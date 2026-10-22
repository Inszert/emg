"""
Myo acquisition: SampleRing, Capture, MyoAcquisition (no Qt). The only code
that touches the serial port.
"""

import struct
import threading
import time
import traceback
from collections import deque

import numpy as np

from emg_app import config as cfg
from emg_app.utils import nominal_rate
from emg_app.recording import Recording, STAGE_RAW


# ==============================================================================
# SECTION 5 -- MYO ACQUISITION (background thread)
#
# This section is deliberately Qt-free so that it can be tested headless.
#
# It does NOT reimplement the Myo protocol: it drives the existing, working
# `myo_raw.MyoRaw` wrapper exactly the way test_myo.py does (construct ->
# add_emg_handler -> connect -> run() in a loop), and adds around it:
#   * a bounded, thread-safe ring buffer for the live plot
#   * zero-loss capture straight out of the EMG callback
#   * stall detection and automatic reconnection with back-off
# ==============================================================================

from emg_app.myo.myo_raw import MyoRaw                      # the working communication layer

# Attribute handles of the Myo's four unfiltered-EMG characteristics. Only used
# when EMG_MODE == "raw"; the default "filtered" path never touches these.
RAW_EMG_DATA_ATTRS = (0x2b, 0x2e, 0x31, 0x34)
RAW_EMG_NOTIFY_ATTRS = (0x2c, 0x2f, 0x32, 0x35)

# Connection states, shown verbatim in the GUI.
CONN_DISCONNECTED = "DISCONNECTED"
CONN_CONNECTING = "CONNECTING"
CONN_CONNECTED = "CONNECTED"
CONN_RECONNECTING = "RECONNECTING"
CONN_ERROR = "ERROR"
CONN_STOPPED = "STOPPED"


class SampleRing:
    """
    Bounded thread-safe ring buffer for the live display.

    Bounded on purpose: a four-hour session must not grow memory. Recording
    data does NOT come from here (see Capture) -- losing a display sample is
    cosmetic, losing a recorded sample is not, so the two paths are separate.
    """

    def __init__(self, capacity: int):
        self.capacity = int(max(64, capacity))
        self._emg = np.zeros((self.capacity, cfg.N_CHANNELS), dtype=np.int32)
        self._t = np.zeros(self.capacity, dtype=np.float64)
        self._write = 0
        self._total = 0
        self._lock = threading.Lock()

    def push(self, t: float, values) -> None:
        with self._lock:
            i = self._write
            self._emg[i] = values
            self._t[i] = t
            self._write = (i + 1) % self.capacity
            self._total += 1

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._total = 0
            self._emg[:] = 0
            self._t[:] = 0.0

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def snapshot(self, n: int = None):
        """Last `n` samples in chronological order, as fresh copies."""
        with self._lock:
            have = min(self._total, self.capacity)
            if have == 0:
                return (np.zeros((0, cfg.N_CHANNELS), np.int32), np.zeros(0, np.float64))
            n = have if n is None else int(min(max(1, n), have))
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return (self._emg[start:start + n].copy(), self._t[start:start + n].copy())
            tail = self.capacity - start
            emg = np.concatenate([self._emg[start:], self._emg[:n - tail]])
            t = np.concatenate([self._t[start:], self._t[:n - tail]])
            return emg, t

    def latest(self):
        """Most recent sample as (values, t), or (zeros, 0.0) if empty."""
        with self._lock:
            if self._total == 0:
                return np.zeros(cfg.N_CHANNELS, np.int32), 0.0
            i = (self._write - 1) % self.capacity
            return self._emg[i].copy(), float(self._t[i])


class Capture:
    """
    An in-progress recording.

    Samples are appended from inside the Myo callback, in the acquisition
    thread, so nothing can be dropped by a slow GUI frame or a blocked event
    loop. The GUI only ever reads the finished arrays.
    """

    def __init__(self, label: str, repetition: int, max_samples: int,
                 meta: dict = None):
        self.label = label
        self.repetition = int(repetition)
        self.max_samples = int(max_samples)
        self.meta = dict(meta or {})
        self.started_unix = time.time()
        self.finished_unix = None
        self.had_disconnect = False
        self.overflowed = False
        self.label_start_index = 0
        self.label_end_index = None
        self._emg = []
        self._t = []
        self._lock = threading.Lock()

    def append(self, t: float, values) -> None:
        with self._lock:
            if len(self._emg) >= self.max_samples:
                self.overflowed = True
                return
            self._emg.append(values)
            self._t.append(t)

    def mark_label_start(self) -> None:
        with self._lock:
            self.label_start_index = len(self._emg)

    def mark_label_end(self) -> None:
        with self._lock:
            self.label_end_index = len(self._emg)

    def __len__(self) -> int:
        with self._lock:
            return len(self._emg)

    def arrays(self):
        with self._lock:
            if not self._emg:
                return (np.zeros((0, cfg.N_CHANNELS), np.int32), np.zeros(0, np.float64))
            return (np.asarray(self._emg, dtype=np.int32),
                    np.asarray(self._t, dtype=np.float64))

    def to_recording(self, person_id: str, session_id: str,
                     extra_meta: dict = None) -> Recording:
        emg, t = self.arrays()
        meta = dict(self.meta)
        meta.update(extra_meta or {})
        meta.setdefault("emg_mode", cfg.EMG_MODE)
        meta.setdefault("capture_overflowed", self.overflowed)
        meta.setdefault("capture_had_disconnect", self.had_disconnect)

        end = self.label_end_index if self.label_end_index is not None else emg.shape[0]
        return Recording(
            emg=emg, timestamps=t, label=self.label,
            person_id=person_id, session_id=session_id,
            repetition=self.repetition, stage=STAGE_RAW,
            created_unix=self.started_unix,
            label_start_index=min(self.label_start_index, max(0, emg.shape[0] - 1)),
            label_end_index=max(1, min(end, emg.shape[0])) if emg.shape[0] else 0,
            meta=meta,
        )


class MyoAcquisition(threading.Thread):
    """
    Background Myo reader.

    Lifecycle:  start()  ->  (connect -> stream -> on error reconnect)*  ->  stop()

    Callbacks (invoked from THIS thread, keep them short):
        on_state(state, message)    connection state changed
        on_sample(t, values)        optional per-sample tap (live prediction)
        on_log(text)               human-readable log line
    """

    def __init__(self, port=None, mode: str = None, on_state=None,
                 on_sample=None, on_log=None):
        super().__init__(name="MyoAcquisition", daemon=True)
        self.port = port if port is not None else cfg.MYO_PORT
        self.mode = mode or cfg.EMG_MODE

        fs = nominal_rate(self.mode)
        self.ring = SampleRing(int(cfg.RING_BUFFER_SECONDS * fs * 1.5) + 1024)

        self._on_state = on_state or (lambda state, msg: None)
        self._on_sample = on_sample
        self._on_log = on_log or (lambda text: None)

        self._myo = None
        self._myo_lock = threading.Lock()
        self._capture = None
        self._capture_lock = threading.Lock()

        self._vibrate_request = None
        self._vibrate_lock = threading.Lock()

        self._running = threading.Event()
        self._state = CONN_DISCONNECTED
        self._state_message = ""

        # Timebase: monotonic deltas pinned to one wall-clock reading, so
        # timestamps are both absolute and free of clock-adjustment jumps.
        self._t0_unix = time.time()
        self._t0_perf = time.perf_counter()

        self._known_addr = None           # cached armband address
        self._connect_started = 0.0       # non-zero while a scan is in flight
        self._error_hint = ""             # friendly reason for the last failure
        self._last_sample_perf = 0.0
        self._sample_count = 0
        self._dropped_packets = 0
        self._rate_estimate = 0.0
        self._rate_window = deque(maxlen=64)
        self._reconnects = 0
        self._backlog_events = 0
        self._worst_backlog = 0
        self._stats_lock = threading.Lock()

    # -- public API -----------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def state_message(self) -> str:
        return self._state_message

    @property
    def connected(self) -> bool:
        return self._state == CONN_CONNECTED

    def stats(self) -> dict:
        # Read the port depth BEFORE taking the stats lock: nesting
        # _stats_lock around _myo_lock would be the only place in this class
        # that orders those two that way.
        backlog = self._input_backlog()
        with self._stats_lock:
            return {
                "samples": self._sample_count,
                "rate": self._rate_estimate,
                "dropped": self._dropped_packets,
                "reconnects": self._reconnects,
                "ring_fill": min(1.0, self.ring.total / self.ring.capacity),
                "backlog_events": self._backlog_events,
                "worst_backlog": self._worst_backlog,
                "backlog": backlog,
                "last_sample_age": (time.perf_counter() - self._last_sample_perf
                                    if self._last_sample_perf else None),
            }

    def _input_backlog(self) -> int:
        """Bytes waiting in the serial port right now (0 when not connected)."""
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            return 0
        try:
            return int(myo.bt.ser.in_waiting)
        except Exception:
            return 0

    def unix_time(self) -> float:
        return self._t0_unix + (time.perf_counter() - self._t0_perf)

    def start_capture(self, label: str, repetition: int, meta: dict = None,
                      max_seconds: float = None) -> Capture:
        """
        Begin recording. Samples land in the returned Capture immediately.

        `max_seconds` overrides the MAX_RECORDING_SECONDS safety valve, which
        is sized for a cued take of a few seconds. A FREE TAKE passes its own,
        much larger cap; the 1.5x headroom stays, because the real rate runs
        above the nominal one often enough to matter.
        """
        seconds = float(max_seconds or cfg.MAX_RECORDING_SECONDS)
        capture = Capture(
            label, repetition,
            max_samples=int(seconds * nominal_rate(self.mode) * 1.5),
            meta=meta,
        )
        capture.had_disconnect = not self.connected
        with self._capture_lock:
            self._capture = capture
        return capture

    def stop_capture(self):
        """End the current recording and hand it back."""
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.finished_unix = time.time()
        return capture

    @property
    def capturing(self) -> bool:
        with self._capture_lock:
            return self._capture is not None

    def vibrate(self, length: int = 1) -> None:
        """
        Request a haptic cue. Safe to call from the GUI thread.

        The write is NOT performed here. `BT` shares one serial port and one
        packet-assembly buffer between all callers and takes no lock while
        reading, so writing from a second thread desynchronises the byte
        stream: packets get misframed and EMG notifications start arriving as
        "data with unknown attr" garbage. The request is therefore handed to
        the acquisition thread, which performs it between two packets.
        """
        if not cfg.VIBRATE_CUES:
            return
        with self._vibrate_lock:
            self._vibrate_request = int(np.clip(length, 1, 3))

    def _service_vibrate(self, myo) -> None:
        """Perform a pending haptic cue. Called only from the acquisition thread."""
        with self._vibrate_lock:
            length = self._vibrate_request
            self._vibrate_request = None
        if length is None:
            return
        try:
            payload = struct.pack("<3B", 3, 1, length)
            if cfg.MYO_BULK_READ:
                # Fire and forget. myo_raw's write_attr blocks waiting for the
                # command response, but in bulk-read mode `_pump` has already
                # consumed and discarded it, so that wait would hang forever.
                self._write_attr_nowait(myo, 0x19, payload)
            else:
                myo.write_attr(0x19, payload)
        except Exception:
            pass          # a buzz is never worth losing the connection over

    @staticmethod
    def _write_attr_nowait(myo, attr: int, value: bytes) -> None:
        """
        Write a BLE attribute without waiting for the acknowledgement.

        Same framing as myo_raw's BT.write_attr/send_command -- BGAPI command
        class 4, id 5 -- just without the blocking read of the response.
        """
        if myo.conn is None:
            return
        payload = struct.pack("<BHB", myo.conn, attr, len(value)) + value
        myo.bt.ser.write(struct.pack("<4B", 0, len(payload), 4, 5) + payload)

    def stop(self) -> None:
        """
        Ask the thread to finish.

        Closing the serial port from here is intentional: MyoRaw.connect() can
        block in a bare serial read, and closing the handle turns that hang
        into an exception the loop already knows how to handle.
        """
        self._running.clear()
        with self._myo_lock:
            myo = self._myo
        if myo is not None:
            try:
                myo.bt.ser.close()
            except Exception:
                pass

    # -- thread body ----------------------------------------------------------

    def run(self) -> None:
        self._running.set()
        delay = cfg.RECONNECT_DELAY_S
        attempts = 0

        watchdog = threading.Thread(target=self._watchdog,
                                    name="MyoWatchdog", daemon=True)
        watchdog.start()

        while self._running.is_set():
            try:
                self._error_hint = ""
                self._connect_started = time.perf_counter()
                self._set_state(CONN_CONNECTING, "Opening the Myo dongle...")
                self._connect()
                self._connect_started = 0.0
                self._set_state(CONN_CONNECTED, "Streaming EMG.")
                delay = cfg.RECONNECT_DELAY_S
                attempts = 0
                self._stream()

            except Exception as exc:
                if not self._running.is_set():
                    break

                # A hint set by the watchdog explains the cause far better
                # than the serial error that closing the port produced.
                hint = self._error_hint
                self._log(hint or f"Myo error: {exc}")
                self._mark_capture_disconnect()
                self._teardown()

                if not cfg.AUTO_RECONNECT:
                    self._set_state(CONN_ERROR, hint or str(exc))
                    break

                attempts += 1
                if cfg.RECONNECT_MAX_ATTEMPTS and attempts > cfg.RECONNECT_MAX_ATTEMPTS:
                    self._set_state(
                        CONN_ERROR,
                        f"Giving up after {attempts - 1} reconnect attempts.",
                    )
                    break

                self._set_state(
                    CONN_RECONNECTING,
                    (f"{hint}  " if hint else "")
                    + f"Attempt {attempts} in {delay:.0f} s...",
                )
                self._sleep(delay)
                delay = min(delay * 1.5, cfg.RECONNECT_MAX_DELAY_S)

        self._teardown()
        watchdog.join(timeout=1.0)
        self._set_state(CONN_STOPPED, "Acquisition stopped.")

    # -- connection -----------------------------------------------------------

    def _apply_link_settings(self) -> None:
        """
        Push the BLE connection parameters into myo_raw before it connects.

        They live as module constants there so this can tune them without
        forking the connection logic.
        """
        from emg_app.myo import myo_raw as _mr
        interval = max(1, int(round(cfg.MYO_CONN_INTERVAL_MS / 1.25)))
        _mr.CONN_INTERVAL_MIN = interval
        _mr.CONN_INTERVAL_MAX = interval
        _mr.CONN_TIMEOUT = int(np.clip(round(cfg.MYO_CONN_TIMEOUT_MS / 10.0),
                                       10, 3200))
        _mr.CONN_LATENCY = 0

    def _connect(self) -> None:
        """Construct and connect a MyoRaw exactly the way the working code does."""
        self._apply_link_settings()
        myo = MyoRaw(self.port)
        myo.add_emg_handler(self._on_emg_filtered)

        if self.mode == "raw":
            self._install_raw_emg_support(myo)

        # Publish the instance BEFORE connect(). connect() blocks in the BLE
        # scan loop -- indefinitely if the armband is asleep -- and stop()
        # needs the serial handle to break that read. Publishing afterwards
        # left the thread stuck in the scan and the port open until the
        # process exited.
        with self._myo_lock:
            self._myo = myo

        # Reconnect straight to the address we already know, when we have one.
        # If that fails the address is forgotten so the next attempt scans.
        addr = self._known_addr if cfg.MYO_FAST_RECONNECT else None
        try:
            myo.connect(addr)
        except Exception:
            if addr is not None:
                self._known_addr = None
                self._log("Direct reconnect failed; will scan on the next try.")
            raise
        self._known_addr = getattr(myo, "addr", None)

        if cfg.MYO_NEVER_SLEEP:
            try:
                myo.set_sleep_mode(True)
                self._log("Sleep mode disabled (armband will not doze off).")
            except Exception as exc:
                self._log(f"Could not disable sleep mode: {exc}")

        if self.mode == "raw":
            self._enable_raw_emg(myo)

        self._last_sample_perf = time.perf_counter()
        self._log(
            f"Myo connected on {myo.bt.ser.port} (mode: {self.mode}, "
            f"supervision timeout {cfg.MYO_CONN_TIMEOUT_MS} ms"
            + (", direct" if addr is not None else ", after scan") + ")."
        )

    def _stream(self) -> None:
        """
        Pump packets until stopped or the link breaks.

        `myo.run()` is called with NO timeout argument, exactly as the
        known-good test_myo.py does, and this matters a great deal:

        `BT.recv_packet(timeout)` re-assigns `self.ser.timeout` before every
        single byte it reads, and on Windows each assignment reconfigures the
        COM port. With `run(1)` that is thousands of port reconfigurations per
        second -- measured on this machine it throttled a 50 Hz stream down to
        4.6 Hz and then failed outright with "A device attached to the system
        is not functioning". With no timeout, `recv_packet` sets the timeout
        once per packet and the stream runs at full rate.

        The cost is that this call blocks in a bare serial read when nothing
        arrives, so it cannot notice a stall by itself. `_watchdog` does that
        instead, and `stop()` closes the port to break out of the read.
        """
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            raise RuntimeError("not connected")

        if cfg.MYO_FLUSH_ON_START:
            try:
                myo.bt.ser.reset_input_buffer()
            except Exception:
                pass

        if cfg.MYO_BULK_READ:
            # Set the timeout ONCE. myo_raw re-assigns it per packet, which is
            # the expensive part on Windows.
            try:
                myo.bt.ser.timeout = None
            except Exception:
                pass
            while self._running.is_set():
                self._pump(myo)
                self._service_vibrate(myo)
            return

        while self._running.is_set():
            myo.run()
            # Between packets is the only safe moment to write to the port.
            self._service_vibrate(myo)

    def _pump(self, myo) -> None:
        """
        Read everything waiting and parse it with myo_raw's own byte parser.

        Blocks on at least one byte so the loop does not spin. Command
        responses (typ 0) are dropped: during streaming nothing waits for
        them, and `_service_vibrate` writes fire-and-forget for that reason.
        """
        ser = myo.bt.ser
        bt = myo.bt

        waiting = ser.in_waiting
        if cfg.MYO_MAX_BACKLOG_BYTES and waiting > cfg.MYO_MAX_BACKLOG_BYTES:
            self._handle_backlog(ser, waiting)
            waiting = ser.in_waiting

        data = ser.read(waiting if waiting > 0 else 1)
        if not data:
            return

        for byte in data:
            packet = bt.proc_byte(byte)
            if packet is not None and packet.typ == 0x80:
                bt.handle_event(packet)

    def _handle_backlog(self, ser, waiting: int) -> None:
        """We are behind. Either catch up by discarding, or say so."""
        approx_seconds = waiting / max(1.0, 25.0 * nominal_rate(self.mode))
        with self._stats_lock:
            self._backlog_events += 1
            self._worst_backlog = max(self._worst_backlog, waiting)

        if self.capturing:
            self._log(
                f"Running {approx_seconds:.1f} s behind ({waiting} bytes "
                "queued) -- not flushing during a recording."
            )
            return

        try:
            ser.reset_input_buffer()
        except Exception:
            return
        self._log(
            f"Dropped a {approx_seconds:.1f} s backlog ({waiting} bytes) to "
            "stay live."
        )

    def _watchdog(self) -> None:
        """
        Notice a silent link.

        An armband going to sleep does not raise a serial error, it just stops
        sending. Closing the port from here turns the blocked read in
        `_stream` into an exception, which the reconnect loop already handles.
        """
        while self._running.is_set():
            time.sleep(0.25)

            # A scan that never finds anything: MyoRaw.connect() loops on
            # recv_packet() with no bound, so without this the thread sits
            # there forever and the UI only ever says "Opening the dongle".
            started = self._connect_started
            if cfg.SCAN_TIMEOUT_S and started and self._state == CONN_CONNECTING:
                if time.perf_counter() - started > cfg.SCAN_TIMEOUT_S:
                    self._connect_started = 0.0
                    self._error_hint = (
                        f"Armband did not answer in {cfg.SCAN_TIMEOUT_S:.0f} s -- "
                        "it is probably asleep. Pick it up, flex, or move it "
                        "to wake it."
                    )
                    self._log(self._error_hint)
                    self._force_close()
                    continue

            if not cfg.STALL_TIMEOUT_S or self._state != CONN_CONNECTED:
                continue
            last = self._last_sample_perf
            if not last:
                continue
            age = time.perf_counter() - last
            if age > cfg.STALL_TIMEOUT_S:
                self._log(
                    f"no EMG packet for {age:.1f} s -- the armband went to "
                    "sleep or the link dropped; forcing a reconnect"
                )
                # Debounce so the reconnect gets a chance before we fire again.
                self._last_sample_perf = time.perf_counter()
                self._force_close()

    def _force_close(self) -> None:
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            return
        try:
            myo.bt.ser.close()
        except Exception:
            pass

    def _teardown(self) -> None:
        self._connect_started = 0.0
        with self._myo_lock:
            myo = self._myo
            self._myo = None
        if myo is None:
            return
        try:
            myo.disconnect()
        except Exception:
            pass
        try:
            myo.bt.ser.close()
        except Exception:
            pass

    def _mark_capture_disconnect(self) -> None:
        """Flag the in-flight recording instead of discarding it."""
        with self._capture_lock:
            if self._capture is not None:
                self._capture.had_disconnect = True

    # -- raw (unfiltered, 200 Hz) EMG ----------------------------------------
    #
    # The default "filtered" mode needs none of this. Raw mode is additive: it
    # registers an extra BT packet handler and enables four more notification
    # characteristics. The connection logic in myo_raw.py is untouched.

    def _install_raw_emg_support(self, myo) -> None:
        def handle_raw(packet):
            if (packet.cls, packet.cmd) != (4, 5):
                return
            try:
                _, attr, _ = struct.unpack("<BHB", packet.payload[:4])
            except struct.error:
                return
            if attr not in RAW_EMG_DATA_ATTRS:
                return
            payload = packet.payload[5:]
            if len(payload) < 16:
                return
            vals = struct.unpack("<16b", payload[:16])
            # Each notification carries two consecutive 8-channel samples.
            self._on_emg_raw(vals[0:8])
            self._on_emg_raw(vals[8:16])

        myo.bt.add_handler(handle_raw)

    def _enable_raw_emg(self, myo) -> None:
        for attr in RAW_EMG_NOTIFY_ATTRS:
            myo.write_attr(attr, b"\x01\x00")
        # set_mode(emg = send_emg_raw, imu = on, classifier = off)
        myo.write_attr(0x19, b"\x01\x03\x03\x01\x00")
        self._log("Unfiltered 200 Hz EMG enabled.")

    # -- sample path ----------------------------------------------------------

    def _on_emg_filtered(self, emg, moving) -> None:
        if self.mode == "filtered":
            self._ingest(emg)

    def _on_emg_raw(self, emg) -> None:
        self._ingest(emg)

    def _ingest(self, values) -> None:
        """
        The hot path: called once per sample, in the acquisition thread.

        Order matters. The capture is fed FIRST, because that is the data that
        must not be lost; display and statistics come after.
        """
        t = self.unix_time()
        values = tuple(int(v) for v in values)

        with self._capture_lock:
            capture = self._capture
        if capture is not None:
            capture.append(t, values)

        self.ring.push(t, values)

        now = time.perf_counter()
        with self._stats_lock:
            if self._last_sample_perf:
                dt = now - self._last_sample_perf
                if dt > 0:
                    self._rate_window.append(dt)
                    self._rate_estimate = 1.0 / (sum(self._rate_window)
                                                 / len(self._rate_window))
                expected = 1.0 / max(1.0, nominal_rate(self.mode))
                if dt > expected * 3.0:
                    self._dropped_packets += int(dt / expected) - 1
            self._last_sample_perf = now
            self._sample_count += 1

        if self._on_sample is not None:
            try:
                self._on_sample(t, values)
            except Exception:
                pass

    # -- plumbing -------------------------------------------------------------

    def _set_state(self, state: str, message: str = "") -> None:
        self._state = state
        self._state_message = message
        if state == CONN_RECONNECTING:
            with self._stats_lock:
                self._reconnects += 1
        try:
            self._on_state(state, message)
        except Exception:
            traceback.print_exc()

    def _log(self, text: str) -> None:
        print(f"[myo] {text}")
        try:
            self._on_log(text)
        except Exception:
            pass

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep -- stop() must not have to wait out a back-off."""
        end = time.perf_counter() + seconds
        while self._running.is_set() and time.perf_counter() < end:
            time.sleep(0.05)

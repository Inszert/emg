# Architecture

How `emg_dataset_tool.py` is put together, and why.

---

## 1. Data flow

```
   Myo armband
       | BLE
   BLED112 dongle (COM3)
       | pyserial, 9600 baud
   myo_raw.BT.recv_packet()                 BGAPI framing            [unchanged]
       |
   myo_raw.MyoRaw.handle_data()             attr 0x27 -> on_emg      [unchanged]
       |  callback, acquisition thread  (raw mode: our own bt handler instead)
   MyoAcquisition._ingest(values)           <-- THE HOT PATH
       |
       +--> Capture.append()        recording buffer   (FIRST: must not lose data)
       +--> SampleRing.push()       display buffer     (bounded, lossy is fine)
       +--> rate / drop statistics
       +--> optional on_sample tap  (live classifier)

   ... GUI thread, on its own timers ...

   QTimer(1000/GUI_FPS) -> SampleRing.snapshot() -> EmgPlotWidget.update_data()
   QTimer(20 ms)        -> SequenceController.tick() -> StatusBanner
   QTimer(100 ms)       -> status bar, channel readouts, button enable states
```

The acquisition thread **never touches a widget**. It reports connection state
through `AcquisitionBridge` (a `QObject` with signals — Qt queues cross-thread
emissions), and the GUI reads samples itself.

### Why not emit samples as signals?

At 200 Hz × 8 channels that is 200 queued events per second per subscriber.
The event loop would spend its time on marshalling, and a stall in the GUI
would back-pressure into the acquisition thread. Pulling from a ring buffer
decouples the two completely: the plot renders whatever is current, and a
dropped frame costs nothing.

## 2. Threads

| Thread | Owner | Responsibility |
|---|---|---|
| Main / GUI | Qt | widgets, timers, state machine, file writing |
| `MyoAcquisition` | `threading.Thread`, daemon | serial I/O, connect/reconnect, sample ingestion |
| `MyoWatchdog` | `threading.Thread`, daemon | notices a silent link and forces a reconnect |
| `DatasetExportWorker` | `QThread`, transient | walks the dataset, builds the export bundle |

**Only `MyoAcquisition` may read or write the serial port.** `BT` shares one
port and one packet-assembly buffer between all callers and takes no lock
while reading, so a write from any other thread desynchronises the byte stream
and EMG notifications start arriving as misframed `data with unknown attr`
garbage. This was observed with a `vibrate()` call from the GUI thread; the
haptic cue is now queued and performed by the acquisition thread between two
packets. The watchdog is the one exception, and it only ever *closes* the port
— deliberately, to break the reader out of a blocking read.

Locks, all short and uncontended:

- `SampleRing._lock` — around index arithmetic and the array copy.
- `Capture._lock` — around the append and the final array conversion.
- `MyoAcquisition._capture_lock` — swapping the active `Capture` in and out.
- `MyoAcquisition._myo_lock` — swapping the `MyoRaw` instance on reconnect.
- `MyoAcquisition._stats_lock` — rate/drop counters.

No lock is ever held while doing I/O or while calling into Qt.

### Subprocesses

Two things deliberately run *outside* the recorder process rather than on a
thread inside it, both `QProcess`, both reported back through Qt signals:

| Process | Started by | Why not a thread |
|---|---|---|
| `train_model.py` | RETRAIN ON ALL DATA | scikit-learn with `n_jobs=-1` starves the acquisition thread, and a training crash must not take the recorder down |
| the compute probe | 400 ms after startup | it imports torch to ask whether CUDA is usable — ~7 s, plus a GPU context the recorder has no use for |

The compute probe is `COMPUTE_PROBE_SNIPPET`, a dependency-free snippet in the
data layer that prints one JSON line; `parse_compute_probe()` reads it back.
Both are Qt-free and testable headless, and `detect_compute_device()` is the
blocking version for scripts. The recorder therefore **never imports torch**,
and reports the device as unknown until the probe actually answers rather than
assuming either way.

## 3. Buffers

### `SampleRing` — display, bounded

Pre-allocated numpy arrays, `capacity = RING_BUFFER_SECONDS × fs × 1.5`.
Fixed memory for a session of any length. `snapshot(n)` returns copies in
chronological order, so the GUI never reads a buffer that is being written.

### `Capture` — recording, growing but capped

A plain Python list, appended from the acquisition callback. Capped at
`max_seconds × fs × 1.5`, where `max_seconds` defaults to
`MAX_RECORDING_SECONDS`; hitting the cap sets `overflowed` (which is recorded
in the metadata) rather than dropping samples silently.

**The two are separate on purpose.** Losing a display sample is cosmetic.
Losing a recorded sample is data loss. If recording read from the ring, a GUI
stall longer than the ring would silently punch a hole in a recording.

### One capture slot, two owners

`MyoAcquisition` holds exactly one `Capture` at a time, swapped under
`_capture_lock`. Two things in the GUI want it:

| Owner | Opened by | Cap |
|---|---|---|
| the cued sequence | `SequenceController._begin_capture` | `MAX_RECORDING_SECONDS` (120 s) |
| a free take | `MainWindow.start_free_take` | `FREE_TAKE_MAX_MINUTES × 60` (15 min) |

A free take deliberately does **not** go through `SequenceController`: it has
no plan, no countdown and no review step, so routing it through the state
machine would mean inventing states for a recording that simply runs until the
operator stops it. It calls `start_capture(..., max_seconds=...)` directly.

Because the slot is single, the two can never run at once.
`MainWindow._refresh_controls()` is what enforces it: while `free_capture` is
set, START SESSION and RECORD are disabled, and while the sequence is running
START FREE TAKE is. Nothing relies on the acquisition layer to arbitrate —
`start_capture` would simply overwrite the slot and the first capture's
samples would stop arriving.

`LABEL ON GRAPH` and the markers panel need no slot at all: they cut from the
**ring buffer** after the fact, which is why they work during a free take.

## 4. Timebase

```python
t = self._t0_unix + (time.perf_counter() - self._t0_perf)
```

One wall-clock reading at startup plus monotonic deltas. Timestamps are
absolute (so recordings can be correlated with anything else) and immune to
NTP steps or DST changes mid-session.

The sample rate is **measured**, never assumed:
`measured_rate()` takes the span average `(n-1) / (t[-1]-t[0])` (not the median
interval: delivery is bursty, see section 5a), so a few dropped packets barely skew it. `NOMINAL_SAMPLE_RATE` is only used to size buffers and
to sanity-check the measurement.

## 5. Connection lifecycle

```
run():
  loop:
    CONNECTING     MyoRaw(port); add_emg_handler; connect()
                   [raw mode only] enable notify attrs, set_mode(emg=0x03)
    CONNECTED      _stream():  myo.run()  until stopped or the link breaks
    on exception:
       flag the in-flight capture as had_disconnect
       disconnect() + close the serial port
       RECONNECTING, back-off 2s -> 3s -> 4.5s ... capped
    repeat
```

Decisions that matter:

- **A fresh `MyoRaw` per reconnect.** A dead instance cannot be revived; this
  is what `test_myo.py` established and it is preserved here.
- **`myo.run()` with NO timeout argument.** This is not a detail.
  `BT.recv_packet(timeout)` re-assigns `self.ser.timeout` *before every single
  byte*, and on Windows each assignment reconfigures the COM port. Measured on
  this machine, `run(1)` throttled a 50 Hz stream to **4.6 Hz** and then failed
  with *"A device attached to the system is not functioning"*. With no
  timeout the assignment happens once per packet and the stream runs at full
  rate — which is exactly why `test_myo.py` calls `m.run()`.
- **A scan that finds nothing is also bounded by the watchdog**
  (`SCAN_TIMEOUT_S`). `MyoRaw.connect()` loops on `recv_packet()` with no
  limit, and the armband stops advertising a minute or two after it comes off,
  so without this the thread sits in the scan forever while the UI says only
  "Opening the Myo dongle...". The watchdog closes the port, and
  `_error_hint` carries a readable reason ("probably asleep, move it to wake
  it") into the `RECONNECTING` state message instead of the serial error that
  closing the handle produced.
- **Stall detection lives in `MyoWatchdog`**, because the no-timeout read
  blocks when nothing arrives. The armband going to sleep does not raise a
  serial error, it just goes quiet; the watchdog notices
  (`STALL_TIMEOUT_S`) and closes the port, which turns the blocked read into
  an exception the reconnect loop already handles.
- **`stop()` closes the serial port.** `MyoRaw.connect()` can block in a bare
  read while scanning; closing the handle converts that hang into an exception
  the loop already handles. Deliberate, not a leak.
- **An in-flight recording is flagged, not silently accepted.**
  `had_disconnect` becomes a quality error so the operator can judge it.
- **A running session auto-pauses on link loss and auto-resumes on reconnect**
  (`PAUSE_SESSION_ON_LINK_LOSS`), so repetitions do not count down into a dead
  link and an unattended subject is not left staring at a frozen screen.
- **An involuntary pause discards the partial recording and retries the
  repetition** (`pause(discard_in_flight=True)`); the plan index is not
  advanced. A deliberate PAUSE by the operator instead finishes and saves the
  recording. Without the distinction, every dropout would quietly cost one
  sample of whichever gesture was being recorded.

## 5a. Measuring the sample rate

The dongle delivers two to four buffered packets at once, microseconds apart,
then goes quiet until the next burst. So:

- `measured_rate()` uses the **span average** `(n-1) / (t[-1]-t[0])`.
  `1/median(diff(t))` measures an *intra-burst* interval and reported
  **9258 Hz** for a 57 Hz stream before this was fixed.
- `stats()["rate"]` is a moving average over 64 intervals, which smooths the
  bursts out and reads ≈50 Hz live.
- The `dropped` counter is therefore **approximate** — burst delivery inflates
  it a little. A handful of counts over thousands of samples is normal; a
  number comparable to the sample count is not.

## 6. EMG modes

| | `filtered` (default) | `raw` |
|---|---|---|
| BLE attr | `0x27` | `0x2b`, `0x2e`, `0x31`, `0x34` |
| set_mode | `01 03 01 01 01` (via `MyoRaw.start_raw()`) | `01 03 03 01 00` |
| Payload | `8H` + moving byte | `16b` = **two** samples per packet |
| Encoding | uint16, rectified | int8, zero-centred |
| Rate | ~50 Hz | 200 Hz |

Raw mode is implemented **without changing the connection logic**: the app
registers its own extra handler via `myo.bt.add_handler(...)` and enables four
more notification characteristics after `connect()` has succeeded. The only
edit inside `myo_raw.py` is silencing the `unknown attr` print for those four
handles.

## 7. Sequence state machine

`SequenceController` is pure logic — no Qt, no timers of its own, driven by
`tick()`. That makes the recording protocol testable without an armband.

```
IDLE --start()--> REST -> PREPARE -> COUNTDOWN -> RECORDING -> STOPPING --+
                   ^                                                      |
                   |                         REVIEW_EACH_RECORDING = False |
                   |                            +--------- SAVED <---------+
                   |                            |            |
                   +---- AUTO_ADVANCE ----------+            |
                   |                                         |
                   |                    REVIEW_EACH_RECORDING = True
                   |                                         |
                   |                                      REVIEW  (no deadline)
                   |                                      /     \
                   +---- redo_review() ------------------+       +-- accept_review()
                         (index unchanged)                            (index += 1)
                                                                            |
                                                      index >= len(plan) -> DONE
PAUSED  <-- pause() from any running state (finishes the in-flight recording first)
```

`REVIEW` has `state_duration = 0`, which is how "wait indefinitely" is
expressed: `tick()` returns early on a zero duration, so nothing advances
until `accept_review()` or `redo_review()` is called. `pause()` is a no-op in
`REVIEW` — the machine is already waiting on the operator.

`redo_review()` deliberately does **not** advance the index, so the plan still
owes that gesture. It also never touches the filesystem: quarantining the
rejected file is `MainWindow._quarantine()`'s job, which keeps the controller
testable without a disk.

Timing is **wall-clock against a deadline**, not tick counting:

```python
if self.elapsed < self.state_duration: return
```

A dropped frame or a slow save delays nothing — the machine simply notices it
is already past the deadline.

### The capture window

```
          COUNTDOWN                RECORDING              STOPPING
 ... 3 ....... 2 ....... 1 |======== cue window ========| ....... |
                       ^                                ^         ^
            capture opens                    label window      capture
         (PRE_GESTURE_MS early)                  closes          closes
```

`label_start_index` / `label_end_index` mark the cue window inside a strictly
wider raw file. A subject who reacts early, or an explosive contraction that
overshoots the cue, stays fully inside the saved data.

`pause()` and `stop()` finish the in-flight recording rather than freezing
mid-gesture — a truncated sample carrying a full label is worse than no sample.

## 8. Plot rendering

One `PlotItem` per channel in a `GraphicsLayoutWidget`, x-axes linked, x in
**seconds relative to now** (range `[-window, 0]`), so the renderer never sees
large float64 values.

Performance measures, in order of importance:

1. `setData` on persistent `PlotDataItem`s — items are never recreated.
2. `setDownsampling(auto=True, method="peak")` — **peak mode, not mean.**
   Mean-mode downsampling smooths away exactly the spikes this project cares
   about; peak mode keeps the extremes of every bin.
3. `setClipToView(True)` — off-screen samples cost nothing.
4. Antialiasing off by default (`PLOT_ANTIALIAS`), line width 1.
5. Text items updated at ~8 Hz, not every frame.
6. The countdown overlay is a `QLabel` on top of the canvas, not a scene item.
   `COUNTDOWN_OVERLAY` defaults to `"corner"`: a centred 150 px digit hides
   exactly the traces the operator is trying to watch while performing the
   gesture, and the big version of the cue already lives in the banner.

### Review mode

The same eight curves are reused to display a finished recording:
`EmgPlotWidget.review(rec, activity)` sets `review_mode`, pushes the
recording's samples with x = `t_rel`, rescales each channel to that take, and
shows a draggable `LinearRegionItem` per channel marking what will be kept.
All eight bands are kept in sync through `_on_trim_region_changed` behind a
`_syncing_trim` guard, and `trim_changed` reports the bounds to the window.

`update_data()` and `_tick_plot()` both return early while `review_mode` is
set, so the live stream keeps filling the ring buffer untouched — the
acquisition thread never pauses for the review. `exit_review()` restores the
rolling `[-window, 0]` x range.

Autoscale is asymmetric — instant growth, slow decay:

```python
self._y_limit[i] = target if target > current else current*0.93 + target*0.07
```

A peak can never be clipped out of view, and the axis does not jitter.

## 9. Export

- **PNG** — `pyqtgraph.exporters.ImageExporter` on the scene.
- **SVG** — Qt's `QSvgGenerator` + `QGraphicsScene.render()`. PyQtGraph 0.14's
  `SVGExporter` raises `ValueError: not enough values to unpack` in
  `correctCoordinates` on these plots. Do not switch back.

The plot's title `LabelItem` is live-updated with person, session, gesture,
mode, measured rate and timestamp, so it is already a correct caption whenever
an image is exported.

## 10. Where the config lives

The `USER CONFIGURATION` block is the single source of truth. GUI edits to
person / session / handedness / notes / repetitions / order are mirrored back
into those module globals by `MainWindow._set_global()` — otherwise
`recording_config_snapshot()` would write metadata that disagrees with what
actually happened.

## 11. Splitting the file later

Sections 1–6 contain no Qt import, which is the seam. A natural split:

```
emg/config.py        USER CONFIGURATION
emg/signal.py        Section 2
emg/quality.py       Section 3
emg/dataset.py       Section 4
emg/acquisition.py   Section 5
emg/sequence.py      Section 6
emg/ui/*.py          Section 7
emg/__main__.py      Section 8
```

Nothing in Sections 1–6 reaches forward into the GUI, so this is a move, not a
rewrite.

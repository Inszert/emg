# SDK for the Myo band

A standalone developer guide to the Myo layer of this project: how the armband
is reached, what you can call from your own code **without the GUI**, what the
data looks like on disk, and what a *new* band needs. It is written so that
you can read it without the other documents; where a rule comes from
`CLAUDE.md` the section is named.

How this document was checked (2026-10-26, Windows 11, CPython 3.14.0,
numpy 2.3.4, pyserial 3.5, scikit-learn 1.7.2, no armband and no dongle
attached): every snippet marked **Verified** was executed from the project root
(`$env:PYTHONPATH = "."` on PowerShell, or `PYTHONPATH=.` on POSIX) and the
output quoted under it is real. Anything that needs the real band is marked
**Requires hardware, not run here** and was only read, not run. Hardware
behaviour (rates, amplitudes) is quoted from `CLAUDE.md` and
`tests/README.md`, where it was measured on the owner's Myo; it was not
re-measured for this document.

---

## 1. The layers

```
Myo armband  (8 dry EMG pods, BLE, name "Myo_TUKE" on the owner's unit)
     |  Bluetooth Low Energy
BLED112 USB dongle (Bluegiga, USB VID:PID 2458:0001) -- shows up as a serial port
     |  pyserial, 9600 baud setting (it is a USB CDC port, the number is nominal)
emg_app/myo/myo_raw.py   class BT      BGAPI framing: bytes -> Packet, commands, events
     |
emg_app/myo/myo_raw.py   class MyoRaw  Myo protocol: scan, connect, set_mode, notifications
     |  add_emg_handler(fn(emg, moving))
emg_app/acquisition.py   class MyoAcquisition   background thread, ring buffer,
     |                                          capture slot, reconnect, watchdog
     |  Qt signals (state only) + ring-buffer reads on a timer
emg_app/gui/...          MainWindow             (not needed for headless use)
```

`emg_app/myo/common.py` only holds the `pack`/`unpack` helpers.
`legacy/myo64.dll` is a leftover of the official Myo SDK and **is not used**;
everything goes over the serial protocol in `myo_raw.py`.

Facts about the stream (from `CLAUDE.md` section 2, measured on hardware):

| | filtered mode (default) | raw mode |
|---|---|---|
| config | `EMG_MODE = "filtered"` | `EMG_MODE = "raw"` |
| sample | 8 x uint16, filtered and rectified on the device | 8 x int8, unfiltered |
| rate | about 50 Hz (49.9 Hz over 6 s) | 200 Hz nominal |
| BLE attribute | `0x27` | `0x2b / 0x2e / 0x31 / 0x34` (two samples per notification) |
| amplitude | baseline about 15-40, maximum-effort fist 1100-1444 | -128..127 |
| status | the path used for all recorded data | implemented, **not validated on a long session** (CLAUDE.md section 12) |

Nominal rates and valid ranges live in `config.NOMINAL_SAMPLE_RATE` and
`config.EMG_RANGE`; the *measured* rate is what is written into each
recording's metadata.

---

## 2. Hard rules (read before touching the connection)

These come from `CLAUDE.md` section 3. The first two were learned the hard way
on real hardware.

1. **Call `myo.run()` with no timeout argument.** `BT.recv_packet(timeout)`
   re-assigns `ser.timeout` before every byte, and on Windows every assignment
   reconfigures the COM port. `run(1)` measured **4.6 Hz instead of 50 Hz** and
   then failed with "A device attached to the system is not functioning".
   Stall detection is therefore done by a separate watchdog thread
   (`MyoAcquisition._watchdog`), which closes the port to break the blocking read.
   If a live test reports a rate near 5 Hz, a timeout has been reintroduced.
2. **Only the acquisition thread may touch the serial port.** `BT` shares one
   port and one packet buffer and takes no lock while reading. A `vibrate()`
   write from the GUI thread desynchronised the framing and EMG notifications
   started arriving as `data with unknown attr` garbage. Haptic cues are
   queued (`MyoAcquisition.vibrate()` only stores the request) and written by
   the acquisition thread between two packets. Anything else that writes to the
   armband must do the same.
3. **Do not swap `myo_raw.py` for another library** (myo-python, pyomyo, ...).
4. **Reconnect = build a fresh `MyoRaw`** and close the old serial port.
   Reusing a dead instance does not work.
5. `MyoRaw.connect()` blocks while scanning (indefinitely if the armband is
   asleep). `MyoAcquisition.stop()` closes the serial handle to break that
   read; that is deliberate, not a leak.
6. Only one process can hold the dongle's port. Close other tools
   (`legacy/test_myo.py`, another copy of the app) first.

### The known-good call sequence

`MyoRaw(port)` -> `add_emg_handler(...)` -> `connect()` -> `run()` in a loop.
`MyoAcquisition._connect` and `_stream` do exactly this and nothing more:

```python
# Requires hardware, not run here.
from emg_app.myo.myo_raw import MyoRaw

def on_emg(emg, moving):          # emg = tuple of 8 ints
    print(emg)

m = MyoRaw()                      # tty=None -> detect_tty() looks for USB PID 2458:0001
m.add_emg_handler(on_emg)
m.connect()                       # scans, connects, prints firmware version
try:
    while True:
        m.run()                   # NO timeout argument (rule 1)
except KeyboardInterrupt:
    pass
finally:
    m.disconnect()
```

`legacy/test_myo.py` is the original matplotlib monitor and is the reference
for "the connection definitely works like this" (it additionally needs
`matplotlib`, which is not in `requirements.txt`).

What `connect()` does, in order (read from `myo_raw.py`): end any scan, drop
connection handles 0-2, scan for the Myo's service UUID (so any Myo is found,
not only one called `Myo_TUKE`), connect, read the firmware version, then for
firmware `1.x` (the tested `1.5.1970.2`): enable notifications and call
`start_raw()`. If you pass `connect(addr)` with the address cached in `m.addr`
from an earlier connect, the scan is skipped (about one second instead of five
to ten).

### The protocol layer without a band

You can exercise the BGAPI byte parser with no hardware. This builds one EMG
notification (attribute `0x27`, 8 x uint16 + 1 byte) and feeds it through
`BT.proc_byte`, the same function the serial loop calls:

```python
import struct
from emg_app.myo.myo_raw import BT

emg = (11, 22, 33, 44, 55, 66, 77, 88)
pay = struct.pack("<8HB", *emg, 0)
body = struct.pack("<BHBB", 0, 0x27, 0, len(pay)) + pay     # conn, attr, type, len
raw = bytes([0x80, len(body), 4, 5]) + body                  # event, len, class 4, cmd 5

bt = BT.__new__(BT)          # skip __init__: it would open a serial port
bt.buf = []
packet = None
for byte in raw:
    packet = bt.proc_byte(byte) or packet
conn, attr, typ = struct.unpack("<BHB", packet.payload[:4])
vals = struct.unpack("<8HB", packet.payload[5:])
print("attr=0x%02X" % attr, "emg=", vals[:8])
assert attr == 0x27 and vals[:8] == emg
```

**Verified:** prints `attr=0x27 emg= (11, 22, 33, 44, 55, 66, 77, 88)`.

### Finding the dongle

```python
import re
from serial.tools.list_ports import comports

ports = list(comports())
for p in ports:
    print(p.device, "|", p.description, "|", p.hwid)
dongle = [p.device for p in ports if re.search(r"PID=2458:0*1", p.hwid)]
print("BLED112 dongle:", dongle or "not found")
```

**Verified:** runs; on the documentation machine (no dongle attached) it
printed `BLED112 dongle: not found`. The regular expression is the exact test
`MyoRaw.detect_tty()` applies to the hardware-id string. To override the
automatic choice set `config.MYO_PORT = "COM3"` (Windows) or
`"/dev/ttyACM0"` (Linux; see `install_linux.md`).

---

## 3. `MyoAcquisition`: the headless acquisition API

`from emg_app.acquisition import MyoAcquisition` (also `T.MyoAcquisition` after
`import emg_dataset_tool as T`). It is a `threading.Thread` and contains no Qt.

```python
acq = MyoAcquisition(port=None, mode=None,
                     on_state=fn(state, message),   # connection state changed
                     on_sample=fn(t, values),       # per-sample tap (keep it short)
                     on_log=fn(text))
```

`port=None` uses `config.MYO_PORT` (auto-detect when that is `None`);
`mode=None` uses `config.EMG_MODE`.

| Member | What it does |
|---|---|
| `start()` / `stop()` / `join(timeout)` | thread lifecycle. `stop()` closes the serial port to break a blocked read |
| `state`, `state_message`, `connected` | one of `DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR, STOPPED` (constants `CONN_*` in `emg_app.acquisition`) |
| `ring` | `SampleRing`: `snapshot(n)` -> `(emg int32 [n, channels], t float64 [n])` oldest first; `latest()`; `total`; `capacity`; `clear()` |
| `start_capture(label, repetition, meta=None, max_seconds=None)` -> `Capture` | open the single capture slot. Samples are appended in the acquisition thread, zero-loss, independent of the display ring |
| `stop_capture()` -> `Capture` | close the slot; `Capture.to_recording(person, session)` makes a `Recording` |
| `capturing` | is a capture open (there is exactly one slot: a free take and the cued sequence cannot overlap) |
| `stats()` | dict: `samples, rate, dropped, reconnects, ring_fill, backlog_events, worst_backlog, backlog, last_sample_age` |
| `vibrate(1..3)` | request a haptic cue; safe from any thread because it is queued (rule 2) |
| `unix_time()` | the acquisition clock: monotonic deltas pinned to one wall-clock reading |

Notes that matter when you build on it:

* **`stats()["last_sample_age"]`** (seconds, or `None` before the first sample)
  is the right "is data flowing" test. `acq.connected` can be true while a
  sleeping band supplies nothing. The app's calibration dialog uses it, and
  `tests/test_calibration.py` pins that `MyoAcquisition` has **no** `running`
  attribute (an older version asked for one and could never capture).
* `stats()["rate"]` is a sliding average over the last 64 inter-sample
  intervals, and `dropped` is approximate: the dongle delivers 2-4 buffered
  packets at once, so a handful of "drops" over thousands of samples is normal.
* `Capture.had_disconnect` and `.overflowed` are flagged, never silently
  discarded; `MAX_RECORDING_SECONDS` (120 s) bounds a cued capture, and
  `max_seconds` overrides it for free takes.
* Callbacks run **in the acquisition thread**: do not block in them.
* Settings are read as `cfg.NAME` at call time (`from emg_app import config as cfg`).
  Change them with `setattr(cfg, "NAME", value)` / `cfg.NAME = value` before
  creating the object; never copy a value with `from ... import NAME`.
  Link tuning that matters: `MYO_CONN_TIMEOUT_MS` (3000), `MYO_NEVER_SLEEP`,
  `MYO_FAST_RECONNECT`, `STALL_TIMEOUT_S` (6), `SCAN_TIMEOUT_S` (20),
  `AUTO_RECONNECT`, `RECONNECT_*`, `MYO_BULK_READ`.

### Using the acquisition API against a real band

```python
# Requires hardware, not run here.
import time
import emg_dataset_tool as T

acq = T.MyoAcquisition(on_log=print)
acq.start()
t_end = time.time() + 45
while not acq.connected and time.time() < t_end:
    time.sleep(0.2)
assert acq.connected, (acq.state, acq.state_message)

time.sleep(6)
print(acq.stats())                       # expect rate near 50 (filtered mode)

cap = acq.start_capture("FIST", 1)
time.sleep(3.5)                          # perform the gesture
cap = acq.stop_capture()
rec = cap.to_recording("me", "session_01")
print(rec, rec.emg.shape)                # about 175 samples for 3.5 s at 50 Hz
acq.stop(); acq.join(3)
```

### The same API without hardware: a replay backend

Because the ring, the capture slot and `stats()` all hang off one method,
`_ingest(values)`, a backend that replays a recording needs to override only
`run()`. This is also the **template for a different band**: whatever produces
samples calls `self._ingest(values)` from its own thread.

```python
import time
import numpy as np
import emg_dataset_tool as T
from emg_app.acquisition import MyoAcquisition, CONN_CONNECTED


class ReplayAcquisition(MyoAcquisition):
    """A MyoAcquisition-shaped backend that replays a recording, no hardware.

    Only the thread body changes. The ring buffer, capture slots, stats()
    and the _ingest() hot path are inherited unchanged.
    """

    def __init__(self, recording, speed=20.0, **kw):
        super().__init__(**kw)
        self.rec = recording
        self.speed = speed

    def run(self):
        self._running.set()
        self._set_state(CONN_CONNECTED, "replaying")
        period = 1.0 / (T.nominal_rate(self.mode) * self.speed)
        while self._running.is_set():
            for row in self.rec.emg:
                if not self._running.is_set():
                    break
                self._ingest(row)
                time.sleep(period)
        self._set_state("STOPPED", "replay stopped")


rec = T.load_dataset("datasets", persons=["Dano"], gestures=["FIST"])[0]
acq = ReplayAcquisition(rec)
acq.start()
time.sleep(0.3)
cap = acq.start_capture("FIST", 1)          # a capture slot
time.sleep(1.0)
cap = acq.stop_capture()
emg, t = acq.ring.snapshot(20)              # ring buffer read
s = acq.stats()
print("state:", acq.state, "| ring:", emg.shape, "| capture samples:", len(cap))
print("stats keys:", sorted(s))
print("samples>0:", s["samples"] > 0, "| last_sample_age < 1 s:", s["last_sample_age"] < 1.0)
acq.stop(); acq.join(2)
out = cap.to_recording("demo", "session_01")
print(out, out.emg.shape)
assert len(cap) > 100 and emg.shape == (20, 8)
```

**Verified** (replay at 20x speed, no serial port opened):

```
state: CONNECTED | ring: (20, 8) | capture samples: 592
stats keys: ['backlog', 'backlog_events', 'dropped', 'last_sample_age', 'rate', 'reconnects', 'ring_fill', 'samples', 'worst_backlog']
samples>0: True | last_sample_age < 1 s: True
Recording(demo/session_01 FIST rep1 592 samples raw) (592, 8)
```

(The capture count varies with OS timer resolution: the replay is paced by
`time.sleep(0.001)`, and on Windows that sleeps longer than asked, so about 590
samples arrived in the 1 s capture instead of the nominal 1000.) Run it from the project root so that `datasets` resolves; with
`PYTHONPATH=.` it works from any folder if you pass an absolute dataset path.

The GUI also needs from the acquisition object: `start, stop, join, ring,
state, state_message, connected, capturing, start_capture, stop_capture,
stats, vibrate` (found by searching the code for `acq.<name>`). `MainWindow`
constructs it as `MyoAcquisition(port=, mode=, on_state=, on_log=)`.

---

## 4. Recordings and the dataset API

All of this is importable with no Qt and no torch (`CLAUDE.md` section 4); it
is also re-exported through `emg_dataset_tool`.

```python
import tempfile
import numpy as np
import emg_dataset_tool as T

# --- load recordings and cut windows ---------------------------------------
recs = T.load_dataset("datasets", persons=["Dano", "Betka"], stage="prefer_trimmed")
X, y, groups, meta = T.make_windows(recs, window_ms=1000, overlap=0.75)
print(len(recs), "recordings ->", X.shape, "windows; fs =", meta["fs"],
      "Hz; persons =", sorted(set(groups)))
feats, names = T.make_feature_matrix(X, meta["fs"])
print("features:", feats.shape, len(names), "names")
for train, test, held_out in T.leave_one_person_out(groups):
    print("held out:", held_out, "train", len(train), "test", len(test))

# --- write a recording (to a temp folder, never into datasets/) ------------
rec = recs[0]
with tempfile.TemporaryDirectory() as root:
    w = T.DatasetWriter(root, "demo_person", "session_01")
    written = w.save(rec)
    print(sorted(written))
    back = T.load_dataset(root)
    assert len(back) == 1
    assert np.array_equal(back[0].emg, rec.emg)
    print("round trip identical:", back[0])
```

**Verified** on the five-subject repository data restricted to two wearers:

```
85 recordings -> (811, 50, 8) windows; fs = 49.88085674052743 Hz; persons = ['Betka', 'Dano']
features: (811, 104) 104 names
held out: Betka train 484 test 327
held out: Dano train 327 test 484
['csv', 'meta', 'npz']
round trip identical: Recording(Betka/session_01 FIST rep1 151 samples raw)
```

(`DatasetWriter.save` writes the folder from the writer's person/session, while
the `Recording` keeps the `person_id` it was created with; build the
`Recording` with the right ids from the start. The `groups` returned by
`make_windows` are the recordings' `person_id`s, which is what the folds split on.)

Key objects:

| Name | Role |
|---|---|
| `Recording(emg, timestamps, label, person_id, session_id, repetition, stage, ...)` | one labelled take. `emg` is `(n_samples, n_channels)` int32 counts, **never mutated**; channel count follows the array |
| `DatasetWriter(root, person_id, session_id)` | the on-disk layout; `save(rec)` writes CSV/NPZ/meta and appends the manifest; every write goes through `unique_path()` so nothing is overwritten |
| `load_recording(path)`, `load_dataset(root, persons, sessions, gestures, stage)` | loaders. `stage="prefer_trimmed"` uses a trimmed copy when one exists. Folders starting with `_` (`_takes`, `_deleted`, `_graphs`) are never walked |
| `make_windows(recs, window_ms, overlap, ...)` | `(X [n, win, channels], y, groups, meta)`; windows are resampled to a fixed grid |
| `make_feature_matrix(X, fs)` | the hand-crafted features and their names |
| `leave_one_person_out(groups)` | yields `(train_idx, test_idx, held_out_person)`; this is the honest split |
| `dataset_fingerprint()`, `model_data_status()` | "is the saved model current?" |

Formats and filenames are specified in `docs/DATASET.md`. In short:
`datasets/<person>/<session>/<GESTURE>/<GESTURE>_rep001_<stamp>_<uid>_raw.{csv,npz}`
plus `_meta.json`, a `manifest.jsonl`, and the safety rules of `CLAUDE.md`
section 9 (raw files are immutable, deletes are moves to `_deleted/`,
`person_id` is the key that folds and calibrations are stored under; getting it
wrong corrupts the protocol).

### Calibration is part of the data contract

The shipped model was fitted on per-wearer `affine` calibrated features and
**refuses to run for a wearer without a calibration** (CLAUDE.md sections 8k and
12). A calibration is two per-channel vectors (rest, reference) stored in
`datasets/<person>/calibration.json`, so it is sized by the channel count of
the *record*, and `apply_calibration` returns the block untouched on a width
mismatch rather than broadcasting.

---

## 5. Prediction API

### `LivePredictor` (scikit-learn, CPU, no torch)

```python
import emg_dataset_tool as T

pred = T.LivePredictor()                 # reads cfg.MODEL_PATH
ok = pred.load()                         # refuses (False) if it cannot be reproduced
print("loaded:", ok, "|", pred.status if not ok else pred.describe())

# NOTE: the model needs the wearer's calibration (cfg.PERSON_ID, default "Dano").
rec = T.load_dataset("datasets", persons=["Dano"], gestures=["FIST"])[0]
n = int(pred.window_s * rec.fs) + 2
res = pred.predict(rec.emg[10:10 + n], rec.timestamps[10:10 + n])
print(res.label, res.percent, dict(zip(res.labels, [round(p, 2) for p in res.probabilities])))
```

**Verified** with the model on disk at the time of writing (the numbers change
whenever the model is retrained):

```
loaded: True | Vote-soft-4  |  1000 ms @ 60 Hz  |  level features  |  affine calibration, 74 % per-gesture CV
FIST 48 % {'FIST': 0.48, 'OPEN': 0.31, 'REST': 0.02, 'THUMB_UP': 0.18, 'WAVE_LEFT': 0.01, 'WAVE_RIGHT': 0.0}
```

On a live band, feed it the newest window of the ring buffer exactly as the
app does:

```python
# Requires hardware, not run here (acq is a connected MyoAcquisition).
from emg_app.prediction import latest_model_window
emg, t = latest_model_window(acq, pred.window_s)
res = pred.predict(emg, t)               # None until there are enough samples
```

Properties worth relying on:

* `predict(emg, t)` resamples the window onto the model's own grid first
  (`MODEL_FS = 60 Hz`), so the answer does not depend on how the BLE link is
  behaving.
* There is one feature function, `model_feature_vector`, shared by the trainer
  and the predictor, so the two cannot drift apart.
* A model whose feature count disagrees with its declared feature set, or one
  trained on a calibration the current wearer lacks, is **refused at `load()`**
  rather than answering with nonsense.
* Gestures can be masked out without retraining (`emg_app/class_mask.py`,
  `cfg.CLASSIFIER_OFF`); `Prediction.off`, `.answered` and `.kept` describe the
  mask. An unanswerable window gives `label == "?"` and `answered == False`.
* Loading calls `serialise_estimator()` (single-thread forests): a live row
  costs about 53 ms for the shipped vote instead of 148 ms. Any new code path
  that predicts one live row on a saved model must do the same (CLAUDE.md 8n).

### `NeuralEngine` (the second opinion; torch lives only in a subprocess)

```python
import sys
import emg_dataset_tool as T

eng = T.NeuralEngine()                     # nothing is started by the constructor
print("state:", eng.state, "| trained checkpoint on disk:", eng.trained())
print("torch imported in this process:", "torch" in sys.modules)
```

**Verified:** `state: stopped | trained checkpoint on disk: True` and
`torch imported in this process: False`.

`eng.start()` spawns `training/nn_serve.py` (which imports torch and may take
the GPU; **not run here**), then you call `eng.submit(emg, t)` once per window
(returns `False` when a request is still in flight, which is normal) and
`eng.poll()` to collect `NeuralPrediction`s. States: `stopped, starting, ready,
error`. By design (CLAUDE.md section 8L, rule 13) the net never contributes to
`LivePredictor`'s answer; it returns a different type (`NeuralPrediction`) so
that it cannot be mixed in by accident.

---

## 6. Channel count, device profiles and a NEW band

Since 2026-10-19 the channel count is not a literal. `config.py` holds:

```python
DEVICE_PROFILES = {"myo": {"channels": 8, ...}, "generic_16": {"channels": 16, ...}}
EMG_DEVICE = "myo"
N_CHANNELS = int(device_profile()["channels"])   # derived
set_device("generic_16")                          # recomputes N_CHANNELS, CHANNEL_NAMES
```

**The rule that makes this safe: anything holding real data follows the DATA,
not the config.** A recording made on 8 channels keeps loading after
`EMG_DEVICE` moves to a wider band. Verified below, with the config switched to
16 and back to 8:

```python
import tempfile
import numpy as np
import emg_dataset_tool as T
from emg_app import config as cfg

print("default:", cfg.EMG_DEVICE, cfg.N_CHANNELS, cfg.CHANNEL_NAMES[:3])

cfg.set_device("generic_16")
ring = T.SampleRing(256)
ring.push(0.0, list(range(16)))
print("ring width under generic_16:", ring.snapshot()[0].shape)

rng = np.random.default_rng(0)
emg = rng.integers(0, 500, size=(120, 16))
t = np.arange(120) / 50.0
rec = T.Recording(emg, t, "FIST", "demo16", "session_01")
with tempfile.TemporaryDirectory() as root:
    T.DatasetWriter(root, "demo16", "session_01").save(rec)
    cfg.set_device("myo")                                  # back to 8 ...
    back = T.load_dataset(root)[0]                         # ... still reads 16
    print("config N_CHANNELS =", cfg.N_CHANNELS, "| loaded width =", back.emg.shape[1])
    assert back.emg.shape == (120, 16) and np.array_equal(back.emg, emg)
    X, y, g, meta = T.make_windows([back], window_ms=400, overlap=0.5)
    print("windows:", X.shape)
```

**Verified:**

```
default: myo 8 ['CH1', 'CH2', 'CH3']
ring width under generic_16: (1, 16)
config N_CHANNELS = 8 | loaded width = 16
windows: (11, 20, 16)
```

### What is ready and what is not (CLAUDE.md section 2b)

| Ready (data side) | NOT ready |
|---|---|
| `Recording`, npz/CSV writers and readers at any width (verified at 4, 8, 12, 16 channels; the CSV reader sorts `ch10` after `ch9`) | **acquisition is Myo-only.** `myo_raw.py` is the only band driver; a profile does not make the app talk to anything else |
| metadata `n_channels`, `make_windows`, feature-name generators, calibration sized from the record | **no cross-band model.** A model trained on 8 channels has 8 channels' worth of features; `LivePredictor` refuses a feature-count mismatch at load. There is no channel mapping between bands and none should be invented |
| `SampleRing`, plot, CHANNELS panel follow the config | `CHANNEL_COLORS` and the plot are built once at construction: call `set_device()` **before** starting the GUI |
| | `generic_16` is a placeholder; nothing has been validated against a real second band |

### A second Myo unit versus a different band

* **Another Myo armband** (same hardware family): nothing in the code changes.
  `connect()` scans for the Myo service UUID rather than a name, so the owner's
  `Myo_TUKE` name is only informational. Calibrations are per **wearer**
  (`datasets/<person>/calibration.json`), not per armband; the band-fit
  differences between two units show up as a different per-channel pattern,
  which is the thing calibration does *not* fix (CLAUDE.md 8h). Firmware other
  than `1.5.1970.2` is untested; `connect()` branches on the major version
  (`0` takes the old path), and the new unit prints its version on connect.
  Record the version in your hardware log.
* **A different product** (other channel count, rate or radio) needs, in order:
  1. a profile in `DEVICE_PROFILES` (channels, channel names, label) and
     `EMG_DEVICE` set to it, **before the GUI starts**;
  2. an acquisition backend with the `MyoAcquisition` interface above (the list
     in section 3). The shortest route is a subclass that overrides `run()` and
     calls `self._ingest(values)` per sample, as the replay class does; take
     care that `SampleRing` and `Capture` size themselves from
     `cfg.N_CHANNELS` and `nominal_rate(mode)`, so `NOMINAL_SAMPLE_RATE` and
     `EMG_RANGE` need entries for the band's mode names;
  3. all the rules of section 2 re-derived for that band's transport (one thread
     owns the port, no per-byte reconfiguration, a watchdog for silence);
  4. a **new dataset and new models**. Per-channel calibration, features and
     every trained model are tied to the channel count; the existing 8-channel
     models will refuse to load against 16 channels.

---

## 7. Hardware-test checklist for the day the new band arrives

Do these in order and stop at the first failure; each step assumes the
previous one passed. Run everything from the project root with the project
virtual environment active (see `install_windows.md` / `install_linux.md`).

**0. Before plugging anything in.** Charge the band. Close other programs that
might hold the dongle (`legacy/test_myo.py`, a second app). Note the band's
firmware, serial number and which arm it is worn on.

**1. The dongle is seen.** Plug in the BLED112 and run the "Finding the dongle"
snippet of section 2.
*Expect:* one port with hardware id containing `VID:PID=2458:0001` and
`BLED112 dongle: ['COM3']` (any COM number on Windows, `/dev/ttyACM*` on Linux).
*Failure:* `not found` means a driver/permissions problem, not the band (see the
install documents).

**2. The offline checks still pass** (no hardware involved), to prove the
environment is healthy before blaming the radio:

```
python emg_dataset_tool.py --summary
python tests/test_datalayer.py
python tests/test_device.py
```

**3. The reference monitor connects** (optional, needs `matplotlib`):
`python legacy/test_myo.py`. Wear the band, wait for the eight traces; it
should print the firmware version. A connection here but not in step 4 points
at the app's config, not the band.

**4. The measured stream.** Close the monitor, wear the band snugly in the position you will use for the study, wake it by
flexing, then:

```
python tests/test_live.py
```

*Expect* (from `tests/README.md`, owner's band, `EMG_MODE = "filtered"`): connects
within about 45 s; **measured rate about 50 Hz** (49.6-49.9 over 6 s), about 299
samples in 6 s; baseline amplitude about 15-40 and maximum-effort fist
**1100-1444** (not clipped at 1023; the code allows up to 2047); a few
"dropped" counts are normal (about 5; burst delivery inflates the counter);
a 3.5 s capture gives about 175 samples; the forced-reconnect check recovers.
*Failures:*
* rate near **5 Hz** -> a timeout was reintroduced into `myo.run()` (rule 1);
* `data with unknown attr` lines in the console -> something wrote to the port
  from a second thread (rule 2);
* `Armband did not answer in 20 s` -> the band is asleep or not advertising
  (it stops advertising a minute or two after it is taken off); flex, move, or
  re-seat it;
* `A device attached to the system is not functioning` -> port reconfiguration
  storm or a half-dead dongle; unplug/replug the dongle, then re-check rule 1;
* all channels near zero or constant -> band not worn / poor skin contact;
* a **different** number of values per sample -> not a Myo-shaped stream; stop
  and read section 6.

**5. Link stability** (a measurement, not pass/fail):
`python tests/test_live_stability.py 300` streams for 300 s and reports
dropouts, uptime, effective rate and every gap above 1 s. Compare with
`--baseline` (old settings) if the link is poor. The three link fixes are
`MYO_NEVER_SLEEP`, `MYO_CONN_TIMEOUT_MS` and `MYO_FAST_RECONNECT`.

**6. The application against the band:**
`python tests/test_live_gui.py` and `python tests/test_live_review.py` (follow
the on-screen cues), then `python emg_dataset_tool.py`: watch the connection
state reach CONNECTED and the plot move (the rate itself is easiest to read
from step 4 or from a recording's `sample_rate_measured` in its metadata).

**7. Data sanity before collecting anything.** Record a few REST and FIST
takes; open them and check: 8 channels, rate recorded in the metadata near 50
Hz, REST near baseline, FIST well above it. The dataset `person_id` must be the
actual wearer. Then run **CALIBRATE** for that wearer (six seconds), because
the shipped classifier will not run without it.

**8. Only then** test PREDICT/TEST MODEL on the new wearer and band. Treat any
accuracy as "same wearer, new sitting" until the band has been refitted between
sittings; a different band unit should be reported as a new condition, not mixed
into earlier numbers.

If the new band is *not* a Myo, steps 3-6 do not apply; use section 6.

---

## 8. Known issues found while writing this

* `NeuralEngine.trained` is a **method**, not a property (`eng.trained()`);
  `if eng.trained:` is always true. Documented here so nobody trips on it.
* `legacy/test_myo.py` needs `matplotlib`, which is not in `requirements.txt`.
* There is no public "synthetic Myo" in the code base: the replay backend in
  section 3 was written for this document. The existing GUI tests simulate
  samples by calling `_ingest` directly, which is the same hook.

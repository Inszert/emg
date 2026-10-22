# PROBLEMS.md — everything that went wrong, and what was done about it

A running log of the failures, annoyances and dead ends hit while building the
Myo EMG dataset tool. Written 2026-10-14 from the code, the docs and the
measurements on disk.

This is deliberately **not** a success story. Each entry says what the fix
actually bought, and where a fix is a workaround rather than a cure it says
so. `CLAUDE.md` says what the rules are; this file says why they exist and
what they cost.

### How to read the verdicts

| Verdict | Meaning |
|---|---|
| **SOLVED** | Root cause found and removed. Not expected to come back. |
| **MITIGATED** | Cause understood but not removable (hardware/library). The symptom is handled, reliably. |
| **WORKAROUND** | The symptom is gone, but by avoiding the broken path, not fixing it. Fragile — a refactor can reintroduce it. |
| **PARTIAL** | Better than before, still shows up. |
| **ACCEPTED** | Real, understood, left alone on purpose. |
| **OPEN** | Still a problem. Nothing shipped yet. |

"Observed" = it happened on this hardware and was measured.
"Guarded" = it was reasoned about and designed against before it could bite.

---

# 1. The link: latency, dropouts, dead streams

Everything in this section is about the chain
`Myo → BLE → BLED112 dongle → COM3 → pyserial → myo_raw`. It was by far the
worst area of the project, and almost every problem in it looked like a
different problem than it was.

## 1.1 Gestures showing up ~8 seconds late — while the sample rate looked perfect  ·  **Observed**

**Symptom.** You clench your fist, the graph does nothing, and several seconds
later the burst appears. The rate readout still said ~50 Hz, so for a while
this looked like a GUI/plot problem rather than a data problem. Sometimes the
lag grew slowly over minutes until it was unusable.

**Why it was hard to see.** A *constant* delay does not change the spacing
between samples. Every rate estimate was correct and every sample was present
— they were just all old. The delay was in the OS serial buffer, where
nothing in the app was looking.

**Cause.** `myo_raw.BT.recv_packet()` reads **one byte per `ser.read()` call**
and re-assigns `ser.timeout` around it. On Windows, assigning `ser.timeout`
reconfigures the COM port. That reader only just keeps up with a 50 Hz,
8-channel stream, so any hiccup (a slow GUI frame, another process, a USB
power event) left a backlog in the kernel buffer that the reader could never
drain, because it could not read faster than the armband sends.

**Fix.** Three settings, all in the `USER CONFIGURATION` block:

| Setting | Where | What it does |
|---|---|---|
| `MYO_BULK_READ = True` | `emg_dataset_tool.py:539` | each pass reads **everything** waiting in one `ser.read(in_waiting)` and feeds it to `myo_raw`'s own byte parser (`MyoAcquisition._pump`, `:4478`). The timeout is set **once** instead of per byte. |
| `MYO_FLUSH_ON_START = True` | `:543` | drop whatever is queued before streaming, so a session starts at "now" instead of replaying the connect handshake |
| `MYO_MAX_BACKLOG_BYTES = 4096` | `:550` | a watchdog on the backlog itself (`_handle_backlog`, `:4503`) |

**Verdict: SOLVED** for the growing lag — the reader now outruns the armband
and the backlog sits at zero. A `lag` readout was added to the status bar so
this can never be invisible again.

**Is the fix ideal?** No, and knowingly so. It **bypasses** `myo_raw`'s own
read loop while still using its byte parser, so there are now two ways to pump
the port (`_stream` falls back to `myo.run()` when `MYO_BULK_READ = False`).
That is duplicated responsibility. The ideal fix is to repair
`BT.recv_packet()` upstream, which was rejected on purpose: `myo_raw.py` is
the one layer that is known to work, and `test_myo.py` / `myo.py` still depend
on its original behaviour.

**The backlog guard is weaker than the read fix, and is a deliberate
compromise.** When the backlog exceeds the limit it *discards* data to stay
live — but only **outside** a recording. During a capture it only logs
`"Running N s behind"`, because a hole in a recording is worse than latency.
So during a long recording the lag can still build up. Accepted.

## 1.2 Stream throttled to 4.6 Hz, then "A device attached to the system is not functioning"  ·  **Observed**

**Symptom.** 50 Hz stream measured at **4.6 Hz**, then the port died outright
with a Windows device error. Looked like a broken dongle or cable.

**Cause.** Calling `myo.run(1)` instead of `myo.run()`. With a timeout
argument, `recv_packet` re-assigns `self.ser.timeout` **before every single
byte** — thousands of COM-port reconfigurations per second. Windows
eventually gave up on the port.

**Fix.** Call `myo.run()` with **no argument**, exactly as the known-good
`test_myo.py` does. Documented in `MyoAcquisition._stream`'s docstring
(`:4431`) and in `CLAUDE.md` §3 as a do-not-touch rule.

**Verdict: SOLVED.**

**Is the fix ideal?** It is correct, but it is *load-bearing and invisible*: a
one-character "improvement" by anyone who does not know this re-breaks the
whole tool, and the failure mode (a slow stream, not an exception) does not
look like a regression. Three separate mitigations exist only because of it:
the watchdog, the bulk reader, and a troubleshooting row in `GUIDE.md`
("Rate readout shows ~5 Hz → something reintroduced a timeout"). So the fix
works perfectly and is still fragile.

**Cost of the fix.** `run()` with no timeout blocks in a bare serial read when
nothing arrives, so the stream thread **cannot notice its own stall**. That
created problem 1.5.

## 1.3 The link dropped every couple of minutes  ·  **Observed**

**Symptom.** `MYO DISCONNECTED`, repeatedly, during otherwise normal
sessions. Initially blamed on distance and interference, which was wrong.

**Cause.** Three independent causes, which is why single fixes kept
"half-working":

1. **The armband puts itself to sleep** after a short idle period, and
   streaming EMG does **not** keep it awake.
2. **The BLE supervision timeout** in `myo_raw` is 640 ms — short enough that
   a hand over the dongle or a busy 2.4 GHz band kills the link.
3. **Connection interval** needed to stay at the Myo default (7.5 ms).

**Fix.** One setting per cause, all applied in `MyoAcquisition._connect`:

- `MYO_NEVER_SLEEP = True` (`:513`) — `set_sleep_mode(never_sleep)`
- `MYO_CONN_TIMEOUT_MS = 3000` (`:519`) — raised from 640 ms
- `MYO_CONN_INTERVAL_MS = 7.5` (`:525`)

`myo_raw.py` keeps its own original defaults so `test_myo.py` and `myo.py` are
unaffected; the values are pushed in from the application.

**Verdict: MITIGATED → effectively solved.** Measured result: a link that
dropped every couple of minutes now holds. This was the single biggest
quality-of-life improvement in the project.

**Is the fix ideal?** Close to it, with one real cost: `MYO_NEVER_SLEEP`
**burns battery**, so the armband must be charged between sessions, and a flat
battery produces the *same* disconnect symptom as the original bug. Noted in
`GUIDE.md` §6 ("`MYO DISCONNECTED` every few minutes → low battery").

## 1.4 Stuck forever on "Opening the Myo dongle…"  ·  **Observed**

**Symptom.** The app hangs on scanning. No error, no timeout, no way to tell
whether the dongle, the armband or the app was at fault.

**Cause.** `MyoRaw.connect()` loops on `recv_packet()` with **no bound** while
scanning. If the armband is not advertising, that loop never exits. The
armband stops advertising a minute or two after you take it off.

**Fix.** `SCAN_TIMEOUT_S = 20.0` (`:571`), enforced from
`MyoAcquisition._watchdog` (`:4526`) — not from `connect()`, which cannot be
interrupted from inside. On timeout the port is force-closed, which turns the
blocked read into an exception the reconnect loop already handles, and the
user gets an actual explanation: *"it is probably asleep. Pick it up, flex, or
move it to wake it."*

**Verdict: SOLVED** as a usability problem. The armband still has to be woken
by hand — that is the hardware.

**Is the fix ideal?** No. **Closing the serial handle to break a blocking read
in another thread is a sledgehammer**, and it is used in three places now
(scan timeout, stall, `stop()`). It works and is deliberate — `CLAUDE.md` §3
calls it out so nobody "fixes the leak" — but it means shutdown and recovery
both rely on provoking an exception rather than on cooperative cancellation.

## 1.5 EMG data just stops — no error, no exception, nothing  ·  **Observed**

**Symptom.** The traces flatline and stay flat. Connection still shows green.
No serial error, because nothing is wrong with the port: the armband simply
stopped talking.

**Cause.** A sleeping or disconnected armband is silent, not faulty. And after
fix 1.2, the stream thread is blocked inside a bare serial read and cannot
time out to check.

**Fix.** `MyoAcquisition._watchdog` (`:4526`), a separate thread ticking every
250 ms. If no EMG sample has arrived for `STALL_TIMEOUT_S = 6.0` (`:578`) it
logs the cause and force-closes the port, which converts the blocked read into
an exception the existing reconnect loop handles. `STALL_TIMEOUT_S` is kept
comfortably above the 3 s supervision timeout on purpose — tearing the link
down and rebuilding it costs more than waiting out a brief gap. There is a
debounce so it cannot fire again while a reconnect is in flight.

**Verdict: MITIGATED.** Recovery is automatic and the interrupted repetition
is retried rather than saved half-empty.

**Is the fix ideal?** It is the only option given 1.2, but it is a 6-second
blind spot by construction: a stall is detected no faster than
`STALL_TIMEOUT_S`, and lowering it risks tearing down a healthy link over a
normal BLE gap. A cooperative reader with a real non-blocking select loop
would be better and would also let 1.1's bulk reader go away. Not attempted.

## 1.6 Reconnects took 5–10 seconds, or failed completely  ·  **Observed**

**Symptom.** After a dropout, recovery was slow, and if the armband had been
off the arm for a minute it never reconnected at all — it re-scanned forever
(and then hit 1.4).

**Cause.** Reconnecting re-ran the **BLE scan**, which only succeeds while the
armband is advertising.

**Fix.** `MYO_FAST_RECONNECT = True` (`:556`) — reconnect straight to the
armband's already-known address, skipping the scan, with an automatic fallback
to scanning on the next attempt if the direct connect fails. Plus exponential
back-off (`RECONNECT_DELAY_S` 2.0 → `RECONNECT_MAX_DELAY_S` 10.0,
`RECONNECT_MAX_ATTEMPTS = 0` = retry forever).

**Verdict: SOLVED.** Measured ~1 s with it versus 5–10 s with a scan.

**Is the fix ideal?** Yes for the cost. It depends on having seen the armband
once in this process (`self._known_addr`), so the *first* connect of a session
is still a scan and still slow. Acceptable.

## 1.7 Reusing a dead `MyoRaw` does not work  ·  **Observed (in `test_myo.py`)**

**Symptom.** After a dropout, reconnecting on the existing object silently
fails to stream.

**Fix.** Reconnect = construct a **fresh** `MyoRaw` and explicitly close the
old serial port. `MyoAcquisition._connect` does exactly this.

**Verdict: SOLVED.** No downside found. Established before the GUI existed,
which is the only reason it did not cost a debugging session later.

## 1.8 EMG notifications turning into "data with unknown attr" garbage  ·  **Observed**

**Symptom.** The stream desynchronised mid-session: EMG stopped arriving and
the console filled with unknown-attribute messages. Looked like random
corruption.

**Cause.** A `vibrate()` **write from the GUI thread**. `BT` shares one serial
port and one packet-assembly buffer across all callers and takes **no lock**
while reading, so a write from a second thread lands in the middle of a packet
and the framing never recovers.

**Fix.** Haptic cues are now *queued*, never written directly:
`MyoAcquisition.vibrate()` (`:4245`) only sets `_vibrate_request`, and
`_service_vibrate()` (`:4261`) performs the write **from the acquisition
thread, between two packets**. Promoted to a project-wide rule in `CLAUDE.md`
§3: only the acquisition thread may touch the serial port, and anything else
that wants to write must use the same queue.

**Verdict: SOLVED.**

**Is the fix ideal?** The pattern is right. The enforcement is not — it is a
**convention, not a mechanism**. `myo.bt.ser` is still reachable from anywhere
in the process, so nothing stops the next feature from writing directly and
reintroducing exactly this bug. A lock inside `BT`, or a wrapper that refuses
writes off-thread, would be the real fix. Not done, because it means editing
`myo_raw.py`.

## 1.9 `EMG_MODE = "raw"` flooded the console 200×/s  ·  **Observed**

**Cause.** `myo_raw.handle_data` printed `data with unknown attr` for the raw
EMG attributes `0x2b/0x2e/0x31/0x34`.

**Fix.** One of only two edits made to `myo_raw.py`: those four attrs are now
silently ignored. Purely additive; no existing behaviour changes.

**Verdict: SOLVED.** Ideal — it removes noise and nothing else. It is also the
reason the project tolerates having modified `myo_raw.py` at all.

## 1.10 `vibrate()` always raised `NameError`  ·  **Observed**

**Cause.** Python-2 `xrange` left in `myo_raw.vibrate()`. Haptic cues had
never worked.

**Fix.** `range`. The second and last edit to `myo_raw.py`.

**Verdict: SOLVED.** Ideal.

## 1.11 Missing packets / the `dropped` counter that cries wolf  ·  **Observed, ACCEPTED**

**Symptom.** A non-zero `dropped` count on sessions that looked perfectly
clean, and `gap` warnings from `check_quality()` (`:1710`) on good data.

**Cause.** The dongle hands over **2–4 buffered packets at once**, microseconds
apart, then goes quiet until the next burst. Delivery is bursty by nature, so
any heuristic based on the inter-sample interval over-counts. Measured
distribution over 8061 intervals in the current session: median 20.6 ms, p95
28.0 ms, max 41.2 ms — the *mean* rate is an honest 49.9 Hz but **almost no
sample arrives on time**.

**Fix / decision.** The counter is kept and documented as **approximate**, in
`CLAUDE.md` §12 and as a `GUIDE.md` §6 row ("a handful over thousands of
samples is fine; a number near the sample count is not"). Real loss is caught
separately by the gap check and the dropout flag, and nothing downstream is
allowed to count samples instead of seconds.

**Verdict: ACCEPTED.** Not ideal — an operator still sees a scary non-zero
number and has to be told to ignore it. A proper fix would compare against the
armband's own sequence numbering rather than wall-clock spacing. Worth doing;
not done.

## 1.12 Sample rate reported as 9258 Hz for a 57 Hz stream  ·  **Observed**

**Symptom.** Absurd sample rates in metadata, which would have silently
corrupted every window length and every feature downstream.

**Cause.** Estimating the rate as `1/median(diff(t))`. Because of the burst
delivery in 1.11, the median interval is an **intra-burst** interval.

**Fix.** `measured_rate()` (`:1836`) uses the **span average** — samples per
second over the recording's whole duration. A long dropout correctly *lowers*
this number, because that is the rate actually achieved; gaps are reported
separately.

**Verdict: SOLVED,** and ideal for this hardware. This one is cheap to
re-break, so it carries its own warning in `CLAUDE.md` §12.

## 1.13 A session that kept counting down while nothing was arriving  ·  **Guarded**

**Symptom (anticipated and designed against).** A dropout mid-sequence meant
repetitions kept advancing against a dead link, collecting empty recordings
that look like real ones.

**Fix.** `PAUSE_SESSION_ON_LINK_LOSS = True` (`:563`) pauses the running
sequence on link loss and resumes it when the armband is back; the interrupted
repetition is **not** counted, so the planned per-gesture counts still come
out right. The partial recording is discarded rather than saved
(`emg_dataset_tool.py:4963`).

**Verdict: SOLVED.** Ideal. This is the one place where throwing data away is
correct, and it is the only such place.

## 1.14 An armband on the desk produces dropout warnings  ·  **Observed, ACCEPTED**

**Cause.** An armband that is **not being worn stops sending EMG** after a few
seconds. The watchdog correctly calls that a dropout.

**Verdict: ACCEPTED — working as designed.** Unpleasant anyway: every bench
test without wearing the band produces alarming log output, and it trains the
operator to ignore stall messages, which is the opposite of what you want.
Documented in `GUIDE.md` §6 and `CLAUDE.md` §12 rather than suppressed,
because suppressing it would hide real dropouts.

## 1.15 `could not open port 'COM3': Access is denied`  ·  **Observed, ACCEPTED**

**Cause.** Only one process can hold the dongle. `test_myo.py`, Myo Connect,
or a second copy of the tool will take it.

**Fix.** Documentation only (`GUIDE.md` §6, `CLAUDE.md` §10) plus
`MyoRaw.detect_tty()` finding the dongle by USB PID so `MYO_PORT = None`
normally works.

**Verdict: ACCEPTED.** The error message is clear enough. A friendlier
"something else is using the dongle" dialog would be nicer; low priority.

---

# 2. Recording and data integrity

## 2.1 The gesture happening outside the labeled window  ·  **Observed → designed around**

**Symptom.** An early reaction to the countdown, or an explosive contraction
overshooting the end of the window, meant the interesting part of the gesture
was **outside** the saved label window — and in the earliest version, outside
the file entirely.

**Fix.** The **capture window is wider than the labeled window**:
`PRE_GESTURE_MS` before and `POST_GESTURE_MS` after. The label boundaries are
stored as metadata (`label_start_index` / `label_end_index`) and as the
`in_label_window` CSV column rather than by cropping. Nothing is cut at record
time.

**Verdict: SOLVED,** and the best decision in the recording path — it makes a
labelling mistake *recoverable* instead of permanent. Cost: files are bigger
than the labels they carry, and every consumer has to decide whether it wants
the label window or the whole file.

## 2.2 Scrolling a panel silently changed recording settings  ·  **Observed**

**Symptom.** The mouse wheel passing over a combo box or spin box in a tall
side panel changed its value — the gesture, the window length, a
hyperparameter — with no indication. Those values go into recording metadata,
so this is a **data-integrity bug, not a UI nuisance**.

**Fix.** `WheelGuard` (`:5279`), an application-wide event filter: a wheel
event over a combo box, spin box or slider is only accepted when that control
has **keyboard focus**, i.e. the operator clicked into it deliberately.
Otherwise the event goes to the scroll area, which is what they meant.

**Verdict: SOLVED.** Ideal — one filter, installed once, covers every current
and future control of those types.

## 2.3 "Could a bug overwrite my data?"  ·  **Guarded**

The whole of `CLAUDE.md` §9 exists because of this worry, which is the right
worry for a dataset tool where a lost recording cannot be re-recorded: raw
files are immutable, every write goes through `unique_path()`, processing
creates **new** files with a new stage suffix, DELETE LAST quarantines into
`_deleted/` and never calls `os.remove`, and bad recordings are flagged rather
than discarded.

**Verdict: SOLVED by construction,** and it held — see 4.3, where a dataset
was lost anyway, but *not* by the tool.

## 2.4 A five-minute free take becoming one "gesture"  ·  **Guarded**

**Risk.** Loading a long, mostly-rest free take as one labelled gesture would
quietly wreck the model — and quietly is the dangerous part.

**Fix.** Two independent mechanisms, on purpose: takes live in `_takes/` and
`discover_recordings()` prunes every folder beginning with `_`; **and** they
carry stage `take`, which no stage selector asks for. Both are checked in
`tests/test_free_take.py` (195 checks).

**Verdict: SOLVED.** Belt-and-braces deliberately, because a single guard that
silently fails looks exactly like a guard that works.

## 2.5 Twenty clips that looked like twenty versions of one recording  ·  **Caught in design**

**Cause.** `uid` is what pairs a raw file with its trimmed copies. Clips cut
from one take would have inherited the take's uid.

**Fix.** Each clip gets its **own** uid; provenance moves to metadata
(`parent_take_uid`, `take_start_index` / `take_end_index`,
`capture_style: "cut_from_take"`). Subject metadata on a clip is copied from
the **take**, not from today's config — otherwise a clip cut weeks later would
claim the wrong subject.

**Verdict: SOLVED.** Ideal.

## 2.6 Annotating a recording must not touch the recording  ·  **Guarded**

**Fix.** Notes on a saved take go to a `*_notes.json` **sidecar**; the marker
log (`markers.jsonl`) is append-only; a correction is another marker, never an
edit. `*_raw.*` stays byte-identical, which `tests/test_markers.py` checks at
both the data layer and through the dialog.

**Verdict: SOLVED.** Ideal.

## 2.7 Auto-trim cutting the gesture in half, or keeping the idle part  ·  **Observed, PARTIAL**

**Symptom.** The proposed cut either split a two-phase gesture at its dip or
swallowed seconds of idle signal.

**Causes and fixes.** `detect_activity()` (`:1239`) picks the **strongest**
region, not the longest, so a 150 ms explosive burst beats a second of drift;
`TRIM_MIN_INACTIVE_MS` bridges the dip between two phases; the operator always
sees the proposed cut drawn over the frozen take in the review step and can
REDO. When detection finds nothing, the band covers the **whole** take — it
never drops data because detection failed.

**Verdict: PARTIAL / ACCEPTED.** The thresholds are subject- and
placement-dependent and there is no automatic calibration, so four separate
`GUIDE.md` §6 rows exist to tell the operator which knob to turn. The real fix
is per-session calibration of the thresholds; that is part of the open work in
4.2.

---

# 3. Application and tooling

## 3.1 PyQtGraph's SVG exporter crashes on these plots  ·  **Observed**

**Symptom.** `ValueError: not enough values to unpack` inside
`correctCoordinates` in pyqtgraph 0.14 — every SVG export failed.

**Fix.** SVG goes through Qt's own `QSvgGenerator` with a `QPainter` render of
the scene (`EmgPlotWidget.export_image`, `:6231`). PNG still uses pyqtgraph's
`ImageExporter`.

**Verdict: WORKAROUND, and it is the right one.** It produces a true vector
file and additionally lets the caption and metadata be embedded in the SVG
header. Not ideal in one respect: the output is a *scene render*, so some
browsers render it oddly (`GUIDE.md` §6 says to use PNG for sharing).
`CLAUDE.md` §12 explicitly forbids "simplifying" it back to the pyqtgraph
exporter.

## 3.2 Training starved the acquisition thread and hung the test suite  ·  **Observed**

**Symptom.** scikit-learn with `n_jobs=-1` saturated every core; the
acquisition thread stopped getting scheduled and **the test suite timed out**.
On top of that, a crash inside training would have taken the recorder down
with it.

**Fix.** Training runs as a **subprocess**, never a thread (`train_model.py`
launched from the GUI). RETRAIN is also disabled while a recording session is
running.

**Verdict: SOLVED.** Ideal — it fixes the starvation *and* the blast radius
with one decision. Cost: progress and errors have to be parsed out of the
subprocess's stdout, and the same subprocess pattern then had to be repeated
for the compute probe (3.4).

## 3.3 The GUI retrained on, and overwrote, the wrong dataset and model  ·  **Observed**

**Symptom.** A changed `DATA_FOLDER` / `MODEL_PATH` in the GUI meant the
trainer silently used **its own defaults** — so RETRAIN read a different
dataset and overwrote a different model than the one the app had loaded. This
actually happened in testing.

**Fix.** The GUI passes `--root`, `--out-dir` and `--model-path` explicitly on
every retrain, and `--spec` so that what the subprocess trains is always what
the MODEL SPEC dialog was showing.

**Verdict: SOLVED.** Ideal in effect, but a reminder that the trainer having
usable defaults at all is what made the bug possible.

## 3.4 The recorder must not import torch  ·  **Guarded**

**Problem.** Only torch can answer "is a CUDA GPU actually going to be used?",
but importing torch into the recorder adds seconds to startup and a GPU
context to a process whose single job is to not drop EMG samples.

**Fix.** The probe runs as a tiny, dependency-free subprocess snippet
(`COMPUTE_PROBE_SNIPPET`, `:2308`) that prints one JSON line;
`parse_compute_probe()` never raises, and "the probe did not answer" is
reported as a *different* thing from "no GPU". The GUI runs the same snippet
through `QProcess` so the event loop never stalls; `detect_compute_device()`
(`:2364`) is the blocking version for headless callers and tests.

**Verdict: SOLVED.** Ideal. One related honesty fix: the badge's headline says
what the **hardware** offers, and the detail says what each trainer actually
uses — because the scikit-learn sweep behind RETRAIN has no GPU backend and
runs on the CPU whatever card is installed. A badge reading "GPU" next to a
CPU-only button would have been a lie.

## 3.5 A hand-edited `model_spec.json` could crash a sweep an hour in  ·  **Guarded**

**Fix.** `validate_model_spec()` is the single gate, and it **never raises and
never returns something the trainer cannot build**: an unknown kind is
dropped, a bad value coerced, an out-of-range value clamped, a duplicate name
renamed, and a spec with everything disabled gets one model switched back on.
Every correction is returned as a string the GUI shows and the trainer logs.
It is Qt-free *and* sklearn-free so there is exactly one definition of what a
spec means. 332 checks in `tests/test_model_spec.py`.

**Verdict: SOLVED.** Ideal.

## 3.6 Smaller annoyances that were fixed and are worth not re-breaking

| Problem | Fix | Verdict |
|---|---|---|
| Per-sample Qt signals at 200 Hz flooded the event loop | acquisition signals **state** only; the GUI reads samples on its own timer ("GUI pulls, acquisition pushes") | SOLVED, ideal |
| A dropped GUI frame shifted the whole sequence | `SequenceController` compares against wall-clock **deadlines**, not tick counts | SOLVED, ideal |
| Clock adjustments mid-session corrupted timestamps | monotonic timestamps pinned to one wall-clock reading | SOLVED, ideal |
| A peak got clipped out of view, or the axis jittered | autoscale **grows instantly, shrinks slowly** | SOLVED, ideal |
| Mean-mode downsampling smoothed away the spikes that matter | peak-mode downsampling in the plot | SOLVED, ideal |
| A subject starting early poisoned the baseline estimate | **median**, not mean, for baselines | SOLVED, ideal |
| Fatigue and electrode drift baked into the last gesture | interleaved gesture order by default | MITIGATED — spreads it, cannot remove it |
| Headless / offscreen runs crashed with no screen | `available_screen_size()` falls back to the primary screen, then to 1280×720 | SOLVED |
| Fusion's selection colour was a saturated magenta that appears nowhere else | `SELECTION_QSS` uses the accent colour | cosmetic, SOLVED |

---

# 4. Data quality and modelling — the expensive problems

The link problems cost days. These cost a **dataset**.

## 4.1 43 % on six gestures, and it was the data  ·  **Observed 2026-10-06**

**Symptom.** 72 recordings, one subject, one session. Best result ≈ **43 %**
per window / 47 % per gesture against a 16.7 % chance level.

**Diagnosis.** Not the model. **15 of 72 recordings were flagged
`no_activity`** (5 of 12 FIST), and FIST's mean envelope varied **26-fold
across its own repetitions**. No classifier recovers from that.

**Action.** Recorded in `CLAUDE.md` §8 as "the dataset is the bottleneck, not
the model: more and stronger repetitions are worth far more than further model
tuning", and model tuning was stopped.

**Verdict: OPEN — correctly diagnosed, not yet fixed.** The honest conclusion
was reached instead of chasing it with bigger models, which is the only good
part of this entry.

## 4.2 Rest was louder than the contractions, and quality control passed it  ·  **Observed, OPEN**

This is the worst failure in the project.

**Symptom.** In the surviving 6-gesture bundle, median total activation is
`REST` **451** and `FIST` **208** — rest was *louder than the fist*. **83 % of
all non-REST windows sit below REST's own 90th percentile.** The gestures are
not separable and no model will fix it.

**Cause, in two parts.**

1. **Recording.** That session's resting baseline was ~130 counts on CH3 and
   ~83 on CH2, against 24/23 in a good session — five times the resting level.
   Loose band, poor contact, or dry/unshaved skin.
2. **Quality control missed it.** `check_quality()` flags a non-REST recording
   that never exceeds `QC_MIN_PEAK_AMPLITUDE` — **40 counts, absolute**. A
   130-count resting baseline clears 40 on its own, so every bad recording
   **passed**.

**Fix attempted.** None in code yet. The fix is specified in
`recommendation.md` §4.1–4.2 and is the #1 item in its priority table
("small effort, **decisive** — without it the next dataset fails the same
way"): a **contrast check relative to the session's own rest**
(`QC_CONTRAST_MIN_RATIO ≈ 3.0`, a `low_contrast` warning raised while the
session is still running and therefore still fixable), plus a live
"**× rest**" readout next to the amplitude, because a subject who can see
"1.2× rest" will squeeze harder while one reading raw counts has no idea
anything is wrong.

**Verdict: OPEN.** `grep QC_CONTRAST emg_dataset_tool.py` returns nothing —
the check exists only as a recommendation. **Until it ships, the next dataset
can fail in exactly the same way, silently.** That makes this the
highest-value open item in the project.

## 4.3 The 6-gesture dataset is no longer in `datasets/`  ·  **Observed, OPEN**

**Symptom.** `python emg_dataset_tool.py --summary` today:

```
Raw recordings: 119   Trimmed: 0
  Dano / session_01 / FIST 56, REST 63
```

The six-gesture set described in `CLAUDE.md` §8 is **gone from the dataset
tree**. It survives only as one export bundle,
`exports/dataset_20261006-143332-241.npz` (1284 windows, 72 recordings, six
labels). On 2026-10-13 the tree held 62 recordings and **all 62 were `REST`**.

**Cause.** Not established. The tool's own preservation rules (2.3) mean it
did not delete anything — nothing in it calls `os.remove`, and `_deleted/`
quarantines rather than erases. The loss happened outside the tool.

**Verdict: OPEN, and partly unrecoverable.** The windowed export is usable for
analysis but it is **not** raw recordings: no per-recording files, no
trimming, no re-windowing at a different length. The practical consequence is
that `CLAUDE.md` §8's "current state of the model" describes data that is no
longer on disk, and `recommendation.md` §4.3 ("re-collect the 6-gesture set
before any modelling") is now mandatory rather than advisory. No backup scheme
exists — worth adding, since the tool protects against its own bugs but not
against anything else.

## 4.4 99.7 % accuracy that means almost nothing  ·  **Observed 2026-10-14**

**Symptom.** `train_torch.py` (CUDA CNN-LSTM) scored **99.7 % per window /
99.2 % per recording** on the 119-recording FIST/REST set. The grouped
cross-validation is sound, so the number is real.

**Why it is nearly worthless.** A **single hand-picked amplitude threshold**
(53.5 counts on mean label-window amplitude) scores **98.3 %** on the same
data, and an LDA on the sklearn feature vector scores **100 %**. REST median
23.7, FIST median 108.8 — a 4.6× gap, nearly disjoint. So the entire deep
stack buys **+0.9 points** over one `if`, and *loses* to the cheapest linear
model in the roster.
FIST-vs-REST asks "is the muscle contracting at all", the easiest question on
EMG. The hard problem is discriminating among **active** gestures, where
amplitude is useless and only the spatial pattern across the 8 pods separates
them — which is where the 43 % in 4.1 comes from.

**Fix.** Report the **majority-class baseline (52.9 %)** and the one-threshold
baseline alongside any accuracy on this dataset, and do not treat a high
two-class score as validation of the model family or of the CUDA path.
Re-measure once a third active gesture exists.

**Verdict: SOLVED as a reporting problem, OPEN as a validation problem.** The
trap here is that nothing was broken — a correct pipeline produced a correct
number that invited a wrong conclusion. Caught by checking a stupid baseline,
which is now the standing rule.

## 4.5 ~99 % accuracy from splitting windows randomly  ·  **Guarded**

Windows from the same recording land in both train and test, so the model
memorises recordings. Fixed by making `groups` and `leave_one_person_out()`
the default path (`docs/ML_PIPELINE.md` §5), and `GUIDE.md` §6 carries the row
"Accuracy is ~99 % and you do not believe it → you are right not to."

**Verdict: SOLVED.** Ideal. Leave-one-subject-out is the number that did not
pick its own folds.

## 4.6 Two feature-engineering choices that measurably hurt  ·  **Observed**

| What was tried | Result | Outcome |
|---|---|---|
| `extract_features()` (median-centres each channel) on the `filtered` stream | the filtered stream is already a rectified envelope, so centring **deleted the activation level**. Fisher ratio 25.5 (level) vs 15.1 (centred) | switched to `extract_level_features()` (`:1423`) for `filtered`; accuracy 33.5 % → **43.2 %** on half the features. SOLVED |
| Baseline subtraction per recording | Fisher 24.9 → **11.5–17.2** — within a session the offsets are constant, and a per-recording estimate partly cancels the gesture | reverted; absolute amplitude is kept. Offset/gain invariance belongs in a future per-user calibration. SOLVED |

Both were real regressions caught by measuring rather than by reasoning, which
is why the numbers are kept in `CLAUDE.md` §13 — a future "obvious cleanup"
would reintroduce either one.

## 4.7 Train and inference features drifting apart  ·  **Guarded**

**Risk.** If `train_model.py` and `LivePredictor` computed features even
slightly differently, live predictions would be garbage while the
cross-validation score stayed perfect — the hardest class of bug to notice.

**Fix.** One function, `model_feature_vector()` (`:1494`), called by both. A
model whose feature count disagrees with its declared `feature_set` is
**rejected at load time with an explanation** rather than producing nonsense
at 7 Hz.

**Verdict: SOLVED.** Ideal.

## 4.8 Uneven sampling breaking fixed-length windows  ·  **Observed**

**Cause.** The stream is 28–70 Hz and unevenly sampled (1.11), so a fixed
*sample count* means a varying real *duration*.

**Fix.** Everything is resampled to `MODEL_FS = 60 Hz` via
`resample_uniform()` (`:1572`), in training and in `LivePredictor` alike, and
a window means a fixed **duration**.

**Verdict: SOLVED.** Ideal, and it is the reason the bursty sampling in 1.11
can be accepted rather than fought.

## 4.9 ~50 Hz is thin, and the `level` features delete wave direction  ·  **Observed, OPEN**

The filtered stream runs at ~50 Hz, which is marginal for fast gestures, and
the `level` features average over time — deleting exactly what separates
`WAVE_LEFT` from `WAVE_RIGHT`. `recommendation.md` §6.3 specifies dynamic
features and multi-scale windows so one model covers both a 150 ms burst and a
2 s hold.

**Verdict: OPEN.** `EMG_MODE = "raw"` (200 Hz) would help and is implemented,
but it is the **less-travelled path** — reviewed, never validated against a
live armband over a long session. Sample rate and amplitudes must be verified
before collecting a real dataset with it (`CLAUDE.md` §12).

## 4.10 A silent CPU fallback that looks exactly like a working GPU run  ·  **Guarded (`train_torch.py`)**

**Problem.** A torch trainer that cannot reach CUDA and quietly runs on the
CPU produces an identical-looking run that simply takes ~20× longer. You find
out an hour in, if at all.

**Fix.** The device contract is **fail loudly, never fall back**: `--device
cuda` is the default and `resolve_device()` raises `SystemExit` with the exact
install command rather than downgrading. Every run prints a boxed
`COMPUTE: >>> GPU <<<` / `>>> CPU <<<` banner at the **start and at the end**,
so a run cannot be misremembered either.

**Verdict: SOLVED.** Ideal. Verified on this machine 2026-10-14: GTX 1650,
torch 2.9.1+cu128, CUDA 12.8, `tests/test_torch.py` 158/158 on the real GPU.

## 4.11 Two ways the torch sweep would have cheated  ·  **Guarded**

| Trap | Why it is a trap | What was done |
|---|---|---|
| Scaling fitted over the whole dataset | `log1p` + per-channel standardisation over all data leaks the test distribution into training; the score goes up and means less | fitted on the **training fold only** |
| Early stopping on the held-out fold | stopping when the test fold looks best *is* fitting the test set | **no early stopping**; epochs are fixed and declared |

**Verdict: SOLVED, both.** These cost nothing to get right up front and are
close to unfixable after the fact, because the inflated number looks normal.

## 4.12 "Same seed" fits that were not the same  ·  **Observed**

**Symptom.** Two runs with an identical seed produced different models.

**Cause.** The seed was taken in `train_one()`, but **weight initialisation
happens at construction** — so by the time the seed was set, the starting
point was already wherever the global RNG had got to.

**Fix.** Seed in `build_model()`, before construction. `tests/test_torch.py`
checks it by comparing **probabilities, not the argmax**, because on a
separable problem like FIST/REST the argmax agrees even between two completely
unrelated fits — an argmax test here would have passed while broken.

**Verdict: SOLVED.** Ideal, and the test design is the more valuable half:
the obvious assertion would have hidden the bug.

## 4.13 Selection bias in AUTO mode  ·  **ACCEPTED, stated not hidden**

`expand_auto_spec()` turns each enabled model into up to `max_per_model`
concrete candidates, which then go through the **existing** evaluation path —
there is no second scoring route to keep honest, every combination appears as
its own row, and the sampling is seeded so a sweep is reproducible.

**The cost is real:** choosing the best of N candidates on the same folds makes
the winner's score optimistic.

**Verdict: ACCEPTED.** The bias is stated in the dialog, in the training log
and in MODEL STATS rather than hidden, and leave-one-subject-out is quoted as
the number that did not pick its own folds. Not ideal — a nested
cross-validation would be — but honest, and `CLAUDE.md` §9 rule 11 makes
"selecting a model never flatters it" an invariant.

## 4.14 Nothing in the app can load the CUDA model  ·  **ACCEPTED**

`train_torch.py` writes `models/gesture_model_torch.pt`, and the LIVE
CLASSIFIER loads only the scikit-learn payload `models/gesture_model.joblib`.
A `.pt` checkpoint therefore cannot be used live at all.

**Why it is accepted.** `model_feature_vector()` is the single contract binding
`train_model.py` to `LivePredictor` (4.7); a conv net learns its own features
and must not be mixed into that path. The trainer **says so at the end of
every run** instead of leaving it to be discovered.

**Verdict: ACCEPTED,** and surprising enough that it is worth stating twice —
you can train the best model in the project and the app will keep predicting
with the old one.

---

# 5. Still open

Ranked by what they will cost if left alone.

| # | Problem | Why it matters | Where the fix is specified |
|---|---|---|---|
| 1 | **No contrast check at record time** (4.2) | the next dataset can fail exactly like the Oct-6 one, silently | `recommendation.md` §4.1–4.2 |
| 2 | **The 6-gesture raw dataset is gone** (4.3) | no multi-gesture raw data exists; no backup scheme | `recommendation.md` §4.3 — re-collect |
| 3 | **Six-gesture accuracy is 43 %** (4.1) | the actual deliverable does not work yet | fix the data first |
| 4 | **Two-class scores prove nothing** (4.4) | invites wrong conclusions about the model family and the CUDA path | quote the one-threshold baseline every time |
| 5 | **No dynamics in the features** (4.9) | wave direction is unlearnable as built | `recommendation.md` §6.3 |
| 6 | **`EMG_MODE = "raw"` unvalidated** (4.9) | the obvious fix for 50 Hz is itself unverified | verify rate and amplitudes on hardware |
| 7 | **Serial-port single-writer is a convention, not a mechanism** (1.8) | the garbage-framing bug can come back with the next feature | a lock or a wrapper in `myo_raw.py` |
| 8 | **Stall detection has a 6 s blind spot** (1.5) | lost samples at every dropout | a non-blocking reader would remove 1.1 and 1.5 together |
| 9 | **The `dropped` counter over-counts** (1.11) | trains the operator to ignore warnings | use the armband's sequence numbers |
| 10 | **Auto-trim thresholds need manual tuning per subject** (2.7) | four troubleshooting rows instead of one calibration step | per-session calibration |

---

# 6. What the failures taught, in one place

1. **A constant delay is invisible to every rate measurement.** Measure
   *latency* separately from throughput, or you will debug the plot for a day
   (1.1).
2. **Measure the symptom, not the suspicion.** 4.6 Hz, 9258 Hz, 26-fold
   envelope variation, 98.3 % from one threshold — every real diagnosis in
   this project came from a number, and every wrong one from a plausible
   story.
3. **One shared resource, one owner.** Every serial problem was two callers
   touching one port (1.8), and the fix is always a queue, never a retry.
4. **Three symptoms can be three causes.** The dropouts in 1.3 needed three
   independent fixes; each one alone looked like it "sort of helped", which is
   the signature of a multi-cause bug.
5. **Catch bad data while it is still being recorded.** Everything in §4 that
   cost a dataset was knowable *during* the session and only discovered after
   (4.2).
6. **Always compute the stupid baseline.** A majority-class number and a
   one-threshold number next to every accuracy (4.4).
7. **Write down why the ugly code is ugly.** `myo.run()` with no argument, the
   `QSvgGenerator` detour, the span-average rate — all three are one
   "cleanup" away from re-breaking, and comments in the code plus a rule in
   `CLAUDE.md` are the only thing holding them.

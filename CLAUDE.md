# CLAUDE.md — Myo EMG gesture dataset project

Orientation file for future sessions. Read this before changing anything.

---

## 1. Purpose

Collect high-quality, labeled 8-channel EMG gesture data from the Thalmic Myo
armband — from the project owner and from other subjects — and keep it in a
form that can later train Random Forest / SVM / XGBoost / CNN / LSTM /
Transformer models, with **honest subject-independent validation**.

The deliverable is a recording tool, not a model. No classifier is trained yet.

## 2. Hardware

| Item | Value |
|---|---|
| Device | Thalmic Myo armband, 8 dry EMG pods. **The channel count is no longer hardcoded** -- it comes from `DEVICE_PROFILES[EMG_DEVICE]`, see §2b. |
| BLE name | `Myo_TUKE` |
| Firmware | `1.5.1970.2` (verified on hardware) |
| Link | Bluegiga BLED112 BLE dongle, USB VID:PID `2458:0001` |
| Port | `COM3` on this machine (auto-detected, see below) |
| Filtered EMG stream | 8 × uint16, on-device filtered/rectified, BLE attr `0x27` |
| **Measured rate (filtered)** | **≈ 50 Hz** (49.9 Hz over a 6 s window) |
| **Measured amplitude (filtered)** | baseline ~15-40, max-effort fist **1100-1444** (so NOT clipped at 1023) |
| Raw EMG stream | 8 × int8, 200 Hz, BLE attrs `0x2b/0x2e/0x31/0x34` |

`myo64.dll` is a leftover from the official Myo SDK. **It is not used** —
everything goes over the serial protocol in `myo_raw.py`.

## 2b. A second band — what is ready, and what is not

Added 2026-10-19, before the band is known. `N_CHANNELS = 8` used to be a
bare literal in the middle of the module, used at 44 sites, which made a
second armband a search-and-replace across all of them. It is now **derived**:

```python
DEVICE_PROFILES = {"myo": {"channels": 8, ...}, "generic_16": {...}}
EMG_DEVICE = "myo"
N_CHANNELS = int(device_profile()["channels"])      # derived
set_device("generic_16")                            # recomputes the derived globals
```

**The rule that makes this safe: anything holding real data follows the DATA,
not the config.** A recording made on an 8-channel band must keep loading
after `EMG_DEVICE` moves to a wider one — otherwise changing one config line
would silently make the whole existing dataset unreadable. So:

| Follows the data | Follows the config |
|---|---|
| `Recording.n_channels` / `.channel_names` (read off the array) | `SampleRing`, the live plot, the CHANNELS panel |
| the metadata sidecar's `n_channels` | the feature-name generators' *default* |
| `csv_header(n_channels)` and the CSV reader (channel columns taken from the **header**, sorted numerically so `ch10` does not land between `ch1` and `ch2`) | |
| `calibration_levels` / `resolve_calibration` / `apply_calibration` (sized from the record) | |
| `make_windows`, `model_feature_names(..., n_channels)` | |

**Verified at 4, 8, 12 and 16 channels** while the config still said 8: npz
and CSV both round-trip bit-identically at every width, the metadata reports
the real count, `make_windows` returns the right third dimension, and the
feature vector and its names stay the same length (50/154/104 at 8 channels,
98/306/208 at 16).

**`apply_calibration` refuses a width mismatch rather than broadcasting.** An
8-wide map onto a 16-wide window would either raise or — worse — succeed on a
multiple. It returns the block untouched, which is the same "behave as the
uncalibrated pipeline" answer an unusable record gets: degrade to no
calibration, never to a wrong one.

**What is NOT done, and must not be assumed:**

- **acquisition is still Myo-only.** `myo_raw.py` is the working
  communication layer and §3's rules still apply to it in full. A profile
  makes the DATA side band-agnostic; it does not make the app talk to
  anything else. A second band needs its own backend behind the same
  `MyoAcquisition`-shaped interface.
- **no cross-band model.** A model trained on 8 channels has 8 channels'
  worth of features; `LivePredictor` already refuses a feature-count
  mismatch at load time, so the failure is a refusal rather than nonsense,
  but there is no channel mapping between bands and inventing one would be
  guesswork.
- **`CHANNEL_COLORS` and the plot are built once** at construction, so a
  `set_device()` call mid-session does not rebuild the GUI. Set the profile
  before starting.
- the `generic_16` profile is a **placeholder** showing the shape of an
  entry. It is not a real band and nothing has been validated against one.

## 3. Myo connection architecture — DO NOT REPLACE

The connection works. It must stay working. The layering is:

```
BLED112 dongle on COM3
        |  pyserial
   myo_raw.BT                 BLE/BGAPI packet framing
        |
   myo_raw.MyoRaw             Myo protocol, set_mode, notifications
        |  add_emg_handler(fn(emg, moving))
   emg_dataset_tool.MyoAcquisition      background thread, ring buffer, capture
        |  Qt signals (state only) + ring buffer reads
   emg_dataset_tool.MainWindow         GUI
```

Rules — the first two were learned the hard way on real hardware:

- **Call `myo.run()` with NO timeout argument.** `BT.recv_packet(timeout)`
  re-assigns `self.ser.timeout` *before every single byte*, and on Windows
  every assignment reconfigures the COM port. `run(1)` measured **4.6 Hz**
  instead of 50 Hz and then died with *"A device attached to the system is not
  functioning"*. `test_myo.py` calls `m.run()` for this reason. Stall
  detection lives in `MyoAcquisition._watchdog` instead, which closes the port
  to break the blocking read.
- **Only the acquisition thread may touch the serial port.** `BT` shares one
  port and one packet buffer across callers and takes no lock while reading.
  A `vibrate()` write from the GUI thread desynchronised the framing and EMG
  notifications started arriving as `data with unknown attr` garbage. Haptic
  cues are now queued (`_vibrate_request`) and written by the acquisition
  thread between packets. Anything else that writes to the armband must do the
  same.
- **Do not swap `myo_raw.py` for another library** (myo-python, pyomyo, …).
- The known-good call sequence is: `MyoRaw(port)` → `add_emg_handler(...)` →
  `connect()` → `run()` in a loop. `MyoAcquisition._connect` / `_stream` do
  exactly this and nothing more.
- `MyoRaw.detect_tty()` finds the dongle by USB PID, so `MYO_PORT = None`
  normally works. Set it explicitly only to override.
- `MyoRaw.connect()` blocks while scanning. `MyoAcquisition.stop()` therefore
  closes the serial handle to break the blocking read — that is deliberate,
  not a leak.
- Reconnect = construct a **fresh** `MyoRaw` and close the old serial port.
  Reusing a dead instance does not work. (`test_myo.py` established this.)

### Changes made to `myo_raw.py` (both additive)

1. `handle_data` now silently ignores attrs `0x2b/0x2e/0x31/0x34` instead of
   printing `data with unknown attr`. Without this, `EMG_MODE = "raw"` floods
   the console 200×/s. No existing behaviour changes.
2. `vibrate()` used Python 2 `xrange` and always raised `NameError`;
   now `range`.
3. (2026-10-22) moved to `emg_app/myo/` with `common.py`; the only edit is
   its import line, `from emg_app.myo.common import *`.

`common.py` is untouched. `legacy/myo.py` (the old k-NN demo) and
`legacy/test_myo.py` changed only their imports (a project-root `sys.path`
line and `emg_app.myo.*`) and still run.

## 4. Important files

**Project layout (reorganised 2026-10-22).** Only the entry point and the
top-level docs live in the root; everything else is in a folder.

```
emg_dataset_tool.py   THE ENTRY POINT and stable import surface (see below)
emg_app/              the application, as a package -- emg_app/README.md
training/             trainers, the job runner, the NN worker, the comb test
tools/                report / screenshot generators
tests/                runnable check scripts
docs/                 every other document
legacy/               retired code kept for the record (not run)
datasets/ models/ exports/   data
```

| File | Role |
|---|---|
| **`emg_dataset_tool.py`** | **The entry point** (`python emg_dataset_tool.py`) **and the one import surface**: it re-exports every name in `emg_app`, so `import emg_dataset_tool as T` works as it did when the whole tool was this one file. USER CONFIGURATION names are forwarded live in both directions -- `T.CALIBRATION_MODE = "off"` sets `emg_app.config.CALIBRATION_MODE`, which is what every module reads. |
| **`emg_app/`** | The application, split by topic (it was one 21 400-line file until 2026-10-22). **Module map and the two rules that keep it correct: `emg_app/README.md`.** Settings are read as `cfg.NAME` at call time, never copied; imports point one way (config → data layer → acquisition/neural/jobs → prediction → gui); the data layer imports no Qt and no torch. `MainWindow` is `gui/main_window/window.py` plus one mixin per feature. |
| **`training/train_model.py`** | **Three-stage** model sweep (screen → shortlist probe → full) + honest validation + saves `models/gesture_model.joblib`. Imports the data layer from `emg_dataset_tool`. Builds its candidates from `models/model_spec.json` when the GUI passes `--spec`; honours a pinned `setup` (§8n). See §8g for the cost controls. |
| **`training/train_nn.py`** | **The neural-net trainer (CUDA).** The separate second opinion, under the *same* protocol as `train_model.py`; saves `models/gesture_nn.pt` + a plain-JSON stats sidecar. Its output is **never** fused into the deployed answer. See §8L. |
| **`training/train_job.py`** | **The background job runner.** Every trainer the GUI starts runs through it, detached, so closing the app does not stop training; also chains ML + NN. Talks to the GUI only through files in `models/jobs/<id>/`. See §8m. |
| **`training/explain_votes.py`** | Rebuilds `models/gesture_panel.joblib` for the model that is *already* saved, without re-running the sweep. `--json` also writes `models/reports/decisions.json`. See §8e. |
| **`training/nn_arch.py`** | The NN architectures, and the only place a layer is defined. Imports **torch and nothing else** — shared by the trainer and the inference worker. §8L. |
| **`training/nn_serve.py`** | **The live inference worker.** The only process that imports torch; the app starts it (`NN_WORKER_SCRIPT`). §8L. |
| **`training/nn_ml_comb_test.py`** | The ML + NN combination search (§8n), with its plain-text progress view. `tools/comb_test_report.py` turns a run into xlsx/csv + `docs/COMB_TEST.md`. |
| **`emg_app/myo/myo_raw.py`** | Myo BLE protocol wrapper. **The working communication layer** (§3). `common.py` beside it has the `pack`/`unpack` helpers. |
| `legacy/` | Retired, kept for the record: `train_torch.py` (**SUPERSEDED by `train_nn.py`**, §8b), `myo.py` (the original k-NN demo), `test_myo.py` (the original matplotlib monitor -- the reference for "the connection definitely works like this"), `myo64.dll` (unused Myo SDK leftover). |
| `docs/GUIDE.md` | **User-facing manual**: what every file and setting does, and how to set the tool up for different jobs. |
| **`docs/MANUAL.md`** | **The complete manual.** Install, architecture, the interface, the ML pipeline, the honesty machinery, and a condensed engineering record of every problem solved. |
| **`docs/UI_GUIDE.md`** | **The UI, screen by screen, with annotated screenshots** (regenerated by `tools/make_ui_guide.py`, which uses the **native** Qt platform plugin — offscreen renders every glyph as a tofu box). |
| **`docs/MODELS_COMPARED.md`** | **ML models vs soft voting vs the NN** -- accuracy (per window, per gesture, per person, per class, nested), the DC-only controls, speed per live window, size, training time, and which to use. §8n. |
| **`docs/COMB_TEST.md`** | The 2026-10-22 ML + NN combination test: every number. §8n. |
| **`docs/EXCLUDING_PEOPLE.md`** | Rotation augmentation on the hard fold, and training with / without a person, with graphs (`tools/make_exclusion_report.py`). §8o. |
| `tools/bench_models.py` | Times the deployed ML model (and each vote member) and the NN worker (CUDA and CPU) per live window, on real recordings. Output kept in `models/reports/bench_models_*.json`. |
| **`docs/how_to_fix_long_training.md`** | **Why the sweep took 3h 16m and what was done about it.** Read it before touching `DEDUPE_WINDOWS` or `A2_SHORTLIST`. §8g. |
| **`docs/calib_plan.md`** | **Per-wearer calibration: the design study.** Tier 1 of it is built — §8k; tiers 2–3 are not. |
| `docs/problems.md`, `docs/recommendation.md` | Earlier problem log and recommendations, kept for the record. |
| `docs/DATASET.md` | On-disk format, filenames, CSV/NPZ schema. |
| **`docs/EDA.md`** | **The ARCHIVED first dataset, measured** (why it could not give an honest score). §8f. The working dataset is `docs/DATASET_SUMMARY.md`. |
| **`docs/DATASET_SUMMARY.md`** | **What is in `datasets/` now**, measured: people x gestures, seconds, sample rates, quality flags, takes, windows scored, saved vs derived calibration. Generated by `tools/dataset_summary.py` (-> `results/dataset_summary.json`). |
| **`docs/DATA_MAP.md`** | **Every saved result, where it is and what answers what** (results/, models/zoo, comb_test, reports, datasets). Start here when writing a report. |
| `docs/REPORT.md` | The generated report: `tools/dataset_summary.py` -> `tools/collect_results.py` (-> `results/experiments.json`) -> `tools/make_report.py`; figures by `tools/make_figures*.py`. Re-run all after new data. |
| `docs/ARCHITECTURE.md` | Threads, buffers, state machine, data flow. |
| `docs/ML_PIPELINE.md` | From recordings to a validated model. |
| `requirements.txt` | Dependencies. |
| `tests/` | Runnable check scripts; see §10 and `tests/README.md`. They import trainers from `training/` themselves. |
| `datasets/` | Recorded data. Also `<person>/calibration.json` + `calibration_log.jsonl` — the per-wearer calibration, §8k. |
| `exports/` | Combined ML bundles from the EXPORT button. |
| `models/` | `gesture_model.joblib` (loaded by the app), `gesture_nn.pt` + `.json`, `model_spec.json`, `nn_spec.json`, `gesture_panel.joblib` (§8e), timestamped archives, `reports/`, `comb_test/`, `jobs/`. |

### Where the old sections went

The single file's numbered sections map onto modules (full map:
`emg_app/README.md`):

```
USER CONFIGURATION   emg_app/config.py
SECTION 1  helpers   emg_app/utils.py
SECTION 2  signal    signal_processing.py, calibration.py (maths), features.py
SECTION 3  quality   quality.py
SECTION 4  dataset   recording.py, markers.py, calibration.py (storage),
                     model_spec.py, compute.py, dataset.py
SECTION 5  acq.      acquisition.py (+ myo/)
SECTION 5b neural    neural.py, jobs.py
SECTION 6  sequence  sequence.py, model_test.py
SECTION 7  GUI       gui/ (qt, tokens, theme, widgets, plot, decision_trail,
                     workers, dialogs/, main_window/), decision_panel.py,
                     prediction.py
SECTION 8  entry     app.py  (+ emg_dataset_tool.py at the root)
```

Everything outside `gui/` is Qt-free and importable headless — that is what
lets a training script do `import emg_dataset_tool as T; T.load_dataset(...)`
on a machine with no display.

**Rules this adds:**

- **Read settings as `cfg.NAME`, never `from emg_app.config import NAME`.**
  A from-import copies the value once; the GUI and the tests change settings
  at runtime, and a copied one would silently keep the old value.
- **Never write a setting through `globals()`** -- in a split module that
  rebinds a private copy. `setattr(cfg, name, value)` (what
  `MainWindow._set_global` does).
- **Imports point one way.** A lower module never imports a higher one at
  top level; a type hint that would is imported under `TYPE_CHECKING`.
- **A new script outside the root** needs the project root on `sys.path`
  before `import emg_dataset_tool` (see the top of `training/train_model.py`).

## 5. Data format

```
datasets/person_01/
    calibration.json             the wearer's CURRENT calibration (8k), if any
    calibration_log.jsonl        append-only; every calibration they ever had
datasets/person_01/session_01/
    session_config.json          config snapshot for the session
    manifest.jsonl               append-only log, one line per saved file set
    markers.jsonl                append-only log of timestamped notes/breaks
    _takes/                      long unlabelled free takes (NOT training data)
        FIST_take001_20261012-164530-123_a1b2c3_take.csv
        FIST_take001_20261012-164530-123_a1b2c3_take.npz
        FIST_take001_20261012-164530-123_a1b2c3_take_meta.json
    FIST/
        FIST_rep001_20261006-114530-123_a1b2c3_raw.csv
        FIST_rep001_20261006-114530-123_a1b2c3_raw.npz
        FIST_rep001_20261006-114530-123_a1b2c3_raw_meta.json
        FIST_rep001_20261006-114530-123_a1b2c3_trimmed.csv      (if trimmed)
        FIST_rep001_20261006-114530-123_a1b2c3_raw_notes.json   (if annotated)
    REST/ OPEN/ ...
    _graphs/                     exported PNG/SVG
    _deleted/                    quarantined recordings (never erased)
```

Underscore folders are excluded from dataset discovery.
Full schema: `docs/DATASET.md`.

## 6. Configuration

Everything lives in the `USER CONFIGURATION` block at the top of
`emg_dataset_tool.py`. Nothing below that block needs editing for normal use.
`GUIDE.md` documents every setting and gives ready-made presets.

The GUI mirrors edits to person/session/handedness/notes/repetitions/order
back into those module globals (`MainWindow._set_global`) so that the metadata
written to disk always matches what actually happened.

## 7. Recording workflow

```
START SESSION
  -> build_plan()  =  GESTURES x REPETITIONS, ordered
  -> for each (gesture, repetition):
        REST       relax
        PREPARE    "GET READY: FIST"
        COUNTDOWN  3 . 2 . 1          <- capture opens PRE_GESTURE_MS before the end
        RECORDING  "RECORD NOW"       <- label window opens
        STOPPING   POST_GESTURE_MS    <- label window closed, buffer still running
                                      <- raw files written, quality checked
        REVIEW     graph freezes on the take, proposed cut drawn over it,
                   waits for ACCEPT & NEXT (writes the trimmed copy) or
                   REDO (quarantines it and repeats the same repetition)
                   -- skipped entirely when REVIEW_EACH_RECORDING = False,
                      in which case SAVED -> REST runs on AUTO_ADVANCE
```

The graph stays live and visible throughout: `COUNTDOWN_OVERLAY = "corner"`
keeps the cue out of the traces, and the review step reuses the same eight
curves to show the finished take without ever pausing acquisition.

The capture window is **wider** than the labeled window, so an early reaction
or an overshooting explosive contraction is still inside the raw file. The
label boundaries are stored as `label_start_index` / `label_end_index` and as
the `in_label_window` CSV column.

### Free takes

The "record now, cut later" path, independent of both the cued sequence and
LABEL ON GRAPH:

```
START FREE TAKE (Ctrl+Shift+R)   one continuous capture, minutes long
STOP & SAVE TAKE                 the whole stretch -> <session>/_takes/
OPEN TAKE & CUT CLIPS            carve labelled clips out of it, any time later
    EXTRACT CLIP (Ctrl+Space)      -> an ordinary stage-raw recording
    NEXT >> (Ctrl+Right)           -> step one clip length along
DELETE TAKE(S)                   the take -> <session>/_deleted/
```

**DELETE TAKE(S)** moves the selected take(s) out of `_takes/` and into
`_deleted/`, through `DatasetWriter.delete_take()` — never `os.remove`
(safety rule 4). A take is minutes of a subject's time and the one thing
clips cannot be re-cut from, so a mis-click costs a folder move. Every file
the take was written in goes together (csv, npz, parquet, the metadata and
notes sidecars), matched as `<stem>.` or `<stem>_` so a take whose name
merely *starts* the same cannot be caught, and a `stage: "deleted"` line with
`deleted_stage: "take"` goes into the append-only manifest.

**Clips already cut from a take are not touched.** They are ordinary
recordings in the gesture folders with their own uids, and deleting the take
does not make them wrong — only unre-cuttable. `clips_from_take()` counts
them from their own `parent_take_uid`, and the confirmation dialog says how
many there are, because that is what someone needs to know before throwing a
take away.

**A take is not training data and must never become it.** Two independent
mechanisms keep it out: it lives in `_takes/`, and `discover_recordings()`
prunes every folder starting with `_`; and it carries stage `take`, which no
stage selector asks for. Loading five minutes of mostly-rest signal as one
labelled gesture would quietly wreck the model, so both are checked in
`tests/test_free_take.py`.

A free take does **not** go through `SequenceController`: it has no plan, no
countdown and no review step. It calls `MyoAcquisition.start_capture()`
directly with a `max_seconds` override (`FREE_TAKE_MAX_MINUTES`), because the
120 s `MAX_RECORDING_SECONDS` valve is sized for a cued take. The acquisition
thread has **one** capture slot, so a free take and the sequence can never run
at the same time -- `_refresh_controls()` enforces that by disabling START
SESSION and RECORD while a take is open.

Clips cut from a take are ordinary stage-`raw` recordings in the gesture
folder, each with its **own** uid (uid is what pairs a raw file with its
trimmed copies; twenty clips sharing the take's uid would look like twenty
versions of one recording). Provenance lives in the metadata:
`parent_take_uid`, `take_start_index`/`take_end_index`, `capture_style:
"cut_from_take"`. Subject metadata on a clip is copied from the **take**, not
from today's config.

`LABEL_DEFAULT_SELECTION_S` is the single default cut length, shared by
SELECT LAST n s on the live graph and the cut dialog, and settable in both.

**TAKES ON DISK is scoped to the current subject AND session, and now says
so.** A take lives in `<person>/<session>/_takes/`, so a take recorded with
`SESSION_ID = "session_03"` is simply not in the list when the app is reopened
on the config file's `session_01` — and because every sitting carries the same
six gesture labels, the list looks populated and correct while the takes you
just made are missing. That is exactly how it was reported, on 2026-10-19,
with six takes sitting safely in `Dano/session_03/_takes/`.

Three things now make the scope visible rather than leaving it to be deduced:
the header reads `TAKES ON DISK -- <person> / <session>`, the empty message
names the pair instead of saying "this subject/session", and
`takes_by_session()` (SECTION 4, Qt-free) reports the subject's *other*
sessions so the panel can name them. It deliberately does **not** switch
session on its own: `SESSION_ID` decides where the next recording is written,
so changing it to make a list look right would be the wrong trade. The hint
points at OPEN FILE(S)... instead, which reaches any take on disk without
moving where new data goes.

### Markers and notes

Independent of the cued sequence, and usable at any time while streaming:

```
ADD NOTE (Ctrl+M)      timestamped text -> markers.jsonl, drawn on the graph
BREAK    (Ctrl+B)      a dashed separator; same thing, no text required
CUT SINCE LAST MARKER  the span marker -> now, saved as a labelled recording
    (Ctrl+Shift+Space) through the SAME path as ADD SELECTION
OPEN FILE...           pull up any recording on disk: graph, notes, trim
```

The marker's timestamp is the newest **sample's** time, not the GUI clock, so
it lines up with the signal. `markers.jsonl` is append-only like the manifest:
a marker is never edited or deleted, and a correction is simply another
marker. Any marker inside a saved span is copied into that recording's
metadata under `markers`.

Notes on a take that is already saved go to a `*_notes.json` sidecar beside
it, never into the data file — `*_raw.*` stays byte-identical, which
`tests/test_markers.py` checks on both the data layer and through the dialog.
`MARKER_CUT_ENABLED` (and the checkbox mirroring it) decides whether cutting
at markers is offered at all.

## 7a. The label set, and recording the same gesture back to back

**`GESTURES` is the set in use; `OPTIONAL_GESTURES` is what else the tool
knows.** Added 2026-10-19 with `PINCH`, `THUMB_OVER_INDEX` and `POINT`, all
**off by default**. `GESTURES...` (`Ctrl+Shift+G`) edits the set at runtime
and writes back to the global, so the folder layout, the plan, every label
combo and each recording's config snapshot follow it.

Off by default on purpose: every extra class costs recording time from every
future subject **and makes the problem harder**. Six classes on an 8-pod
forearm band already collide — §8L measured `OPEN` against `WAVE_RIGHT` at
**0.978 cosine**, i.e. nearly the same signal — so a seventh is only worth it
when it is driven by a different muscle group rather than by a
different-looking hand. `GESTURE_CUES` / `OPTIONAL_GESTURES` carry the
instruction to show, so a cue can say what to do instead of naming a
constant.

**Switching a gesture off never touches data already on disk.** The folder
and its recordings stay and the trainer still finds them, because discovery
walks the tree rather than this list. A label set is a plan for what to
record next, not a filter on what has been recorded.

### Adding a NEW gesture from the GUI — and why it never retrains (2026-10-22)

GESTURES... → ADD YOUR OWN takes a name and an **instruction** (the cue).
`emg_app/gestures.py` (Qt-free) makes it last and reach everything:

| Where | What happens |
|---|---|
| `<DATA_FOLDER>/gestures.json` | the in-use list + every added gesture's cue; written by `edit_gestures`, restored by `app.main()` (**not** by `MainWindow`, so a test's own `GESTURES` is never overridden). It lives with the DATA because it describes what is recorded. An unreadable file is ignored, never allowed to empty the set |
| `cfg.CUSTOM_GESTURES` | name -> cue; `known_gestures()` and `gesture_cue()` include it |
| every dropdown, the plan, the calibration sweep, the session banner, TEST MODEL's cue, the recordings counts | follow `cfg.GESTURES` / `cfg.gesture_cue()` as before -- now with the instruction shown |
| colour | `cfg.label_color(name)`: the six built-ins keep `LABEL_COLORS`; any other gesture gets one of `EXTRA_LABEL_COLORS` in `known_gestures()` order, so it is stable and never one already in use. Used by the plot, trim dialog, counts, FEATURE SPACE and the caption of every probability bar (ML and NN) |
| the models | **unchanged.** A model knows exactly the labels it was trained on. The dialog says per gesture which models know it; the LIVE CLASSIFIER panel adds `NOT IN THE MODEL YET: X (n rec) -- RETRAIN when you choose` once recordings exist (`model_gesture_gaps()`, from the cached fingerprint -- no extra disk walk) |

**Nothing retrains automatically, by design** -- the operator asked for it
that way, and it is also right: a gesture recorded by **one person** makes
`subject_grouping_viable()` fail, the trainer falls back to
recording-grouped folds, and its headline number becomes the optimistic
within-person one (~97 %), not comparable with the subject-held-out 77.6 %.
The dialog warns per gesture ("record it on a second person before trusting
a retrain's score"). Record a new gesture on at least two people, calibrate
them (the sweep now includes it), then RETRAIN.

`tests/test_custom_gesture.py` checks the whole story on a temp COPY of the
dataset and models: add -> saved -> restored -> dropdowns / plan / sweep /
banner / colour -> nothing retrained -> recorded on three people -> panel
warns -> explicit retrain -> both models know it: 7 probability bars in its
colour, 7x7 confusion matrices in MODEL STATS and NN STATS, its points in the
FEATURE SPACE (PCA) in its colour, TEST MODEL cues it, the NN worker loads it.

### Switching a gesture off in the classifiers — without retraining (2026-10-26)

The **gear icon** in the header (`Ctrl+,`) opens SETTINGS → Gestures. For each
gesture there are two independent switches:

| Switch | Means | Retrain? |
|---|---|---|
| **RECORD** (the name's checkbox) | is it cued, offered in every label combo, calibrated | never (§7a above) |
| **USE IN CLASSIFIER** | may the live **ML model and neural net** answer with it | **no — in either direction** |

`emg_app/class_mask.py` (Qt-free, torch-free) is the whole mechanism: a
gesture that is off has its probability **zeroed and the rest renormalised**
on the model's *output*. The model, its file and the worker process are never
touched, so switching back on is instant and returns the original answers bit
for bit (`tests/test_class_mask.py` compares them and the model file's bytes).
`cfg.CLASSIFIER_OFF` holds the set; it is saved in `<DATA_FOLDER>/gestures.json`
as `classifier_off` (older files without the key mean "nothing off") and
restored by `app.main()` with the rest of the gesture set.

**Where it is applied, and why each one:**

| Where | What |
|---|---|
| `LivePredictor.predict` | after `predict_proba`, **before** smoothing |
| `NeuralEngine._on_out` | the same, in the main process — the worker still answers for every class, so no restart |
| `DecisionPanel.step_for_features` | each member masked, fusion = mask of the plain mean, so the DECISION TRAIL cannot name a gesture the LIVE CLASSIFIER was told to skip. The *recorded* (out-of-fold) trail is unmasked history |
| both probability-bar panels | an off gesture is dimmed, `(off)`, value `off` |
| TEST MODEL | cues and scores only `predictor.active_labels` |

Three decisions worth knowing before touching it:

1. **The mask is part of the smoothing key.** Smoothing averages the last few
   windows' probabilities; a window scored with FIST on must not be averaged
   with one scored with FIST off. Both predictors compare `off_key(labels)` and
   clear their history when it moves.
2. **"No answer" is a state, not a guess.** A forest readily puts *all* its
   probability on one class, so with that class off there is nothing to
   renormalise. The result is `label = "?"`, `answered = False`, uniform
   probabilities and a NaN confidence — never an arbitrary argmax. One such
   window does not blank an already-established vote.
3. **Renormalising inflates confidence, so `kept` says by how much.**
   `Prediction.kept` / `NeuralPrediction.kept` is the share of the model's *own*
   probability that sat on gestures still on. A window the model was 99.9 % sure
   was an off gesture can come out as "100 % OPEN"; below `LOW_KEPT` (0.5) the
   panels add "only N % of the model's own probability was on gestures that are
   on". Read the answer as "the best of what is left", not as a belief.

What a mask is **not**: it is not the answer a model *trained without* that
gesture would give, and the app says "masked out of the answer, nothing
retrained" wherever it shows one. MODEL STATS / NN STATS still describe the
model as trained, on all its classes.

A gesture **no model has been trained on** has no output column to switch: its
switch reads `n/a`, and the editor's retrain note says what to do. Adding one
still means ADD → record (≥ 2 people) → calibrate → RETRAIN, exactly as above;
the settings dialog's **APPLY + RETRAIN** button (shown only when a recorded
gesture is missing from a trained model) chains into the usual RETRAIN ML +
NEURAL NET confirmation. A request that would leave a model with fewer than two
gestures is refused with the model named (`validate_off`), and switching REST
off warns once, because REST is the only "no gesture" answer.

While a session runs, what is *recorded* is locked (the plan is already built)
but the classifier switches still work — they cannot disturb a recording.

`GestureEditor` (`gui/dialogs/gesture_setup.py`) is the one editor, hosted by
both SETTINGS and the older GESTURES… dialog (`GestureSetupDialog`, a thin
shell that keeps the API the tests use). `MainWindow._apply_gesture_setup` is
the one place an edit is applied, so the two entry points cannot disagree.

### "Five of the same gesture in a row" — `GESTURE_ORDER = "blocked"`

This already existed and did exactly that; what was missing was that it was
undiscoverable (the dropdown's tooltip described only "interleaved") and the
hold length was not adjustable in the GUI. Both fixed.

```
blocked      REST REST REST REST REST  FIST FIST FIST FIST FIST  OPEN ...
interleaved  one round of every gesture, shuffled within the round
random       fully shuffled
```

**`RECORD_TIME` is 2.5 s** as of 2026-10-19, down from 4.0, and settable per
session in the PLAN panel. The reason is specific to blocked order: five 4 s
holds back to back is enough sustained contraction to drive measurable
fatigue drift (§8f measured FIST losing 59 % of its amplitude across a 52 s
hold). 2.5 s still yields ~7 windows per repetition at the model's
0.3 s / 0.5 setting and is close to the 3 s median of the clips already on
disk.

**Blocked is the comfortable choice; interleaved is the honest one.** In
blocked order all of a gesture's data comes from one stretch of time, which
is the §8f confound being created deliberately — one take per gesture per
sitting. Use blocked to collect comfortably, then collect *more sittings*
rather than trusting the within-session score.

## 7b. Model specification

Which classifiers the sweep trains is data, not code. `MODEL SPEC...`
(`Ctrl+Shift+M`) edits it and writes `models/model_spec.json`; the GUI passes
`--spec` on every retrain, so what the subprocess trains is always what the
GUI was showing.

```
emg_dataset_tool.py     MODEL_KINDS (the schema), MODEL_SEARCH_SPACES,
                        validate_model_spec, expand_auto_spec,
                        describe_estimator        -- no sklearn
train_model.py          _make_estimator / _make_pipeline / resolve_spec
                        build_models_from_spec    -- sklearn lives here
```

The split matters: the app must run without scikit-learn, so the schema and
the validation are Qt-free *and* sklearn-free, and only the trainer turns a
spec entry into an estimator. There is exactly one definition of what a spec
means.

**`validate_model_spec()` is the gate.** It never raises and never returns
something the trainer cannot build: an unknown kind is dropped, a bad value is
coerced, an out-of-range value is clamped, a duplicate name is renamed, and a
spec with everything disabled gets one model switched back on. Every
correction is reported as a string the GUI shows and the trainer logs. A
hand-edited `model_spec.json` therefore cannot crash a sweep an hour in.

**Voting.** The ensemble is added as one more *candidate*, not as a
replacement, so it is scored against its own members by the same
cross-validation. `voting.always` forces it to be saved regardless, which is a
choice rather than an improvement.

**AUTO mode expands, it does not search.** `expand_auto_spec()` turns each
enabled model into up to `max_per_model` concrete candidates, which then go
through the *existing* evaluation path. That is deliberate: there is no second
scoring route to keep honest, every combination appears as its own row, and
the sampling is seeded so a sweep is reproducible. The configured values are
always combination #1.

The cost is selection bias, and it is stated in the dialog and in the training
log rather than hidden: choosing the best of N candidates on the same folds
makes the winner's score optimistic. Leave-one-subject-out is the number that
did not pick its own folds.

**`describe_estimator()` reads the object, not the spec.** MODEL STATS shows
the fitted pipeline's real stages and, for an ensemble, how the member outputs
are combined. If a payload's spec ever disagreed with its estimator, this
shows the estimator — which is what actually predicts.

## 7c. The model zoo, soft voting, adaptation, augmentation and the two questions (2026-10-27)

Everything here is **off or equal to the old behaviour by default**
(`emg_app/config.py`, "MODEL SELECTION, TRAINING OPTIONS AND PROMPTS"), saved in
`<DATA_FOLDER>/app_settings.json` (`emg_app/settings_store.py`) and restored by
`app.main()` -- not by `MainWindow`, so a test's own values are never overridden.
The measurements behind every default are in `docs/REPORT.md` and
`results/experiments.json` (built by `tools/collect_results.py` from the raw runs
-- nothing in it is typed by hand).

**The zoo** (`emg_app/zoo.py`, `training/train_zoo.py`, `training/zoo_roster.py`).
`models/zoo/<id>/` holds `model.joblib` (an ML payload exactly like
`gesture_model.joblib`) or `model.pt` + `model.json` (a checkpoint exactly like
`gesture_nn.pt`) and a `meta.json`: what it is, its leave-one-person-out scores,
its size, what it was trained with, what two models must share to be voted. The
roster is the 16 classifiers of the combination test (all at the winning
bandpass / `level` / 1.0 s setup) and every architecture in `nn_arch.KINDS`
(0.7 s). `train_zoo.py eval|fit|ensembles|all|status`; a model is **never saved
without scores measured on the dataset now on disk** (`digest`), and the
deployed `gesture_model.joblib` / `gesture_nn.pt` are never touched.

**Stars** are computed, never stored: per family the top 3 by window accuracy
(ties by macro-F1) -- never a training-set number -- plus the best soft-voting
subset only if its NESTED estimate beats the best single model measured on the
same out-of-fold probabilities. They are recomputed whenever the zoo is listed,
which includes after every training because training rewrites `meta.json`.

**Selection and voting.** `cfg.MODEL_SELECTION_ENABLED` False (default) = the
configured `MODEL_PATH` / `NN_MODEL_PATH`, bit for bit as before. True = the
zoo ids in `SELECTED_ML` / `SELECTED_NN`: one id is that model, several are a
**soft vote** -- `LivePredictor` wraps the fitted members in `MeanProba`; the NN
worker (`nn_serve.py --model A --model B`) holds all the nets in one process
and answers with the mean of their softmax plus `members`. A vote is **refused,
with the reason**, if the members disagree on labels, window, features,
calibration mode or adaptation (`zoo.check_vote`, `_vote_problem`, the worker
itself) -- never averaged anyway. **ML and nets are never voted together**
(rule 13 stands).

**A selected vote is described as itself (2026-10-28).** A vote has no `stats`
block, and the first member's numbers are that model's, not the vote's -- the
statistics screens used to show the deployed model's (or `?`). Now:
`zoo.vote_stats()` scores the vote from its members' stored out-of-fold
probabilities (`results/zoo_eval/`, same folds, checked against the trainer's
own `scores_from_probs` in `tests/test_vote_view.py`) and returns None rather
than a number when it cannot: fewer than two models, a model trained *without*
someone, a missing file, or probabilities made on another dataset. The
combination is the user's choice, not a search, so it carries no selection
bias. `LivePredictor` puts that, the members' own scores (`members`) and the
newest member's date into `stats`; MODEL STATS shows a members table; NN STATS
reads the SELECTED nets (`engine.active_model_path` / `model_paths()`), never
`engine.model_path`, which is only the deployed one. Both panels show *how each
member voted on this window* under the bars (`emg_app/vote_view.py`: members'
probabilities go through the same class mask; display only, the answer is still
the mean). The DECISION TRAIL is still built from `gesture_panel.joblib` of the
DEPLOYED model and does not follow the selection -- MODEL STATS says so in a
banner when they differ. Open statistics windows are closed when the selection
changes. **The zoo's NN numbers are 2 seeds** and move a lot (NN-gru 72.4 %
on one seed, 63.0 % on the other, 67.7 % mean): the 72.4 % quoted elsewhere is
a single seed, and nets within ~2 points of each other are not ranked.

**Adaptation to the person** (`emg_app/adaptation.py`): standardise each
person's input with statistics from ~10 % of their own UNLABELLED windows (about
9 s -- the length of a calibration sweep). A model trained with it records
`adaptation` in its payload and carries global statistics as the fallback; the
person's statistics come from `adaptation_signal.npz`, saved beside
`calibration.json` whenever a calibration is saved, so "use the last calibration"
keeps adapting after a restart. `ADAPT_LIVE` off = every model uses its stored
global statistics. It is a different thing from the per-wearer CALIBRATION
(8k), which is unchanged, and a model trained WITHOUT adaptation ignores the
statistics entirely.

**Augmentation** (`training/nn_augment.py`: gain, noise, timewarp, rotate,
chandrop) acts on raw calibrated windows at training time only. `train_nn.py
--augment/--adapt` and `train_zoo.py --augment-nn/-ml --adapt-nn/-ml` take
them; with them off the code path is the original one (`tests/test_zoo.py`
compares weights, fold accuracies and confusion matrices with the previous
commit's trainer). The switches are **per family** (`TRAIN_AUGMENT_NN/ML`,
`TRAIN_ADAPT_NN/ML`) because the two were measured separately and disagree.

**The questions** (`emg_app/prompts.py` decides, `gui/main_window/prompts.py`
asks). *Calibrate the hand first?* before PREDICT / START NN, once per run per
person; "use the last calibration" is offered only if it was made before this
run started, is under 8 h old, and it always says the band must have stayed on
the arm (a live rest check is reported, never used to refuse). *Retrain all or
only the selected?* at startup (`app.main`, 1.5 s after the window is up) when a
model is stale or a recorded gesture is unknown to it; never while a training
or a session runs; "do not ask when the application opens" is a setting.
"Selected" with selection off = the deployed classifier and net.
RETRAIN ML + NN can chain a third step, `zoo`, that re-measures and re-fits the
chosen zoo models (`retrain_zoo`).

## 8. ML workflow

```python
import emg_dataset_tool as T

recs = T.load_dataset("datasets", stage="prefer_trimmed")
X, y, groups, meta = T.make_windows(recs)        # (n, win, 8), labels, person ids
feats, names = T.make_feature_matrix(X, meta["fs"])

for train, test, held_out in T.leave_one_person_out(groups):
    ...   # train on the rest, test on `held_out`
```

`groups` is the whole point. See `docs/ML_PIPELINE.md`.

In practice just run the trainer, which does all of this and sweeps models:

```powershell
python training/train_model.py            # saves models/gesture_model.joblib
```

Then the app's LIVE CLASSIFIER panel works: one-shot or continuous mode, the
prediction drawn over the graph, per-class probability bars, and a MODEL STATS
dialog showing the cross-validated numbers with their protocol.

**Current state of the models (2026-10-22, FIVE subjects) -- see §8n.**
ML: soft vote of `ET-500` + `RF-500-leaf3` on `bandpass` / `level`, 1.0 s
windows at 0.75 overlap, 211 recordings from Barbora, Betka, Bruno, Dano and
Miso: **77.6 % per window (74.0 % nested, the honest figure), 83.7 % per
recording**, subject held out. NN: `gru` at 0.7 s, **72.4 % / 78.9 %**.
Barbora held out is still the hard fold for both (46 % / 28 %).

*(Superseded, kept for the trend:)* **State on 2026-10-19, THREE subjects.** The payload on
disk is `SVM-rbf-C10` on `raw` / `both`, 0.5 s windows at 0.5 overlap with
frequency features, six labels, trained on **128 recordings from three
wearers** -- `Dano` 44, `Betka` 41, `Barbora` 43 (1313 deduped windows, 154
features) -- on the **`affine` per-wearer calibrated scale** (§8k).

It still **requires a calibration**: every wearer must calibrate once before
the classifier will run for them, and `LivePredictor` refuses rather than
predicting on the wrong scale.

| Protocol | Window | Recording vote |
|---|---|---|
| **subject-grouped** (the chosen one) | **64.8 % ± 14.8** | 74.0 % |
| recording-grouped (contrast) | 94.8 % | 98.4 % |
| DC-only control, subject-grouped | 61.7 % | 69.3 % |
| majority class | 19.3 % | -- |

Per wearer, held out: `Dano` 76.4 %, `Betka` 74.1 %, **`Barbora` 43.9 %**.
Macro F1 60.2 %. The neural net (§8L), scored the same way, reads 63.8 % ±
16.3 per window with the same Barbora asymmetry (38.4 / 72.6 / 78.6).

### Read the ± before the mean: Barbora is the hard fold

Both models agree to within a few points: holding out Dano or Betka gives
**73-79 %**, holding out Barbora gives **38-44 %**. That is a consistent
35-point asymmetry, not noise, and it is the whole of the ± 15.

**Measured, so it is not a guess.** Mean per-gesture profile similarity
between people: `Betka`/`Dano` 0.933, `Barbora`/`Betka` 0.912,
`Barbora`/`Dano` **0.884** -- worst on WAVE_LEFT (0.81-0.83) and THUMB_UP
(0.84). It is **not** amplitude: her active/rest contrast is 6.6x against
9.7x and 8.8x, and her absolute levels are in line (239 active against 210
and 181). What differs is the per-channel *pattern*, which is what a
different band rotation looks like -- the one thing §8h lists as unhandled
and which calibration explicitly does not fix, because a permutation of the
channels is not a rescaling of them. **(Superseded 2026-10-27, §8o: the
rotation explanation was tested and is NOT supported -- rotation augmentation
and a one-pod shift do not explain her.)**

With three wearers, two cannot span the third. This is what n = 3 looks like
honestly.

> **The previous entry here (2026-10-16, "two subjects", 86.8 %) was
> arithmetically fine and factually wrong.** `Barbora/session_01` had been
> recorded with `PERSON_ID = "Dano"` still in the SUBJECT box, so 43 of her
> recordings sat inside Dano's fold: the "held-out" person was partly in
> training, and the calibration derived for Dano pooled two people's signal.
> Corrected 2026-10-19 -- every clip and take re-saved under `Barbora` with
> uid, timestamp and samples unchanged (EMG verified bit-identical), the
> originals archived to `datasets/_corrected_Dano_session_03_to_Barbora/`,
> and each person re-calibrated from their own signal alone. Dano's
> calibration came back to exactly its known-good values. **The lesson is
> that `person_id` is not a caption -- it is the key the folds split on and
> the key a calibration is stored under, so getting it wrong corrupts the
> protocol rather than the labelling.**

> **The DC-only control is now the thing to worry about.** On this dataset
> the winner's 146 non-DC features add only **+3.1 points** over the 8
> channel means, and the neural net's control came out at **-1.0** -- the
> identical net with all temporal structure removed scored 64.9 % against
> 63.8 %. Both models are substantially reading per-channel amplitude
> levels, which transfer between wearers *because* calibration makes them
> transfer. That is §8k's warning landing exactly as written: a rising
> DC-only score is not good news. Gesture shape is not carrying this result.

Retrain after each new sitting and after each new wearer.

*(Superseded, kept for the trend: on 2026-10-06 the dataset was 72 recordings
and the best score ≈ 43 % per window against a 16.7 % chance level, with 15
of 72 recordings flagged `no_activity`. The free-take route fixed the
weak-contraction problem — `no_activity` is now 8 flags in 308 — and created
a different one, below.)*

## 8b. The CUDA path (`train_torch.py`) — RETIRED 2026-10-14

> **SUPERSEDED 2026-10-19 by `train_nn.py` (§8L).** A neural net is back in
> the project, but not this one: §8L's trainer shares `train_model.py`'s
> protocol (dedup, subject-grouped folds, calibration, the DC-only control),
> runs its inference in a subprocess so the recorder still never imports
> torch, and is kept strictly out of the deployed answer. `train_torch.py`
> itself is still retired, is still not run, and its numbers are **not**
> comparable with anything current — it had no window-level dedup and chose
> its winner on recording-grouped folds. Read §8L for what replaced it; the
> rest of this section is the record of why it was retired, which §8L's
> headline result has now confirmed a second time.

> **Not part of the workflow any more.** The project trains only the
> traditional classifiers in `models/model_spec.json`. `train_torch.py`, its
> checkpoints and its sweep reports are kept as a recorded experiment, and
> nothing in the application reads them. Do not add it back without a reason
> that the numbers support — see "Why it was retired" below.
>
> **Why it was retired.** It was never better. On the FIST/REST dataset the
> CNN-LSTM scored 99.2 % per recording against **100 %** for a plain
> LDA-shrinkage with no hyperparameters, and **98.3 %** for a single
> amplitude threshold, on a 52.9 % majority-class baseline. A deep model that
> loses to an LDA, on a task one threshold almost solves, is not evidence for
> a deep model. It also made the application's deployment story worse: torch
> is a ~200 MB dependency, and the classifier is destined for an SBC on a
> robot where CUDA may not exist at all.
>
> **What stays true regardless:** the recorder never imported torch, and
> `LivePredictor` has always loaded `models/gesture_model.joblib` and run on
> the CPU. Retiring the experiment changed nothing about what the app loads.

The rest of this section is kept for the record.

It fed the **windowed time series** `(win, 8)` to a neural net instead of the
hand-crafted feature vector:

```powershell
python legacy/train_torch.py                     # CUDA, full grid
python legacy/train_torch.py --quick             # one model, one window
python legacy/train_torch.py --device cpu        # no GPU
python legacy/train_torch.py --epochs 80 --kind cnn_lstm --window 0.5
```

Three architectures (`cnn`, `lstm`, `cnn_lstm`) x three window lengths
(0.30 / 0.40 / 0.50 s). It writes `models/gesture_model_torch.pt` and a full
sweep report under `models/reports/`.

**What is deliberately identical to `train_model.py`**, and must stay so:
`MODEL_FS = 60 Hz` resampling, the `label_start_index:label_end_index` span,
`StratifiedGroupKFold` grouped by **recording uid**, leave-one-subject-out,
and the per-window/per-recording pair of numbers.

**What is deliberately different:**

- **No shared feature function.** `model_feature_vector()` is the one contract
  binding `train_model.py` to `LivePredictor`; a conv net learns its own
  features and must not be mixed into that path.
- **Its own output file.** `models/gesture_model.joblib` stays a scikit-learn
  payload because that is what the app's LIVE CLASSIFIER loads. Nothing in the
  recorder reads the `.pt` checkpoint — the trainer says so at the end of
  every run.
- **Fold-local scaling.** `log1p` + per-channel standardisation fitted on the
  **training fold only**. Fitting it over the whole dataset would leak the
  test distribution into training.
- **No early stopping.** Stopping when the held-out fold looks best is fitting
  the test set. Epochs are fixed and declared.

**The device contract: fail loudly, never fall back.** `--device cuda` is the
default, and if CUDA is not usable `resolve_device()` raises `SystemExit` with
the install command rather than quietly running on the CPU — a silent
fallback looks identical to a working run except that it takes twenty times
as long. Every run prints a boxed `COMPUTE: >>> GPU <<<` / `>>> CPU <<<`
banner at the start **and** at the end.

**Seeding is taken in `build_model()`, not in `train_one()`.** Weight
initialisation happens at construction, so seeding afterwards leaves the
starting point to wherever the global RNG had got to, and two "same-seed"
fits differ. `tests/test_torch.py` checks this by comparing probabilities,
not the argmax — on a separable problem the argmax agrees even between two
unrelated fits.

**Verified on this machine (2026-10-14):** NVIDIA GeForce GTX 1650, torch
2.9.1+cu128, CUDA 12.8, capability 7.5, 4095 MB. `tests/test_torch.py` =
163/163 on the real GPU.

**A high two-class score here is not evidence.** On the current FIST/REST
dataset the CNN-LSTM scores 99.7 % per window — but a single amplitude
threshold scores 98.3 %, and an LDA on the sklearn feature vector scores
100 %. FIST vs REST asks "is the muscle contracting at all", which is the
easiest question on EMG. Quote the majority-class baseline (52.9 %) next to
any number from this dataset, and re-measure once a third **active** gesture
exists.

## 8c. Model provenance — "is a model trained, on what, and is it current?"

Both trainers write a `data` block into the payload:

```json
"data": {"root": "...", "n_recordings": 119, "labels": {"FIST": 56, "REST": 63},
         "subjects": ["Dano"], "sessions": ["Dano/session_01"],
         "newest_stamp": "20261013-225241-161", "uid_digest": "2868d8e7e91543b6"}
```

`dataset_fingerprint()` builds the same structure from the dataset on disk,
and `model_data_status()` compares the two. Four states, and the GUI shows
exactly one of them in the LIVE CLASSIFIER panel:

| State | Shown as | Means |
|---|---|---|
| `untrained` | `NO MODEL TRAINED` (grey) | no payload at `MODEL_PATH` |
| `unknown` | `TRAINED on <date>` (amber) | a model, but no `data` block — it cannot say what it used |
| `current` | `UP TO DATE -- trained on <date>` (green) | `uid_digest` matches the dataset exactly |
| `stale` | `OUT OF DATE -- trained on <date>` (amber) | the dataset has changed; the tooltip says how many recordings are new, or how many it used are gone |

A second line names the provenance itself (`trained on 119 recordings |
FIST 56, REST 63 | 1 subject(s): Dano | 1 session(s)`), and MODEL STATS has
the full block including the dataset id and the newest recording in the
training set. With no model, that line shows what is on disk instead.

**The fingerprint is built from FILENAMES, not from file contents.** Every
recording's uid and timestamp are in its name and `unique_path()` guarantees
the name, so the whole comparison costs one directory walk — 30 ms for 119
recordings, which is what makes it affordable on a GUI timer (throttled to
once every 5 s, forced on model load, on a retrain finishing, and whenever
the panel state changes).

**`unknown` is a state, not a default.** A model that does not record its
training data is reported as not knowing, never as up to date. Same rule as
the compute badge: the honest answer to an unanswerable question is to say
it is unanswerable.

### The GPU/CPU badge in the GUI

> **Two stale-signal bugs fixed 2026-10-19, same family, both found by
> tests.** `QTimer.singleShot` cannot be cancelled, so a probe that answered
> in 8 s still had a timeout due at 30 s; when it landed it killed whatever
> probe was running *then* and reported "the device probe did not answer" --
> on a machine with a working GPU. The identical flaw existed on the
> `finished` signal: a killed probe's signal arrives asynchronously and was
> read as the *replacement* probe answering with empty output. Both handlers
> are now bound to their own `QProcess` and ignore a signal from any other.
>
> Worth remembering as a shape rather than as two fixes: **anything
> asynchronous that outlives its subject must carry the subject's identity.**
> `NeuralEngine` (§8L) is not affected -- its start timeout is computed from
> an elapsed time that `start()` resets, so there is no detached timer to go
> stale -- but the next `QProcess` added here will have the same trap.
>
> They surfaced only because `tests/test_gui.py` grew past the 30 s mark, so
> the regression check now simulates a stale timer directly instead of
> relying on the suite happening to be slow.


The recorder must never import torch (see §13), so it asks a subprocess:
`COMPUTE_PROBE_SNIPPET` + `parse_compute_probe()` live in the data layer
(Qt-free, testable headless) and `MainWindow.start_compute_probe()` runs them
through `QProcess` so the ~7 s torch import cannot stall the event loop. Two
indicators:

- the full badge in the LIVE CLASSIFIER panel — `GPU READY -- <card>` in
  green, `CPU ONLY` in amber, with a tooltip naming CUDA/torch versions;
- a compact chip in the status bar, which cannot be folded away.

Until the probe answers, both read `checking...` / `?` — **never** a guess in
either direction.

The badge states what the *hardware* offers, and its tooltip and the RETRAIN
dialog both say that the scikit-learn sweep behind that button runs on the
**CPU** regardless: sklearn has no GPU backend. A green "GPU" badge next to a
button that only ever uses the CPU would be a lie, so the distinction is
written out in both places.

## 8d. Why the sweep does not use the GPU, and what made it faster

**Asked and answered 2026-10-14: CUDA cannot accelerate this workload.**

RAPIDS cuML is the only GPU implementation of these algorithms, and it does
not exist for this machine. Resolved against the real interpreter:

```
INFO:wheel-stub: cuml_cu12-25.8.0-cp310-cp310-manylinux_2_28_x86_64.whl ...
INFO:wheel-stub: cuml_cu12-25.8.0-cp311-cp311-manylinux_2_28_aarch64.whl ...
INFO:wheel-stub: cuml_cu12-25.8.0-cp312-cp312-manylinux_2_28_x86_64.whl ...
error: metadata-generation-failed
```

Every wheel is **manylinux** and **cp310-cp312**. This machine is
`win-amd64` on **Python 3.14**. Two independent blockers; RAPIDS on Windows
means WSL2. The GTX 1650 itself is fine (compute 7.5). Even if it installed,
the data is far too small: the worst setup is ~6300 windows x 154 features,
and GPU tree/SVM kernels start winning around 10^5 rows.

Per algorithm: LDA, ExtraTrees and HistGradientBoosting have no cuML
equivalent at all; SVM, RandomForest and LogReg do, but only on Linux.
XGBoost *is* installable here and has `device="cuda"`, but it is a different
algorithm from HistGradientBoosting, so it would change the spec rather than
accelerate it.

**What actually made it faster, measured:**

| Change | Effect | Evidence |
|---|---|---|
| `probability=False` while scoring | **5.7x** on every SVM fit | 54.26 s -> 9.52 s at 6300x154; `predict()` identical, same support vectors |
| CV folds on threads | **~1.3-2.9x** on single-threaded estimators | 6.17 -> 4.66 s at 6300x154; identical scores |
| ~~`SVC(cache_size=1000)`~~ | **nothing** | 7.22 s vs 7.21 s. Predicted win, measured nothing, dropped |
| Rebuild instead of cache | memory stops growing | the old `cache[key]` held all 432 feature matrices for the whole sweep |

The SVM change is the important one and the one to understand before touching
it. `probability=True` makes libsvm run its own internal 5-fold CV to fit a
Platt sigmoid for `predict_proba`. `evaluate()` only ever calls `predict()`,
which comes from the decision function — so that work changes no score. The
saved model is built separately **with** probability, because `LivePredictor`
and the soft vote both need `predict_proba`.

**The trap, and why `voting_pool` exists.** Soft voting picks its members with
`hasattr(clf, "predict_proba")`, and an SVC built for scoring does not have
that attribute. Sharing the scoring pipelines with the ensemble would have
silently dropped the SVM from the vote and changed what was being measured.
The ensemble is therefore built from its own always-probability copies.
`train_model.py` asserts nothing about this; `tests/test_model_spec.py` does.

## 8e. The decision panel — MODEL STATS → DECISION TRAIL

Answers "how did each model decide, how do they become one, and what is the
final answer?" — live, on the armband, and on the recorded dataset.

```
train_model.py  build_decision_panel()   ->  models/gesture_panel.joblib
emg_dataset_tool.py  DecisionPanel           loads it, runs the members
                     TrailStep               one window's worth of opinions
                     DecisionTrailWidget     draws input -> 7 rows -> fusion
                     DecisionTrailView       LIVE / RECORDED + the controls
explain_votes.py                             rebuilds it without a sweep
```

**It is a SEPARATE FILE, and that is the whole design.**
`gesture_model.joblib` keeps holding exactly one estimator, because that is
what ships to the SBC on the robot. `LivePredictor` never opens the panel.
Delete `gesture_panel.joblib` and the only thing that changes is that the
view says it is missing and tells you how to rebuild it. `--no-panel` skips
writing it.

**One feature path, not two.** The live trail runs on
`LivePredictor.last_features` — the exact row the deployed model was just
given. Recomputing features inside the panel would create a second path that
could drift from `model_feature_vector()`, which is the one contract binding
training to inference. `tests/test_trail.py` checks the vector is identical
and unmodified.

**Two sources of numbers, never mixed:**

| | What it is | Honest? |
|---|---|---|
| **LIVE** | the stored members, fitted on the **whole dataset**, run on the window you just made | **No** — it demonstrates how the vote forms. The view says so in the caption, every time. |
| **RECORDED** | out-of-fold probabilities captured during training, scrubbable per window | **Yes** — the window's recording was held out of the fold that scored it. |

**Every candidate is scored on the WINNER's setup**, not on its own best one.
That is what makes seven rows comparable: one feature vector, seven opinions.
It also means these numbers are *lower* than `best_per_model` in the sweep
report, where each model picked its own preprocessing — on this dataset SVM
reads 99.15 % here against 99.8 % there. The dialog says this in words next
to the table rather than leaving someone to discover it.

**The fusion rule is measured, not claimed.** Soft voting is the unweighted
mean of the members' `predict_proba`. `build_decision_panel()` computes the
mean itself, compares it against the stored `VotingClassifier` over the whole
dataset, and records the result (`fusion.max_abs_diff_vs_mean`, measured
1.2e-07 — float32 rounding). The live path therefore does **not** call the
ensemble: doing so re-runs all seven members for a number already in hand,
and that doubling is what made the view miss windows. When the check does
*not* pass the view shows the ensemble's recorded output instead of trusting
the mean.

**The diagram must not imply the ensemble ships.** It does not: the sweep
saved the single winner. Every frame marks the deployed row with `*` and the
fusion box carries a `DEPLOYED <name> -> <label>` line. Rule 11 in §9 covers
reporting; this is the same idea applied to a picture.

**Performance, measured on this machine** (and the reason for two
non-obvious lines of code):

| | Cost | Why |
|---|---|---|
| members at `n_jobs=-1` | **2700 ms** per window | joblib stands up a worker pool per call, for one row through 400 trees |
| members at `n_jobs=1` | **50-78 ms** per window, now **161 ms** with kNN members in the spec (§12) | `DecisionPanel._serialise_members()` forces this on load |
| first call ever | **~2500 ms** | HistGradientBoosting builds its prediction kernel on demand (2416 ms first, 2.5 ms second) |
| after `warm_up()` | — | run once on a worker thread (`PanelWarmUp`) when the dialog opens, so no live window eats it |

`n_jobs` changes only how a fitted forest is evaluated, never what it
answers. The panel is loaded **lazily**, on the first MODEL STATS open, so
startup does not unpickle two 400-tree forests for a view nobody asked for.

**MODEL STATS is not modal any more.** The trail follows the live classifier,
and a modal dialog would mean you could only watch it if PREDICT happened to
be running already — you could not reach the button to start it.

**A mismatched panel is refused, not fed.** `matches()` compares labels,
feature set, representation, frequency features and window samples; on any
difference the view explains which field disagrees and stops. Feeding a panel
vectors it was never fitted on would produce confident nonsense.

## 8f. What the dataset actually is — EDA, 2026-10-15

Full analysis with every number in **`docs/EDA.md`**. The short version,
because it changes how any result from this project must be read:

> **SUPERSEDED 2026-10-15 — the dataset analysed below was archived.** All
> 325 recordings were moved whole to
> `datasets/_archive_pre_r1_20261015/` (not deleted: safety rules 4 and 5, and
> the leading underscore keeps `discover_recordings()` from ever walking back
> into it). Its README says why. **Keep it** — it is the baseline that answers
> "did the re-record change anything", and its DC-only control of 98.9 % is
> the number to beat.
>
> **What is on disk now (updated 2026-10-16): TWO wearers, 85 recordings.**
>
> | Wearer | Recordings | Takes | Sitting |
> |---|---|---|---|
> | `Dano` | 44 | 6 | 2026-10-15 23:27–23:46 |
> | `Betka` | 41 | 6 | 2026-10-16 15:19–15:33 |
>
> Both are 3–4 s clips of *relaxed* holds — the R4 fix, applied, and it
> worked. Within each wearer the old structure is unchanged: **one take per
> gesture**, so label is still confounded with take *inside* a person and the
> recording-grouped contrast still reads 97.7 %.
>
> **What the second wearer bought is the split that was impossible before.**
> `SELECTION_GROUPING = "auto"` switched the folds to subject-grouped by
> itself (§8h), and the winner now scores **83.3 % per window on a person it
> never saw**, against 19.2 % for the majority class. That is the first
> honest generalisation estimate in the project's history. The within-subject
> number did not get better — it got *exposed*: 97.7 % recording-grouped
> against 83.3 % subject-grouped, a 14-point gap that is precisely the
> subject-specificity §8h said would show up.
>
> Still open: more *sittings per wearer*, band off and refitted between, which
> is what makes leave-one-take-out valid and what would break the confound
> inside a person. Two wearers fixed the cross-person question; the
> cross-fitting question is untouched. **Per-wearer calibration (§8k, added
> 2026-10-16) took the cross-person number from 80.2 % to 86.8 % and did
> nothing at all to this one** — the within-wearer contrast moved 97.8 % to
> 98.2 %, which is the point: calibration is a between-wearer correction and
> this confound is inside a wearer.
>
> The analysis below is kept in full because the failure mode it documents is
> the one being recorded against.

**308 recordings, 1 subject, 2 sessions, 6 labels, 15.1 minutes of signal.**
`FIST` 56, `REST` 63, `OPEN` 55, `THUMB_UP` 39, `WAVE_LEFT` 42,
`WAVE_RIGHT` 53. Measured rate 49.4–50.2 Hz, `filtered` mode throughout. The
stream itself is clean: stable rate, no clipping (global max 1348 of a 2047
ceiling), no zeros.

**Nothing came from the cued sequence.** 280 of 308 clips were cut from free
takes, 27 from `label_on_graph`, 1 untagged. `SequenceController` produced
none of it, which is why `label_start_index == 0 && label_end_index ==
n_samples` for all 308 — there is no PRE/POST_GESTURE_MS context in this
dataset and `use_label_window` is a no-op on it.

**Each active class is exactly ONE take.** FIST = one 58 s take, OPEN = one
53 s take, THUMB_UP = 52 s, WAVE_LEFT = 54 s, WAVE_RIGHT = 53 s; all four
session_02 gestures were recorded back-to-back between 11:01 and 11:07 on one
day without re-donning the armband. REST is the only class with more than one
source (two takes plus the live-graph spans). So **label is perfectly
confounded with electrode state and wall-clock block.**

Three consequences, all measured:

1. **Grouped CV by parent take is impossible**, not merely optimistic: a
   leave-one-take-out split puts a whole class in the test fold with none of
   it in training. Combined with one subject (so `leave_one_person_out()` has
   nothing to split), **this dataset currently admits no honest
   generalisation estimate at all.**
2. **The ~99 % scores do not need the gesture.** Reproducing the trainer's
   protocol gives LDA 0.996. Handing the same classifier *only the 8
   per-channel mean levels* — no shape, no spectrum, no timing — gives RF
   0.993. One number, the overall mean level, gives 0.757 against a 0.187
   baseline. The model is reading which 50-second block it is looking at.
3. **74 of the 308 recordings are byte-identical duplicates** of another
   recording (THUMB_UP: 39 clips from 13 distinct signals), and the rest tile
   their take with ~50 % overlap. Grouping by recording uid does not separate
   them, because every clip legitimately has its own uid. After dropping
   duplicates and overlaps, 105 distinct observations remain.
   `train_model.py`'s `source_timeline()` / `dedupe_windows()` remove this
   leak at window level; they do not touch 1 or 2.

**Also worth knowing:** sustained 52 s holds drive heavy fatigue drift (FIST
loses 59 % of its amplitude across its take, Pearson r = −0.83; WAVE_RIGHT
r = −0.90), and because every take is ~100 % contraction there are **no
onsets, offsets or gesture-to-gesture transitions anywhere in the dataset** —
which is most of what a live classifier sees.

**This already breaks live use, visibly.** A held thumbs-up is recognised for
a few seconds and then flips to `REST` with the hand still in the pose.
Measured per-window mean amplitude: `REST` reaches 43 at its 95th percentile
and **nothing labelled `THUMB_UP` exists below 171**, so the band 43–171
carries no label at all and `REST` is its nearest neighbour. Same void for
`OPEN` (43–128) and both waves (43–254); `FIST` has the mirror problem, its
p5 of 52 sitting on REST's p95. The cause is that a 52 s hold can only be
recorded as an *effortful* contraction, while a real gesture is made and then
relaxed into — and the relaxed hold lands in the void. It is not fixable by
scaling: a THUMB_UP window still reads THUMB_UP at 20 % gain, because `share`
carries the spatial profile with amplitude divided out. `docs/EDA.md` R4 has
the full table and the recording fix. On the good side, the six
classes have genuinely distinct spatial signatures (WAVE_LEFT peaks on
CH6/7/8, WAVE_RIGHT on CH2/3/1 — near mirror images), label fidelity is
clean (no active clip is more than 50 % quiet), and REST now reports no
activity 57 times in 63, which is the right answer.

**Rules this adds for anyone working here:**

- **Never quote an accuracy from this dataset without the confound.** Same
  discipline as §8b's majority-class baseline rule: the sentence "each class
  is one take, so this is not a generalisation estimate" belongs next to the
  number.
- **Do not interpret a high score as progress on gesture recognition.**
  Verify against the DC-only control in `docs/EDA.md` §3 first: if 8 channel
  means still score as well as the full vector, nothing has been learned
  about gestures.
- **Group by `meta["parent_take_uid"]`, not by uid**, when adding any new
  validation path.
- The fix is in the recording, not the model: several separate takes per
  gesture, armband removed and refitted between them, gestures interleaved.
  `docs/EDA.md` §8 has this as R1–R10, in priority order.

## 8g. Why the sweep is fast now — cost controls, 2026-10-15

Full write-up in **`how_to_fix_long_training.md`**. The sweep went from a
**measured 3 h 16 m** (11 747 s, 119 recordings) to **~14 min** on 308
recordings. Four changes, all with measured costs:

| Change | Effect | Accuracy cost |
|---|---|---|
| `DEDUPE_WINDOWS` — drop windows that re-cover covered signal | 5 598 → 2 425 windows | **negative**: it removes a leak |
| identical `EXCLUDE_SETS` folded together | 360 → 120 distinct setups | none; it was the same computation three times |
| stage A1 screens on `SCREEN_FRACTION` of windows with `SCREEN_PROBES` | cheap probes only, ~0.7 s/setup | none reported; A1 scores are never quoted |
| `A2_SHORTLIST = 32` instead of all 360 | 14× fewer expensive probes | **0.02 points**, measured |

**The dedup is correctness, not a speed-up.** 280 of 308 recordings are clips
cut from a few free takes, tiling them with ~50 % overlap, and 74 are
byte-identical to another clip (§8f). Grouping CV by recording uid does *not*
separate them, because every clip legitimately owns its uid — so without dedup
the same samples sat in train and test. Removing it moved the score 99.7 % →
99.0 %, which is the leak closing. `dedupe_windows()` works at **window**
level, not clip level: it maps each window's centre onto the timeline it was
cut from (`parent_take_uid` + `take_start_index`) and keeps the first window
per step-sized bucket, which is exactly windowing each source timeline once.
Clip-level dropping was tried and is measurably worse (108 recordings kept,
98.0 %).

**Why a 32-setup shortlist is safe**, checked against the last full-grid sweep
where all 360 setups were scored by all three probes: the best per-window
accuracy reachable inside a 40-setup shortlist was 99.63 % against 99.65 % for
the whole grid, and the per-recording vote was identical. Dozens of setups tie
at the top. The shortlist is 24-best **+ 8 seeded-random** because the cheap
probes' own top-20 was 0.28 points short on the vote while a purely random 40
was not short at all — the screen ranks well but not perfectly, so the
shortlist deliberately does not trust it completely.

**A1 rows are not comparable with A2/B rows** and are kept out of `results`
for that reason: they are measured on half the windows. They go to the report
under `screen`, and `cost_controls` records exactly how the sweep was cut
down, so a fast run can never be mistaken for an exhaustive one.

**The trainer now runs the §8f control on every run.** It prints the
majority-class baseline and re-scores the winner using only the 8 per-channel
mean levels, then says how much the other features added. On the current
dataset: baseline 28.4 %, DC-only 97.2 %, full vector 99.3 % — so the shape,
spectrum and timing are worth about two points and the model is mostly reading
which 50-second block it is looking at. Under 5 points of gain the run prints
a warning naming the confound. Both numbers go into the sweep report
(`confound`) and the saved payload (`stats.dc_only_control`), so §8f's "never
quote an accuracy without the confound" rule is enforced by the tool rather
than by memory.

## 8h. Multi-subject readiness — what switches on by itself

Built 2026-10-15, before the data exists, and tested on a synthetic
three-subject dataset. **More wearers change what "honest" means, and the
pipeline now follows that change on its own.**

`SELECTION_GROUPING = "auto"` in `train_model.py`:

| Subjects on disk | Folds grouped by | Why |
|---|---|---|
| 1 | recording uid | a subject-independent split is impossible |
| 2 or more | **subject** | the held-out person's data is never in training |

**Why this had to change, and it is not a detail.** The sweep picks the
winner from cross-validated scores. With several wearers but folds grouped by
*recording*, every subject sits in both train and test — so a setup that leans
on one person's absolute amplitudes scores well and gets **selected**, then
fails on the next wearer. That is the opposite of what collecting more people
is for. `leave_one_subject_out()` already reported the honest number, but it
ran *after* the winner had been chosen, so it could only document a bad choice,
not prevent it.

The split key and the vote key are deliberately separate in `evaluate()`:
`split_groups` decides the folds, `groups` stays the recording id because the
per-recording vote must still be counted per recording. Passing one for the
other would either leak a subject or destroy the vote.

**Both protocols are always reported.** The winner block prints the chosen
one and then the *other* as a contrast, and warns when they disagree by more
than 5 points. On the synthetic three-subject set: subject-independent 94.4 %
against within-subject 74.1 %, flagged as a 20.4-point disagreement. The gap
*is* the subject-specificity; showing one number alone would let a
within-subject score be read as a cross-wearer one. `validation.grouping`,
`validation.grouping_contrast` and the payload's `selection_grouping` carry
the same pair.

**The DC-only control (§8f) becomes a useful signal rather than a verdict.**
Measured: on the current one-subject dataset it reads **97.2 %** against the
full vector's 99.3 % — the red flag, because absolute levels identify the
take. On the synthetic three-subject set the same control reads **18.5 %**
against a 25 % baseline, because levels do not transfer between wearers. So
the control flips from "nothing was learned" to "all clear" exactly when it
should, and that is how to read it going forward: **cross-subject, a high
DC-only score means the subjects differ in gain, not that the gestures were
recognised.**

**What is still NOT automatic:**

- **Take-grouped CV.** Once a gesture has two or more takes, leave-one-take-out
  becomes possible (verified synthetically: with 3 takes per class "every class
  survives every split"). The trainer does not do it yet, because on today's
  data a leave-one-take-out split removes a whole class from training (§8f
  consequence 1). Add it when the takes exist, not before.
- ~~**Amplitude-invariant preprocessing.**~~ **DONE 2026-10-16, but not the
  way this bullet expected.** The prediction was that the sweep would start
  preferring `normalized`, which divides out amplitude per window. It did
  not — `normalized` still loses badly (22–48 % in every measurement), because
  dividing each *window* by its own scale throws away the activation level
  along with the wearer's gain. What worked is a per-WEARER constant measured
  once: `CALIBRATION_MODE = "affine"` (§8k), 80.2 % → 86.8 % subject-grouped.
  The distinction is the whole lesson — per-window normalisation and
  per-wearer calibration both "divide out amplitude" and only one of them
  keeps the information.
- **Electrode rotation between wearers.** Nothing here handles the armband
  being donned at a different angle, which permutes the channels. The `share`
  feature carries the spatial profile with amplitude divided out, and §8k's
  `per_channel`/`affine` modes equalise per-channel *gain*, but neither is
  rotation alignment: a permutation of the channels is not a rescaling of
  them. Expect this to be the next real problem.
- **Per-subject sample balance.** With unequal recordings per person the
  subject-grouped folds are unequal too. `StratifiedGroupKFold` balances the
  labels, not the people.

## 8i. The band-pass filter, and why "filtered" gets no high-pass

Added 2026-10-15. `bandpass_filter()` in SECTION 2 is a real zero-phase
Butterworth, and before it existed the only filtering in the project was
`moving_rms`/`moving_mean` (crude low-passes) and `baseline_level`
subtraction (a crude DC block).

**It does not import scipy.** The `HAVE_SCIPY` comment records that
`scipy.signal` costs ~7.3 s on a cold cache and that nothing here actually
imports it; a filter that runs in the live prediction path is not the place to
start. `_butter2_coeffs()` writes a 2-pole section out via the bilinear
transform instead, with `tan()` pre-warping so the cutoff lands where it was
asked for. Verified against `scipy.signal.butter` to 1e-12 when scipy happens
to be installed, and against theory when it is not: -6 dB at the cutoff (the
forward-backward pass doubles one section's -3 dB), monotone roll-off, DC gain
exactly 1 low-pass / 0 high-pass.

**Zero phase, and that is not a nicety.** A causal filter delays the
envelope, and a delayed envelope moves a gesture's onset relative to the
`label_start_index`/`label_end_index` it was cut with -- it would shift the
label boundaries the whole dataset is built on. `_filtfilt()` runs the section
forwards then backwards so the phase cancels exactly.

**`EMG_BANDPASS_HZ["filtered"] = (None, 8.0)` -- low-pass only. The missing
high-pass is a decision, not an omission.** Three reasons, and the third is
measured:

1. The Myo band-passes and rectifies **on-device**. What arrives in
   `filtered` mode is already an envelope; the textbook 20-450 Hz stage
   happened upstream in hardware.
2. The envelope arrives at ~50 Hz, so Nyquist is ~25 Hz. A 450 Hz low-pass is
   not merely unnecessary, it is unrepresentable, and a 20 Hz high-pass sits
   at 0.8x Nyquist.
3. A high-pass on an envelope removes its DC, and **the DC here is the
   activation level**. This project measured that twice, and both times it
   cost accuracy: median-centring moved the Fisher ratio 25.5 -> 15.1 and
   accuracy 43.2 % -> 33.5 %, and baseline subtraction moved Fisher
   24.9 -> 11.5-17.2. See the "Absolute amplitude is kept" decision in §13.

`EMG_BANDPASS_HZ["raw"] = (20.0, 95.0)` is the opposite case: a true 200 Hz
bipolar stream with nothing applied to it, where a band-pass is standard and
genuinely absent, and 95 Hz fits inside its 100 Hz Nyquist.

**A cutoff at or above Nyquist is SKIPPED, not clamped**, and the reason goes
into the caller's `report`. Clamping 450 Hz down to 24.9 Hz would apply a
completely different filter and still call it 20-450 Hz. At these sample rates
that is not a corner case -- it is what happens the first time someone pastes
a textbook band from a 1 kHz setup.

**It is a REPRESENTATION, not a file transform.** `"bandpass"` joins `raw`,
`normalized` and `envelope` in `REPRESENTATIONS`, so it is applied inside
`model_feature_vector()` -- the one function shared by the trainer and
`LivePredictor`. Three consequences worth knowing:

- Nothing on disk is rewritten, so safety rules 1 and 3 are untouched. There
  is no filtered copy of the dataset to keep in sync.
- The existing dataset is filtered on every run, automatically, because
  filtering happens at feature time rather than at record time.
- Whether it *helps* is measured by the sweep like everything else rather than
  asserted. Measured on the 325-recording dataset at one fixed setup
  (0.40 s / 0.5 overlap / `level`, LDA-shrinkage): `raw` 98.07 % window /
  99.10 % vote, `bandpass` 98.15 % / **99.55 %**, `envelope` 87.75 % /
  95.05 %, `normalized` 22.34 % / 24.32 %. So it is a hair ahead of `raw` and
  clearly ahead of the other two -- a candidate, not a default.

The fourth representation grows the sweep from 120 to 160 setups.

## 8j. MODEL STATS -> FEATURE SPACE

Added 2026-10-15. Every training window as one point in two dimensions, with
the live window drawn on the same axes.

```
train_model.py        feature_space_projection()  -> panel["space"]
emg_dataset_tool.py   DecisionPanel.project()        one live vector -> (x, y)
                      FeatureSpaceWidget             the scatter
                      FeatureSpaceView               controls + caption
```

**PCA, not LDA, and that is safety rule 11 applied to a picture.** An LDA
chooses its axes to separate the labels, so an LDA scatter looks beautifully
separated even when the separation comes from electrode drift rather than from
the gesture -- it would flatter the model by construction, the same way the
§8e diagram must not imply the ensemble ships. PCA never sees `y`, so what it
draws is the structure that is really in the features.

**One transform, used in both directions.** The trainer stores
`mean`/`scale`/`components`, and `DecisionPanel.project()` is the only thing
that applies them. Fitting a projection on live data would move the axes under
the scatter and make the two incomparable -- the same reason the trail reuses
`LivePredictor.last_features` instead of recomputing features. A vector of the
wrong width is refused rather than reshaped, because a reshaped vector lands
somewhere plausible and means nothing.

**Three colourings, because each answers a different question:**

| Colour by | Question it answers |
|---|---|
| gesture | do the labels form their own clusters? |
| take | or do the clusters follow the recording session? If this looks *tidier* than "gesture", the model is reading the block -- §8f made visible |
| amplitude level | where the effort sits, and the gap a relaxed hold falls into -- `docs/EDA.md` R4 made visible |

The live ring leaves a short fading tail, so a hold decaying out of its own
cluster reads as a direction rather than a jumping dot. A live point outside
the training cloud is still drawn: that it *is* outside is the interesting
case.

It shares the DECISION TRAIL's 120 ms timer rather than adding a second one,
and the scatter is thinned to `max_points` (6000) evenly rather than randomly,
so every recording keeps representation. `has_space` is gated on `loaded`
rather than `ready`: drawing the scatter runs no model, so a panel whose
members are unusable can still show its projection. A panel built before this
existed has no `space` block, and the view says so and names
`explain_votes.py` instead of drawing empty axes.

## 8k. Per-wearer calibration — optional, measured, and it won

Added 2026-10-16, answering §13's "offset/gain invariance belongs in the
future per-user calibration" and §8h's open item on wearers differing in
gain. It shipped **optional and off**, was measured by three full sweeps, and
`CALIBRATION_MODE` is now **`"affine"`** because it is worth **+5.3 points**
on the only honest number this project has — measured with the calibration
itself held out, which is the gate `calib_plan.md` §5 demanded.

**It is still one config line.** `CALIBRATION_MODE = "off"` plus a retrain
reproduces every number this project reported before 2026-10-16, bit for bit,
and the code path when it is off is the old one unchanged — calibration
resolves to `None` and `None` means "do nothing" everywhere.

**The price of the new default is real, and it is enforced rather than hoped
for:** a model fitted on calibrated features obliges every wearer to
calibrate before it will predict for them, and `LivePredictor.load()` refuses
with instructions rather than answering on the wrong scale. That is the
trade: +5.3 points, at the cost of six seconds per wearer and a hard failure
if they skip it.

```
emg_dataset_tool.py  SECTION 2   CALIBRATION_MODES, calibration_levels,
                                 validate_calibration, resolve_calibration,
                                 apply_calibration, calibration_record,
                                 calibration_from_captures,
                                 calibration_from_recordings,
                                 describe_calibration
                     SECTION 4   calibration_path / _log_path,
                                 load/save/clear_calibration,
                                 load_calibrations, calibration_table,
                                 calibration_status
                     SECTION 7   CalibrationDialog,
                                 MainWindow.open_calibration
train_model.py                   --calibration MODE, threaded through
                                 build_dataset -> feature_vector
```

**What a calibration is.** Two levels per channel — a REST level and a
REFERENCE (strong-contraction) level — measured once for one wearer, six
seconds of signal, labels never involved. Together they define an affine map
onto a common scale, so that a given number means the same amount of effort
on anybody. Four modes, and the choice is a real trade-off rather than a
quality setting:

| Mode | What it divides by | What it costs |
|---|---|---|
| `off` | nothing | — (the default) |
| `gain` | ONE scalar, all 8 channels | nothing: every ratio between channels survives exactly, so the spatial profile the gestures actually differ in is untouched. `tests/test_calibration.py` asserts this bit-for-bit, and that every `_share` feature is literally unchanged |
| `per_channel` | each channel by its own reference | flattens per-channel differences that may be real, not just differences in where the pods sat |
| `affine` | each channel, rest subtracted first | the most aggressive; below-rest signal is floored at 0 |

**It stays valid until it is replaced.** RECALIBRATE measures a new one.
Nothing expires, nothing is re-estimated per recording, and a calibration
survives restarts because it lives in the wearer's own folder:
`datasets/<person>/calibration.json`, with every calibration that wearer has
ever had in the append-only `calibration_log.jsonl` beside it. Per wearer and
not next to the model on purpose — that placement is what lets the trainer
calibrate each training subject with that subject's own numbers while the GUI
calibrates whoever is wearing the band right now, out of the same files.

**Why `affine` is not the §13 baseline-subtraction mistake.** §13 records that
baseline subtraction measurably hurt (Fisher 24.9 → 11.5–17.2). That was a
**per-recording** baseline, estimated from the recording itself, which partly
cancels the gesture it is supposed to be a baseline for. A calibration
baseline is one constant per wearer, measured once from a dedicated rest
capture, and cannot cancel a gesture because it never sees one. Different
operation, different risk — which is why it is a mode to measure rather than
a default.

### How a calibration is measured — the guided gesture sweep

**Cue REST, then each gesture in turn, and take each channel's reference as
the loudest that channel ever got across the set.**

```
reference[c] = max over ACCEPTED gestures of percentile(capture_g[:, c], 95)
rest[c]      = median of the REST capture
```

**One generic squeeze is not enough, and the gap is measured.** A squeeze only
drives the muscles a squeeze uses, so the channels it leaves quiet get their
full scale set from almost nothing — and then `per_channel`/`affine` divide
those channels by noise. Real evidence for the shape of the problem: `Dano`'s
derived reference is `[553, 657, 154, 219, 919, 992, 364, 404]`, so CH3 and
CH4 top out at 154 and 219 where CH6 reaches 992. On the synthetic stand-in
built from those profiles, a one-gesture reference **under-scales seven of
eight channels by 1.9×–3.2×**, the two quietest by 2.2× and 2.9×.
`tests/test_calibration.py` asserts that gap rather than describing it.

The **max** rather than a pooled percentile, for two reasons: "how loud does
this channel ever get" should not be diluted by the four gestures that leave
it quiet, and the answer then does not depend on how long each capture ran.

**Each capture is judged as it lands, so a gesture that did not register can
be redone on the spot.** `calibration_capture_quality()` returns the sample
count, the peak channel and the contrast against the wearer's own rest, and a
capture is accepted as evidence only if it clears **both**
`CALIBRATION_MIN_GESTURE_CONTRAST` (2×) **and**
`CALIBRATION_MIN_GESTURE_LEVEL` (60 counts). Both gates are needed, and the
second one was added because the first is not enough: noise on a channel
sitting near rest inflates a high percentile, so a shrug at 32 counts against
a rest of 21 reads as 2.5× and sails through. The absolute floor is set from
the measured spread on this hardware — REST p95 is 24–27 counts and the
weakest real gesture any wearer produced is 183 (`Betka`'s THUMB_UP) — so 60
has clear headroom in both directions.

**A rejected capture contributes nothing; it does not drag its channels
down.** It stays on screen with the reason ("peak only 41 counts, need 60 —
hold it harder") and is excluded from the max until it is redone. That is the
"keep going until it is right" part, and it is a gate rather than a loop: the
wearer decides when to stop, and the dialog will not pretend a shrug was a
gesture.

**REST is calibrated too, with its own row and its own gate** (added
2026-10-21). It is the first row of the sweep table, is judged by
`calibration_rest_quality()`, and can be redone like any gesture. It is
rejected if any channel's median exceeds `CALIBRATION_MAX_REST_LEVEL` (50
counts; measured relaxed rest is 19–26) — a hand left tense would raise the
floor `affine` subtracts and quietly fail every gesture judged against it.
A rejected rest builds **nothing**: SAVE stays off until it is redone, and the
gestures are judged against the hardware floor meanwhile. The median, so one
twitch does not fail it; and the ceiling sits below the 60-count gesture
floor, so a rest can never pass as a gesture.

The quick **rest + one squeeze** path is kept, because six seconds is better
than nothing, and `DERIVE FROM RECORDED DATA` is kept because it needs no
armband. All three write the same record shape; `source` says which produced
it (`live_gesture_sweep`, `live`, `derived_from_recordings`) and the sweep
also stores every gesture's contrast and verdict under `gesture_sweep`, so a
stored calibration can always say how it was obtained.

> **The two wearers on disk were calibrated by the DERIVED route**, not by a
> sweep — the feature did not exist when their data was recorded. The derived
> route pools all of a wearer's samples and takes per-channel percentiles,
> which estimates the same quantity by a different and slightly cruder path:
> it cannot balance across gestures, it just happens to see all of them.
> The **+5.3 points** in §8k's leak gate was measured with derived
> calibrations, so it is a *floor* for what the sweep should achieve rather
> than a measurement of it. Re-measuring with swept calibrations needs a
> wearer and an armband, and has not been done.

### Readiness is judged on samples, not on the link

`CalibrationDialog.stream_state()` reads `stats()["last_sample_age"]`, because
what a capture needs is samples: a connected armband that has gone to sleep
supplies none, and §12 records that it does exactly that when idle. Three
distinct answers — no acquisition, no samples yet (naming the connection
state), and no sample for N seconds — shown live in the dialog the whole time
it is open rather than only after a button press fails. A capture that loses
the stream half way through is abandoned with a message instead of storing a
stub.

> **This replaced a hard bug, worth recording because the class of it recurs
> here.** The first version asked `getattr(acq, "running", False)`, and
> `MyoAcquisition` has no `running` attribute — it has a private
> `_running` Event and a `connected` property. The check was therefore
> **always False**, so CALIBRATE could never capture anything, with or without
> hardware, and said "Not streaming" either way. Same shape as
> `free_capture` vs `free_take_capture` and `log_box` vs `log_view` before
> it: a `getattr` default silently turning a wrong attribute name into a
> plausible answer. `tests/test_calibration.py` now pins that
> `MyoAcquisition` has no `running` attribute and that `last_sample_age` is
> in `stats()`, so the next person to reach for one gets a failing test
> rather than a dead button.

### A bad calibration is REFUSED, never coerced

`validate_model_spec()` coerces, because a clamped hyperparameter still trains
something sensible. `validate_calibration()` **refuses**, because a gain does
not work that way: a "maximum effort" capture where the wearer did nothing
would divide their whole signal by noise. Four gates —
`CALIBRATION_MIN_REFERENCE` (40 counts), `CALIBRATION_MIN_CONTRAST` (2×),
`CALIBRATION_MIN_SAMPLES` (24), and reference-above-rest on every channel —
and the refusal happens before the record reaches the disk. `resolve_*`
returns `None` for anything that fails, and `None` everywhere means "behave
exactly as the uncalibrated pipeline", so an unusable calibration degrades to
no calibration and never to a wrong one.

### One feature path, and a refusal rather than a wrong answer

Calibration is applied inside `model_feature_vector()` — the one function
`train_model.py` and `LivePredictor` both call, for the same reason the
representations live there. The trainer resolves the table once per wearer
(`calibration_table()`) and hands each recording's windows the numbers of the
wearer who produced them; a wearer with no calibration is left uncalibrated
rather than borrowing someone else's gain, and the trainer prints a MIXED
SCALES warning when that happens.

**The payload records what was APPLIED, not what was asked for.** `mode` is
the scale the features were actually built on and `requested` is the flag the
run was given; they differ only when a mode was set and *no* wearer had a
usable calibration, in which case nothing was applied and `mode` is `"off"`.
Getting this wrong is not cosmetic — stamping `"affine"` on plain device
counts produces a model the app refuses to load, which is to say a trainer
that locks its own output out of its own application. The run says so in
words when it happens.

The payload also records whose calibrations built it, and
**`LivePredictor.load()` refuses a model it cannot reproduce the scale for.**
A model fitted on calibrated features and then fed device counts does not
break — it answers at 7 Hz with every amplitude feature off by that wearer's
gain, which is worse. Same rule as §8e's "a mismatched panel is refused, not
fed". The panel carries `calibration_mode` too, so `matches()` catches a panel
rebuilt on a different scale.

### Applied to the existing data: derived, with labels unused

The 85 recordings predate the feature, so both wearers' calibrations were
**derived** from signal already on disk — `calibration_from_recordings()`,
which pools all of one wearer's samples and takes the 10th and 95th
percentiles per channel. It deliberately does not read `rec.label`:
computing the rest level from the recordings labelled `REST` would use the
held-out wearer's **labels** to build the transform their own test fold is
scored through, which is leakage. Pooling everything uses only their
unlabelled signal. `tests/test_calibration.py` shuffles every label in the
fixture and asserts not one number moves.

Measured, both wearers, 2026-10-16:

| | rest (per channel) | reference (per channel) | contrast |
|---|---|---|---|
| `Dano` | 20 22 21 20 22 25 20 19 | 553 657 154 219 919 992 364 404 | 25.2× |
| `Betka` | 19 21 22 22 22 22 26 20 | 397 528 285 280 910 1077 496 471 | 25.5× |

**Read those two rows carefully — they predict which mode will work.** The
wearers' pooled p95 differ by only **9 %** overall (737 vs 673) and their
rest levels are identical to within a count, so between these two people
there is almost nothing for a single master-volume divisor to correct. That
is why `gain` does nothing (measured below: 79.7 % against 80.2 %
uncalibrated). What *does* differ is the per-channel profile, by up to
**1.85×** (CH3) — where the pods sat — and that is the difference `affine`
removes, which is why `affine` is the mode that pays.

Adding a calibration does **not** move `dataset_fingerprint()` — it is not a
recording and discovery never sees it — so no saved model goes stale when one
appears. That is asserted, because the alternative would be every model on
disk reading `OUT OF DATE` the moment anybody calibrated.

### What it is worth, measured — and `affine` won

**Measure this with the SWEEP, not at a fixed setup.** Both were run, and they
disagree, because the choice of preprocessing is not independent of the
calibration. The fixed-setup A/B is kept below as the record of a wrong
method.

**Three full sweeps, same 85 recordings, same spec, same folds**, differing
only in `--calibration`. Subject-grouped (the held-out wearer's data is never
in training), against a 19.3 % majority class:

| | `off` | **`affine`** | `gain` |
|---|---|---|---|
| winner | LDA-shrinkage | LDA-shrinkage | Vote-soft-6 |
| its setup | 0.5 s / 0.5 / `raw` / **`level`** (50 feat) | 0.5 s / 0.5 / `raw` / **`both`** (154 feat) | 0.5 s / 0.5 / `raw` / `level` (50 feat) |
| **per window** | 80.2 % ± 0.1 | **86.8 % ± 4.5** | 79.7 % ± 0.4 |
| **per recording** | 80.0 % | **87.1 %** | 80.0 % |
| macro F1 | 73.8 % | **83.2 %** | 73.3 % |
| recording-grouped contrast | 97.8 % | 98.2 % | 98.0 % |
| **protocol gap** | 17.6 pts | **11.4 pts** | 18.2 pts |
| DC-only, same protocol | 54.4 % (+25.8) | 68.2 % (+18.6) | 74.3 % (+5.5) |
| held out `Betka` / `Dano` | 80.1 / 80.3 | **91.3 / 82.4** | 79.4 / 80.1 |

`affine` is **+6.6 points per window and +7.1 on the vote** here, both wearers
improve when held out, and the gap between the two protocols — which *is* the
subject-specificity (§8h) — narrows from 17.6 to 11.4 points. That is
calibration doing the one thing it is for.

**These three columns are how the mode was CHOSEN, and they are optimistic.**
Each wearer's calibration was derived from all of their own recordings,
including the ones scored in their held-out fold. The honest re-measurement is
the next subsection and it is the number to quote: **+5.3 points**.
`CALIBRATION_MODE` was set to `"affine"` and the model retrained only after
that gate passed.

**Why it helps is more interesting than that it helps.** All three sweeps
picked the same preprocessing; the only thing that changed is which *feature
family* won. Uncalibrated, `level` wins and `both` is dead weight — §13
records why: `extract_features()` median-centres each channel, which on this
rectified envelope deletes the activation level. Calibrated with `affine`,
the wearer's rest level has already been removed and the scale normalised, so
the classic shape features (waveform length, zero crossings, slope sign
changes) become **comparable between wearers** and start paying: 50 features
→ 154, and macro F1 up 9.4 points, a bigger relative move than the accuracy.
Calibration did not merely rescale the inputs — it made a feature family
transferable that was not.

**`gain` is the control that confirms the mechanism.** It leaves the full
score flat (79.7 vs 80.2) while lifting the DC-only control by **20 points**
(54.4 → 74.3). That is exactly what dividing by one number per wearer should
do: it makes the eight channel means transfer between people without adding
one bit of information about the gesture. A reminder that a *rising* DC-only
score is not good news.

### The leak gate — run it, because the sweep number is optimistic

`calib_plan.md` §5 names the trap, and the sweep walked into it: every wearer's
calibration was **derived from all of that wearer's recordings, including the
ones then scored in the held-out fold.** That is fitting the transform on the
test set. It needs measuring, not describing, so:

```
for each wearer W:
    model <- LDA-shrinkage fitted on every OTHER wearer
    C_W   <- derived from a DISJOINT third of W's recordings (every 3rd per
             label), labels unused
    score <- that model on W's REMAINING recordings, calibrated with C_W
```

Setup fixed to the sweep's winner, nothing selected, 571 windows scored:

| Protocol | window | vote | macro F1 | `Betka` | `Dano` |
|---|---|---|---|---|---|
| uncalibrated | 81.3 % | 80.4 % | 80.4 % | 83.1 | 80.0 |
| `affine`, **leaky** (what the sweep did) | 86.9 % | 90.2 % | 86.0 % | 91.9 | 83.3 |
| **`affine`, honest** (held-out calibration) | **86.5 %** | **88.2 %** | **85.3 %** | 90.7 | 83.6 |
| `per_channel`, honest | 85.6 % | 86.3 % | 84.6 % | 91.5 | 81.5 |
| `gain`, honest | 81.1 % | 80.4 % | 80.2 % | 83.1 | 79.7 |

**The leak was worth 0.4 points per window and 2.0 on the vote. The honest
gain is +5.3 per window and +7.8 on the vote, and it clears the gate.**

**Why the leak is so small is the argument for keeping the scheme stupid.** A
calibration here is sixteen numbers — two percentiles per channel — and a
statistic with that little capacity has nowhere to memorise a test set. The
honest and leaky reference vectors agree to within 10 % on every channel
(ratios 0.93–1.10). A richer fit, such as `calib_plan.md` §3.4's 64-parameter
tier 3, has no such defence and would have to clear this gate on its own
merits.

It also separates the two things `affine` does: `per_channel` (divide by the
per-channel reference) is worth +4.3, and subtracting the wearer's rest level
on top of it adds the remaining **+0.9**. So the per-channel map is where
nearly all of it comes from, exactly as `calib_plan.md` §2.6 predicted, and
`gain` at **−0.2** is its prediction for a single scalar confirmed to the
decimal.

**Caveats, and they are not small:**

- **Two wearers means two folds.** The ± on `affine` is 4.5 against 0.1 for
  `off`, and that spread *is* the Betka/Dano difference (91.3 vs 82.4): the
  calibration helps one wearer a great deal and the other modestly. With
  n = 2 there is no way to tell which of those is the typical case. **This is
  the biggest remaining caveat on the whole result** — +5.3 points is the
  average of +7.6 and +3.6 on a sample of two people.
- **Nothing here was validated on a wearer who was not in the reference.**
  Both wearers' calibrations map onto a fixed `CALIBRATION_TARGET_LEVEL`
  rather than toward a stored reference frame, so there is no frame to leak —
  but there is also no third wearer to check the scheme transfers to.
- **935 windows** survive dedup at the winning setup. Small.
- The DC-only numbers are **not comparable across the three columns**,
  because a different feature set won in each; only the gain within one
  column is.
- The within-subject contrast barely moves (97.8 → 98.2), as expected:
  calibration is a *between*-wearer correction and the §8f confound is
  *within* a wearer.

**Kept for the record: the fixed-setup A/B, which pointed the other way.**
At the previously-saved model's own setup (0.30 s @ 0.75, `bandpass`/`level`
+freq, 3573 windows), subject-grouped per window:

| Mode | LDA-shr | ExtraTrees-400 | RF-400 | SVM-rbf-C10 |
|---|---|---|---|---|
| `off` | 74.8 | **83.3** | 78.7 | 73.5 |
| `gain` | 75.2 | **83.5** | 79.6 | 74.1 |
| `per_channel` | 74.0 | 80.7 | 76.3 | 78.2 |
| `affine` | **80.7** | 82.0 | 77.4 | 79.7 |

Read on its own this says calibration is a wash — `affine` worth +5.9 to LDA
and +6.2 to SVM but −1.3 to both forests, nothing beating uncalibrated
ExtraTrees. **It is the wrong measurement.** Holding the setup fixed holds it
at the setup that won *uncalibrated*, and the whole point is that calibration
changes which setup is best. The lesson generalises: anything that alters the
feature distribution has to be compared by re-running the sweep, not by
swapping it in underneath the previous winner.

**It is NOT a sweep axis, and that is deliberate.** Representation is one
because the choice is invisible to the user; calibration is not, because a
calibrated model obliges every future wearer to calibrate before they can use
it. That is a deployment decision, not a hyperparameter, so it is a flag the
operator sets (`--calibration`, which the GUI passes from `CALIBRATION_MODE`)
and the runs are compared deliberately. It also keeps the sweep at 160 setups
instead of 640.

### Rules this adds

- **A calibrated subject-grouped score is NOT zero-shot.** The held-out
  wearer's own unlabelled signal set their gain, so it answers "how well does
  this work on a new wearer *who calibrates first*" — a real deployment, but a
  different question. The trainer prints this in words on every calibrated
  run, and the uncalibrated run is the comparison.
- **Calibration does not fix the §8f confound and does not claim to.** The
  confound is *within* a wearer — one take per gesture per person — and a
  per-wearer constant is a *between*-wearer correction. Measured: the
  recording-grouped contrast barely moves (97.8 % → 98.2 %). More sittings
  per wearer, band off and refitted between, is still the fix.
- **A rising DC-only control is not good news.** `gain` lifted it by 20
  points (54.4 % → 74.3 %) while leaving the full score flat. Making the
  eight channel means transfer between people is not the same as learning
  anything about gestures — §8h already said to read a high cross-subject
  DC-only score as "the wearers differ in gain", and here it was *caused*
  deliberately.
- **Compare anything that changes the feature distribution by re-running the
  sweep.** A fixed-setup A/B held at the uncalibrated winner's setup said
  calibration was a wash; the sweep said +6.6 points and the leak gate
  confirmed +5.3, because calibration
  changes which setup and which feature family is best. Both measurements are
  below, and the fixed-setup one is kept as the record of a wrong method.
- **Never mix scales.** Calibrate every wearer or none. The trainer warns, but
  a feature space where some people are calibrated and some are not is not one
  scale and its scores mean nothing.

## 8L. The neural network — back, on CUDA, and deliberately NOT in the answer

Added 2026-10-19. The retired experiment (§8b) is back as a **separate second
opinion**: it has its own trainer, its own checkpoint, its own process, its
own START button, its own probability bars and its own statistics dialog, and
its output is never averaged into anything.

```
nn_arch.py            the five architectures -- torch, and nothing else
train_nn.py           the sweep: same protocol as train_model.py, own output
nn_serve.py           the live worker: the ONLY process that imports torch
emg_dataset_tool.py   SECTION 2   nn_input_block
                      SECTION 5b  nn_stats_path / load_nn_stats,
                                  nn_encode_request / nn_parse_line,
                                  NeuralPrediction, NeuralEngine
                      SECTION 7   NeuralStatsDialog,
                                  MainWindow._group_neural / toggle_neural /
                                  _tick_neural / train_neural_net
tests/test_nn.py      158 checks, and the first 30 are the separation rule
```

### The separation rule — three mechanisms, not a promise

**`LivePredictor` never opens the checkpoint, and the net is not a member of
the DECISION TRAIL's soft vote.** The app's LIVE CLASSIFIER answer is
bit-identical with the net running, stopped, or never trained. Three
independent mechanisms enforce it, and `tests/test_nn.py` §1 checks all three
by reading the code (with comments and strings tokenised out, so a docstring
explaining the rule cannot be mistaken for a breach):

1. **Different process.** `nn_serve.py` holds the net; the recorder talks to
   it over stdin/stdout. That is primarily about §13's no-torch invariant, but
   it also means the net physically cannot reach into the classifier's state.
2. **Different type.** The engine yields `NeuralPrediction`, never
   `Prediction`. The two carry nearly the same fields, which is exactly why
   they are different classes: with one type, a net answer could be appended
   to `LivePredictor.history`, handed to the decision panel or drawn in the
   deployed bars by a single misplaced line, and nothing would complain.
3. **Different everything else.** Its own file, timer, smoothing deque,
   button and dialog. `_show_nn_prediction` writes only to the net's widgets
   and deliberately does **not** claim the graph overlay — two labels
   fighting over one corner of the plot is the exact blurring this section
   exists to prevent.

The panel does show **where the two disagree**, which is the point of having
two: "DISAGREES: the classifier says OPEN 71 %, the net says WAVE_RIGHT 64 %.
The app's answer is the classifier's." A single blended number would hide
precisely that.

### It really does use CUDA, and the recorder still never imports torch

`NN_DEVICE = "auto"` resolves in the worker. **Measured on this machine
(2026-10-19):** the worker comes up on `cuda:0`, GTX 1650, torch 2.9.1+cu128,
and one live window costs **24.3 ms round trip of which 7.7 ms is the forward
pass**, against the 150 ms `NN_INTERVAL_MS` budget. The panel prints that
measurement next to the budget rather than claiming it fits.

`tests/test_no_torch.py` still passes **11/11 with the net enabled** — with
`import torch` blocked outright, the application starts, loads the sklearn
model and predicts. That is the whole reason for the subprocess, and it is
the same trick as the GPU/CPU badge (§8c).

**Two device questions, two answers, never conflated.** The badge in LIVE
CLASSIFIER says what the *hardware* offers and that RETRAIN uses the CPU
regardless. The chip in the NEURAL NETWORK panel says what the *worker is
actually running on*, and reads `NN DEVICE --` until the worker reports —
never a guess in either direction.

### Which architecture, and what the sweep actually said

Five candidates, and the ranking is measured rather than asserted. The
default — first in the grid, and what `--quick` takes — is **`tcn`**: three
depthwise-separable dilated conv blocks (dilations 1/2/4, kernel 3) plus
attention pooling, **3 943 parameters**. The reasoning is that the input is
18–30 time steps and the dataset is ~1 700 windows from two wearers, so
reaching across the window should cost parameters logarithmically in its
length (dilation) rather than linearly in hidden units (recurrence), and
depthwise separation buys the same receptive field for a third of the weights.

**Attention pooling over time, not `out[:, -1, :]`.** The retired experiment
fed the last time step to the head, which asks the net to compress a gesture
into whatever state it holds at an arbitrary window boundary. `AttentionPool`
learns a weight per step and returns the weighted mean **concatenated with
the plain mean** — the second half keeping the activation level, because
removing the level is the mistake §13 records measuring twice.
`tests/test_nn.py` pins that the output really does change with amplitude.
`lstm` and `cnn` keep the old last-step / global-average heads on purpose:
they are the historical baseline and modernising them would destroy the
comparison.

**Full sweep, 85 recordings, two wearers, `affine` calibration,
subject-grouped, 433 s on the GTX 1650:**

| arch | window | per window | per gesture | macro F1 | params |
|---|---|---|---|---|---|
| **`cnn_gru`** (saved) | 0.50 s | **57.5 % ± 1.9** | 55.3 % | 47.6 % | 6 503 |
| `tcn` | 0.50 s | 54.3 % | **58.8 %** | 46.4 % | **3 943** |
| `cnn_gru` | 0.40 s | 55.8 % | 54.1 % | 45.7 % | 6 503 |
| `gru` | 0.50 s | 47.5 % | 50.6 % | 39.4 % | 4 455 |
| `lstm` | 0.30 s | 42.9 % | 44.7 % | 31.9 % | 14 022 |
| `cnn` | 0.50 s | 39.6 % | 34.1 % | 31.6 % | 18 246 |

Majority class 17.6 %. Per wearer held out: `Dano` 59.4 %, `Betka` 53.2 %.

So the prediction was half right: the conv front end matters (both top rows
have one, and both bare recurrent nets lose to it), but a **GRU on top of it
beat the pure dilated stack on windows** — 57.5 against 54.3 — while `tcn`
won the per-recording vote at 1.6× fewer parameters. The sweep saves the
window winner; `tcn` remains the default *candidate* because that is the
ordering, not the verdict.

### The headline finding: it loses to the LDA, again

| | per window | per gesture | macro F1 | input |
|---|---|---|---|---|
| **LDA-shrinkage (deployed)** | **86.8 %** | **87.1 %** | **83.2 %** | 154 features |
| NN-cnn_gru (separate) | 57.5 % | 55.3 % | 47.6 % | 30 × 8 raw samples |

Both subject-grouped on the same 85 recordings with the same `affine`
calibration, so they are comparable. **The net is 29 points worse than a
linear discriminant with no hyperparameters.** That is §8b's lesson repeating
itself with a better-specified experiment behind it, and it is the reason the
net is a diagnostic rather than a deployment: nothing here argues for shipping
a 200 MB torch dependency to an SBC.

**What is new is that the net is not merely losing — it is losing while
reading the right thing.** Its DC-only control (the identical net with every
window flattened to its eight channel means, so all time removed) scores
**47.3 % against the full input's 57.5 %, so the temporal structure is worth
+10.2 points**. The net genuinely uses how the signal changes in time; it is
just that eight channels of envelope at 50 Hz do not carry enough of it to
beat good hand-crafted features on a thousand windows.

**And the recording-grouped contrast is brutal: 95.6 % per window against
57.5 % subject-grouped — a 38.2-point gap, against the deployed model's
11.4.** Reading those two together: the net has plenty of capacity for this
problem and spends it memorising the wearer. That is the one concrete reason
to expect the net to become interesting later — the gap is what more wearers
would close, and it is much larger here than for the LDA.

### What is identical to `train_model.py`, and why that matters

`train_nn.py` imports `source_timeline`, `dedupe_windows`,
`subject_grouping_viable`, `resolve_grouping`, `load_recordings` and
`person_vector` **from `train_model`** rather than re-implementing them, so
the two trainers cannot drift into measuring different things. It therefore
inherits, for free: `MODEL_FS = 60 Hz` resampling of the
`label_start_index:label_end_index` span, window-level dedup (§8g — the
retired trainer had none, which is one reason its numbers were higher than
they should have been), subject-grouped fold selection (§8h), per-wearer
calibration (§8k), leave-one-subject-out, the majority-class baseline, and
the `PROGRESS`/`TRAINEVENT` lines — so **one GUI panel and one parser serve
both trainers**, the only difference being that this one's `device` field can
say `cuda`.

What is deliberately different: **no shared feature function.**
`model_feature_vector()` is the contract binding `train_model.py` to
`LivePredictor`; a net learns its own features and must not be mixed into that
path. The net's contract is `T.nn_input_block()` — resample, calibrate, cast
to float32, stop — and `nn_serve.py` calls the same function. Also fold-local
scaling (`log1p` + per-channel standardise, fitted on the training fold only,
shipped inside the checkpoint), and no early stopping, because stopping when
the held-out fold looks best is fitting the test set.

### The statistics sidecar, and why it is not the checkpoint

`train_nn.py` writes `models/gesture_nn.json` beside the `.pt`: everything
except the weights. **The application cannot read its own net's metrics out
of the checkpoint, because a `.pt` is a torch pickle and the recorder does not
import torch.** So NN STATS reads the sidecar, which is what lets the dialog
work with the net stopped, with no GPU, and on a machine where torch was never
installed. The running worker reports the same numbers back, and both come
from one write.

The sidecar also carries the same `data` provenance block both trainers emit,
so `model_data_status()` answers "is a net trained, on what, and is it
current?" for the net exactly as it does for the model (§8c).

### The refusals — a wrong answer is never preferred to no answer

Same discipline as §8e's panel and §8k's calibration gate:

- **no checkpoint** → START NN refuses and names `train_nn.py`;
- **a calibrated net, an uncalibrated wearer** → refused **before a
  subprocess is spawned**, naming CALIBRATE and `--calibration off`. There is
  no point attaching a GPU context to discover the wearer has not calibrated;
- **the worker disagrees with the sidecar about the calibration mode** → the
  **checkpoint wins** and the engine re-resolves. The weights are the fact;
  the sidecar is a convenience that can be stale;
- **a probability vector of the wrong width** → the whole engine is refused,
  not that one window, because the only other outcome is a confidently
  mislabelled answer;
- **a window the worker cannot use** → one `NNERR` with `fatal: false`; the
  engine keeps going. A transient must not cost a GPU context;
- **`strict=True` on `load_state_dict`** → a shape or name mismatch raises
  rather than leaving layers at their initialisation;
- **a worker that dies before reporting** → the engine goes to `error`
  carrying the worker's own last words (e.g. the torch ImportError), not a
  generic failure.

### Two real bugs the tests caught, both worth remembering

1. **The scaler shape bug, which only the inference path could hit.**
   `fold_scaler` originally returned `mean`/`std` with `keepdims=True`, shape
   `(1, 1, 8)`. That broadcasts correctly against the trainer's batched
   `(n, win, 8)` — so every training run passed — but turns a single live
   `(win, 8)` window into `(1, win, 8)`, which reaches `conv1d` as a 4-D
   tensor and raises. **No training run can find this.** Both sides now use a
   per-channel `(8,)` vector, and `nn_serve.py` flattens whatever it finds so
   earlier checkpoints still load.
2. **A refusal that resolved the wrong way round.** `_load_calibration()`
   always re-read the mode from the sidecar, so when `_on_meta` tried to let
   the worker's checkpoint override it, the sidecar's answer went straight
   back. The mode is now passed in explicitly.

### Rules this adds

- **The net never votes, and nothing may make it vote.** Not in
  `LivePredictor`, not in the decision panel's soft vote, not as a tiebreak.
  If a future version wants a two-model ensemble, that is a new deployed
  model with its own cross-validated number, not a quiet averaging of these
  two.
- **Quote the two models side by side, never a combined figure.** NN STATS →
  AGAINST THE DEPLOYED MODEL does exactly that, and *refuses to subtract*
  when the two were scored on different protocols.
- **A high DC-only score on the net means the same thing it means on the
  model.** The control is the identical net with time removed; read a small
  gain as "this did not need to be a sequence model" (§8f, §8k).
- **One trainer at a time.** RETRAIN and TRAIN NEURAL NET lock each other
  out, and START NN is disabled while the net trains: the worker holds a CUDA
  context on the card the trainer wants.
- **Re-measure the live cost after any architecture change.** The panel
  prints the round trip against `NN_INTERVAL_MS`, and §12 records what
  happened the last time something quietly outgrew its budget.

## 8m. Background training, ML + NN in one go, and TEST MODEL (2026-10-21)

### Training survives closing the window

Both trainers used to run as a `QProcess` owned by the window, so closing
the app killed them. They now run through **`train_job.py`**, started
**detached** (`start_training_job()`: no console, own process group, broken
away from the parent's Windows job object where allowed), and the GUI and the
job talk only through one folder per job:

```
models/jobs/<stamp>_<kind>_<id>/
    job.json      what to run: steps [{name, argv}] -- written by the GUI
    log.txt       every step's merged output + JOBSTEP / JOBSTEPEND lines
    status.json   state, runner pid, per-step exit codes -- written by the runner
    cancel        appears when the GUI asks it to stop (kills the step's TREE)
    seen          the GUI has reported the outcome
```

`TrainingJobWatcher` replaced the QProcess in `training_process` /
`nn_training_process`: it tails the log (`JobLogTail` hands out complete lines
only -- a half-written `TRAINEVENT` is held until its newline), feeds the same
two line handlers the QProcess pumps did, and each `JOBSTEPEND` calls the same
finish handlers. On close the app **asks**: keep training (default) or stop.
On startup `_resume_training_job()` reattaches to a live job and replays its
log from the start, and reports once any job that finished while no window
was open.

**Two Windows traps, both pinned by `tests/test_jobs.py`:**
- `os.kill(pid, 0)` on Windows is not a probe -- it calls TerminateProcess.
  `pid_alive()` asks the kernel (`OpenProcess` + `GetExitCodeProcess`).
- `os.replace` onto a file another process has open fails with *Access is
  denied* -- measured 1712 failures in 2000 writes against a polling reader.
  The GUI polls `status.json` three times a second, so the unretried runner
  crashed now and then and left `"running"` behind for ever. `write_status()`
  retries; the runner's `main()` also writes a terminal status on any crash;
  and the watcher treats a dead runner pid as finished.

### RETRAIN ML + NEURAL NET

One job, two steps, **strictly in sequence** (the one-trainer-at-a-time rule
holds inside the chain), and the net only starts if the classifier exited 0.
The classifier's completion box is suppressed in the chain; the net's box at
the end carries both results. The button is the chain's cancel button while
it runs, and so are RETRAIN (during step 1) and TRAIN NEURAL NET (step 2).

### TEST MODEL -- the number cross-validation does not measure

The CV headline is subject-grouped: a person the model never saw. Live use is
a wearer who usually IS in the training data, judged on a held gesture.
`ModelTestDialog` measures that directly: it cues every label `rounds` times
(interleaved, never the same twice in a row), predicts through
`latest_model_window()` -- the same window PREDICT uses -- skips the first
`MODEL_TEST_SETTLE_S` of each hold, resets the predictor's smoothing at each
hold, and reports three numbers, each with a matrix it can be recomputed from
(`confusion_metrics()`):

| | comparable with |
|---|---|
| per trial (majority of a hold's windows) | the trainer's per-recording vote |
| smoothed (what the panel shows) | what you experience |
| per window (raw) | the trainer's per-window score |

It **says whether the sitting is in the model's training data** (from the
payload's `data.sessions`), because a test on a trained-on sitting is
optimistic in exactly the §8f way: re-don the band and use a new Session id
for an honest number. Every result is appended to
`datasets/<person>/model_tests.jsonl` with the model it was measured on;
nothing is recorded, trained or saved otherwise.

## 8n. The ML + NN combination test, and what training calibrates with (2026-10-22)

`nn_ml_comb_test.py` (plain-text progress view: `--watch`; resumable:
`--resume DIR`; `--ml-only`; `COMB_ML_WORKERS=2` on this 8 GB machine, 5
workers ran it out of memory) searched 16 classifiers x 37 setups with every
soft vote scored from stored out-of-fold probabilities, plus a NESTED check
that redoes the whole choice without each person. Full results:
**`docs/COMB_TEST.md`** and `models/comb_test/20261022-003918/comb_test_results.xlsx`
(regenerate with `python tools/comb_test_report.py <run dir>`).

| | per window | nested | per recording |
|---|---|---|---|
| previous model | 56.5 % | -- | -- |
| **ML: vote(ET-500, RF-500-leaf3), bandpass/level 1.0 s** | 77.6 % | **74.0 %** | 83.7 % |
| best single (ET-500, same setup) | 77.4 % | -- | 83.3 % |
| **NN: gru 0.7 s** (search stopped after N1 by choice) | 72.4 % | -- | 78.9 % |

What it found: ML wants **longer windows** (1.0-1.3 s), **`level`** features
win every top setup, **forests take the whole top 15**, and voting adds only
0.2 points over the best single model. **Read the DC-only control:** at the
winning setup the 8 channel means alone score 77.8 % -- the model is reading
calibrated per-channel levels (§8f/§8k), not gesture shape. The NN's timing
is worth +9.2 points over its own DC-only control.

`model_spec.json` now carries a **`setup` pin** (`train_model.py` "PINNED
SETUP" block: the sweep collapses to that one setup) and `voting.always`, so
RETRAIN reproduces the winner rather than re-sweeping. Two trainer bugs this
exposed, fixed: a pinned `level` setup with `freq_features: false` was
skipped (the level family only ran with the flag on), and a spec with none of
the named probe models screened with nothing ("Nothing could be evaluated").
`nn_spec.json` / `NN_SPEC_DEFAULTS` hold `gru` @ 0.7 s, 150 epochs, the only
architecture selected, marked RECOMMENDED in the dialog.

**Training always calibrates with DERIVED calibrations**
(`TRAINING_CALIBRATION_SOURCE = "derived"`, `training_calibrations()`),
whatever `calibration.json` holds. A live REST+SQUEEZE record measures
*maximum* effort while the recordings are relaxed holds -- Dano's live and
derived references differ by **1.2x-3.1x per channel** (measured with the
record he had then; his current guided-sweep record of 2026-10-26 differs by
0.8x-2.2x, per person in `docs/DATASET_SUMMARY.md` section 5; Bruno and Miso
have no saved calibration at all) -- and a live record
describes how the band sits *today*, not when the data was recorded. Mixing
the two put wearers on different scales. The saved (live) calibration is
still what PREDICTING uses. **Open question, to measure with TEST MODEL:**
does a model trained on the derived scale predict well on a live
REST+SQUEEZE scale? If not, the live calibration should measure the way the
derived one does.

### Live speed, and a fix (2026-10-22)

`tools/bench_models.py`, 300 real windows, median: the ML vote takes
**53 ms** per live window (two 500-tree forests at ~25 ms each, features
1.7 ms, SVM/LDA under 1 ms); the NN round trip is **1.8 ms on CUDA**, 2.9 ms
on the CPU. **Before the fix the vote took 148 ms against the 150 ms
prediction interval**: the forests are saved with `n_jobs=-1`, which stood up
a joblib pool for every single-row predict. `LivePredictor.load()` now calls
`serialise_estimator()` (`emg_app/model_spec.py`, the same walk the DECISION
TRAIL panel always did) -- identical answers, a third of the time. **Any new
code path that predicts one live row on a saved model must do the same.**

**The spec in use is not the search's pick.** After the search the operator
added SVM and LDA to `model_spec.json` (commit `eb70845`); the 4-model vote
scores 69.5 % against the pair's 77.6 %, because an unweighted soft vote gives
13-14 points weaker members an equal say (docs/MODELS_COMPARED.md). Restoring
the pair is the operator's call -- the spec is data they edit in MODEL SPEC.

## 8o. Barbora, rotation augmentation, and training without a person (2026-10-27)

Full write-up, tables and graphs: **`docs/EXCLUDING_PEOPLE.md`**, generated by
`tools/make_exclusion_report.py` from the saved runs (`results/exclusion.json`).

- **Rotation augmentation does not fix her.** From the ablation runs, per held-out
  person: ML 45.9 -> 41.6 % on her; NN 36.7 -> 56.1 % but seeds 42-65 against
  30-49 and Miso -25 / Betka -13 (mean over five: ML -6.6, NN -4.2). With the band
  slid one pod her score barely moves (33 -> 37 %, 37 -> 39 %) while everybody
  else's rises by 11-33 points: **a one-pod roll is not what makes her hard**.
  The §8 "looks like a rotated band" reading is NOT supported -- do not repeat it.
- **What did move her is ML adaptation** (already the ML default): 46 -> 60 %.
- **Training without her is a setting, off by default.** Settings -> Training ->
  *Train without these people* (`cfg.TRAIN_EXCLUDE_PERSONS`, saved in
  `app_settings.json`). RETRAIN / TRAIN NEURAL NET pass `--person` for everyone
  who stays (`settings_store.person_args`); the zoo passes `--exclude-person`
  (`train_zoo.py`), which saves **`<id>-without-<name>`** entries beside the
  ordinary ones (`zoo.variant_id`, `meta.excluded_persons`) with their own eval
  folder `results/zoo_eval_without_<name>/`. `train_model.load_recordings(...,
  exclude_persons=)` is an explicit argument, never read from config, so the
  research tools keep measuring the whole dataset.
- **Variants are never ranked or given a yellow star** (their score is over fewer,
  easier people; `compute_stars` skips them; the Models tab lists them after the
  ordinary models). Never compare a variant's headline with an ordinary model's:
  compare per person, same held-out person, with and without
  (`scores.base_per_person`). A variant earns a **blue star** (`zoo.variant_stars`)
  when it scores HIGHER than its yellow star on the same held-out people -- any
  gain, no margin (`zoo.BLUE_STAR_MIN_GAIN` = 0), computed live from the entries'
  scores every time the zoo is listed, never stored; yellow stays the default.
  The reference is the yellow star the variant was made from (else the family's
  best yellow star). `collect_results.variants_block` exports them separately
  from the ranked `models`.
- **The training-switch ablation was re-run without her** (`tools/ablation_ml.py`
  / `ablation_nn.py --exclude-person Barbora`, stage A only, results in
  `results/ablation_without_barbora/`): each switch's effect on the four people who
  stay, with and without her in training, tables + figure + CSVs in
  `docs/EXCLUDING_PEOPLE.md` (generated by `tools/make_exclusion_report.py`, which
  also fills the marked block in `docs/MODELS_COMPARED.md`). Note the judging rule
  needs 4 people better, which with four left means all four: compare the points,
  not only the verdict.
- **GUI:** the Settings tabs scroll (`ContainedTable`: each model list keeps the
  mouse wheel when it can scroll, and is never disabled -- a disabled widget gets no
  wheel events, which used to move the whole page).
- **Result:** the six starred models were trained without Barbora. For the other
  four people the change is **ML +0.9, NN +0.9 points on average** -- mixed per
  person (-5.7 .. +9.5), not separable from noise. She herself reads ~60 % (ML) /
  ~39 % (NN) from a model that never saw her. Not a clear win, hence off.

## 9. Safety / data-preservation rules

These are invariants. Breaking one is a bug, not a trade-off.

1. **Raw is written once and never rewritten.** `*_raw.*` files are immutable.
2. **Nothing is overwritten.** Every write goes through `unique_path()`.
3. **Trimming/processing creates new files** with a new stage suffix.
4. **Deleting moves files to `_deleted/`**; nothing in this project calls
   `os.remove` on recorded signal. DELETE LAST, REDO and DELETE TAKE(S) all
   go through a folder move (`DELETED_DIRNAME`), and each logs a
   `stage: "deleted"` manifest line saying where the files went.
5. **Bad recordings are flagged, never discarded.** `check_quality` returns
   findings; the operator decides.
6. **Samples are captured in the acquisition thread**, not the GUI thread, so
   a slow frame cannot drop a sample.
7. **Every file gets a config snapshot** (`recording_config_snapshot()`), so
   sessions recorded weeks apart stay comparable.
8. **Annotating never edits data.** A note on a saved recording goes to a
   `*_notes.json` sidecar; the marker log is append-only. Nothing in the
   markers/notes path opens a signal file for writing.
9. **A free take is never training data, and cutting from it never alters
   it.** Takes live in `_takes/` with stage `take` -- pruned by discovery and
   asked for by no stage selector. EXTRACT CLIP copies samples out; it does
   not rewrite, rename or delete the take. DELETE TAKE(S) is the only thing
   that moves a take, and it moves the take alone -- never the clips cut
   from it, which are independent recordings.
10. **No fake predictions.** The classifier panel shows output only from a
    real model's `predict()`.
11. **Selecting a model never flatters it.** The model spec decides what gets
    TRAINED, never how it is scored: every candidate -- hand-written, voted or
    AUTO-generated -- goes through the same cross-validation. Where selection
    bias exists (AUTO mode), it is stated in the dialog, the training log and
    MODEL STATS rather than hidden.
12. **Calibration never rewrites signal, and a bad one is refused.** A
    calibration is two level vectors per wearer, applied at FEATURE time in
    `model_feature_vector()`; nothing on disk is scaled, so there is no
    calibrated copy of the dataset to keep in sync and rules 1 and 3 are
    untouched. `validate_calibration()` refuses rather than coerces -- a
    clamped hyperparameter still trains something, a clamped gain divides a
    wearer's signal by noise -- and an unusable record resolves to None,
    which everywhere means "behave as the uncalibrated pipeline". Clearing
    one writes the full record to the append-only `calibration_log.jsonl`
    before the file goes, so the measurement is never actually lost. §8k.
13. **The neural net never contributes to the answer.** It is a separate
    checkpoint, loaded by a separate process, yielding a separate TYPE
    (`NeuralPrediction`, not `Prediction`). `LivePredictor` never opens it,
    the DECISION TRAIL's soft vote has no neural member, and the LIVE
    CLASSIFIER prediction is bit-identical whether the net is running,
    stopped or has never been trained. Where the two models are shown
    together they are two independent cross-validated results with their
    protocols named — never one blended figure, and never a subtraction
    across different protocols. §8L.
14. **Switching a gesture off in the classifier never retrains, and never
    touches a model.** It is a mask on the output (`class_mask.py`), applied
    identically to the ML model and the net, saved with the gesture set, and
    undone by ticking the box. It must not rewrite `gesture_model.joblib` /
    `gesture_nn.pt`, must not restart the NN worker, and must say it is a mask
    wherever it shows an answer. A mask that leaves nothing to answer with is
    reported as "no answer", never as a guess. §7a.

15. **Choosing models never alters a model, and a vote is never forced.**
    Selecting zoo models, turning voting or adaptation on or off, and the two
    questions change what the application LOADS and how it standardises input;
    they never rewrite a model file, never touch `gesture_model.joblib` /
    `gesture_nn.pt`, and never start a training by themselves. Models that cannot
    be averaged are refused with the reason, the two families are never voted
    together, and a model is saved to the zoo only with scores measured on the
    dataset now on disk. Stars are computed from those scores, not stored. §7c.

16. **Leaving a person out of training never touches data or an existing model.**
    `TRAIN_EXCLUDE_PERSONS` only narrows what the NEXT training reads; nothing on
    disk is moved, an ordinary model is never overwritten (the zoo writes separate
    `-without-` entries), a variant is never ranked against ordinary models, and at
    least two people must stay in. §8o.

## 10. How to run

```powershell
python -m pip install -r requirements.txt     # once
python emg_dataset_tool.py                    # the application
python emg_dataset_tool.py --summary          # dataset stats, no GUI
python legacy/test_myo.py                            # original reference monitor

python training/train_model.py --spec models\model_spec.json   # THE trainer (CPU)
python training/train_model.py --quick --spec models\model_spec.json   # reduced grid
python training/train_model.py --no-panel                      # skip the DECISION TRAIL file
python training/train_model.py --calibration affine            # per-wearer calibration on (§8k)
python training/explain_votes.py                               # rebuild it for the saved model
python training/explain_votes.py --json                        # ...and a JSON copy of the trail

python training/train_nn.py                      # THE NEURAL NET (CUDA) -- separate model, §8L
python training/train_nn.py --quick              # the default architecture, one window
python training/train_nn.py --device cpu         # no GPU
python training/train_nn.py --kind tcn --window 0.5          # restrict the grid
python training/train_nn.py --calibration off    # reproduce the uncalibrated numbers
python training/nn_serve.py --model models\gesture_nn.pt --device auto   # the worker, by hand

python tests\test_datalayer.py                # 150 checks, no hardware
python tests\test_gui.py                      # 109 checks, no hardware
python tests\test_review.py                   # 50 checks, no hardware
python tests\test_label_on_graph.py              # 48 checks, no armband needed
python tests\test_markers.py                  # 88 checks, no hardware
python tests\test_free_take.py                # 268 checks, no hardware
python tests\test_model_spec.py               # 346 checks, no hardware
python tests\test_predict.py                  # 109 checks, no hardware
python tests\test_trail.py                    # 103 checks, the DECISION TRAIL
python tests\test_calibration.py              # 161 checks, per-wearer calibration (§8k)
python tests\test_jobs.py                     # 48 checks, background training + TEST MODEL (§8m)
python tests\test_custom_gesture.py           # adding a gesture reaches everything, never auto-retrains (§7a; ~5 min, trains on a temp copy)
python tests\test_nn.py                       # 162 checks, the neural net and the SEPARATION rule (§8L)
python tests\test_class_mask.py               # 121 checks, switching a gesture off in the classifiers, no retrain (§7a)
python tests\test_zoo.py                     # 117 checks, the zoo, voting, adaptation, augmentation, the settings store, the questions' logic (§7c)
python tests\test_app_settings.py            # 97 checks, the settings tabs, applying/restoring, the two questions on the real window (§7c)
python tools\run_tests.py [--fast]            # every offline suite, one line each; writes tests/last_run.json
python tests\test_device.py                # 104 checks, device profile / channel count (§2b), gestures (§7a)
python tests\test_sweep_cost.py               # 48 checks, the sweep cost controls (§8g)
python tests\test_filters.py                  # 52 checks, the band-pass (§8i) + FEATURE SPACE (§8j)
python tests\test_no_torch.py                 # 11 checks, proves the app needs no torch
python tests\test_torch.py                    # 163 checks, the RETIRED deep trainer
python tests\test_live.py                     # 35 checks, NEEDS the armband
python tests\test_live_gui.py                 # 25 checks, NEEDS the armband
python tests\test_live_review.py              # 29 checks, NEEDS the armband
```

Only one process can hold `COM3`. Close `test_myo.py` before starting the tool
or the live tests.

## 11. Dependencies

Required: `pyserial`, `numpy`, `PySide6`, `pyqtgraph`.
`torch` is also what `train_nn.py`, `nn_serve.py`, `nn_arch.py` and
`tests/test_nn.py` need (§8L) — still only in subprocesses, still never in
the recorder.
Recommended: `pandas` (DataFrames/parquet), `scipy`.
Optional: `scikit-learn` + `joblib` (training, live classifier),
`pyarrow` (only if `SAVE_PARQUET = True`),
`torch` (only `train_torch.py` and `tests/test_torch.py`).

`torch` is deliberately **not** installable from `requirements.txt`: pip's
default index serves the CPU-only build, which `train_nn.py` refuses rather
than silently running twenty times slower. Install it explicitly:

```powershell
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The recorder never imports torch. It asks a subprocess whether a GPU exists
(§8b) and runs the neural net in another one (§8L), so the application
starts, records and predicts exactly the same with torch absent —
`tests/test_no_torch.py` proves it with the net enabled.

The data layer degrades gracefully: without pandas you lose
`to_dataframe()`/parquet, nothing else.

## 12. Known issues

- **`EMG_MODE = "raw"` is the less-travelled path.** The filtered mode is the
  one the original working code uses and is the default. Raw mode has been
  implemented and reviewed but not yet validated against a live armband over a
  long session. Verify sample rate and amplitudes before collecting a real
  dataset with it.
- **PyQtGraph 0.14 `SVGExporter` crashes** on these plots
  (`ValueError: not enough values to unpack` in `correctCoordinates`). SVG
  export therefore uses Qt's `QSvgGenerator` directly. Do not "simplify" it
  back to the pyqtgraph exporter.
- **Filtered mode runs at roughly 50 Hz**, which is thin for fast gestures.
  `NOMINAL_SAMPLE_RATE` is only a hint — the measured rate is what gets stored.
- **The armband sleeps** when idle. `STALL_TIMEOUT_S` catches that and triggers
  a reconnect; the first reconnect after a sleep can take a few seconds.
- **The DECISION TRAIL now costs MORE than the prediction interval: 161 ms
  of GUI thread per live window against the 150 ms interval.** It was
  50-78 ms (§8e). `tests/test_trail.py` fails on exactly this check
  (`a live window costs well under the prediction interval -- 161 ms vs
  150 ms`), and that failure is the budget being exceeded, not a bad test.
  **The cause is the model spec, not the dataset:** enabling `kNN-5` and
  `kNN-15` put two members in the panel whose prediction is a distance to all
  3573 training rows, and `RandomForest-800-deep` added 800 more trees. So
  with MODEL STATS open the trail misses windows. No samples are lost --
  capture is on the acquisition thread -- but the view is no longer
  keeping up. Close the dialog, set `ENABLE_DECISION_PANEL = False`, or
  disable the kNN candidates in the spec. Re-measure after any spec change
  that adds members.
- **`gesture_panel.joblib` tracks the spec, not the dataset, and the spec can
  swing it by 6×.** It is a desktop diagnostic — do not copy it to the robot.
  Measured: 12 MB at the 44-recording model, 49 MB at the 325-recording
  bandpass model, **57 MB** at the 85-recording two-subject model with
  `kNN-5`/`kNN-15` enabled (kNN stores the training matrix outright), and
  **9.5 MB** at the current calibrated model, whose spec is six models and a
  soft vote with one forest. The 100 MB GitHub hard limit is no longer close,
  but it is one spec edit away — gitignoring this file and the timestamped
  archives is still an open decision. The saved model itself is **209 KB**.
- **AUTO-TRIM ALL destroys a relaxed hold. Do not run it on clips cut from a
  take.** `detect_activity` picks the *strongest* region by design (§13, "a
  150 ms explosive burst must beat a second of drift"), and in the recording
  protocol the relaxed hold is the **weakest** part of the clip. Measured on
  the 17 relaxed-hold clips of 2026-10-15: it kept a median of 70.8 % of the
  clip, worst case 42.4 %, and in **16 of 17** clips the part it discarded was
  weaker than the part it kept — ratios 0.28–0.65, at absolute amplitudes of
  50–90, which is precisely the 43–171 void the re-record exists to fill. The
  function is behaving exactly as specified and is exactly wrong for this job.
  It is also unnecessary: a clip cut from a take is written with
  `label_start_index = 0, label_end_index = n_samples`, and `make_windows()`
  trains on that span, so **the operator's selection already is the trim.**
- **The dataset cannot currently be validated honestly** — every active class
  is a single free take, so a take-grouped split has no training data for the
  class it is testing, and one subject means `leave_one_person_out()` cannot
  split either. 74 of 308 recordings are also byte-identical duplicates that
  uid-grouped CV does not separate. §8f and `docs/EDA.md`.
- **A NEW WEARER MUST CALIBRATE BEFORE THE CLASSIFIER WILL RUN FOR THEM.**
  This is the cost of `CALIBRATION_MODE = "affine"` (§8k) and it is the thing
  most likely to look like a bug. The saved model was fitted on calibrated
  features, so for a wearer with no calibration `LivePredictor.load()`
  refuses — "this model was trained on 'affine' per-wearer calibration, and
  <person> has none. Press CALIBRATE, or retrain with calibration off" — and
  the LIVE CLASSIFIER shows that instead of a prediction. It refuses because
  it *would* otherwise answer, at 7 Hz, with every amplitude feature off by
  that wearer's gain. Press CALIBRATE (six seconds), or set the mode to `off`
  and retrain. The same refusal appears if the mode is changed without
  retraining; the panel reloads the model the moment the mode or the wearer
  changes, so the message arrives then rather than at the next PREDICT.
- **The neural net is 5 points behind the ML vote** (2026-10-22, five
  wearers: GRU 72.4 % against the ET+RF vote's 77.6 % per window, both with a
  person held out; docs/MODELS_COMPARED.md). On two wearers it was 29 points
  behind (57.5 % against 86.8 %, §8L) -- more people closed most of the gap,
  as §8L's 38-point recording-vs-subject gap predicted. It is still a second
  opinion and never part of the answer (rule 13); it is ~300x smaller and
  ~30x faster per window, and its timing is worth +9.2 points over its own
  DC-only control, where the ML vote's features add nothing over channel
  means.
- **START NN takes a few seconds**, and the panel says `STARTING` while it
  does. The worker has to import torch (~7 s cold) and build a CUDA context
  before it can answer; that cost is exactly why it is a subprocess. If it
  has not reported within `NN_START_TIMEOUT_S` (40 s) the panel gives up with
  the worker's own last words.
- **`MAX_RECORDINGS = 0`** means unlimited, not zero.
- **The `dropped` counter is approximate.** The dongle delivers 2–4 buffered
  packets at once, so the inter-sample interval is bursty and the heuristic
  over-counts slightly. A handful over thousands of samples is normal.
- **Sample rate must be measured as a span average**, not as
  `1/median(diff(t))` — burst delivery made the median estimate report
  9258 Hz for a 57 Hz stream. See `measured_rate()`.
- **An armband that is not being worn stops sending EMG** after a few seconds.
  The watchdog treats that as a dropout and reconnects, which is correct, but
  it means bench-testing without wearing the band produces dropout warnings.
- **The GPU is small.** 4 GB on a GTX 1650, and `train_torch.py` moves the
  whole fold onto the device in one tensor rather than streaming batches.
  That is fine at the current scale (a few thousand windows of 30x8 floats)
  and will not be once the dataset is twenty times bigger; the fix then is a
  `DataLoader`, not a smaller model.
- **The torch sweep is grid x folds x epochs.** Three architectures x three
  windows x six folds at 60 epochs took ~9 minutes on this dataset. Use
  `--quick` while changing the code and the full grid only for a real run.
- **The CUDA path is single-GPU, device 0.** `resolve_device()` does not look
  for a second card.
- Windows only, in practice: the COM-port handling and `detect_tty()` have not
  been tried elsewhere in this project.
- **Windows display scaling shrinks the screen Qt lays out on.** Qt 6 works
  in logical pixels, so a 1920x1080 panel is 1536x826 at 125 % and 1280x688
  at 150 %. Checked 2026-10-21 at 100/125/150/175 % with `QT_SCALE_FACTOR`:
  the calibration dialog (950 px of content) and MODEL SPEC (~700 px) used to
  run off the bottom; both now go through `scroll_dialog_body()` with the
  button row pinned, and every window's minimum fits at 175 %. Tables are
  sized by `fit_table_height()` from their real row height, not 24 px; the
  banner drops to 64 px on a screen under 800 logical px; and the plot lanes
  no longer add pyqtgraph's "(x0.001)" to the channel name, which overlapped
  at ~40 px per lane. **Re-check with `QT_SCALE_FACTOR=1.5` after adding a
  tall dialog** — a layout minimum taller than the screen is something
  `fit_to_screen` cannot shrink.

## 13. Design decisions

- **One file.** Requested. Kept maintainable via strict section ordering and
  by keeping Sections 1–6 free of Qt.
- **Ring buffer for display, separate capture list for recording.** Dropping a
  display sample is cosmetic; dropping a recorded sample is data loss. The two
  paths are deliberately not shared.
- **GUI pulls, acquisition pushes.** Per-sample Qt signals at 200 Hz would
  flood the event loop, so the acquisition thread only signals *state* and the
  GUI reads samples on its own timer.
- **Wall-clock state machine.** `SequenceController` compares against
  deadlines rather than counting ticks, so a dropped frame delays nothing.
- **Monotonic timestamps pinned to one wall-clock reading** — absolute and
  immune to clock adjustments mid-session.
- **Median, not mean, for baselines.** A subject who starts the gesture early
  would otherwise poison the baseline estimate.
- **Autoscale grows instantly, shrinks slowly.** A peak must never be clipped
  out of view; the asymmetry also stops the axis from jittering.
- **Peak-mode downsampling in the plot.** Preserves spikes that mean-mode
  downsampling would smooth away — the whole point for explosive gestures.
- **Strongest region wins in `detect_activity`**, not the longest: a 150 ms
  explosive burst must beat a second of drift.
- **Interleaved gesture order by default.** Spreads fatigue and electrode
  drift evenly across the label set instead of baking it into the last gesture.
- **Identifying columns repeated in every CSV row.** Redundant, but it makes a
  single CSV self-describing and `pd.concat`-able without care.
- **Two feature families, and `level` is the default.** The `filtered` stream
  is an already-rectified envelope. `extract_features()` median-centres each
  channel, which is right for raw bipolar EMG and deletes the activation level
  here — measured Fisher ratio 25.5 (level) vs 15.1 (centred), and switching
  took accuracy from 33.5 % to 43.2 % on half the features. Use
  `extract_level_features()` for `filtered` and `extract_features()` for
  `raw`. Details in `docs/ML_PIPELINE.md` §3a.
- **One feature function shared by training and inference.**
  `model_feature_vector()` is called by both `train_model.py` and
  `LivePredictor`. If those drifted apart by one feature, live predictions
  would be garbage while the cross-validation score stayed perfect.
- **Everything is resampled to `MODEL_FS = 60 Hz` for the model.** The stream
  is 28–70 Hz and unevenly sampled, so a fixed sample count would mean a
  varying real duration. `resample_uniform()` is applied in training and in
  `LivePredictor` alike.
- **Absolute amplitude is kept per recording, and removed per WEARER.**
  Baseline subtraction *per recording* was tried and measurably hurt
  (Fisher 24.9 → 11.5–17.2): within a session the offsets are constant, and a
  per-recording estimate partly cancels the gesture it is meant to baseline.
  A **per-wearer** calibration is the opposite case and it measurably helps
  (+5.3 points subject-grouped with the calibration itself held out, §8k):
  one constant measured once from a
  dedicated rest capture, which never sees a gesture and so cannot cancel
  one. The distinction between those two is the whole content of §8k, and
  getting it backwards would undo either result.
- **Calibration is a deployment decision, not a sweep axis.** The sweep
  chooses the representation because that choice is invisible to whoever uses
  the model. A calibrated model obliges every future wearer to calibrate
  before they can use it at all, so the mode is a flag the operator sets and
  the two runs are compared by hand. §8k.
- **Retraining always uses the whole dataset.** `train_model.py` reads the
  entire tree, so the GUI's RETRAIN button adds new recordings to all earlier
  ones. There is deliberately no "train on the latest batch only" path.
- **The GUI passes `--root`, `--out-dir` and `--model-path` to the trainer**
  rather than letting it fall back to its own defaults. Without that, a
  changed `DATA_FOLDER` or `MODEL_PATH` meant the GUI retrained on, and
  overwrote, a different dataset and model than the one it was using — which
  is exactly what happened in testing.
- **Training runs as a subprocess, not a thread.** scikit-learn with
  `n_jobs=-1` starves the acquisition thread (observed: it made the test suite
  time out), and a training crash must not take the recorder down.
- **Three separate fixes keep the link up**, see `MYO_NEVER_SLEEP`,
  `MYO_CONN_TIMEOUT_MS` and `MYO_FAST_RECONNECT`. The armband sleeps on its
  own unless told not to; the original 640 ms supervision timeout is too
  short; and re-scanning on reconnect fails once the armband stops
  advertising. `myo_raw.py` keeps its original defaults so `test_myo.py` and
  `myo.py` are unaffected.
- **The classifier panel cannot fabricate output.** Every number comes from a
  real `predict_proba()`. A model whose feature count disagrees with its
  declared `feature_set` is rejected at load time with an explanation rather
  than producing nonsense at 7 Hz.
- **Two models, never one blended number.** The neural net is a second
  opinion with its own button, not a member of an ensemble (§8L). Averaging
  two models whose accuracies differ by 29 points would make the worse one
  invisible and the better one worse; showing them side by side with their
  protocols named makes a disagreement informative. The separation is
  enforced by a different process, a different TYPE and different widgets
  rather than by discipline — safety rule 13.
- **No torch in the recorder, and the GPU question answered honestly.**
  `train_torch.py` is a separate process with a separate output file; the app
  learns whether CUDA exists by running `COMPUTE_PROBE_SNIPPET` in a
  subprocess through `QProcess`, never by importing torch itself (the import
  costs ~7 s and attaches a GPU context to the process that must not drop
  samples). Until the probe answers, the badge says `checking...` rather than
  guessing — and because the scikit-learn sweep has no GPU backend, both the
  badge tooltip and the RETRAIN dialog say in words that RETRAIN runs on the
  CPU even when the badge is green.

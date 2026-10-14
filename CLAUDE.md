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
| Device | Thalmic Myo armband, 8 dry EMG pods |
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

`common.py`, `myo.py` (the old k-NN demo) and `test_myo.py` are untouched and
still run.

## 4. Important files

| File | Role |
|---|---|
| **`emg_dataset_tool.py`** | The whole application. Single file by design, internally split into numbered sections so it can be broken into modules later. |
| **`train_model.py`** | Two-stage model sweep + honest validation + saves `models/gesture_model.joblib`. Imports the data layer from `emg_dataset_tool`. Builds its candidates from `models/model_spec.json` when the GUI passes `--spec`. |
| **`explain_votes.py`** | Rebuilds `models/gesture_panel.joblib` for the model that is *already* saved, without re-running the sweep. For a model trained before the panel existed, or a panel that was deleted. `--json` also writes `models/reports/decisions.json`. See §8e. |
| `train_torch.py` | **RETIRED.** The CUDA neural-net experiment (1D-CNN / LSTM / CNN-LSTM). Kept for the record, not part of the workflow: it lost to a plain LDA. Nothing in the app reads its output. See §8b. |
| `myo_raw.py` | Myo BLE protocol wrapper. **The working communication layer.** |
| `common.py` | `pack`/`unpack` helpers used by `myo_raw.py`. |
| `myo.py` | Original k-NN pose demo. Not part of the tool; kept for reference. |
| `test_myo.py` | Original matplotlib live monitor. Kept as the reference for "the connection definitely works like this". |
| `GUIDE.md` | **User-facing manual**: what every file and setting does, and how to set the tool up for different jobs. |
| `docs/DATASET.md` | On-disk format, filenames, CSV/NPZ schema. |
| `docs/ARCHITECTURE.md` | Threads, buffers, state machine, data flow. |
| `docs/ML_PIPELINE.md` | From recordings to a validated model. |
| `requirements.txt` | Dependencies. |
| `tests/` | Fourteen runnable check scripts (1443 checks total, three need the armband). See `tests/README.md`. |
| `datasets/` | Recorded data (created on first run). |
| `exports/` | Combined ML bundles from the EXPORT button. |
| `models/` | `gesture_model.joblib` (loaded by the app), `gesture_panel.joblib` (the DECISION TRAIL side file, §8e), timestamped archives, and `reports/` with every sweep's full results. |

### Section map of `emg_dataset_tool.py`

```
USER CONFIGURATION          all tunables, top of file
SECTION 1  helpers          paths, slugs, unique_path, time formatting
SECTION 2  signal           moving RMS, envelope, normalisation,
                            detect_activity, extract_features, windowing,
                            augmentation
SECTION 3  quality          check_quality + QualityIssue
SECTION 4  dataset          Recording, DatasetWriter, loaders, make_windows,
                            leave_one_person_out, dataset_summary,
                            Marker + marker log + notes sidecar,
                            free takes (save_take / list_takes)
SECTION 5  acquisition      SampleRing, Capture, MyoAcquisition  (no Qt)
SECTION 6  sequence         build_plan, SequenceController         (no Qt)
SECTION 7  GUI              theme, EmgPlotWidget, StatusBanner,
                            QualityDialog, TrimDialog, MainWindow
SECTION 8  entry point      main()
```

Sections 1–6 are Qt-free and importable headless — that is what lets a
training script do `import emg_dataset_tool as T; T.load_dataset(...)` on a
machine with no display.

## 5. Data format

```
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
```

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
python train_model.py            # saves models/gesture_model.joblib
```

Then the app's LIVE CLASSIFIER panel works: one-shot or continuous mode, the
prediction drawn over the graph, per-class probability bars, and a MODEL STATS
dialog showing the cross-validated numbers with their protocol.

**Current state of the model (2026-10-06):** 72 recordings, one subject
(`Dano`), one session. Best so far ≈ 43 % per window / 47 % per gesture
against a 16.7 % chance level. The dataset is the bottleneck, not the model:
15 of 72 recordings are flagged `no_activity` (5 of 12 FIST), and FIST's mean
envelope varies 26-fold across its own repetitions. More and stronger
repetitions are worth far more than further model tuning.

## 8b. The CUDA path (`train_torch.py`) — RETIRED 2026-10-14

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
python train_torch.py                     # CUDA, full grid
python train_torch.py --quick             # one model, one window
python train_torch.py --device cpu        # no GPU
python train_torch.py --epochs 80 --kind cnn_lstm --window 0.5
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
| members at `n_jobs=1` | **50-78 ms** per window | `DecisionPanel._serialise_members()` forces this on load |
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

## 9. Safety / data-preservation rules

These are invariants. Breaking one is a bug, not a trade-off.

1. **Raw is written once and never rewritten.** `*_raw.*` files are immutable.
2. **Nothing is overwritten.** Every write goes through `unique_path()`.
3. **Trimming/processing creates new files** with a new stage suffix.
4. **DELETE LAST moves files to `_deleted/`**; it never calls `os.remove`.
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
   not rewrite, rename or delete the take.
10. **No fake predictions.** The classifier panel shows output only from a
    real model's `predict()`.
11. **Selecting a model never flatters it.** The model spec decides what gets
    TRAINED, never how it is scored: every candidate -- hand-written, voted or
    AUTO-generated -- goes through the same cross-validation. Where selection
    bias exists (AUTO mode), it is stated in the dialog, the training log and
    MODEL STATS rather than hidden.

## 10. How to run

```powershell
python -m pip install -r requirements.txt     # once
python emg_dataset_tool.py                    # the application
python emg_dataset_tool.py --summary          # dataset stats, no GUI
python test_myo.py                            # original reference monitor

python train_model.py --spec models\model_spec.json   # THE trainer (CPU)
python train_model.py --quick --spec models\model_spec.json   # reduced grid
python train_model.py --no-panel                      # skip the DECISION TRAIL file
python explain_votes.py                               # rebuild it for the saved model
python explain_votes.py --json                        # ...and a JSON copy of the trail

python tests\test_datalayer.py                # 150 checks, no hardware
python tests\test_gui.py                      # 91 checks, no hardware
python tests\test_review.py                   # 50 checks, no hardware
python tests\test_label_on_graph.py              # 48 checks, no armband needed
python tests\test_markers.py                  # 88 checks, no hardware
python tests\test_free_take.py                # 195 checks, no hardware
python tests\test_model_spec.py               # 346 checks, no hardware
python tests\test_predict.py                  # 109 checks, no hardware
python tests\test_trail.py                    # 103 checks, the DECISION TRAIL
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
Recommended: `pandas` (DataFrames/parquet), `scipy`.
Optional: `scikit-learn` + `joblib` (training, live classifier),
`pyarrow` (only if `SAVE_PARQUET = True`),
`torch` (only `train_torch.py` and `tests/test_torch.py`).

`torch` is deliberately **not** installable from `requirements.txt`: pip's
default index serves the CPU-only build, which `train_torch.py` refuses
rather than silently running twenty times slower. Install it explicitly:

```powershell
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The recorder never imports torch. It asks a subprocess whether a GPU exists
(§8b), so the application starts, records and predicts exactly the same with
torch absent.

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
- **The DECISION TRAIL costs 50-78 ms of GUI thread per live window**
  (§8e), against a 150 ms prediction interval. The plot can look slightly
  less smooth while MODEL STATS is open; no samples are lost, because
  capture happens on the acquisition thread. Close the dialog, or set
  `ENABLE_DECISION_PANEL = False`, on a slow machine.
- **`gesture_panel.joblib` is ~4 MB** and grows with the forests in the
  spec. It is a desktop diagnostic — do not copy it to the robot.
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
- **Absolute amplitude is kept, not baseline-subtracted.** Baseline
  subtraction was tried and measurably hurt (Fisher 24.9 → 11.5–17.2): within
  a session the offsets are constant, and a per-recording estimate partly
  cancels the gesture. Offset/gain invariance belongs in the future per-user
  calibration, not here.
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
- **No torch in the recorder, and the GPU question answered honestly.**
  `train_torch.py` is a separate process with a separate output file; the app
  learns whether CUDA exists by running `COMPUTE_PROBE_SNIPPET` in a
  subprocess through `QProcess`, never by importing torch itself (the import
  costs ~7 s and attaches a GPU context to the process that must not drop
  samples). Until the probe answers, the badge says `checking...` rather than
  guessing — and because the scikit-learn sweep has no GPU backend, both the
  badge tooltip and the RETRAIN dialog say in words that RETRAIN runs on the
  CPU even when the badge is green.

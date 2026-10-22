# emg

Labeled 8-channel EMG gesture recording tool for the Thalmic Myo armband, plus
the training and validation pipeline that goes with it.

**The deliverable is the recording tool, not a model.** The point is a clean,
honestly-labelled dataset with subject-independent validation — because on
this problem the dataset is the bottleneck and the classifier is not.

![The main window](docs/images/ui_main.png)

---

## Contents

- [Install and run](#install-and-run)
- [What it does](#what-it-does)
- [Recording workflows](#recording-workflows)
- [From recordings to a model](#from-recordings-to-a-model)
- [The honesty machinery](#the-honesty-machinery)
- [Safety rules](#safety-rules)
- [Where things live](#where-things-live)
- [Documentation](#documentation)
- [Tests](#tests)
- [Current state](#current-state)

---

## Install and run

```powershell
python -m pip install -r requirements.txt     # once

python emg_dataset_tool.py                    # the application
python emg_dataset_tool.py --summary          # dataset stats, no GUI

python training/train_model.py --spec models\model_spec.json    # THE trainer (CPU)
python training/train_model.py --quick --spec models\model_spec.json
python training/explain_votes.py                       # rebuild the DECISION TRAIL file
```

Required: `pyserial`, `numpy`, `PySide6`, `pyqtgraph`.
Optional: `scikit-learn` + `joblib` (training and the live classifier),
`pandas`, `scipy`, `pyarrow`.

Needs a Myo armband and a Bluegiga BLED112 dongle. Windows, in practice — the
COM-port handling has not been tried elsewhere. Only one process can hold the
port, so close `legacy/test_myo.py` before starting the tool.

**The recorder never imports torch.** It asks a subprocess whether a GPU exists, and runs the neural net in another one, so the app starts, records and predicts identically with torch absent.

---

## What it does

```
BLED112 dongle on COM3
        │  pyserial
   myo_raw.BT                      BLE/BGAPI packet framing
        │
   myo_raw.MyoRaw                  Myo protocol, set_mode, notifications
        │  add_emg_handler(fn)
   MyoAcquisition                  background thread, ring buffer, capture
        │  Qt signals (state only) + ring buffer reads
   MainWindow                      the GUI
```

Eight dry EMG pods at ~50 Hz (`filtered` mode, rectified on-device) or 200 Hz
(`raw` mode). Live plot, labelled capture, quality checks, trimming,
annotation, export, a scikit-learn model sweep, and a live classifier with its
own diagnostics.

**Everything in `emg_app/` outside `gui/` is Qt-free** and importable headless,
which is what lets a training script do `import emg_dataset_tool as T;
T.load_dataset(...)` on a machine with no display.

---

## Recording workflows

Three independent paths. **The free-take route is the one in use.**

### Free take — record now, cut later

```
START FREE TAKE (Ctrl+Shift+R)   one continuous capture, minutes long
STOP & SAVE TAKE                 the whole stretch -> <session>/_takes/
OPEN TAKE(S) & CUT CLIPS         carve labelled clips out of it, any time
    EXTRACT CLIP (Ctrl+Space)      -> an ordinary stage-raw recording
    NEXT >> (Ctrl+Right)           -> step one clip length along
```

### Cued sequence

`START SESSION` walks `GESTURES × REPETITIONS`:
`REST → PREPARE → 3-2-1 → RECORD → review`. The capture window is wider than
the labelled window, so an early reaction or an overshooting contraction is
still in the raw file.

### Label on graph

Select a span on the live plot and save it. No countdown, no plan.

### Markers, any time

`Ctrl+M` note · `Ctrl+B` break · `Ctrl+Shift+Space` cut since last marker.
Timestamped off the newest **sample**, not the GUI clock.

> **How to actually record**, in one line: make the gesture, let it settle,
> then **hold it relaxed at the effort you would really use**, and take the
> armband off between sittings. The full protocol is in
> [`docs/UI_GUIDE.md`](docs/UI_GUIDE.md#the-five-minute-version).

---

## From recordings to a model

```python
import emg_dataset_tool as T

recs = T.load_dataset("datasets", stage="prefer_trimmed")
X, y, groups, meta = T.make_windows(recs)      # (n, win, 8), labels, person ids
feats, names = T.make_feature_matrix(X, meta["fs"])

for train, test, held_out in T.leave_one_person_out(groups):
    ...
```

`groups` is the whole point.

In practice just press **RETRAIN ON ALL DATA**, which runs `train_model.py` as
a subprocess. It is a **three-stage sweep** — cheap screen, shortlist probe,
full evaluation — over window length × overlap × feature set × representation
× frequency features. It went from a measured **3 h 16 m to ~14 min**; see
[`docs/how_to_fix_long_training.md`](docs/how_to_fix_long_training.md).

Four representations: `raw`, `normalized`, `envelope`, `bandpass` (a hand-
written zero-phase Butterworth that deliberately does not import scipy).
Which one wins is measured, never asserted.

One feature function, `model_feature_vector()`, is shared by the trainer and
the live predictor. If those drifted apart by one feature, live predictions
would be garbage while the cross-validation score stayed perfect.

---

## The honesty machinery

This is the part worth reading. The project assumes any good-looking number is
wrong until proven otherwise, and the tooling enforces that rather than
relying on memory.

**The DC-only control runs on every training run.** It re-scores the winner
using *only* the 8 per-channel mean levels — no shape, no spectrum, no timing
— and prints how much the other features added. If 8 numbers score as well as
50, nothing has been learned about gestures. Under 5 points of gain the run
prints a warning naming the confound, and both numbers go into the sweep
report and the saved payload.

**Both validation protocols are always reported.** With one subject, folds are
grouped by recording; with two or more they switch to subject-grouped
automatically, because otherwise the sweep *selects* a setup that leans on one
person's amplitudes. The winner block prints the chosen protocol and the other
one as contrast, and warns when they disagree by more than 5 points.

**Window-level deduplication is correctness, not speed.** Clips cut from a
take tile it with ~50 % overlap, and grouping CV by recording uid does not
separate them because every clip legitimately owns its uid. Without dedup the
same samples sit in train and test.

**Selecting a model never flatters it.** The model spec decides what gets
trained, never how it is scored. Where selection bias exists (AUTO mode), it
is stated in the dialog, the log and MODEL STATS rather than hidden.

**The feature space view is a PCA, not an LDA** — an LDA would separate the
labels by construction and look clean even when the separation is electrode
drift.

**The GPU badge never guesses.** Until the subprocess probe answers it reads
`checking…`, and because the sklearn sweep has no GPU backend, both the
tooltip and the RETRAIN dialog say in words that RETRAIN runs on the CPU even
when the badge is green.

---

## Safety rules

Invariants. Breaking one is a bug, not a trade-off.

1. **Raw is written once and never rewritten.** `*_raw.*` are immutable.
2. **Nothing is overwritten.** Every write goes through `unique_path()`.
3. **Trimming creates new files** with a new stage suffix.
4. **Deleting quarantines into `_deleted/`** — never `os.remove`.
5. **Bad recordings are flagged, never discarded.** The operator decides.
6. **Samples are captured on the acquisition thread**, so a slow frame cannot
   drop one.
7. **Every file gets a config snapshot.**
8. **Annotating never edits data.** Notes go to a sidecar; the marker log is
   append-only.
9. **A free take is never training data**, and cutting from it never alters
   it.
10. **No fake predictions.** Only a real model's `predict()`.
11. **Selecting a model never flatters it.**

---

## Where things live

```
emg_dataset_tool.py   the entry point: python emg_dataset_tool.py
emg_app/              the application package (module map: emg_app/README.md)
training/             trainers, job runner, NN worker, the comb test
tools/                report and screenshot generators
tests/                runnable check scripts
docs/                 every other document
legacy/               retired code, kept for the record
datasets/ models/ exports/
```

| Path | Role |
|---|---|
| `emg_dataset_tool.py` | the entry point, and `import emg_dataset_tool as T` still exposes every name in the package |
| `emg_app/` | the application, split by topic: config, signal processing, calibration, features, dataset, acquisition, neural engine, jobs, prediction, and `gui/` (widgets, one file per dialog, the main window as one mixin per feature) |
| `emg_app/myo/myo_raw.py` | the Myo BLE protocol wrapper — **the working communication layer** |
| `training/train_model.py` | the three-stage sweep, honest validation, saves `models/gesture_model.joblib` |
| `training/explain_votes.py` | rebuilds the DECISION TRAIL side file without re-running a sweep |
| **`training/train_nn.py`** | trains the **neural network** -- a separate second opinion, on CUDA, never part of the answer |
| `training/nn_arch.py` / `nn_serve.py` | its architectures, and the subprocess that serves it live |
| `training/train_job.py` | runs training in the background, so closing the app does not stop it |
| `training/nn_ml_comb_test.py` | the ML + NN combination search ([`docs/COMB_TEST.md`](docs/COMB_TEST.md)) |
| `tools/make_ui_guide.py` | regenerates the annotated screenshots in `docs/images/` |
| `legacy/` | the **retired** `train_torch.py`, the original `myo.py` demo and `test_myo.py` monitor |
| `datasets/` | recorded data (`_`-prefixed folders are never discovered) |
| `models/` | the models, specs, the panel, timestamped archives, `reports/`, `comb_test/` |
| `exports/` | combined ML bundles |

---

## Documentation

| File | What it covers |
|---|---|
| [`docs/MANUAL.md`](docs/MANUAL.md) | **the complete manual** — install, architecture, the interface, the ML pipeline, and the full engineering record of what broke and how it was fixed |
| [`docs/UI_GUIDE.md`](docs/UI_GUIDE.md) | **the UI, screen by screen, with annotated screenshots** |
| [`docs/GUIDE.md`](docs/GUIDE.md) | user manual: every file, every setting, ready-made presets |
| [`CLAUDE.md`](CLAUDE.md) | orientation and the invariants — read before changing anything |
| [`emg_app/README.md`](emg_app/README.md) | the package's module map and the two rules that keep the split correct |
| [`docs/COMB_TEST.md`](docs/COMB_TEST.md) | the ML + NN combination test of 2026-10-22, every number |
| [`docs/EDA.md`](docs/EDA.md) | **what is actually in `datasets/`, measured** — read before quoting any accuracy |
| [`docs/how_to_fix_long_training.md`](docs/how_to_fix_long_training.md) | why the sweep took 3 h 16 m and what was done |
| [`docs/DATASET.md`](docs/DATASET.md) | on-disk format, filenames, CSV/NPZ schema |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | threads, buffers, state machine, data flow |
| [`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md) | recordings to a validated model |
| [`docs/problems.md`](docs/problems.md) | every failure hit, the fix tried, and whether it worked |
| [`tests/README.md`](tests/README.md) | the check scripts; three need the armband |

---

## Tests

**1603 checks across sixteen scripts.** Twelve need no hardware at all:

```powershell
python tests\test_datalayer.py        # 150    python tests\test_trail.py       # 103
python tests\test_gui.py              #  91    python tests\test_sweep_cost.py  #  48
python tests\test_review.py           #  50    python tests\test_filters.py     #  52
python tests\test_label_on_graph.py   #  48    python tests\test_predict.py     # 109
python tests\test_markers.py          #  88    python tests\test_model_spec.py  # 346
python tests\test_free_take.py        # 268    python tests\test_no_torch.py    #  11
python tests\test_nn.py               # 158
```

`test_torch.py` needs the GPU; `test_live*.py` need the armband.

---

## Current state

**Two wearers, and the first honest number.** `Dano` 44 recordings + `Betka`
41 = **85**, six gestures, 3–4 s clips of relaxed holds.

A second wearer tripped the automatic switch the pipeline was built for: folds
moved from recording-grouped to **subject-grouped** on their own, so the winner
was selected on data from a person it had never seen.

```
subject-grouped  (the chosen protocol)   83.3 % window   82.4 % vote
recording-grouped (contrast)             97.7 % window   97.6 % vote
majority class                           19.2 %
held out: Betka 89.3 %   Dano 77.2 %
```

**The within-subject score did not improve — it got exposed.** That 14-point
gap between the two protocols *is* the subject-specificity, and the whole point
of reporting both is that one number alone would have let 97.7 % be read as a
cross-wearer result. 83.3 % on an unseen wearer against a 19.2 % baseline is
the first result here that is not a measurement of electrode state.

The previous 325-recording dataset remains archived in
`datasets/_archive_pre_r1_20261015/` — not deleted — because its 99.6 %
accuracy was not gesture recognition: the DC-only control scored **98.9 %**
against the full vector's 98.9 %, so the other 42 features added **+0.0
points**. Every active gesture in it was a single free take; 325 recordings
held only 52 distinct sources.

Still open: more **sittings per wearer**, armband off and refitted between.
Two wearers fixed the cross-person question; within a person each gesture is
still one take, which is why the recording-grouped contrast is still 97.7 %.

**Do not quote an accuracy from this project without its protocol and the
DC-only control next to it.** That rule is the whole reason the tooling prints
them.

# GUIDE — what every file does, what every setting does, how to set it up

Start here. This is the user manual.

- **Section 1** — install and run.
- **Section 2** — what each file in the folder is for.
- **Section 3** — the window, button by button.
- **Section 4** — **every configuration setting**, what it does, when to change it.
- **Section 5** — **ready-made setups** for different jobs.
- **Section 6** — troubleshooting.

---

# 1. Install and run

### Install (once)

```powershell
python -m pip install -r requirements.txt
```

That is everything. **PyTorch is not needed** — the retired neural-network
experiment was the only thing that used it, and the application has never
imported it.

### Run the application

```powershell
python emg_dataset_tool.py
```

### Other commands

```powershell
python emg_dataset_tool.py --summary     # print dataset stats, no GUI
python legacy/test_myo.py                       # the old reference monitor

python tests\test_datalayer.py           # 150 checks, no armband needed
python tests\test_gui.py                 # 91 checks, no armband needed
python tests\test_review.py              # 50 checks, no armband needed
python tests\test_label_on_graph.py         # 48 checks, no armband needed
python tests\test_predict.py             # 109 checks, no armband needed
python tests\test_trail.py               # 103 checks, the DECISION TRAIL
python tests\test_nn.py                  # 158 checks, the neural net (and that it stays separate)
python tests\test_no_torch.py            # 11 checks, proves no PyTorch is needed
python tests\test_live.py                # 35 checks, NEEDS the armband
python tests\test_live_gui.py            # 25 checks, NEEDS the armband
python tests\test_live_review.py         # 29 checks, NEEDS the armband
```

If a test says `ModuleNotFoundError: emg_dataset_tool`, run it as
`$env:PYTHONPATH = "."; python tests\test_live.py`.

**Only one program can use `COM3` at a time.** If the tool says it cannot open
the port, close `test_myo.py` (and Myo Connect, if installed) first.

### First run, step by step

1. Plug in the BLED112 dongle. Put the armband on, logo up, towards the wrist.
2. Flex once — the armband wakes up and the pill at the top turns green
   **MYO CONNECTED**.
3. Set **Person id** and **Session id** in the right-hand panel.
4. Watch the graph while you make each gesture. All 8 channels should move.
5. Press **START SESSION** and follow the big banner.

---

# 2. What each file is for

## Files you use

| File | What it is |
|---|---|
| **`emg_dataset_tool.py`** | **The application.** The config block at the top is the only part you normally edit. |
| **`GUIDE.md`** | This file. |
| `requirements.txt` | Dependency list for `pip install -r`. |
| `CLAUDE.md` | Technical orientation for a developer (or a future AI session). |
| `docs/DATASET.md` | Exact on-disk format: filenames, CSV columns, NPZ keys, metadata. |
| `docs/ARCHITECTURE.md` | Threads, buffers, state machine, rendering. |
| **`train_model.py`** | Trains the gesture model: sweeps models and settings, validates honestly, saves `models/gesture_model.joblib`. **This is the one the app loads.** Runs on the CPU. |
| **`explain_votes.py`** | Builds the DECISION TRAIL file for the model you already have, without re-running a three-hour sweep. Use it if MODEL STATS says there is no decision panel. |
| **`train_nn.py`** | **Trains the NEURAL NETWORK** -- the separate second opinion behind the NEURAL NETWORK panel. Sweeps five small sequence architectures on the GPU and saves `models/gesture_nn.pt`. It does **not** touch the model the app predicts with. |
| `nn_arch.py` | The five architectures. Shared by the trainer and the live worker, so a saved net is always rebuilt the way it was trained. |
| `nn_serve.py` | The neural net's live worker. The only file that imports PyTorch while the app is running -- it runs as a separate process, which is how the net can use CUDA without the recorder ever loading torch. |
| `train_torch.py` | **Retired experiment, replaced by `train_nn.py`.** The original GPU neural-net trainer. Not run any more -- it lost to a plain LDA and had none of the validation the current one shares with `train_model.py`. Kept for the record. |
| `docs/ML_PIPELINE.md` | How to turn the data into a validated model, which features matter and why, and the route to a wearer-independent model. |
| `tests/` | Nineteen scripts that verify the tool. Run them after changing config — see `tests/README.md`. |

## Files that make the armband work — do not touch

| File | What it is |
|---|---|
| **`myo_raw.py`** | The Myo BLE protocol driver. **This is the working communication layer.** Everything else depends on it. |
| `common.py` | `pack`/`unpack` helpers used by `myo_raw.py`. |

## Leftovers kept for reference

| File | What it is |
|---|---|
| `test_myo.py` | The original matplotlib live monitor. Kept because it is the proof of how the connection must be driven. Still runs. |
| `myo.py` | The original k-NN pose demo from the myo-raw project. Not used by the tool. |
| `myo64.dll` | From the official Myo SDK. **Not used** — this project talks to the dongle directly. Safe to ignore. |
| `__pycache__/` | Python bytecode cache. Ignore. |

## Folders created while you work

| Folder | What goes in it |
|---|---|
| `datasets/` | All recorded data: `datasets/<person>/<session>/<GESTURE>/…` |
| `datasets/…/_graphs/` | Graph images you export. |
| `datasets/…/_deleted/` | Recordings removed with DELETE LAST. **Still there** — nothing is erased. |
| `datasets/<person>/calibration.json` | That wearer's current calibration, if they have one. Optional — see CALIBRATE below. |
| `datasets/<person>/calibration_log.jsonl` | Append-only: every calibration that wearer has ever had, including ones you cleared. |
| `exports/` | Combined ML bundles from EXPORT DATASET. |
| `models/` | `gesture_model.joblib` (what the app loads), timestamped archives, and `reports/` with every sweep's scores. Created by `train_model.py`. |
| `models/gesture_panel.joblib` | Every candidate the sweep trained, for MODEL STATS → DECISION TRAIL. Currently 9.5 MB, and it follows what is in MODEL SPEC — it has been as large as 57 MB. The app predicts fine without it and never loads it to predict — delete it freely, or rebuild it with `python training/explain_votes.py`. |

---

# 3. The window

```
┌──────────────────────────────────────────────────────────────────────────┐
│ MYO EMG DATA COLLECTION   [MYO CONNECTED]     PERSON  GESTURE  RECORDED  │
├──────────────────────────────────────────────────────────────────────────┤
│                        GET READY: FIST                                   │  <- banner
│                      Hold the start position                             │
│  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░                          │
├────────────────────────────────────────────┬─────────────────────────────┤
│ Window 5s  [x]Autoscale  Sensitivity ────  │ SUBJECT / SESSION           │
│ [ ]RMS  [Freeze]  [Reset]   Channels 1-8   │ RECORDING PLAN              │
│ ┌────────────────────────────────────────┐ │ CHANNELS                    │
│ │ CH1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ │ │ RECORDINGS (this run)       │
│ │ CH2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ │ │ LIVE CLASSIFIER             │
│ │  …  (8 channel bands)                 │ │ LOG                         │
│ └────────────────────────────────────────┘ │                             │
├────────────────────────────────────────────┴─────────────────────────────┤
│ [START SESSION][PAUSE][STOP][SKIP][RECORD]      [SAVE GRAPH][EXPORT]     │
├──────────────────────────────────────────────────────────────────────────┤
│ CONNECTED    50 Hz    buffer 42 %    samples 128,430  dropped 0  recon 0 │
└──────────────────────────────────────────────────────────────────────────┘
```

## The banner states

| State | Colour | Meaning |
|---|---|---|
| `READY` | grey | idle, press START SESSION |
| `RELAX` | blue | rest period before the next repetition |
| `GET READY: FIST` | amber | the next gesture is announced |
| `3` `2` `1` | orange | countdown — also shown huge over the graph |
| `RECORD NOW` | red | **do the gesture**; the labeled window is open |
| `STOP` | orange | label closed, still capturing the release |
| `CHECK: FIST` | cyan | **the review step** — the graph is frozen on what you just did, with the proposed cut drawn over it. Nothing happens until you decide. |
| `SAVED` | green | written to disk, shows `Repetition 12 / 180` |
| `PAUSED` | violet | paused; the reason is shown underneath |
| `SESSION COMPLETE` | green | the whole plan is done |
| `MYO DISCONNECTED` | amber | link lost, reconnecting; a running session auto-pauses and auto-resumes |

## Live prediction — the LIVE CLASSIFIER panel

Right-hand panel, below the recordings list. It only works once you have
trained a model (`python training/train_model.py`); with no model it says so and the
buttons stay disabled. **It never shows an invented label or percentage.**

```
┌─ LIVE CLASSIFIER ─────────────────────────┐
│ Mode  [One shot (stop after each)   ▾]    │
│                                           │
│                 FIST                      │  <- big result
│                 87 %                      │  <- confidence
│                                           │
│  REST  ▇░░░░░░░░░░░░░░░░░░░░░░░░    3 %   │  <- every class,
│  FIST  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░   87 %   │     so you can see
│  OPEN  ▇▇▇░░░░░░░░░░░░░░░░░░░░░░   10 %   │     what it nearly said
│  ...                                      │
│                                           │
│ ExtraTrees-400 | 200 ms @ 60 Hz | level…  │
│ [    PREDICT    ]  [ MODEL STATS ]        │
│ [   RETRAIN ON ALL DATA   ]               │
│ [   MODEL SPEC...         ]               │
│ COMPUTE  GPU READY -- GeForce GTX 1650    │  <- green: a GPU exists
└───────────────────────────────────────────┘      amber CPU ONLY: none
```

The COMPUTE line (and the matching chip in the status bar) says which
processor is available — see **GPU or CPU** under *Training a model*. It
reads `checking...` for the first seconds after startup, and a green GPU
there does **not** mean RETRAIN uses it.

### Is a model trained, and is it still the right one?

The two lines above COMPUTE answer that without opening anything:

```
MODEL   NO MODEL TRAINED
on disk  119 recordings | FIST 56, REST 63 | 1 subject(s): Dano | 1 session(s)
```

```
MODEL   UP TO DATE  --  trained on 2026-10-14 11:38
trained on  119 recordings | FIST 56, REST 63 | 1 subject(s): Dano | 1 session(s)
```

```
MODEL   OUT OF DATE  --  trained on 2026-10-13 22:10
trained on  96 recordings | FIST 44, REST 52 | 1 subject(s): Dano | 1 session(s)
```

| What it says | What it means |
|---|---|
| **NO MODEL TRAINED** (grey) | nothing trained yet. The second line shows what is on disk waiting to be used. |
| **UP TO DATE** (green) | every recording currently on disk went into this model. |
| **OUT OF DATE** (amber) | you have recorded since. Hover it: it says how many recordings are new (or how many it used have been deleted). The model still works — it is just not using your latest data. |
| **TRAINED on …** (amber) | an older model that does not record which recordings it used, so the question cannot be answered. Retrain once and it will. |

It updates by itself: record a gesture and the line turns amber within a few
seconds. For the full detail — the exact session list, the newest recording
in the training set, the dataset id — open **MODEL STATS**, which now has a
TRAINING DATA section.

The result is also **drawn over the graph**, top-left, so you can watch the
traces and the prediction at the same time.

### When exactly do I make the gesture?

**There is no 3-2-1 countdown here.** That belongs to *recording* training
data. The classifier is a different thing, and the rule is simple:

> **Make the gesture and HOLD it until the answer appears.**

Not before, not already finished. The model was trained on the middle of a
sustained contraction, so the muscle has to be active during the slice it
reads.

#### One shot — about 1.3 s end to end

```
press PREDICT ──0.8 s──┬────── 0.5 s ──────┬── result (held 3 s)
                       │                   │
 "DO THE GESTURE"      │  "HOLD..."        │
 hold it -- recording  │  reading the      │
 the signal in 0.8 s   │  signal now       │
                       ↑                   ↑
            start the gesture      THIS slice is what
            as you press           gets classified
```

The panel and the graph overlay tell you which phase you are in: amber
**DO THE GESTURE** while you get into it, red **READING THE SIGNAL** while the
classified window is being taken. Keep holding through both. The two durations
are `ONE_SHOT_DELAY_S` and the model's own window length (shown in MODEL
STATS).

#### Continuous — hold and watch

Press **PREDICT** once; it classifies the last window every
`PREDICTION_INTERVAL_MS` and keeps going until you press **STOP**. Just hold a
gesture and read the panel.

Because it soft-votes `PREDICTION_SMOOTHING` windows, there is roughly
*window + smoothing* of lag — about **1 s** with the defaults. **Hold each
gesture for 1–2 seconds** before believing the reading, and expect the bars to
wander while you transition between gestures.

#### Which mode for which gesture

| | Best mode | Why |
|---|---|---|
| Held postures (`FIST`, `OPEN`, `THUMB_UP`, `REST`) | either | the signal is steady, so a single window is representative |
| Movements (`WAVE_LEFT`, `WAVE_RIGHT`) | **continuous** | a 0.5 s slice catches only part of a movement; watching it live is far more informative than one snapshot |

Switching mode always stops a run in progress. Prediction and recording are
independent — you can leave continuous mode running during a session.

A result below `PREDICTION_MIN_CONFIDENCE` is shown as `FIST?` in amber with
"below the confidence threshold" rather than as a confident answer.

Prediction and recording are independent — you can record a session while
continuous prediction runs, and the review step works normally.

### RETRAIN ON ALL DATA

The button under MODEL STATS. Press it after recording more takes and it
retrains **using every recording on disk** — the ones you just made *plus*
every earlier session and subject. The confirmation dialog lists exactly what
will be used, for example:

```
Retrain using ALL 148 recordings on disk.

2 subject(s):
    Dano: session_01 (72), session_02 (60)
    person_02: session_01 (16)

Everything is used together -- the recordings from this run are added to
the earlier ones, not trained on separately.
```

There is deliberately **no "train on just the new data" option**: that would
throw away most of your dataset and give a worse model every time.

- **Quick** — reduced grid, about a minute. Use this between recording blocks.
- **Full sweep** — every model enabled in MODEL SPEC against every setting.
  Budget **hours**, not minutes: with seven models enabled on ~120
  recordings the grid is over 1300 combinations. The dialog tells you how
  many models are enabled before you start it.

It runs as a separate process, so the GUI stays responsive and a training
crash cannot take the recorder down. Progress streams into the LOG panel. When
it finishes, the new model is loaded automatically and you are told how it
compares to the previous one (`+4.2 points vs the previous model`). The old
model is **archived**, not overwritten — `models/gesture_model_<timestamp>.joblib`.

Two things to know:

- **Retraining is refused while a recording session is running.** Training
  saturates the CPU and would disturb the live stream.
- A retrain only becomes a *wearer-independent* model once the dataset
  contains several people. With one subject it stays subject-specific no
  matter how many times you retrain, and MODEL STATS will say so.

### MODEL STATS

Opens everything measured about the loaded model: per-gesture and per-window
accuracy, macro F1, chance level, the **validation protocol and its limits**,
per-class recall/precision, a colour-coded confusion matrix, the exact
preprocessing, and a **TRAINING DATA** section — when it was trained, how
many recordings per gesture, which subjects and sessions, the newest
recording that went into it, the dataset id, and whether that is still what
is on disk. It also states plainly when a cross-subject claim cannot be
made. There is a **Reload model** button for after you retrain.

## CALIBRATE — making one number mean the same effort on everybody

**CALIBRATE…** (`Ctrl+Shift+C`, in the LIVE CLASSIFIER panel) is **on since
2026-10-16, and the current model needs it.** It is still one setting
(`CALIBRATION_MODE`): set it to `off` and retrain, and you have exactly what
you had before.

It was switched on because it was measured: on a wearer the model has never
seen it is worth **+5.3 points** per window (81.3 % → 86.5 %) and **+7.8 on
the per-gesture vote** (80.4 % → 88.2 %) — measured with each wearer's
calibration fitted on recordings that were then left out of the scoring, so it
is not a number the calibration gave itself. The table and the caveats are in
CLAUDE.md §8k.

> **What this means for you in practice:** every new wearer has to spend six
> seconds calibrating before the classifier will work for them. If they skip
> it, the LIVE CLASSIFIER refuses and says so — it does **not** guess.

### Why it exists

The armband reads a different number on every wearer for the same effort.
Skin, muscle, how tight the band is, where pod 1 ended up — all of it scales
the signal. So a model trained on you can read a new wearer's hard squeeze as
your gentle one.

A calibration fixes that by measuring two levels per channel, once, for one
person:

* **rest** — what that channel reads when they are relaxed;
* **reference** — what it reads during a strong contraction.

Those two numbers map the wearer onto a common scale. Six seconds of signal,
measured once.

### How to do it — the guided sweep

1. Put the band on properly and connect, so the graph is running. The dialog
   shows a green **STREAM: streaming, 50 Hz** line when it can actually
   capture; if it says NO STREAM it tells you why.
2. Check the **Person id** in the SUBJECT panel is the person wearing it. The
   calibration is saved for *that* person.
3. Press **CALIBRATE…**, then **START SWEEP**.
4. It cues **RELAX** first — stay relaxed — and then each gesture in turn,
   with a short `GET READY: FIST` beat before each one. Hold each gesture
   firmly and steadily for its countdown. Strong, not explosive: it is
   setting the top of the scale, so a spike is worse than a firm hold.
5. Watch the table as you go. Each gesture gets a row saying how many samples
   arrived, which pod it peaked on, and `accepted` — or why not.
6. **If a row says `hold it harder`, select it and press REDO SELECTED.**
   That gesture contributes nothing to the calibration until it registers, so
   repeating it is the point, not a fallback. Do that until every row you
   care about reads `accepted`.
7. Press **SAVE CALIBRATION**.

The button then reads **RECALIBRATE…**, and the status line underneath says
`CALIBRATED`.

**Why a whole sweep and not just one squeeze.** The reference is meant to be
each channel's *full scale*, and one gesture only drives the muscles that
gesture uses. On the real data one wearer's channels top out anywhere between
154 and 992 depending on which gesture you ask — so calibrating from a single
fist sets the quiet channels' scale from almost nothing, and then those
channels get divided by noise. Measured on a stand-in built from the real
profiles: a one-gesture reference under-scales **seven of eight channels by up
to 3.2×**.

You do not have to do all six. Three well-spread gestures (something that
peaks on the inside of the forearm, something on the outside, something that
uses the whole arm) cover the eight pods better than six sloppy ones.
`CALIBRATION_GESTURES` sets the list.

### The quick way, if you are in a hurry

**RELAX** then **SQUEEZE HARD**, two captures, six seconds. It is in the
QUICK box and it is worse than the sweep for the reason above — but it is
much better than nothing, and it is what to use when you just need the
classifier running for somebody.

**It stays valid until you press RECALIBRATE.** It does not expire, nothing
re-measures it behind your back, and it survives closing the application. It
is stored with the person, in `datasets/<person>/calibration.json`, so
switching **Person id** switches which calibration is in force.

### No armband? DERIVE FROM RECORDED DATA

The dialog also works offline. **DERIVE FROM RECORDED DATA** takes the same
two level vectors from percentiles of everything that person has already
recorded — the quiet 10 % stands in for the rest capture and the loud 5 % for
the reference. It is how a dataset recorded before this feature existed gets
calibrated, and it is how `Dano` and `Betka` were calibrated.

It **ignores the labels**. That is deliberate rather than lazy: using the
recordings labelled `REST` to set the rest level would mean a cross-wearer
score was partly built from the held-out person's answers.

### Which mode to use

| Mode | What it does | Measured, on a wearer the model never saw, with the calibration held out too |
|---|---|---|
| **off** | nothing is applied; the model sees raw device counts | 81.3 % — what every number before 2026-10-16 was measured with |
| **gain** | one divisor for all eight channels | 81.1 % — nothing. These two wearers' overall volumes differ by only 9 %, so there is nothing for one divisor to fix |
| **per channel** | each channel gets its own divisor | 85.6 % — most of the benefit is already here |
| **affine** | per channel, with the wearer's rest level subtracted first | **86.5 %** — the default, and the one that pays |

**Which one is best was measured, not decided.** Three full sweeps, same
recordings, same classifiers, same folds, only the calibration different.
What differs between these two wearers is not their overall volume but their
per-channel profile — up to 1.85× on one channel, because of where the pods
ended up — and `affine` is the mode that removes that.

The interesting part is *why* it helps: calibrated, a whole family of
features that was useless before starts working. The classic EMG features
(waveform length, zero crossings) centre each channel, which throws away the
activation level and made them worthless on this stream — but once the
wearer's own rest level has already been taken out, they become comparable
*between* wearers and start paying. The winner went from 50 features to 154,
and its macro F1 from 73.8 % to 83.2 %.

**It does not fix the dataset's real problem**, and it is not claimed to.
Within one wearer each gesture is still a single take, so the model can still
win by recognising which block of time it is looking at — the within-wearer
score barely moved (97.8 % → 98.2 %). More sittings per wearer, band taken
off and refitted between them, is still the fix for that.

### Two things that will trip you up

**Changing the mode makes the loaded model refuse to predict, and that is
correct.** A model remembers which scale it was trained on. If you switch
calibration on after training, the LIVE CLASSIFIER says the model was trained
with a different calibration and asks you to retrain. It is refusing because
it *would* still produce labels — confident, and wrong by that wearer's gain.
Retrain, or put the mode back.

**Calibrate everybody or nobody.** A dataset where some wearers are calibrated
and some are not is not on one scale, and scores from it mean nothing. The
trainer prints a `MIXED SCALES` warning when it sees that.

### What it never does

* It never changes a recording. Calibration is applied when features are
  computed, not when signal is saved, so every file on disk stays
  byte-identical and there is no second, calibrated copy of your dataset.
* A bad calibration is **refused**, not quietly accepted. A single gesture
  capture is only counted as evidence if its peak reaches 60 counts *and*
  twice the wearer's rest level, with at least half a second of samples —
  otherwise the row says `hold it harder` and that gesture is left out. And
  the finished record is refused outright if its reference is under 40 counts,
  under twice rest, or short of samples: the dialog says so and SAVE stays
  disabled. Dividing a wearer's whole signal by noise is worse than not
  calibrating.
* **CLEAR** does not destroy the measurement. The full record is written to
  `datasets/<person>/calibration_log.jsonl` first, which is append-only and
  holds every calibration that person has ever had.

## MODEL SPEC — which classifiers compete, and how they vote

**MODEL SPEC…** (`Ctrl+Shift+M`, in the LIVE CLASSIFIER panel) decides what the
next retrain actually trains. It writes `models/model_spec.json`, which
`train_model.py` reads via `--spec`.

### The roster

A checklist of candidates. Each row is one classifier with its own
hyperparameters, and only the ticked ones are trained:

| Control | What it does |
|---|---|
| the tick box per row | whether this candidate is trained at all |
| **ALL ON / ALL OFF** | bulk toggle. Everything off is repaired — there is always at least one trainable model |
| **ADD** | another candidate of the chosen kind. You can have several of the same kind with different settings |
| **REMOVE SELECTED** | drop a candidate. The last one is protected |
| **RESET TO DEFAULTS** | back to the built-in roster |

Nine kinds are available: LDA, SVM (RBF and linear), Random Forest, Extra
Trees, Histogram Gradient Boosting, Logistic Regression, k-NN and a small MLP.

### Hyperparameters

Select a row and its settings appear on the right, generated from the schema —
so every value offered is one that really builds. This is where the training
aids live:

| Aid | Where |
|---|---|
| **L1 / L2 / elastic net** | Logistic Regression → `penalty`, with `l1_ratio` for elastic net. L1 drives weights to zero (feature selection), L2 shrinks them |
| **L2 by another name** | `C` on both SVMs and LogReg is *inverse* strength — lower C = more regularisation. `alpha` on the MLP. `l2_regularization` on the booster |
| **early stopping** | MLP and Histogram Gradient Boosting, with `validation_fraction` (how much is held back) and `n_iter_no_change` (patience) |
| **batching** | MLP → `batch_size` (0 = auto) and `learning_rate_init`. Ignored by the `lbfgs` solver, which is full-batch — the app does not pass it settings it cannot use |
| **class balancing** | `class_weight = balanced` on the SVMs, both forests, the booster and LogReg. Reweights rare gestures so a lopsided dataset does not just predict the common one |
| **tree regularisation** | `min_samples_leaf` up, `max_leaf_nodes` down, `max_depth` (0 = unlimited) |

Invalid combinations are fixed rather than left to explode mid-sweep: `lbfgs`
cannot do L1, so `solver = auto` picks `saga`, and an explicit mismatch is
corrected with a note in the training log.

### Voting

With **VOTING ENSEMBLE** on, an ensemble of every enabled model is added as
*one more candidate*:

* **soft** averages the members' predicted probabilities (usually better; any
  member without `predict_proba` is left out of the ensemble and told so).
* **hard** counts one vote per model.

By default the ensemble **competes** — it is scored by the same
cross-validation as its own members and the best score wins. If the vote is
worse than its best member, that is worth knowing. **"always save the
ensemble"** overrides that, which is a deliberate choice rather than an
improvement.

### AUTO MODE

Ticking **AUTO MODE** generates hyperparameter combinations for each enabled
model and trains every one of them. Each combination becomes its own named
candidate (`SVM-rbf-C10[C=100.0, class_weight=balanced, gamma=auto]`), scored
by exactly the same cross-validation and shown as its own row — there is no
separate, hidden tuning path.

* **Max combinations per model** caps it. The full grid is sampled down to
  that many with a seeded shuffle, so **a rerun gives the same candidates**.
* Your configured values are always combination #1, so AUTO can never lose a
  setting you made.
* A model can **opt out** (`auto-tune` in its own form) to stay pinned while
  the others are tuned.
* The dialog shows how many candidates that is and roughly what multiple of
  the training time it costs.

**The honest-result warning the dialog shows is not boilerplate.** Picking the
best of 28 candidates on the same folds makes the *winner's* score optimistic
— the more combinations, the more optimistic. The number that did not get to
pick its own folds is leave-one-subject-out, in MODEL STATS.

## MODEL STATS — what the model actually is

The dialog is **not modal**: leave it open and keep using the window behind
it. That is deliberate, because the top section follows the armband live.

### DECISION TRAIL — how each model decided, and how they become one

This is the picture of the whole decision, and it moves while you use the
armband:

```
  INPUT           LDA-shrinkage *  ████████░░  FIST  0.998  ✓ ╲
  CH1 ███ 412     LDA-svd          ████████░░  FIST  0.996  ✓  ╲
  CH2 ██  287     SVM-rbf-C10      ██████░░░░  FIST  0.842  ✓   ╲        SOFT VOTE
  CH3 ████ 503    RandomForest-400 █████░░░░░  FIST  0.630  ✓    ●──  mean of 7 = 0.743
  CH4 ██  251     ExtraTrees-400   ██████░░░░  FIST  0.805  ✓   ╱
  CH5 ███ 388     HistGradBoost    ███████░░░  FIST  0.923  ✓  ╱         FIST
  CH6 ████ 467    LogReg           ░░░░░░████  REST  0.991  ✗ ╱          0.743  6 of 7 agree
  CH7 ███ 350
  CH8 ███ 402                                                   DEPLOYED  LDA-shrinkage -> FIST
```

Left is the window every model is looking at — the eight channel levels.
Each row is one classifier from your MODEL SPEC: its probability split, the
gesture it picked, how sure it is, and whether that matches. The curves carry
each verdict into the vote, thicker and brighter the more confident it is, so
a split vote looks like two colours arriving at the same point. The box on
the right averages the seven probabilities and takes the larger side.

Two buttons choose where the numbers come from, and the view always says
which:

- **LIVE** — follows PREDICT. Every window the classifier handles is re-run
  through all the candidates, on the *same* data the deployed model just
  used. Press PREDICT in the LIVE CLASSIFIER panel and watch the rows move
  as you contract. These candidates were fitted on your whole dataset, so
  **what they agree about here is not an accuracy figure** — it shows you how
  the vote forms. The caption says so every time.
- **RECORDED** — scrub the decisions captured during training, one window at
  a time, with a recording picker and a slider. These *are* the honest
  numbers: each window was judged by a model that had never seen its
  recording. **NEXT DISAGREEMENT** jumps straight to the windows where the
  candidates did not agree, which on a clean dataset are the only interesting
  ones (28 windows out of 3162 on the current data).

The row marked `*` and the `DEPLOYED` line are the important detail: the app
loads **one** model, the sweep's winner. The vote is drawn so you can see how
the candidates combine, but it is not what predicts and not what goes on the
robot. Switching to it would be a decision, not an upgrade.

Under the drawing, **EVERY CANDIDATE, SAME SETUP** scores all of them on the
winner's preprocessing. Those numbers are lower than the ones in the sweep
report, where each model was allowed to pick its own window length and
feature set — that is the point, because only one setup at a time makes the
seven comparable.

If the section says there is no decision panel, run a retrain (or
`python training/explain_votes.py` to build one for the model you already have). The
first window after opening the dialog takes a moment while the candidates
warm up; after that it keeps pace with the classifier.

### The rest of the dialog

Beyond the accuracy and its protocol, it also shows:

**WHAT THIS MODEL IS** — every stage of the fitted estimator, in order, read
off the object on disk rather than off the spec that was meant to produce it.
If the two ever disagree, this is what will run:

```
VotingClassifier   --   4 models, soft voting
  combines: average of each model's predicted probabilities, highest average wins
  weights : equal
  member 1: LDA-shrinkage
        Pipeline  (2 stages, in order)
             [1] scale: StandardScaler
          -> [2] clf: LinearDiscriminantAnalysis
                 solver=lsqr, shrinkage=auto
  member 3: SVM-rbf-C10
        Pipeline  (2 stages, in order)
             [1] scale: StandardScaler
          -> [2] clf: SVC
                 kernel=rbf, C=10, gamma=scale
```

**WHAT IT COMPETED AGAINST** — the spec the model was trained from: every
candidate with its hyperparameters, the winner marked, and whether voting and
AUTO mode were on. When AUTO was on it repeats the optimism warning, because
that is exactly the case where the headline number flatters the model.

**TRAINING DATA** — which recordings went in, and whether that is still the
dataset:

```
status         : OUT OF DATE  --  trained on 2026-10-13 22:10
trained        : 2026-10-13T22:10:51.004
recordings     : 96 of 96 on disk at the time
per gesture    : FIST 44, REST 52
subjects       : Dano
sessions       : Dano/session_01
newest in set  : 20261013-220914-770
dataset id     : 9f31c0ab77e21d04
excluded       : clean
```

The **dataset id** is a digest of every recording that went in, which is how
"up to date" is decided — not by counting, so a deletion and an addition
cannot cancel out. A model trained before this existed shows
`not recorded` and is reported as unknown rather than as current.

## Training a model

```powershell
python training/train_model.py              # full sweep, saves the best model
python training/train_model.py --quick      # fewer combinations, a minute or two
python training/train_model.py --no-save    # evaluate only
```

It sweeps window length, overlap, signal representation, **feature set**,
frequency features, quality filtering and 21 classifiers, validates by
leave-recordings-out, and writes:

- `models/gesture_model.joblib` — what the app loads
- `models/reports/sweep_<timestamp>.json` — every score

Then set `ENABLE_LIVE_PREDICTION = True` (it already is) and press PREDICT.
Details and the cross-subject roadmap: `docs/ML_PIPELINE.md`.

### GPU or CPU — the COMPUTE badge

Two places in the window tell you which processor is in play:

- **the COMPUTE line** in the LIVE CLASSIFIER panel — green
  `GPU READY -- NVIDIA GeForce GTX 1650` or amber `CPU ONLY`. Hover it for
  the CUDA and torch versions;
- **a chip at the right of the status bar** (`GPU NVIDIA…` / `CPU ONLY`),
  which stays visible even with every panel folded away.

For the first few seconds after startup both read `checking...` / `?`. That
is not a bug: the answer comes from a short helper process, and the tool
would rather say "not known yet" than guess.

**A green badge does not mean RETRAIN uses the GPU.** It does not, ever:
scikit-learn has no GPU backend, so the sweep behind that button is CPU work
whatever card you own. The retrain dialog says so before you commit to it.
The GPU is used only by `train_torch.py`.

### Watching training happen — the TRAINING panel

Press RETRAIN and the **TRAINING** section unfolds by itself. It shows, from
events the trainer actually emits as it finishes work:

```
train_model.py  --  119 recordings (full sweep)
[=========                    ]  32%  (41:12 left)
status          running
stage           stage A
algorithm       SVM-rbf-C10
preprocessing   0.20s  ov 0.75  raw/both  (6307 windows x 154 feat)
candidate       430 / 1336
folds           6 per candidate
device          CPU
elapsed         22:41
remaining       about 41:12
saved to        --
```

`status` goes **queued → running → completed**, or **failed** / **cancelled**.
`remaining` says `estimating...` until enough candidates have finished for
the rate to mean anything — it is measured from real completions, never a
fixed guess. `device` says CPU because scikit-learn has no GPU backend; see
below. When it finishes, the panel shows the winner's scores and the file it
wrote, and a popup tells you the same.

Training runs as a separate process, so the graph keeps streaming and the
window stays responsive throughout.

### Why training does not use your GPU

It cannot, and this was checked rather than assumed. The only GPU
implementation of these algorithms is RAPIDS cuML, which ships **Linux-only
wheels for Python 3.10–3.12**. This machine is Windows on Python 3.14, so
there is no installable build — running it would mean putting the whole
project inside WSL2. Your GTX 1650 is not the problem.

Even if it installed, the data is too small to benefit: the largest setup is
~6300 windows × 154 features, and GPU implementations of these algorithms
start winning somewhere around 100,000 rows.

What *did* make it faster is in `CLAUDE.md` §8d — mainly skipping SVM
probability calibration during scoring, which is 5.7× faster and provably
changes no result.

## NEURAL NETWORK — the second opinion, on the GPU

The panel under LIVE CLASSIFIER runs a completely different model on the same
signal: a small sequence network that looks at how the muscle signal *changes*
across the window, instead of boiling the window down to statistics first.

**It is not part of the app's answer, and that is on purpose.** Whatever it
says, the prediction in LIVE CLASSIFIER comes from the scikit-learn model
alone. The two are never averaged. What the panel adds is the ability to see
them *disagree* — "the classifier says OPEN 71 %, the net says WAVE_RIGHT
64 %" — which a single blended number would hide.

### Using it

| Button | What it does |
|---|---|
| **START NN** (Ctrl+Shift+N) | Starts the net on the live signal. It runs in its own process, so it can use the GPU without slowing the recorder down. Takes a few seconds — the panel says `STARTING` while it loads. |
| **NN STATS** | Its own cross-validated numbers, every architecture the sweep tried, and a side-by-side against the deployed model. Works with the net stopped. |
| **TRAIN NEURAL NET** | Sweeps the architectures on every recording on disk and saves its own checkpoint. Does not touch the deployed model. |

Both models can run at once. The chip under the buttons says what the net is
**really** running on (`NN ON GPU -- NVIDIA GeForce GTX 1650`) — which is a
different question from the COMPUTE badge above, that one being about what
hardware exists. Below it, the measured cost per window against its budget.

Like the classifier, **a new wearer must CALIBRATE before the net will run
for them**, and it says so rather than answering on the wrong scale.

### What it is worth, measured

| | per window | per gesture | input |
|---|---|---|---|
| **LDA (the deployed classifier)** | **86.8 %** | **87.1 %** | 154 features |
| Neural net (separate) | 57.5 % | 55.3 % | 30 × 8 raw samples |

Both measured the same way, on the same 85 recordings from two wearers, with
the held-out person never in training. **The net loses by 29 points**, which
repeats what the retired 2026-10-14 experiment found and is why it is a
diagnostic rather than the thing that ships.

Two numbers in NN STATS are worth reading anyway:

- **the DC-only control**: the same net with all time removed scores 47.3 %
  against 57.5 %, so the temporal structure really is worth **+10 points**.
  The net is reading the right thing — there just is not enough of it in eight
  channels of envelope at 50 Hz to beat good hand-crafted features on a
  thousand windows;
- **the gap between protocols**: 95.6 % when folds are split by recording
  against 57.5 % when they are split by *person*. A 38-point gap means the net
  is spending its capacity memorising the wearer. More wearers is what would
  close it — which is the one concrete reason to look at this again later.

### Train it yourself

```powershell
python training/train_nn.py                 # the full sweep on the GPU, ~7 minutes
python training/train_nn.py --quick         # one architecture, one window
python training/train_nn.py --device cpu    # no GPU
```

It writes `models/gesture_nn.pt` plus `models/gesture_nn.json` — the second
one holds the statistics in plain JSON, because the application cannot read a
PyTorch checkpoint (it never imports PyTorch at all).

To switch the whole thing off, set `ENABLE_NEURAL_NET = False`: the panel
disappears and nothing else changes.

### The original neural-network experiment (retired)

`train_torch.py` trained CNN / LSTM / CNN-LSTM models on the GPU and was
retired on 2026-10-14. **It is not run any more** — `train_nn.py` above
replaced it, with the same validation protocol as the main trainer, which the
old one did not have. Its numbers (99.2 % per recording against 100 % for a
plain LDA, on a task a single amplitude threshold solves 98.3 % of the time)
are kept so the experiment stays on the record.

### If the accuracy is low, fix the DATA first

The current model on 72 recordings of one session scores **45.9 % per window /
52.8 % per gesture** against a 16.7 % chance level. That is the dataset
talking, not the model — a 1140-combination sweep over 21 classifiers only
spread the top results across about 5 points.

What the sweep itself proves, in order of impact:

| Finding | Evidence |
|---|---|
| **Weak recordings are actively harmful** | Dropping the ones flagged `no_activity` raised mean accuracy from 26.1 % to **34.1 %** — the single biggest factor in the whole sweep |
| Longer windows help a little | 0.4–0.5 s beat 0.2–0.3 s |
| Absolute amplitude matters | `raw` 35.4 % vs `normalized` **14.7 %** — do not normalise away the level for a single-subject model |
| Level features beat classic ones | `level`/`both` 29 % vs `classic` 26.6 % |
| The model barely matters | LogReg, ExtraTrees, MLP, SVM and LDA all land within ~4 points |

So: **re-record, don't re-tune.** Concretely —

1. **Contract harder and more consistently.** 15 of 72 recordings had no
   detectable activation, including **5 of 12 FIST** — which is why FIST
   recall is 5 %. FIST's mean envelope varies **26-fold** across its own
   repetitions.
2. **Watch the live graph while you record.** If the traces barely move during
   `RECORD NOW`, that repetition is worthless. Use **REDO** (Ctrl+R) on it —
   that is exactly what the review step is for.
3. **25–30 repetitions per gesture**, not 12.
4. **Add sessions on different days**, each with a new `SESSION_ID`.
5. **Make the gestures more distinct.** `WAVE_LEFT` and `WAVE_RIGHT` swap
   constantly because they differ mainly in one channel's magnitude; exaggerate
   the movement. Everything weak collapses into `REST`.
6. Keep `QC_PAUSE_ON = "error"` so you are told when a take is bad.

Then retrain. Expect a large jump: the ceiling here is set by how separable
your gestures actually are on the armband.

## LABEL ON GRAPH — the recommended way to collect data

Instead of being told what to do by a countdown, you stream continuously, do a
gesture whenever you like, then **draw a box around it on the graph and label
it**. Nothing enters the dataset until you select it, so hesitation, false
starts and fidgeting never get recorded.

This is better than the cued workflow for one reason: **you see the signal
before you commit to it.** The old approach recorded whatever happened during
`RECORD NOW`, which is how 15 of the first 72 recordings ended up with no
detectable contraction.

### How to use it

1. **Set the gesture first** — pick it in the LABEL ON GRAPH panel *before*
   you do it, so afterwards you only have to press ADD.
2. **START LABEL MODE.**
3. **Do the gesture.** Hold it clearly. Watch the traces move.
4. **SELECT LAST n s** — a white box drops onto the right-hand edge of the
   graph. **Drag its edges** on any channel to fit the gesture exactly; all
   eight stay in sync. `n` is the **Default cut (s)** spin box in the same
   panel: set it once and every selection from then on is that long, here and
   in the cut-clips dialog. The button relabels itself so you can see what it
   will do.
5. **ADD SELECTION** (or `Ctrl+Space`). It is saved, and the region turns into
   a **coloured band labelled with the gesture**, which scrolls left with the
   signal until it leaves the screen.
6. Repeat. Change the gesture in the combo whenever you want.

**UNDO LAST** removes the most recent mark and moves its files to `_deleted/`.

The graph window is **30 s** by default and the buffer holds **180 s**, so you
can still go back and label something you did half a minute ago.

| Setting | Default | What it does |
|---|---|---|
| `PLOT_WINDOW_S` | `30.0` | Visible time window. The whole point of the big window is having room to select in. |
| `RING_BUFFER_SECONDS` | `180.0` | How far back you can still select. Must exceed `PLOT_WINDOW_S`. |
| `LABEL_MIN_SELECTION_S` | `0.25` | Shortest selection that may be labelled. |
| `LABEL_DEFAULT_SELECTION_S` | `2.0` | Default cut length: the width of a fresh selection box. The **Default cut (s)** spin box sets it live, for both this panel and the cut-clips dialog. |
| `LABEL_MAX_VISIBLE_MARKS` | `14` | How many labelled bands stay drawn (a drawing-cost cap only). |
| `LABEL_COLORS` | per gesture | Band colour per gesture. |

Everything saved this way is an ordinary recording — same files, same format,
same `RETRAIN ON ALL DATA` button. The metadata records
`capture_style: "label_on_graph"` and `selected_by_operator: true`, and the
whole selection is the label window (no pre/post roll, because you drew the
boundaries yourself).

Label mode and the cued session are mutually exclusive; the app refuses to
start one while the other is running.

## FREE TAKE — record the whole session now, cut it up later

This is the workflow for *"let me just record for five minutes and sort it out
at home."* You press one button, stream for as long as you like, press it
again, and the **whole stretch is written to disk in one piece**. Nothing is
labelled yet. Later — tomorrow, on a laptop with no armband attached — you
open the take and carve the useful parts out of it.

### Why it is not the same as LABEL ON GRAPH

Label mode only reaches back `RING_BUFFER_SECONDS` (180 s by default), because
nothing is on disk until you select it. Once those three minutes have scrolled
past, that signal is gone. A free take is on disk the moment you stop, so
*"later"* has no deadline.

### Recording one

1. **Intended gesture** — pick what you sat down to collect, e.g. `FIST`. This
   names the take and becomes the default label when you cut it up. It does
   **not** label the data: every clip is labelled when you cut it out.
2. **START FREE TAKE** (`Ctrl+Shift+R`).
3. The banner turns red and reads **FREE TAKE — RECORDING**, with the gesture,
   the elapsed time, the limit and the live sample count. The panel below says
   the same thing, and the graph carries a `REC` overlay. If the armband drops
   out, the banner says so *and* names the open take, so it is clear the take
   itself did not fail.
4. Do your gestures, with rest in between. **Drop a BREAK (`Ctrl+B`) before
   each one** — those markers are saved with the take and become your cut
   points later. This is the single most useful habit for this workflow.
5. **STOP & SAVE TAKE.** The take appears in **TAKES ON DISK**.

A take stops itself at `FREE_TAKE_MAX_MINUTES` (15 by default) rather than
growing in RAM without limit, and says so in the log when it does. A take
shorter than `FREE_TAKE_MIN_SECONDS` is discarded instead of littering the
folder.

### A take is never training data

It goes to `<session>/_takes/` with the stage `take`. Underscore folders are
pruned by dataset discovery and no stage selector asks for `take`, so
`load_dataset()`, the EXPORT button and `train_model.py` **cannot** see it.
That is deliberate: five minutes of mostly-rest signal loaded as one `FIST`
recording would quietly poison the model. `tests/test_free_take.py` checks
this both on the data layer and through the GUI.

### Cutting it up

Select a take in **TAKES ON DISK** and press **OPEN TAKE & CUT CLIPS**
(or double-click it). You can also reach any take with **OPEN FILE…** — a take
is recognised by its name and gets this dialog instead of the trim dialog.

The dialog shows the whole take across all eight channels, with every marker
you dropped drawn as a separator, and an **EXTRACT TRAINING CLIPS** panel:

| Control | Shortcut | What it does |
|---|---|---|
| **Gesture** | — | What the next clip will be labelled as. Change it as often as you like — one take can yield clips of several gestures. |
| **Clip length (s)** | — | The default cut length. Same value as the **Default cut (s)** box on the main window. |
| **FIT LENGTH** | — | Resize the selection to exactly that length, keeping its left edge. |
| **\|< FIRST** | — | Jump to the start of the take. |
| **<< PREV / NEXT >>** | `Ctrl+Left` / `Ctrl+Right` | Slide the selection one clip length along, keeping its width. This is how you walk a long take. |
| **EXTRACT CLIP** | `Ctrl+Space` | Write the selection as a labelled training recording and step forward. |

So the rhythm is: **NEXT** until the selection sits on a gesture, drag the
edges to fit, pick the gesture, **EXTRACT CLIP**. Repeat. Each clip you write
is drawn back onto the graph as a coloured band in its gesture's colour, so
you can see at a glance which parts of the take you have already harvested,
and the list below records every one.

Press **DONE** when you are finished. The clips are added to the RECORDINGS
table, so they are immediately part of the session.

### What a clip is

An ordinary stage-`raw` recording in its gesture folder — the same files, the
same format, and the same `RETRAIN ON ALL DATA` button as everything else. Its
samples are **bit-exact copies** of the take's, and its metadata records where
it came from:

* `capture_style: "cut_from_take"` and `selected_by_operator: true`
* `parent_take_uid`, `parent_take_label`, `parent_take_file`
* `take_start_index` / `take_end_index` and `take_start_s` / `take_end_s`
* any marker that falls inside the clip, under `markers`

Each clip gets its **own** uid, not the take's, and its own repetition number
within its label — so cutting ten FIST clips gives you `rep001`…`rep010`. The
handedness, arm and device recorded on a clip come from the **take**, not from
today's config, because that is what was true when the signal was recorded.

**The take is never modified.** It is not rewritten, renamed or deleted when
you cut from it, so you can reopen it and cut more clips next week. Two of the
checks in `tests/test_free_take.py` do nothing but compare the take file's
bytes before and after extracting.

### Throwing a take away

Once you have cut what you want out of a 50 MB take — or recorded one you do
not want at all — select it in **TAKES ON DISK** and press
**DELETE TAKE(S)**. Ctrl/Shift+click to pick several.

Nothing is erased. The take's files move into `<session>/_deleted/`, the same
folder DELETE LAST and REDO use, and a line goes into `manifest.jsonl` saying
where they went. To get a take back, move its files from `_deleted/` to
`_takes/` — `_deleted` is an underscore folder, so a take sitting in it is
invisible to discovery either way and cannot leak into the dataset.

Every format the take was written in goes together: `.csv`, `.npz`,
`.parquet`, the `_meta.json` sidecar and any `_notes.json`.

**Clips you already cut are not deleted.** They are ordinary recordings in
their gesture folders and they stay in the dataset exactly as they are. The
confirmation dialog tells you how many there are before you commit, because
the one thing you lose is the ability to cut *more* clips from that take:

```
Move 1 take(s) out of _takes?

    FIST_take003_20261013-225031-124_91a797e8_take   (56 clip(s) cut from it)

The files move into the _deleted folder, not erased, so the signal can
always be recovered.

56 clip(s) were cut from these takes. Those clips are separate recordings
and are NOT deleted -- they stay in the dataset. You will not be able to
cut any more clips from these takes, though.
```

The button is greyed out while a free take is still recording — the file does
not exist yet — and when nothing is selected.

| Setting | Default | What it does |
|---|---|---|
| `FREE_TAKE_ENABLED` | `True` | Master switch. `False` hides the panel. |
| `FREE_TAKE_MAX_MINUTES` | `15.0` | Hard cap on one take. Samples are held in RAM until you stop, so this is a memory bound (~1 MB per 10 min). The recorder stops itself at the cap. |
| `FREE_TAKE_MIN_SECONDS` | `2.0` | Shorter than this and the take is discarded instead of saved. |
| `FREE_TAKE_FOLDER` | `_takes` | Folder inside the session. **Must start with `_`** or takes would enter the dataset. |
| `FREE_TAKE_WARN_FRACTION` | `0.85` | When the status line turns amber as the cap approaches. |

## MARKERS & NOTES — writing down what happened

A **marker** is a timestamped piece of text dropped onto the live stream while
you record. It is drawn on the graph as a vertical separator with its text, so
a long free-running take stops being an undifferentiated wall of signal.

### While recording

| Button | Shortcut | What it does |
|---|---|---|
| **ADD NOTE** | `Ctrl+M` | Type the text in the field first, then press it. The note lands at the instant you press, with the text drawn on the graph. |
| **BREAK** | `Ctrl+B` | A dashed separator line. Needs no text — use it where one block of work ends and the next begins. |
| **CUT SINCE LAST MARKER** | `Ctrl+Shift+Space` | Saves everything between the newest marker and now as a labelled recording, using the gesture chosen in the LABEL ON GRAPH panel above. |

So the mid-recording workflow you probably want is: drop a **BREAK**, do the
gesture, then **CUT SINCE LAST MARKER**. The break is the visible boundary and
the cut turns what follows it into training data.

The tick box **allow cutting at markers** decides whether cutting is offered
at all; untick it for a pure observation session where you want the
annotations but nothing cut from them. It mirrors `MARKER_CUT_ENABLED`, so you
can also set the default in the config before you start.

The marker's timestamp is the time of the newest **sample**, not of the GUI
clock, so the line sits where the signal actually was. Markers go to
`markers.jsonl` in the session folder, which is **append-only**: a marker is
never edited or deleted, and a correction is simply another marker. The panel
lists the ones logged this session, and they are reloaded if you restart the
tool mid-session. Switching subject swaps the log, so notes are never drawn
over — or cut from — somebody else's session.

Any marker that falls inside a saved span is copied into that recording's
metadata under `markers`, so the context travels with the data.

### Annotating a recording you already saved

Select it in the RECORDINGS list and press **TRIM** (or double-click it), or
use **OPEN FILE…** to pull up *any* recording on disk, including one from an
earlier session. The dialog shows:

* the full graph of the take, with any marker inside it drawn as a separator
* **NOTES ON THIS RECORDING** — every note already attached, plus a field to
  add another. A note is saved the moment you press ADD NOTE; you do not have
  to press Save, and pressing Cancel does not undo it.
* the usual start/end selection, so you can **cut a span out and save it** as
  a `*_trimmed.*` training file.

Notes are written to a `*_notes.json` sidecar next to the recording. The
signal file is never opened for writing — annotating a take leaves `*_raw.*`
byte-identical, which is checked in `tests/test_markers.py`.

| Setting | Default | What it does |
|---|---|---|
| `MARKERS_ENABLED` | `True` | Master switch. `False` hides the panel and the notes box in the dialog. |
| `MARKER_CUT_ENABLED` | `True` | Whether CUT SINCE LAST MARKER is offered. Mirrored by the tick box. |
| `MARKER_MAX_VISIBLE` | `12` | How many markers stay drawn (a drawing-cost cap only; the file keeps all of them). |
| `MARKER_LABEL_MAX_CHARS` | `28` | Where the caption on the graph is elided. The full text always goes to the file. |
| `MARKER_COLORS` | note/break | Colour per kind. |
| `MARKERS_NAME` | `markers.jsonl` | Name of the session marker log. |
| `NOTES_SIDECAR_SUFFIX` | `_notes.json` | Suffix of the per-recording notes sidecar. |

## The review step — you set the pace

This is on by default (`REVIEW_EACH_RECORDING = True`). After every
repetition:

```
 ... RECORD NOW ...  ->  CHECK: FIST
                         ┌──────────────────────────────────────────┐
                         │ CH1  ───╮▁▁▁▟█████▙▁▁▁╭───               │
                         │ CH2  ───╯   ╎     ╎   ╰───               │
                         │  …      ╔═══╪═════╪═══╗  <- green band:  │
                         │         ║   ╰ cut ╯   ║     what is kept │
                         └──────────────────────────────────────────┘
      FIST rep 3:  keeping 94 of 142 samples (66 %), 1.88 s  |  quality: ok

              [ACCEPT & NEXT]   [REDO]
```

1. **The graph freezes on the take you just did** — all 8 channels, scaled to
   that recording. The live stream keeps running underneath; nothing is lost.
2. **The green band is the automatic cut** — the part that actually matters.
   **Drag either edge** on any channel to adjust it; all 8 move together and
   the read-out underneath updates live.
3. **ACCEPT & NEXT** (or **Space** / **Enter**, since the button has focus, or
   **Ctrl+Enter**) keeps the take, writes the trimmed copy for the band you
   see, and starts the next repetition.
4. **REDO** (or **Ctrl+R**) rejects it: the files are **moved to `_deleted/`**
   (never erased) and the **same repetition is recorded again** — the plan
   still owes you that gesture, so your counts stay correct.

The raw file is written before the review and is never modified by it. If you
accept without moving the band and the band spans the whole recording, no
trimmed file is written — there would be nothing to trim.

To run straight through without stopping, set `REVIEW_EACH_RECORDING = False`.

## Buttons

| Button | What it does |
|---|---|
| **START SESSION** | Builds the plan (`GESTURES × REPETITIONS`, in `GESTURE_ORDER`) and runs the whole automatic sequence. |
| **ACCEPT & NEXT** | Review step only. Keeps the take, writes the trimmed copy, next repetition. `Space` / `Enter` / `Ctrl+Enter`. |
| **REDO** | Review step only. Rejects the take (moved to `_deleted/`) and repeats the same repetition. `Ctrl+R`. |
| **PAUSE / RESUME** | Pauses between repetitions. A recording already in progress is **finished and saved** first. |
| **STOP** | Aborts the session. Anything already saved stays. |
| **SKIP** | Drops the current repetition and moves on. Useful when the subject fumbles. |
| **RECORD (manual)** | Records the gesture chosen in *Manual gesture* until pressed again. For one-off samples. |
| **SAVE GRAPH** | Saves the current graph as PNG or SVG, captioned with person, session, gesture, mode, rate and timestamp. |
| **EXPORT DATASET** | Scans `datasets/`, builds windows + features for **every person**, writes an `.npz` bundle plus an index CSV and a summary JSON into `exports/`. Read-only. |
| **TRIM** | Opens the trim dialog for the selected recording (or double-click the row). |
| **AUTO-TRIM ALL** | Runs activity detection over every recording of this run and writes trimmed copies. Raw files untouched. |
| **DELETE LAST** | **Moves** the last recording into `_deleted/`. Nothing is erased. |
| **CLEAR LIST** | Empties the on-screen list and the graph buffer. Files on disk are kept. |

## Graph controls

| Control | What it does |
|---|---|
| **Window** | Visible time span, 0.25–60 s. |
| **Autoscale** | Per-channel vertical scaling. Grows instantly on a peak, shrinks slowly — a spike can never be clipped out of view and the axis does not jitter. |
| **Sensitivity** | Manual vertical scale. Only used when Autoscale is off. |
| **RMS envelope** | Overlays a dashed moving-RMS envelope — useful for judging where a gesture really starts. |
| **Freeze** | Stops the plot. **Recording and saving continue.** |
| **Reset view** | Undoes mouse zoom/pan. |
| **Channels 1–8** | Show/hide individual channel bands. |
| Mouse | Drag to pan, wheel to zoom, right-click for the PyQtGraph menu. |

## The trim dialog

```
|  useless  |      gesture      |  useless  |
            ^                   ^
          start                end
```

- Drag the blue region on any plot, or type exact seconds into **Start/End**.
- The bottom plot shows the RMS envelope with the detection thresholds drawn
  in, so you can see *why* auto-detect chose what it chose.
- A faint green band marks the window the app originally cued.
- **Auto-detect** re-runs activity detection. **Full range** selects everything.
  **Preview selection** zooms to it.
- **Save trimmed copy** writes `*_trimmed.csv` / `*_trimmed.npz` **next to**
  the raw file. The raw file is never modified.

## The channel readouts

Live `value / min / max / rms` per channel. Use these to check electrode
contact before recording: a channel stuck at a constant number has lost
contact, and a channel that is loud while you are relaxed has a noise problem.

---

# 4. Every configuration setting

All of these are at the top of `emg_dataset_tool.py`, in the
`USER CONFIGURATION` block. Edit, save, restart the app.

> The GUI mirrors edits to person, session, handedness, arm, device, notes,
> repetitions and order back into these values, so the metadata on disk always
> matches what actually happened.

## 4.1 Where the data goes

| Setting | Default | What it does |
|---|---|---|
| `DATA_FOLDER` | `"datasets"` | Root folder for all data. Relative to the script; an absolute path works too (e.g. a fast local disk instead of OneDrive). |
| `PERSON_ID` | `"person_01"` | Subject id. **Use a new one for every new person.** |
| `SESSION_ID` | `"session_01"` | **Start a new session every time the armband is taken off and put back on** — the electrodes land in different places and the signal changes. |
| `HANDEDNESS` | `"right"` | Stored as metadata. |
| `ARM_WORN_ON` | `"right"` | Which arm wears the band. Stored as metadata. |
| `DEVICE_NAME` | `"Myo armband"` | Free text; useful if you own more than one band. |
| `SUBJECT_NOTES` | `""` | Free text saved with every recording. Editable in the GUI. |

## 4.2 Gestures

| Setting | Default | What it does |
|---|---|---|
| `GESTURES` | 6 labels | **The label set.** Drives the GUI, the folders and the plan. Add a gesture by adding a string — nothing else needs changing. Use `UPPER_SNAKE_CASE`; the name becomes a folder name. |
| `REPETITIONS` | `30` | Repetitions **per gesture**. 6 × 30 = 180 recordings. |
| `GESTURE_ORDER` | `"interleaved"` | `"interleaved"` = one round of every gesture, then the next round (**recommended** — spreads fatigue and electrode drift across all labels). `"blocked"` = all 30 FIST, then all 30 OPEN (easiest for the subject, but bakes drift into whichever gesture came last). `"random"` = fully shuffled. |
| `RANDOM_SEED` | `42` | Makes the shuffling reproducible. |
| `MAX_RECORDINGS` | `0` | Hard cap on recordings per session. **`0` means unlimited**, not zero. |

## 4.3 The automatic sequence

```
REST ──> GET READY ──> 3 2 1 ──> RECORD NOW ──> STOP ──> pause ──> (next)
REST_TIME  PREPARE_TIME  COUNTDOWN_TIME  RECORD_TIME  POST…  PAUSE_TIME
```

| Setting | Default | What it does |
|---|---|---|
| `REST_TIME` | `2.0` s | Relax / return-to-neutral before each repetition. Raise it for tiring gestures. |
| `PREPARE_TIME` | `0.8` s | How long "GET READY: FIST" is shown. |
| `COUNTDOWN_TIME` | `3.0` s | Length of the countdown. `3.0` gives three ticks, `2.0` gives two. |
| `RECORD_TIME` | `2.0` s | **Length of the labeled recording.** 1.0–1.5 s for quick gestures, 2–3 s for sustained holds. |
| `PAUSE_TIME` | `1.0` s | Breather after each recording. |
| `PRE_GESTURE_MS` | `300` | Signal captured **before** the RECORD cue. This is why a fast gesture never loses its onset — the buffer is already running when the subject reacts. Raise to 400–500 for explosive gestures. |
| `POST_GESTURE_MS` | `300` | Signal captured **after** the window closes, so the release is kept. |
| `AUTO_SAVE` | `True` | Write each repetition to disk as it finishes. Leave this on. |
| `AUTO_ADVANCE` | `True` | Move to the next repetition automatically. `False` pauses after each one. **Ignored when `REVIEW_EACH_RECORDING` is on** — the review step decides. |
| `VIBRATE_CUES` | `True` | Short buzz on the armband at record start/stop — lets the subject keep their eyes off the screen. |

## 4.3a The review step

| Setting | Default | What it does |
|---|---|---|
| `REVIEW_EACH_RECORDING` | `True` | **After every repetition, stop and wait for you.** The graph freezes on the take with the proposed cut drawn over it; nothing advances until ACCEPT or REDO. `False` = run straight through the plan. |
| `REVIEW_SHOW_TRIM` | `True` | Draw the auto-detected region as a draggable green band. Drag either edge to change what gets kept. `False` shows the take without a band. |
| `REVIEW_SAVE_TRIM_ON_ACCEPT` | `True` | On ACCEPT, write `*_trimmed.*` for the band you see. Skipped when the band spans the whole recording (nothing to trim). The raw file is untouched either way. |
| `REVIEW_QUARANTINE_ON_REDO` | `True` | On REDO, **move** the rejected take to `_deleted/`. `False` leaves it in the dataset — only useful if you want to keep every attempt. |
| `COUNTDOWN_OVERLAY` | `"corner"` | Where the cue is drawn on the graph. **`"corner"`** = compact badge, top-right, **traces stay fully visible**. `"center"` = big centred digits (covers the traces). `"off"` = banner only. |

Session length ≈ `len(GESTURES) × REPETITIONS ×`
(`REST + PREPARE + COUNTDOWN + RECORD + POST + PAUSE`).
The GUI shows the estimate under *Recording plan*.

## 4.4 Acquisition

| Setting | Default | What it does |
|---|---|---|
| `MYO_PORT` | `None` | `None` auto-detects the dongle by USB id. Set `"COM3"` to force it. |
| `EMG_MODE` | `"filtered"` | **`"filtered"`** = the stream the known-good code uses: 8 × uint16, on-device filtered, **measured at 50 Hz on this armband**. **`"raw"`** = true unfiltered 200 Hz, 8 × int8 — much better for fast gestures and for CNN/LSTM, but the less-tested path (see `CLAUDE.md` §12). |
| `NOMINAL_SAMPLE_RATE` | 50 / 200 | A *hint* used to size buffers and sanity-check. The **measured** rate is what gets stored, and it is measured as samples-per-second over the recording's span — the dongle delivers packets in bursts, so per-interval estimates are meaningless. |
| `EMG_RANGE` | `(0, 2047)` / `(-128, 127)` | Valid amplitude range, used by the quality checks and the plot. The filtered ceiling is **measured, not documented**: a maximum-effort fist on this armband reached 1100–1444, so it is *not* clipped at 1023. Widen it further if you see `out_of_range` on a healthy-looking signal. |
| `SATURATION_FRACTION` | `0.98` | Fraction of the range above which a sample counts as clipped. |
| `EMG_BANDPASS_HZ` | `{"filtered": (None, 8.0), "raw": (20.0, 95.0)}` | `(high_pass_Hz, low_pass_Hz)` per mode; `None` disables that half. Used by the `"bandpass"` representation, which the sweep treats as one more candidate. **The `filtered` row has no high-pass on purpose.** The Myo already band-passes and rectifies on-device, so what you get in that mode is an envelope at ~50 Hz — Nyquist is ~25 Hz, which makes the textbook 20–450 Hz band unrepresentable, and a high-pass on an envelope deletes the DC *activation level* that carries most of the signal (measured twice: Fisher 25.5 → 15.1 and accuracy 43.2 % → 33.5 %). So `filtered` gets an 8 Hz smoother only. `raw` is a genuine 200 Hz bipolar stream with nothing applied, where 20–95 Hz is both standard and actually missing. A cutoff at or above Nyquist is **skipped with a message, never clamped** — clamping would apply a different filter and still call it by the name you asked for. See `CLAUDE.md` §8i. |
| `EMG_BANDPASS_ORDER` | `2` | Order per half. The filter is zero-phase (run forwards then backwards), which doubles the effective order to 4 and — more importantly — introduces no group delay. A causal filter would delay the envelope and shift gesture onsets relative to the label boundaries clips are cut on. |

## 4.5 Memory

| Setting | Default | What it does |
|---|---|---|
| `RING_BUFFER_SECONDS` | `60.0` | Length of the rolling display buffer. Fixed memory regardless of session length. |
| `MAX_RECORDING_SECONDS` | `120.0` | Safety cap on a single recording. Raise only if you deliberately record long takes. |

## 4.6 The graph

| Setting | Default | What it does |
|---|---|---|
| `GUI_FPS` | `60` | Graph refresh rate. **Drop to 30 if the UI feels heavy.** |
| `PLOT_WINDOW_S` | `5.0` | Starting visible time window. |
| `PLOT_AUTOSCALE` | `True` | Start with autoscale on. |
| `PLOT_Y_PADDING` | `1.15` | Headroom above the peak when autoscaling. |
| `PLOT_ANTIALIAS` | `False` | `True` looks slightly smoother and costs real performance. |
| `PLOT_USE_OPENGL` | `False` | Try `True` with a good GPU; it can also make things worse. |
| `PLOT_LINE_WIDTH` | `1` | **Anything above 1 is much slower in PyQtGraph.** |
| `SHOW_ENVELOPE` | `False` | Start with the RMS overlay on. |
| `ENVELOPE_MS` | `50.0` | Window for the envelope and the RMS readout. Shorter = more responsive to bursts. |
| `CHANNEL_COLORS` | 8 hex colours | Per-channel colours. |

## 4.7 Windowing (affects later ML, not recording)

| Setting | Default | What it does |
|---|---|---|
| `WINDOW_MS` | `200` | Analysis window length. 200 ms is the usual accuracy/latency compromise for real-time control. |
| `WINDOW_OVERLAP` | `0.5` | Overlap between windows. `0.5` = 50 %. More overlap = more training windows, but they are **not** independent — never split them randomly. |

## 4.7a Per-wearer calibration (affects later ML, not recording)

Nothing here changes a single recorded sample. See **CALIBRATE** above for
what it is and how to use it.

| Setting | Default | What it does |
|---|---|---|
| `CALIBRATION_ENABLED` | `True` | Whether the CALIBRATE button exists at all. |
| `CALIBRATION_MODE` | `"affine"` | `off` / `gain` / `per_channel` / `affine`. `affine` since 2026-10-16, because it measured **+5.3 points** on a wearer the model has never seen, with the calibration held out too. `off` plus a retrain reproduces every number this project reported before then. Changing it means the saved model no longer matches — retrain. |
| `CALIBRATION_REST_SECONDS` | `3.0` | Length of the relaxed capture. |
| `CALIBRATION_EFFORT_SECONDS` | `3.0` | Length of the one-squeeze capture on the QUICK path. |
| `CALIBRATION_GESTURE_SECONDS` | `2.5` | Length of each capture in the guided sweep. |
| `CALIBRATION_GESTURES` | `None` | Which gestures the sweep cues. `None` = every entry in `GESTURES` except REST, which is always captured first and separately because it sets the floor. A shorter, well-spread list is a valid trade. |
| `CALIBRATION_PREPARE_SECONDS` | `1.2` | The `GET READY: FIST` beat before each capture, so the countdown does not start while the hand is still moving. |
| `CALIBRATION_MIN_GESTURE_CONTRAST` | `2.0` | A sweep capture counts as evidence only if its peak channel reaches this multiple of rest. |
| `CALIBRATION_MIN_GESTURE_LEVEL` | `60.0` | …and this many counts in absolute terms. Both are needed: noise near rest inflates a percentile, so a 32-count shrug can read as 2.5× and sail through the ratio alone. |
| `CALIBRATION_STALE_SECONDS` | `2.0` | No sample for this long and the dialog treats the stream as down. Judged on the newest **sample**, not the link: a connected armband that has gone to sleep supplies none. |
| `CALIBRATION_REST_PERCENTILE` | `50.0` | Which percentile of the rest capture becomes the stored rest level. A median, so one twitch does not move it. |
| `CALIBRATION_EFFORT_PERCENTILE` | `95.0` | Same for the reference. Not the maximum, so one spike does not define the wearer's whole scale. |
| `CALIBRATION_TARGET_LEVEL` | `600.0` | What a reference contraction maps to. Not arbitrary: the level features have absolute terms, so calibrated signal is scaled back into the range the Myo delivers rather than into 0–1. |
| `CALIBRATION_MIN_SAMPLES` | `24` | Fewer samples than this in either capture and it is refused (about half a second at 50 Hz). |
| `CALIBRATION_MIN_CONTRAST` | `2.0` | The effort capture must be at least this many times the rest capture. |
| `CALIBRATION_MIN_REFERENCE` | `40.0` | …and at least this many counts. Both guard the same mistake: a "maximum effort" capture where nothing happened. |
| `CALIBRATION_DERIVE_REST_PERCENTILE` | `10.0` | Used by DERIVE FROM RECORDED DATA, where everything is pooled and the labels are ignored, so the quiet percentile has to stand in for rest. |
| `CALIBRATION_DERIVE_EFFORT_PERCENTILE` | `95.0` | Same, for the reference. |

## 4.8 Automatic trimming

Finds the gesture inside the recording so the idle padding can be cut away.

| Setting | Default | What it does |
|---|---|---|
| `AUTO_TRIM` | `True` | Run detection on every recording and store the suggestion in the metadata. |
| `AUTO_TRIM_WRITE_FILE` | `False` | `True` also writes the trimmed file immediately. `False` = suggest only; trim from the dialog or with AUTO-TRIM ALL. |
| `AUTO_TRIM_SKIP_LABELS` | `["REST"]` | Labels never auto-trimmed. REST has no activity to find, by definition. |
| `TRIM_ENVELOPE_MS` | `50.0` | Envelope window for detection. Keep it short so a 100 ms burst still makes a full-height peak. |
| `TRIM_BASELINE_MS` | `200.0` | How much of the start is used to estimate resting noise. |
| `TRIM_THRESHOLD_HIGH` | `3.0` | Multiple of baseline needed to **start** a gesture. **Raise if noise is being detected as gesture; lower if gentle gestures are missed.** |
| `TRIM_THRESHOLD_LOW` | `1.8` | Multiple of baseline below which the gesture **ends**. Keeping it under HIGH is the hysteresis that stops flickering. |
| `TRIM_ABS_FLOOR` | 12 / 3 | Minimum absolute threshold. Stops a dead-quiet baseline from making the detector hair-trigger. |
| `TRIM_MIN_ACTIVE_MS` | `120.0` | Bursts shorter than this are noise, not gestures. Lower it for genuinely explosive gestures. |
| `TRIM_MIN_INACTIVE_MS` | `250.0` | Quiet stretches shorter than this do **not** end a gesture. **This is what keeps a two-phase wave as one gesture.** Raise it if a gesture gets split and only the louder half survives; lower it to separate rapid repeats. |
| `PRE_PADDING_MS` | `100.0` | Kept before the detected start. |
| `POST_PADDING_MS` | `150.0` | Kept after the detected end. |
| `TRIM_MIN_ACTIVE_CHANNELS` | `2` | How many of the 8 channels must agree. `1` is sensitive but noisy; `3`–`4` is stricter. |

When several bursts are found, the **strongest** one wins (not the longest) —
so a 150 ms explosive contraction beats a second of slow drift.

## 4.9 Quality checks

Nothing is ever deleted by these. They raise a visible warning and write the
finding into the metadata; keeping or excluding the recording is your call.

| Setting | Default | What it does |
|---|---|---|
| `QC_ENABLED` | `True` | Master switch. |
| `QC_PAUSE_ON` | `"error"` | When to interrupt the session: `"never"` (log only), `"error"` (serious problems — disconnect, dead channel), `"warning"` (anything suspicious; thorough but slow). |
| `QC_MIN_SAMPLE_FRACTION` | `0.80` | Warn below this fraction of the expected sample count. |
| `QC_MAX_GAP_MS` | `120.0` | Warn on a gap longer than this between consecutive samples. |
| `QC_MAX_SATURATED_FRACTION` | `0.02` | Warn above this fraction of clipped samples. |
| `QC_FLAT_CHANNEL_TOLERANCE` | `1e-9` | A channel that never changes is reported as a dead electrode. |
| `QC_MAX_BASELINE_RMS` | 60 / 12 | Warn on a noisy resting baseline — bad skin contact or mains interference. |
| `QC_MIN_PEAK_AMPLITUDE` | 40 / 8 | Warn when a non-REST recording never gets loud: the gesture was probably missing. Also used to flag a REST recording that was not actually relaxed. |

When a check fires you get a dialog with **Repeat this gesture** (re-queues it
at the current position) or **Keep and continue**. The raw file is already on
disk either way.

## 4.10 File formats

| Setting | Default | What it does |
|---|---|---|
| `SAVE_CSV` | `True` | Human-inspectable, one row per sample. |
| `SAVE_NPZ` | `True` | Compact and fast; the preferred format for loading. **Keep at least one of CSV/NPZ on.** |
| `SAVE_PARQUET` | `False` | Needs `pyarrow`. Worth it only for very large datasets. |
| `SAVE_META_JSON` | `True` | The per-recording metadata sidecar. Leave on — this is where provenance lives. |
| `CSV_FLOAT_FORMAT` | `"%.6f"` | Float formatting in CSV. |
| `CSV_INCLUDE_LABEL_COLUMNS` | `True` | Repeat gesture/person/session/repetition in every row. Redundant, but it makes a single CSV self-describing and safe to `pd.concat`. |

## 4.11 Reconnection

| Setting | Default | What it does |
|---|---|---|
### Latency — if gestures show up seconds late

| Setting | Default | What it does |
|---|---|---|
| `MYO_BULK_READ` | `True` | **Drains the serial port in bulk instead of one byte at a time.** `myo_raw`'s reader calls `ser.read()` for a single byte and reconfigures the port once per packet; on Windows that only just keeps up, so any hiccup leaves a backlog the reader never clears — and everything you see is that far behind. Symptom: gestures appearing **~8 s late while the reported sample rate still looked correct**, because a constant delay does not change the spacing between samples. |
| `MYO_FLUSH_ON_START` | `True` | Throw away whatever was queued before streaming began, so a session starts at "now". |
| `MYO_MAX_BACKLOG_BYTES` | `4096` | Safety valve. If the queue exceeds this (~1.5 s), the buffer is flushed to restore responsiveness — but **only outside a recording**; during one it is just logged, because a gap in a recording is worse than latency and the quality checks will flag it. `0` disables. |

**The status bar shows the lag**: `lag 0.0s (0B)` in grey is healthy, amber
over 0.35 s, red over 1 s. That readout is the thing to watch — a rate
readout alone cannot reveal this problem.

### Keeping it connected

Three separate causes, one setting each. Together they turn a link that
dropped every couple of minutes into one that holds.

| Setting | Default | What it does |
|---|---|---|
| `MYO_NEVER_SLEEP` | `True` | **The big one.** The armband puts *itself* to sleep after a short idle period, and streaming EMG does **not** keep it awake. This sends `set_sleep_mode(never_sleep)` right after connecting. Costs battery — charge it between sessions. |
| `MYO_CONN_TIMEOUT_MS` | `3000` | BLE supervision timeout: how long the radio tolerates silence before declaring the link dead. The original value was **640 ms**, short enough that a hand over the dongle or a busy 2.4 GHz band drops it. Range 100–32000. |
| `MYO_CONN_INTERVAL_MS` | `7.5` | Connection interval. 7.5 ms is the Myo default; shorter means lower latency and more airtime. Rarely worth changing. |
| `MYO_FAST_RECONNECT` | `True` | Reconnect straight to the armband's known address instead of re-scanning. The scan only works while the armband is advertising — which it stops doing shortly after coming off — and it is the slow part: **~1 s with this on vs 5–10 s** with a scan. Falls back to a scan automatically if the direct attempt fails. |

If it still drops: charge it, move the dongle to a front USB port (not a hub),
keep it within a metre or two, and clean the skin — a loose or dry contact
makes the armband think it is not being worn.

### Recovering when it does drop

| Setting | Default | What it does |
|---|---|---|
| `AUTO_RECONNECT` | `True` | Reconnect automatically when the link drops. |
| `PAUSE_SESSION_ON_LINK_LOSS` | `True` | Pause a running session on disconnect and **resume automatically** once the armband is back. The repetition that was interrupted is **discarded and retried**, so a dropout does not quietly cost you a sample. Without this, repetitions would count down into a dead link and save empty recordings. |
| `SCAN_TIMEOUT_S` | `20.0` | Give up a BLE scan after this long and retry with a clear message. **The armband stops advertising a minute or two after you take it off**, and without this the app would sit forever on "Opening the Myo dongle..." with no explanation. `0` disables it. |
| `RECONNECT_DELAY_S` | `2.0` | Wait before the first retry. |
| `RECONNECT_MAX_DELAY_S` | `10.0` | Back-off ceiling. |
| `RECONNECT_MAX_ATTEMPTS` | `0` | `0` = retry forever. |
| `STALL_TIMEOUT_S` | `3.0` | No packet for this long while "connected" = treat the link as lost. This is what catches the armband falling asleep, which does not raise an error on its own. |

## 4.12 Live classifier

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_LIVE_PREDICTION` | `True` | Master switch for the classifier panel. Harmless when no model exists — the panel just says so. |
| `MODEL_PATH` | `"models/gesture_model.joblib"` | Where the model is loaded from. Written by `train_model.py`. |
| `PREDICTION_MODE` | `"one_shot"` | Startup default for the mode selector: `"one_shot"` or `"continuous"`. Switchable in the GUI. |
| `PREDICTION_SMOOTHING` | `5` | Majority vote over N consecutive windows in continuous mode. Higher = steadier but laggier. `1` disables it. |
| `PREDICTION_INTERVAL_MS` | `150` | How often continuous mode classifies. |
| `ONE_SHOT_DELAY_S` | `0.8` | Grace period after pressing PREDICT before the window is captured, so you have time to make the gesture. |
| `ONE_SHOT_HOLD_S` | `3.0` | How long a one-shot result stays on the graph. |
| `PREDICTION_MIN_CONFIDENCE` | `0.45` | Below this the result is shown as uncertain (`FIST?`, amber) instead of as a confident answer. `0` disables the distinction. |
| `PREDICTION_OVERLAY` | `True` | Also draw the prediction over the live graph. |

With no model the panel explains why and the buttons stay disabled.
**It never shows an invented prediction.** Model format and the route to a
wearer-independent model: `docs/ML_PIPELINE.md` §5b and §8.

---

# 5. Ready-made setups

Copy the lines into the config block.

### 5.1 Quick test that everything works (about 1 minute)

```python
GESTURES = ["REST", "FIST"]
REPETITIONS = 2
REST_TIME = 1.0
COUNTDOWN_TIME = 2.0
RECORD_TIME = 1.0
PAUSE_TIME = 0.5
```

### 5.2 Serious dataset from yourself (the default, ~45 min)

```python
PERSON_ID = "person_01"
SESSION_ID = "session_01"
GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]
REPETITIONS = 30
GESTURE_ORDER = "interleaved"
RECORD_TIME = 2.0
AUTO_TRIM = True
```

Do **2–3 sessions on different days**, each with a new `SESSION_ID`. Different
days and re-placements are what make a model survive contact with reality.

### 5.3 Fast / explosive / aggressive gestures

Short bursts, high amplitude, sudden transitions.

```python
EMG_MODE = "raw"            # 200 Hz -- a 100 ms burst is 20 samples, not 5
RECORD_TIME = 1.2           # short: the burst is over quickly
PRE_GESTURE_MS = 500        # catch an early reaction
POST_GESTURE_MS = 400       # catch the release
REST_TIME = 3.0             # explosive work is tiring

TRIM_ENVELOPE_MS = 30.0     # short window -> full-height peak for a short burst
TRIM_MIN_ACTIVE_MS = 60.0   # accept genuinely brief bursts
TRIM_MIN_INACTIVE_MS = 150.0
TRIM_THRESHOLD_HIGH = 4.0   # bursts are loud; be choosy
PRE_PADDING_MS = 150.0

WINDOW_MS = 150             # shorter windows react faster
SHOW_ENVELOPE = True
```

Note: at `EMG_MODE = "raw"` the data is signed int8 centred on zero, so the
graph shows ±. Also raise `QC_MIN_PEAK_AMPLITUDE`'s `"raw"` entry if
everything is being flagged as "no activity".

### 5.4 Slow, sustained holds (force / fatigue studies)

```python
RECORD_TIME = 4.0
REST_TIME = 4.0
PRE_GESTURE_MS = 200
POST_GESTURE_MS = 200
AUTO_TRIM = False           # the whole window IS the gesture
ENVELOPE_MS = 150.0
WINDOW_MS = 400
```

### 5.5 Recording another person

1. In the GUI set **Person id** to the next free id (`person_02`, …) and
   **Session id** to `session_01`.
2. Set **Handedness** / **Armband on** and put anything relevant in **Notes**
   (arm size, hairy skin, athlete, …).
3. Show them each gesture yourself first, then do a short warm-up run.
4. Watch the CHANNELS readouts before starting — all 8 must respond.

```python
# or in the config, before launching:
PERSON_ID = "person_02"
SESSION_ID = "session_01"
HANDEDNESS = "left"
ARM_WORN_ON = "left"
SUBJECT_NOTES = "slim forearm, band one notch tighter"
```

The folder layout keeps people apart automatically, which is what makes
leave-one-person-out validation possible later. **Never reuse a `person_id`
for a different person.**

### 5.6 Unattended / hands-free collection

Nobody watching the screen — the session must never stop and wait.

```python
REVIEW_EACH_RECORDING = False # <- important: do not wait for a verdict
VIBRATE_CUES = True           # the subject can keep their eyes shut
AUTO_SAVE = True
AUTO_ADVANCE = True
AUTO_TRIM = True
AUTO_TRIM_WRITE_FILE = True   # trim automatically since nobody will
QC_PAUSE_ON = "never"         # never block waiting for an operator
PAUSE_SESSION_ON_LINK_LOSS = True   # auto-pause and auto-resume
REST_TIME = 3.0
```

### 5.6b Supervised collection — you check and cut every take

The default, and the right setup when you are sitting there and want control
over what enters the dataset.

```python
REVIEW_EACH_RECORDING = True
REVIEW_SHOW_TRIM = True
REVIEW_SAVE_TRIM_ON_ACCEPT = True
REVIEW_QUARANTINE_ON_REDO = True
COUNTDOWN_OVERLAY = "corner"  # keep the traces visible during the countdown
AUTO_TRIM = True              # needed: this is what proposes the band
AUTO_TRIM_WRITE_FILE = False  # the review writes the trimmed copy instead
PAUSE_TIME = 0.3              # you control the pace, so keep the gap short
QC_PAUSE_ON = "never"         # the review shows quality inline; no popups
```

Workflow: perform the gesture → look at the frozen trace → drag the band if
the automatic cut is off → **Space** to accept, or **Ctrl+R** to redo it.

### 5.7 Adding a new gesture to an existing dataset

```python
GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT",
            "PINCH"]          # <- added
```

Then either record only the new label (choose it under *Manual gesture* and use
**RECORD (manual)**), or run a session with `GESTURES = ["PINCH"]` into a new
`SESSION_ID`. Existing files are untouched; the new folder appears by itself.
Filter at load time with `load_dataset(..., gestures=[...])`.

### 5.8 Weak PC / stuttering UI

```python
GUI_FPS = 30
PLOT_WINDOW_S = 3.0
PLOT_ANTIALIAS = False
PLOT_LINE_WIDTH = 1
SHOW_ENVELOPE = False
SAVE_PARQUET = False
```

Also hide a few channels with the checkboxes while recording — hidden bands
are not drawn at all.

### 5.9 Maximum paranoia about data loss

```python
SAVE_CSV = True
SAVE_NPZ = True
SAVE_META_JSON = True
AUTO_SAVE = True
AUTO_TRIM_WRITE_FILE = False      # keep raw as the only writer
QC_PAUSE_ON = "warning"           # inspect everything suspicious
DATA_FOLDER = r"D:\emg_datasets"  # a local disk, not a syncing cloud folder
```

A OneDrive folder (which is where this project lives) can lock files mid-write.
A local path is safer for long sessions; copy it to the cloud afterwards.

---

# 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `could not open port 'COM3': Access is denied` | Another program holds the port. Close `test_myo.py`, Myo Connect, or another copy of the tool. |
| `Myo dongle not found!` | Dongle unplugged or a driver issue. Check Device Manager for *Bluegiga Bluetooth Low Energy*, then set `MYO_PORT = "COM3"` explicitly. |
| Stuck on `scanning…` / `RECONNECTING` | The armband is asleep — it stops advertising a minute or two after you take it off. **Pick it up / flex / shake it**; it wakes on motion. After `SCAN_TIMEOUT_S` the app says so explicitly and retries by itself. Make sure it is charged. |
| `out_of_range` warnings | Your contractions exceed the configured `EMG_RANGE` ceiling. Raise it — the filtered ceiling is empirical (1100–1444 measured here) and varies by subject and placement. |
| `noisy_baseline` on most recordings | First check skin contact and tightness. If the signal looks genuinely fine, the threshold is simply tight for you: raise `QC_MAX_BASELINE_RMS` for your mode. It is placement- and subject-dependent, and it is only a warning — nothing is discarded. |
| Graph is flat, connection is green | The armband is on backwards or too loose. Logo up, pointing at the wrist; tighten one notch. Wipe the skin — dry skin is a common cause. |
| Only some channels move | Normal: different pods sit over different muscles. All 8 should respond to *something*. A channel stuck on one number is a dead electrode. |
| `MYO DISCONNECTED` every few minutes | Low battery, distance, or USB power saving. Charge it, move the dongle to a front port, avoid USB hubs. |
| "no EMG packet for 3.1 s" in the log | Stall detection working as designed — the armband went quiet. It reconnects by itself, and the interrupted repetition is retried. **An armband that is not being worn stops sending EMG after a few seconds**, so this is normal when bench-testing with the band on the desk. |
| `dropped` counter is not zero | Expected. The dongle hands over 2–4 buffered packets at once, which makes the interval bursty and the counter slightly pessimistic. A handful over thousands of samples is fine; a number near the sample count is not. |
| Rate readout shows ~5 Hz | Something reintroduced a timeout into the `myo.run()` call. See `CLAUDE.md` §3 — it must be called with no argument. |
| Many `gap` warnings | BLE packet loss. Move closer, reduce 2.4 GHz interference (Wi-Fi, other BT devices). |
| `noisy_baseline` warnings | Poor skin contact or mains hum. Clean the skin, tighten the band, move away from chargers and monitors; try unplugging the laptop. |
| `no_activity` on gestures you definitely did | The threshold is too high for you. Lower `QC_MIN_PEAK_AMPLITUDE` for your mode, or `TRIM_THRESHOLD_HIGH`. |
| Auto-trim cuts the gesture in half | Raise `TRIM_MIN_INACTIVE_MS` (bridges the dip between two phases). |
| Auto-trim keeps the idle part | Raise `TRIM_THRESHOLD_HIGH`, or `TRIM_MIN_ACTIVE_CHANNELS` to 3. |
| Auto-trim finds nothing | Lower `TRIM_THRESHOLD_HIGH` and the `TRIM_ABS_FLOOR` entry for your mode. The dialog's bottom plot shows the thresholds against the envelope — look there first. |
| UI stutters | See preset 5.8. |
| SVG export looks wrong in a browser | It is rendered by Qt's `QSvgGenerator`. Inkscape and Illustrator handle it; some browsers are fussy. Use PNG for sharing. |
| `EXPORT DATASET` says "no windows could be built" | `WINDOW_MS` is longer than your recordings. Lower it, or raise `RECORD_TIME` and record again. |
| Deleted something by accident | It is in `datasets/<person>/<session>/_deleted/`. Move it back into the gesture folder. Same for anything you rejected with REDO. |
| The session stops after every gesture and waits | That is the review step working as intended. Press **Space** to accept, **Ctrl+R** to redo. Set `REVIEW_EACH_RECORDING = False` to run straight through. |
| The countdown number hides the graph | Set `COUNTDOWN_OVERLAY = "corner"` (the default) or `"off"`. |
| No green band appears during the review | `AUTO_TRIM` must be `True` — it is what proposes the band. Also check `REVIEW_SHOW_TRIM`. If the detector found nothing the band covers the whole take, which is deliberate: it never drops data because detection failed. |
| `Space` does nothing in the review | The ACCEPT button lost focus (you clicked elsewhere). Click it, or use **Ctrl+Enter**. |
| REDO does not seem to lower my recording count | Correct — REDO does not advance the plan, so the same repetition is recorded again and the final counts per gesture stay exactly as planned. |
| Accuracy is ~99 % and you do not believe it | You are right not to. You probably split windows randomly. Read `docs/ML_PIPELINE.md` §5 and use `leave_one_person_out`. |
| Gestures appear seconds late | Watch the `lag` readout in the status bar. `MYO_BULK_READ` must be `True`; if lag still grows, lower `MYO_MAX_BACKLOG_BYTES` so the guard kicks in sooner. |
| Live prediction is wrong a lot | Check MODEL STATS first — if the cross-validated accuracy is low, the live behaviour is simply honest. Fix the data (see "If the accuracy is low" above), not the model. |
| Everything gets predicted as REST | Your contractions are too weak to separate from rest. This is the most common failure and it is a recording problem. |
| PREDICT button is greyed out | No model loaded. Run `python training/train_model.py`, then MODEL STATS → Reload model (or restart the app). |
| Retrain button greyed out | A recording session is running. Stop it first — training would saturate the CPU and disturb the live stream. |
| Did the retrain use my new recordings? | Yes, and all the old ones too. The confirmation dialog lists every subject and session it will use, and MODEL STATS shows the final count. |
| Prediction says a label but a different one has a higher bar | It cannot: the panel soft-votes the probability vectors, so the label shown is always the argmax of the bars shown. If you see this, it is a bug. |
| Predictions lag behind the gesture | Lower `PREDICTION_SMOOTHING` (fewer windows averaged) and/or `PREDICTION_INTERVAL_MS`. Smoothing trades latency for stability. |
| One-shot fires before you are ready | Raise `ONE_SHOT_DELAY_S`. |
| Not sure when to make the gesture | Make it as you press and **hold it until the answer appears** — the panel shows amber "DO THE GESTURE" then red "READING THE SIGNAL". There is no 3-2-1 countdown in the classifier; that belongs to recording. |
| One-shot keeps guessing REST | You probably relaxed before the red "READING THE SIGNAL" phase. Hold through the whole ~1.3 s. |

## Checking a dataset from the command line

```powershell
python emg_dataset_tool.py --summary
```

```python
import emg_dataset_tool as T
print(T.dataset_summary("datasets"))

recs = T.load_dataset("datasets", stage="prefer_trimmed")
for r in recs[:5]:
    print(r, r.fs, r.worst_issue_level, [i.code for i in r.issues])
```

## Golden rules

1. **New person → new `PERSON_ID`.** Never reuse one.
2. **Band off and on → new `SESSION_ID`.**
3. **`*_raw.*` files are never modified.** If something wants to change one,
   that is a bug.
4. **Nothing is deleted.** DELETE LAST moves to `_deleted/`.
5. **Bad recordings are flagged, not thrown away.** You decide at training
   time, with `skip_quality_errors=True`.
6. **Report the leave-one-person-out score**, not a random split.

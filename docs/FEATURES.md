# Features

One section per feature: what it does, how to turn it on and off, what the
default is and why, where it is in the code. The defaults table is written by
`tools/make_report.py` from `results/experiments.json`; the measurements behind
it are in [REPORT.md](REPORT.md).

Defaults overview:

<!-- TABLE: defaults -->

| setting | default | why |
|---|---|---|
| settings / gesture switches | all gestures on | no retrain either way (CLAUDE.md 7a) |
| model selection | off | the original deployed models stay the default; select from the zoo in SETTINGS -> Models |
| soft voting | off | only with two or more selected models; see the voting table in the report |
| augmentation (NN) | off | no augmentation passed the rule (>= 1 point and 2x the seed noise, better for >= 4 of 5 people, or the same on robustness) -- stays off, selectable in the settings |
| adaptation (NN) | off | did not pass the rule (+4.5 points, better for 4/5 people, no evidence) -- stays off, selectable in the settings |
| augmentation (ML) | off | no augmentation passed the rule (>= 1 point and 2x the seed noise, better for >= 4 of 5 people, or the same on robustness) -- stays off, selectable in the settings |
| adaptation (ML) | on | helps robustness: +4.3 points per window, better for 3/5 people |
| calibration question | on | asked for; turn off in General |
| retrain question | on | asked for; turn off in General |
| application icon | on | assets/icons/ |

---

## 1. Settings (gear icon, Ctrl+,) and the gesture switches

STATUS: implemented

**What it does.** The gear icon in the header opens SETTINGS. Its Gestures page
has two independent switches per gesture: **RECORD** (is it cued, offered in
label combos, calibrated) and **USE IN CLASSIFIER** (may the live ML model and
neural net answer with it). The second switch needs no retraining in either
direction: the gesture's probability is zeroed and the rest renormalised on the
model's output (CLAUDE.md section 7a).

**How to turn it on.** Click the gear or press `Ctrl+,`; untick USE IN
CLASSIFIER for a gesture.

**How to turn it off.** Tick the box again. This restores the original
behaviour bit for bit; the model files and the NN worker are never touched
(checked by `tests/test_class_mask.py`).

**Default.** All gestures on (a switch, not a measurement).

**Where it is in the code.** `emg_app/class_mask.py` (the mask),
`emg_app/gestures.py`, `emg_app/gui/dialogs/gesture_setup.py` (`GestureEditor`),
`cfg.CLASSIFIER_OFF`, saved in `<DATA_FOLDER>/gestures.json`.

## 2. Model selection with star markers

STATUS: implemented

**What it does.** SETTINGS -> Models lists every model of the model zoo
(`models/zoo/`), the ML classifiers and the neural nets separately, with its
accuracy per window, macro-F1, size on disk, parameters, latency and what it
was trained with. The **top 3 of each family** (by leave-one-person-out
accuracy per window, ties by macro-F1 -- never a training-set number) carry a
star, and so does the best soft-voting subset if its *nested* estimate beats
every single model. Stars are computed whenever the zoo is listed, so they are
recomputed after every training (training rewrites each model's `meta.json`).
The tooltip on a star gives the rank, the score and the size.

**How to turn it on.** SETTINGS -> Models -> tick "Use the models selected
below", then tick models (or press *Select starred models* / *Select best
combination*). Saved in `<DATA_FOLDER>/app_settings.json`.

**How to turn it off.** Untick the same box. The application then loads
`MODEL_PATH` / `NN_MODEL_PATH` exactly as before (checked bit for bit by
`tests/test_zoo.py` and `tests/test_app_settings.py`); your ticks stay saved, so
switching on again is one click.

**Default.** Off (the original deployed models). The zoo is trained with
`python training/train_zoo.py all`.

**Where it is in the code.** `emg_app/zoo.py`, `training/train_zoo.py`,
`training/zoo_roster.py`, `emg_app/gui/dialogs/settings_tabs.py`.

## 3. Soft-voting of selected models (ML, and separately NN)

STATUS: implemented

**What it does.** Select several ML models and the live classifier answers with
the **mean of their probabilities**; select several nets and the NN worker holds
them in one process and answers with the mean of their softmax outputs (each
net's own answer is also returned). Models that cannot be averaged -- different
labels, window, features, calibration mode or adaptation -- are **refused with
the reason**, in the dialog and again by the loader; they are never averaged
anyway. ML models and neural nets are never voted together (CLAUDE.md rule 13).

**How to turn it on.** Tick two or more models of one family.

**How to turn it off.** Tick one (or none: the original model).

**Default and why.** Off. The measured voting table in [REPORT.md](REPORT.md)
shows whether voting beats its members: the ET-500 + RF-500-leaf3 vote is within
the seed noise of its best member, and voting everything in the four-model spec
(which is what `gesture_model.joblib` currently is) is much worse, because an
unweighted vote gives weak members an equal say.

**Where it is in the code.** `MeanProba` / `LivePredictor._load_vote` in
`emg_app/prediction.py`, `training/nn_serve.py`, `NeuralEngine` in
`emg_app/neural.py`, `zoo.check_vote`.

## 4. Size of the selected models

STATUS: implemented

**What it does.** The Models tab shows each model's size on disk and parameter
count, and under each table the count, total size and total parameters of the
selection. (`size on disk` is read from the file, not from metadata.)

**How to turn it on / off.** Always shown with the Models tab.

**Where it is in the code.** `zoo.selection_summary`, `ModelsTab`.

## 5. Data augmentation

STATUS: implemented (per family, off unless measured to help)

**What it does.** Extra training windows made by perturbing real ones, on the
raw calibrated window and for training only: **gain** (overall and per channel),
**noise** (multiplicative jitter), **time warp** (speed and shift), **band
rotation** (a circular channel roll of one pod) and **dead electrode** (one
channel reads zero). Each is its own switch; none is applied when predicting.

**How to turn it on.** SETTINGS -> Training, per family (neural nets / ML
classifiers). The neural trainer takes `--augment gain,rotate`; the zoo trainer
`--augment-nn` / `--augment-ml`. A model already trained keeps what it learned;
train again.

**How to turn it off.** Untick; the trainer's code path is then the original
one (the test compares weights, fold accuracies and confusion matrices with the
previous commit's trainer).

**Default and why.** See the defaults table above and [REPORT.md](REPORT.md),
section 6: a switch is on only if it passed the rule (>= 1 point and 2x the seed
noise, better for >= 4 of 5 people, or the same on robustness).

**Where it is in the code.** `training/nn_augment.py`, `train_nn.py`
(`--augment`), `training/ml_cv.py` (`augmented_features`), `train_zoo.py`.

## 6. Adaptation to the person

STATUS: implemented (per family; only a model trained with it uses it)

**What it does.** Standardises each person's input with statistics from about
9 s of their own **unlabelled** signal (10 % of their windows in the
experiments; the signal captured while calibrating at run time). For the nets
that is per-channel log-statistics, for the ML models per-feature statistics. It
is different from the per-wearer *calibration*, which is unchanged. A model
trained with it stores global statistics as the fallback.

**How to turn it on.** SETTINGS -> Training: "Train with adaptation" (per
family) makes new models adapted; "Use adaptation live" decides whether an
adapted model uses the person's statistics at run time. The statistics come
from the signal stored beside each calibration (`adaptation_signal.npz`), so
calibrate once.

**How to turn it off.** Untick "Use adaptation live": adapted models then use
their stored global statistics, which the experiment shows costs nothing
measurable ([REPORT.md](REPORT.md), section 8). Untick "Train with adaptation"
to train plain models. A model trained without adaptation ignores the
statistics entirely.

**Default and why.** See the defaults table; the evidence is thin (five people,
gain not uniform) and the report says so.

**Where it is in the code.** `emg_app/adaptation.py`, `LivePredictor.set_adaptation`,
`NeuralEngine.set_adaptation` + the worker's `adapt` command,
`PromptsMixin.update_adaptation`.

## 7. Calibration question before PREDICT / START NN

STATUS: implemented

**What it does.** Before PREDICT or START NN the application asks whether to
calibrate the hand: **CALIBRATE NOW**, **USE LAST CALIBRATION**, Skip, Cancel.
"Use last calibration" is offered only when the stored calibration was made
*before this run of the application started* (i.e. the application was
restarted), is under 8 h old, and the question always says the band must have
stayed on the arm; a live check of the signal at rest against the calibration
is shown, never used to refuse. Asked once per run per person.

**How to turn it on / off.** SETTINGS -> General -> "Before PREDICT / START NN,
ask whether to calibrate the hand". Off restores the old behaviour: no question
(a model that needs a calibration still refuses to run without one,
CLAUDE.md section 12).

**Default.** On (it was asked for).

**Where it is in the code.** `emg_app/prompts.py` (decision),
`emg_app/gui/main_window/prompts.py` (`calibration_gate`).

## 8. Retrain question on new gestures or new data

STATUS: implemented

**What it does.** At startup, when a model is out of date or a recorded gesture
is unknown to it, asks: **Retrain ALL models** (the deployed classifier and net,
then every zoo model) / **Retrain only the selected** (the selected zoo models;
with selection off, the deployed classifier and net) / Not now. Nothing retrains
automatically. Never asked while a training or a session runs.

**How to turn it on / off.** SETTINGS -> General -> "When new gestures or new
recordings make models out of date, ask ...", or the checkbox in the question
itself ("Do not ask when the application opens"). Off restores the old
behaviour: the panels still say what is out of date.

**Default.** On.

**Where it is in the code.** `PromptsMixin.maybe_ask_retrain`, `retrain_zoo`
and the `zoo` step of `retrain_everything` in
`emg_app/gui/main_window/training.py`.

## 9. Application icon

STATUS: implemented

**What it does.** The window, dialog and taskbar icon (a band ring with an EMG
trace). Generated by `tools/make_icons.py` into `assets/icons/` at 16, 32, 48,
64, 128 and 256 px (PNG) and `app.ico`. A missing or unreadable file gives the
default icon and never an error.

**How to turn it off.** SETTINGS -> General -> Appearance -> untick "Show the
application icon" (takes effect at the next start), or remove `assets/icons/`;
nothing else changes.

**Where it is in the code.** `emg_app/gui/icon.py` (`apply_app_icon`, called
from `emg_app/app.py`), `tests/test_icon.py`.

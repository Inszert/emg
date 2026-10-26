# Features

One section per feature. `STATUS` is `implemented` or `planned`; main flips it
when the feature lands. Cells marked `TODO(main)` are filled from the ablation
(`results/experiments.json`, table `defaults`).

Defaults overview:

<!-- TABLE: defaults -->

| setting | default | reason | delta_acc |
|---|---|---|---|
| settings dialog / gesture switches | TODO(main) | TODO(main) | TODO(main) |
| model selection (stars) | TODO(main) | TODO(main) | TODO(main) |
| soft-voting ensemble | TODO(main) | TODO(main) | TODO(main) |
| size display | TODO(main) | TODO(main) | TODO(main) |
| data augmentation | TODO(main) | TODO(main) | TODO(main) |
| adaptation to the person | TODO(main) | TODO(main) | TODO(main) |
| calibration prompt | TODO(main) | TODO(main) | TODO(main) |
| retrain question | TODO(main) | TODO(main) | TODO(main) |
| app icon | TODO(main) | TODO(main) | TODO(main) |

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

**Default and why.** All gestures on. TODO(main): from the ablation.

**Where it is in the code.** `emg_app/class_mask.py` (the mask),
`emg_app/gestures.py`, `emg_app/gui/dialogs/gesture_setup.py` (`GestureEditor`),
`cfg.CLASSIFIER_OFF`, saved in `<DATA_FOLDER>/gestures.json`.

## 2. Model selection with star (*) markers

STATUS: planned

**What it does.** After every training the app automatically marks with a star
(*) the top 3 NN and top 3 ML models, ranked by test accuracy then macro-F1,
plus the best soft-voting ensemble if it beats every single model. The list is
recomputed after every training. A tooltip shows the score and the size.

**How to turn it on.** TODO(main): setting name.

**How to turn it off.** TODO(main). Off restores the current behaviour: one
deployed ML model and one NN, with no stars.

**Default and why.** TODO(main): from the ablation.

**Where it is in the code.** TODO(main).

## 3. Soft-voting ensemble of selected models (NN and ML)

STATUS: planned

**What it does.** Averages the probabilities of the selected models (ML and NN
together). Today the ML vote is two forests and the NN is never averaged in
(CLAUDE.md safety rule 13; `docs/MODELS_COMPARED.md`). A mixed vote is only
offered as a separate, separately cross-validated model, never as a silent
blend.

**How to turn it on.** TODO(main).

**How to turn it off.** TODO(main). Off restores the two independent answers.

**Default and why.** TODO(main): from the ablation (`ablation_voting`).

**Where it is in the code.** Existing ML vote: `training/train_model.py`,
`emg_app/model_spec.py`. Mixed vote: TODO(main).

## 4. Size display of selected models

STATUS: planned

**What it does.** Shows each selected model's size on disk and its parameter
count (feeds table `model_sizes`).

**How to turn it on.** TODO(main).

**How to turn it off.** TODO(main). Off restores the current panels.

**Default and why.** TODO(main): from the ablation.

**Where it is in the code.** TODO(main). Sizes already measured in
`docs/MODELS_COMPARED.md`.

## 5. Data augmentation

STATUS: planned (on/off)

**What it does.** Extra training windows made by perturbing real ones.
TODO(main): list the augmentations used.

**How to turn it on.** TODO(main): setting / trainer flag.

**How to turn it off.** TODO(main). Off reproduces the current training exactly.

**Default and why.** TODO(main): from the ablation (`ablation_nn`, `ablation_ml`).

**Where it is in the code.** TODO(main).

## 6. Adaptation to the person

STATUS: planned (on/off)

**What it does.** Adapts a model to the current wearer. TODO(main): method.

**How to turn it on.** TODO(main).

**How to turn it off.** TODO(main). Off reproduces the current per-wearer
calibration-only behaviour.

**Default and why.** TODO(main): from the experiment (table `adaptation`).

**Where it is in the code.** TODO(main).

## 7. Calibration prompt before the predictor starts

STATUS: planned

**What it does.** Before the predictor starts, the app asks the wearer to
calibrate. "Use last calibration" is offered only if the app was restarted
but the band stayed on the arm; otherwise a fresh calibration is required.
Today a model trained with calibration refuses to run for an uncalibrated
wearer (CLAUDE.md section 12).

**How to turn it on.** TODO(main).

**How to turn it off.** TODO(main). Off restores the current behaviour: the
refusal message.

**Default and why.** TODO(main).

**Where it is in the code.** TODO(main). Calibration itself: `emg_app/calibration.py`,
`CalibrationDialog`.

## 8. Retrain question on new gestures or new data

STATUS: planned

**What it does.** When new gestures or new training data appear, asks whether
to retrain: all models or only the selected ones. A setting shows or hides the
question at every start.

**How to turn it on.** TODO(main): setting name.

**How to turn it off.** TODO(main). Off restores the current behaviour:
nothing retrains automatically (CLAUDE.md section 7a); the panel only says the
models do not know the data yet.

**Default and why.** TODO(main).

**Where it is in the code.** TODO(main). Existing hints: `model_gesture_gaps()`,
`model_data_status()`.

## 9. App icon

STATUS: planned

**What it does.** Window and taskbar icon. Files: `assets/icons/`
(`icon_128.png`).

**How to turn it on / off.** TODO(main). Off restores the default Qt icon.

**Default and why.** On. TODO(main).

**Where it is in the code.** `emg_app/gui/icon.py`, `tools/make_icons.py` (Agent C).

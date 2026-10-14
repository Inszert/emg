# ML pipeline

From recorded EMG to a model whose reported accuracy you can believe.

The tool does not train anything. It provides the pieces and, more importantly,
the `groups` array that keeps validation honest.

---

## 1. The intended path

```
record  ->  validate  ->  trim  ->  window  ->  features  ->  train
                                                                |
                                            validate (leave-one-person-out)
                                                                |
                                                  test on a fully unseen person
                                                                |
                                                        live prediction
```

Everything up to and including `features` is implemented. Training is yours to
write; the loaders are built so that it is a short script.

## 2. Loading

```python
import emg_dataset_tool as T

recs = T.load_dataset(
    "datasets",
    stage="prefer_trimmed",   # trimmed where available, else raw
    persons=None,             # e.g. ["person_01", "person_02"]
    gestures=None,            # e.g. ["REST", "FIST", "OPEN"]
    skip_quality_errors=True, # leave out recordings flagged with an error
)
print(len(recs), recs[0])
```

`stage` options: `"raw"`, `"trimmed"`, `"processed"`, `"augmented"`,
`"prefer_trimmed"`.

A `Recording` gives you `.emg` (n, 8) int32 raw counts, `.timestamps`, `.fs`
(measured), `.label`, `.person_id`, `.session_id`, `.repetition`,
`.label_start_index` / `.label_end_index`, `.issues` and `.meta`.

Derived views are computed on demand and never overwrite the raw array:

```python
rec.envelope()               # moving-RMS linear envelope
rec.normalized("mad")        # (array, scaling params) -- reversible
rec.detect_activity()        # ActivityResult
```

## 3. Windowing

```python
X, y, groups, meta = T.make_windows(
    recs,
    window_ms=200,            # default WINDOW_MS
    overlap=0.5,              # default WINDOW_OVERLAP
    use_label_window=True,    # cue window only, drop the pre/post roll
    representation="raw",     # "raw" | "normalized" | "envelope"
)
# X      (n_windows, window_samples, 8) float32
# y      (n_windows,) int  -> index into meta["labels"]
# groups (n_windows,) str  -> person_id
```

A trailing partial window is **dropped, not zero-padded** — padding teaches a
model to recognise padding.

`X` is already the right shape for a CNN (`Conv1d` over time, 8 input
channels) or an LSTM/Transformer (sequence of 8-dim vectors). Flatten it for
an MLP.

## 3a. Which features — this matters more than the model

**The Myo's `filtered` stream is an already-rectified envelope, not a
zero-centred oscillation.** `extract_features()` subtracts each channel's
median before computing its shape features. That is correct for raw bipolar
EMG and *wrong* here: it deletes the activation level, which is exactly what
identifies a gesture.

Measured on this project's dataset (between-class / within-class Fisher ratio,
higher is better):

| What the model sees | Fisher ratio |
|---|---|
| per-channel mean (level kept) | **25.5** |
| per-channel `share` (profile, amplitude divided out) | 25.2 |
| median-centred RMS (`extract_features` default) | 15.1 |

Switching from `classic` to `level` features took per-window accuracy from
33.5 % to 43.2 % *and* cut the feature count from 104 to 50.

So there are three families:

| `feature_set` | Function | Use for |
|---|---|---|
| `classic` | `extract_features()` | true raw EMG (`EMG_MODE = "raw"`, int8, zero-centred) |
| `level` | `extract_level_features()` | the rectified `filtered` stream — **the default** |
| `both` | `extract_all_features()` | when you have enough data to let the model choose |

`extract_level_features()` returns, per channel: `mean`, `median`, `p90`,
`std`, `share`, `log_mean`, plus two globals (`total_log`, `focality`).

### `share` is the cross-subject feature

`share` is each channel's fraction of the total activation — the spatial
pattern with absolute amplitude divided out. It survives a harder contraction,
a tighter band and different skin impedance, so it is the part that should
transfer to another wearer. `mean` is the part that will not.

### A warning about baselines

Subtracting a resting baseline looks obviously right and **made things worse**
here (Fisher ratio 24.9 → 11.5 with a pre-roll baseline, 14.9 with a p10
baseline). Within one session the resting offsets are constant, so they add no
within-class variance, and a per-recording estimate partly cancels the gesture
itself. Leave levels absolute for a single-subject model; handle offset and
gain differences in the per-user calibration step instead.

## 3b. Rate independence

The filtered stream is **not evenly sampled**: the dongle delivers samples in
pairs ~32 ms apart, and the effective rate across this dataset ranges from
**28 Hz to 70 Hz** (median 63). A model trained on "13 samples" would see
windows covering 185–460 ms of real time.

`train_model.py` therefore resamples everything onto a fixed grid
(`MODEL_FS = 60 Hz`) with `resample_uniform()`, and `LivePredictor` does the
same to the live window. A window means a fixed *duration* on both sides.

## 4. Features (for Random Forest / SVM / XGBoost)

```python
feats, names = T.make_feature_matrix(X, meta["fs"])   # (n_windows, 104)
```

13 features × 8 channels. Per channel, computed after removing the channel's
own median (so a DC offset is not mistaken for signal) while amplitude
features stay on the original scale:

| Feature | Meaning |
|---|---|
| `rms` | root mean square — overall activation |
| `mav` | mean absolute value |
| `var`, `std` | variance, standard deviation |
| `wl` | waveform length — sum of absolute differences |
| `zc` | zero crossings (with a dead-band so noise does not inflate it) |
| `ssc` | slope sign changes |
| `peak` | maximum absolute deviation |
| `iemg` | integrated EMG |
| `ptp` | peak-to-peak of the original signal |
| `skew` | skewness |
| `mnf` | mean frequency (from the periodogram) |
| `mdf` | median frequency |

`rms`/`mav`/`wl`/`zc`/`ssc` are the classic Hudgins set and are a strong
baseline. `mnf`/`mdf` are the usual fatigue indicators — useful if you want to
study drift across a long session.

For a single window: `T.extract_features(window, fs)`, names from
`T.feature_names()`.

## 5. Validation — the part that is easy to get wrong

### Do not do this

```python
#  WRONG
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.2, shuffle=True)
```

Two independent leaks make this meaningless:

1. **Window overlap.** At 50 % overlap, adjacent windows share half their
   samples. A random split puts near-duplicates on both sides.
2. **Subject identity.** EMG is dominated by anatomy and electrode placement.
   A model can score ~99 % by recognising *the person* and then guessing their
   most likely gesture — and collapse on anybody new.

### Do this instead

```python
for train, test, held_out in T.leave_one_person_out(groups):
    model.fit(feats[train], y[train])
    score = model.score(feats[test], y[test])
    print(f"held out {held_out}: {score:.3f}")
```

Or an explicit split — train on persons 1–3, test on person 4:

```python
train, test = T.split_by_person(groups, ["person_04"])
```

Or with scikit-learn directly:

```python
from sklearn.model_selection import GroupKFold, cross_val_score
cross_val_score(model, feats, y, groups=groups, cv=GroupKFold(n_splits=4))
```

**Report the leave-one-person-out number.** That is the one that predicts what
happens when somebody new puts the armband on. Expect it to be
*substantially* lower than a within-subject score — that gap is the honest
measurement, not a bug.

### Session-independent validation

Even for one person, holding out a whole **session** tests robustness to
electrode shift, which is the second-biggest failure mode after subject
identity:

```python
sessions = np.array([r["session_id"] for r in meta["records"]])
train, test = np.flatnonzero(sessions != "session_03"), np.flatnonzero(sessions == "session_03")
```

## 5a. Just run the trainer

`train_model.py` does all of the above and picks the winner for you.

```powershell
python train_model.py              # full two-stage sweep, saves the best model
python train_model.py --quick      # fewer combinations (a minute or two)
python train_model.py --no-save    # evaluate only
python train_model.py --person Dano
```

It sweeps window length, overlap, representation, **feature set**, frequency
features, quality filtering and 21 classifiers. Stage A finds the best
preprocessing with three fast probe models; stage B then runs every model on
the best setups, because the full cartesian product would take hours.

Output:

- `models/gesture_model.joblib` — the model the app loads
- `models/gesture_model_<timestamp>.joblib` — an archived copy
- `models/reports/sweep_<timestamp>.json` — every combination's score

It prints a leaderboard, the best result per model, the marginal effect of
each choice, a confusion matrix, and the per-recording vote accuracy.

### What the literature says to expect

Published sEMG gesture work reports 1D-CNN and CNN-LSTM around 96–97 %, a
Convolutional Vision Transformer at 96.6 % on Myo armband data vs 94.2 % for a
CNN, and LDA/SVM at 98 %+ within-subject. Those come from much larger,
multi-subject datasets at 200 Hz and above. At 72 recordings of a ~60 Hz
envelope stream, deep architectures have nothing to work with — which is why
`build_models()` is feature-based classifiers plus a small MLP, and why LDA is
listed first: no hyperparameters to overfit, trains in milliseconds, and it is
the standard choice for per-user recalibration.

Sources: [1D Dilated CNN / LDA / SVM comparison](https://soar.wichita.edu/server/api/core/bitstreams/22e6d0c5-d4b9-4eb6-981d-15de7df98e53/content), [CNN feature selection for sEMG](https://pmc.ncbi.nlm.nih.gov/articles/PMC9269838/), [Hybrid Convolutional Vision Transformer vs CNNs on Myo data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12730683/), [multi-stream time-varying feature enhancement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11436881/).

## 5a-ter. Every model records what it was trained on

Both trainers write a `data` block into the payload — the dataset as it stood
when training ran:

```python
"data": {"n_recordings": 119, "labels": {"FIST": 56, "REST": 63},
         "subjects": ["Dano"], "sessions": ["Dano/session_01"],
         "newest_stamp": "20261013-225241-161",
         "uid_digest": "2868d8e7e91543b6"}
```

`T.dataset_fingerprint(root)` builds the same structure from disk and
`T.model_data_status(payload, fingerprint)` compares them, returning
`untrained` / `unknown` / `current` / `stale`. The app shows that in the
LIVE CLASSIFIER panel and in MODEL STATS; a script can ask the same question:

```python
import emg_dataset_tool as T, joblib
payload = joblib.load("models/gesture_model.joblib")
print(T.model_data_status(payload, T.dataset_fingerprint("datasets")))
```

The digest is over the sorted recording uids, so a deletion plus an addition
cannot cancel out into a false "up to date". A payload with no `data` block
is reported as `unknown`, never as current — the comparison is only as honest
as the thing it compares against.

## 5a-bis. The deep path on the GPU — `train_torch.py` (RETIRED)

> **Retired 2026-10-14.** Kept below as a recorded negative result. The
> project trains only the traditional classifiers in `model_spec.json`; the
> application never imported torch and never read this trainer's output.
>
> It was retired because it lost. On the FIST/REST dataset the CNN-LSTM
> managed 99.2 % per recording against **100 %** for a plain LDA-shrinkage,
> on a task a single amplitude threshold already solves 98.3 % of the time.
> A deep model that is beaten by an LDA with no hyperparameters, on a task
> that barely needs a model, is not an argument for deep models — and it
> added a ~200 MB dependency to a classifier headed for an SBC on a robot.
>
> The honest lesson is in §5a-quater: this dataset cannot rank models at all.
> The question is worth reopening only when there are several *active*
> gestures to tell apart.

## 5a-quater. Why training is CPU-only, and what made it fast

**scikit-learn has no GPU backend.** The only GPU implementation of these
algorithms is RAPIDS cuML, and it has no Windows wheel and none for Python
3.14 (every wheel is `manylinux`, cp310–cp312). Using it would mean moving
the project into WSL2. The GPU itself is not the limitation.

Nor would it help: the largest setup in the sweep is ~6300 windows × 154
features, and GPU implementations of SVM and forests start paying off around
10⁵ rows. Below that, kernel-launch and transfer overhead dominate.

What did work, all measured on this machine and all result-neutral:

| Change | Effect |
|---|---|
| `SVC(probability=False)` while scoring | **5.7×** — 54.26 s → 9.52 s, `predict()` identical |
| CV folds on a thread pool | 1.3–2.9× on single-threaded estimators |
| Rebuilding setups instead of caching all 432 | memory stops growing without bound |
| `SVC(cache_size=1000)` | **nothing** — 7.22 s vs 7.21 s. Measured, then dropped |

The SVM one matters most and is worth understanding. `probability=True` makes
libsvm fit a Platt sigmoid via its own internal 5-fold CV. `evaluate()` only
calls `predict()`, which comes from the decision function — so that work
cannot change any score. The **saved** model is built separately with
probability, because `LivePredictor` and the soft vote need `predict_proba`.

The trap: soft voting picks members by `hasattr(clf, "predict_proba")`, and a
scoring-built SVC does not have it. Sharing those pipelines with the ensemble
would have silently dropped the SVM from the vote. The ensemble is therefore
built from its own always-probability copies, and
`tests/test_model_spec.py` pins that.

## 5a-quinquies. Seeing every candidate decide — the decision panel

The sweep scores many candidates and saves one. The panel is how you look at
the rest of them afterwards, per window, instead of only at a table of means.

`train_model.build_decision_panel()` writes `models/gesture_panel.joblib`
after the model is saved (skip it with `--no-panel`;
`python explain_votes.py` rebuilds it for a model that already exists). It
holds two different things, and the distinction is the whole point:

```python
panel["members"]          # each candidate fitted on the WHOLE dataset
panel["trail"]["proba"]   # out-of-fold probabilities, per window, per candidate
panel["cv"]               # out-of-fold scores, all on the WINNER's setup
panel["fusion"]           # the vote rule, and the measured check of it
```

**`members` is for demonstration, `trail` is for evidence.** The members run
on live windows in MODEL STATS, so you can make a gesture and watch seven
classifiers disagree about it. They were fitted on everything, so their
agreement is not a performance estimate and the GUI says so on every frame.
The `trail` is the honest half: each probability comes from a fold that never
saw the window's recording.

**All candidates are scored on the winner's setup**, not on their own best.
A sweep row lets each model pick its own window length, overlap and feature
set, which is right for choosing a winner and wrong for comparing opinions
about one window. On the current dataset the same model reads 99.15 % here
and 99.8 % in `best_per_model`; neither is wrong, they answer different
questions.

**The fusion rule is verified, not asserted.** Soft voting is the unweighted
mean of the members' `predict_proba`. The builder computes that mean itself
and compares it with the fitted `VotingClassifier` across the whole dataset:

```
soft vote vs mean of members: max|diff| = 1.19e-07     (float32 rounding)
```

Because that holds, the live view computes the mean from the members it has
already run instead of calling the ensemble, which would re-run all seven a
second time. When the check fails, the view falls back to the ensemble's own
recorded output rather than trusting the mean.

**It is a side file, never the model.** `gesture_model.joblib` keeps holding
exactly one estimator — that is what goes to the SBC — and `LivePredictor`
never opens the panel. The live trail is fed
`LivePredictor.last_features`, the same row the deployed model was given, so
no second feature path exists to drift from `model_feature_vector()`.

## The retired deep path, for the record

A second trainer, not a replacement for `train_model.py`. Same data layer,
same validation protocol, different model family: it feeds the **windowed
time series** `(win, 8)` to a neural net rather than the hand-crafted feature
vector.

```powershell
python train_torch.py                     # CUDA, 3 architectures x 3 windows
python train_torch.py --quick             # one model, one window
python train_torch.py --device cpu        # no GPU (slow)
python train_torch.py --kind cnn_lstm --window 0.5 --epochs 80
```

Architectures: `cnn` (3 conv layers → global average pool), `lstm` (2 stacked
layers, last time step), `cnn_lstm` (conv front end compresses time, one LSTM
reads the sequence). All about 25–30 k parameters — small on purpose at this
dataset size.

### What is held identical to the sklearn sweep, and why

| Shared | Why it must not drift |
|---|---|
| `MODEL_FS = 60 Hz` resampling | the stream is 28–70 Hz and uneven; a fixed sample count would mean a varying real duration |
| `label_start_index:label_end_index` | the capture window is deliberately wider than the labeled one |
| `StratifiedGroupKFold` on **recording uid** | windows overlap 75 %; a random split tests on near-copies of the training rows |
| leave-one-subject-out | the only number that did not pick its own folds |
| per-window **and** per-recording vote | the second is what a live session actually does |

### What differs, deliberately

- **No `model_feature_vector()`.** That function is the single contract
  binding `train_model.py` to `LivePredictor`; a conv net learns its own
  features and must not be spliced into that path.
- **`log1p` + per-channel standardisation, fitted on the training fold
  only.** `log1p` because the stream spans baseline ~15 to max-effort ~1444,
  and on a linear scale the strong contractions own the gradient. Absolute
  level is kept — §3a records that removing it measurably hurt.
- **Fixed epochs, no early stopping.** Stopping when the held-out fold looks
  best is fitting the test set.
- **Its own output file**, `models/gesture_model_torch.pt`, carrying the
  state dict, the labels, the window geometry **and the scaler** — inference
  that does not reproduce the scaler is feeding the model noise. The app does
  not load this file; the LIVE CLASSIFIER still uses the joblib payload.

### The device contract

`--device cuda` is the default and it **fails loudly**: if CUDA is not usable
the script exits with the install command instead of falling back to the CPU,
because a silent fallback is indistinguishable from a working run except that
it takes twenty times longer. Each run is bracketed by a `COMPUTE: >>> GPU
<<<` / `>>> CPU <<<` banner, and the device description is written into both
the checkpoint and the sweep report.

Measured here: NVIDIA GeForce GTX 1650 (4 GB, capability 7.5), torch
2.9.1+cu128, CUDA 12.8. The full 3x3 grid at 60 epochs takes about nine
minutes on the 119-recording dataset.

### Reading its numbers honestly

On the current FIST/REST data the CNN-LSTM reports **99.7 % per window /
99.2 % per recording**. The grouped CV behind that is sound, and the number
is still nearly worthless as evidence:

| Model | Per-recording |
|---|---|
| CNN-LSTM (this trainer) | 99.2 % |
| LDA on the feature vector | 100 % |
| **One threshold on mean amplitude** | **98.3 %** |
| Majority class | 52.9 % |

FIST vs REST asks "is the muscle contracting at all", the easiest question on
EMG — REST median amplitude 23.7 against FIST's 108.8, nearly disjoint. The
deep stack buys about one recording out of 119 over `if amplitude > 53.5`.
The hard problem is discriminating among **active** gestures, where amplitude
is useless and only the spatial pattern across the eight pods separates them;
that is where the six-gesture 43 % figure comes from. Re-measure once a third
active gesture exists, and quote the majority-class and one-threshold
baselines next to any accuracy from this dataset.

## 5b. Getting to a wearer-independent model

This is the eventual goal: the app should not care who put the armband on,
after at most a short calibration. The route, in order of value per hour spent:

1. **More subjects.** Nothing substitutes for this. Subject count buys more
   generalisation than repetition count, and until there are two subjects
   `leave_one_person_out()` correctly yields nothing.
2. **Train on `share`-style features** (`feature_set="level"` already includes
   them) or `representation="normalized"`, so absolute amplitude — the most
   subject-specific thing in the signal — carries less weight.
3. **Augment with `augment_rotate_channels`.** Armband rotation is the single
   biggest cross-wearer nuisance and this is nearly free. Augment inside the
   training fold only.
4. **Then calibrate per user.** The literature's practical answer is a general
   model plus a small per-user adaptation: pre-training on a pool of source
   subjects cuts the calibration data a new subject needs, meta-learning
   frameworks report ~97 % on a 7-class protocol from 10 % subject-specific
   data, and few-shot prototype adaptation is designed for exactly this. For a
   classical pipeline the cheap version is to re-fit LDA's class means on a few
   calibration repetitions while keeping the shared covariance.

Sources: [MyoFlow cross-session/cross-subject HD-sEMG](https://arxiv.org/html/2609.17194v1), [few-shot prototype adaptation](https://www.nature.com/articles/s41598-026-40352-6), [meta-learning few-shot gesture recognition](https://pubmed.ncbi.nlm.nih.gov/42490288/), [personalising a cross-user sEMG encoder on a small calibration budget](https://arxiv.org/pdf/2609.20296), [EMG-UP unsupervised personalisation](https://arxiv.org/pdf/2509.21589).

Note what this means for the dataset: keep `person_id` honest and never reuse
one, because every one of those methods needs clean subject boundaries.

## 6. A complete baseline

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import emg_dataset_tool as T

recs = T.load_dataset("datasets", stage="prefer_trimmed", skip_quality_errors=True)
X, y, groups, meta = T.make_windows(recs)
feats, names = T.make_feature_matrix(X, meta["fs"])

print(f"{X.shape[0]} windows, {len(meta['labels'])} labels, "
      f"{len(set(groups))} subjects")

scores = []
for train, test, held_out in T.leave_one_person_out(groups):
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
    )
    clf.fit(feats[train], y[train])
    score = clf.score(feats[test], y[test])
    scores.append(score)
    print(f"  held out {held_out}: {score:.3f}")

print(f"subject-independent accuracy: {np.mean(scores):.3f} "
      f"+/- {np.std(scores):.3f}")
```

With only one subject in the dataset, `leave_one_person_out` yields nothing —
that is correct behaviour, not a failure. Collect a second subject before
claiming anything about generalisation.

## 7. Augmentation

Generate augmented data into a **separate** array or a separate `augmented`
stage. Never modify a raw recording.

```python
rng = np.random.default_rng(0)
extra_X, extra_y, extra_g = [], [], []
for i in range(X.shape[0]):
    w = T.augment_amplitude(X[i], (0.85, 1.15), rng)
    w = T.augment_noise(w, 0.03, rng)
    extra_X.append(w); extra_y.append(y[i]); extra_g.append(groups[i])
```

| Function | Models |
|---|---|
| `augment_amplitude` | different contraction strength / skin impedance |
| `augment_time_warp` | a faster or slower performance of the gesture |
| `augment_noise` | electrical noise, poorer contact |
| `augment_channel_dropout` | a lifted electrode |
| `augment_baseline_shift` | electrode drift between sessions |
| `augment_rotate_channels` | **the armband worn rotated** — usually the single most valuable one for cross-subject robustness |

Two rules:

- Augment **inside the training fold only**. Augmenting before the split puts
  variants of a test window into training.
- `augment_time_warp` changes the window length. Re-window or resample before
  feeding a fixed-input model.

## 8. Live prediction

The GUI has a classifier panel that is inert until a real model exists. It
never displays an invented number.

Save a model as:

```python
import joblib
joblib.dump({
    "model": clf,                 # anything with .predict / .predict_proba
    "labels": meta["labels"],     # index -> name, matching training
    "fs": meta["fs"],
    "window_ms": 200,
    "representation": "raw",      # must match make_windows()
    "features": True,             # True = model expects extract_features()
}, "models/gesture_model.joblib")
```

Then set in the config block:

```python
ENABLE_LIVE_PREDICTION = True
MODEL_PATH = "models/gesture_model.joblib"
PREDICTION_SMOOTHING = 5      # majority vote over N consecutive windows
```

The panel shows the smoothed label and, when the model exposes
`predict_proba`, the confidence. If loading or prediction fails it says so and
disables itself rather than showing stale output.

## 9. Practical notes

- **Class balance.** `REST` accumulates faster than anything else if your
  recordings have long pre/post rolls. Check
  `..._summary.json`'s `windows_per_label` and use class weights.
- **`window_ms` vs. latency.** 200 ms at 200 Hz is 40 samples — a reasonable
  accuracy/latency trade-off for real-time control. Shorter windows react
  faster and classify worse.
- **Filtered mode (~50 Hz) gives 10 samples per 200 ms window.** Enough for
  the feature-based models, thin for a CNN. Consider `EMG_MODE = "raw"` if you
  are going to train deep models — see the caveat in `CLAUDE.md` §12.
- **Normalise per recording, not globally,** when crossing subjects:
  `representation="normalized"` uses a robust median/MAD scaling that one
  explosive burst cannot distort.
- **Keep the quality flags.** `skip_quality_errors=True` is the cheapest
  accuracy improvement available, and nothing is deleted to achieve it.
- **How much data?** Aim for ≥30 repetitions per gesture per session, ≥2
  sessions per person, and as many people as you can get. Subject count buys
  more generalisation than repetition count.

# RECOMMENDATION — data, models, gesture speed, placement-free calibration

Written 2026-10-13 against the tree as it is on that date. Sections 1–3 are
**measured from the files on disk**, not quoted from the other docs. Sections
4–9 are **recommendations and assumptions**, and are labelled as such.

Companion reading: `docs/DATASET.md` (format), `docs/ML_PIPELINE.md` (the
pipeline as built), `GUIDE.md` §5 (recording presets). This file is the
forward-looking one: what to collect, what to train, and what to build next.

---

## 0. The short version

1. **The current dataset cannot train a 6-class model.** On disk right now
   there are 62 recordings and all 62 are `REST`. The 6-gesture set
   `CLAUDE.md` §8 describes is no longer in `datasets/` — it survives only as
   an export bundle.
2. **In that surviving bundle the gestures are not separable, and no model
   will fix it.** Median total activation: `REST` 451, `FIST` 208. Rest was
   louder than the contractions. 83 % of all non-REST windows sit below
   `REST`'s own 90th percentile.
3. **The cheapest big win is a record-time contrast check**, not a better
   classifier. Reject a repetition whose activation does not clear the
   session's own rest level by a factor you set.
4. **Fast vs slow gestures need two things**: dynamics in the feature vector
   (the current `level` features average over time and delete exactly what
   separates `WAVE_LEFT` from `WAVE_RIGHT`), and multi-scale windows so one
   model covers a 150 ms burst and a 2 s hold.
5. **"Placement does not matter, calibration decides which channel is which"
   is the right target and is achievable** — the invariance you need is a
   circular rotation + a mirror + a per-channel gain, and all three can be
   estimated from a 60-second calibration block. Section 7 is the design.

---

## 1. What the data looks like — measured, 2026-10-13

### 1.1 What is actually in `datasets/`

```
python emg_dataset_tool.py --summary
  Raw recordings: 62   Trimmed: 0
  Dano / session_01 / REST  62
```

| Property | Measured |
|---|---|
| Subjects | 1 (`Dano`) |
| Sessions | 1 (`session_01`) |
| Labels present | `REST` only |
| Recordings | 62 raw, 0 trimmed |
| Duration | min 1.22 s, median 1.99 s, max 5.26 s |
| Samples per recording | min 62, median 100, max 263 |
| Measured rate | 49.4 – 50.2 Hz (median **49.9**) |
| Label window | equals the whole file in every recording (median 100 of 100) |
| Capture style | 35 `cut_from_take`, 27 `label_on_graph` |
| Quality flags | 2 × `rest_not_quiet`, nothing else |

`EMG_MODE = "filtered"` throughout. No cued-sequence recordings at all: this
session was collected entirely with LABEL ON GRAPH and by cutting clips out of
free takes, which is why the label window always covers the full file — there
is no pre/post roll to exclude.

### 1.2 Sampling is bursty, not uniform

Inter-sample intervals over all 62 recordings (8061 gaps):

| Interval | Share |
|---|---|
| 0–10 ms | 3.2 % |
| 10–15 ms | 15.1 % |
| 15–20 ms | 28.0 % |
| 20–25 ms | 37.0 % |
| 25–30 ms | 14.1 % |
| 30–41 ms | 2.5 % |

Median 20.6 ms, mean 20.1 ms, p95 28.0 ms, max 41.2 ms. The mean rate is
honest (~49.9 Hz) but **almost no sample arrives on time**. This is why
`resample_uniform()` onto `MODEL_FS = 60 Hz` exists, and why nothing
downstream may count samples instead of seconds. Confirmed-good design; keep
it.

### 1.3 Amplitudes in this session (device counts, all 62 REST recordings)

| | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 | CH8 |
|---|---|---|---|---|---|---|---|---|
| median | 20 | 23 | 24 | 24 | 25 | 23 | 21 | 20 |
| p95 | 23 | 28 | 52 | 32 | 66 | **144** | 25 | 24 |
| max | 65 | 139 | 93 | 92 | 348 | **361** | 297 | 86 |

A clean, flat resting baseline around 20–25 counts on every channel, with
occasional bursts on CH5/CH6 (the two `rest_not_quiet` flags, plus
sub-threshold movement). **This is what a good baseline looks like on this
hardware** — these numbers are the reference for §4.

### 1.4 The surviving 6-gesture data

`exports/dataset_20261006-143332-241.npz` — 1284 windows, 13 samples each,
`fs = 64.68`, 72 recordings, six labels, one subject. It is the only
multi-gesture data left. Per-gesture mean amplitude per channel:

| gesture | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 | CH8 | total |
|---|---|---|---|---|---|---|---|---|---|
| FIST | 27 | 51 | 139 | 59 | 35 | 36 | 23 | 36 | **406** |
| OPEN | 54 | 91 | 155 | 91 | 53 | 38 | 35 | 30 | 546 |
| REST | 22 | 83 | 129 | 51 | 42 | 33 | 24 | 19 | **402** |
| THUMB_UP | 22 | 46 | 97 | 49 | 29 | 49 | 24 | 27 | **343** |
| WAVE_LEFT | 25 | 54 | 286 | 59 | 83 | 86 | 32 | 60 | 684 |
| WAVE_RIGHT | 22 | 31 | 224 | 39 | 74 | 63 | 29 | 29 | 512 |

Note the resting baseline: ~130 counts on CH3 and ~83 on CH2, versus 24/23 in
the current session (§1.3). That session had **five times the resting level**
of the current one — poor contact, a loose band, or dry/unshaved skin.

Per-window total activation:

| gesture | p10 | median | p90 |
|---|---|---|---|
| FIST | 170 | **208** | 766 |
| OPEN | 199 | 466 | 981 |
| REST | 160 | **451** | 745 |
| THUMB_UP | 174 | 280 | 623 |
| WAVE_LEFT | 282 | 643 | 1094 |
| WAVE_RIGHT | 205 | 528 | 667 |

**`FIST` was half as loud as `REST`.** 83 % of non-REST windows fall below
`REST`'s p90. In-sample nearest-centroid on `level` features (descriptive
only, no cross-validation — do not read it as an accuracy):

```
             FIST   OPEN   REST THUMB_ WAVE_L WAVE_R   recall
FIST            2     15     25    107     17     14     1.1 %
OPEN           13     90     25     62     10     19    41.1 %
REST            2      2     82    105     16     23    35.7 %
THUMB_UP        3      9     43    128     16     15    59.8 %
WAVE_LEFT       0      0      0     45    125     53    56.1 %
WAVE_RIGHT      0      1      0     45     65    107    49.1 %
```

Two failure patterns, and they need different fixes:

- **`FIST` → `THUMB_UP` (107 of 180 windows), `REST` → `THUMB_UP` (105).**
  A collection failure. The contractions never left the noise floor, so three
  classes became one blob. Fix at record time (§4).
- **`WAVE_LEFT` ↔ `WAVE_RIGHT` (53 + 65 windows).** A *feature* failure. Those
  two gestures differ mainly in the **order** in which channels fire, and
  `extract_level_features()` averages the whole window, which deletes order.
  Fix in the features (§6.3).

Per-window Fisher ratios (between/within variance) over the 50 `level`
features peak at only **0.28** (`CH1_log_mean`, `CH2_share`) and bottom out at
0.046 (`CH8_std`). The `25.5` in `docs/ML_PIPELINE.md` §3a is a different
statistic (per-recording means, not per-window); both are true, they measure
different things, and the per-window number is the one a window-level
classifier actually has to work with. 0.28 is weak.

### 1.5 Two corrections to the existing docs

- **`CLAUDE.md` §8 is stale.** "72 recordings … Best so far ≈ 43 % per window"
  describes the Oct-6 bundle. Those recordings are not in `datasets/` any
  more, there is no `models/` directory, and no model has been trained since.
- **`exports/*.npz` `groups` is `person_id`, not the recording uid.**
  `make_windows()` fills `groups` with the person
  (`emg_dataset_tool.py:3550`), so an export bundle alone **cannot** support
  the grouped-by-recording CV that `train_model.py` performs.
  `train_model.py` is correct because it rebuilds groups from the tree
  (`train_model.py:541`). Anyone training from an export bundle will group by
  person only and leak across overlapping windows. Worth a one-line note in
  `docs/DATASET.md`, or an extra `rec_uids=` array in the bundle.

---

## 2. Format — what a consumer can rely on

Unchanged from `docs/DATASET.md`; repeated here as the contract a model-side
script may assume.

### 2.1 Per recording, three files

```
<GESTURE>_rep<NNN>_<YYYYmmdd-HHMMSS-mmm>_<uid8>_<stage>.csv
                                                 .npz
                                            _meta.json
```

`stage` ∈ `raw` | `trimmed` | `processed` | `augmented` | `take`. `raw` is
immutable; `uid8` is what pairs a raw file with its trimmed copies.

### 2.2 CSV

```
sample_index,t_rel_s,t_unix,ch1..ch8,in_label_window,gesture,person_id,session_id,repetition
```

One row per sample, identifying columns repeated on every row (deliberately
redundant — any single CSV is self-describing and blindly `pd.concat`-able).
`filtered` mode: `ch*` are non-negative integers, observed 15 … 1444.
`raw` mode: signed int8, −128 … 127.

### 2.3 NPZ / `Recording`

`.emg` (n, 8) int32 · `.timestamps` · `.t_rel` · `.fs` (**measured**, span
average) · `.label` · `.person_id` · `.session_id` · `.repetition` ·
`.label_start_index` / `.label_end_index` · `.issues` · `.meta`.

### 2.4 `_meta.json` — the fields that matter for training

`uid`, `stage`, `label`, `n_samples`, `duration_s`,
**`sample_rate_measured`**, `label_start_index`/`label_end_index`,
`amplitude_min`/`amplitude_max` per channel, `quality_issues`,
`activity_detected` (with `baseline`, `threshold_high`, `core_start/end`),
`meta.capture_style`, and a full `config` snapshot.

Quality codes in use: `empty`, `missing_samples`, `rate_mismatch`, `gap`,
`disconnect`, `flat_channel`, `out_of_range`, `saturated`, `invalid_values`,
`noisy_baseline`, `rest_not_quiet`, `no_activity`.

### 2.5 What is *missing* from the format for what comes next

Four fields I would add, all additive, all required by sections 6–8:

| Field | Where | Why |
|---|---|---|
| `gesture_speed` | `meta` | `"fast"` / `"slow"` / `"natural"` — lets you stratify and report per-speed accuracy (§6) |
| `calibration_uid` | `meta` | which calibration block this recording was canonicalised against (§7) |
| `rotation_reference` | `meta` | where the band's reference pod sat relative to an anatomical landmark (§7.8) |
| `rec_uids` | export bundle | so an export can be grouped by recording — see §1.5 |

---

## 3. What "expected" data looks like — targets

Assumption-based, anchored on §1.3's measured baseline.

### 3.1 Per recording

| Check | Target | Rationale |
|---|---|---|
| Resting level per channel | 15 – 35 counts | measured good session (§1.3) |
| Resting level, worst channel | < 60 | existing `QC_MAX_BASELINE_RMS` |
| Gesture p90 total / rest p90 total | **≥ 3×** | the Oct-6 set was ≈ 1× and that killed it |
| Loudest single channel during a gesture | 150 – 1400 | 1444 measured at max effort; above ~1800 suspect clipping |
| Active channels during a gesture | ≥ 2 | `TRIM_MIN_ACTIVE_CHANNELS` |
| Samples present | ≥ 80 % of expected | `QC_MIN_SAMPLE_FRACTION` |
| Largest gap | < 120 ms | `QC_MAX_GAP_MS` |
| Duration (cued) | 1.5 – 2.5 s + pre/post roll | current defaults are right |

### 3.2 Per dataset

| Level | Minimum worth training | Comfortable | Why |
|---|---|---|---|
| Repetitions per gesture per session | 25 | 40 | ≥ 6 CV folds grouped by recording |
| Sessions per subject | 2 | 3–4, different days | electrode shift is failure mode #2 |
| Subjects | 1 (subject-specific only) | **5+** | nothing cross-subject is claimable below 2; LOSO is noise below ~5 |
| Windows per class | 400 | 1500 | at 2 s and 50 % overlap, ~19 windows per recording at 200 ms |
| Rotations per subject | 1 | 3 (0, +1, +2 pods) | the only way to validate §7 |

### 3.3 Accuracy worth expecting — assumptions, not promises

| Protocol | Expect | Note |
|---|---|---|
| 6 classes, 1 subject, 1 session, grouped-by-recording CV | per-window 75–88 %, per-gesture vote 85–95 % | with §4 fixed data; the literature's 98 % is 200 Hz and many subjects |
| Same, held-out **session** | 10–20 points lower | electrode shift |
| Cross-subject (LOSO), no calibration | 40–65 % | the honest floor |
| Cross-subject **with** the §7 calibration | 60–80 % | the thing worth building |
| Current data, any model | ≤ 45 % per window | already measured; a ceiling of the data, not of the model |

If the first row does not clear ~75 % per-window once the data is fixed,
suspect the pipeline, not the models.

---

## 4. Fix the data first — concrete changes

The highest-value change in this document, and the cheapest.

### 4.1 A contrast check that would have caught the Oct-6 failure

`check_quality()` currently flags a non-REST recording that never exceeds
`QC_MIN_PEAK_AMPLITUDE` (40 counts, absolute). The Oct-6 `FIST` recordings
**passed** that, because a 130-count resting baseline is already above 40. The
check has to be relative to that session's own rest.

```python
# proposed, in the USER CONFIGURATION block
QC_CONTRAST_MIN_RATIO = 3.0              # gesture p90 total / session rest p90 total
QC_CONTRAST_REFERENCE = "session_rest"   # "session_rest" | "recording_prefix"
```

Implementation sketch (Section 3 of `emg_dataset_tool.py`, additive):

```python
def contrast_ratio(x, rest_reference):
    """p90 of total activation vs the session's resting p90. >=3 is healthy."""
    total = np.abs(x).sum(axis=1)
    return float(np.percentile(total, 90) / max(rest_reference, 1e-6))
```

`DatasetWriter` already knows the session folder, so the rest reference can be
cached from the `REST` recordings as they are written and a new `low_contrast`
warning raised on every later gesture. With `QC_PAUSE_ON = "warning"` the
operator is told *during* the session, which is the only time it is fixable.

### 4.2 Show contrast live

The CHANNELS readout shows amplitude. Next to it, it should show "× rest",
computed against the last REST block. A subject who can see "1.2× rest" will
squeeze harder; one looking at raw counts has no idea anything is wrong. That
is the whole explanation for why the Oct-6 set is unusable.

### 4.3 Re-collect the 6-gesture set

Before any modelling. With `REVIEW_EACH_RECORDING = True`, 30 reps,
interleaved, and a deliberate max-effort warm-up on each gesture so the
operator sees the achievable ceiling first. Keep the current session's
baseline (§1.3) as the acceptance reference — the band was clearly sitting
well on 2026-10-12.

---

## 5. Models and hyperparameters — what to use, when

Everything below is already expressible in `MODEL_KINDS` /
`DEFAULT_MODEL_SPEC_MODELS` or in the MODEL SPEC dialog, except where marked.

### 5.1 At today's scale (one subject, ~1.3 k windows, 50 features)

Roughly 25 windows per feature. That is **regularisation territory**, not
capacity territory: the roster should be short and biased toward bias.

| # | Model | Params | Why this one |
|---|---|---|---|
| 1 | **LDA** | `solver="lsqr"`, `shrinkage="auto"` | the myoelectric-control baseline for forty years; no hyperparameters to overfit; and its class means can be re-fit per user later (§7.7) — no other model gives you that for free |
| 2 | **Extra Trees** | `n_estimators=600`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"` | random split points beat RF on small, noisy, correlated EMG features; `min_samples_leaf=2` is the single most useful regulariser here |
| 3 | **HistGradientBoosting** | `learning_rate=0.06`, `max_iter=400`, `max_leaf_nodes=15`, `l2_regularization=1.0`, `early_stopping="yes"`, `validation_fraction=0.15` | the in-tree stand-in for XGBoost; shallow + slow + early-stopped is the right shape at this n |
| 4 | **SVM-RBF** | `C=10`, `gamma="scale"`, `class_weight="balanced"`, standard scaler | strongest single model in most small-EMG comparisons |
| 5 | **LogReg** | `C=0.3`, `penalty="l2"`, `class_weight="balanced"` | mediocre alone, but the best-calibrated member of a soft vote |
| 6 | **Soft vote** of 1+2+5 | `MODEL_VOTING_MODE="soft"`, `MODEL_VOTING_ALWAYS=False` | mixes one linear, one non-parametric and one calibrated model; keep `ALWAYS=False` so it only wins if it earns it |

Leave off for now: `kNN` (collapses the moment amplitude shifts — exactly the
cross-session/subject failure you care about), `MLP` (1.3 k windows is not
enough; keep it in the spec but disabled), deep models (§5.4).

**Expected spread between models 1–6: a few points, inside the fold-to-fold
std** — `fold_accuracies` in the sweep report gives you that band. Do not
chase a 2 % difference; at 6 folds over ~70 recordings, 2 % is noise. Spend
the time on sections 4, 6 and 7.

### 5.2 Search-space additions

`MODEL_SEARCH_SPACES` already sweeps sensible values. Three I would add:

```python
"extra_trees":  {"n_estimators": [400, 800], "min_samples_leaf": [1, 2, 4, 8],
                 "max_features": ["sqrt", "log2"],          # <- add
                 "class_weight": ["none", "balanced"]},
"svm_rbf":      {"C": [1.0, 3.0, 10.0, 30.0],               # <- 3 / 30 added
                 "gamma": ["scale", "auto"],
                 "class_weight": ["none", "balanced"]},
"logreg":       {"C": [0.03, 0.1, 0.3, 1.0, 10.0],          # <- lower end
                 "penalty": ["l2", "l1"],
                 "class_weight": ["none", "balanced"]},
```

The lower `C` values matter: at this sample count the optimum is usually more
regularised than the current grid's floor.

### 5.3 XGBoost, if you add the dependency

Not installed and not in `MODEL_KINDS`. `HistGradientBoostingClassifier` is
the same algorithm family and is already there, so this is optional. If you do
add it, start here and do not tune further at this n:

```python
XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4,
              min_child_weight=2, subsample=0.8, colsample_bytree=0.6,
              reg_lambda=2.0, reg_alpha=0.0, objective="multi:softprob",
              tree_method="hist", n_jobs=-1, random_state=0)
```

Adding it means a new entry in `MODEL_KINDS` (the param rules), a branch in
`build_models_from_spec()`, and a line in `requirements.txt` marked optional —
the trainer must still run when xgboost is absent, the same way it already
survives a missing pandas.

### 5.4 When deep models become worth it

Not now. The gate is **the raw stream**, not the dataset size:

| Prerequisite | Status |
|---|---|
| `EMG_MODE = "raw"` validated on hardware (200 Hz, int8) | **not done** — `CLAUDE.md` §12 |
| ≥ 8 subjects × 2 sessions × 30 reps | not done |
| ≥ 20 k windows | not done (≈ 1.3 k) |

At 50 Hz a 200 ms window is 10 samples; a 1D-CNN over 10 timesteps is a
dressed-up MLP, which is why the current roster is right to be feature-based.
Once raw mode is validated, 200 ms = 40 samples and 1 s = 200, and this is
where I would start:

```
1D-CNN:  input (8, 200)    # 1 s @ 200 Hz, channels-first
  4 x [Conv1d(k=5, pad=2) -> BatchNorm -> ReLU -> MaxPool(2)]   ch: 32,64,64,128
  GlobalAvgPool -> Dropout(0.3) -> Linear(n_classes)
  AdamW lr=3e-4, weight_decay=1e-4, batch=64, <=60 epochs,
  early stop on a held-out SUBJECT (never a random split)
  augment in-fold: rotate_channels (all 8), amplitude 0.8-1.25, time_warp 0.9-1.1
```

A CNN + single GRU layer (64 hidden) on top of the conv stack is the natural
next step, and the architecture that would finally capture wave direction
properly. Both must stay out of `emg_dataset_tool.py` (no torch dependency in
the recorder): a separate `train_deep.py` importing the same data layer, with
the same grouped protocol.

### 5.5 Keep the validation protocol exactly as it is

`StratifiedGroupKFold` on the recording uid, plus LOSO when >1 subject, plus
the per-window/per-recording pair of numbers. That is already better than most
published EMG work. The one addition worth making: **held-out-session CV**
once there are 2+ sessions, reported alongside the other two, because
electrode shift is the failure you will hit before you hit cross-subject.

---

## 6. Fast gestures vs slow gestures

### 6.1 Why the current pipeline treats them identically (and shouldn't)

`train_model.py` sweeps **one** window length (0.20 / 0.30 / 0.40 / 0.50 s)
and applies the winner to every class. The feature vector is
`extract_level_features()`: per-channel `mean`, `median`, `p90`, `std`,
`share`, `log_mean` plus two globals. Every one of those is
**time-symmetric** — reverse the window and all 50 features are unchanged.

Consequences, in order of severity:

1. **Direction is unrecoverable.** `WAVE_LEFT` and `WAVE_RIGHT` differ by the
   order in which pods fire. Time-symmetric features cannot represent order.
   §1.4 shows the cost: 118 windows confused between the two.
2. **A 150 ms burst is 7 ± 2 samples** at 49.9 Hz with 10–30 ms jitter (§1.2).
   After `resample_uniform` to 60 Hz it is 9 interpolated points, mostly
   invented. A fast `FIST` is barely represented at all.
3. **A fixed window mismatches both ends.** 300 ms is a third of a burst's
   useful life and a sixth of a hold's. Whichever the sweep picks, one class
   family pays for it.
4. **Smoothing penalises transients.** `PREDICTION_SMOOTHING = 5` at
   `PREDICTION_INTERVAL_MS = 150` is a 750 ms averaging window: a 200 ms flick
   can never dominate the average, so it can never be reported.

### 6.2 Multi-scale windows — one model, both regimes

Keep one feature vector. The single-feature-path rule in `CLAUDE.md` §13 is
non-negotiable: if training and inference diverge by one feature, live
predictions are garbage while cross-validation stays perfect. So instead of
choosing a scale, **concatenate scales** that all end at the same instant:

```python
MODEL_WINDOW_SCALES_S = (0.15, 0.35, 0.70)   # new config entry

def model_feature_vector(window, representation="raw", feature_set="level",
                         include_freq=True, fs=None,
                         scales=None):                     # <- new arg
    """Longest scale is the window passed in; shorter scales are its tails."""
    if not scales:
        return _single_scale(window, representation, feature_set,
                             include_freq, fs)
    n, longest, parts = window.shape[0], max(scales), []
    for s in sorted(scales, reverse=True):
        k = max(4, int(round(n * s / longest)))
        parts.append(_single_scale(window[-k:], representation,
                                   feature_set, include_freq, fs))
    return np.concatenate(parts)
```

Cost: 3 × 50 = 150 features instead of 50, and the live window must be the
longest scale (700 ms instead of 300 ms of ring buffer — free, the ring holds
180 s). Payoff: the short scale sees the burst, the long scale sees the hold,
and the classifier chooses per class. Sweep `MODEL_WINDOW_SCALES_S` as one
more axis, with candidates `(0.3,)`, `(0.15, 0.35)`, `(0.15, 0.35, 0.70)`.

### 6.3 Dynamic features — the fix for wave direction

Add a third feature family alongside `level` and `classic`. These are the
features a time-symmetric set cannot have:

```python
DYNAMIC_FEATURES = [
    "slope",          # per channel: linear fit of the envelope, counts/s
    "rise_frac",      # time-to-peak as a fraction of the window
    "half_ratio",     # log1p(mean(second half) / mean(first half))
    "env_max_rate",   # max positive dEnvelope/dt
    "active_frac",    # fraction of the window above 3x the window's own p10
]
DYNAMIC_GLOBAL_FEATURES = [
    "centroid_angle",   # circular mean of `share` around the 8-pod ring
    "centroid_drift",   # SIGNED change of that angle, first third -> last
    "centroid_speed",   # |drift| / window seconds
    "total_slope",      # slope of the summed envelope
]
```

`centroid_drift` is the important one. Treat the 8 channels as points on a
circle at angles `2*pi*c/8`, compute the activation centroid per third of the
window, and take the signed, wrapped difference. **A wave left and a wave
right have opposite signs on that one number.** Nothing in the current 50
features does.

```python
def ring_centroid(share):
    ang = 2 * np.pi * np.arange(8) / 8.0
    return float(np.arctan2((share * np.sin(ang)).sum(),
                            (share * np.cos(ang)).sum()))

def signed_drift(a, b):                       # wrapped to (-pi, pi]
    return float((b - a + np.pi) % (2 * np.pi) - np.pi)
```

This depends on the pod-ring assumption (A1 in §10) — and it is also the
cheapest way to *test* that assumption: if `centroid_drift` separates the two
waves, CH1…CH8 really are adjacent around the arm.

Add `feature_set` values `"level+dyn"` and `"all"` and put them in the sweep's
`FEATURE_SETS`. Expected effect, as an assumption: most of the
WAVE_LEFT/WAVE_RIGHT confusion, i.e. roughly 8–10 points of 6-class
per-window accuracy on a properly collected set.

### 6.4 Align the window to the onset, not to the cue

For transient gestures the informative part is the first ~300 ms **after
onset**, and the onset drifts by hundreds of milliseconds relative to the cue
— the whole reason `PRE_GESTURE_MS` exists. `detect_activity()` already
returns `core_start`. Use it in `build_dataset()`:

```python
ONSET_ALIGN = True            # new config
ONSET_ALIGN_PRE_MS = 100.0    # keep this much before the detected onset
```

Windows then tile from `core_start - ONSET_ALIGN_PRE_MS` rather than from
`label_start_index`. For `REST` and other no-activity labels, fall back to the
label window — there is no onset, by definition. Training-time change only;
live inference is inherently "now"-aligned.

### 6.5 Per-class decision timing

One model, but per-class post-processing. Transients must be allowed to win on
one window; holds should have to persist:

```python
CLASS_MIN_DWELL_WINDOWS = {      # new config
    "REST": 4, "FIST": 2, "OPEN": 3, "THUMB_UP": 3,
    "WAVE_LEFT": 1, "WAVE_RIGHT": 1,
}
```

In `LivePredictor`, keep the current soft-probability averaging — it is the
right call, and the comment at `emg_dataset_tool.py:7850` explains why — but
let a class fire once its *own* dwell requirement is met, using the mean over
that many windows instead of a fixed 5. A flick becomes reportable without
making holds jittery.

### 6.6 Collect speed explicitly

Record fast and slow versions of the same gesture under the same label, with
`meta["gesture_speed"]` set, using the existing GUIDE §5.3/§5.4 presets. Two
reasons: the model must see both (otherwise you have a model for whichever
speed you happened to record), and you can then report accuracy per speed
instead of guessing which half is failing. Suggested split per gesture: 40 %
natural, 30 % fast, 30 % slow-sustained.

### 6.7 And if fast gestures are the priority: raw mode

Everything above is a workaround for 50 Hz. The actual fix is
`EMG_MODE = "raw"` (200 Hz, int8), where a 150 ms burst is 30 samples instead
of 7. Required first, per `CLAUDE.md` §12: a long live session verifying rate
and amplitudes, and `feature_set="classic"` (median-centring is *correct* for
a zero-centred bipolar stream) or `"both"`. Note that the `level` features and
all of §4's contrast logic are written for the rectified stream, so switching
modes means re-deriving the thresholds in `QC_*` and `TRIM_ABS_FLOOR` — both
already have per-mode entries, which is why this is a config change and not a
rewrite.

---

## 7. "I don't care how I put it on" — calibration-defined channels

The target: the subject puts the armband on any way round, performs a short
calibration, and from then on the model works. This is achievable, and it is
also what makes sessions and subjects poolable.

### 7.1 What actually varies between placements

| Nuisance | Group | Fixable by calibration? |
|---|---|---|
| Rotation around the arm | circular shift of the 8 channels (continuous; 8 discrete steps plus a fraction) | **yes** |
| Which arm / band orientation (flipped) | mirror of the channel order | **yes** |
| Proximal/distal position along the forearm | changes the muscle mix, not a permutation | partly (gain), not fully |
| Band tightness, skin impedance, hair, sweat | per-channel gain + offset | **yes** |
| Anatomy, muscle volume, fat layer | a genuinely different signal | no — needs subjects |

So the calibration has to estimate **one rotation, one flip, eight gains,
eight offsets**: 18 numbers from a 60-second block. Entirely feasible.

### 7.2 The calibration block (proposed protocol, ~70 s)

| Step | Duration | Yields |
|---|---|---|
| 1. Relax, arm supported | 5 s | `offset[8]` = per-channel median |
| 2. Max-effort **FIST**, 3 × 2 s with 2 s rest | 12 s | `gain[8]`, and the flexor profile |
| 3. Max-effort **OPEN** (wrist/finger extension), 3 × 2 s | 12 s | the second, roughly antagonist, profile |
| 4. Slow **WAVE_RIGHT** then **WAVE_LEFT**, 1 × 3 s each | 6 s | rotation **sign** / flip disambiguation |
| 5. Relax again | 5 s | drift check against step 1 |

Steps 2 and 3 are chosen because flexors and extensors sit on roughly opposite
sides of the forearm: their two share-vectors together pin the rotation far
better than either alone. Step 4 resolves the mirror, which a single pair of
opposite profiles cannot — flexor/extensor is symmetric under reflection,
wave direction is not.

### 7.3 Estimating rotation and flip

```python
def alignment(fist_share, open_share, template):
    """
    Return (shift, flip, score). `template` is the canonical concatenated
    [fist_share, open_share] profile: 16 numbers.
    """
    best = (0, 0, -1.0)
    for flip in (0, 1):
        f = fist_share[::-1] if flip else fist_share
        o = open_share[::-1] if flip else open_share
        for shift in range(8):
            v = np.concatenate([np.roll(f, shift), np.roll(o, shift)])
            s = float(v @ template / (np.linalg.norm(v) * np.linalg.norm(template)))
            if s > best[2]:
                best = (shift, flip, s)
    return best
```

Sixteen candidates, cosine similarity, microseconds. Refinements worth having:

- **Fractional rotation.** The band never lands exactly on a pod boundary.
  Interpolate: upsample each share vector circularly ×8 (`np.interp` on a
  wrapped axis), align at 1/8-pod resolution, and resample the signal's
  channel axis at the fractional offset. Expect ~1–3 points over the integer
  version; add it only once the integer version works.
- **Reject bad calibrations.** If `score < CALIB_MIN_SCORE` (start at 0.80),
  or the second-best shift scores within `CALIB_MIN_MARGIN` (0.05) of the
  best, the alignment is ambiguous — tell the operator to reseat the band and
  redo it. A silently wrong rotation is far worse than no canonicalisation at
  all.
- **Where the template comes from.** First subject/session: their own aligned
  profile, `shift=0, flip=0`, written to `models/channel_template.json`. After
  that: the mean of all enrolled, aligned sessions. The template is a dataset
  artifact, versioned next to the model, and a model must record which
  template it was trained against.

### 7.4 Gain and offset

```python
canon[:, c] = (x[:, perm[c]] - offset[perm[c]]) / scale[perm[c]]
scale[k]    = max(p90_during_max_effort[k] - offset[k], CALIB_MIN_SPAN)
```

This is deliberately an **MVC-style normalisation**: every channel ends up
roughly on 0…1, where 1 is that person's own maximum effort on that pod. It is
the standard answer to "person A's signal is three times person B's", and it
is what would make `mean`-type features — not just `share` — transferable.

**One honest caveat.** `docs/ML_PIPELINE.md` §3a records that baseline
subtraction was *measured* to hurt (Fisher 24.9 → 11.5–17.2). That measurement
used per-recording baselines within one session, where offsets are constant
and a per-recording estimate partly cancels the gesture. This proposal differs
in both respects: the constants come from one dedicated calibration block, not
per recording, and they are applied identically in training and inference.
**It must still be re-measured.** Run the same sweep with
`CALIB_APPLY ∈ {none, offset, offset+gain, offset+gain+rotation}` as a new
axis and let the numbers decide. If offset subtraction hurts again, keep
rotation and drop the offset — they are independent.

### 7.5 Where it plugs into the code

Additive, and it preserves every invariant in `CLAUDE.md` §9:

```
SECTION 1   calibration_path(session_dir)
SECTION 2   canonicalize(window, cal)        <- pure, returns a new array
            ring_centroid(), signed_drift()  (§6.3)
SECTION 4   Calibration dataclass; load/save calibration.json;
            written to <session>/calibration.json -- append-only, a new uid
            per re-calibration, never overwritten
SECTION 6   CalibrationController (no Qt) -- the 5-step state machine, reusing
            SequenceController's wall-clock deadline pattern
SECTION 7   CalibrateDialog + a CALIBRATE button; optional hard gate on
            START SESSION when CALIBRATION_REQUIRED = True
```

The contract that keeps training and inference honest:

- `calibration.json` lives in the session folder; each recording's
  `_meta.json` gets `calibration_uid` **and a copy of the 18 numbers**, so a
  recording can always be re-canonicalised from itself alone, years later.
- Raw files are still written un-canonicalised. Canonicalisation is a *view*,
  computed on load — the same rule as `envelope()` and `normalized()`.
  Invariant 1 is untouched.
- `canonicalize()` is called **inside** `model_feature_vector()`, the one path
  `train_model.py` and `LivePredictor` share. Nowhere else.
- The model payload records `canonicalize: True` and the template id.
  `LivePredictor.load()` already refuses a model whose feature count
  disagrees with its declared `feature_set`; extend that check to refuse a
  canonicalising model when the current session has no valid calibration,
  with the same loud, explained failure.

### 7.6 Rotation-invariant features — the cheap complement

Independent of calibration, and worth having anyway: features that do not
change under a circular shift at all. Take the DFT across the **channel** axis
(length 8) of the per-channel mean vector and keep the magnitudes:

```python
mags = np.abs(np.fft.rfft(mean_per_channel))     # 5 numbers, shift-invariant
```

`|F0|` is total activation, `|F1|` is "how one-sided is the activation", `|F2|`
"how two-lobed", and so on. Eight channels give five magnitudes, all exactly
invariant to how the band is rotated — and the *phase* of `F1` is precisely
the ring centroid angle from §6.3. Add them to the `level` family as
`ring_mag0..4`; they cost nothing and they degrade gracefully when the
calibration is mediocre.

### 7.7 Still worth doing: train for rotation, and adapt per user

- **Augment with `augment_rotate_channels` inside the training fold only.**
  All 8 shifts turn N windows into 8N and directly teach rotation tolerance.
  This works *today*, needs no calibration, and is the highest
  value-per-line-of-code item in this whole section.
- **Per-user LDA recalibration.** LDA is in the roster for this reason: re-fit
  only the class means from the new subject's calibration block while keeping
  the pooled covariance from the whole dataset. A handful of repetitions, a
  closed-form update, no retraining. This is the classical answer to
  "calibration at the beginning decides everything", and it composes with
  §7.3 rather than competing with it.

### 7.8 What to do before any of this exists

Costs nothing, needs no code, and makes today's recordings usable later:
**mark the band**. Put a dot on one pod, always seat that pod over the same
anatomical landmark (e.g. directly over the wrist flexors, logo toward the
wrist), and write where it sat into `SUBJECT_NOTES` / the proposed
`rotation_reference` field. A recording with a known reference pod can be
rotated into canonical form retroactively; one without it cannot.

### 7.9 How to prove any of it works

None of §7 is verifiable with one subject in one placement. The minimum
experiment, about two hours of recording:

1. One subject, one day, **four sessions**: band at 0, +1 pod, +2 pods, and
   reversed orientation. 20 reps × 6 gestures each.
2. Train on session 0 only. Test on each of the others, with and without
   canonicalisation, with and without rotation augmentation.
3. Report the 2 × 2. Canonicalisation earns its place only if it closes most
   of the gap that +1/+2 opens.

Do this **before** building the GUI wizard. The data layer and the
`alignment()` function are enough to run it.

---

## 8. New-user / new-subject configuration

### 8.1 Paste-ready preset — enrolling a new subject

```python
# ---- identity (the only lines that MUST change per subject) ----
PERSON_ID = "person_03"          # never reuse an id; LOSO depends on it
SESSION_ID = "session_01"        # new id for every re-placement of the band
HANDEDNESS = "right"
ARM_WORN_ON = "right"
SUBJECT_NOTES = "forearm 27 cm, band notch 4, dot pod over wrist flexors"

# ---- what to record ----
GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]
REPETITIONS = 30
GESTURE_ORDER = "interleaved"    # spreads fatigue/drift across the label set

# ---- pacing for someone who has never done this ----
REST_TIME = 3.0                  # a new subject needs longer to relax
PREPARE_TIME = 1.2               # and longer to understand the cue
COUNTDOWN_TIME = 3.0
RECORD_TIME = 2.0
PAUSE_TIME = 1.5
PRE_GESTURE_MS = 400             # their reaction time is unknown; be generous
POST_GESTURE_MS = 400

# ---- supervision ----
REVIEW_EACH_RECORDING = True     # watch every take from a new subject
QC_PAUSE_ON = "warning"          # stop and fix contact problems immediately
VIBRATE_CUES = True
AUTO_TRIM = True
AUTO_TRIM_WRITE_FILE = False
```

### 8.2 The 12-minute enrollment routine

1. **Seat the band.** Dot pod on the landmark, one notch tighter than feels
   necessary. Wipe the skin.
2. **Watch the CHANNELS readout at rest.** All eight in the 15–35 band (§1.3).
   Any channel above 60, or flat, reseat before going further.
3. **Ceiling check.** Max-effort fist. The loudest channel should pass ~400
   counts, ideally 800+. If nothing clears 150, the band is too loose or too
   proximal — reseat.
4. **Calibration block** (§7.2) once it exists; until then, record the
   rotation reference in the notes.
5. **Teach the gestures.** Demonstrate each, have them do two of each
   untracked. Same-looking gestures from a subject who misunderstood the
   instruction are indistinguishable from bad electrodes in the data.
6. **Warm-up run.** `REPETITIONS = 2`, discard. This is also when you find out
   that their `OPEN` is a splayed-finger stretch and their `THUMB_UP` involves
   the whole wrist.
7. **Record.** Watch the review graph; REDO anything where the trace does not
   obviously differ from rest.
8. **Second session on a different day**, new `SESSION_ID`. One session per
   subject teaches the model that placement, nothing more.

### 8.3 Proposed new configuration block

For the `USER CONFIGURATION` block, in the project's existing style. All
default to off / current behaviour, so pasting this changes nothing until it
is switched on.

```python
# ------------------------------------------------------------------------------
# CALIBRATION  (placement-independent channels -- see recommendation.md 7)
# ------------------------------------------------------------------------------
CALIBRATION_ENABLED = False      # master switch for the whole feature
CALIBRATION_REQUIRED = False     # True -> START SESSION is gated on a valid one
CALIBRATION_NAME = "calibration.json"
CALIB_REST_S = 5.0               # step 1: offsets
CALIB_EFFORT_S = 2.0             # one max-effort contraction
CALIB_EFFORT_REPS = 3            # per calibration gesture
CALIB_GESTURES = ["FIST", "OPEN"]                        # profile gestures
CALIB_DIRECTION_GESTURES = ["WAVE_RIGHT", "WAVE_LEFT"]   # resolve the mirror
CALIB_MIN_SCORE = 0.80           # cosine fit below this -> reseat and redo
CALIB_MIN_MARGIN = 0.05          # best vs second-best shift; below -> ambiguous
CALIB_MIN_SPAN = 20.0            # floor on (max-effort p90 - offset), counts
CALIB_FRACTIONAL_ROTATION = False        # 1/8-pod alignment; needs the integer
                                         # version working first
CALIB_APPLY = "none"             # "none" | "offset" | "offset_gain" |
                                 # "offset_gain_rotation"   <- sweep this axis
CHANNEL_TEMPLATE_PATH = "models/channel_template.json"

# ------------------------------------------------------------------------------
# GESTURE DYNAMICS  (fast vs slow -- see recommendation.md 6)
# ------------------------------------------------------------------------------
MODEL_WINDOW_SCALES_S = (0.35,)  # (0.15, 0.35, 0.70) for multi-scale features
ONSET_ALIGN = False              # tile training windows from the detected onset
ONSET_ALIGN_PRE_MS = 100.0
CLASS_MIN_DWELL_WINDOWS = {}     # e.g. {"WAVE_LEFT": 1, "REST": 4}
DEFAULT_GESTURE_SPEED = "natural"        # written into every recording's meta

# ------------------------------------------------------------------------------
# QUALITY -- contrast against the session's own rest (see recommendation.md 4.1)
# ------------------------------------------------------------------------------
QC_CONTRAST_MIN_RATIO = 0.0      # 3.0 to enable; 0 disables the check
QC_CONTRAST_REFERENCE = "session_rest"
```

### 8.4 Metadata fields to start writing now

Free to add, impossible to backfill: `forearm_cm`, `band_notch`,
`rotation_reference`, `skin_prep`, `gesture_speed`, `fatigue_level` (1–5,
asked at the end of each block), `hours_since_last_session`. Each one is a
column you will wish you had the first time a subject's session looks wrong.

---

## 9. Priority order

| # | Action | Effort | Payoff | Section |
|---|---|---|---|---|
| 1 | Contrast QC + live "× rest" readout | small | **decisive** — without it the next dataset fails the same way | §4.1–4.2 |
| 2 | Re-collect the 6 gestures, 30 reps, reviewed | half a day of recording | **decisive** | §4.3 |
| 3 | Rotation augmentation inside the training fold | ~20 lines | high | §7.7 |
| 4 | Dynamic features (`centroid_drift` above all) | ~80 lines | high — should fix the wave confusion | §6.3 |
| 5 | Multi-scale window features | ~40 lines + a sweep axis | medium-high | §6.2 |
| 6 | Second and third subject | recording time | **unlocks LOSO at all** | §3.2 |
| 7 | The rotation experiment (4 placements, 1 subject) | 2 h recording | tells you whether §7 is worth building | §7.9 |
| 8 | Calibration: data layer + `alignment()` + canonicalise | 1–2 days | high, if #7 says so | §7.3–7.5 |
| 9 | Calibration GUI wizard | 1–2 days | quality-of-life, after #8 | §7.5 |
| 10 | Validate `EMG_MODE = "raw"` on hardware | one long session | gates every deep model | §6.7 |
| 11 | XGBoost / deep models | days | **last** — the data is the bottleneck | §5.3–5.4 |

Items 1 and 2 are worth more than 3–11 combined. The measured evidence for
that claim is §1.4: a dataset where `REST` is louder than `FIST` has no
model-side fix.

---

## 10. Assumptions, and how to check each one

| # | Assumption | How to verify |
|---|---|---|
| A1 | CH1…CH8 are physically adjacent around the arm, so rotation = `np.roll` | Press each pod in turn and watch which channel readout jumps; or check whether `centroid_drift` (§6.3) separates the two waves. If the mapping is not circular, the rotation math needs a fixed permutation table first — everything else in §7 still holds |
| A2 | Flexor and extensor profiles are roughly antipodal, so FIST+OPEN pin the rotation | Record both at known rotations 0/+1/+2 and confirm the estimated shift matches |
| A3 | MVC gain normalisation transfers better than absolute counts | the `CALIB_APPLY` sweep axis (§7.4) |
| A4 | Calibration-derived offset subtraction does **not** repeat the harm measured in `ML_PIPELINE.md` §3a | same sweep axis; the two are independent, keep whichever wins |
| A5 | Multi-scale features beat a single swept scale | one more sweep axis, same protocol |
| A6 | Wave confusion is a feature problem, not a collection problem | add `centroid_drift` to the Oct-6 export bundle and recheck the confusion — testable **today**, on existing data, with no recording |
| A7 | 50 Hz is the limiting factor for fast gestures | compare per-speed accuracy at `filtered` vs validated `raw` |
| A8 | A new subject needs ≤ 2 sessions to be useful in a pooled model | only answerable with 5+ subjects |

A6 is the one to run first: it costs one script, no hardware, and it either
confirms or kills the largest feature-side recommendation in this document.

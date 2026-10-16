# Per-wearer calibration — a plan

> ## STATUS, 2026-10-16: TIER 1 IS BUILT AND SHIPPED. TIERS 2–3 ARE NOT.
>
> See **CLAUDE.md §8k** for what exists, `tests/test_calibration.py` for its
> guard rails, and GUIDE.md for the operator's view. What landed is this
> document's **tier 1** (§3.4), reached by a simpler route than §3 proposed,
> and it was built in parallel with this plan rather than from it — so read
> the two as independent work that agrees, not as spec and implementation.
>
> **What shipped, and how it differs from §3:**
>
> | | this plan (§3) | what was built |
> |---|---|---|
> | form | tier 3, a full 8×8 `M`, regularised toward `I` | **tier 1**, a diagonal per-channel map |
> | fitted toward | a stored `calibration_reference` frame in the payload (§3.5) | a fixed `CALIBRATION_TARGET_LEVEL`; no frame, so no frame to leak |
> | calibration ritual | ~30 s, **six cued labelled gestures** (§3.6) | **6 s, two unlabelled captures** — rest and one strong contraction |
> | fit | regularised least squares | two percentiles per channel (16 numbers) |
> | the model | **frozen**; no retrain per wearer | frozen per wearer, but retrained **once** for everyone, on the calibrated scale |
>
> **What this plan got right, confirmed by measurement:**
>
> - §2.2 "a single scalar gain has nothing to correct" — the `gain` mode
>   measured **−0.2 points**. Exact.
> - §2.6 "per-channel spatial map is the only candidate left standing" — it
>   is where nearly all of the benefit is: `per_channel` +4.3 of the +5.3.
> - §2.5 `normalized` does not solve it — confirmed again, and CLAUDE.md §8h's
>   open item is now closed with that distinction written down.
> - §5 **the leak trap**, which the sweep walked straight into. The gate was
>   run as §5 specifies (calibration fitted on a disjoint third of each
>   wearer's recordings, scored on the rest): the leak was worth 0.4 points
>   and the honest gain is **+5.3 per window, +7.8 on the vote**, against a
>   gate of "must beat uncalibrated". It cleared. CLAUDE.md §8k has the table.
> - §5 "fix the DC-only split-key defect first" — done, and it turned a
>   nonsensical −14.3 points into +25.8.
>
> **What this plan got wrong:** §3.4's "tier 1 is cheap and probably
> insufficient". Tier 1 is worth +5.3 points on a held-out wearer. The
> reasoning — that §2.4's failure is a peak channel *moving*, which a diagonal
> map cannot do — still stands as far as it goes; it just turns out there was
> a lot of per-channel gain mismatch to collect first.
>
> **Still open, and still worth doing:** tier 3 (§3.4), the stored reference
> frame (§3.5), the cued six-gesture ritual (§3.6), and the FIST↔THUMB_UP
> confusion §2.4 diagnoses — which the calibrated model still has (recall
> 88.8 % and 82.9 %, with 15 FIST windows going to THUMB_UP and 22 back). A
> richer fit has no defence against §5's leak, so it must clear that gate on
> its own merits.

**Status of the original text below: design, not built. Every number in it is
measured; the scheme is not.**

This answers two questions:

1. What would per-wearer calibration actually have to fix?
2. How does it make the classifier work for a new wearer **without retraining
   the model every time?**

It supersedes the placeholder in [`docs/EDA.md`](docs/EDA.md) §11, and it
answers the open question that section left standing. The headline is that
**the obvious scheme — normalise each wearer's amplitude — would do almost
nothing on this hardware**, and the measurements below are why.

---

## Contents

- [1. The short answer](#1-the-short-answer)
- [2. What calibration has to fix, measured](#2-what-calibration-has-to-fix-measured)
- [3. The design](#3-the-design)
- [4. Why this beats retraining](#4-why-this-beats-retraining)
- [5. How it gets validated honestly](#5-how-it-gets-validated-honestly)
- [6. What calibration cannot fix](#6-what-calibration-cannot-fix)
- [7. Safety and honesty requirements](#7-safety-and-honesty-requirements)
- [8. The prerequisite](#8-the-prerequisite)
- [9. Staged plan, with gates](#9-staged-plan-with-gates)
- [10. Open questions](#10-open-questions)

---

## 1. The short answer

A trained model is a frozen function: 50 numbers in, one of six labels out.
Retraining moves **the function**. Calibration moves **the input** — it maps a
new wearer's signal into the coordinate frame the frozen model was trained in,
using a handful of numbers fitted from about thirty seconds of their time.

```
         TODAY                              WITH CALIBRATION

  wearer -> signal -> features        wearer -> signal -> CALIB -> features
                 |                                                   |
           frozen model                                        frozen model
                 |                                                   |
               label                                               label

  new wearer => refit the model        new wearer => refit 8-64 numbers
  (15 min recording + 8m21s sweep,     (~30 s, no sweep, model byte-identical,
   new model identity, validation       the validation number stays valid)
   number no longer describes it)
```

The model file never changes. `uid_digest` never changes, so
`model_data_status()` keeps reading `UP TO DATE`. The robot ships one model,
not one per person. And the project's honest number — **83.3 % on a wearer the
model never saw** — stays the baseline that calibration has to beat, instead
of being invalidated by the act of measuring.

**But the measurements below change what the calibration should be.** It is not
an amplitude normaliser. It is a spatial channel map.

---

## 2. What calibration has to fix, measured

All numbers from the 85-recording two-wearer dataset on disk
(`Dano` 44, `Betka` 41), per-channel medians over each label's labelled spans.

### 2.1 Resting level is NOT wearer-specific

| | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 | CH8 |
|---|---|---|---|---|---|---|---|---|
| `Betka` REST | 19 | 21 | 22 | 21 | 22 | 22 | 25 | 20 |
| `Dano` REST | 19 | 21 | 21 | 19 | 21 | 23 | 18 | 18 |
| ratio | 1.00 | 1.00 | 0.95 | 0.90 | 0.95 | 1.05 | 0.72 | 0.90 |

The two wearers' resting baselines are **the same to within 5 % on seven of
eight channels.** That makes physical sense: in `filtered` mode the Myo
rectifies on-device, so at rest what arrives is the electronics' noise floor —
a property of the *armband*, not of the arm in it.

**Consequence: a rest-anchored calibration is a no-op.** It would estimate a
gain of ~1.0 per channel and change nothing. This rules out the cheapest and
most tempting scheme — "record ten seconds of rest and divide".

### 2.2 Global gain is already ~1

| | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 | CH8 |
|---|---|---|---|---|---|---|---|---|
| `Betka` active | 105 | 130 | 86 | 108 | 315 | 434 | 194 | 148 |
| `Dano` active | 169 | 139 | 56 | 62 | 292 | 345 | 165 | 124 |
| ratio | 1.61 | 1.07 | 0.65 | 0.58 | 0.93 | 0.80 | 0.85 | 0.84 |

(active = mean over the five non-REST labels)

**Global active gain `Dano`/`Betka` = 0.89×.** Neither wearer is meaningfully
stronger than the other. So a single scalar gain — the MVC normaliser from the
textbooks — has nothing to correct either.

What *does* vary is the **per-channel** ratio: 0.58 to 1.61, a **2.79× spread
across channels**. The difference between these two wearers is where the energy
lands, not how much there is.

A further check: the ratio of the rest-estimated gain to the active-estimated
gain, per channel, is `0.62 0.93 1.47 1.56 1.03 1.31 0.85 1.07`. If rest
predicted active gain these would all be 1.0. They are not, which is §2.1
again from the other direction.

### 2.3 It is not a band rotation either

CLAUDE.md §8h predicted electrode rotation would be "the next real problem".
Tested directly: for each gesture, circularly shift one wearer's eight channels
by `k` and correlate the amplitude-normalised spatial profile against the
other's.

| gesture | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | best |
|---|---|---|---|---|---|---|---|---|---|
| FIST | **+0.59** | +0.06 | +0.37 | −0.30 | −0.66 | −0.42 | −0.22 | +0.58 | k=0 |
| OPEN | **+0.90** | +0.50 | +0.02 | −0.21 | −0.50 | −0.67 | −0.40 | +0.37 | k=0 |
| THUMB_UP | +0.40 | −0.25 | −0.65 | −0.69 | −0.38 | +0.46 | +0.48 | **+0.63** | k=7 |
| WAVE_LEFT | +0.64 | **+0.70** | +0.04 | −0.48 | −0.18 | −0.17 | −0.37 | −0.19 | k=1 |
| WAVE_RIGHT | **+0.99** | +0.47 | −0.25 | −0.48 | −0.53 | −0.48 | −0.21 | +0.49 | k=0 |

**The best shift disagrees across gestures: k ∈ {0, 1, 7}.** A band rotation is
*one* shift for every gesture simultaneously — that is what a rotation means. A
residual that needs a different shift per gesture is not a rotation, and no
renumbering of channels can remove it.

`k=0` already wins on three of five, and the band is evidently donned
consistently: WAVE_RIGHT matches at **r = +0.99**, OPEN at **+0.90**. So there
is no rotation to undo on this pair of wearers. (That could change the first
time someone puts the band on sideways; §9 keeps a rotation term as an
optional, cheap tier.)

### 2.4 What actually differs: two gestures, spatially

| gesture | profile correlation | `Betka` peak | `Dano` peak |
|---|---|---|---|
| WAVE_RIGHT | **+0.99** | CH6 | CH6 |
| OPEN | **+0.90** | CH6 | CH6 |
| WAVE_LEFT | +0.64 | CH2 | CH2 |
| FIST | +0.59 | **CH7** | **CH5** |
| THUMB_UP | **+0.40** | **CH6** | **CH1** |

Three gestures transfer between wearers essentially as-is. **FIST and THUMB_UP
do not — their peak channel moves.** And that is exactly what the model gets
wrong: the recording-grouped confusion matrix sends **31 FIST windows to
THUMB_UP and 16 back**, which is the bulk of its error.

So the between-wearer problem is not a diffuse gain mismatch. It is two
gestures whose spatial signature is genuinely different on the two arms.

### 2.5 The question EDA.md §11 left open, now answered

> *"Does `normalized` already solve it? It divides amplitude out entirely. If
> the sweep starts choosing it once subjects are grouped, an explicit
> calibration step may be unnecessary. This should be checked before anything
> is built."*

**Checked. It does not.** Best score per representation in the 2026-10-16
sweep, chosen under the **subject-grouped** protocol:

| representation | window | recording vote |
|---|---|---|
| `bandpass` | **83.3 %** | 82.4 % |
| `raw` | 83.2 % | 81.2 % |
| `envelope` | 72.3 % | 72.9 % |
| **`normalized`** | **32.8 %** | 51.8 % |

`normalized` is last by fifty points even with the folds grouped by subject —
the exact condition under which it was supposed to start winning. Dividing
amplitude out destroys the activation level that separates REST from active
(the same effect measured twice before: median-centring took Fisher 25.5 → 15.1
and accuracy 43.2 % → 33.5 %; baseline subtraction took Fisher 24.9 → 11.5).

**So an explicit calibration step is warranted, and brute amplitude-invariance
is not the way to get it.**

### 2.6 Summary of the target

| Candidate scheme | Measured verdict |
|---|---|
| rest-level normalisation | **dead** — rest is identical between wearers (§2.1) |
| global / MVC gain | **dead** — global gain is 0.89× (§2.2) |
| rotation alignment | **dead on this data** — best shift disagrees per gesture (§2.3) |
| `normalized` representation | **dead** — 32.8 % subject-grouped (§2.5) |
| **per-channel spatial map** | **the only candidate left standing** (§2.2, §2.4) |

---

## 3. The design

### 3.1 One transform, at the input, inside the one shared function

```python
def model_feature_vector(window, representation="raw", feature_set="level",
                         include_freq=True, fs=None, calib=None):
    block = np.asarray(window, dtype=np.float64)
    if calib is not None:
        block = apply_calibration(block, calib)   # <- the whole change
    ...                                            # everything below unchanged
```

`model_feature_vector()` is the single function `train_model.py` and
`LivePredictor` both call — the one contract binding training to inference.
Calibration goes **there and nowhere else**. A calibration applied in only one
of the two paths is the worst class of bug this project has: live predictions
become garbage while the cross-validation score stays perfect.

`calib=None` must be bit-identical to today's behaviour, so every existing
payload keeps working untouched.

### 3.2 It is applied to the SIGNAL, not to the feature vector

The 50 features are: 8 × `mean`, `median`, `p90`, `std`, `share`, `log_mean`,
plus `log` and `focality`. **48 of 50 are per-channel**, so a per-channel
signal transform propagates into almost the whole vector for free.

It must be applied to the signal rather than patched into the features,
because the propagation is not linear: `log_mean` picks up an additive
`log(gain)`, `share` is a ratio that renormalises, and `focality` is a spatial
concentration measure. Transforming the signal and recomputing is correct by
construction; patching 50 features by hand is a second implementation waiting
to drift.

### 3.3 One map per wearer — shared across all gestures

**This is the hardest constraint, and the easiest one to get wrong: at
inference time you do not know which gesture is being made.** So the map must
be gesture-independent. One matrix per wearer-fitting, used for every window.

Fitting a different map per gesture would not be a calibration — it would be a
classifier that needs the label it is supposed to predict. A single map fitted
to match *all* gestures jointly is also what keeps it honest: it cannot quietly
map one wearer's REST onto the reference's FIST, because the same map has to
serve REST and FIST at once.

### 3.4 Three tiers, in order of power and risk

Let `x` be one 8-channel sample, `C` the calibration.

| Tier | Form | Params | Fits what |
|---|---|---|---|
| **1** | `x' = g ⊙ x` (diagonal) | 8 | per-channel gain (§2.2's 0.58–1.61 spread) |
| **2** | tier 1 + circular shift `k` | 8 + 1 | tier 1, plus a band donned rotated |
| **3** | `x' = M x`, `M` ∈ ℝ^(8×8) | 64 | energy moving between channels (§2.4's FIST/THUMB_UP) |

**Tier 1 is cheap and probably insufficient**, because §2.4's failure is the
peak channel *moving*, which a diagonal map cannot do. **Tier 3 is what the
measurements actually point at**, and it is the one that needs care:

- **Parameter budget.** Fitted against six per-gesture median profiles, `M`
  has 64 unknowns from 48 equations — underdetermined, and it would fit noise.
  Fitted against per-window profiles (6 gestures × ~30 windows × 8 channels ≈
  **1440 equations**) it is comfortably determined. So **fit on windows, not on
  gesture medians.**
- **Regularise toward identity**: minimise `||MX − R||² + λ||M − I||²`. The
  wearer whose arm already matches the reference should get `M ≈ I`, and three
  of five gestures already transfer at r ≥ 0.64 — most of the map should be
  doing nothing.
- A full `M` is powerful enough to map anything to anything. §5's held-out
  measurement is what stops that being a problem, and it is not optional.

### 3.5 The reference frame

Fit toward the **mean per-channel per-gesture profile of the training
wearers**, not toward one privileged person. Computed at train time and stored
in the payload:

```json
"calibration_reference": {
  "labels": ["FIST", "OPEN", "REST", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"],
  "profile": {"FIST": [58, 44, 39, 94, 190, 108, 181, 137], "...": []},
  "n_subjects": 2, "representation": "bandpass", "fs": 60.0
}
```

Storing it in the payload makes the frame part of the model's identity, which
is what lets a prediction always say which frame it was made in (§7).

### 3.6 The ritual, and the file

About thirty seconds, and it reuses machinery that already exists:

```
CALIBRATE (new wearer / band refitted)
  -> one short free take: ~3 s of each of the six gestures, cued
  -> fit C against the payload's calibration_reference
  -> write datasets/<person>/<session>/calibration.json
  -> show the fit quality, and REFUSE it if the take is bad (section 8)
```

```json
{
  "person_id": "Betka", "session_id": "session_01",
  "fitted_iso": "2026-10-16T17:40:00", "tier": 3,
  "model_digest": "e75b40192e06c8b7",
  "reference_n_subjects": 2,
  "matrix": [[0.98, 0.01, "...8 per row..."], "...8 rows..."],
  "fit": {"residual_before": 0.41, "residual_after": 0.12,
          "per_gesture_r_before": {"THUMB_UP": 0.40},
          "per_gesture_r_after":  {"THUMB_UP": 0.78}},
  "source_take_uid": "d040d49b", "n_windows": 183
}
```

`model_digest` is the binding in §7. `source_take_uid` is the leak guard in §5.

---

## 4. Why this beats retraining

The question, directly: **what does calibration buy over just retraining
whenever someone new shows up?**

### 4.1 The structural reason

Training solves "which function maps features to labels". Calibration solves
"what coordinate frame is this wearer's signal in". These are different
problems, and only the second one is per-wearer. Conflating them means
re-solving a 50-dimensional model-selection problem every time you want to fix
an 8-dimensional frame mismatch.

### 4.2 The cost

| | Retrain per wearer | Calibrate per wearer |
|---|---|---|
| wearer's time | ~15 min recording + clip cutting | **~30 s** |
| compute | **8 m 21 s** sweep, 160 setups | a least-squares fit, **milliseconds** |
| artefact | a new 11 MB model (+57 MB panel) | a **~1 KB** JSON file |
| models to ship | one **per wearer** — and the robot must pick | **one, for everyone** |
| `model_data_status()` | flips to `OUT OF DATE` on every new wearer | stays `UP TO DATE` |
| feasible on the robot's SBC | no | yes — it is a matrix multiply |
| needs *labelled* data | yes, the full protocol | yes, but **six clips** |
| selection bias | **compounds** — a fresh winner picked each time | **none** — nothing is selected |
| the honest number | **destroyed** (see below) | **preserved** (see below) |

### 4.3 The part that matters most: retraining destroys the number

The project's one honest result is **83.3 % per window on a wearer the model
never saw**, because `SELECTION_GROUPING = "auto"` held that wearer out of both
training *and* model selection.

Retrain on a new wearer's own labelled data and that estimate no longer
describes what is deployed. What you would be able to quote is the
**within-subject** figure — **97.7 %** — which this project has spent its
entire history establishing is a measurement of electrode state and not of
gesture recognition (DC-only control: 97.6 %, i.e. eight channel means score
the same). You would have traded the only trustworthy number for the number
the whole repository exists to warn about.

Calibration keeps 83.3 % as the live baseline, and its own benefit is measured
*against* it, held out, by §5. That is the difference between a result and a
reassurance.

### 4.4 And retraining does not even address the measured problem

§2.4: FIST and THUMB_UP have different spatial signatures on the two arms. A
model retrained on wearer W learns W's signatures — which works for W and
teaches you nothing transferable. Fit a *map* instead and you have learned
something reusable: how W's arm relates to the reference frame, in 8 or 64
numbers that can be refitted in thirty seconds when the band is re-donned.

### 4.5 The bonus: measuring calibration is itself cheap

Because the model is frozen, the effect of a calibration is measured by
re-scoring the **same** fitted estimator on calibrated features. No sweep, no
refit, seconds not minutes. A calibration change can therefore be evaluated as
often as a unit test — which is the difference between a scheme that stays
honest and one that is too expensive to check.

---

## 5. How it gets validated honestly

Calibration is a candidate, not an improvement, until this has been run.
Safety rule 11 applies to an input transform exactly as it does to a model.

```
for each wearer W:
    model <- fit on all wearers EXCEPT W          (already what the folds do)
    C     <- fit on W's CALIBRATION TAKE ONLY
    score <- model.predict(features(W's other recordings, calib=C))
compare against the uncalibrated subject-independent baseline
```

**The gate.** Must beat the measured uncalibrated numbers:

| | uncalibrated | calibrated |
|---|---|---|
| mean | **83.3 %** | ? |
| held-out `Betka` | 89.3 % | ? |
| held-out `Dano` | 77.2 % | ? |

If it does not clear this, it does not ship, no matter how principled it looks.

**The leak trap, and it is the whole game.** The calibration take must be
excluded from the scored data. Fitting `C` on the recordings you then score is
fitting on the test set, and it would produce a beautiful number that means
nothing — the same failure as the class↔take confound, one level up. This is
what `source_take_uid` in the calibration file is for: the evaluation asserts
that no scored recording has that `parent_take_uid`.

**Second trap: the reference frame must not contain W.** §3.5's reference is
the mean over training wearers. When W is held out, W must be out of the
reference too, or the frame itself has seen the test wearer.

**Report the DC-only control alongside**, per §8f — and with the winner's split
key, which is the defect noted in CLAUDE.md §8 and has to be fixed before that
number means anything.

---

## 6. What calibration cannot fix

Stated now, so a disappointing result later is not mistaken for a bug.

**A wearer doing a different gesture.** `Betka`'s THUMB_UP peaks on CH6;
`Dano`'s on CH1 — and `Betka`'s is weak, **6.6× over rest against `Dano`'s
10.4×**, with most channels under 2× rest. Her clips are also **1.0–2.1 s**,
short of the 2.5–3 s the protocol asks for. Some of that gap is not a
coordinate-frame mismatch at all; it is two people performing a thumbs-up
differently, one of them faintly. **No input transform fixes that.** The fix is
the recording protocol: show the wearer the gesture, check the contrast live
(§8), and cut longer clips.

**Irreducible anatomy.** Muscle geometry under eight fixed pods differs between
arms. A full `M` can rotate energy between channels, but it cannot create a
distinction that is absent from the eight signals.

**The within-wearer confound.** Each gesture is still one take per person,
which is why the recording-grouped contrast is 97.7 %. Calibration is
orthogonal to this; only more *sittings per wearer* fix it.

**Fatigue drift.** A calibration fitted once cannot track a 59 % amplitude loss
across a long hold. Deliberately out of scope — a per-recording correction is
the thing that was measured to hurt, twice (§2.5).

---

## 7. Safety and honesty requirements

**Safety rule 10, no fake predictions.** A model calibrated for one wearer
predicting for another is a fake prediction. So the live panel reports
calibration with the same four-state discipline as `model_data_status()`, and
never guesses:

| State | Shown as | Means |
|---|---|---|
| `uncalibrated` | `NO CALIBRATION` (grey) | no file; predictions are in the reference frame |
| `current` | `CALIBRATED -- <person>, <date>` (green) | `model_digest` matches the loaded model |
| `stale` | `CALIBRATION STALE` (amber) | band refitted / new session since the fit |
| `mismatched` | `CALIBRATION REJECTED` (amber) | fitted against a different model or label set |

A mismatched calibration is **refused, not applied** — the same rule as
`DecisionPanel.matches()`. Feeding a model a frame it was not fitted against
produces confident nonsense, which is worse than no calibration.

**Nothing on disk is rewritten.** The calibration is a sidecar, like
`*_notes.json`. No recording is modified, no filtered copy of the dataset
exists, and safety rules 1, 2 and 3 are untouched.

**It is a candidate in the sweep**, like `bandpass` was: tiers 0/1/2/3 become
something the sweep scores, so whether calibration helps is measured rather
than asserted.

---

## 8. The prerequisite

**The contrast check must ship first.** `check_quality()` flags a non-REST
recording that never exceeds `QC_MIN_PEAK_AMPLITUDE` — **40 counts, absolute**.
`grep QC_CONTRAST emg_dataset_tool.py` still returns nothing.

This matters more for calibration than for recording, because a calibration
fitted from a bad take is **silently worse than none**: it corrupts every
subsequent prediction through the one function both paths share, and nothing
downstream would flag it. The calibration ritual has to be able to reject its
own input.

Measured peak-over-rest contrast on the current data, which is what such a
check would read:

| | FIST | OPEN | THUMB_UP | WAVE_LEFT | WAVE_RIGHT |
|---|---|---|---|---|---|
| `Betka` | 10.1× | 32.6× | **6.6×** | 20.1× | 42.2× |
| `Dano` | 10.0× | 20.3× | 10.4× | 24.8× | 39.7× |

A `QC_CONTRAST_MIN_RATIO ≈ 3.0` passes every gesture of both wearers, so the
threshold is well clear of good data while still catching the 2026-10-06
failure (resting baseline ~130 counts against a 40-count absolute floor, so
every bad recording passed and the dataset was lost). **`Betka`'s THUMB_UP at
6.6× is the weakest thing in the dataset and would warrant a live warning**
even though it clears 3.0.

---

## 9. Staged plan, with gates

Each stage ends in a measurement, and a stage that fails its gate stops there.

| # | Build | Gate |
|---|---|---|
| 0 | — | **done.** §2.5: `normalized` scores 32.8 % subject-grouped, so brute amplitude-invariance is not the answer and this plan is warranted |
| 1 | `QC_CONTRAST_MIN_RATIO` + a live "× rest" readout (§8) | catches a take whose contrast is under 3.0; passes all ten gesture/wearer pairs above |
| 2 | `calib=None` plumbed through `model_feature_vector()`, `apply_calibration()`, the sidecar format, the four-state display | **byte-identical** predictions with `calib=None`; a mismatched file is refused; tests in `tests/test_calib.py` |
| 3 | tier 1 (8 per-channel gains), fitted per §3, evaluated per §5 | beats 83.3 % mean / 77.2 % on held-out `Dano`. *Expected to be marginal* — §2.4's failure is a moving peak |
| 4 | tier 3 (8×8, ridge toward identity, fitted on windows) | beats tier 1 **and** 83.3 %; `M ≈ I` for the wearer who already matches |
| 5 | the CALIBRATE ritual in the GUI, cued, ~30 s | an operator who has not read this file can complete it and gets a pass/fail |
| 6 | tier 2's rotation term | only if a wearer ever actually dons the band rotated; §2.3 says nobody has yet |

Stages 1 and 2 are worth doing regardless: stage 1 is the highest-value open
item in the project independently of calibration, and stage 2 is inert by
construction.

**A third wearer outranks all of it.** Two wearers give one point of comparison
and no way to tell "`Betka` differs from `Dano`" from "`Betka` differs from
people". §3.5's reference frame is a mean over two profiles, which is barely a
mean. If a third wearer is available before stage 3, record them first — it
would change the fitted reference, and it might change §2.4's conclusion about
which gestures actually fail to transfer.

---

## 10. Open questions

- **Is tier 3 fitting a frame, or memorising a wearer?** 64 parameters against
  ~1440 equations is determined, but §5's held-out gate is the only thing that
  can tell the difference. If tier 3 beats tier 1 on the fitted wearer but not
  held out, it is memorising.
- **Should `share` and `focality` be excluded from the fit objective?** They are
  already amplitude-invariant, so they carry the spatial information
  calibration is trying to correct — possibly the most informative target,
  possibly double-counting.
- **How often must it be refitted?** Per wearer, per fitting, or per session?
  §2.1 suggests the *band* is stable; nothing here measures how much a
  re-donning on the same arm moves the profile. One wearer donning the band
  five times would answer it, and that take is cheap.
- **Does calibration interact with `bandpass`?** The winner filters before
  features. Calibrate before or after the filter? Both are linear and
  per-channel, so a diagonal map commutes with it — a full `M` does not.
- **What does a wearer with a genuinely different thumbs-up do?** If §6's
  conclusion holds, the answer is a protocol change, not a maths change.

---

## What this does not change

No file on disk is rewritten. No recording is modified. The model payload
format gains one optional block, and `calib=None` keeps every existing payload
working. If every stage above failed its gate, the correct outcome is a
documented negative result and `QC_CONTRAST_MIN_RATIO` shipped anyway — which
is why stage 1 comes first.

---

## Where the numbers came from

| § | Source |
|---|---|
| 2.1, 2.2, 2.4, 8 | per-channel medians over `label_start_index:label_end_index` of all 85 recordings on disk, grouped by `person_id` |
| 2.3 | circular-shift correlation of amplitude-normalised profiles, rest floor (~20 counts) removed first |
| 2.5 | `models/reports/sweep_20261016-160208-546.json`, best row per `representation` |
| 4.2, 4.3, 5 | the same sweep: `validation.grouping`, `validation.grouping_contrast`, `validation.subject_independent`, `confound.dc_only_control`, `elapsed_s` |
| 2.5 (Fisher) | the two earlier amplitude-removal experiments recorded in CLAUDE.md §13 and `docs/EDA.md` §11 |

Reproduce §2 with the two scripts used to write it, or re-derive from the
dataset directly — nothing here needs the armband.

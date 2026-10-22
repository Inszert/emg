# How to fix the long training

Why `train_model.py` grew to **3h 16m**, what was changed, and what each
change costs in accuracy. Everything here is measured on this machine against
this dataset; nothing is an estimate unless it says so.

Written 2026-10-15. Companion to `docs/EDA.md` (what the dataset *is*) and
CLAUDE.md §8d (why the GPU cannot help).

---

## 1. The short version

| | Before | After |
|---|---|---|
| Stage A wall-clock, projected over 360 setups | **203.7 min** | **~14 min** |
| Stage A1, **measured** on all 308 recordings | — | **161 s** (240 screens, 120 setups) |
| Last recorded full sweep | 11 747 s = **3 h 16 m** (119 recordings) | — |
| Distinct setups after folding identical filters | 360 | **120** |
| ...then **160** once `bandpass` was added as a 4th representation (§10) | — | **160** |
| Setups fully probed | 360 of 360 | 32 of 120 |
| Windows in the winning setup | 5 598 | 2 425 |
| Per-window accuracy | 99.7 % | 99.0 % |
| Per-recording vote | 99.4 % | 99.0 % |

**~14× faster, and the 0.4–0.7 points it "loses" are a cross-validation leak
being removed, not accuracy being given up.** The sweep got cheaper in four
independent ways, three of which cost nothing measurable at all.

The uncomfortable part is in §6: on this dataset **98 of those 99 points do
not come from gesture recognition**, and that was true before any of these
changes.

---

## 2. The issues, in order of how much time they cost

> **Measured on the real dataset, 2026-10-15:** stage A1 screened all 120
> distinct setups in **161 s**, and the run logged
> `no_errors->all, clean->all` — the fold in §3.2 working on live data. The
> banner now reads `120 preprocessing setups x 8 models = 960 combinations,
> of which 376 are evaluated`, where the old one claimed 432 setups and then
> swept 360 of them three times over.

### Issue 1 — the grid was 3× redundant (free to fix)

`EXCLUDE_SETS` has three entries (`all`, `no_errors`, `clean`) and they are a
grid axis, so every setup was swept three times. Measured on the current
dataset:

```
all        kept  308  dropped   0
no_errors  kept  308  dropped   0
clean      kept  308  dropped   0
all == no_errors ? True      all == clean ? True
```

The quality checks flagged nothing that any of the three filters removes, so
**all three selected the identical 308 recordings** and a third of the sweep
was the same computation run three times. 360 setups collapse to 120.

### Issue 2 — the expensive probe ran on every setup

Per-setup cost, measured on the current dataset:

| Work | Cost |
|---|---|
| featurising one setup | 1.0 – 3.5 s (scales with window count) |
| `LDA-shrinkage` scoring it | 0.25 s |
| `SVM-rbf-C10` scoring it | 0.31 s |
| `ExtraTrees-400` scoring it | **3.4 s, almost independent of data size** |

ExtraTrees is per-tree overhead, not per-row work: shrinking the data 2.3×
only took it from 4.14 s to 3.41 s. It was being paid 360 times to rank
setups that were never going to win.

### Issue 3 — most of the dataset is the same signal counted twice

This is the big one, and it is a correctness bug that happened to also be
slow. 280 of 308 recordings are clips cut from a handful of free takes, and
they tile those takes with heavy overlap:

```
label       clips   clip_s   uniq_s  redundancy
FIST           56    169.0     58.4   2.89x
OPEN           55    155.9     52.9   2.94x
REST           64    168.1    138.1   1.22x
THUMB_UP       40    158.7     54.7   2.90x
WAVE_LEFT      42    114.0     53.7   2.12x
WAVE_RIGHT     53    147.8     52.6   2.81x
TOTAL         310    913.5    410.4   2.23x
```

913 seconds of clip material from **410 seconds of unique signal**.
`docs/EDA.md` found the same thing from the other direction: 74 of the 308
recordings are byte-identical duplicates of another recording.

The cross-validation groups by recording uid — and that does not help here,
because every clip legitimately owns its own uid. Two clips that overlap
inside the take **share real samples**, so one can land in training and the
other in test. The same samples were on both sides of the split.

### Issue 4 — the screen was paying full price for a ranking job

Stage A's only job is to rank preprocessing setups. It was doing that on every
window of every setup, at full fidelity, to produce numbers that were then
thrown away as soon as the shortlist was picked.

---

## 3. What was changed

```
BEFORE   360 setups x 3 probes, full data, no dedup
AFTER    A1 SCREEN  every setup, deduped, 50 % of windows, cheap probes
         A2 PROBE   32-setup shortlist, full deduped data, all probes
         B          every model on the best 5 setups, full deduped data
```

**1. `dedupe_windows()` — drop windows that re-cover covered signal.**
Each window's centre is mapped onto the timeline it was cut from
(`parent_take_uid` + `take_start_index`/`take_end_index`, or the session wall
clock for `label_on_graph` spans) and quantised to the step size; the first
window to claim a bucket keeps it. That is exactly the set of windows you get
by windowing each source timeline **once**, so coverage of the unique signal
is preserved in full and only duplicates go.

It deliberately is *not* "drop the overlapping clip". Clip-level dropping also
throws away the parts that overlapped nothing — measured, it left 108 of 308
recordings and scored 98.0 %, where window-level dedup keeps every recording
contributing its non-duplicate windows and scores 99.0 %.

**2. Identical exclude filters are folded together.** Fingerprint the selected
recording uids; a filter matching an earlier one is skipped and the report
says so. The axis is not deleted — on a dataset with real quality flags it
still does its job.

**3. Stage A1 screens cheaply, stage A2 probes properly.** The screen uses
`SCREEN_PROBES` (LDA/SVM, ~0.25 s each) on `SCREEN_FRACTION = 0.5` of the
windows, evenly spaced rather than randomly drawn. Only the shortlist pays for
featurising at full fidelity and for the 400-tree forest.

**4. The shortlist is 24 best + 8 seeded-random.** Not purely the screen's
top-32 — see §4 for why that random slice is load-bearing.

New knobs, all at the top of `train_model.py`:

```python
DEDUPE_WINDOWS   = True
SCREEN_FRACTION  = 0.5
SCREEN_PROBES    = ("LDA-shrinkage", "SVM-rbf-C10", "LDA-svd", "LogReg")
A2_SHORTLIST     = 32
A2_RANDOM        = 8
```

Set `DEDUPE_WINDOWS = False` to reproduce the old (leaky) numbers exactly.

---

## 4. Why a shortlist is safe — measured, not argued

Shortlisting is the one change that *could* have cost accuracy, so it was
checked against the last full-grid sweep
(`models/reports/sweep_20261014-141319-030.json`), where all 360 setups were
scored by all three probes. For each shortlist rule: the best score reachable
*inside* that shortlist, against the best in the whole grid.

| Shortlist rule | Setups | Best window | Regret | Best vote | Regret |
|---|---|---|---|---|---|
| full grid (the old behaviour) | 360 | 99.65 % | 0.00 | 99.44 % | 0.00 |
| top-20 by `ExtraTrees-400` | 20 | 99.65 % | 0.00 | 99.44 % | 0.00 |
| top-20 by `LDA-shrinkage` | 20 | 99.65 % | 0.00 | 99.16 % | 0.28 |
| top-20 by `SVM-rbf-C10` | 20 | 99.65 % | 0.00 | 99.16 % | 0.28 |
| **random 40 (mean of 200 draws)** | 40 | **99.63 %** | **0.02** | **99.44 %** | **0.00** |
| random 60 (mean of 200 draws) | 60 | 99.64 % | 0.01 | 99.44 % | 0.00 |

Two things fall out, and both shaped the design:

- **The grid is far wider than the differences it resolves.** Dozens of setups
  tie at the top, so paying for all 360 full probes bought **0.02 points**.
- **The cheap probes rank well but not perfectly** — their own top-20 was 0.28
  points short on the vote, while *random* 40 was not short at all. That is
  why the shortlist is 24 best **+ 8 random** rather than simply the top 32:
  the random slice covers the case where the cheap probes' ranking is wrong,
  and it is seeded on `RANDOM_STATE` so a sweep stays reproducible.

---

## 5. Impact on the data

| | Before | After |
|---|---|---|
| Recordings read | 308 | 308 (unchanged) |
| Recordings contributing windows | 308 | 205 |
| Windows (winning setup) | 5 598 | 2 425 |
| Unique signal represented | 410 s | 410 s (unchanged) |
| Recordings voted on | 308 | 205 |

**Nothing is deleted from disk, and no unique signal is lost.** The 103
recordings that stop contributing are the ones whose every window was already
covered by an earlier clip of the same take — including the 74 byte-identical
duplicates `docs/EDA.md` found. The windows that remain cover exactly the same
410 seconds of real signal as before.

Per-class windows after dedup: FIST 383, OPEN 345, REST 689, THUMB_UP 325,
WAVE_LEFT 346, WAVE_RIGHT 337. Better balanced than before (REST was
relatively under-represented at 1 004 of 5 598).

### What it costs in accuracy

| Setup (0.30 s, 0.50 overlap, raw/both) | Windows | LDA | ExtraTrees | SVM |
|---|---|---|---|---|
| no dedup (old) | 5 598 | 99.7 % | 99.8 % | 99.7 % |
| **deduped** | **2 425** | **99.0 %** | **98.9 %** | **98.8 %** |
| deduped + 50 % screen | 1 216 | 99.1 % | 99.0 % | 98.9 % |

The 0.7–0.9 point drop is **the leak closing**. The old number was partly
measuring the model's ability to recognise samples it had already been trained
on. The new number is what leave-recordings-out actually supports.

The screen column also shows why `SCREEN_FRACTION = 0.5` is safe for ranking:
halving the windows moved the scores by ±0.1 points.

---

## 6. The thing that matters more than any of this

**98 of the 99 points are not gesture recognition.** The trainer now runs the
control CLAUDE.md §8f asks for, on the winning setup:

```
majority-class baseline: 28.4%
DC-only control (8 channel means): 97.2% per window, 97.6% per recording
-> the other 42 features add +2.1 points.
```

A classifier handed **nothing but the 8 per-channel mean levels** — no shape,
no spectrum, no timing — scores 97.2 % against the full vector's 99.3 %.
`docs/EDA.md` §3 found the same: the dataset's five active gestures are each a
*single continuous take*, recorded back-to-back without re-donning the
armband, so **label is perfectly confounded with electrode state and
wall-clock block.** The model can win by recognising which 50-second block it
is looking at.

Speeding the sweep up does not touch this, and neither would a better model.
The sweep now prints the control and the majority baseline next to the winner,
and both go into the sweep report (`confound`) and the saved payload
(`stats.dc_only_control`), so the number cannot be quoted innocently.

**The fix is in the recording, not the trainer** — `docs/EDA.md` §8 R1–R10.
The shortest version: several separate takes per gesture, armband removed and
refitted between them, gestures interleaved rather than in one block per
gesture.

---

## 7. What was considered and rejected

| Idea | Why not |
|---|---|
| **Run the sweep on the GPU** | No GPU implementation exists for LDA / ExtraTrees / HistGradientBoosting at all; SVM / RandomForest / LogReg only via cuML, whose wheels are Linux + cp310–312 against this machine's win-amd64 + Python 3.14. XGBoost *does* install here with `device="cuda"` but is a different algorithm from HistGradientBoosting, so it would change the spec rather than accelerate it — and at 6 000 × 154 the data is far below where GPU kernels win. CLAUDE.md §8d. |
| **Process-parallelism across setups** | ExtraTrees already saturates all 8 cores with `n_jobs=-1`; nesting would oversubscribe. The existing folds-on-threads path already covers the single-threaded estimators. |
| **Caching window arrays across setups** | Measured: resample + window for the *whole dataset* is 0.03 s against 1–3.5 s to featurise it. There was nothing to win. |
| **Shrinking the grid by hand** | The screen covers all 120 distinct setups for ~1.7 s each, so nothing has to be guessed away. Deleting axes would lose the "effect of each choice" table, which is still informative. |
| **Dropping whole overlapping clips** | Measured worse: 108 of 308 recordings kept and 98.0 %, against 205 kept and 99.0 % for window-level dedup. It throws away the non-overlapping parts too. |
| **Grouping CV by `parent_take_uid`** | The honest grouping, and currently impossible: each active class is exactly one take, so a leave-one-take-out split puts an entire class in test with none of it in training. `docs/EDA.md` §4 consequence 1. Recorded as a known limitation instead of faked. |

---

## 8. When more subjects arrive

Prepared 2026-10-15, before the data exists; full detail in CLAUDE.md §8h.
It matters here because it changes **what the sweep optimises**, and a fast
sweep that optimises the wrong thing is worse than a slow one.

`SELECTION_GROUPING = "auto"` switches the folds from recording-grouped to
**subject**-grouped as soon as a second person is on disk. Without that, the
winner would be chosen on folds where every subject appears in both train and
test — so a setup leaning on one person's absolute amplitudes would win and
then fail on the next wearer.

Verified on a synthetic three-subject dataset (96 recordings, 4 gestures,
per-subject baseline offset and gain):

```
Validation: StratifiedGroupKFold grouped by SUBJECT across 3 wearers
per-window acc     94.4% +- 7.9          <- subject-independent
within-subject contrast: 74.1% per window
   the two protocols disagree by 20.4 points
leave-one-subject-out: alice 100.0%, bob 100.0%, carol 83.3%
DC-only control: 18.5% against a 25.0% baseline
```

Two things to notice:

- **Both protocols are printed, and a >5-point disagreement is called out.**
  The gap is the subject-specificity; one number alone would let a
  within-subject score pass as a cross-wearer one.
- **The DC-only control flips meaning and that is the point.** On the current
  one-subject data it reads 97.2 % — the red flag from §6. Across subjects it
  reads 18.5 %, below the baseline, because absolute levels do not transfer
  between wearers. High DC-only across subjects would mean the people differ
  in gain, not that the gestures were recognised.

**Effect on sweep length: slightly faster, and noisier.** Subject-grouped
folds are capped at the number of subjects, so three wearers means three folds
rather than six — half the fits, and a larger per-fold variance (±7.9 above).
With only two subjects the score is a two-point average and should be read as
a direction, not a measurement.

Still not automatic, in priority order: take-grouped CV (possible once a
gesture has 2+ takes, not implemented), amplitude-invariant preprocessing
(`normalized` exists and the sweep should start preferring it by itself), and
electrode-rotation alignment between wearers (nothing handles it; expect it to
be the next real problem).

## 10. What the band-pass added back, 2026-10-15

A fourth representation, `bandpass` (CLAUDE.md §8i), joined `raw`,
`normalized` and `envelope`. Representations multiply the setup grid, so:

| | Before | After |
|---|---|---|
| Distinct setups | 120 | **160** |
| Stage A1 screens | 240 | 320 |
| Stage A2 probes | 32 setups | 32 setups (unchanged) |
| Evaluations in the run | ~333 | **445** |

**Stage A2 and B did not grow**, because `A2_SHORTLIST` is an absolute count
rather than a fraction — the expensive part of the sweep is fixed at 32 setups
no matter how many are screened. Only the cheap A1 screen scales with the
grid, which is exactly the property the three-stage split was built for. The
run cost roughly a third more than the 120-setup version, in the stage that
costs ~0.7 s per setup.

**Why it was added rather than hard-coded.** Whether an 8 Hz low-pass on the
envelope helps is a measurable question, and the project's rule is to measure
it. At one fixed setup (0.40 s / 0.5 overlap / `level`, LDA-shrinkage) on 325
recordings: `raw` 98.07 % window / 99.10 % vote, `bandpass` 98.15 % /
**99.55 %**, `envelope` 87.75 % / 95.05 %, `normalized` 22.34 % / 24.32 %. A
hair ahead of `raw` on the vote — worth having as a candidate, not worth
asserting as a default. The sweep picks it or it does not.

**If the sweep ever needs to get cheaper again**, dropping a representation is
the cleanest lever available: it is a straight 40-setup saving per
representation, with no effect on the shortlist's regret, and
`REPRESENTATIONS` is one line. `normalized` is the obvious candidate on
single-subject data, where it scores 22 % — but see §8 before removing it, as
it is expected to start winning once several wearers exist.

---

## 9. Verification

```powershell
# 1100 checks, no hardware, all passing after the change
python tests\test_datalayer.py        # 150
python tests\test_gui.py              #  91
python tests\test_model_spec.py       # 346
python tests\test_predict.py          # 109
python tests\test_trail.py            # 103
python tests\test_review.py           #  50
python tests\test_free_take.py        # 195
python tests\test_markers.py          #  88
python tests\test_label_on_graph.py   #  48
python tests\test_no_torch.py         #  11
python tests\test_sweep_cost.py       #  48   (new: everything in this note)

# the sweep itself
python training/train_model.py --spec models\model_spec.json
python training/train_model.py --quick --no-save --no-panel --spec models\model_spec.json
```

Reproducing the old numbers: set `DEDUPE_WINDOWS = False` and
`A2_SHORTLIST = 360`.

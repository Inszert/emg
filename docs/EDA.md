# EDA.md — what is actually in `datasets/`

Exploratory analysis of the recorded data, run 2026-10-15 against the tree as
it stands after `982064c Add session_02: four new gestures from Dano`.

Every number here was measured, not assumed. The analysis used the project's
own data layer (`load_dataset` / `make_windows` / `model_feature_vector`), so
it sees exactly what a trainer sees.

**Read §3 before trusting any accuracy number from this dataset.** The
headline is that the dataset currently cannot support episode-, session- or
subject-independent validation, and the ~99 % scores it produces are a
measurement of electrode state, not of gesture recognition.

---

## 1. Inventory

| | |
|---|---|
| Recordings (stage `raw`) | **308** |
| Subjects | **1** (`Dano`) |
| Sessions | **2** (`session_01`, `session_02`) |
| Labels | 6: `FIST`, `OPEN`, `REST`, `THUMB_UP`, `WAVE_LEFT`, `WAVE_RIGHT` |
| Total signal | **905 s = 15.1 min** |
| Free takes | 7 (in `_takes/`, correctly excluded from discovery) |
| Quarantined | 2 (`session_01/_deleted/`) |
| Sample rate measured | 49.42 – 50.23 Hz, median **49.88 Hz** |
| EMG mode | `filtered` for every single file |

```
Dano  session_01  FIST          56      Dano  session_02  OPEN          55
Dano  session_01  REST          63      Dano  session_02  THUMB_UP      39
                                        Dano  session_02  WAVE_LEFT     42
                                        Dano  session_02  WAVE_RIGHT    53
```

Per-label shape:

| label | n | duration med [min,max] | n_samples med [min,max] |
|---|---|---|---|
| FIST | 56 | 2.01 s [1.97, 5.01] | 101 [100, 251] |
| OPEN | 55 | 2.01 s [1.98, 4.02] | 101 [100, 202] |
| REST | 63 | 1.99 s [1.22, 5.26] | 100 [62, 263] |
| THUMB_UP | 39 | 3.99 s [3.99, 4.02] | 200 [200, 201] |
| WAVE_LEFT | 42 | 2.01 s [1.98, 4.02] | 101 [100, 201] |
| WAVE_RIGHT | 53 | 2.01 s [1.98, 4.01] | 101 [100, 201] |

The sample rate is stable and close to the documented ≈ 50 Hz. Nothing in the
stream itself looks broken.

---

## 2. Provenance — the single most important fact

**280 of 308 recordings were cut out of a free take. 27 came from
`label_on_graph`. One has no `capture_style` at all.** The cued
`SequenceController` path (REST → PREPARE → COUNTDOWN → RECORDING → REVIEW)
produced **none** of the current data.

A consequence visible in the metadata: `label_start_index == 0` and
`label_end_index == n_samples` for **308 of 308** recordings. There is no
PRE/POST_GESTURE_MS context anywhere in the dataset, so `use_label_window`
is a no-op on it.

### Every active class is one continuous take

| take uid | label | recorded | length | clips cut |
|---|---|---|---|---|
| `f00d9cdc` | REST | 2026-10-12 17:31 | 10.7 s | 1 |
| `fb91ea5e` | REST | 2026-10-13 17:55 | 71.9 s | 34 |
| `91a797e8` | FIST | 2026-10-13 22:50 | 58.4 s | 56 |
| `7567f69c` | OPEN | 2026-10-15 11:01 | 52.9 s | 55 |
| `287090eb` | THUMB_UP | 2026-10-15 11:03 | 51.8 s | 39 |
| `9d3d4c61` | WAVE_LEFT | 2026-10-15 11:05 | 53.7 s | 42 |
| `8d72f1b3` | WAVE_RIGHT | 2026-10-15 11:06 | 52.6 s | 53 |

Plus 27 `label_on_graph` REST spans and 1 untagged REST clip.

So **`FIST`, `OPEN`, `THUMB_UP`, `WAVE_LEFT` and `WAVE_RIGHT` each exist in
exactly one 52–58 s episode.** All four session_02 gestures were recorded
back-to-back inside a six-minute window (11:01 → 11:07) with no electrode
repositioning between them, and each active class falls on its own calendar
day relative to FIST:

```
FIST        2026-10-13
REST        2026-10-12, 2026-10-13
OPEN        2026-10-15      THUMB_UP   2026-10-15
WAVE_LEFT   2026-10-15      WAVE_RIGHT 2026-10-15
```

Label is therefore **perfectly confounded with "which block of wall-clock
time"**, which on dry EMG electrodes means skin impedance, sweat, exact pod
rotation and fatigue state.

### The clips overlap each other, and 74 of them are duplicates

The clips tile each take with roughly 50 % overlap, and in places the same
span was extracted more than once:

| take | label | clips | unique spans | take coverage | redundancy |
|---|---|---|---|---|---|
| `91a797e8` | FIST | 56 | 42 | 100 % | 2.91× |
| `7567f69c` | OPEN | 55 | 41 | 100 % | 2.96× |
| `287090eb` | THUMB_UP | **39** | **13** | 100 % | 3.02× |
| `9d3d4c61` | WAVE_LEFT | 42 | 41 | 100 % | 2.14× |
| `8d72f1b3` | WAVE_RIGHT | 53 | 41 | 100 % | 2.83× |
| `fb91ea5e` | REST | 34 | 27 | 76.6 % | 1.95× |
| `f00d9cdc` | REST | 1 | 1 | 11.6 % | 1.00× |

Hashing the signal arrays:

```
unique signal arrays : 234 of 308 recordings
duplicate groups     : 22        extra copies: 74
extra copies/label   : THUMB_UP 26, FIST 14, OPEN 14, WAVE_RIGHT 12, REST 7, WAVE_LEFT 1
largest group        : OPEN x10  (rep46..rep55, byte-identical)
```

`THUMB_UP` is the worst case: 39 recordings carrying 13 distinct signals.

After dropping byte-identical copies **and then** keeping only
non-overlapping spans, **105** of the 308 recordings remain. That is the
honest count of distinct 2–4 s observations on disk.

Nothing here is a bug in the tool — `unique_path()` held, and a clip
legitimately gets its own uid (CLAUDE.md §7, free takes). The duplicates come
from pressing EXTRACT CLIP without advancing, and the overlap from the
NEXT >> step length. The tool recorded faithfully what it was asked to
record.

---

## 3. Why the ~99 % scores mean nothing yet

Reproducing the trainer's protocol (leave-recordings-out,
`StratifiedGroupKFold` grouped by recording uid, `level` features plus
frequency, LDA-shrinkage — the model the sweep actually picked):

| what is grouped | n windows | groups | majority baseline | LDA accuracy |
|---|---|---|---|---|
| all 308 clips, grouped by recording uid *(as trained)* | 8749 | 308 | 0.187 | **0.9960 ± 0.0033** |
| byte-duplicates removed | 6145 | 234 | 0.212 | 0.9926 ± 0.0068 |
| duplicates **and** overlap removed | 2407 | 105 | 0.394 | 0.9849 ± 0.0148 |
| grouped by **parent take** | 2407 | 8 | — | **impossible** |

The last row is the finding. A leave-one-take-out split puts an entire class
in the test fold and none of it in training, because each active class *is*
one take. **There is currently no way to compute an honest
episode-independent score on this dataset**, and `leave_one_person_out()` has
nothing to work with either — one subject.

And the score does not need the gesture to be present at all:

| features given to the classifier | accuracy (random clip split) |
|---|---|
| the full 50-feature `level` vector | LDA 0.997 / RF 0.993 |
| **just the 8 per-channel mean levels** | LDA 0.977 / **RF 0.993** |
| **one number: the overall mean level** | RF 0.757 |

Eight DC numbers — no shape, no spectrum, no timing — reach the same 99 % as
the full feature vector. That is the signature of a dataset in which class
and electrode state are the same variable.

### Fatigue drift inside each take

Overall level against position in the parent take:

| label | Pearson r | first third | last third |
|---|---|---|---|
| WAVE_RIGHT | **−0.904** | 378.0 | 283.7 |
| FIST | **−0.826** | 163.1 | **66.6** |
| WAVE_LEFT | −0.787 | 385.8 | 301.2 |
| OPEN | −0.506 | 166.0 | 144.0 |
| REST | −0.267 | 31.4 | 22.1 |
| THUMB_UP | +0.133 | 233.0 | 226.5 |

FIST loses **59 %** of its amplitude between the start of its take and the
end. A clip's position in time is nearly as predictive as its class — both a
confound and a warning that a 52 s sustained hold is too long.

---

## 4. What the signal itself looks like

The gestures *do* have distinct spatial signatures, which is the encouraging
part of the analysis. Per-class median channel level (device counts):

| label | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 | CH7 | CH8 | strongest |
|---|---|---|---|---|---|---|---|---|---|
| REST | 20 | 23 | 24 | 24 | 25 | 23 | 20 | 20 | — flat |
| FIST | 115 | 49 | 28 | 33 | 133 | 155 | 66 | 286 | CH8, CH6, CH5 |
| OPEN | 214 | 317 | 121 | 69 | 54 | 64 | 216 | 119 | CH2, CH7, CH1 |
| THUMB_UP | 328 | 139 | 260 | 58 | 54 | 122 | 428 | 412 | CH7, CH8, CH1 |
| WAVE_LEFT | 212 | 201 | 76 | 52 | 119 | **787** | 724 | 602 | CH6, CH7, CH8 |
| WAVE_RIGHT | 428 | **972** | 727 | 156 | 88 | 68 | 67 | 98 | CH2, CH3, CH1 |

WAVE_LEFT and WAVE_RIGHT are near mirror images across the pod ring, which is
physiologically what you would expect. CH4 and CH5 are quiet in every class.

Fisher ratios (6-class, on the 50-feature `level` vector) agree:

```
CH2_median 47.8   CH2_mean 46.6   CH2_p90 41.8   CH3_p90 36.7
CH7_median 31.4   CH7_mean 29.8   CH7_p90 25.1   CH3_mean 24.3
...   mean 12.5   median 7.5
worst: CH5_std 0.55, CH8_std 0.89, CH5_p90 0.97, CH5_median 1.02, focality 1.05
```

Amplitude and clipping — the signal is **not** clipped:

```
all samples n=363776   min 13  max 1348  mean 189.9  p99 995  p99.9 1161
>= 2047 (configured ceiling) : 0.0000 %
>= 1023                      : 0.775 %
== 0                         : 0.000 %
```

Per class, the share of samples above 1000 counts: WAVE_RIGHT **5.31 %**,
WAVE_LEFT 0.67 %, FIST 0.018 %, everything else 0 %. WAVE_RIGHT peaks at 1348,
close to the 1444 maximum this armband has been observed to produce (CLAUDE.md
§2) — so the wave gestures are the ones to watch for headroom, not FIST.

REST separates cleanly on amplitude alone:

```
REST   median level  20.0 .. 22.0 .. 78.0
active median level  34.0 .. 137.0 .. 372.5
25 of 245 active clips sit below the loudest REST clip
```

### Label fidelity is good — the takes are 100 % contraction

For every active-class clip, the fraction of samples quieter than the REST
p95 level (43.8 counts) has median **0.0 %**, and not one clip is more than
50 % quiet. On the parent takes:

| take | label | quiet | active |
|---|---|---|---|
| `91a797e8` | FIST | 0.6 % | 99.4 % |
| `7567f69c` | OPEN | 0.0 % | 100 % |
| `287090eb` | THUMB_UP | 0.0 % | 100 % |
| `9d3d4c61` | WAVE_LEFT | 0.0 % | 100 % |
| `8d72f1b3` | WAVE_RIGHT | 0.0 % | 100 % |
| `fb91ea5e` | REST | 86.7 % | 13.3 % |
| `f00d9cdc` | REST | 0.0 % | **100 %** |

So no clip is secretly rest mislabelled as a gesture — the "five minutes of
mostly-rest signal as one label" failure that CLAUDE.md §7 warns about did
not happen. The flip side is that every active take is one **unbroken 52 s
hold**: there are no onsets, no offsets, no gesture-to-gesture transitions
and no inter-repetition relaxation anywhere in the dataset. A live classifier
spends much of its time on exactly those transitions.

### `f00d9cdc` is a suspect REST take

The first REST take (2026-10-12) sits at median level **103.8** against
**24.2** for the second one — more than 4× the amplitude, and 100 % "active"
by the threshold above. Only one clip was cut from it, and that clip's median
level is 78 against 22–25 for the 61 good REST clips. It is the loudest REST
in the dataset by a wide margin.

| REST source | n | median level min / med / max |
|---|---|---|
| `label_on_graph` | 27 | 20 / 21 / 22 |
| take `fb91ea5e` | 34 | 22 / 23 / 25 |
| take `f00d9cdc` | 1 | 78 / **78** / 78 |
| untagged | 1 | 46 / 46 / 46 |

---

## 5. Quality flags, as the tool recorded them

```
rest_not_quiet   3      noisy_baseline   4      missing_samples  1
```

Eight flags across 308 recordings. `detect_activity` results:

| label | activity found |
|---|---|
| FIST | 55 / 56 |
| OPEN | 55 / 55 |
| THUMB_UP | 39 / 39 |
| WAVE_LEFT | 41 / 42 |
| WAVE_RIGHT | 53 / 53 |
| REST | **6 / 63** |

REST finding no activity 57 times out of 63 is the correct answer, not a
problem — it is the behaviour you want from a rest class. This is a clear
improvement on the state recorded in CLAUDE.md §8 (15 of 72 flagged
`no_activity`, 5 of 12 FIST, FIST's mean envelope varying 26-fold): the
free-take route produces far stronger and far more consistent contractions
than the cued route did.

---

## 6. Bookkeeping gaps

| | |
|---|---|
| `session_01/session_config.json` | present |
| `session_01/markers.jsonl` | present, 8 markers |
| `session_01/manifest.jsonl` | 182 lines |
| `session_02/session_config.json` | **MISSING** |
| `session_02/markers.jsonl` | **MISSING** |
| `session_02/manifest.jsonl` | 193 lines |

The per-file `config` snapshot is present in all 308 metadata files and is
**identical across every one of them** (`emg_mode` filtered,
`nominal_sample_rate` 50.0, `record_time` 2.0, `window_ms` 200,
`window_overlap` 0.5, `repetitions` 30, the 6-gesture list), so no
information was actually lost — safety rule 7 held. But session_02 has no
session-level snapshot of its own and no timestamped operator notes.

Clip length is inconsistent, which changes how many windows each class
contributes:

```
FIST        100:8  101:28  102:1  250:7  251:12
OPEN        100:14 101:16  102:2  199:1 200:15 201:6 202:1
REST        62:1   64:1    100:38 101:11 262:6 263:6
THUMB_UP    200:27 201:12                      <- all 4 s
WAVE_LEFT   100:7  101:20  200:8  201:7
WAVE_RIGHT  100:12 101:20  200:14 201:7
```

Windows produced by `make_windows()` at the defaults (10 samples, step 5,
fs 49.88): 8749 total, from 12.6 % (WAVE_LEFT) to 18.7 % (FIST) per class —
reasonably balanced despite the above.

---

## 7. The saved model is stale

`models/gesture_model.joblib`, read directly:

```
labels              : ['FIST', 'REST']          <- 4 classes missing
model_name          : LDA-shrinkage
window_accuracy     : 1.0   (std 0.0)
recording_vote_acc  : 1.0
n_recordings        : 119   of 119
n_windows           : 3162     n_features : 50
validation_protocol : leave-recordings-out (StratifiedGroupKFold), 6 splits
subject_independent : null
data                : {}        <- EMPTY
```

Two things follow. It predates session_02, so it cannot name four of the six
gestures the dataset now contains. And its `data` block is empty, so
`model_data_status()` reports **`unknown`**, not `stale` — the GUI shows
amber `TRAINED on 2026-10-14` and cannot say what it was trained on.
CLAUDE.md §8c describes the `data` block as written by both trainers; this
payload does not have one.

The `1.0` is also exactly the number §3 explains: a two-class
contract-or-not question, scored on overlapping clips cut from one FIST take
and two REST takes.

---

## 8. Recommendations

Ordered by how much each one changes what the numbers mean. The first three
are the ones that matter.

### R1 — Record each gesture in several separate episodes *(blocking)*

The dataset cannot be honestly validated until a class exists in more than
one take. Target **at least 3–5 separate takes per gesture**, and make them
genuinely separate: take the armband **off and put it back on** between them,
or at minimum record them on different days. Interleaving the gestures inside
one wearing (OPEN, THUMB_UP, OPEN, THUMB_UP, …) is a large improvement over
the current block-per-gesture layout even without re-donning, because it
breaks the class↔time-block identity. This is the same reasoning as the
"interleaved gesture order by default" decision in CLAUDE.md §13, applied at
the level of takes instead of repetitions.

Until then, quote the confound next to every score, the way CLAUDE.md §8b
already requires the majority-class baseline to be quoted.

### R2 — Group cross-validation by parent take, not by recording uid

`StratifiedGroupKFold(groups=uid)` is not protective here: two clips can be
byte-identical and still land on opposite sides of the split. Every clip
carries `meta["parent_take_uid"]`; group on that, falling back to the uid for
`label_on_graph` clips. Today that makes the split *impossible* rather than
optimistic, which is the correct and informative outcome — and once R1 is
done it becomes the honest number this project is after.

Worth making the default grouping in `train_model.py` — which still groups by
recording uid, even with `dedupe_windows()` in place (see R3) — and the stated
protocol in `docs/ML_PIPELINE.md`.

### R3 — Deduplicate and de-overlap before training *(partly done)*

74 byte-identical recordings and ~50 % span overlap inflate `n`, shrink the
apparent variance, and tilt the class balance (THUMB_UP's 39 rows are 13
signals). Keep the files — safety rule 5 — and filter at load time.

`train_model.py` in the working tree already does this at **window** level:
`source_timeline()` places each clip on its parent take's index axis (or its
session wall clock for `label_on_graph` clips) and `dedupe_windows()` drops
windows whose samples another kept window already covers. That removes the
shared-sample leak, which is the sharpest part of the problem, and it is the
right layer to do it at — a window is what the model sees.

What it does not change: the class↔take confound (R1), the grouping (R2), or
the fact that `load_dataset()` still reports 308 recordings where there are
105 distinct observations. A guard in the cut dialog — refuse a span identical
to one already extracted from this take, warn above 50 % overlap — would stop
the duplication at the source rather than filtering it downstream every time.

### R4 — Shorten the holds, record the transitions, and record them at the
### effort you will actually use *(this one is already breaking live use)*

A 52 s unbroken contraction costs FIST 59 % of its amplitude by the end and
produces no onsets. Record in **3–5 s contractions with 2–3 s of rest between
them** — which is what the cued `SequenceController` path already does, and
it has produced none of the current data. Either use it, or add explicit
relax periods inside the free takes and cut clips that straddle the
transitions.

**The effort level is the half of this that is already observable live.**
Reported symptom: live, a thumbs-up is recognised for a few seconds and then
flips to `REST` while the hand is still held in the pose. The cause is a
train/serve mismatch in *effort*, and it is measurable in the dataset. Per
window mean amplitude, 24 samples at `MODEL_FS`:

| label | min | p5 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| REST | 20 | 22 | 22 | 23 | 25 | **43** | 108 |
| FIST | 43 | **52** | 69 | 105 | 129 | 251 | 445 |
| OPEN | 121 | **128** | 139 | 154 | 165 | 188 | 214 |
| THUMB_UP | 104 | **171** | 222 | 236 | 263 | 294 | 345 |
| WAVE_LEFT | 99 | **254** | 309 | 348 | 371 | 462 | 539 |
| WAVE_RIGHT | 230 | **254** | 286 | 334 | 367 | 410 | 464 |

REST tops out at 43 at its 95th percentile, and **nothing labelled THUMB_UP
exists below 171**. The band 43–171 is therefore signal that carries no label
at all, and `REST` is its nearest neighbour. The same void exists for OPEN
(43–128) and both waves (43–254). FIST is the opposite problem: its p5 of 52
sits almost on REST's p95, with 11 REST windows already above it, so a
low-effort fist is near-indistinguishable from rest in this data.

This is a direct consequence of the 52 s holds: every labelled example of a
gesture is an *effortful* contraction, because that is the only way to hold
one for 52 s. A real gesture is made and then **relaxed into**, and that
relaxed hold lands in the void.

Two simulations confirm the void is not reachable by amplitude alone, which is
why this needs recording rather than preprocessing. Scaling a real THUMB_UP
window down to 20 % gain still reads THUMB_UP at ~100 % confidence, and
blending it 70 % toward the subject's measured REST profile
(`20 23 24 24 26 23 21 20`) still reads THUMB_UP — the winning `level` feature
set includes `share`, the amplitude-divided spatial profile, so shape alone
carries it. Relaxing is not a uniform scale-down: specific motor units drop
out, so the *profile* changes too. Only the real relaxed hold produces it, and
the dataset contains none.

So, concretely, for each clip:

```
onset  ->  settle  ->  RELAXED HOLD  ->  release  ->  rest
   (the ramps R4 asks for)   (the part that fixes the live flip)
```

Hold at **natural, comfortable effort** — the effort the gesture will be used
at — not at maximum. Do not squeeze harder to get a cleaner-looking trace;
that is precisely what produced the 43–171 void. The labelled span should
cover the settle and the relaxed hold, not just the strong part.

### R5 — Retrain, and give the payload a `data` block

The saved model knows only FIST and REST. Retrain over all six classes so the
LIVE CLASSIFIER matches the dataset, and make sure `dataset_fingerprint()`
lands in the payload so the panel can report `current` / `stale` instead of
`unknown`. Then rebuild `gesture_panel.joblib` (`python explain_votes.py`).
Expect the reported accuracy to stay near 99 % and to mean no more than it
does now, until R1 is done.

### R6 — Re-examine or retire REST take `f00d9cdc`

4× the amplitude of the other REST take and 100 % "active". Its one clip
(median level 78) is the loudest REST in the dataset. Either confirm it was a
different arm/electrode state and keep it as deliberate REST variety, or
quarantine it to `_deleted/` — do not leave it sitting there unexplained.
Flag the untagged clip (no `capture_style`, median level 46) at the same
time.

### R7 — Write the missing session_02 bookkeeping

`session_config.json` and `markers.jsonl` are absent for session_02. The
per-file snapshots make it recoverable, so reconstruct `session_config.json`
from any session_02 metadata file's `config` block. Then check why recording
free takes without starting a session skips writing them — the session-level
snapshot should not depend on which recording path was used.

### R8 — Add a second subject

One subject means `leave_one_person_out()` is dead code and
`subject_independent` is `null` in every report. The project's stated goal is
honest subject-independent validation (CLAUDE.md §1); that needs a second
person more than it needs any further model tuning.

### R9 — Standardise clip length

THUMB_UP is 4 s, most other classes are 2 s, and FIST has 5 s clips. Pick one
`LABEL_DEFAULT_SELECTION_S` and keep it, so per-class window counts reflect
per-class data rather than per-class cutting habits.

### R10 — Watch WAVE_RIGHT for headroom

5.31 % of its samples exceed 1000 counts and it peaks at 1348 against an
observed device maximum near 1444. It is not clipping yet, but a stronger
repetition could. Check the per-channel maxima on the next wave take before
assuming the range is safe.

---

## 9. Reproducing this

The analysis is plain use of the data layer; nothing in it was persisted:

```python
import emg_dataset_tool as T
recs = T.load_dataset("datasets", stage="prefer_trimmed")      # 308
# provenance:   rec.meta["parent_take_uid"], ["take_start_index"], ["capture_style"]
# duplicates:   hashlib.sha1(rec.emg.tobytes()).hexdigest()
# the confound: classify numpy.array([r.emg.mean(axis=0) for r in recs]) on its own
```

Re-run it after any session that adds data, and in particular re-check §2
(is a class still one take?) and §3 (does the DC-only classifier still win?)
before quoting a new accuracy figure.

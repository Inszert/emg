# UI guide — what every control does

A visual walkthrough of the recorder. Every arrow in these pictures was placed
from the widget's real geometry by `tools/make_ui_guide.py`, not drawn by
hand, so an arrow either lands on the live control or the generator fails.

The screenshots show the project's real state: the real model, the real
decision panel, a real take off disk. Nothing in them is mocked up.

> New here? Read [the five-minute version](#the-five-minute-version) first,
> then come back for the detail.

---

## 1. The main window

![The main window](images/ui_main.png)

The layout never changes: **graph on the left, controls on the right, banner
across the top, status bar along the bottom.**

The graph is the part to trust. It stays live through every other operation —
even the review step draws a finished take over these same eight curves
without pausing capture, because samples are collected on the acquisition
thread and a slow frame must never cost you data.

**The banner is the thing to watch during a session.** It is deliberately
oversized: during a recording you are looking at your hand, not at a 10 px
label. It tells you what to do right now (`REST` → `GET READY` → `RECORD NOW`)
and turns orange the moment the link drops.

### Bottom bar

| Control | What it does |
|---|---|
| **START SESSION** | runs the cued plan: for every gesture × repetition, `REST → GET READY → 3-2-1 → RECORD` |
| **RECORD (manual)** | one take right now, outside the plan |
| **SAVE GRAPH** | current view as PNG or SVG into the session's `_graphs/` |
| **EXPORT DATASET** | combined ML bundle (`.npz` + index CSV + summary) into `exports/` |

### Status bar

Connection, sample rate, buffer fill, counts, and the **GPU/CPU chip**. That
chip is answered by a subprocess probe and never by importing torch into the
recorder — the import costs ~7 s and attaches a GPU context to a process that
must not drop samples. Until it answers it reads `checking…`, never a guess.

---

## 2a. The control panel — top half

![Control panel, top](images/ui_panel_1.png)

Each section folds away by clicking its header. On a laptop screen they do not
all fit at once, and scrolling past the one you want is the single most common
irritation in this UI — so fold the ones you are not using.

**SUBJECT / SESSION.** Use a **new person id for every new subject.** Data is
stored per person, and that is the whole mechanism behind
leave-one-person-out validation. Edits here are mirrored back into the module
globals, so the metadata written to disk always matches what actually
happened.

**LABEL ON GRAPH.** Select a span on the live plot and save it as a labelled
clip — no countdown, no plan. `SELECT LAST n s` sets the span, `ADD SELECTION`
writes it, `UNDO LAST` takes it back.

**FREE TAKE.** Record now, cut later. One continuous capture, minutes long,
which you carve clips out of whenever you like.

> **A take is never training data, and two independent mechanisms enforce
> that.** It lives in `_takes/`, which `discover_recordings()` prunes, and it
> carries stage `take`, which no stage selector asks for. Loading five minutes
> of mostly-rest signal as one labelled gesture would quietly wreck a model,
> so `tests/test_free_take.py` checks both.

**MARKERS & NOTES.** `Ctrl+M` for a note, `Ctrl+B` for a break. The timestamp
comes from the newest **sample**, not the GUI clock, so markers line up with
the signal. The log is append-only: a correction is another marker, never an
edit.

**RECORDING PLAN.** Gestures × repetitions for the cued sequence. Leave the
order **interleaved** — it spreads fatigue and electrode drift across every
label instead of baking it all into whichever gesture you recorded last.

---

## 2b. The control panel — bottom half

![Control panel, bottom](images/ui_panel_2.png)

**CHANNELS** hides pods from the display only. All eight are always recorded.

**RECORDINGS (this run)** is what this sitting produced, with quality and trim
state per row. `DELETE LAST` moves files to `_deleted/` — it never calls
`os.remove`.

> ⚠️ **Do not press AUTO-TRIM ALL on clips cut from a take.** `detect_activity`
> keeps the *strongest* region by design, and in the relaxed-hold recording
> protocol the hold is the **weakest** part of the clip. Measured on 17 real
> relaxed-hold clips: it kept a median of 70.8 %, worst case 42.4 %, and in
> **16 of 17** the discarded part was weaker than the kept part — at
> amplitudes of 50–90, precisely the band the protocol exists to capture. It
> is also unnecessary: a clip cut from a take has the whole clip as its label
> window, so **your selection already is the trim.**

**LIVE CLASSIFIER.** One-shot or continuous, drawn over the graph with
per-class probability bars. The provenance line reads one of four states and
never guesses:

| State | Means |
|---|---|
| `NO MODEL TRAINED` | no payload on disk |
| `TRAINED on <date>` | a model, but it cannot say what it was trained on |
| `UP TO DATE` | its dataset fingerprint matches the disk exactly |
| `OUT OF DATE` | the dataset changed; the tooltip says how |

**TRAINING.** `RETRAIN` runs `train_model.py` as a subprocess — not a thread,
because scikit-learn with `n_jobs=-1` starves the acquisition thread, and a
training crash must not take the recorder down. It runs on the **CPU** even
when the GPU badge is green: sklearn has no GPU backend, and both the tooltip
and the dialog say so rather than letting a green badge imply otherwise.

---

## 3. MODEL STATS

![Model stats](images/ui_model_stats.png)

Opened from the LIVE CLASSIFIER panel. **It is not modal**, on purpose: the
trail follows the live classifier, and a modal dialog would mean you could
only watch it if `PREDICT` happened to already be running.

**DECISION TRAIL** — one window, every candidate's vote, and how they fuse.

- **LIVE** runs the stored members on the window you just made. It is a
  *demonstration*, not an honest score — those members were fitted on the
  whole dataset — and the caption says so every time.
- **RECORDED** replays out-of-fold probabilities captured during training,
  where each window's recording was held out of the fold that scored it. This
  is the honest half.

**FEATURE SPACE** — every training window as one point, with the live window
drawn on the same axes.

It is a **PCA, not an LDA**, and that matters. An LDA picks axes that separate
the labels, so an LDA scatter looks beautifully separated even when the
separation comes from electrode drift rather than from the gesture — it would
flatter the model by construction. PCA never sees the labels, so what you see
is the structure really in the features.

Three colourings, because each asks a different question:

| Colour by | Question |
|---|---|
| **gesture** | do the labels form their own clusters? |
| **take** | …or do the clusters follow the recording session? **If this looks tidier than "gesture", the model is reading the block, not the gesture.** |
| **amplitude level** | where the effort sits, and the gap a relaxed hold falls into |

The **take** colouring is the single most useful diagnostic in this
application. Use it.

---

## 4. OPEN TAKE(S) & CUT CLIPS

![Cutting clips](images/ui_cut_clips.png)

Carving labelled training clips out of a free take, any time after recording.

Drag the green band or type the bounds. `NEXT >>` (`Ctrl+Right`) steps one
clip length along, so you can walk a take end to end.

**Where to put the span:** start just after the signal has clearly risen, end
before the release. Keep it on the settled, **relaxed** part of the hold.
Exclude flat rest at either edge — a pure-rest window labelled as a gesture is
how you get false positives firing while your hand sits still.

**How long:** about 2.5–3 s. At the winning setup (0.3 s windows, 0.75
overlap) that is ~30–37 training windows. Going longer does not buy more data
— windows are deduplicated against the source timeline anyway — it just drags
in ramp.

Each `EXTRACT CLIP` (`Ctrl+Space`) writes an ordinary stage-`raw` recording
with its **own** uid and full provenance (`parent_take_uid`,
`take_start_index`, `capture_style: cut_from_take`). Its own uid matters:
twenty clips sharing the take's uid would look like twenty versions of one
recording. The take itself stays byte-identical.

---

## 5. MODEL SPEC

![Model spec](images/ui_model_spec.png)

`Ctrl+Shift+M`. **Which classifiers the sweep trains is data, not code.** It
is written to `models/model_spec.json`, and the GUI passes `--spec` on every
retrain, so what the subprocess trains is always what this dialog was showing.

You can have several candidates of the same kind with different
hyperparameters — SVM at C=1, C=10 and C=100 are three rows, each scored
separately by the same cross-validation.

**Voting.** The ensemble is added as one more *candidate*, not as a
replacement, so it is scored against its own members. `soft` averages the
member probabilities; `hard` counts the votes. "Always save" forces it to be
kept even if a single model scored higher — a choice, not an improvement.

**AUTO mode expands, it does not search.** Each enabled model becomes up to
`max_per_model` concrete candidates which go through the *existing* evaluation
path. There is no second scoring route to keep honest, every combination gets
its own row, and the sampling is seeded so a sweep is reproducible. The
configured values are always combination #1.

> The cost is **selection bias**: choosing the best of N candidates on the same
> folds makes the winner's score optimistic. That is stated in this dialog, in
> the training log and in MODEL STATS rather than hidden — safety rule 11,
> "selecting a model never flatters it".

A hand-edited `model_spec.json` cannot crash a sweep an hour in.
`validate_model_spec()` never raises and never returns something the trainer
cannot build: an unknown kind is dropped, a bad value coerced, an
out-of-range value clamped, a duplicate name renamed, and a spec with
everything disabled gets one model switched back on. Every correction is
reported as a string the GUI shows and the trainer logs.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+Shift+R` | START / STOP free take |
| `Ctrl+Space` | EXTRACT CLIP (cut dialog) · ADD SELECTION (live graph) |
| `Ctrl+Right` / `Ctrl+Left` | step one clip length forward / back |
| `Ctrl+M` | add a note marker |
| `Ctrl+B` | add a break marker |
| `Ctrl+Shift+Space` | CUT SINCE LAST MARKER |
| `Ctrl+Shift+M` | MODEL SPEC editor |
| `Ctrl+Return` | ACCEPT & NEXT (review step) |
| `Ctrl+R` | REDO this repetition (review step) |

The review shortcuts use a modifier deliberately. A bare `Space` or `R` window
shortcut would swallow those characters while you are typing in the Notes
field.

---

## The five-minute version

Recording one sitting, start to finish:

1. Put the armband on. Wait for the link pill to go green.
2. Set **Person id** and **Session id**. A new subject gets a new id.
3. `START FREE TAKE` (`Ctrl+Shift+R`).
4. Per repetition: rest ~3 s → make the gesture → let it settle → **hold it
   relaxed for ~4 s at the effort you would really use** → release.
   Interleave the gestures; do ~6 of each.
5. `STOP & SAVE TAKE`.
6. `OPEN TAKE(S) & CUT CLIPS` → cut ~3 s clips from the settled part of each
   hold. **Do not press AUTO-TRIM ALL.**
7. `RETRAIN ON ALL DATA`.
8. **Take the armband off and walk away.** Repeat on another day.

Step 8 is not optional padding — it is the entire point. A gesture recorded
on only one electrode fitting cannot be told apart from that fitting, no
matter how good the model is. See [`EDA.md`](EDA.md) and
[`../CLAUDE.md`](../CLAUDE.md) §8f.

---

## Regenerating these pictures

```powershell
python tools\make_ui_guide.py . docs\images
```

It builds the real window with `MYO_PORT` pointed at a port that cannot exist,
so no hardware is touched, and renders with `WA_DontShowOnScreen` so nothing
appears on your display. It uses the **native** platform plugin rather than
`offscreen`, because the offscreen plugin ships no font database and renders
every glyph as a tofu box.

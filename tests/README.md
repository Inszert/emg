# Tests

Plain scripts, no pytest. Each prints `PASS`/`FAIL` per check and exits
non-zero if anything failed. Run them from the **project root**:

```powershell
python tests\test_datalayer.py      # 150 checks -- no hardware, no GUI
python tests\test_gui.py            # 91 checks  -- no hardware (simulated armband)
python tests\test_review.py         # 50 checks  -- no hardware (simulated armband)
python tests\test_label_on_graph.py         # 48 checks, no armband needed
python tests\test_markers.py        # 88 checks  -- no hardware (markers, notes, cutting at a marker)
python tests\test_free_take.py      # 195 checks -- no hardware (free takes, cutting clips out of them)
python tests\test_model_spec.py     # 346 checks -- no hardware (model roster, hyperparameters, voting, AUTO)
python tests\test_predict.py        # 109 checks  -- no hardware (trains a toy model, and runs a real retrain)
python tests\test_trail.py          # 103 checks -- the DECISION TRAIL (every candidate's vote, the fusion, the refusals)
python tests\test_no_torch.py       # 11 checks  -- the app runs with no PyTorch at all
python tests\test_sweep_cost.py     # 48 checks  -- window dedup, the screen, the shortlist, the DC-only control, subject-grouped selection
python tests\test_torch.py          # 163 checks -- the RETIRED deep trainer (uses the GPU if present)
python tests\test_live.py           # 35 checks  -- NEEDS the real Myo on COM3
python tests\test_live_gui.py       # 25 checks  -- NEEDS the real Myo on COM3
python tests\test_live_review.py    # 29 checks  -- NEEDS the real Myo on COM3
python tests\test_live_stability.py        # link-hold measurement -- NEEDS the Myo
python tests\test_live_stability.py 300 --baseline   # same, with the OLD settings
```

If `ModuleNotFoundError: emg_dataset_tool`, run them with the project root on
the path:

```powershell
$env:PYTHONPATH = "."; python tests\test_live.py
```

| Script | Needs armband | What it covers |
|---|---|---|
| `test_datalayer.py` | no | envelope/RMS, normalisation reversibility, explosive-burst and two-phase detection, all quality checks, trimming provenance, CSV/NPZ bit-exact round trips, non-overwriting writes, multi-person loading, windowing, features, leave-one-person-out leakage, augmentation purity, the full sequence state machine, ring buffer and capture caps, and the model-provenance fingerprint (it changes when a recording is added, a model matching it reads as current, one that does not reads as stale with a count, a deletion is reported as missing rather than as new, and a model with no provenance is reported as unknown rather than as up to date) |
| `test_gui.py` | no | builds the real `MainWindow` offscreen, feeds synthetic EMG through the real `_ingest` path, runs a compressed automatic session, exercises every graph control, the trim dialog, PNG+SVG export, DELETE LAST quarantine, auto-trim-all, dataset export, and verifies the classifier panel shows no fake prediction. Points `MYO_PORT` at a non-existent port so the reconnect path runs too. |
| `test_label_on_graph.py` | no | LABEL ON GRAPH: the 30 s window, and the property that matters most — a selection is anchored in ABSOLUTE time, so its unix bounds do not drift while the axis scrolls left underneath it. Also that ADD saves exactly the boxed samples, the coloured band tracks the signal and carries the gesture caption, UNDO quarantines rather than erases, a too-short selection is refused, the hand-labelled data loads and windows for training, and label mode and the cued session refuse to run together. |
| `test_markers.py` | no | MARKERS & NOTES: the append-only marker log survives a damaged line, unicode and a restart; a marker's timestamp comes from the newest sample rather than the GUI clock; CUT SINCE LAST MARKER writes marker->now through the same code path as ADD SELECTION and records which marker it came from; switching subject swaps the marker log; and -- checked both on the data layer and through the dialog -- annotating a saved recording leaves its signal file byte-identical. |
| `test_free_take.py` | no | FREE TAKE: a long unlabelled take is written to `_takes/` with stage `take` and -- checked on the data layer *and* through the GUI -- `load_dataset()`, `discover_recordings()` and the windowing path never see it; the capture cap really is the FREE TAKE cap and not the 120 s one; a take and the cued sequence cannot both own the acquisition thread's single capture slot; the banner says what is going on, and a dead link outranks the take while still naming it; markers dropped during a take travel in its metadata and markers from outside it do not; the default cut length set in the GUI reaches the config, the SELECT button and the live selection; and EXTRACT CLIP writes bit-exact copies into the gesture folder, with its own uid, incrementing repetitions per label and full provenance, while the take file stays byte-identical. |
| `test_model_spec.py` | no | Also the SVM scoring shortcut: the scoring estimator drops Platt scaling while the saved one keeps it, both predict identically, and -- the trap -- the soft-voting ensemble still contains the SVM, because soft voting selects members by `hasattr(predict_proba)` and sharing the scoring pipelines would have silently dropped it. MODEL SPEC: the schema is self-consistent (every default inside its own range, every search space naming only real parameters); `validate_model_spec` drops an unknown kind, coerces a non-numeric value, clamps an out-of-range one, renames a duplicate, ignores an invented parameter, parses typed layer sizes, and repairs a spec with everything disabled; a corrupt spec file falls back instead of crashing; AUTO mode is reproducible for a seed, capped per model, keeps the configured values as a candidate and honours a per-model opt-out; **every model kind builds, fits and predicts at its schema defaults**, as does every scaler, every `penalty` x `solver` combination of logistic regression, and every MLP solver with early stopping and batching; the voting ensemble is built from the enabled models only, in the requested mode, and is absent for a single model; `describe_estimator` reports the stages in order, the members and how their outputs are combined, and follows the OBJECT rather than the spec; and the dialog's edits, clamping, add/remove, voting, AUTO preview and save round-trip through the file. |
| `test_review.py` | no | the operator-paced review step: the sequence stops at `REVIEW` and provably never advances on its own, the graph freezes on the take with a draggable trim band, dragging changes what gets written, ACCEPT writes exactly the banded slice while the raw files stay byte-identical, REDO quarantines the take and repeats the **same** repetition, the countdown overlay stays out of the traces in `"corner"` mode, and the fully automatic path still works with the review off. |
| `test_no_torch.py` | no | The deployment guarantee: `import torch` is blocked outright, then the real application module is imported, a real saved model is loaded through `LivePredictor` and a real prediction is produced on the CPU. This is what stops the retired neural-network experiment creeping back in through an import -- the classifier is headed for an SBC where no NVIDIA GPU need exist. |
| `test_sweep_cost.py` | no | The sweep's cost controls, on a synthetic dataset built with a KNOWN 4x overlap so it does not depend on what is in `datasets/` today. Checks that window dedup is correctness and not a speed-up: it removes roughly the overlap factor, it is **window**-level so a partly-overlapping clip still contributes its unique windows (clip-level dropping was measured worse and must not creep back), it leaves genuinely non-overlapping clips untouched, it keeps every class populated, and a recording whose provenance cannot be placed keeps all its windows rather than vanishing. Also: two builds of the same setup are byte-identical, so a "seeded, reproducible" sweep is one; `screen_fraction` changes only the row count, never the label set or feature count; `DEDUPE_WINDOWS = False` still reproduces the old leaky counts exactly, so the before/after numbers in `how_to_fix_long_training.md` stay checkable; and the section 8f DC-only control really uses just the 8 channel means, really detects a level-only difference, and cannot change what gets trained. Finally the multi-subject switch: `resolve_grouping` picks recording-grouped folds with one subject and **subject**-grouped folds with two or more (and fails over rather than raising when asked for the impossible), `person_vector` lines up window-for-window with the recording groups, the per-recording vote still counts recordings and not people, and no subject appears on both sides of a subject-grouped split. |
| `test_trail.py` | no | MODEL STATS -> DECISION TRAIL. Builds a panel through the trainer's own `build_decision_panel()` on synthetic data, then checks what keeps it honest: the trail is out-of-fold and no recording straddles a fold; soft voting really is the unweighted mean of the members, recomputed here rather than taken from the stored flag; the live path runs on the **same** feature vector `LivePredictor` built, byte-identical and unmodified, so there is one feature path and not two; the deployed row is identified and is never the ensemble; the loaded forests are pinned to one thread, which is the 2.7 s -> 50 ms difference per live window; a panel built for a different feature set or feature count is REFUSED with the disagreeing field named, instead of being fed vectors it was never fitted on; a missing file, and a model file handed over as a panel, are states with messages rather than crashes; the widget paints at four widths and with no channel strip; and the view shows no numbers at all when there is no usable panel. Finally, where `models/gesture_panel.joblib` exists, the same properties are checked against the real seven candidates, including that one live window costs well under the prediction interval. |
| `test_torch.py` | no | **The RETIRED deep trainer**, `train_torch.py`. Kept so the experiment stays reproducible; nothing in the app depends on it. Runs on the CPU where there is no GPU, and additionally exercises the real GPU where there is one. Covers: the device contract (`--device cuda` fails loudly with the install command rather than falling back to the CPU silently); all three architectures build and emit one logit per class; the `log1p` standardisation is really fitted on the training fold only, and including the test fold would have changed it; `build_sequences` returns owned, writeable, contiguous float32 rather than the read-only strided view `sliding_windows` hands back; no recording straddles a cross-validation split; the seed reaches weight initialisation, so two same-seed fits agree to the last probability; the checkpoint round-trips and reproduces its own predictions; a free take never enters the training set; `--no-save` writes nothing and `--out-dir` is honoured for the report as well as for the model; and the trainer never touches the recorder's `.joblib`. Also the GPU/CPU probe the GUI badge is built on: it agrees with torch, a probe that printed nothing is not read as "no GPU", and the badge never lets a GPU imply that the scikit-learn sweep uses one. On a GPU it additionally checks that the parameters really live there, that two seeded runs agree exactly, that GPU and CPU reach the same conclusion, and that a GPU-trained checkpoint loads and runs where there is no GPU. |
| `test_predict.py` | no | the live classifier: trains a real LDA on synthetic channel profiles, saves it in the production payload format, then checks train/inference feature parity (byte-identical vectors), rate independence (same answer at 35 Hz and 125 Hz), one-shot mode predicting exactly once and then waiting, continuous mode tracking a change of gesture, the overlay on the graph, probability bars, the stats dialog, that recording and predicting coexist, and that a feature-mismatched model is refused rather than guessed with. Also the TRAINING panel: it starts queued rather than at a stale percentage, the bar follows real completions, the algorithm, preprocessing, candidate count, fold count and device come from the trainer's own events, no ETA is invented from one completed unit, completion reports the winner and the file it wrote, and a malformed event cannot raise. Also the model-provenance panel: a model with no `data` block reports that it cannot say rather than claiming to be current, one matching the dataset reads as UP TO DATE, one new recording turns it OUT OF DATE with a count, and the dataset walk behind all of it is skipped while samples are being captured. |
| `test_live.py` | **yes** | real connection, measured sample rate, value ranges, haptic cue, a real capture through `MyoAcquisition`, quality checks on real data, save + bit-exact reload, and a **forced reconnect** (closes the serial port underneath the reader). |
| `test_live_gui.py` | **yes** | the whole application against the armband: a real short automatic session, files on disk, bit-exact reload, graph export, and trimming a real recording. |
| `test_live_stability.py` | **yes** | Not pass/fail checks but a measurement: streams for N seconds while you wear the armband and reports dropouts, uptime percentage, effective rate, every gap over 1 s and the full state-transition log. `--baseline` reverts to the original settings (sleep on, 640 ms supervision timeout, scan on reconnect) so the before/after is directly comparable. |
| `test_live_review.py` | **yes** | the review step on live EMG: the graph freezes on a real take, the band comes from real activity detection, dragging it and accepting writes exactly that slice of real samples while the raw files stay byte-identical, and REDO quarantines the take and repeats the same repetition. Follow the on-screen cues while it runs. |

The live tests write into a temporary folder and delete it afterwards — they
never touch `datasets/`.

Close any other program using `COM3` (including `test_myo.py`) before running
the live tests.

## Known-good reference results

On this machine (Windows 11, CPython 3.14, Myo firmware 1.5.1970.2,
`EMG_MODE = "filtered"`):

- measured rate **≈ 50 Hz** (49.6-49.9 Hz over a 6 s window)
- amplitude: baseline ~15-40, maximum-effort fist **1100-1444** (filtered mode
  is NOT clipped at 1023, which is why `EMG_RANGE` allows up to 2047)
- 299 samples in 6 s, ~5 approximate "dropped" counts (burst delivery inflates
  this counter slightly; see `CLAUDE.md` §12)
- a 3.5 s capture yields ~175 samples

If the live test reports a rate around **5 Hz**, something has reintroduced a
timeout into the `myo.run()` call — see `CLAUDE.md` §3.

For `test_torch.py`, on this machine (NVIDIA GeForce GTX 1650, torch
2.9.1+cu128, CUDA 12.8): **163/163**, about three minutes, and the run opens
by printing which device it found. Without `torch` installed the script exits
1 with the install command — that is a missing optional dependency, not a
failure of the recorder, which never imports torch.

The armband stops advertising a minute or two after you take it off, so a live
run may open with `Armband did not answer in 20 s`. Keep it on, or keep moving
it, while the live tests run.

# NOTES -- decisions, open questions, ideas

Working log of the finishing pass (2026-10-26 onwards). Decisions made without
asking are written here with the reasoning, so they can be overruled later.
Newest at the bottom of each section.

## Resources (measured, Phase 0)

- CPU i5-9300H, 4 cores / 8 threads. RAM 7.9 GB (about 1 GB free while training
  runs). One GPU, GTX 1650 with 4 GB.
- Consequence: **no two GPU jobs at once**, sub-agents may not train or import
  torch, ML ablations are CPU-only and may overlap a GPU job (accuracies are not
  affected by contention, only wall time; latencies are measured separately in a
  quiet moment).

## Decisions

1. **One results file.** `results/experiments.json` is the single source for the
   documentation tables, the GUI stars and the figures. It is *generated* from the
   raw run files (`results/ablation/*/runs.jsonl`, the OOF probabilities and the
   model metadata) by `tools/collect_results.py`; nothing is typed in by hand.
2. **Fair measurement.** Leave-one-person-out over the 5 people on disk, the
   project's own windows / dedup / calibration, 3 seeds. A setting becomes a
   default only if it passes `training/ablation_stats.py::judge`: mean gain over
   the baseline of at least max(1 point, 2 x seed noise) **and** better for at
   least 4 of the 5 people (or the same on robustness without losing more than
   1 point of clean accuracy). Everything else stays off but selectable.
3. **Adaptation to the person** = standardising each person's input with
   statistics measured on a random 10 % of that person's own *unlabelled*
   windows (about 9 s of signal, the length of a calibration sweep): per-channel
   log-statistics for the nets, per-feature statistics for the ML. It is a
   different thing from the per-wearer calibration, which is unchanged.
   Because a model trained with adaptation sees differently distributed inputs,
   the switch in the settings only applies to models that were *trained* with it;
   with it off such a model falls back to its stored global statistics (the
   original pipeline for that model). Honest limitation, repeated in FEATURES.md.
4. **Augmentation** acts on raw calibrated windows at training time only (five
   types, `training/nn_augment.py`). Switching it off in the settings means the
   *trainer* does not use it; models already trained keep what they learned.
   With augmentation off the trainer's code path is the original one (checked:
   identical weights).
5. **Stars** are computed, never stored: rank by leave-one-person-out window
   accuracy, ties by macro-F1; top 3 per family plus the best soft-voting
   subset if it beats every single model. Recomputed whenever the zoo is listed,
   which includes after each training because training rewrites the model's
   metadata.
6. **Soft voting** over selected models: ML members must share the feature setup
   (same window, representation, feature set, calibration); NN members must share
   window length and labels (one worker process holds all of them and averages
   their softmax). A mismatching selection is refused with the reason, never
   silently averaged. NN+ML in one vote stays an explicit opt-in because CLAUDE.md
   rule 13 forbids a *silent* blend; see the entry below once measured.
7. **Docs/REPORT.md** could not be written by the documentation sub-agent (the
   Write tool refused a file named "report"); main wrote it from the agent's
   skeleton.

## Open questions / ideas

- Sub-agent C found things only main may change (widgets.py `_stat_tile` nested
  frames, header cards, MODEL SPEC / CALIBRATE layout); see `docs/UI_REVIEW.md`.
- `n_jobs=-1` forests and the NN worker share the machine with the recorder; the
  latency numbers in the report were measured on a BUSY machine (see Open).
- A second sitting of the weakest person (Barbora) is still the most valuable data
  to collect (see docs/MODELS_COMPARED.md); no model change substitutes for it.
- A new band: acquisition is Myo-only (CLAUDE.md 2b). `docs/sdk_myo.md` lists what a
  new backend needs.

## Findings worth keeping (2026-10-27) -- numbers are in REPORT.md / results/experiments.json

- **Newer architectures did not beat the GRU** (one seed, 150 epochs, person held
  out): GRU 72.4 %, ROCKET-GPU 69.7 %, channel transformer 67.7 %, TSMixer 65.2 %,
  CircConv-GRU 62.0 %, InceptionTime-S 60.6 %. The Phase 2 condition ("train
  anything that beats GRU 07 in real time") therefore did not trigger; all of
  them are in the zoo and selectable anyway. The 2-epoch smoke run of the same
  script ranked them differently (Inception first) -- a reminder that nothing
  here should be judged on a short or single run.
- **Seed noise is large for the nets.** The GRU baseline's mean over people was
  72.4 % (seed 0) but ~63 % for seed 1; over 3 seeds 67.7 % +- 4.7. The decision
  rule uses 2 x the seed spread, which is why NN adaptation (+4.5 points, 4/5
  people, +3.5 on robustness) is "no evidence" and stays off. It is suggestive;
  more seeds (or more people) is the experiment that would settle it.
- **ML adaptation passed on robustness** (+4.3 points, 3/5 people on accuracy,
  4/5 on the robustness probes) and is on by default for the zoo's ML models. An
  adapted model scored WITHOUT the person's statistics reads 78.2 % against
  78.5 % with them and 77.6 % for a plain model, so it is safe to deploy before a
  calibration signal exists (`tools/adapt_fallback_ml.py`). The gain comes
  mainly from the weakest person and varies a lot with which windows were
  sampled; five people cannot say more.
- **Augmentation: nothing helped.** For the ML pair gain / rotation / dead
  electrode hurt by 4 - 7 points (trees read absolute levels, which these
  distort); rotation helps the band-shifted probe a little but costs more on
  clean data. For the GRU nothing cleared the noise.
- **Calibration mode.** `affine` stays. `off` and `per_channel` hurt the ML pair
  clearly. A single global **gain** scored +3.6 points on average (Barbora 46 ->
  75 %, Miso 88 -> 71 %) -- not judged better (3/5 people, within noise) but it
  is the most interesting open lead: Barbora is the person a per-channel map
  seems to hurt. Worth repeating with more wearers before touching the default.
- **Voting.** The deployed pair (ET-500 + RF-500-leaf3) is 0.1 points better
  than ET alone (noise). The four-model spec the app currently trains is 7.6
  points WORSE than the pair. That spec is the operator's call (CLAUDE.md 8n);
  the zoo makes the pair one click (select ET-500 and RF-500-leaf3) without
  touching `model_spec.json`.
- **Zoo vote star.** The best ML subset searched on the folds (79.3 %) falls to
  77.0 % when the subset is chosen without the held-out person (nested), below
  the best single model on the same basis (77.6 %), so it is not starred --
  deliberately: the optimistic number would have starred it.
- **Things found by the sub-agents** (all handled or listed): scikit-learn is now
  pinned `<1.8` (the models were pickled with 1.7.2); `train_job.kill_tree` kills
  only the trainer on POSIX, not joblib workers (open, Linux only);
  `NeuralEngine.trained` is a method (no caller treats it as a property); the UI
  asks for Segoe UI / Consolas fonts (cosmetic on Linux).

## Open

- Quiet-machine latency benchmark for the zoo (`tools/bench_models.py` covers the
  deployed pair only); the latencies in the tables were measured while other jobs
  ran.
- ML augmentation / adaptation are applied by the zoo trainer only; the classic
  RETRAIN sweep (train_model.py) trains plain models. Folding them into the
  sweep means threading the switches through `build_dataset` for every setup of
  the screen -- not done, because the zoo trains the same models directly at
  the winning setup.
- UI items not fixable in the theme layer (docs/UI_REVIEW.md items 3 - 5):
  MODEL SPEC hyper-parameter scroll area, CALIBRATE dead space, hint font sizes.

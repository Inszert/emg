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
  latency numbers in the report are from a quiet machine.
- A second sitting of the weakest person (Barbora) is still the most valuable data
  to collect (see docs/MODELS_COMPARED.md); no model change substitutes for it.
- A new band: acquisition is Myo-only (CLAUDE.md 2b). `docs/sdk_myo.md` lists what a
  new backend needs.

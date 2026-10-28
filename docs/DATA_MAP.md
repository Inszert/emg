# DATA_MAP.md -- every saved result, where it is, and what answers what

For writing a report: where the numbers live, how each file was made, which
document already summarises it. Counts below were taken from the checkout on
2026-10-28; the generated documents are re-made by the tools named in each row.

## 1. Which question, which file

| question | open | made by |
|---|---|---|
| what is in the dataset (people, gestures, seconds, sample rates, quality flags, calibrations, windows)? | [DATASET_SUMMARY.md](DATASET_SUMMARY.md), `results/dataset_summary.json` | `tools/dataset_summary.py` |
| how well does each model do on a person it never saw, how big and how fast? | [REPORT.md](REPORT.md) section 4, `results/experiments.json` (`models`) | `tools/collect_results.py` then `tools/make_report.py` |
| does a training switch (augmentation, adaptation, calibration mode) help? | [REPORT.md](REPORT.md) section 6, `results/experiments.json` (`ablation`, `defaults`) | `tools/ablation_ml.py`, `tools/ablation_nn.py`, rule in `training/ablation_stats.py` |
| does voting help? what does a selected vote score? | [REPORT.md](REPORT.md) sections 4 and 6, [MODELS_COMPARED.md](MODELS_COMPARED.md), `experiments.json` (`voting`, `ensembles`, `starred_votes`) | same; the vote of any selection: `emg_app/zoo.py` `vote_stats` |
| what if the band is shifted / the signal weaker? | [REPORT.md](REPORT.md) section 7, `experiments.json` (`ablation` robustness columns) | the ablation tools |
| what does leaving a person out of training do? | [EXCLUDING_PEOPLE.md](EXCLUDING_PEOPLE.md), `results/exclusion.json`, `exclusion_per_person.csv`, `exclusion_switches.csv` | `tools/make_exclusion_report.py` |
| which ML setup, window length, features won the search? | [COMB_TEST.md](COMB_TEST.md), `models/comb_test/20261022-003918/` | `training/nn_ml_comb_test.py`, `tools/comb_test_report.py` |
| how do the models fail (per gesture, per person, confidence, timing)? | [MODELS_COMPARED.md](MODELS_COMPARED.md), `docs/figures/` | `tools/make_figures*.py` |
| newer architectures | [REPORT.md](REPORT.md) section 5, `models/reports/new_models_test_run1/` | `tools/new_models_test.py`, `tools/make_figure_new_models.py` |
| how fast is a live window? | [REPORT.md](REPORT.md) section 9, `models/reports/bench_models_20261022.json` | `tools/bench_models.py`, `tools/bench_zoo.py` |
| do the automatic checks pass? | [REPORT.md](REPORT.md) section 11, `tests/last_run.json` | `tools/run_tests.py` |
| the check of `publication.tex` against the data (what was corrected, added, left open) | [PUBLICATION_CHECK.md](PUBLICATION_CHECK.md), figures made for it by `tools/make_figures_paper.py` | (this pass) |
| the archived first dataset and why it gave dishonest scores | [EDA.md](EDA.md), `datasets/_archive_pre_r1_20261015/` | (analysis of 2026-10-15) |
| how did the numbers get here, and what did each training run cost? | [TRAINING_HISTORY.md](TRAINING_HISTORY.md), `results/training_history.json` | `tools/training_history.py` (reads every sweep, log and job file) |

## 2. The recordings (`datasets/`)

| path | holds |
|---|---|
| `<person>/<session>/<GESTURE>/` | one clip = `*_raw.csv`, `*_raw.npz`, `*_raw_meta.json` (label, duration, measured sample rate, quality findings, activity detection, config snapshot, parent take); format in [DATASET.md](DATASET.md) |
| `<person>/<session>/_takes/` | the long free takes the clips were cut from (never training data) |
| `<person>/<session>/_deleted/` | recordings moved out by DELETE / REDO (moved, never erased) |
| `<person>/calibration.json`, `calibration_log.jsonl` | the person's current live calibration and every calibration they ever had (append-only) |
| `_archive_pre_r1_20261015/` | the first 325-recording dataset (one person, plus 3 quarantined), kept as a baseline; README inside |
| `_corrected_Dano_session_03_to_Barbora/` | the originals of the 43 recordings (plus 13 quarantined) that had been saved under the wrong person (README inside) |
| `app_settings.json` | the application's saved settings: model selection, training switches, prompts (`gestures.json`, the edited gesture set, is written only once the set is edited; there is none now) |

Empty folders (`Michal/`, `Dano/session_02`, `Dano/session_09`) were never
filled; discovery ignores them.

## 3. Results (`results/`)

| path | holds |
|---|---|
| `experiments.json` | **everything the report quotes**, in one file: `meta`, `models` (25 zoo models with scores, sizes, latency, per-person), `ensembles`, `ablation`, `variants`, `without`, `voting`, `new_models`, `defaults`, `tests`, `dataset`, `starred_votes` |
| `dataset_summary.json` | the dataset description of section 1 above |
| `exclusion.json`, `exclusion_per_person.csv`, `exclusion_switches.csv` | trained with / without a person |
| `ablation/ml/`, `ablation/nn/` | `runs.jsonl` (one line per run and fold: accuracy, F1, robustness, size, latency), `oof_<config>_s<seed>.npz` (the out-of-fold probabilities of every window), `log.txt`; ML also `fallback.json` (adaptation statistics missing) |
| `ablation_without_barbora/` | the same ablation with the weakest person left out of training |
| `zoo_eval/` | `data_ml.npz`, `data_nn.npz` (labels, person and recording of each scored window, dataset digest) and `oof_<model>_s<seed>.npz` (every zoo model's probabilities, 3 seeds ML, 2 nets) -- the raw material of every zoo score and of `vote_stats` |
| `zoo_eval_without_barbora/` | the same for the models trained without her |

## 4. Models and their reports (`models/`)

| path | holds |
|---|---|
| `gesture_model.joblib`, `gesture_nn.pt` + `.json` | the **deployed** classifier and net (with the statistics sidecar the app reads) |
| `gesture_panel.joblib` | the DECISION TRAIL's data for the deployed classifier |
| `model_spec.json`, `nn_spec.json` | what the next RETRAIN trains |
| `zoo/<id>/` | `meta.json` (scores, setup, compat, dataset digest) and the model; `zoo/ensembles.json` the best voting subsets |
| `comb_test/20261022-003918/` | the ML + NN combination search: `ml_result.json`, `nn_result.json`, `nn_results.jsonl`, `csv/*.csv` (all models, best vote per setup, nested check, NN configs, window length, confusion), `comb_test_results.xlsx`, `ml_oof/`, logs |
| `reports/sweep_*.json` (29), `nn_sweep_*.json` (8) | the full report of each classifier / net training sweep (timestamped, kept); all read and tabulated in [TRAINING_HISTORY.md](TRAINING_HISTORY.md) |
| `reports/bench_models_20261022.json` | per-window timing of the deployed model and the net |
| `reports/decisions.json` | **out of date**: a decision trail written 2026-10-14 for the then 7-member vote on the one-person, 119-recording dataset (3162 windows). The current deployed model's trail is in `gesture_panel.joblib`; regenerate this file with `training/explain_votes.py --json` (which also rebuilds the panel) before using it |
| `reports/new_models_test_run1/` | the newer-architectures test (out-of-fold probabilities, losses) |
| `jobs/` | the logs of background training jobs |
| `gesture_model_<stamp>.joblib`, `gesture_nn_<stamp>.pt` | archived earlier deployed models (provenance of past numbers) |

## 5. Things to know before quoting a number

- **Mean over people, not pooled.** Every accuracy in the report is the mean
  over the five held-out people. Pooling windows gives a different number
  (79.1 % against 77.6 % for the ET + RF pair).
- **Neural-net numbers are few-seed.** The zoo's nets have 2 seeds and move a
  lot: NN-gru 72.4 % on one seed, 63.0 % on the other (mean 67.7 %); the 72.4 %
  quoted in older documents is one seed.
- **Same data, same folds.** All zoo and ablation probabilities come from
  leave-one-person-out folds on the dataset with digest `237a717be2327de9`
  (211 recordings); a score is only comparable with another measured on those
  folds.
- **ML zoo models use adaptation to the person; the nets do not;** so a zoo ML
  score differs from the same model in the combination test.
- **The scores are for people who calibrated:** a calibrated subject-grouped
  score is not zero-shot (CLAUDE.md 8k).

## 6. What was checked by re-computing, not trusting (2026-10-28)

- **Raw recordings**: all 211 clips and 30 free takes read; no NaN, timestamps
  monotonic, no clipping, no dead channel, every csv identical to its npz
  ([DATASET_SUMMARY.md](DATASET_SUMMARY.md) section 1, re-run by
  `tools/dataset_summary.py`).
- **Manifests against disk**: one gap, Betka's first 7 THUMB_UP clips
  (DATASET_SUMMARY section 4d; CLAUDE.md 8p).
- **Combination search**: all 592 CSV rows re-scored from `ml_oof/*.npz` to
  0.005 points (576 are distinct); every probability row sums to 1, no NaN;
  the NN CSV equals `nn_results.jsonl`.
- **Ablations**: every one of the 99 + 21 runs (and 72 + 21 without Barbora)
  re-computes exactly from its `oof_*.npz`; the six models trained without
  Barbora and the 25 zoo models match their `meta.json`; the newer-models run
  matches for its 60 held-out-person folds.
- **Logs**: no traceback, failure or stall in any log. The one failed run is
  the first retrain after the combination search ("Nothing could be
  evaluated", a trainer bug fixed the same day, CLAUDE.md 8n); the second
  retrain succeeded.
- **Reproducibility**: the final GRU run was made twice
  (`log_final_nn_gru07.txt`, `_2.txt`) and gives the identical 72.42 % per
  window, so a seed is deterministic; the zoo's second seed gives 63.0 %.

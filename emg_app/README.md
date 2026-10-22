# `emg_app` — the application package

Until 2026-10-22 this was one 21 400-line file, `emg_dataset_tool.py`. It is
now split by topic. `emg_dataset_tool.py` at the project root is still the
entry point (`python emg_dataset_tool.py`) and still re-exports every name,
so `import emg_dataset_tool as T` keeps working everywhere.

## The two rules that make the split safe

1. **Settings are read live from `config`.** Every module refers to a
   USER CONFIGURATION value as `cfg.NAME` (`from emg_app import config as
   cfg`), never as a copied global. A value changed at runtime — by the GUI's
   `_set_global`, by a test, or as `T.NAME = ...` on the entry point, which
   forwards the assignment — is therefore seen by every module at once.
2. **Imports point one way only.** Lower layers never import higher ones:
   `config` → data layer → `acquisition` / `neural` / `jobs` / `sequence` →
   `prediction` → `gui`. The data layer imports no Qt and no torch. A type
   hint that would point the other way is imported under `TYPE_CHECKING`.

## Module map

| Module | What is in it |
|---|---|
| `config.py` | USER CONFIGURATION: every tunable, the device profile, optional-dependency probes |
| `utils.py` | paths, slugs, `unique_path`, time formatting |
| `signal_processing.py` | moving RMS, envelope, band-pass, normalisation, activity detection |
| `calibration.py` | per-wearer calibration: maths, quality gates, storage, training tables (CLAUDE.md 8k, 8n) |
| `features.py` | feature families, `model_feature_vector` (the training/inference contract), `nn_input_block`, windowing, augmentation |
| `quality.py` | recording quality checks |
| `markers.py` | markers and notes sidecars |
| `recording.py` | `Recording`, the writers, `DatasetWriter`, single-file loaders, filename parsing |
| `model_spec.py` | model-spec schema and validation (sklearn-free), estimator descriptions |
| `compute.py` | the GPU/CPU probe (torch runs in a subprocess) |
| `dataset.py` | discovery, `load_dataset`, `make_windows`, person folds, fingerprint, model provenance |
| `acquisition.py` | `SampleRing`, `Capture`, `MyoAcquisition` — the only code that touches the serial port |
| `myo/` | the Myo BLE driver (`myo_raw.py`, `common.py`) — CLAUDE.md section 3 |
| `neural.py` | the neural-net engine and its line protocol to `training/nn_serve.py` (no torch) |
| `jobs.py` | background training jobs (CLAUDE.md 8m) |
| `sequence.py` | `build_plan`, `SequenceController` |
| `model_test.py` | TEST MODEL scoring and its log |
| `decision_panel.py` | the DECISION TRAIL panel model |
| `prediction.py` | `LivePredictor` |
| `app.py` | `main()`, `--summary` |
| `gui/qt.py` | the guarded Qt imports |
| `gui/tokens.py`, `gui/theme.py` | design tokens; stylesheet and class colours |
| `gui/widgets.py` | small shared widgets and layout helpers (`fit_to_screen`, `scroll_dialog_body`, ...) |
| `gui/plot.py` | `EmgPlotWidget` |
| `gui/decision_trail.py` | MODEL STATS views: DECISION TRAIL and FEATURE SPACE |
| `gui/workers.py` | dataset export worker, `TrainingJobWatcher` |
| `gui/dialogs/` | one file per dialog: calibration, model_spec, model_stats, model_test, neural_setup, neural_stats, gesture_setup, quality, trim |
| `gui/main_window/` | `MainWindow` = `window.py` (layout, timers, control state, closing) + one mixin per feature: `session`, `recordings`, `labeling`, `takes`, `markers`, `prediction`, `neural`, `training` |

`MainWindow`'s mixins are plain classes whose methods run on the window and
share its widgets and state through `self`; only `window.py` calls `super()`.

# Verification: independent check of the finished application

Agent D, 2026-10-27, Windows 11 (GTX 1650, 7.9 GB RAM), plus WSL Ubuntu 24.04.
Written from running things, not from reading the code. Evidence paths point to a
scratch folder outside the repository (`...\scratchpad\verify\`, abbreviated `V\`).

**What was and was not touched.** Only this file is new in the repository. Test
runs rewrote `tests/last_run.json`; it was restored from git afterwards, and
`tests/last_run_partial.json` (git-ignored) was deleted. The three uncommitted
changes below were already present when the check started and were not made by
it. Nothing was trained, `train_zoo.py`, `ablation_*.py` and `make_ui_guide.py`
were not run, and `datasets/`, `models/` and `results/` were only read (the GUI
checks worked on a temporary copy of `datasets/`).

**Known limitation, not hidden.** No Myo armband and no BLED112 dongle were
available. Everything that needs the hardware was checked on stored recordings
and simulated input only, or not at all (see item 7, icon and taskbar).

## Summary

| # | Item | Verdict |
|---|---|---|
| 1 | Clean start (venv, `--summary`, GUI start, WSL repeat) | **NESPLNENÉ** (Windows steps pass; WSL `test_jobs.py` fails on Linux) |
| 2 | Automatic tests (`tools/run_tests.py`, all suites) | **SPLNENÉ** (22/22 suites, 2233/2233 checks, with torch) |
| 3 | New features in the running window | **NESPLNENÉ** (one of 59 checks fails: adaptation is not in the vote pre-check) |
| 4 | OFF means original behaviour | **SPLNENÉ** (OBMEDZENIE: trainer equivalence by diff and test, not by re-training) |
| 5 | Numbers in REPORT / FEATURES vs `experiments.json` | **NESPLNENÉ** (all 455 numbers match; prose claims do not, see below) |
| 6 | Screenshots and links | **NESPLNENÉ** (two question screenshots show typed-in states; links resolve) |
| 7 | Definition of done | **NESPLNENÉ** (icon has no settings switch; docs stale; repo not clean) |
| 8 | Other findings | see section 8, ordered by severity |

### Every NESPLNENÉ, with the place to fix it

| Where | What to change |
|---|---|
| `models/gesture_model.joblib` vs `docs/REPORT.md:142`, `docs/NOTES.md:98`, `docs/FEATURES.md:97`, `docs/FEATURES.md:202`, `tools/make_report.py:263`, `tools/make_figure_new_models.py:70` | The deployed model is a four-member soft vote (RF-500-leaf3, ET-500, svm_rbf, lda), not the ET-500 + RF-500-leaf3 pair. Correct the text (or deploy the pair; that is the operator's decision per CLAUDE 8n). |
| `emg_app/zoo.py:283-288` (`COMPAT_KEYS`) and `emg_app/zoo.py:291` (`check_vote`) | Add adaptation to the compatibility check, as `emg_app/prediction.py:89-112` already does at load time. CLAUDE 7c and FEATURES 3 say the dialog refuses it. |
| `tools/make_ui_guide.py:613-637` and `docs/images/ui_question_calibrate.png`, `docs/images/ui_question_retrain.png` | Build the screenshots from the real stored state, or caption them as examples. |
| `tools/make_figure_new_models.py:70` and `docs/figures/novesie_modely_test.png` | The label "ML vote ET+RF (deployed) 77.6 %" is wrong on two counts (not deployed; 77.0 % in REPORT). |
| `tools/make_figures_ablation.py:119` and `docs/figures/ablation_effects.png` | The grey band uses only the baseline's seed spread; the judge uses the larger of the two. |
| `emg_app/config.py`, `emg_app/gui/dialogs/settings_tabs.py` (`GeneralTab`, line 344), `emg_app/app.py:88` | No switch for the icon exists. Add one or change the requirement. |
| `docs/install_windows.md:51`, `:58`, `:201`, `:205-216`; `docs/install_linux.md:63`, `:68-69` | scikit-learn 1.9.1 and the `InconsistentVersionWarning` are no longer true: `requirements.txt` pins `<1.8` and the saved model loads with 1.7.2 without a warning. |
| `docs/MANUAL.md:1350` | `test_gui.py` is 109 checks, not 91. |
| `tests/test_zoo.py:333` | `import torch` is unguarded in section 4; the suite fails with 0/0 in a torch-free install. |
| `tests/test_jobs.py:297` | `taskkill` is Windows-only; the suite fails on Linux. |
| `tools/make_report.py:382` (REPORT:270) | "the slow suites are run separately" is wrong: `tools/run_tests.py` runs them unless `--fast`. |
| `tools/make_report.py:110`, `docs/REPORT.md:17` and the defaults text | The rule says "2 x the seed noise"; `training/ablation_stats.py:74` uses the larger of two spreads. Say so, and give the sensitivity (section 5). |
| `git status` | `datasets/Dano/calibration.json`, `datasets/Dano/calibration_log.jsonl` and `emg_app/gui/main_window/prompts.py` are uncommitted; three agent worktrees are still registered (`git worktree list`). |

## 1. Clean start

**Verdict: NESPLNENÉ** -- the Windows steps pass; the WSL repeat fails in one
suite, and the install guide's stated versions are out of date.

Windows, fresh venv outside the repo (`V\venv_clean`, `python -m venv`, Python
3.14.0):

- `pip install -r requirements.txt`: exit 0. Resolved `scikit-learn 1.7.2`,
  `numpy 2.5.4`, `PySide6 6.12.0`, `pyqtgraph 0.14.0`. No torch installed.
  Evidence: `V\verify\pip_install.log`. **The guide says 1.9.1 (install_windows.md
  line 51); that is out of date, see the table above.**
- `python emg_dataset_tool.py --summary`: exit 0, `Raw recordings: 211`, five people
  (Barbora 43, Betka 41, Bruno 43, Dano 44, Miso 40 = 211). Evidence:
  `V\verify\summary.log`.
- Real entry point started from the venv, offscreen, under `timeout 45`:
  `window up after 5.6 s`, `Myo error: Myo dongle not found!`, still running at the
  timeout (exit 124). Evidence: `V\verify\venv_app_start.log`.
- The saved model loads from the venv: `LivePredictor(MODEL_PATH).load()` -> True,
  model name `Vote-soft-4`, with no version warning under scikit-learn 1.7.2.

WSL Ubuntu 24.04 (Python 3.12.3, a fresh venv in `~/venv_emg`, project copied to
`~/emg_verify`, not into the repo; script `V\verify\wsl\setup_wsl.sh`):

- `pip install -r requirements.txt`: exit 0; `--summary`: 211 recordings.
- The GUI started offscreen: `window up after 2.7 s` / `4.5 s`, dongle message.
- `tools/run_tests.py --fast`: **18/20 suites, 1867/1867 checks**.
  - `jobs` FAILS, 0/0: `tests/test_jobs.py:297` calls `taskkill` (Windows only),
    `FileNotFoundError: 'taskkill'`. Evidence: `V\verify\wsl\run_tests_linux_fast.txt`.
  - `zoo` FAILS, 0/0: needs torch (section 2).

## 2. Automatic tests

**Verdict: SPLNENÉ** -- the full run passes and matches the recorded numbers.

`python tools/run_tests.py` with the system Python (it has torch and CUDA, so
`nn` and `custom_gesture` run, not just skipped). Output:
`V\verify\run_tests_system_full.log`; the results file it writes was saved to
`V\verify\last_run_NEW_system.json` and then restored from git.

| suite | checks | suite | checks |
|---|---|---|---|
| datalayer | 150/150 | predict | 109/109 |
| gui | 109/109 | trail | 103/103 |
| review | 50/50 | calibration | 161/161 |
| label_on_graph | 48/48 | jobs | 48/48 |
| markers | 88/88 | class_mask | 121/121 |
| free_take | 268/268 | device | 104/104 |
| model_spec | 346/346 | sweep_cost | 48/48 |
| filters | 52/52 | no_torch | 11/11 |
| icon | 15/15 | zoo | 108/108 |
| app_settings | 66/66 | defaults | 18/18 |
| nn | 162/162 | custom_gesture | 48/48 |

**22/22 suites, 2233/2233 checks** (suite time for `custom_gesture`: 187.6 s). The
per-suite counts equal `tests/last_run.json` at HEAD and `docs/REPORT.md` section
11 for all 22 suites. `docs/figures/tests_overview.png` shows the same numbers.

Torch-free environments: in the fresh venv (no torch), `run_tests.py --fast` gave
**19/20 suites, 1915/1915** (`zoo` fails, `V\verify\run_tests_venv_fast.log`).
The install guide does not promise `zoo` without torch, but `tools/run_tests.py`
calls every non-slow suite "offline", so the label is wrong (section 8).

## 3. The new features in the running window

**Verdict: NESPLNENÉ** -- 58 of 59 checks pass; the one failure is real.

Script: `V\verify\gui_checks.py`, on the real `MainWindow` (offscreen, no dongle),
against a temporary copy of `datasets/`. Log: `V\verify\gui_checks.log`, results
`V\verify\gui_checks_result.json`. The NN checks were also run with the system
Python (torch, CPU worker).

Checked and passing:

- **Settings tabs** are Gestures, Models, Training, General.
- **Zoo listing**: the ML table has 16 rows, the NN table 9 (25 entries on disk).
- **Stars, recomputed independently** from `models/zoo/*/meta.json` (top 3 per
  family by `window_acc`, ties by `macro_f1`): ML = ET-500-leaf3, RF-500, RF-500-leaf3;
  NN = NN-chan_transformer, NN-gru, NN-tsmixer. The UI shows the same sets.
- **Vote stars**: `Select best combination` is disabled for both families. Recomputed
  from `models/zoo/ensembles.json`: nested 0.7703 < best single 0.7764 (ML),
  nested 0.6909 < best single 0.7242 (NN), so no vote is starred. Correct.
- **Size display**: ticking the three top ML models shows `3 selected | total size
  22.0 MB`; the independently summed bytes give 22.0 MB.
- **One ML model** loads (`ml-rf-500-leaf3`, file path checked); probabilities on a real
  recording window sum to 1.
- **Soft vote, ML** (`ml-et-500` + `ml-rf-500-leaf3`): the live answer equals the mean of
  the two single-model predictors' probabilities on the same window: **max difference
  0.00e+00**.
- **Refusal with reason**: different windows (`window_s` 1.0 vs 0.5) is refused with
  `"... disagree on window_s (1.0 against 0.5), so their probabilities cannot be averaged"`.
  ML + NN in one selection is refused (`zoo.check_vote`: "not voted together").
- **Soft vote, NN** (`nn-gru` + `nn-tsmixer`, CPU worker): the vote equals the mean of the
  two nets' softmax outputs (each run on its own), **max difference 2.98e-08**; each
  net's output sums to 1 (0.99999992).
- **Augmentation / adaptation switches reach the trainers.** With the Training tab set
  to gain+rotate and adapt on: `--augment gain,rotate --adapt`. Defaults: `--augment none
  --no-adapt`; zoo: `--augment-ml none --adapt-ml on --adapt-nn off`. The command lines
  are accepted by the real argparse (`python -u training/train_nn.py <args> --help`
  and `train_zoo.py ... --help` both exit 0 with the GUI's own arguments). No training
  started.
- **Calibration question**: the stored Dano calibration (2026-10-26 12:30, 17.4 h old at run
  time) is **not** offered (reason: "more than 8 h"). A calibration made before the run and
  under 8 h old **is** offered, with the arm warning. One made during the run is not offered.
  The question is asked **once** per run per person; CANCEL stops the start; USE LAST
  logs its use; setting OFF means no question.
- **Retrain question**: with reasons present it is asked once; never while a training job or a
  session runs; setting OFF means not asked; "do not ask again" switches the setting off
  and saves it to the temporary `app_settings.json`; "Not now" starts nothing. On this
  dataset the real reason list is empty (zoo digests match), so the question correctly
  does not appear now.
- **Adaptation statistics** change the adapted ML model's answer (max change 0.397).
  With `ADAPT_LIVE` off the statistics are not applied (`adapt_applied` False), and the
  answer equals a fresh load **exactly** (max difference 0.00e+00).
- **App icon**: all six PNG sizes and `app.ico` exist and load (16 to 256 px). A missing
  folder gives an empty icon and no exception; `apply_app_icon` sets the window and the
  application icon; the AppUserModelID call returns a bool and is guarded.

Failing:

- **FAIL: "a vote mixing adapted and plain models is refused with the reason"** (the
  Models-tab pre-check, `zoo.check_vote`). `check_vote` accepts the pair; `COMPAT_KEYS`
  (`emg_app/zoo.py:283-288`) has no adaptation key. CLAUDE 7c and FEATURES 3 both state
  that the dialog refuses. The loader does refuse: `emg_app/prediction.py` `_vote_problem`
  returns "differ in adaptation to the person" (checked directly, PASS in the same run).
  So the user sees "Soft vote" in the dialog and gets the refusal only at load.

## 4. OFF means the original behaviour

**Verdict: SPLNENÉ, with one limitation** (training was not re-run; see below).

| Feature | Check | Result |
|---|---|---|
| Model selection OFF (ticks saved) | the deployed model answers, compared with a fresh `LivePredictor(MODEL_PATH)` on the same window | **bit-identical** (`np.array_equal`) |
| Model selection OFF, NN | `NeuralEngine()` (default path) vs `NeuralEngine(model_path=NN_MODEL_PATH)` on the same window, both on CPU | **bit-identical** (`V\verify\nn_off.py`: max difference 0.0) |
| Model selection OFF, ticks | the saved ticks are kept | yes (one click to switch back) |
| ADAPT_LIVE OFF | person statistics not applied; answer equals a fresh load | **yes**, exact equality |
| Augmentation / adaptation OFF in `train_nn` | command line `--augment none --no-adapt` | accepted; code path compared with `git show 4077bc4:training/train_nn.py` (see below) |
| Two questions OFF | no question, no training | yes (section 3) |
| Icon missing | `assets/icons` replaced by a missing folder (monkeypatched path) | no exception, default icon, window still starts |

**Limitation (OBMEDZENIE).** Training may not be run here, so the weights were not
compared. What was checked instead:

- `git diff 4077bc4 HEAD -- training/train_nn.py` (142 insertions, 67 deletions) shows
  that with augmentation and adaptation off `prepare_fold` calls the same `fold_scaler`
  path as before and `train_one` receives the same arrays (`augmenter=None` leaves the
  batch untouched).
- `tests/test_zoo.py` section 10 runs the previous commit's trainer
  (`git show 4077bc4:training/train_nn.py`) and compares folds and confusion matrices;
  it passed in the full run (108/108).

## 5. Numbers

**Verdict: NESPLNENÉ** for the prose; the tables are correct.

**Tables.** `V\verify\check_numbers.py` parses every generated table in
`docs/REPORT.md` and `docs/FEATURES.md` and compares it with
`results/experiments.json` and `tests/last_run.json`: **455 values, 0 mismatches**
(ML and NN rows including sizes, latency, stars and trained-with; ablation,
robustness, voting, new-models, latency, test counts and the defaults rows).
Negative control: two planted errors (80.8 -> 80.9, 76.9 -> 77.9) were both caught.

**Raw data, recomputed from the files on disk:**

- The 25 zoo models (`results/zoo_eval/oof_*_s*.npz` against `data_ml.npz` / `data_nn.npz`):
  per-window accuracy (mean over the five held-out people, averaged over seeds)
  **matches meta.json and experiments.json for all 25**. Macro-F1 with the trainer's own
  definition (`training/train_zoo.py` `macro_f1`, per person, six fixed classes, a class
  with no true positive scores 0, then averaged) **matches for all 25**
  (`V\verify\check_f1.py`).
- The ablation vote `vote(ET+RF)`: 77.01 % from the raw OOF files (seeds 77.58, 76.67,
  76.79) against 77.01 in experiments; ET-500 alone 76.90 against 76.90.
- **NN ablation, all 21 runs** (`results/ablation/nn/runs.jsonl`, judged with
  `training/ablation_stats.judge`): 51 values, every verdict and every delta **reproduces**
  (`V\verify\check_raw.py`).
- **ML adaptation, from the raw runs.** The shipped vote (`vote(ET+RF)`):
  delta +4.30 points, 3/5 people better on accuracy, robustness delta +3.74 with 4/5
  people better, seed spread 0.29 (baseline) and 1.67 (adapted) -> bar 2 x 1.67 = 3.34.
  **It passes by 0.4 points** on robustness only; accuracy fails (bar 4.94). The single
  ET-500 with the same setting does not pass (robustness +3.48, `no evidence`).

**Prose the numbers do not support.**

1. "The deployed pair's vote is 77.0 %" (REPORT:142). The file the app loads, and
   that the app answers with, is `models/gesture_model.joblib`: a `VotingClassifier` of
   four members (RF-500-leaf3, ET-500, svm_rbf, lda) with **no adaptation** (payload
   `adaptation` is None). The live predictor reports `Vote-soft-4`. REPORT's own table
   measures that four-model vote at **69.4 %**. So the live classifier is about eight points
   below what the report calls "deployed". NOTES:98, FEATURES:97 and FEATURES:202 repeat
   the claim, and so does the figure label (section 6). CLAUDE 8n already records the
   four-model vote at 69.5 %, so the project knows; the report and the figure do not say it.
2. The rule (REPORT:17, FEATURES, `tools/make_report.py:110`) says "at least max(1 point,
   2 x the seed noise)". The code (`training/ablation_stats.py:74`) uses the **larger** of the
   baseline's and the candidate's seed spreads. For the NN adaptation row, the baseline's
   spread is 4.69 (bar 9.38), but the candidate's own spread 1.48 gives bar 2.96, and the
   gain of 4.51 with 4/5 people better would **pass**. So "no evidence" depends on that
   unstated choice. The NN adaptation gain is also one person: Barbora +26.6 points, the
   others between -7.4 and +2.1 (`check_raw.py` output). The "no evidence" verdict is
   defensible for that reason, but the report should say so.
3. ML adaptation default **ON**: passes only on robustness, by 0.4 points, and only for the
   two-model vote (section above). The text "helps robustness: +4.3 points per window"
   gives the accuracy gain, which does not pass the rule.
4. REPORT:270 says the slow suites "are run separately". `run_tests.py` runs them by default.
5. The defaults apply only to the zoo trainer. NOTES "Open" says the classic RETRAIN sweep
   (`train_model.py`, the one that produces the deployed model) trains plain models and
   ignores both switches. FEATURES presents the defaults without that scope.
6. Macro-F1 is undefined in the report. It is the per-person mean with six fixed classes and
   absent classes scored 0 (`train_zoo.py:86-94`). It is not sklearn's pooled macro-F1, which
   would give different numbers (for example 80.3 instead of 77.8 for ET-500).
7. `GRU (current NN)` appears as 72.4 % (one seed, 150 epochs, `new_models_test`) and as 67.7 %
   (three seeds, zoo and ablation). Both are labelled, and NOTES explains the seed spread;
   a reader can still be confused.

## 6. Screenshots and links

**Verdict: NESPLNENÉ** (the links pass; the screenshots do not all match).

**Links.** An inline Python run (not saved): **122 relative links** in `README.md`,
`docs/*.md`, `tests/README.md` and `emg_app/README.md`, including every image reference.
All resolve except the three links to `docs/VERIFICATION.md`, which this file now
satisfies.

**Screenshots I looked at.**

| File | Matches the current GUI and data? |
|---|---|
| `docs/images/ui_main.png` | Yes: gear icon top right, banner, status bar, cards without nested outlines. Callout text overlaps a little (callout 1 over "capture.", callout 9 on the START SESSION label). |
| `docs/images/ui_settings.png` | Four tabs. The counts (44, 29, 35, 32, 36, 35 recordings) equal the `--summary` totals. The FIST row shows "on" beside a box that looks unchecked while the callout says FIST is off; not resolved by zooming here. |
| `docs/images/ui_settings_models.png` | Yes: ticks and stars and the numbers (RF-500-leaf3 80.8 %, 78.6 %, 4.8 MB; NN-gru 67.7 %) match the zoo; total 22.0 MB for three ticked models. |
| `docs/images/ui_settings_training.png` | Yes: "Defaults in this build -- NN: augmentation none, adaptation off; ML: augmentation none, adaptation on." Some callout badges cover labels. |
| `docs/images/ui_settings_general.png` | Yes (two switches). |
| `docs/images/ui_question_calibrate.png` | **No.** It shows "reuse the last calibration made 2026-10-27 08:10 (2.3 h ago)". The stored Dano calibration is from 2026-10-26 12:30, more than 8 hours old, and would not be offered. The image comes from a hard-coded offer (`tools/make_ui_guide.py:613`). |
| `docs/images/ui_question_retrain.png` | **No.** It says "WRIST_FLEX ... 40 recordings of it exist". There are 0 WRIST_FLEX files on disk, and `ui_settings.png` says "nothing recorded yet" for the same gesture. Built from fixed text in `tools/make_ui_guide.py`. |
| `docs/figures/ablation_effects.png` | The numbers and verdicts match. The ML grey band (about +-1 point) is drawn from the baseline spread only; the judge's bar for the ML adaptation row is 4.9. Source: `tools/make_figures_ablation.py:119`. |
| `docs/figures/novesie_modely_test.png` | The new-architecture numbers match REPORT section 5. The label "ML vote ET+RF (deployed) 77.6 %" is wrong (not the deployed model; 77.0 % in REPORT). Source: `tools/make_figure_new_models.py:70`. |
| `docs/figures/tests_overview.png` | Yes: 2233/2233 and each suite's count equal `tests/last_run.json`. |

## 7. Definition of done

| Item | Verdict | Evidence |
|---|---|---|
| All models trained, selectable and correctly starred | **SPLNENÉ** | 16 ML + 9 NN on disk, OOF scores reproduce (section 5); selectable (section 3); stars recomputed independently (section 3). The deployed model is not the one the report calls deployed (section 5, item 1). |
| Defaults backed by the Phase-3 numbers and justified | **OBMEDZENIE** | Every default is generated from `experiments.json`, and the verdicts reproduce from the raw runs. The NN adaptation verdict depends on an unstated choice; the ML adaptation default passes on robustness by 0.4 points (section 5). |
| Every new feature switchable off in the settings | **NESPLNENÉ** | Model selection, voting (untick), augmentation and adaptation (Training tab), both questions (General tab) are switchable and verified. **The icon has no switch** (`emg_app/app.py:88`, `emg_app/gui/icon.py`); FEATURES 9 says to remove the folder. |
| Icon in the window corner and taskbar | **OBMEDZENIE** | Verified headless: files and sizes; `apply_app_icon` sets window and application icon (`app.py:88` runs before `MainWindow()` at line 105, so the AppUserModelID is set before the first window). The taskbar and the corner need a real desktop; not seen. |
| Tests ran and results visualised | **SPLNENÉ** | Section 2; `docs/figures/tests_overview.png` matches. |
| Documentation, tables, README, screenshots, report, Myo SDK, Windows and Linux install current | **NESPLNENÉ** | The install guides' versions and warning are out of date (section 8, item 5); the MANUAL count is 91; the "deployed pair" text; two question screenshots; the "run separately" line. The Myo SDK guide (`docs/sdk_myo.md`, 658 lines) was not checked against hardware (OBMEDZENIE). |
| Agents communicated on the board | **OBMEDZENIE** | Sections for A, B, C and D are present in `scratchpad\board\BOARD.md`. `coordination/BOARD.md` does not exist yet; main's copy step is outstanding. C's request (nested frames in `_stat_tile`) is in place: the tile uses `QFrame#card` (`emg_app/gui/widgets.py:434-435`), so no nested outline is drawn. |
| Everything committed and pushed | **NESPLNENÉ** | `origin/main` = `main` = 39c1dea (nothing unpushed), but `git status` shows three modified files that are not committed; three agent worktrees are still registered (all their branches are merged). |

## 8. Other findings, ordered by severity

No data-loss risk was found. Nothing deletes or rewrites recordings in the checks
above; the retrain question starts nothing when declined; calibration history is kept
in `calibration_log.jsonl` (the modified copy is one line longer than the committed one).

1. **HIGH, honesty: the "deployed" model is not the pair the documents describe.**
   Section 5, item 1. The live classifier is a four-member soft vote with no adaptation,
   measured at 69.4 %; REPORT, NOTES, FEATURES and a figure describe ET-500 + RF-500-leaf3
   at 77.0 % as deployed. Whether to deploy the pair is an operator decision (CLAUDE 8n).
2. **MEDIUM: the vote pre-check misses adaptation** (`emg_app/zoo.py:283-291`). The loader
   refuses at load time, so nothing is averaged wrongly, but the dialog promises a refusal
   it does not give.
3. **MEDIUM: two screenshots show states that are not on disk** (`tools/make_ui_guide.py:613-637`);
   a reader could take the question wording as current data.
4. **MEDIUM: the icon cannot be switched off in the settings** (DoD item).
5. **MEDIUM: the install guides are out of date.** `requirements.txt` pins
   `scikit-learn>=1.3,<1.8`, so a clean install gets 1.7.2, which loads the shipped model
   without a warning (checked in both environments). `install_windows.md:51,201-216` and
   `install_linux.md:63,68-69` still say 1.9.1 and warn. Both guides also say the
   `tools/run_tests.py` runner is not in the branch (install_windows.md:156)
   although it is.
6. **MEDIUM: two test-suite portability bugs.** `tests/test_zoo.py:333` imports torch
   unguarded (torch-free install: 0/0 failure, and `run_tests.py` calls it "offline");
   `tests/test_jobs.py:297` calls the Windows-only `taskkill` (Linux: 0/0 failure). On Linux
   the POSIX kill path of `training/train_job.py` is therefore not tested at all.
7. **MEDIUM-LOW: defaults rest on a narrow reading of the rule.** Sections 5, items 2 and 3:
   the NN adaptation verdict depends on taking the larger seed spread (`ablation_stats.py:74`);
   the ML adaptation default passes on robustness by 0.4 points for the vote only.
8. **LOW-MEDIUM: the adaptation default applies only to the zoo trainer** (NOTES "Open"); the
   classic RETRAIN, which trains the deployed model, ignores it (section 5, item 5).
9. **LOW: the ablation figure's grey band** understates the bar for the ML adaptation row
   (`tools/make_figures_ablation.py:119`).
10. **LOW: `docs/REPORT.md:270` and `docs/MANUAL.md:1350`** have the wrong description of the
    slow suites and the old count of 91.
11. **LOW: the repository is not clean.** Two tracked dataset files were rewritten by the
    app's calibration (the log is appended, so the history is kept); `prompts.py:138` has an
    uncommitted two-line change; three agent worktrees are registered.
12. **LOW: macro-F1 is not defined in the report** (section 5, item 6).
13. **LOW: callout overlaps and an ambiguous "on" caption in several screenshots**
    (`ui_main.png`, `ui_settings.png`, `ui_settings_training.png`).
14. **LOW: an empty person folder** `datasets/Michal/session_01` (no recordings; not
    listed by `--summary`). Harmless, but a subject typed as "Michal" would look valid.
15. **Not verified:** the BLED112 dongle and the armband, the taskbar icon, and training
    re-runs (forbidden for this check). The Linux app-exit timing was not measured reliably
    (the start itself worked).

## Evidence index (scratch, outside the repo)

- Install: `V\verify\pip_install.log`, `V\verify\venv_app_start.log`, `V\verify\summary.log`
- Tests: `V\verify\run_tests_system_full.log`, `V\verify\run_tests_venv_fast.log`,
  `V\verify\test_zoo_venv.log`, `V\verify\last_run_NEW_system.json`,
  `V\verify\last_run_HEAD.json`
- GUI: `V\verify\gui_checks.py`, `V\verify\gui_checks.log`, `V\verify\gui_checks_result.json`,
  `V\verify\nn_off.py`, `V\verify\trainnn_help.txt`
- Numbers: `V\verify\check_numbers.py` (455 values, 0 mismatches; `V\verify\neg\` is the
  planted-error copy), `V\verify\check_raw.py` (NN ablation), `V\verify\check_zoo_raw.py`,
  `V\verify\check_nn_raw.py`, `V\verify\check_f1.py`
- Linux: `V\verify\wsl\setup_wsl.sh`, `V\verify\wsl\setup.log`,
  `V\verify\wsl\summary_linux.txt`, `V\verify\wsl\run_tests_linux_fast.txt`
- Board: `V\board\BOARD.md` (Agent D section updated)

Scratch root: `C:\Users\danie\AppData\Local\Temp\claude\C--Users-danie-OneDrive-Po--ta--EMG\bf987bcb-91c4-4fc6-89e4-2e488a141b51\scratchpad\`


---

## Status after the fixes (main, 2026-10-27, after Agent D's check)

Agent D's findings were fixed, or answered, as follows. The check itself was not
repeated by D; the evidence for each fix is the test or command named.

| Finding | Action | Evidence |
|---|---|---|
| "Deployed pair" is wrong: `gesture_model.joblib` is the four-model vote | Texts corrected everywhere (`tools/make_report.py`, FEATURES, NOTES, `novesie_modely_test.png` title and legend now say "combination-test winner / stored result"). Whether to deploy the pair stays the operator's call (CLAUDE.md 8n). | `docs/REPORT.md` section 6, `docs/NOTES.md` |
| Vote pre-check ignored adaptation | `adaptation_mode` added to `zoo.COMPAT_KEYS` (ML); the 16 existing ML metas migrated; `train_zoo.py` writes it; the Models tab now flags and blocks such a vote before the loader would. | `tests/test_zoo.py` (109), `tests/test_app_settings.py` (67) |
| Question screenshots show typed-in values | Captions say "example values" (the numbers in a real box come from disk) | `docs/UI_GUIDE.md` 10d |
| No settings switch for the icon | `APP_ICON_ENABLED` (config, saved settings, General tab "Appearance"; applied at the next start; the saved settings are restored before the icon) | `tests/test_icon.py` (17), `tests/test_app_settings.py` |
| Install guides stale (scikit-learn 1.9.1, version warning) | Rewritten: `requirements.txt` pins `scikit-learn>=1.3,<1.8`, a clean install resolves 1.7.2 and the saved model loads without a warning (D verified exactly that) | `docs/install_windows.md`, `docs/install_linux.md` |
| `MANUAL.md:1350` 91 checks | 109 | `docs/MANUAL.md` |
| `test_zoo.py` needs torch; `test_jobs.py` uses `taskkill` | Torch-dependent sections are skipped (and said so) without torch; `test_jobs` kills the runner portably | `tests/test_zoo.py`, `tests/test_jobs.py` |
| "Slow suites run separately" wrong | Sentence corrected (they are in the count; `--fast` skips them) | `docs/REPORT.md` section 11 |
| Rule text vs `ablation_stats.judge` (larger of two spreads) | Stated in the report's methods; section 8 now says how firm the two adaptation verdicts are (NN: borderline, decided by the baseline's instability; ML: weakest default); the ablation figure now draws the judge's own per-row threshold | `docs/REPORT.md` sections 3 and 8, `ablation_effects.png` |
| macro-F1 undefined | Defined in the methods (per person over six fixed classes, then averaged) | `docs/REPORT.md` section 3 |
| ML adaptation default applies to the zoo only | Written in the report's limitations and NOTES | `docs/REPORT.md` section 12 |
| Repo not clean | Three agent worktrees removed after their branches were merged; `prompts.py` committed. **`datasets/Dano/calibration.json` and `calibration_log.jsonl` stay uncommitted on purpose**: they are the owner's own live calibration from 2026-10-26, not part of this work. | `git status`, `git worktree list` |
| Callout overlaps, empty `datasets/Michal/session_01` | Not changed (cosmetic; the empty folder is the owner's) | -- |

Full suite after the fixes: **22/22 suites, 2237/2237 checks**
(`python tools/run_tests.py`, `tests/last_run.json`).

Still limited, not hidden: no armband or dongle was ever available (hardware paths,
the taskbar icon on a real desktop, native Linux and Wayland are unverified), and
the edit above of the adaptation default rests on five people.

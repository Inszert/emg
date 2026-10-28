# PUBLICATION_CHECK.md -- `publication.tex` against the data (2026-10-28)

Every number in the paper was compared with the files it comes from
(`results/`, `models/comb_test/`, `models/zoo/`, `datasets/`) and the code that
makes it. This is what was found and changed. `publication.tex` itself was edited
in place, in the same TeX format, with CRLF line endings kept, the six red
`\textcolor{red}{...}` notes untouched, no existing image touched, and no
citation invented (the `TODO_*` keys are unchanged).

## 1. Corrected (the text was wrong or imprecise)

| where | was | now | why |
|---|---|---|---|
| Introduction | "from +8.0 to -2.7 points for the same ML model" | "+8.0 (Bruno) to -2.7 (Betka) for the ML models" | those are the mean of the three ML models per person (EXCLUDING_PEOPLE.md section 3), not one model |
| Introduction | "70--90 % for the other four people" | "72--94 % (ML) and 66--80 % (networks)" | per-person scores of the six starred models |
| Introduction | adaptation raised her "ML accuracy 46 -> 60 %" | adds GRU 37 -> 63 % | the GRU gain on her is larger (ablation, 3 seeds: 48.8/29.8/31.6 -> 59.0/71.6/59.4) |
| Abstract | "only adaptation helped the ML models" | calibration worth about four points, augmentation none, adaptation a further four | the conclusion already called calibration the most useful addition; calibration off costs 4.3 points |
| Abstract | "300 times smaller" | "300 times fewer learned parameters" | 1.29 million against 4 455; on disk the gap is larger (17.7 MB against 25 KB) |
| Abstract | 72.4 % for the GRU, no qualifier | "in the same search run (67.7 % averaged over seeds)" | 72.4 % is seed 0; seeds 72.4 / 63.0 / 67.8 |
| Dataset | "almost all are between 2 and 4 s" | "81 % are between 2 and 4 s" | 20 clips are under 2 s, 8 over 4.1 s |
| Model search (i) | "level features best at every window length" | "...except 0.4 s, where they tie" | at 0.4 s: level 71.1 %, both 71.2 % |
| Model search (i), Features | "band-passed signal" | "zero-phase 8 Hz low-pass ... called band-passed in our code" | for the recorded envelope stream the "bandpass" representation is a low-pass at 8 Hz (`EMG_BANDPASS_HZ["filtered"] = (None, 8.0)`) |
| Model search (ii) | "nine architectures ... trained on 0.7 s windows" | the real sequence: five architectures at 0.3-0.5 s, GRU and CNN-GRU at six lengths, four newer ones and ROCKET at 0.7 s | the nine were all at 0.7 s only in the zoo |
| How the best models work | "the total activation" | "the logarithm of the total activation" | `total_log` in `extract_level_features` |
| Evaluation, cosine | "0.978" | "0.98 in the recorded counts, 0.92 after calibration, 0.56 between the two waves" | recomputed on the current five-wearer data (0.977 / 0.916 / 0.560) |
| Evaluation | "the forest and the GRU make largely the same mistakes" | same three gestures, not always the same confusion (GRU fist -> thumb up +14 points, forest open -> right wave +13) | MODELS_COMPARED "The two models also make *different* mistakes" |
| Evaluation, voting | "searched all subsets of two to four of the eight best models (154 subsets). The best vote, of ET-500 and RF-500-leaf3 ..." | the combination test searched every subset of 2-4 of the **ten** best singles per setup plus greedy forward selection; the 154-subset (eight best) search is the zoo's, and its best was a four-model vote | two different searches had been merged |
| Evaluation, per wearer | GRU 92.2 / 69.8 / 73.9 / 74.6 / 28.4 % | three-seed means for ML and GRU (ML 45.9 / 82.7 / 79.8 / 88.9 / 87.7, GRU 36.7 / 86.0 / 69.2 / 74.7 / 72.1) | the old GRU numbers come from a second training pass inside the trainer whose mean is 67.8 %, not the 72.4 % of the run behind the other tables |
| Limitations | "wore it loosely" | "was observed by the operator not to wear it tightly; the fit was not measured" | EXCLUDING_PEOPLE.md: "observation from the operator, not tested" |
| Limitations | "how well the two calibrations agree has not been measured" | Dano's live and derived references differ by a factor 0.8-2.2; the other wearers have no live calibration | DATASET_SUMMARY.md section 5 |
| Rule wording | "twice the seed spread" | "twice the larger of the two seed spreads" | `training/ablation_stats.py`; matters for the GRU adaptation verdict (VERIFICATION.md 5.2) |

## 2. Added (all from measured data)

- **Previously empty subsections written:** Related Works (prose with the existing
  `TODO_*` citation keys and one new red note asking for the real references),
  Features, Classifiers, Soft-voting definition, NN Architecture, Protocol, ML
  experiments, NN experiments, ML/NN comparison, and one-line introductions for
  the calibration and hardest-wearer subsections.
- **Data:** the stream arrives in bursts (why windows are resampled onto a 60 Hz
  grid); every clip and take was read (no NaN, monotonic timestamps, no dead
  channel, highest count 1659; 2047 is an empirical ceiling, so the paper says
  "nominal range" and claims no saturation check); per-wearer channel levels and
  gesture similarity.
- **Results:** the unevenness of calibration across wearers (off: Bruno 80 -> 57,
  Betka 83 -> 90; global gain: Barbora 46 -> 75, Miso 88 -> 71); the unevenness of
  adaptation (ML: Barbora +14.6, Bruno -6.3; GRU: Barbora +26.6, Betka -7.3) and that
  it passes the rule only narrowly on robustness; the fallback without the wearer's
  statistics (78.2 % against 78.5 %); Barbora's errors are substitutions with
  unchanged confidence (0.60 right, 0.61 wrong); half of the windows are right for
  all of eight classical models and 13 % wrong for all of them; the price of an
  unseen wearer (97.4 / 95.3 % against 77.6 / 72.4 %); Betka's 17-window thumb up
  and why (her first seven clips were discarded, DATASET_SUMMARY.md 4d).
- **Two new figures** (new files, `tools/make_figures_paper.py`):
  `docs/figures/channel_profiles.png` (Fig. `fig:profiles`) and
  `docs/figures/generalisation_gap_paper.png` (Fig. `fig:gap`). The older
  `generalisation_gap.png` carries a repository reference in its footnote, so a
  clean version was made instead. Both are included with `\safefig`, so a missing
  file gives the red placeholder, not a LaTeX error.

## 3. Checked and correct (no change)

Dataset: 211 recordings, 5 wearers, Table 1, 10.6 minutes, 113-154 s per wearer,
49.9 Hz (46.5-51.2), median 151 samples / 3.0 s, shortest 1.0 and longest 9.5 s,
47 + 6 flagged recordings, one sitting per wearer on one day, rest 44 / fist 29,
1787 and 2873 windows, ML windows per gesture, majority class 18 %, PCA 61 %.
Models: 16 classifiers, 36 setups (3 x 3 x 4), top 15 all trees, 77.4 / 74.4 /
66.0 %, 1.0 s best, ROCKET 69.7 % with 169 thousand parameters and 8.95 ms, the
whole of Table 2 (sizes, ms, plain, final, gains), the GRU recipe (AdamW 3e-3,
weight decay 1e-3, label smoothing 0.05, class weights, batch 64, cosine, 150
epochs, width 32, dropout 0.25, no early stopping), 4 455 parameters, TSMixer
(24 features, two blocks, mean and max), the whole of Table 3, Table 4, the
gesture table, the amplitude-only control (77.8 / 77.6 and 63.3 / 72.4, losses of
26 / 20 / 14 points), speed (1.7 + 2 x 25 ms = 52 ms, 1.8 ms, 2.9 ms), 154 subsets,
79.3 / 77.0 / 77.6 %, 69.5 %, 77.6 / 74.0 / 77.4 %, the without-Barbora numbers
(85.4 -> 86.3, 73.5 -> 74.4, -5.7 to +9.5, 59.7 / 39.7 %), 17 to 102 windows per
cell, the conclusion's 5 to 13 points.

## 3b. Every document in `docs/` was read, and what it changed

Read in full or in the parts that carry facts: REPORT, MODELS_COMPARED,
EXCLUDING_PEOPLE, COMB_TEST, NOTES, DATASET, DATASET_SUMMARY, DATA_MAP,
TRAINING_HISTORY, ARCHITECTURE, FEATURES, ML_PIPELINE, EDA, calib_plan,
VERIFICATION, recommendation, problems, MANUAL, GUIDE, UI_GUIDE (the settings, model
stats and cutting sections), install_windows, install_linux, how_to_fix_long_training,
sdk_myo, UI_REVIEW (headings). None of them contradicts a number in the paper. What
they did turn up, and the edits made:

| document | finding | action |
|---|---|---|
| DATASET.md, sdk_myo.md | 2047 is an empirical ceiling, not a documented sensor limit | paper: "nominal range 0--2047", integrity sentence no longer says "sensor ceiling" |
| ARCHITECTURE.md section 4 | said `measured_rate()` uses the median interval; section 5a (and the code) use the span average | section 4 corrected |
| calib_plan.md 2.1 | "resting level is NOT wearer-specific" was measured on 2 wearers; with 5 it fails (Miso four channels at 86-112, Bruno two at 81-88, Barbora CH2 76) | dated note added above the table |
| GUIDE.md section 4 | stale defaults: `REPETITIONS` 30 (now 6), `RECORD_TIME` 2.0 (2.5), `RING_BUFFER_SECONDS` 60/180 (600), `STALL_TIMEOUT_S` 3.0 (6.0) | table values corrected against `emg_app/config.py`; other GUIDE prose (21 classifiers, two-wearer net numbers) is dated history and left as it is |
| MANUAL.md, GUIDE.md, UI_GUIDE.md | older two-wearer results (86.8 % LDA, 57.5 % net) appear as historical record | left; current numbers are in MODELS_COMPARED.md and the paper |

## 4. Open (needs you)

- **Related Works** needs real references and the closest studies' cross-subject
  accuracies; no reference was invented.
- **Live accuracy** (the four red notes): `datasets/<person>/model_tests.jsonl`, where
  TEST MODEL writes its results, does not exist for anyone, so nothing was measured.
- **`resources.bib`** is not in the repository (`\addbibresource{resources.bib}`).
- **The raw-signal figure** (`Dano_session_01_REST_...png`) is from the archived
  first dataset; its rest is far more active than the final clips'. The text now
  says so; a figure from the final dataset would be better (Fig. `fig:profiles`
  shows the final data as numbers, not as a time series).
- **Not compiled**: no LaTeX is installed on this machine. Environments are
  balanced, every `\ref` has its `\label`, but check the page count (IEEE
  conference) after adding the two figures and the new subsections.
- Length: the additions are about 150 lines. If the page limit bites, the ML
  experiments, NN experiments and comparison paragraphs are the first to shorten:
  their numbers are also in the tables.

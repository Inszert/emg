# DATASET_SUMMARY.md -- what is in `datasets/` now

*Generated 2026-10-28 17:19 by `tools/dataset_summary.py` from the recordings' own metadata and the stored evaluation data; re-run it after recording. Nothing here is typed by hand. `EDA.md` describes the dataset archived on 2026-10-15; this is the working set.*

## 1. At a glance

- **211 recordings, 5 people** (Barbora, Betka, Bruno, Dano, Miso), 6 gestures (REST, FIST, OPEN, THUMB_UP, WAVE_LEFT, WAVE_RIGHT), 10.6 minutes of labelled signal in total.
- Device: Myo armband, 8 channels, EMG mode `filtered` (on-device filtered envelope, about 50 Hz).
- **One sitting per person** (Barbora: session_01, Betka: session_01, Bruno: session_01, Dano: session_01, Miso: session_01): the band was not taken off and refitted, so the confound inside a person (CLAUDE.md 8f) is still there; the honest number is the person-held-out one.
- Windows scored (ML, 1.0 s windows at 0.75 overlap, after window-level deduplication): **1787**; the majority class is 18.3 % of them (what a model that always answers the most common gesture would score).
- Windows scored (neural nets, 0.7 s windows at 0.75 overlap, after window-level deduplication): **2873**; the majority class is 18.6 % of them (what a model that always answers the most common gesture would score).
- Folders that exist but hold no recordings (nothing was lost, they were never filled): Michal, Dano/session_02, Dano/session_09, Michal/session_01.
- Archives, never read by discovery (leading underscore): `_archive_pre_r1_20261015` (328 recordings); `_corrected_Dano_session_03_to_Barbora` (56 recordings).

## 2. Recordings per person and gesture

| person | REST | FIST | OPEN | THUMB_UP | WAVE_LEFT | WAVE_RIGHT | total |
|---|---|---|---|---|---|---|---|
| Barbora | 13 | 4 | 6 | 5 | 8 | 7 | 43 |
| Betka | 7 | 6 | 7 | 7 | 7 | 7 | 41 |
| Bruno | 8 | 7 | 7 | 6 | 8 | 7 | 43 |
| Dano | 6 | 7 | 7 | 8 | 8 | 8 | 44 |
| Miso | 10 | 5 | 8 | 6 | 5 | 6 | 40 |
| **all** | 44 | 29 | 35 | 32 | 36 | 35 | 211 |

## 3. Per person

| person | labelled seconds | mean clip (min - max) s | sample rate Hz (min - max) | recorded | free takes (minutes) | cut from takes | recordings with a quality flag | gesture clips with no activity found | quarantined recordings |
|---|---|---|---|---|---|---|---|---|---|
| Barbora | 112.6 | 2.62 (1.08 - 9.48) | 49.8 (49.6 - 50.1) | 2026-10-19 | 6 (5.2) | 43 | 10 | 0 | 0 |
| Betka | 113.0 | 2.76 (1.01 - 3.99) | 49.9 (49.7 - 50.3) | 2026-10-16 | 6 (7.5) | 41 | 20 | 0 | 2 |
| Bruno | 135.1 | 3.14 (2.0 - 4.82) | 49.7 (46.5 - 50.5) | 2026-10-21 | 6 (5.2) | 43 | 9 | 0 | 1 |
| Dano | 154.1 | 3.5 (2.98 - 4.09) | 49.9 (49.7 - 51.2) | 2026-10-15 | 6 (8.0) | 44 | 8 | 2 | 6 |
| Miso | 119.1 | 2.98 (2.04 - 3.02) | 49.9 (48.8 - 50.4) | 2026-10-21 | 6 (4.1) | 40 | 6 | 0 | 3 |

Quality flags are findings, never deletions (safety rule 5): Barbora: noisy_baseline 7, rest_not_quiet 3; Betka: noisy_baseline 20; Bruno: noisy_baseline 9; Dano: noisy_baseline 8; Miso: noisy_baseline 3, rest_not_quiet 3. `noisy_baseline` = resting RMS above 60 (skin contact or mains); `rest_not_quiet` = the clip's rest part is not quiet. "Cut from takes" are clips carved from a longer free take; the takes themselves are never training data.

## 4a. Windows the models are scored on (ML, 1.0 s)

| person | FIST | OPEN | REST | THUMB_UP | WAVE_LEFT | WAVE_RIGHT | total |
|---|---|---|---|---|---|---|---|
| Barbora | 102 | 50 | 61 | 23 | 40 | 18 | 294 |
| Betka | 57 | 61 | 62 | 17 | 60 | 63 | 320 |
| Bruno | 62 | 53 | 66 | 53 | 82 | 63 | 379 |
| Dano | 60 | 87 | 52 | 82 | 92 | 79 | 452 |
| Miso | 44 | 67 | 86 | 48 | 45 | 52 | 342 |

## 4b. Windows the models are scored on (neural nets, 0.7 s)

| person | FIST | OPEN | REST | THUMB_UP | WAVE_LEFT | WAVE_RIGHT | total |
|---|---|---|---|---|---|---|---|
| Barbora | 161 | 85 | 104 | 40 | 64 | 36 | 490 |
| Betka | 90 | 98 | 98 | 34 | 98 | 98 | 516 |
| Bruno | 98 | 86 | 107 | 84 | 129 | 98 | 602 |
| Dano | 98 | 138 | 84 | 130 | 143 | 124 | 717 |
| Miso | 70 | 105 | 140 | 79 | 70 | 84 | 548 |

## 5. Calibration

Two calibrations exist per person and they are not the same thing (CLAUDE.md 8n): the **saved** one is what PREDICT uses (measured live, a maximum-effort squeeze or sweep), the **derived** one is what TRAINING uses (percentiles of the person's own recorded relaxed holds). Per channel, in device counts:

| person | saved: when (how) | saved: mean rest | saved: reference min - max | derived: mean rest | derived: reference min - max | saved / derived reference |
|---|---|---|---|---|---|---|
| Barbora | 2026-10-19 (derived_from_recordings) | 26 | 205 - 929 | 26 | 205 - 929 | 1.0x - 1.0x |
| Betka | 2026-10-16 (derived_from_recordings) | 22 | 280 - 1077 | 22 | 280 - 1077 | 1.0x - 1.0x |
| Bruno | -- | -- | -- | 37 | 512 - 1393 | -- |
| Dano | 2026-10-26 (live_gesture_sweep) | 32 | 321 - 1062 | 21 | 154 - 992 | 0.8x - 2.2x |
| Miso | -- | -- | -- | 37 | 91 - 1054 | -- |

A person with no saved calibration cannot use a model trained on calibrated features until they calibrate (CLAUDE.md 12); training is unaffected because it uses the derived one.

## 6. Reading this honestly

- Five people and one sitting each: every accuracy in the other documents is the mean over five held-out people and carries that spread (the weakest person alone moves a mean by about 10 points).
- Recording counts per gesture are uneven (REST has the most clips for some people); the windows tables above are what the majority-class baseline is computed from.
- Where this data is described elsewhere: `DATASET.md` (on-disk format and schema), `EDA.md` (the archived dataset and why it could not give an honest score), `MODELS_COMPARED.md` and `REPORT.md` (results).

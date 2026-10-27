# Report: models, ablation, defaults and verification

*Generated 2026-10-27 06:28 by `tools/make_report.py` from `results/experiments.json`. Re-run it after any new measurement; nothing below is typed by hand.*

## 1. Goal and context

The project records 8-channel EMG from a Thalmic Myo and trains gesture classifiers that must work on a person they have never seen. This pass added: a **model zoo** (every classifier trained, measured and selectable), **soft voting** of selected models, **augmentation** and **adaptation to the person**, two questions the application asks (calibrate the hand; retrain), a settings area for all of it, an application icon, and the documentation and checks needed before the first test on a new band (`docs/sdk_myo.md`, `docs/install_windows.md`, `docs/install_linux.md`). Every new feature is off, or equal to the old behaviour, unless a measurement below says otherwise (`docs/FEATURES.md` lists how to turn each on and off).

## 2. Data

5 people (Barbora, Betka, Bruno, Dano, Miso), 211 recordings, six gestures (REST, FIST, OPEN, THUMB_UP, WAVE_LEFT, WAVE_RIGHT), per-wearer calibration `affine`, derived from each person's own recordings. After window-level deduplication the NN windows (0.7 s at 0.75 overlap) number 2873. **Five people is small**: a difference of a couple of points is inside the spread between people, and the weakest person (Barbora, a differently seated band) alone moves every average by about 10 points. `docs/EDA.md` and `docs/MODELS_COMPARED.md` describe the data in full.

## 3. Methods

- **Protocol.** Leave-one-person-out: every window is scored by a model that never saw that person. The score is the mean over the five held-out people (each counts once). Per-recording vote and macro-F1 are reported beside it.
- **Seeds.** Ablations use 3 seeds and report mean ± seed spread; the zoo uses 2 seeds for the nets and 3 for the ML models.
- **A switch counts as helping** (`training/ablation_stats.py`) only if its mean gain over the baseline is at least max(1 point, 2 x the seed noise) AND it is better for at least 4 of the 5 people; or the same on the robustness probes without losing more than 1 point of clean accuracy. The seed noise is the LARGER of the baseline's and the setting's own spread over seeds (`ablation_stats.judge`); that choice is conservative and decides one verdict (neural-net adaptation, section 8). Anything else is "no evidence", stays off and remains selectable.
- **Macro-F1** is computed per held-out person over the six fixed classes (a class the person never produced or was never predicted scores 0) and averaged over people; it is not scikit-learn's pooled value.
- **Robustness probes**, applied to the held-out person's raw windows after the fact: multiplicative + additive noise; the band slid one pod either way (a circular channel roll); the whole signal 25 % weaker. Statistics for adaptation are taken before the perturbation (a band that drifts after calibration).
- **DC-only control** (CLAUDE.md 8f/8k): the same model on each window flattened to its 8 channel means -- how much of a score is amplitude rather than gesture shape.
- **Adaptation** standardises each person's input with statistics from a random 10 % of their own unlabelled windows (about 9 s, a calibration sweep). It is *not* zero-shot: it answers "a new person who wears the band for a few seconds first".

## 4. Models and sizes

16 ML classifiers and 9 neural nets are in the zoo, all trained on the whole dataset and scored leave-one-person-out. ★ = starred (top 3 per family by accuracy per window, ties by macro-F1).

**The ML models in the zoo were trained with adaptation to the person** (the measured default, section 6), so their scores are not comparable with the ablation baseline further down (no adaptation) -- compare within a table. The neural nets were trained plain, which is why the GRU's score here equals its ablation baseline.

<!-- TABLE: models_ml -->

|  | model | per window | macro-F1 | per recording | size on disk | ms / live window | trained with |
|---|---|---|---|---|---|---|---|
| ★ | RF-500-leaf3 | 80.8 % | 78.6 % | 86.3 % | 4.8 MB | 26.4 | adaptation |
| ★ | RF-500 | 80.5 % | 78.0 % | 85.5 % | 5.7 MB | 26.5 | adaptation |
| ★ | ET-500-leaf3 | 79.4 % | 77.9 % | 84.7 % | 11.5 MB | 26.9 | adaptation |
|  | ET-500 | 79.4 % | 77.7 % | 85.1 % | 13.0 MB | 26.6 | adaptation |
|  | SVM-rbf-C1 | 77.2 % | 75.4 % | 81.7 % | 218.9 KB | 2.47 | adaptation |
|  | SVM-rbf-C3 | 76.4 % | 74.1 % | 81.6 % | 167.4 KB | 2.27 | adaptation |
|  | MLP-256-128 | 75.7 % | 73.1 % | 80.2 % | 564.3 KB | 3.53 | adaptation |
|  | MLP-128 | 74.9 % | 72.0 % | 79.8 % | 102.0 KB | 2.34 | adaptation |
|  | HGB-lr0.05 | 74.6 % | 70.9 % | 79.5 % | 2.9 MB | 28.6 | adaptation |
|  | LogReg-C0.1 | 74.4 % | 71.4 % | 77.4 % | 10.7 KB | 2.25 | adaptation |
|  | SVM-rbf-C10 | 74.4 % | 71.7 % | 79.3 % | 157.0 KB | 2.33 | adaptation |
|  | SVM-rbf-C30 | 73.8 % | 71.1 % | 78.4 % | 149.9 KB | 4.67 | adaptation |
|  | LDA-shrinkage | 73.4 % | 71.0 % | 77.6 % | 31.4 KB | 2.60 | adaptation |
|  | kNN-31 | 72.9 % | 70.3 % | 78.7 % | 371.2 KB | 5.76 | adaptation |
|  | kNN-15 | 72.8 % | 70.4 % | 79.3 % | 371.2 KB | 3.55 | adaptation |
|  | LogReg-C1 | 69.7 % | 66.2 % | 73.5 % | 10.7 KB | 2.17 | adaptation |

<!-- TABLE: models_nn -->

|  | model | per window | macro-F1 | per recording | parameters | size on disk | ms / live window |
|---|---|---|---|---|---|---|---|
| ★ | NN-gru | 67.7 % | 63.9 % | 72.3 % | 4 455 | 23.5 KB | 0.81 |
| ★ | NN-tsmixer | 66.3 % | 61.9 % | 67.4 % | 9 066 | 45.9 KB | 1.81 |
| ★ | NN-chan_transformer | 66.0 % | 63.4 % | 68.7 % | 20 326 | 92.2 KB | 1.25 |
|  | NN-lstm | 63.9 % | 59.9 % | 67.4 % | 14 022 | 61.3 KB | 0.52 |
|  | NN-circ_gru | 63.4 % | 60.3 % | 67.3 % | 9 927 | 47.1 KB | 1.28 |
|  | NN-tcn | 62.8 % | 58.6 % | 66.2 % | 3 943 | 30.2 KB | 2.08 |
|  | NN-cnn | 62.6 % | 58.6 % | 67.2 % | 18 246 | 82.6 KB | 1.09 |
|  | NN-cnn_gru | 62.3 % | 57.3 % | 66.2 % | 6 503 | 34.9 KB | 1.49 |
|  | NN-inception | 61.1 % | 56.8 % | 65.5 % | 6 854 | 42.3 KB | 2.62 |

<!-- TABLE: model_sizes -->

The whole ML zoo is 40.0 MB; the whole neural zoo 460.0 KB. Latency is one live window on an idle machine (`tools/bench_zoo.py`): for the ML models the whole path (resample, features, probabilities, one thread), for the nets one forward pass of batch 1 on CUDA.

**Best ML soft-voting subset:** ml-rf-500, ml-rf-500-leaf3, ml-et-500, ml-logreg-c0.1 -- 79.3 % on the folds it was searched on, **77.0 % nested** (chosen without the held-out person), against 77.6 % for the best single model measured the same way (seed-0 out-of-fold probabilities, so not the seed-averaged scores in the table above). It does not beat the best single model, so it is not starred.

**Best NN soft-voting subset:** nn-gru, nn-chan-transformer -- 72.6 % on the folds it was searched on, **69.1 % nested** (chosen without the held-out person), against 72.4 % for the best single model measured the same way (seed-0 out-of-fold probabilities, so not the seed-averaged scores in the table above). It does not beat the best single model, so it is not starred.

![The zoo](figures/zoo_overview.png)

## 5. Newer architectures

`tools/new_models_test.py` trained five architectures newer than LSTM/GRU under the same protocol, 150 epochs, one seed, on CUDA:

<!-- TABLE: new_models_test -->

| model | parameters | person held out | time removed (DC-only) | within-person (optimistic) | ms / window |
|---|---|---|---|---|---|
| GRU (current NN) | 4 455 | 72.4 % | 63.3 % | 95.6 % | 0.68 |
| ROCKET-GPU + ridge | 168 928 | 69.7 % | 66.5 % | 97.8 % | 8.95 |
| Channel Transformer | 20 326 | 67.7 % | 66.8 % | 96.5 % | 1.31 |
| TSMixer | 9 066 | 65.2 % | 63.4 % | 97.8 % | 3.21 |
| CircConv-GRU + rotation aug | 9 927 | 62.0 % | 60.7 % | 95.3 % | 1.15 |
| InceptionTime-S | 6 854 | 60.6 % | 62.6 % | 98.2 % | 1.98 |

Result: **none of them beat the GRU** -- GRU 72.4 %, best of the new ones ROCKET-GPU + ridge 69.7 %. Every new model is far better within-person (95 - 98 %) than across people, which is the usual sign that capacity goes into memorising the wearer. With one seed, differences of a few points are within noise (the GRU alone varied by about 10 points between seeds in the ablation below).

![Newer models](figures/novesie_modely_test.png)

## 6. Ablation study, and how the defaults were chosen

One switch at a time against the baseline, then only combinations of what helped (stage B).

### Neural net (GRU, 0.7 s)

<!-- TABLE: ablation_nn -->

| setting | seeds | per window | ± seeds | macro-F1 | change | people better | robustness change | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 67.7 % | 4.7 % | 63.9 % | -- | -- | -- | reference |
| aug:gain | 3 | 65.9 % | 3.6 % | 62.2 % | -1.8 | 2/5 | -1.4 | hurts |
| aug:noise | 3 | 68.2 % | 2.4 % | 63.8 % | +0.4 | 3/5 | +0.8 | no evidence |
| aug:timewarp | 3 | 67.3 % | 1.5 % | 63.9 % | -0.4 | 2/5 | -0.2 | no evidence |
| aug:rotate | 3 | 63.6 % | 2.4 % | 58.5 % | -4.2 | 2/5 | +3.6 | hurts |
| aug:chandrop | 3 | 66.7 % | 1.0 % | 64.5 % | -1.1 | 3/5 | +1.7 | hurts |
| adapt | 3 | 72.3 % | 1.5 % | 67.8 % | +4.5 | 4/5 | +3.5 | no evidence |

### ML (soft vote ET-500 + RF-500-leaf3)

<!-- TABLE: ablation_ml -->

| setting | seeds | per window | ± seeds | macro-F1 | change | people better | robustness change | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 77.0 % | 0.5 % | 74.3 % | -- | -- | -- | reference |
| aug:gain | 3 | 72.4 % | 0.4 % | 70.0 % | -4.7 | 1/5 | -1.1 | hurts |
| aug:noise | 3 | 77.1 % | 0.4 % | 74.3 % | +0.0 | 2/5 | -0.3 | no evidence |
| aug:timewarp | 3 | 76.9 % | 0.3 % | 74.3 % | -0.1 | 3/5 | -0.4 | no evidence |
| aug:rotate | 3 | 70.4 % | 0.8 % | 67.3 % | -6.6 | 0/5 | +3.1 | hurts |
| aug:chandrop | 3 | 72.8 % | 1.6 % | 70.6 % | -4.2 | 1/5 | -1.6 | hurts |
| adapt | 3 | 81.3 % | 2.5 % | 79.1 % | +4.3 | 3/5 | +3.7 | helps robustness |
| calib:off | 3 | 72.7 % | 0.5 % | 67.8 % | -4.3 | 2/5 | +2.2 | hurts |
| calib:gain | 3 | 80.6 % | 0.3 % | 76.8 % | +3.6 | 3/5 | +3.3 | no evidence |
| calib:per_channel | 3 | 68.9 % | 0.3 % | 65.3 % | -8.1 | 1/5 | -3.4 | hurts |

![Ablation effects](figures/ablation_effects.png)

### Voting against its members

<!-- TABLE: ablation_voting -->

| model | per window | ± seeds | macro-F1 |
|---|---|---|---|
| ET-500 | 76.9 % | 0.4 % | 74.2 % |
| RF-500-leaf3 | 76.3 % | 0.3 % | 73.9 % |
| lda | 63.1 % | 0.0 % | 61.0 % |
| svm_rbf | 64.3 % | 0.2 % | 60.5 % |
| vote(ET+RF) | 77.0 % | 0.5 % | 74.3 % |
| vote(all 4) | 69.4 % | 0.1 % | 67.5 % |

The combination test's winner, the ET-500 + RF-500-leaf3 vote, reads 77.0 % against 76.9 % for the best single member (ET-500): a gain of +0.1 points, inside the seed spread -- voting does not demonstrably help here. Voting all four models of `models/model_spec.json` -- which is what the deployed `gesture_model.joblib` currently is -- falls to 69.4 %: an unweighted vote gives the 13-point weaker SVM and LDA an equal say. Whether to deploy the pair is the operator's call (CLAUDE.md 8n); the zoo lets you select it without touching the deployed model.

![Voting](figures/voting_vs_members.png)

### Defaults

<!-- TABLE: defaults -->

| family | setting | shipped value | why |
|---|---|---|---|
| NN | augmentation | off | no augmentation passed the rule (>= 1 point and 2x the seed noise, better for >= 4 of 5 people, or the same on robustness) -- stays off, selectable in the settings |
| NN | adaptation | off | did not pass the rule (+4.5 points, better for 4/5 people, no evidence) -- stays off, selectable in the settings |
| ML | augmentation | off | no augmentation passed the rule (>= 1 point and 2x the seed noise, better for >= 4 of 5 people, or the same on robustness) -- stays off, selectable in the settings |
| ML | adaptation | on | helps robustness: +4.3 points per window, better for 3/5 people |

The calibration mode was also measured for the ML pair: the shipped `affine` is kept because switching calibration off or to per-channel scaling measurably hurts. A single global gain was **not** judged better (it helps some people a great deal, notably the weakest, and hurts others), which is a lead to follow with more people, not a default.

## 7. Robustness

<!-- TABLE: robustness -->

| family | setting | noise | band shifted one pod | 25 % weaker |
|---|---|---|---|---|
| NN | baseline | 66.9 % | 45.6 % | 67.7 % |
| NN | aug:gain | 65.3 % | 45.3 % | 65.4 % |
| NN | aug:noise | 67.9 % | 46.4 % | 68.3 % |
| NN | aug:timewarp | 66.6 % | 45.9 % | 67.0 % |
| NN | aug:rotate | 64.0 % | 62.2 % | 64.7 % |
| NN | aug:chandrop | 66.6 % | 49.9 % | 68.9 % |
| NN | adapt | 71.0 % | 48.8 % | 71.0 % |
| ML | baseline | 72.6 % | 43.8 % | 72.7 % |
| ML | aug:gain | 70.7 % | 41.4 % | 73.7 % |
| ML | aug:noise | 72.7 % | 42.7 % | 72.8 % |
| ML | aug:timewarp | 72.2 % | 43.1 % | 72.7 % |
| ML | aug:rotate | 68.0 % | 61.1 % | 69.2 % |
| ML | aug:chandrop | 69.0 % | 42.3 % | 73.1 % |
| ML | adapt | 78.4 % | 44.4 % | 77.5 % |
| ML | calib:off | 71.3 % | 49.8 % | 74.7 % |
| ML | calib:gain | 81.3 % | 44.9 % | 72.8 % |
| ML | calib:per_channel | 70.0 % | 39.6 % | 69.2 % |

![Robustness](figures/ablation_robustness.png)

- **neural net:** a band slid one pod costs 22 points (67.7 % -> 45.6 %); training with `rotate` augmentation brings the shifted score to 62.2 % but costs 4.2 points on clean data, so it does not pass the rule for a default. It is the right switch if the band is likely to be refitted between calibration and use.
- **ML pair:** a band slid one pod costs 33 points (77.0 % -> 43.8 %); training with `rotate` augmentation brings the shifted score to 61.1 % but costs 6.6 points on clean data, so it does not pass the rule for a default. It is the right switch if the band is likely to be refitted between calibration and use.

A shifted band is the probe every model fails and the open problem the weakest person's score reflects; only `rotate` addresses it.

## 8. Adaptation to the person

<!-- TABLE: adaptation -->

| family | per window | ± seeds | macro-F1 | change | people better | verdict |
|---|---|---|---|---|---|---|
| NN | 72.3 % | 1.5 % | 67.8 % | +4.5 | 4/5 | no evidence |
| ML | 81.3 % | 2.5 % | 79.1 % | +4.3 | 3/5 | helps robustness |

**What if the person's statistics are missing?** An ML model trained with adaptation, scored without them (the stored global statistics, which is what `ADAPT_LIVE` off or a missing calibration signal does), reads 78.2 % against 78.5 % with them and 77.6 % for a plain model (seed 0): the fallback costs nothing measurable, so an adapted model is safe to deploy before a calibration signal exists.

**How firm are the two verdicts?** Neural net: the gain is +4.5 points for 4/5 people, but the baseline's seed spread is 4.7 % (the setting's own is 1.5 %), so 2 x the larger spread is 9.4 %: the verdict is *no evidence*. Judged against the setting's own spread alone it would pass, so this one is a borderline call decided by the baseline's instability, not a clear negative. ML: adaptation passes on robustness for the ET+RF vote by a small margin and on accuracy for only 3 of 5 people; a single ET-500 does not pass on its own. The ML default is therefore the weakest of the shipped defaults.

**Caveat.** The gain is not uniform: it comes mostly from the weakest and a couple of other people and is within the seed noise for some, because the statistics depend on which windows were sampled. Five people cannot settle it; the experiment to repeat is the same table with more wearers. Where there is data from one person only, adaptation cannot be tested at all.

## 9. Real-time and latency

<!-- TABLE: latency -->

| family | model | ms per live window | device |
|---|---|---|---|
| ML | RF-500-leaf3 | 26.4 | CPU, 1 thread |
| ML | RF-500 | 26.5 | CPU, 1 thread |
| ML | ET-500-leaf3 | 26.9 | CPU, 1 thread |
| ML | ET-500 | 26.6 | CPU, 1 thread |
| ML | SVM-rbf-C1 | 2.47 | CPU, 1 thread |
| ML | SVM-rbf-C3 | 2.27 | CPU, 1 thread |
| ML | MLP-256-128 | 3.53 | CPU, 1 thread |
| ML | MLP-128 | 2.34 | CPU, 1 thread |
| ML | HGB-lr0.05 | 28.6 | CPU, 1 thread |
| ML | LogReg-C0.1 | 2.25 | CPU, 1 thread |
| ML | SVM-rbf-C10 | 2.33 | CPU, 1 thread |
| ML | SVM-rbf-C30 | 4.67 | CPU, 1 thread |
| ML | LDA-shrinkage | 2.60 | CPU, 1 thread |
| ML | kNN-31 | 5.76 | CPU, 1 thread |
| ML | kNN-15 | 3.55 | CPU, 1 thread |
| ML | LogReg-C1 | 2.17 | CPU, 1 thread |
| NN | NN-gru | 0.81 | CUDA (GTX 1650) |
| NN | NN-tsmixer | 1.81 | CUDA (GTX 1650) |
| NN | NN-chan_transformer | 1.25 | CUDA (GTX 1650) |
| NN | NN-lstm | 0.52 | CUDA (GTX 1650) |
| NN | NN-circ_gru | 1.28 | CUDA (GTX 1650) |
| NN | NN-tcn | 2.08 | CUDA (GTX 1650) |
| NN | NN-cnn | 1.09 | CUDA (GTX 1650) |
| NN | NN-cnn_gru | 1.49 | CUDA (GTX 1650) |
| NN | NN-inception | 2.62 | CUDA (GTX 1650) |

The live window interval is 150 ms. The ML forests cost most (a vote of several forests is the sum of its members); the nets are a few milliseconds. Voting adds the members' costs, and the NN worker runs all its nets in one process so a vote of nets costs the sum of their forward passes, not a second process.

## 10. GUI features

Settings (gear icon, Ctrl+,): Gestures, Models, Training, General. Details, how to turn each feature on and off, and the screenshots: [FEATURES.md](FEATURES.md) and [UI_GUIDE.md](UI_GUIDE.md).

## 11. Testing and verification

<!-- TABLE: tests -->

| suite | checks passed | seconds |
|---|---|---|
| datalayer | 150/150 | 6.1 |
| gui | 109/109 | 46.2 |
| review | 50/50 | 18.3 |
| label_on_graph | 48/48 | 19.7 |
| markers | 88/88 | 2.6 |
| free_take | 268/268 | 11.9 |
| model_spec | 346/346 | 10.5 |
| predict | 109/109 | 54.2 |
| trail | 103/103 | 7.1 |
| calibration | 161/161 | 5.7 |
| jobs | 48/48 | 43.2 |
| class_mask | 121/121 | 5.8 |
| device | 104/104 | 1.2 |
| sweep_cost | 48/48 | 4.4 |
| filters | 52/52 | 4.4 |
| no_torch | 11/11 | 3.7 |
| icon | 17/17 | 1.1 |
| zoo | 109/109 | 29.9 |
| app_settings | 67/67 | 5.3 |
| defaults | 18/18 | 1.1 |
| nn | 162/162 | 9.1 |
| custom_gesture | 48/48 | 180.5 |

2237/2237 checks in 22 suites (`python tools/run_tests.py`; `--fast` skips the slow `nn` and `custom_gesture` suites, which are part of this count).

![Tests](figures/tests_overview.png)

An independent check of the finished application is recorded in [VERIFICATION.md](VERIFICATION.md).

## 12. Limitations

- Five people; one band position each. Every number carries that.
- Seed spread is large for the nets (the GRU baseline moved by about 10 points between seeds), so one-seed rankings (the newer-architectures test, the zoo's nets) are indicative, not final.
- Adaptation is not zero-shot, and its gain is not uniform across people.
- Latencies are from one idle machine (i5-9300H, GTX 1650); a robot's SBC will differ. The nets were timed without the pipe to the worker process (about 1 ms more in the app).
- The adaptation default for ML models applies to the zoo trainer only; the classic RETRAIN sweep (`train_model.py`), which produces the deployed `gesture_model.joblib`, trains plain models.
- Acquisition is Myo-only; nothing here was run against a real second band.
- The nested subset estimate is honest about subset choice, not about the 5-person sample.

## 13. When the new band arrives

Follow the hardware checklist in [sdk_myo.md](sdk_myo.md) (what a new backend must provide, what to run in what order, what numbers to expect) and the install guides ([Windows](install_windows.md), [Linux](install_linux.md)). A model trained on 8 channels refuses a different channel count rather than guessing; record new data with the new band, calibrate, and retrain (the retrain question appears at startup).

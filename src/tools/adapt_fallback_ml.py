"""
How much does an ML model trained WITH adaptation lose if the person's
statistics are missing and the stored global ones are used instead?

    python tools/adapt_fallback_ml.py

Leave-one-person-out, ET-500 + RF-500-leaf3 soft vote, seed 0. Writes
results/ablation/ml/fallback.json (read by tools/collect_results.py).

Three numbers per person:
  baseline                      a model trained and tested on globally standardised input
  adapted                       trained and tested on per-person standardised input
  adapted model, global stats   trained on per-person input, tested WITHOUT the
                                person's statistics (what ADAPT_LIVE off, or a
                                missing calibration signal, does at run time)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import emg_dataset_tool as T                                      # noqa: E402
import ml_cv                                                      # noqa: E402
import train_nn as NN                                             # noqa: E402
import zoo_roster as ZR                                           # noqa: E402
from emg_app import adaptation as AD                              # noqa: E402

D = ml_cv.build_windows(NN.DATASET_ROOT, T.CALIBRATION_MODE)
D["F"] = ml_cv.featurize(D["W"], D["cals"])
y, persons, F = D["y"], D["persons"], D["F"]
specs = {s[0]: s for s in ZR.ML_ROSTER}
members = [specs["ET-500"], specs["RF-500-leaf3"]]
table = AD.person_feature_table(F, persons, seed=0)
Fa = AD.apply_person_features(F, persons, table)
gm, gs = AD.feature_stats(F)
Fg = ((F - gm) / gs).astype(np.float32)
NAMES = ["baseline", "adapted", "adapted model, global stats at test"]
people = sorted(set(persons.tolist()))
res = {name: [] for name in NAMES}
for held in people:
    tr = np.flatnonzero(persons != held)
    te = np.flatnonzero(persons == held)
    for name, Ftr, Fte in (("baseline", F, F), ("adapted", Fa, Fa),
                           (NAMES[2], Fa, Fg)):
        P = 0
        for spec in members:
            pipe = ml_cv.make_pipeline(spec, 0)
            pipe.fit(Ftr[tr], y[tr])
            P = P + pipe.predict_proba(Fte[te])
        res[name].append(float((P.argmax(1) == y[te]).mean()))
    print(held, {k: round(v[-1] * 100, 1) for k, v in res.items()}, flush=True)
mean = {k: float(np.mean(v)) for k, v in res.items()}
print({k: round(v * 100, 1) for k, v in mean.items()})
out = os.path.join(ROOT, "results", "ablation", "ml")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "fallback.json"), "w", encoding="utf-8") as fh:
    json.dump({"protocol": "leave-one-person-out, vote(ET-500 + RF-500-leaf3), "
                           "seed 0",
               "per_person": {k: dict(zip(people, v)) for k, v in res.items()},
               "mean": mean}, fh, indent=1)

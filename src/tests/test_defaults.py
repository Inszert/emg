"""
The shipped defaults are the measured ones.

`results/experiments.json` (built by tools/collect_results.py from the raw
ablation runs) carries, per family and switch, the value the measurements
justify and why. This test checks that `emg_app/config.py` ships exactly that,
that every switch that is ON by default passed the rule in
`training/ablation_stats.py`, and that the decision procedure itself behaves
(a switch that does not pass stays off; ties and missing measurements never
switch anything on).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import emg_dataset_tool as T                                         # noqa: E402
import ablation_stats as ST                                          # noqa: E402
import collect_results as CR                                         # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    shown = "" if detail is None or (isinstance(detail, str) and not detail) \
        else f"  [{detail}]"
    print(f"{'PASS' if condition else 'FAIL'}  {name}{shown}")


print("\n=== the decision procedure ===")


def fold_runs(deltas, base=0.70, seeds=3, robust=None):
    """Synthetic runs: every person scores base + their delta, every seed."""
    people = ["a", "b", "c", "d", "e"]
    runs = []
    for s in range(seeds):
        folds = []
        for p, d in zip(people, deltas):
            f = {"person": p, "acc": base + d + 0.001 * s, "f1": base + d}
            if robust is not None:
                f["robust"] = {k: base + d + robust for k in
                               ("noise", "shift", "gain")}
            folds.append(f)
        runs.append({"folds": folds, "seed": s})
    return runs


base = fold_runs([0, 0, 0, 0, 0], robust=0.0)
check("a clear gain for every person passes",
      ST.judge(base, fold_runs([.03] * 5, robust=0.0))["helps_acc"])
check("a gain of 0.5 points does not (below the 1-point floor)",
      not ST.judge(base, fold_runs([.005] * 5, robust=0.0))["default_worthy"])
check("a big gain for only 3 of 5 people does not pass on accuracy",
      not ST.judge(base, fold_runs([.10, .10, .10, -.02, -.02],
                                   robust=0.0))["helps_acc"])
check("...but a gain for 4 of 5 does",
      ST.judge(base, fold_runs([.05, .05, .05, .05, -.05],
                               robust=0.0))["helps_acc"])
check("a loss is reported as hurts",
      ST.judge(base, fold_runs([-.04] * 5, robust=0.0))["verdict"] == "hurts")
r = ST.judge(base, fold_runs([0.0] * 5, robust=0.05))
check("equal accuracy but better robustness passes on robustness",
      r["helps_robust"] and r["default_worthy"] and not r["helps_acc"], r)
r = ST.judge(base, fold_runs([-.03] * 5, robust=0.10))
check("...unless clean accuracy dropped by more than a point",
      not r["helps_robust"], r)
noisy_base = fold_runs([0] * 5, robust=0.0)
for i, run in enumerate(noisy_base):                  # seeds that disagree a lot
    for f in run["folds"]:
        f["acc"] += (-0.05, 0.0, 0.05)[i]
check("a gain smaller than twice the seed spread is not believed",
      not ST.judge(noisy_base, fold_runs([.04] * 5, robust=0.0))["helps_acc"])

rows = [{"family": "nn", "config": "baseline", "setting": "baseline",
         "default_worthy": False, "acc_mean": .7, "robust_noise": .6,
         "delta_acc": None, "verdict": "reference", "people_better": None,
         "n_people": None},
        {"family": "nn", "config": "aug:gain", "setting": "augmentation",
         "default_worthy": False, "acc_mean": .7, "robust_noise": .6,
         "delta_acc": 0.0, "verdict": "no evidence", "people_better": 2,
         "n_people": 5}]
d = CR.decide(rows, "nn")
check("nothing passes -> augmentation off, adaptation off, both explained",
      [x["value"] for x in d] == [[], False] and all(x["reason"] for x in d), d)
rows[1].update(default_worthy=True, verdict="helps accuracy", delta_acc=.03,
               people_better=5)
d = CR.decide(rows, "nn")
check("a switch that passed is switched on, with its numbers in the reason",
      d[0]["value"] == ["gain"] and "+3.0" in d[0]["reason"], d[0])

print("\n=== the shipped configuration ===")
path = os.path.join(ROOT, "results", "experiments.json")
if not os.path.exists(path):
    check("results/experiments.json exists (python tools/collect_results.py)",
          False)
else:
    doc = json.load(open(path, encoding="utf-8"))
    names = {("nn", "augmentation"): "TRAIN_AUGMENT_NN",
             ("nn", "adaptation"): "TRAIN_ADAPT_NN",
             ("ml", "augmentation"): "TRAIN_AUGMENT_ML",
             ("ml", "adaptation"): "TRAIN_ADAPT_ML"}
    for dec in doc["defaults"]:
        cfg_name = names[(dec["family"], dec["setting"])]
        shipped = getattr(T, cfg_name)
        want = dec["value"]
        check(f"{cfg_name} ships the measured value ({want})",
              (list(shipped) == list(want)) if isinstance(want, list)
              else (bool(shipped) == bool(want)), f"config {shipped!r}")
        if want not in ([], False, None):
            row = [r for r in doc["ablation"] if r["family"] == dec["family"]
                   and r["default_worthy"]]
            check(f"...and something passed the rule to justify turning "
                  f"{cfg_name} on", bool(row))
    check("every default has a written reason",
          all(d["reason"] for d in doc["defaults"]))
    check("the ML ablation measured three seeds or more",
          doc["meta"]["seeds"]["ml"] >= 3, doc["meta"]["seeds"])
    check("the NN ablation measured three seeds or more",
          doc["meta"]["seeds"]["nn"] >= 3, doc["meta"]["seeds"])

print()
passed = sum(OK)
print(f"{'=' * 16}  {passed}/{len(OK)} checks passed  {'=' * 16}")
sys.exit(0 if all(OK) else 1)

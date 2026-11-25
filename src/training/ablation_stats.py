"""
How the ablation study decides that something helped. numpy only.

THE RULE (one place, so every table and every default uses the same one):

A setting is compared with its baseline on the SAME folds -- every person held
out in turn, the same seeds. For each held-out person the score is the mean over
seeds; the paired differences (setting minus baseline) across the people are
what is judged.

  helps (accuracy)    mean difference >= max(1.0 point, 2 x the seed-to-seed
                      standard deviation of the mean score)  AND  better for at
                      least 4 of the 5 people
  helps (robustness)  the same test on the mean of the robustness probes
                      (noise, shift, gain), AND the clean score has not dropped
                      by more than 1 point
  default-worthy      helps (accuracy) or helps (robustness)

Anything else -- including a difference inside the noise, or a gain that comes
from one person only -- is reported as "no evidence" and the setting stays off.
With five people this is deliberately conservative: it would rather leave a
real small gain off than switch a lucky one on.
"""
from __future__ import annotations

import numpy as np

MIN_GAIN = 0.01
MIN_PEOPLE_BETTER = 4
MAX_CLEAN_DROP = 0.01


def per_person(runs, key="acc", robust=None):
    """`{person: mean over seeds}` of a fold metric (or a robustness probe)."""
    table = {}
    for run in runs:
        for fold in run["folds"]:
            if robust is None:
                v = fold[key]
            else:
                v = float(np.mean([fold["robust"][k] for k in robust]))
            table.setdefault(fold["person"], []).append(v)
    return {p: float(np.mean(v)) for p, v in table.items()}


def seed_std(runs, robust=None):
    """Standard deviation over seeds of the mean-over-people score."""
    if len(runs) < 2:
        return 0.0
    scores = []
    for run in runs:
        if robust is None:
            scores.append(np.mean([f["acc"] for f in run["folds"]]))
        else:
            scores.append(np.mean([np.mean([f["robust"][k] for k in robust])
                                   for f in run["folds"]]))
    return float(np.std(scores, ddof=1))


def paired(base_runs, cand_runs, robust=None):
    b = per_person(base_runs, robust=robust)
    c = per_person(cand_runs, robust=robust)
    people = sorted(set(b) & set(c))
    d = np.asarray([c[p] - b[p] for p in people])
    return people, d


def judge(base_runs, cand_runs, robust_kinds=("noise", "shift", "gain")):
    people, d = paired(base_runs, cand_runs)
    _, dr = paired(base_runs, cand_runs, robust=robust_kinds) \
        if base_runs and "robust" in base_runs[0]["folds"][0] \
        and cand_runs and "robust" in cand_runs[0]["folds"][0] \
        else (people, np.zeros(0))
    noise = max(seed_std(base_runs), seed_std(cand_runs))
    out = {"n_people": len(people),
           "delta_acc": float(d.mean()) if d.size else 0.0,
           "people_better": int((d > 0).sum()),
           "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
           "seed_noise": noise}
    need = max(MIN_GAIN, 2.0 * noise)
    out["helps_acc"] = bool(d.size and d.mean() >= need
                            and (d > 0).sum() >= MIN_PEOPLE_BETTER)
    if dr.size:
        out["delta_robust"] = float(dr.mean())
        out["people_better_robust"] = int((dr > 0).sum())
        need_r = max(MIN_GAIN, 2.0 * max(seed_std(base_runs, robust_kinds),
                                         seed_std(cand_runs, robust_kinds)))
        out["helps_robust"] = bool(
            dr.mean() >= need_r and (dr > 0).sum() >= MIN_PEOPLE_BETTER
            and out["delta_acc"] >= -MAX_CLEAN_DROP)
    else:
        out["delta_robust"] = None
        out["helps_robust"] = False
    out["default_worthy"] = bool(out["helps_acc"] or out["helps_robust"])
    if out["helps_acc"]:
        out["verdict"] = "helps accuracy"
    elif out["helps_robust"]:
        out["verdict"] = "helps robustness"
    elif d.size and d.mean() <= -MIN_GAIN:
        out["verdict"] = "hurts"
    else:
        out["verdict"] = "no evidence"
    return out


def summarise(runs):
    """Mean +- std over seeds of the headline numbers, for the tables."""
    accs = [np.mean([f["acc"] for f in r["folds"]]) for r in runs]
    f1s = [np.mean([f["f1"] for f in r["folds"]]) for r in runs]
    out = {"seeds": len(runs), "acc_mean": float(np.mean(accs)),
           "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
           "f1_mean": float(np.mean(f1s)),
           "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0}
    if runs and "robust" in runs[0]["folds"][0]:
        for k in runs[0]["folds"][0]["robust"]:
            out["robust_" + k] = float(np.mean(
                [np.mean([f["robust"][k] for f in r["folds"]]) for r in runs]))
    for key in ("n_parameters", "size_bytes", "latency_ms"):
        v = [r[key] for r in runs if key in r]
        if v:
            out[key] = float(np.mean(v))
    return out

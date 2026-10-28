"""
Collect every measurement into ONE file, results/experiments.json, which the
documentation tables, the GUI stars and the figures read. Nothing in it is typed
by hand: it is built from the raw run files.

    python tools/collect_results.py            # write results/experiments.json
    python tools/collect_results.py --print    # also print the defaults decision

Sources
-------
    models/zoo/*/meta.json                    the models, their scores and sizes
    models/zoo/ensembles.json                 the best soft-voting subsets
    results/ablation/nn/runs.jsonl            augmentation / adaptation, NN
    results/ablation/ml/runs.jsonl            augmentation / adaptation / calibration /
                                              voting, ML
    models/reports/new_models_test_run1/      the newer-architectures experiment
    tests/last_run.json                       the automatic tests

Shape (the documentation skeleton's tables map onto it):

    meta        what was measured, on what, when
    models      one row per zoo model  (+ star, rank)
    ensembles   the measured soft-voting subsets, and whether they are starred
    ablation    one row per (family, config): mean +- std over seeds, deltas against
                the baseline, the verdict of training/ablation_stats.judge
    voting      ML: every member alone against the votes
    new_models  the newer-architectures experiment (3 protocols per model)
    defaults    the decision: per setting, the value that is shipped and WHY
    tests       the automatic test suites and their counts
"""
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablation_stats as ST                                      # noqa: E402
from emg_app import zoo                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "experiments.json")


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def group(runs, kind=None):
    g = {}
    for r in runs:
        if kind is None or r["kind"] == kind:
            g.setdefault(r["config"], []).append(r)
    return g


# ---- models and stars ------------------------------------------------------------


def models_block():
    entries = zoo.list_entries()
    ensembles = zoo.read_ensembles()
    stars = zoo.compute_stars(entries, ensembles)
    rows = []
    for fam in ("ml", "nn"):
        #  Models trained WITHOUT somebody are scored over fewer people: they
        #  are not ranked with the ordinary ones (see variants_block).
        ranked = zoo.rank([e for e in entries
                           if e.family == fam and not e.excluded])
        for place, e in enumerate(ranked, 1):
            sc = e.scores
            rows.append({
                "id": e.id, "family": fam, "name": e.name, "kind": e.kind,
                "rank": place if e.window_acc is not None else None,
                "params": e.params, "size_bytes": e.size_bytes,
                "window_acc": e.window_acc,
                "window_acc_std": sc.get("window_acc_std"),
                "macro_f1": e.macro_f1, "vote_acc": sc.get("vote_acc"),
                "latency_ms": e.meta.get("latency_ms"),
                "dc_only_acc": sc.get("dc_only_acc"),
                "per_person": sc.get("per_person"),
                "seeds": sc.get("seeds"), "protocol": sc.get("protocol"),
                "starred": e.id in stars["models"],
                "star_reason": stars["models"].get(e.id),
                "trained_with": e.trained_with,
                "description": e.meta.get("description")})
    combos = []
    for fam, c in ensembles.items():
        combos.append(dict(c, family=fam, starred=fam in stars["ensembles"],
                           star_reason=stars["ensembles"].get(fam)))
    return rows, combos


def variants_block():
    """
    The models trained without somebody, each against the ordinary model it is a
    variant of ON THE SAME PEOPLE (never its headline score: that one is over
    fewer, easier people). The blue stars are decided in `zoo.variant_stars`.
    """
    entries = zoo.list_entries()
    blue = zoo.variant_stars(entries)
    by_id = {e.id: e for e in entries}
    rows = []
    for e in entries:
        if not e.excluded:
            continue
        sc = e.scores
        mine = sc.get("per_person") or {}
        base = sc.get("base_per_person") or {}
        base_entry = by_id.get(e.meta.get("variant_of"))
        rows.append({
            "id": e.id, "variant_of": e.meta.get("variant_of"),
            "family": e.family, "name": e.name, "excluded": e.excluded,
            "per_person": mine, "base_per_person": base,
            "delta": {p: mine[p] - base[p] for p in mine if p in base},
            "mean_delta": zoo.variant_gain(e),
            "excluded_held_out": sc.get("excluded_held_out"),
            "window_acc": e.window_acc,
            "base_window_acc": None if base_entry is None
            else base_entry.window_acc,
            "size_bytes": e.size_bytes, "seeds": sc.get("seeds"),
            "blue_star": e.id in blue, "blue_star_reason": blue.get(e.id),
            "base_starred": bool(base_entry) and e.meta.get("variant_of") in
            zoo.compute_stars(entries, {})["models"]})
    return rows


def without_blocks():
    """The training-switch ablation re-run WITHOUT somebody, one block per
    results/ablation_without_<name>/ folder, judged by the same rule."""
    out = []
    root = os.path.join(ROOT, "results")
    for name in sorted(os.listdir(root)):
        if not name.startswith("ablation_without_"):
            continue
        folder = os.path.join(root, name)
        nn_runs = read_jsonl(os.path.join(folder, "nn", "runs.jsonl"))
        ml_runs = read_jsonl(os.path.join(folder, "ml", "runs.jsonl"))
        abl = []
        for kind in sorted({r["kind"] for r in nn_runs}):
            abl += ablation_rows("nn", nn_runs, kind)
        if ml_runs:
            abl += ablation_rows("ml", ml_runs, "vote(ET+RF)")
        if not abl:
            continue
        defaults = []
        for fam in ("nn", "ml"):
            rows = [r for r in abl if r["family"] == fam]
            if any(r["config"] == "baseline" for r in rows):
                defaults += decide(rows, fam)
        out.append({
            "excluded": name[len("ablation_without_"):],
            "people": sorted({f["person"] for r in nn_runs + ml_runs
                              for f in r["folds"]}),
            "seeds": {"nn": max([r["seed"] for r in nn_runs], default=-1) + 1,
                      "ml": max([r["seed"] for r in ml_runs], default=-1) + 1},
            "ablation": abl, "defaults": defaults})
    return out


# ---- ablation ------------------------------------------------------------------------


def ablation_rows(family, runs, kind):
    g = group(runs, kind)
    if "baseline" not in g:
        return []
    base = g["baseline"]
    rows = []
    for name, rr in g.items():
        e = ST.summarise(rr)
        j = None if name == "baseline" else ST.judge(base, rr)
        setting = ("baseline" if name == "baseline" else
                   "adaptation" if name == "adapt" else
                   "calibration" if name.startswith("calib:") else
                   "augmentation" if name.startswith("aug:") and "|" not in name
                   and "+" not in name else "combination")
        rows.append({
            "family": family, "model": kind, "setting": setting, "config": name,
            "value": name, "seeds": e["seeds"],
            "acc_mean": e["acc_mean"], "acc_std": e["acc_std"],
            "f1_mean": e["f1_mean"], "f1_std": e["f1_std"],
            "latency_ms": e.get("latency_ms"), "size_bytes": e.get("size_bytes"),
            "robust_noise": e.get("robust_noise"),
            "robust_shift": e.get("robust_shift"),
            "robust_gain": e.get("robust_gain"),
            "delta_acc": None if j is None else j["delta_acc"],
            "delta_robust": None if j is None else j["delta_robust"],
            "people_better": None if j is None else j["people_better"],
            "n_people": None if j is None else j["n_people"],
            "seed_noise": None if j is None else j["seed_noise"],
            "verdict": "reference" if j is None else j["verdict"],
            "default_worthy": False if j is None else j["default_worthy"]})
    return rows


def decide(rows, family):
    """
    The shipped value of each switch for one family, and why. A switch is on
    only if its row is `default_worthy`; among several, the best accuracy
    (then robustness) wins. Pure function of the measurements.
    """
    base = next((r for r in rows if r["config"] == "baseline"), None)
    out = []
    if base is None:
        return out
    singles = [r for r in rows if r["setting"] == "augmentation"]
    combos = [r for r in rows if r["setting"] == "combination"]
    adapt = next((r for r in rows if r["setting"] == "adaptation"), None)
    candidates = [r for r in singles + combos + ([adapt] if adapt else [])
                  if r["default_worthy"]]
    aug_pick = [r for r in singles + combos if r["default_worthy"]
                and "adapt" not in r["config"]]
    best_aug = max(aug_pick, key=lambda r: (r["acc_mean"],
                                            r["robust_noise"] or 0),
                   default=None)
    if best_aug is None:
        out.append({"family": family, "setting": "augmentation", "value": [],
                    "reason": "no augmentation passed the rule (>= 1 point and "
                              "2x the seed noise, better for >= 4 of 5 people, "
                              "or the same on robustness) -- stays off, "
                              "selectable in the settings",
                    "delta_acc": 0.0})
    else:
        names = best_aug["config"].replace("aug:", "").split("+")
        out.append({"family": family, "setting": "augmentation",
                    "value": [n.split("|")[0] for n in names],
                    "reason": f"{best_aug['verdict']}: "
                              f"{best_aug['delta_acc'] * 100:+.1f} points per "
                              f"window, better for {best_aug['people_better']}/"
                              f"{best_aug['n_people']} people",
                    "delta_acc": best_aug["delta_acc"]})
    if adapt is not None and adapt["default_worthy"]:
        out.append({"family": family, "setting": "adaptation", "value": True,
                    "reason": f"{adapt['verdict']}: {adapt['delta_acc'] * 100:+.1f}"
                              f" points per window, better for "
                              f"{adapt['people_better']}/{adapt['n_people']} "
                              "people",
                    "delta_acc": adapt["delta_acc"]})
    elif adapt is None:
        out.append({"family": family, "setting": "adaptation", "value": False,
                    "reason": "not measured yet -- stays off", "delta_acc": 0.0})
    else:
        out.append({"family": family, "setting": "adaptation", "value": False,
                    "reason": "did not pass the rule"
                              + ("" if adapt is None else
                                 f" ({adapt['delta_acc'] * 100:+.1f} points, "
                                 f"better for {adapt['people_better']}/"
                                 f"{adapt['n_people']} people, "
                                 f"{adapt['verdict']})")
                              + " -- stays off, selectable in the settings",
                    "delta_acc": 0.0 if adapt is None else adapt["delta_acc"]})
    return out


# ---- newer architectures ---------------------------------------------------------------


def new_models_block():
    run = os.path.join(ROOT, "models", "reports", "new_models_test_run1")
    results = read_jsonl(os.path.join(run, "results.jsonl"))
    if not results:
        return []
    meta = json.load(open(os.path.join(run, "run.json"), encoding="utf-8"))
    out = {}
    for r in results:
        accs = [f["acc"] for f in r["folds"]]
        entry = out.setdefault(r["model"], {
            "model": r["model"], "name": meta["titles"].get(r["model"]),
            "params": r["n_parameters"]})
        entry[r["protocol"] + "_acc"] = float(np.mean(accs))
        entry[r["protocol"] + "_std"] = float(np.std(accs))
        if r["protocol"] == "loso":
            entry["per_person"] = {f.get("person") or f.get("held_out"): f["acc"]
                                   for f in r["folds"]}
            entry["latency_ms"] = r.get("latency_ms")
            entry["seconds_per_fold"] = float(np.mean(
                [f["seconds"] for f in r["folds"]]))
    return list(out.values())


# ---- tests ----------------------------------------------------------------------------------


def software_block():
    """What the numbers were produced with (read from package metadata, so
    nothing heavy -- torch included -- is imported)."""
    import platform
    import subprocess
    from importlib import metadata
    out = {"python": platform.python_version(), "os": platform.platform()}
    for name in ("numpy", "scikit-learn", "torch", "PySide6", "pyqtgraph",
                 "joblib", "pandas"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["commit"] = None
    return out


def dataset_block():
    """The working dataset, from tools/dataset_summary.py (run it first)."""
    path = os.path.join(ROOT, "results", "dataset_summary.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def starred_votes_block():
    """
    What the 'yellow star' selection scores as a soft vote, per family.

    The three starred models voted together, scored from their stored
    out-of-fold probabilities exactly as the vote is scored in the application
    (zoo.vote_stats). The combination is fixed by the star rule, not searched
    for, so there is no selection bias; the best single model is given beside
    it, measured the same way.
    """
    entries = zoo.list_entries()
    stars = zoo.compute_stars(entries, zoo.read_ensembles())
    out = []
    for fam in ("ml", "nn"):
        members = [e for e in zoo.rank([e for e in entries if e.family == fam
                                        and not e.excluded])
                   if e.id in stars["models"]]
        vs = zoo.vote_stats(members) if len(members) >= 2 else None
        if not vs:
            continue
        best = max(members, key=lambda e: e.window_acc or 0)
        out.append({"family": fam, "members": [e.name for e in members],
                    "window_acc": vs["window_accuracy"],
                    "macro_f1": vs["window_macro_f1"],
                    "vote_acc": vs["recording_vote_accuracy"],
                    "per_person": vs["subject_independent"]["per_subject"],
                    "seeds": vs["seeds"],
                    "best_single": best.name, "best_single_acc": best.window_acc,
                    "member_acc": {e.name: e.window_acc for e in members}})
    return out


def tests_block():
    path = os.path.join(ROOT, "tests", "last_run.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def main(argv):
    nn_runs = read_jsonl(os.path.join(ROOT, "results", "ablation", "nn",
                                      "runs.jsonl"))
    ml_runs = read_jsonl(os.path.join(ROOT, "results", "ablation", "ml",
                                      "runs.jsonl"))
    models, combos = models_block()
    abl = []
    for kind in sorted({r["kind"] for r in nn_runs}):
        abl += ablation_rows("nn", nn_runs, kind)
    ml_primary = "vote(ET+RF)"
    abl += ablation_rows("ml", ml_runs, ml_primary)
    voting = []
    ml_base = group(ml_runs).get("baseline") if False else None
    for kind in sorted({r["kind"] for r in ml_runs}):
        base = [r for r in ml_runs if r["kind"] == kind
                and r["config"] == "baseline"]
        if base:
            e = ST.summarise(base)
            voting.append({"family": "ml", "model": kind, "seeds": e["seeds"],
                           "acc_mean": e["acc_mean"], "acc_std": e["acc_std"],
                           "f1_mean": e["f1_mean"], "f1_std": e["f1_std"],
                           "latency_ms": e.get("latency_ms"),
                           "size_bytes": e.get("size_bytes"),
                           "robust_noise": e.get("robust_noise"),
                           "robust_shift": e.get("robust_shift"),
                           "robust_gain": e.get("robust_gain")})
    defaults = (decide([r for r in abl if r["family"] == "nn"], "nn")
                + decide([r for r in abl if r["family"] == "ml"], "ml"))
    doc = {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "protocol": "leave-one-person-out, 5 people, project windows / "
                        "dedup / derived calibration; robustness probes on "
                        "the held-out person's raw windows (noise, band "
                        "shifted one pod, 25 % weaker)",
            "rule": "training/ablation_stats.py",
            "people": sorted({f["person"] for r in nn_runs + ml_runs
                              for f in r["folds"]}),
            "seeds": {"nn": max([r["seed"] for r in nn_runs], default=-1) + 1,
                      "ml": max([r["seed"] for r in ml_runs], default=-1) + 1},
            "hardware": "i5-9300H 4C/8T, 7.9 GB RAM, GTX 1650 4 GB",
            "software": software_block(),
            "note": "latencies are measured while other jobs ran; see the "
                    "report for the quiet-machine benchmark"},
        "models": models, "ensembles": combos, "ablation": abl,
        "variants": variants_block(), "without": without_blocks(),
        "voting": voting, "new_models": new_models_block(),
        "defaults": defaults, "tests": tests_block(),
        "dataset": dataset_block(), "starred_votes": starred_votes_block()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(models)} models, "
          f"{len(doc['variants'])} trained-without variants, "
          f"{len(abl)} ablation rows, {len(defaults)} default decisions, "
          f"{len(doc['without'])} 'without' ablation(s)")
    if "--print" in argv:
        for d in defaults:
            print(f"  {d['family']:<3} {d['setting']:<13} -> {d['value']!s:<28} "
                  f"{d['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
Ablation study for the ML classifier: which switches earn their place as
defaults, and whether voting beats its members.

    python tools/ablation_ml.py                      # the whole study
    python tools/ablation_ml.py --seeds 1 --out DIR  # a quick smoke test
    python tools/ablation_ml.py --summary-only

The deployed model is the soft vote of ET-500 and RF-500-leaf3 on the
bandpass / `level` features of 1.0 s windows (docs/COMB_TEST.md); that setup is
fixed here, and what varies is one switch at a time:

    aug:<type>      training windows augmented (two extra copies each) -- the
                    same five augmentations the nets use (training/nn_augment)
    adapt           per-person standardisation of the feature vectors from a
                    random 10 % of the person's own windows, no labels
    calib:<mode>    the per-wearer calibration mode (default `affine`)

Stage B tries only the combinations of what helped. The voting question is
answered from the baseline run: every member alone, the deployed pair, and the
four-model vote that `models/model_spec.json` currently asks for.

Leave-one-person-out, several seeds, robustness probes, the judging rule of
`training/ablation_stats.py`: identical to `tools/ablation_nn.py`, so the two
tables are comparable. Output goes to <out>/runs.jsonl (resumable),
<out>/oof_*.npz and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emg_dataset_tool as T                                    # noqa: E402
import ablation_stats as ST                                     # noqa: E402
import nn_augment                                               # noqa: E402
import nn_cv                                                    # noqa: E402
import train_model as TM                                        # noqa: E402
import train_nn as NN                                           # noqa: E402
from emg_app import adaptation as AD                            # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "results", "ablation", "ml")
SETUP = {"window_s": 1.0, "overlap": 0.75, "representation": "bandpass",
         "freq_features": False, "feature_set": "level"}
PRIMARY = "vote(ET+RF)"
MEMBERS = ["ET-500", "RF-500-leaf3", "svm_rbf", "lda"]
COPIES = 2                                   # augmented copies per window
CALIB_MODES = ["off", "gain", "per_channel"]  # baseline is the app's `affine`


def fmt_s(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else \
        f"{s // 60} min {s % 60:02d} s"


def config_name(augment, adapt, calib=None):
    parts = []
    if calib:
        parts.append("calib:" + calib)
    if augment:
        parts.append("aug:" + "+".join(augment))
    if adapt:
        parts.append("adapt")
    return "|".join(parts) or "baseline"


# ---- data -------------------------------------------------------------------


def spec_entries():
    spec = json.load(open(os.path.join(ROOT, "models", "model_spec.json"),
                          encoding="utf-8"))
    return {m["name"]: m for m in spec["models"]}


def make_member(name, entries, seed):
    pipe = TM._make_pipeline(entries[name], for_scoring=False)
    clf = pipe.named_steps["clf"]
    if "random_state" in clf.get_params():
        pipe.set_params(clf__random_state=int(seed))
    if "n_jobs" in clf.get_params():
        pipe.set_params(clf__n_jobs=-1)
    return pipe


def build_windows(root, calib_mode, exclude_persons=()):
    """Uncalibrated windows + the per-window wearer calibration, deduped."""
    recs, _ = TM.load_recordings(root, persons=None,
                                 exclude_codes=NN.EXCLUDE_SETS["no_errors"],
                                 exclude_persons=exclude_persons)
    people = sorted({r.person_id for r in recs})
    cal, _, _ = T.training_calibrations(root, calib_mode, recs, people)
    win = max(4, int(round(SETUP["window_s"] * TM.MODEL_FS)))
    step = max(1, int(round(win * (1.0 - SETUP["overlap"]))))
    labels = sorted({r.label for r in recs})
    index = {n: i for i, n in enumerate(labels)}
    staged = []
    for rec in recs:
        a, b = rec.label_start_index, rec.label_end_index
        if b - a < TM.MIN_LABEL_WINDOW_SAMPLES:
            continue
        seconds = max(1e-6, float(rec.t_rel[b - 1] - rec.t_rel[a]))
        n_out = max(win, int(round(seconds * TM.MODEL_FS)))
        block = T.resample_uniform(rec.emg[a:b], rec.t_rel[a:b], n_out)
        windows = T.sliding_windows(block.astype(np.float32), win, step)
        if windows.shape[0]:
            staged.append((rec, windows, win, step, n_out))
    masked = TM.dedupe_windows(staged, keep_report={})
    W, y, persons, uid, cals = [], [], [], [], []
    for rec, windows, mask in masked:
        for i in np.flatnonzero(mask):
            W.append(windows[i])
            y.append(index[rec.label])
            persons.append(rec.person_id)
            uid.append(rec.uid)
            cals.append(cal.get(rec.person_id))
    return {"W": np.ascontiguousarray(np.asarray(W, dtype=np.float32)),
            "y": np.asarray(y, dtype=np.int64),
            "persons": np.asarray(persons, dtype=object),
            "uid": np.asarray(uid, dtype=object), "cals": cals,
            "labels": labels}


def featurize(W, cals):
    return np.asarray([TM.feature_vector(W[i], SETUP["representation"],
                                         SETUP["freq_features"],
                                         SETUP["feature_set"], cals[i])
                       for i in range(len(W))], dtype=np.float32)


def augmented(W, kinds, seed, copy):
    import torch
    aug = nn_augment.Augmenter(kinds, seed=seed * 100 + copy, device="cpu")
    with torch.no_grad():
        return aug(torch.from_numpy(W)).numpy().astype(np.float32)


def vote(probs_by_name, names):
    return np.mean([probs_by_name[n] for n in names], axis=0)


# ---- one (config, seed) -----------------------------------------------------

VARIANTS = ("noise", "shift+", "shift-", "gain")


def fold_from_probs(P, Pp, rows, y, held, n_classes):
    """One held-out person's metrics from clean and perturbed probabilities."""
    pred = P[rows].argmax(1)
    fold = {"person": held, "n": int(len(rows)),
            "acc": float((pred == y[rows]).mean()),
            "f1": nn_cv.macro_f1(y[rows], pred, n_classes)}
    acc = {v: float((Pp[v][rows].argmax(1) == y[rows]).mean())
           for v in VARIANTS}
    fold["robust"] = {"noise": acc["noise"],
                      "shift": 0.5 * (acc["shift+"] + acc["shift-"]),
                      "gain": acc["gain"]}
    return fold


def run_one(D, F, F_pert, augment, adapt, seed, entries, members, log,
            measure_cost=False):
    """
    Fit `members` with every person held out in turn.

    Returns `({model_name: {"folds", "probs"}}, cost, seconds)`. The votes
    (`vote(ET+RF)`, `vote(all 4)`) are means of the members' probabilities,
    on the clean AND on every perturbed copy of the test windows, so a vote's
    robustness is measured, not inferred from its members.
    """
    import joblib
    y, persons = D["y"], D["persons"]
    people = sorted(set(persons.tolist()))
    n_classes = len(D["labels"])
    extra = []
    if augment:
        for c in range(COPIES):
            extra.append(featurize(augmented(D["W"], augment, seed, c),
                                   D["cals"]))
    table = AD.person_feature_table(F, persons, seed=seed) if adapt else None

    def adapt_rows(A, rows):
        return A if table is None else AD.apply_person_features(
            A, persons[rows], table)

    shape = (len(y), n_classes)
    probs = {m: np.zeros(shape, dtype=np.float32) for m in members}
    pprobs = {m: {v: np.zeros(shape, dtype=np.float32) for v in VARIANTS}
              for m in members}
    cost = {}
    t0 = time.perf_counter()
    for held in people:
        tr = np.flatnonzero(persons != held)
        te = np.flatnonzero(persons == held)
        Xtr = np.concatenate([adapt_rows(F[tr], tr)]
                             + [adapt_rows(E[tr], tr) for E in extra], axis=0)
        ytr = np.concatenate([y[tr]] * (1 + len(extra)))
        Xte = adapt_rows(F[te], te)
        fitted = {}
        for m in members:
            pipe = make_member(m, entries, seed)
            pipe.fit(Xtr, ytr)
            fitted[m] = pipe
            probs[m][te] = pipe.predict_proba(Xte)
            for v in VARIANTS:
                pprobs[m][v][te] = pipe.predict_proba(
                    adapt_rows(F_pert[v][te], te))
        if measure_cost and not cost:
            for m, pipe in fitted.items():
                T.serialise_estimator(pipe)
                buf = io.BytesIO()
                joblib.dump(pipe, buf, compress=0)
                x1 = Xte[:1]
                for _ in range(5):
                    pipe.predict_proba(x1)
                ts = []
                for _ in range(40):
                    a = time.perf_counter()
                    pipe.predict_proba(x1)
                    ts.append((time.perf_counter() - a) * 1000)
                cost[m] = {"size_bytes": buf.tell(),
                           "latency_ms": float(np.median(ts))}
        log(f"      held out {held:<8} " + "  ".join(
            f"{m} {(probs[m][te].argmax(1) == y[te]).mean() * 100:5.1f}"
            for m in members))

    named = {m: (probs[m], pprobs[m]) for m in members}
    for vote_name, group in ((PRIMARY, ["ET-500", "RF-500-leaf3"]),
                             ("vote(all 4)", MEMBERS)):
        if all(m in probs for m in group):
            named[vote_name] = (
                vote(probs, group),
                {v: vote({m: pprobs[m][v] for m in group}, group)
                 for v in VARIANTS})
    out = {}
    for name, (P, Pp) in named.items():
        folds = [fold_from_probs(P, Pp, np.flatnonzero(persons == h), y, h,
                                 n_classes) for h in people]
        out[name] = {"folds": folds, "probs": P}
    if cost:
        for vote_name, group in ((PRIMARY, ["ET-500", "RF-500-leaf3"]),
                                 ("vote(all 4)", MEMBERS)):
            if all(m in cost for m in group):
                cost[vote_name] = {
                    "size_bytes": sum(cost[m]["size_bytes"] for m in group),
                    "latency_ms": sum(cost[m]["latency_ms"] for m in group)}
    return out, cost, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--no-combos", action="store_true")
    ap.add_argument("--no-calib", action="store_true",
                    help="skip the calibration-mode switch")
    ap.add_argument("--exclude-person", action="append", default=[],
                    help="run the whole study WITHOUT this person's "
                         "recordings (repeatable); output defaults to "
                         "results/ablation_without_<name>/ml")
    args = ap.parse_args(argv)
    args.exclude_person = sorted(set(args.exclude_person))
    if args.exclude_person and args.out == DEFAULT_OUT:
        tag = "-".join(p.lower() for p in args.exclude_person)
        args.out = os.path.join(ROOT, "results",
                                f"ablation_without_{tag}", "ml")
    raise SystemExit(_main(args))


def _main(args):
    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.jsonl")

    def log(msg):
        print(msg, flush=True)
        with open(os.path.join(args.out, "log.txt"), "a",
                  encoding="utf-8") as fh:
            fh.write(msg + "\n")

    def load():
        runs = []
        if os.path.exists(runs_path):
            for line in open(runs_path, encoding="utf-8"):
                try:
                    runs.append(json.loads(line))
                except ValueError:
                    pass
        return runs

    def grouped(kind):
        g = {}
        for r in load():
            if r["kind"] == kind:
                g.setdefault(r["config"], []).append(r)
        return g

    if not args.summary_only:
        entries = spec_entries()
        root = NN.DATASET_ROOT
        t_start = time.perf_counter()
        cache = {}

        def dataset(calib):
            if calib not in cache:
                log(f"  building windows + features (calibration {calib}) ...")
                D = build_windows(root, calib, args.exclude_person)
                F = featurize(D["W"], D["cals"])
                pert = {}
                for variant in ("shift+", "shift-", "gain"):
                    pert[variant] = featurize(nn_cv.perturb(D["W"], variant),
                                              D["cals"])
                cache[calib] = (D, F, pert)
                log(f"  {len(D['y'])} windows x {F.shape[1]} features")
            return cache[calib]

        default_calib = T.CALIBRATION_MODE or "off"

        def ensure(augment, adapt, calib=None, members=("ET-500",
                                                         "RF-500-leaf3")):
            name = config_name(augment, adapt, calib)
            have = {r["seed"] for r in grouped(PRIMARY).get(name, [])}
            D, F, pert = dataset(calib or default_calib)
            for seed in range(args.seeds):
                if seed in have:
                    continue
                log(f"[ml] {name}  seed {seed}  "
                    f"({fmt_s(time.perf_counter() - t_start)} elapsed)")
                pert_seed = dict(pert)
                pert_seed["noise"] = featurize(
                    nn_cv.perturb(D["W"], "noise", seed), D["cals"])
                out, cost, secs = run_one(
                    D, F, pert_seed, augment, adapt, seed, entries,
                    list(members), log,
                    measure_cost=(name == "baseline" and seed == 0))
                for kind_name, r in out.items():
                    P = r.pop("probs")
                    tag = "".join(ch for ch in kind_name
                                  if ch.isalnum() or ch in "-_")
                    np.savez_compressed(os.path.join(
                        args.out, f"oof_{tag}_"
                                  f"{name.replace(':', '-').replace('|', '_')}"
                                  f"_s{seed}.npz"), probs=P)
                    rec = {"kind": kind_name, "config": name, "seed": seed,
                           "augment": list(augment), "adapt": bool(adapt),
                           "calib": calib or default_calib,
                           "folds": r["folds"], "seconds": secs,
                           "acc_mean": float(np.mean(
                               [f["acc"] for f in r["folds"]])),
                           "f1_mean": float(np.mean(
                               [f["f1"] for f in r["folds"]]))}
                    rec["robust_mean"] = {k: float(np.mean(
                        [f["robust"][k] for f in r["folds"]]))
                        for k in nn_cv.ROBUST_KINDS}
                    if kind_name in cost:
                        rec.update(cost[kind_name])
                    with open(runs_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    if kind_name == PRIMARY:
                        log(f"      -> {PRIMARY} {rec['acc_mean'] * 100:.1f} %"
                            f"  robust {rec['robust_mean']}")

        # stage A
        ensure((), False, members=tuple(MEMBERS))            # + voting study
        for t in nn_augment.AUG_TYPES:
            ensure((t,), False)
        ensure((), True)
        if not args.no_calib:
            for mode in CALIB_MODES:
                if mode != default_calib:
                    ensure((), False, calib=mode)
        # stage B
        if not args.no_combos:
            g = grouped(PRIMARY)
            base = g["baseline"]
            helped = [t for t in nn_augment.AUG_TYPES
                      if config_name((t,), False) in g
                      and ST.judge(base, g[config_name((t,), False)]
                                   )["default_worthy"]]
            adapt_ok = "adapt" in g and ST.judge(base, g["adapt"]
                                                 )["default_worthy"]
            log(f"[ml] helped alone: augment {helped or 'nothing'}, "
                f"adapt {adapt_ok}")
            if len(helped) >= 2:
                ensure(tuple(helped), False)
            if helped and adapt_ok:
                ensure(tuple(helped), True)

    # summary
    summary = {"setup": SETUP, "primary": PRIMARY, "configs": {}, "voting": {}}
    gp = grouped(PRIMARY)
    if "baseline" not in gp:
        print("no baseline runs yet")
        return 1
    base = gp["baseline"]
    for name, runs in gp.items():
        e = ST.summarise(runs)
        e["vs_baseline"] = None if name == "baseline" else ST.judge(base, runs)
        e["augment"], e["adapt"] = runs[0]["augment"], runs[0]["adapt"]
        e["calib"] = runs[0].get("calib")
        summary["configs"][name] = e
    for kind in sorted({r["kind"] for r in load()}):
        gk = grouped(kind).get("baseline")
        if gk:
            summary["voting"][kind] = ST.summarise(gk)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w",
                            encoding="utf-8"), indent=1)
    print(f"\n{'config':<34}{'acc':>8}{'±':>6}{'F1':>8}{'d acc':>8}"
          f"{'ppl+':>6}{'d rob':>8}  verdict")
    for name, e in summary["configs"].items():
        j = e["vs_baseline"]
        print(f"{name:<34}{e['acc_mean'] * 100:8.1f}{e['acc_std'] * 100:6.1f}"
              f"{e['f1_mean'] * 100:8.1f}"
              + ("" if j is None else
                 f"{j['delta_acc'] * 100:+8.1f}{j['people_better']:>5}/"
                 f"{j['n_people']}"
                 + (f"{j['delta_robust'] * 100:+8.1f}"
                    if j["delta_robust"] is not None else "")
                 + f"  {j['verdict']}"))
    print("\nvoting vs members (baseline config)")
    for kind, e in summary["voting"].items():
        print(f"  {kind:<16}{e['acc_mean'] * 100:7.1f} % ± {e['acc_std'] * 100:.1f}"
              f"   F1 {e['f1_mean'] * 100:.1f}")
    return 0


if __name__ == "__main__":
    main()

"""
Ablation study for the neural nets: which switches earn their place as defaults.

    python tools/ablation_nn.py --kind gru                 # the whole study for one net
    python tools/ablation_nn.py --kind gru --seeds 1 --epochs 5 --out DIR   # a smoke test
    python tools/ablation_nn.py --kind gru --summary-only  # redo summary/judging from saved runs

Stage A, one switch at a time against the baseline: each augmentation alone,
and adaptation to the person. Stage B, only combinations of what helped (so
there are a handful of extra trainings, not hundreds): all the augmentations
that helped together, adaptation, and both.

Everything goes through `training/nn_cv.run_loso` -- leave-one-person-out, the
trainer's own `train_one`, several seeds -- and is judged by
`training/ablation_stats.judge` (the rule is written at the top of that file).
Output: <out>/runs.jsonl (one line per config x seed, resumable), <out>/oof_*.npz
(out-of-fold probabilities, kept for the voting analysis), <out>/summary_<kind>.json.

One GPU: do not run two of these at once.
"""
from __future__ import annotations

import argparse
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

import ablation_stats as ST                                     # noqa: E402
import nn_augment                                               # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "results", "ablation", "nn")


def fmt_s(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else \
        f"{s // 60} min {s % 60:02d} s"


def config_name(augment, adapt):
    parts = []
    if augment:
        parts.append("aug:" + "+".join(augment))
    if adapt:
        parts.append("adapt")
    return "|".join(parts) or "baseline"


def load_runs(path):
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    runs.append(json.loads(line))
                except ValueError:
                    pass
    return runs


def group(runs, kind):
    g = {}
    for r in runs:
        if r["kind"] == kind:
            g.setdefault(r["config"], []).append(r)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", default="gru")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--no-combos", action="store_true",
                    help="stage A only")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.jsonl")
    log_path = os.path.join(args.out, "log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if not args.summary_only:
        import torch
        import nn_cv
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available -- refusing to fall back "
                             "to the CPU.")
        device = torch.device("cuda")
        data = nn_cv.load_windows()
        log(f"{datetime.now():%Y-%m-%d %H:%M}  ablation_nn  kind={args.kind}  "
            f"{data['X'].shape[0]} windows, "
            f"{len(set(data['persons'].tolist()))} people, "
            f"{args.seeds} seed(s), {args.epochs} epochs, "
            f"{torch.cuda.get_device_name(0)}")
        t_start = time.perf_counter()

        def ensure(augment, adapt):
            """Run (or skip, if saved) one config for every seed."""
            name = config_name(augment, adapt)
            have = {r["seed"] for r in group(load_runs(runs_path),
                                             args.kind).get(name, [])}
            for seed in range(args.seeds):
                if seed in have:
                    continue
                log(f"[{args.kind}] {name}  seed {seed}  "
                    f"({fmt_s(time.perf_counter() - t_start)} elapsed)")
                t0 = time.perf_counter()
                r = nn_cv.run_loso(
                    args.kind, data, device, args.epochs, seed=seed,
                    augment=augment, adapt=adapt,
                    measure_cost=(name == "baseline" and seed == 0), log=log)
                probs = r.pop("probs")
                np.savez_compressed(os.path.join(
                    args.out, f"oof_{args.kind}_{name.replace(':', '-')}"
                              f"_s{seed}.npz"), probs=probs)
                r.update(config=name, seconds=time.perf_counter() - t0,
                         epochs=args.epochs)
                with open(runs_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r) + "\n")
                log(f"      -> {r['acc_mean'] * 100:.1f} %  "
                    f"robust {r['robust_mean']}")

        # ---- stage A: one switch at a time ---------------------------------
        ensure((), False)
        for t in nn_augment.AUG_TYPES:
            ensure((t,), False)
        ensure((), True)

        # ---- stage B: only what helped, together ---------------------------
        if not args.no_combos:
            g = group(load_runs(runs_path), args.kind)
            base = g["baseline"]
            helped = [t for t in nn_augment.AUG_TYPES
                      if config_name((t,), False) in g
                      and ST.judge(base, g[config_name((t,), False)]
                                   )["default_worthy"]]
            adapt_ok = "adapt" in g and ST.judge(base, g["adapt"]
                                                 )["default_worthy"]
            log(f"[{args.kind}] helped alone: augment {helped or 'nothing'}, "
                f"adapt {adapt_ok}")
            if len(helped) >= 2:
                ensure(tuple(helped), False)
            if helped and adapt_ok:
                ensure(tuple(helped), True)

    # ---- summary ------------------------------------------------------------
    g = group(load_runs(runs_path), args.kind)
    if "baseline" not in g:
        raise SystemExit("no baseline runs yet")
    base = g["baseline"]
    summary = {"kind": args.kind, "seeds": len(base),
               "epochs": base[0].get("epochs"), "configs": {}}
    for name, runs in g.items():
        entry = ST.summarise(runs)
        entry["vs_baseline"] = None if name == "baseline" else ST.judge(base,
                                                                        runs)
        entry["augment"] = runs[0]["augment"]
        entry["adapt"] = runs[0]["adapt"]
        summary["configs"][name] = entry
    with open(os.path.join(args.out, f"summary_{args.kind}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
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
                    if j["delta_robust"] is not None else f"{'':>8}")
                 + f"  {j['verdict']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

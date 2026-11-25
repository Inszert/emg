"""
Time the deployed models per live window, on real recordings.

    python tools/bench_models.py [--windows 300] [--json out.json]

What it measures, all on this machine:
  * the LIVE CLASSIFIER path -- LivePredictor.predict(): resample, features,
    calibration and predict_proba of the whole deployed model;
  * every member of a soft vote on its own (predict_proba on the same
    feature rows), so the cost of each model in the vote is visible;
  * the NEURAL NET through the real worker (training/nn_serve.py), round
    trip per window, on CUDA and on the CPU.

Windows are cut from the wearer's own recordings at the model's window
length, so the input is the real signal rather than noise. Accuracy is NOT
measured here -- it comes from the trainers' cross-validation
(docs/MODELS_COMPARED.md).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import numpy as np                                               # noqa: E402

import emg_dataset_tool as T                                     # noqa: E402


def real_windows(person, seconds, count, seed=0):
    """(emg, t) slices of `seconds` from `person`'s recordings."""
    recs = T.load_dataset(os.path.join(T.PROJECT_DIR, T.DATA_FOLDER),
                          stage="prefer_trimmed")
    recs = [r for r in recs if r.person_id == person] or recs
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        rec = recs[int(rng.integers(len(recs)))]
        t = rec.timestamps
        if t[-1] - t[0] <= seconds:
            continue
        start = float(rng.uniform(t[0], t[-1] - seconds))
        keep = (t >= start) & (t <= start + seconds)
        if keep.sum() >= 8:
            out.append((rec.emg[keep], t[keep]))
    return out


def stats_ms(samples):
    a = np.asarray(samples) * 1000.0
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n": int(a.size)}


def bench_ml(person, count):
    pred = T.LivePredictor()
    if not pred.load():
        raise SystemExit(f"model not loadable: {pred.status}")
    windows = real_windows(person, pred.window_seconds() * 1.05, count)
    for emg, t in windows[:5]:                                 # warm-up
        pred.predict(emg, t)
    total, feats, rows = [], [], []
    for emg, t in windows:
        t0 = time.perf_counter()
        pred.predict(emg, t)
        total.append(time.perf_counter() - t0)
        rows.append(pred.last_features)
        t1 = time.perf_counter()
        block = T.resample_uniform(emg, t, pred.window_samples)
        T.model_feature_vector(block, pred.representation, pred.feature_set,
                               pred.freq_features, pred.fs, pred.calibration)
        feats.append(time.perf_counter() - t1)
    result = {"model": pred.stats.get("model_name"),
              "window_s": pred.window_seconds(),
              "whole_prediction": stats_ms(total),
              "features_only": stats_ms(feats), "members": {}}
    model = pred.model
    members = getattr(model, "estimators_", None)
    names = [n for n, _ in (getattr(model, "estimators", None) or [])]
    if members is not None:
        for name, member in zip(names, members):
            for row in rows[:5]:
                member.predict_proba(row)
            times = []
            for row in rows:
                t0 = time.perf_counter()
                member.predict_proba(row)
                times.append(time.perf_counter() - t0)
            result["members"][name] = stats_ms(times)
    return result


def bench_nn(person, count, device):
    eng = T.NeuralEngine(device=device)
    sidecar = T.load_nn_stats(eng.model_path) or {}
    seconds = float(sidecar.get("window_s") or 0.7)
    if not eng.start():
        return {"device": device, "error": eng.status}
    t0 = time.time()
    while eng.state not in ("ready", "error") and time.time() - t0 < 90:
        eng.poll()
        time.sleep(0.05)
    if eng.state != "ready":
        status = eng.status
        eng.stop()
        return {"device": device, "error": status}
    windows = real_windows(person, seconds * 1.05, count, seed=1)
    rtt, infer = [], []
    for i, (emg, t) in enumerate(windows):
        if not eng.submit(emg, t):
            continue
        deadline = time.time() + 5
        got = []
        while not got and time.time() < deadline:
            got = eng.poll()
            if not got:
                time.sleep(0.0005)
        if got and i >= 5:                                     # skip warm-up
            rtt.append(got[-1].latency_ms / 1000.0)
            infer.append(eng.infer_ms / 1000.0
                         if np.isfinite(eng.infer_ms) else np.nan)
    status = eng.status
    eng.stop()
    infer = [x for x in infer if np.isfinite(x)]
    return {"device": device, "running_on": status,
            "round_trip": stats_ms(rtt),
            "forward_pass": stats_ms(infer) if infer else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--person", default=T.PERSON_ID)
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {"machine": {"python": sys.version.split()[0],
                       "cpu_count": os.cpu_count()},
           "ml": bench_ml(args.person, args.windows),
           "nn": [bench_nn(args.person, args.windows, d)
                  for d in ("cuda", "cpu")]}
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

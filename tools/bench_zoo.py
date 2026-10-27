"""
Latency of every zoo model for ONE live window, on an idle machine.

    python tools/bench_zoo.py            # measure all, write meta.json["latency_ms"]
    python tools/bench_zoo.py --dry      # measure, print, write nothing

ML: `LivePredictor.predict` on a synthetic window -- resample, features and
`predict_proba`, single-threaded, exactly the live path. NN: one forward pass of
batch 1 on CUDA, synchronised (the worker's own work, without the pipe).
Run it when nothing else is training: the figures in results/experiments.json are
only as quiet as the machine was.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import emg_dataset_tool as T                                       # noqa: E402
from emg_app import zoo                                            # noqa: E402

DRY = "--dry" in sys.argv


def ml_latency(entry):
    p = T.LivePredictor(entry.file)
    if not p.load():
        return None
    rng = np.random.default_rng(0)
    n = max(8, int(p.window_s * 50) + 4)
    emg = np.abs(rng.normal(120, 50, (n, 8))) + 15
    t = np.arange(n) / 50.0
    for _ in range(10):
        p.predict(emg, t)
    ts = []
    for _ in range(100):
        a = time.perf_counter()
        p.predict(emg, t)
        ts.append((time.perf_counter() - a) * 1000)
    return float(np.median(ts))


def nn_latency(entry):
    import torch
    import nn_arch
    ck = torch.load(entry.file, map_location="cpu", weights_only=False)
    model = nn_arch.build(ck["kind"], ck["n_channels"], len(ck["labels"]),
                          ck["win_samples"], hidden=ck["hidden"],
                          dropout=ck["dropout"], seed=0)
    model.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    x = torch.zeros((1, ck["win_samples"], ck["n_channels"]), device=dev)
    with torch.no_grad():
        for _ in range(30):
            model(x)
        ts = []
        for _ in range(300):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            a = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - a) * 1000)
    return float(np.median(ts))


for entry in zoo.list_entries():
    if not entry.exists:
        continue
    ms = ml_latency(entry) if entry.family == "ml" else nn_latency(entry)
    print(f"{entry.id:<22}{'--' if ms is None else f'{ms:8.2f} ms'}")
    if ms is not None and not DRY:
        entry.meta["latency_ms"] = ms
        entry.meta["latency_note"] = ("one live window, measured on an idle "
                                      "machine by tools/bench_zoo.py")
        zoo.write_meta(entry.dir, entry.meta)

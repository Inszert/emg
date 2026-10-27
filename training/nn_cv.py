"""
Leave-one-person-out cross-validation for the neural nets, with the ablation
switches: augmentation (`nn_augment`), adaptation to the person
(`emg_app.adaptation`) and robustness probes. torch is imported lazily.

This is the SAME protocol as `train_nn.py` / `tools/new_models_test.py`:
recordings and filters from `train_model`, derived per-wearer calibration,
0.7 s windows at 0.75 overlap resampled to 60 Hz, window-level dedup,
class-weighted AdamW + cosine, fixed epochs, no early stopping. It exists so
that the ablation study and the shipped trainer cannot measure different
things: `train_nn.train_one` is called, not copied.

With `augment=[]` and `adapt=False` the run is the plain baseline and goes
through the unchanged original code path.

ROBUSTNESS is measured on the held-out person's RAW windows after perturbing
them, scaled with the statistics measured BEFORE the perturbation (what a band
that drifts after calibration looks like):

    noise   multiplicative jitter (sigma 0.15) + additive counts (sigma 15)
    shift   the band slid by one pod, both directions (mean of the two)
    gain    everything 25 % weaker
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emg_dataset_tool as T                                    # noqa: E402
import nn_arch                                                  # noqa: E402
import train_model as TM                                        # noqa: E402
import train_nn as NN                                           # noqa: E402
from emg_app import adaptation as AD                            # noqa: E402

WINDOW_S = 0.7
OVERLAP = 0.75
ROBUST_KINDS = ("noise", "shift", "gain")


def load_windows(root=None, window_s=WINDOW_S, overlap=OVERLAP,
                 exclude_persons=()):
    """The project's windows, calibrated, deduped. Same call as the trainer."""
    root = root or NN.DATASET_ROOT
    recs, _ = TM.load_recordings(root, persons=None,
                                 exclude_codes=NN.EXCLUDE_SETS["no_errors"],
                                 exclude_persons=exclude_persons)
    people = sorted({r.person_id for r in recs})
    mode = T.CALIBRATION_MODE or "off"
    cal, _, _ = T.training_calibrations(root, mode, recs, people)
    built = NN.build_sequences(recs, window_s, overlap, calibration=cal,
                               report={})
    X, y, uid, persons, labels, meta = built
    return {"X": X, "y": y, "uid": uid, "persons": persons, "labels": labels,
            "meta": meta, "calibration": mode if cal else "off",
            "n_recordings": len(recs)}


def perturb(X, kind, seed=0):
    """Raw `(n, T, C)` windows -> perturbed copy (see module docstring)."""
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float32)
    if kind == "noise":
        out = X * np.exp(0.15 * rng.standard_normal(X.shape)).astype(np.float32)
        out = out + 15.0 * rng.standard_normal(X.shape).astype(np.float32)
        return np.clip(out, 0.0, None)
    if kind == "shift+":
        return np.roll(X, 1, axis=2)
    if kind == "shift-":
        return np.roll(X, -1, axis=2)
    if kind == "gain":
        return X * 0.75
    raise ValueError(kind)


def state_dict_bytes(model):
    """Size of the weights as torch would write them."""
    import torch
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               buf)
    return buf.tell()


def latency_ms(model, device, win, n_channels, reps=300):
    """Median time of ONE live window through the net (batch 1, synchronised)."""
    import torch
    model.eval()
    x = torch.zeros((1, win, n_channels), dtype=torch.float32, device=device)
    ts = []
    with torch.no_grad():
        for _ in range(30):
            model(x)
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def run_loso(kind, data, device, epochs, seed=0, augment=(), adapt=False,
             adapt_fraction=AD.ADAPT_FRACTION, robust=True, hidden=None,
             dropout=None, measure_cost=False, log=None):
    """
    One model kind, every person held out in turn.

    Returns a dict: per-fold metrics, the out-of-fold probabilities (`probs`,
    aligned with `data["X"]`) and, with `measure_cost`, size and latency.
    """
    import torch

    X, y, persons = data["X"], data["y"], data["persons"]
    n_classes, win, n_ch = len(data["labels"]), X.shape[1], X.shape[2]
    hidden = nn_arch.HIDDEN if hidden is None else hidden
    dropout = nn_arch.DROPOUT if dropout is None else dropout
    people = sorted(set(persons.tolist()))
    probs_full = np.zeros((len(y), n_classes), dtype=np.float32)
    robust_probs = {}
    folds = []
    cost = {}

    table = (AD.person_log_table(X, persons, adapt_fraction, seed)
             if adapt else None)

    def stats_for(rows, person_of):
        """Per-window (mean, std) arrays for `rows` (adapt) or the global."""
        mean = np.stack([table[p][0] for p in person_of[rows]])
        std = np.stack([table[p][1] for p in person_of[rows]])
        return mean.astype(np.float32), std.astype(np.float32)

    for held in people:
        t0 = time.perf_counter()
        tr = np.flatnonzero(persons != held)
        te = np.flatnonzero(persons == held)
        if adapt:
            m_tr, s_tr = stats_for(tr, persons)
            m_te, s_te = table[held]
            scale_stats = (m_tr, s_tr)
            scaled_tr = ((np.log1p(np.abs(X[tr])) - m_tr[:, None, :])
                         / s_tr[:, None, :]).astype(np.float32)

            def scale_test(Z):
                return ((np.log1p(np.abs(Z)) - m_te) / s_te
                        ).astype(np.float32)
        else:
            scale, mean, std = NN.fold_scaler(X[tr])
            scale_stats = (mean, std)
            scaled_tr = scale(X[tr])
            scale_test = scale

        bins = np.bincount(y[tr], minlength=n_classes).astype(float)
        weights = np.where(bins > 0,
                           bins.sum() / (n_classes * np.maximum(bins, 1)), 1.0)
        model = nn_arch.build(kind, n_ch, n_classes, win, hidden=hidden,
                              dropout=dropout, seed=NN.RANDOM_STATE + seed)
        augmenter = None
        if augment:
            from nn_augment import Augmenter
            augmenter = Augmenter(augment, seed=seed, device=device)
        model = NN.train_one(
            model, X[tr] if augmenter is not None else scaled_tr, y[tr],
            device, epochs, n_classes, class_weights=weights,
            seed=NN.RANDOM_STATE + seed, augmenter=augmenter,
            scale_stats=scale_stats if augmenter is not None else None)

        pred, probs = NN.predict_all(model, scale_test(X[te]), device)
        probs_full[te] = probs
        acc = float((pred == y[te]).mean())
        fold = {"person": held, "n": int(len(te)), "acc": acc,
                "f1": macro_f1(y[te], pred, n_classes)}
        if robust:
            rb = {}
            for name in ROBUST_KINDS:
                if name == "shift":
                    accs = []
                    for sign in ("shift+", "shift-"):
                        p, _ = NN.predict_all(
                            model, scale_test(perturb(X[te], sign, seed)),
                            device)
                        accs.append(float((p == y[te]).mean()))
                    rb[name] = float(np.mean(accs))
                else:
                    p, _ = NN.predict_all(
                        model, scale_test(perturb(X[te], name, seed)), device)
                    rb[name] = float((p == y[te]).mean())
            fold["robust"] = rb
        fold["seconds"] = time.perf_counter() - t0
        folds.append(fold)
        if measure_cost and not cost:
            cost = {"n_parameters": nn_arch.count_parameters(model),
                    "size_bytes": state_dict_bytes(model),
                    "latency_ms": latency_ms(model, device, win, n_ch)}
        if log:
            log(f"      held out {held:<8} {acc * 100:5.1f} %"
                + (f"   noise {fold['robust']['noise'] * 100:4.1f}  shift "
                   f"{fold['robust']['shift'] * 100:4.1f}  gain "
                   f"{fold['robust']['gain'] * 100:4.1f}" if robust else "")
                + f"   {fold['seconds']:5.1f} s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {"kind": kind, "seed": int(seed), "augment": list(augment),
           "adapt": bool(adapt), "folds": folds, "probs": probs_full,
           "acc_mean": float(np.mean([f["acc"] for f in folds])),
           "f1_mean": float(np.mean([f["f1"] for f in folds]))}
    if robust:
        out["robust_mean"] = {k: float(np.mean([f["robust"][k] for f in folds]))
                              for k in ROBUST_KINDS}
    out.update(cost)
    return out


def fit_final(kind, data, device, epochs, seed=0, augment=(), adapt=False,
              adapt_fraction=AD.ADAPT_FRACTION, hidden=None, dropout=None):
    """
    Fit one net on ALL the windows (what gets saved).

    Returns `(model, mean, std, table, train_accuracy)`. `mean`/`std` are the
    GLOBAL scaler -- stored in the checkpoint as its scaler, and what the net
    falls back to when per-person adaptation is switched off. `table` is the
    per-person statistics when `adapt` (else None).
    """
    X, y, persons = data["X"], data["y"], data["persons"]
    n_classes, win, n_ch = len(data["labels"]), X.shape[1], X.shape[2]
    hidden = nn_arch.HIDDEN if hidden is None else hidden
    dropout = nn_arch.DROPOUT if dropout is None else dropout
    scale, mean, std = NN.fold_scaler(X)
    bins = np.bincount(y, minlength=n_classes).astype(float)
    weights = np.where(bins > 0,
                       bins.sum() / (n_classes * np.maximum(bins, 1)), 1.0)
    table = None
    if adapt:
        table = AD.person_log_table(X, persons, adapt_fraction, seed)
        m_all = np.stack([table[p][0] for p in persons]).astype(np.float32)
        s_all = np.stack([table[p][1] for p in persons]).astype(np.float32)
        scaled = ((np.log1p(np.abs(X)) - m_all[:, None, :])
                  / s_all[:, None, :]).astype(np.float32)
        stats = (m_all, s_all)
    else:
        scaled = scale(X)
        stats = (mean, std)
    model = nn_arch.build(kind, n_ch, n_classes, win, hidden=hidden,
                          dropout=dropout, seed=NN.RANDOM_STATE + seed)
    augmenter = None
    if augment:
        from nn_augment import Augmenter
        augmenter = Augmenter(augment, seed=seed, device=device)
    model = NN.train_one(model, X if augmenter is not None else scaled, y,
                         device, epochs, n_classes, class_weights=weights,
                         seed=NN.RANDOM_STATE + seed, augmenter=augmenter,
                         scale_stats=stats if augmenter is not None else None)
    pred, _ = NN.predict_all(model, scaled, device)
    return model, mean, std, table, float((pred == y).mean())

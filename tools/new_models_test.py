"""
Newer sequence models, tried on THIS project's data under THIS project's protocol.

    python tools/new_models_test.py --probe              # time it, predict the total, train nothing
    python tools/new_models_test.py                      # the run (CUDA)
    python tools/new_models_test.py --resume DIR         # continue an interrupted run
    python tools/make_figure_new_models.py [DIR]         # draw docs/figures/novesie_modely_test.png

An EXPERIMENT, not a trainer. It never writes into models/ (no checkpoint, no
sidecar, no spec), never touches the deployed ML model or the net, and puts
everything in its own folder under models/reports/.

What is identical to train_nn.py (imported, not copied, so it cannot drift):
recordings and filters (`TM.load_recordings`), the DERIVED per-wearer
calibration (`T.training_calibrations`), 0.7 s windows at 0.75 overlap
resampled to 60 Hz, window-level dedup, log1p + per-channel standardisation
fitted on the training fold only, and the recipe (AdamW 3e-3, wd 1e-3, cosine,
label smoothing 0.05, batch 64, class-weighted loss, FIXED epochs, no early
stopping on the held-out person).

Protocols, per model:
  loso      leave-one-person-out -- the number to quote. Every window is scored
            by a model that never saw that person.
  dc_only   the same, with each window flattened to its 8 channel means. How
            much of the score is amplitude rather than time (CLAUDE.md 8f).
  contrast  (--contrast) recording-grouped folds: the optimistic within-person
            number, for the generalisation gap.

The held-out accuracy printed / stored every 10 epochs is a MONITOR for the
learning-curve panel. It is never used to pick an epoch, a model or a setting.

Candidates (all small: the data is ~2 900 windows of 42 x 8):
  gru              the current NN, re-run here so the comparison is like for like
  circ_gru         conv over the 8 pods with CIRCULAR padding (the band is a
                   ring) + GRU, trained with random +-1 pod rotations
  chan_transformer each electrode is a token (iTransformer style), 2 layers
  inception        small InceptionTime (3 inception modules, kernels 3/7/15)
  tsmixer          all-MLP time-mixing / feature-mixing blocks (TSMixer)
  rocket           ROCKET-style random convolutional kernels on the GPU +
                   ridge. Not a trained net: only the linear read-out is fitted.
                   Weights are NOT mean-centred (the original centres them,
                   which deletes the level, and the level is the signal here).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emg_dataset_tool as T                                    # noqa: E402
import nn_arch                                                  # noqa: E402
import train_model as TM                                        # noqa: E402
import train_nn as NN                                           # noqa: E402

WINDOW_S = 0.7
OVERLAP = 0.75
EPOCHS = 150
BATCH = 64
LR = 3e-3
WD = 1e-3
SMOOTH = 0.05
HIDDEN = 32
DROPOUT = 0.25
MONITOR_EVERY = 10

MODELS = ["gru", "circ_gru", "chan_transformer", "inception", "tsmixer",
          "rocket"]
TITLES = {
    "gru": "GRU (current NN)",
    "circ_gru": "CircConv-GRU + rotation aug",
    "chan_transformer": "Channel Transformer",
    "inception": "InceptionTime-S",
    "tsmixer": "TSMixer",
    "rocket": "ROCKET-GPU + ridge",
}
NEW = MODELS[1:]


# ==============================================================================
# MODELS
# ==============================================================================


def build_torch_models():
    """The torch classes, built lazily so `--help` does not import torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttentionPool(nn.Module):
        """(B, C, L) -> (B, 2C): attention-weighted mean ++ plain mean (level kept)."""

        def __init__(self, channels):
            super().__init__()
            self.score = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, h):
            w = torch.softmax(self.score(h), dim=-1)
            return torch.cat([(h * w).sum(dim=-1), h.mean(dim=-1)], dim=1)

    class CircConvGRU(nn.Module):
        """
        Conv over (time x pod) with CIRCULAR padding on the pod axis: pod 8 is
        next to pod 1, as on the band. The pod axis is NOT pooled away -- which
        pod is active is what separates WAVE_LEFT from WAVE_RIGHT -- so the net
        keeps pod identity and only learns neighbourhood patterns.
        """

        def __init__(self, n_channels, n_classes, hidden=HIDDEN,
                     dropout=DROPOUT, filters=8):
            super().__init__()
            self.conv = nn.Conv2d(1, filters, kernel_size=(3, 3),
                                  padding=(1, 0))
            self.bn = nn.BatchNorm2d(filters)
            self.rnn = nn.GRU(filters * n_channels, hidden, batch_first=True)
            self.pool = AttentionPool(hidden)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):                                # (B, T, C)
            z = F.pad(x.unsqueeze(1), (1, 1, 0, 0), mode="circular")
            z = F.gelu(self.bn(self.conv(z)))                # (B, f, T, C)
            b, f, t, c = z.shape
            z = z.permute(0, 2, 1, 3).reshape(b, t, f * c)
            out, _ = self.rnn(z)
            return self.head(self.pool(out.transpose(1, 2)))

    class ChanTransformer(nn.Module):
        """Each electrode's whole window is one token (iTransformer style)."""

        def __init__(self, n_channels, n_classes, win, d=32, heads=4,
                     layers=2, dropout=DROPOUT):
            super().__init__()
            self.embed = nn.Linear(win, d)
            self.pos = nn.Parameter(torch.randn(1, n_channels, d) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d, heads, dim_feedforward=2 * d, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            #  Flattened over pods, so pod identity reaches the head.
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d * n_channels, n_classes))

        def forward(self, x):                                # (B, T, C)
            z = self.embed(x.transpose(1, 2)) + self.pos     # (B, C, d)
            z = self.norm(self.enc(z))
            return self.head(z.flatten(1))

    class InceptionModule(nn.Module):
        def __init__(self, c_in, nf=8, kernels=(3, 7, 15)):
            super().__init__()
            self.bottle = nn.Conv1d(c_in, nf, 1, bias=False)
            self.convs = nn.ModuleList(
                [nn.Conv1d(nf, nf, k, padding=k // 2, bias=False)
                 for k in kernels])
            self.pool = nn.MaxPool1d(3, stride=1, padding=1)
            self.pconv = nn.Conv1d(c_in, nf, 1, bias=False)
            self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))

        def forward(self, x):
            b = self.bottle(x)
            out = torch.cat([c(b) for c in self.convs]
                            + [self.pconv(self.pool(x))], dim=1)
            return F.gelu(self.bn(out))

    class InceptionTimeS(nn.Module):
        def __init__(self, n_channels, n_classes, dropout=DROPOUT, nf=8):
            super().__init__()
            width = nf * 4
            self.m1 = InceptionModule(n_channels, nf)
            self.m2 = InceptionModule(width, nf)
            self.m3 = InceptionModule(width, nf)
            self.res = nn.Sequential(nn.Conv1d(n_channels, width, 1,
                                               bias=False),
                                     nn.BatchNorm1d(width))
            #  Mean ++ max: the mean keeps the activation level.
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(width * 2, n_classes))

        def forward(self, x):
            x = x.transpose(1, 2)
            h = self.m3(self.m2(self.m1(x)))
            h = F.gelu(h + self.res(x))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    class MixerBlock(nn.Module):
        def __init__(self, win, d, dropout):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.time = nn.Sequential(nn.Linear(win, win), nn.GELU(),
                                      nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(d)
            self.feat = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d, d),
                                      nn.Dropout(dropout))

        def forward(self, x):                                # (B, T, d)
            x = x + self.time(self.n1(x).transpose(1, 2)).transpose(1, 2)
            return x + self.feat(self.n2(x))

    class TSMixer(nn.Module):
        def __init__(self, n_channels, n_classes, win, d=24, blocks=2,
                     dropout=DROPOUT):
            super().__init__()
            self.inp = nn.Linear(n_channels, d)
            self.blocks = nn.Sequential(*[MixerBlock(win, d, dropout)
                                          for _ in range(blocks)])
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d * 2, n_classes))

        def forward(self, x):
            z = self.blocks(self.inp(x))
            return self.head(torch.cat([z.mean(1), z.amax(1)], dim=1))

    return {"circ_gru": CircConvGRU, "chan_transformer": ChanTransformer,
            "inception": InceptionTimeS, "tsmixer": TSMixer}


def make_net(name, n_channels, n_classes, win, seed):
    import torch
    if name == "gru":
        return nn_arch.build("gru", n_channels, n_classes, win,
                             hidden=HIDDEN, dropout=DROPOUT, seed=seed)
    classes = build_torch_models()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if name == "circ_gru":
        return classes[name](n_channels, n_classes)
    if name == "chan_transformer":
        return classes[name](n_channels, n_classes, win)
    if name == "inception":
        return classes[name](n_channels, n_classes)
    if name == "tsmixer":
        return classes[name](n_channels, n_classes, win)
    raise ValueError(name)


def roll_channels(xb, max_shift=1, p=0.5):
    """Random circular rotation of the pod axis, per sample (training only)."""
    import torch
    b, t, c = xb.shape
    shift = torch.randint(-max_shift, max_shift + 1, (b,), device=xb.device)
    shift = torch.where(torch.rand(b, device=xb.device) < p, shift,
                        torch.zeros_like(shift))
    idx = (torch.arange(c, device=xb.device)[None, :] - shift[:, None]) % c
    return xb.gather(2, idx[:, None, :].expand(b, t, c))


# ---- ROCKET on the GPU -------------------------------------------------------


class RocketGPU:
    """
    Random convolutional kernels (ROCKET) evaluated by `conv1d` on the GPU,
    features = PPV and max per kernel, read out by a ridge classifier whose
    alpha is chosen by recording-grouped CV INSIDE the training fold.

    Differences from the paper, on purpose: weights are not mean-centred (that
    would delete the DC level, which carries most of this dataset's signal), and
    each bias is a random quantile of that kernel's own output on training
    windows (the MiniRocket idea), so PPV is informative for any scale.
    """

    def __init__(self, n_kernels=2000, device="cuda", seed=0):
        self.k = int(n_kernels)
        self.device = device
        self.seed = int(seed)
        self.groups = []

    def _make_kernels(self, n_channels, win):
        import torch
        rng = np.random.default_rng(self.seed)
        spec = defaultdict(list)
        for _ in range(self.k):
            length = int(rng.choice([7, 9, 11]))
            max_exp = math.log2(max(1.0, (win - 1) / (length - 1)))
            dilation = int(2 ** rng.uniform(0, max_exp))
            pad = ((length - 1) * dilation) // 2 if rng.random() < 0.5 else 0
            n_use = int(2 ** rng.uniform(0, math.log2(min(n_channels, length)
                                                      + 1)))
            n_use = max(1, min(n_channels, n_use))
            chans = rng.choice(n_channels, size=n_use, replace=False)
            w = np.zeros((n_channels, length), dtype=np.float32)
            w[chans] = rng.normal(0, 1, (n_use, length)).astype(np.float32)
            spec[(length, dilation, pad)].append((w, rng.uniform(0.1, 0.9)))
        self.groups = []
        for (length, dilation, pad), items in spec.items():
            weights = torch.from_numpy(np.stack([w for w, _ in items])
                                       ).to(self.device)
            quant = np.asarray([q for _, q in items], dtype=np.float32)
            self.groups.append({"w": weights, "dilation": dilation,
                                "pad": pad, "q": quant, "b": None})

    def _conv(self, x, g):
        import torch.nn.functional as F
        return F.conv1d(x, g["w"], dilation=g["dilation"], padding=g["pad"])

    def fit_kernels(self, X):
        """Draw the kernels and set every bias from training windows only."""
        import torch
        n_channels, win = X.shape[2], X.shape[1]
        self._make_kernels(n_channels, win)
        rng = np.random.default_rng(self.seed + 1)
        sample = X[rng.choice(len(X), size=min(256, len(X)), replace=False)]
        xs = torch.from_numpy(np.ascontiguousarray(sample)).to(self.device
                                                                ).transpose(1, 2)
        with torch.no_grad():
            for g in self.groups:
                y = self._conv(xs, g)                       # (n, k, L)
                flat = y.permute(1, 0, 2).reshape(y.shape[1], -1)
                srt = torch.sort(flat, dim=1).values
                pos = torch.from_numpy(
                    (g["q"] * (flat.shape[1] - 1)).astype(np.int64)
                ).to(self.device)
                g["b"] = srt.gather(1, pos[:, None])[:, 0]

    def transform(self, X, chunk=512):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(X), chunk):
                xb = torch.from_numpy(np.ascontiguousarray(X[s:s + chunk])
                                      ).to(self.device).transpose(1, 2)
                feats = []
                for g in self.groups:
                    y = self._conv(xb, g) - g["b"][None, :, None]
                    feats.append((y > 0).float().mean(-1))
                    feats.append(y.amax(-1))
                out.append(torch.cat(feats, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, X, y, uid):
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        self.fit_kernels(X)
        F_tr = self.transform(X)
        self.scaler = StandardScaler().fit(F_tr)
        F_tr = self.scaler.transform(F_tr)
        best, best_acc = 1.0, -1.0
        for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            accs = []
            for a, b in GroupKFold(n_splits=3).split(F_tr, y, uid):
                clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
                clf.fit(F_tr[a], y[a])
                accs.append(float((clf.predict(F_tr[b]) == y[b]).mean()))
            if np.mean(accs) > best_acc:
                best, best_acc = alpha, float(np.mean(accs))
        self.alpha = best
        self.clf = RidgeClassifier(alpha=best, class_weight="balanced")
        self.clf.fit(F_tr, y)
        return self

    def predict_proba(self, X):
        d = self.clf.decision_function(self.scaler.transform(self.transform(X)))
        d = d - d.max(axis=1, keepdims=True)
        e = np.exp(4.0 * d)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    @property
    def n_parameters(self):
        n = sum(int(g["w"].numel()) for g in self.groups)
        return n + int(self.clf.coef_.size)


# ==============================================================================
# TRAIN / PREDICT
# ==============================================================================


def predict_probs(model, X, device, batch=1024):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch])
                                  ).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_net(model, Xtr, ytr, device, epochs, n_classes, aug=None, seed=0,
            monitor=None):
    """
    The train_nn.py recipe. `monitor(model)` -> accuracy is called every
    MONITOR_EVERY epochs for the learning-curve panel only.
    """
    import torch
    import torch.nn as nn

    model = model.to(device)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(ytr)).to(device)
    bins = np.bincount(ytr, minlength=n_classes).astype(float)
    weights = np.where(bins > 0, bins.sum() / (n_classes * np.maximum(bins, 1)),
                       1.0)
    crit = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n = Xt.shape[0]
    losses, mon = [], []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = torch.zeros((), device=device)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            if idx.numel() < 2:
                continue
            xb = Xt[idx]
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yt[idx])
            loss.backward()
            opt.step()
            total += loss.detach() * idx.numel()
        sched.step()
        losses.append(float(total) / n)
        if monitor is not None and ((epoch + 1) % MONITOR_EVERY == 0
                                    or epoch + 1 == epochs):
            mon.append((epoch + 1, float(monitor(model))))
    return model, losses, mon


def latency_ms(name, model_or_rocket, X_one, device, reps=300):
    """Median time to score ONE live window, with the device synchronised."""
    import torch
    x = np.ascontiguousarray(X_one[:1])
    if name == "rocket":
        for _ in range(10):
            model_or_rocket.predict_proba(x)
        ts = []
        for _ in range(60):
            t0 = time.perf_counter()
            model_or_rocket.predict_proba(x)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))
    model_or_rocket.eval()
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        for _ in range(30):
            model_or_rocket(xt)
        ts = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model_or_rocket(xt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


# ==============================================================================
# PROTOCOLS
# ==============================================================================


def run_fold(name, Xtr, ytr, uid_tr, Xte, yte, n_classes, device, epochs,
             seed, curves=True):
    """Fit one model on a training fold, score the test fold."""
    scale, _, _ = NN.fold_scaler(Xtr)
    A, B = scale(Xtr), scale(Xte)
    t0 = time.perf_counter()
    if name == "rocket":
        model = RocketGPU(device=str(device), seed=seed).fit(A, ytr, uid_tr)
        probs = model.predict_proba(B)
        return {"probs": probs, "seconds": time.perf_counter() - t0,
                "n_parameters": model.n_parameters, "losses": [], "monitor": [],
                "alpha": model.alpha, "model": model, "A": A}
    net = make_net(name, A.shape[2], n_classes, A.shape[1], seed)
    n_par = nn_arch.count_parameters(net)
    aug = roll_channels if name == "circ_gru" else None

    def monitor(m):
        return float((predict_probs(m, B, device).argmax(1) == yte).mean())

    net, losses, mon = fit_net(net, A, ytr, device, epochs, n_classes,
                               aug=aug, seed=seed,
                               monitor=monitor if curves else None)
    probs = predict_probs(net, B, device)
    return {"probs": probs, "seconds": time.perf_counter() - t0,
            "n_parameters": n_par, "losses": losses, "monitor": mon,
            "model": net, "A": A}


def protocol_splits(kind, y, uid, persons):
    if kind in ("loso", "dc_only"):
        for p in sorted(set(persons.tolist())):
            yield p, np.flatnonzero(persons != p), np.flatnonzero(persons == p)
    elif kind == "contrast":
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for i, (a, b) in enumerate(cv.split(np.zeros(len(y)), y, uid)):
            yield f"fold{i + 1}", a, b


def run_protocol(name, kind, data, device, epochs, seed, log, keep_model=False):
    X, y, uid, persons = data["X"], data["y"], data["uid"], data["persons"]
    if kind == "dc_only":
        X = NN.dc_only_sequences(X)
    n_classes = len(data["labels"])
    probs_full = np.zeros((len(y), n_classes), dtype=np.float32)
    seen = np.zeros(len(y), dtype=bool)
    folds, last = [], None
    for label, tr, te in protocol_splits(kind, y, uid, persons):
        r = run_fold(name, X[tr], y[tr], uid[tr], X[te], y[te], n_classes,
                     device, epochs, seed, curves=(kind == "loso"))
        probs_full[te] = r["probs"]
        seen[te] = True
        acc = float((r["probs"].argmax(1) == y[te]).mean())
        folds.append({"held_out": label, "n": int(len(te)), "acc": acc,
                      "seconds": r["seconds"], "losses": r["losses"],
                      "monitor": r["monitor"],
                      "n_parameters": r["n_parameters"]})
        log(f"      {kind:<8} {name:<17} held out {label:<8} "
            f"{acc * 100:5.1f} %   {r['seconds']:6.1f} s")
        last = r
        if not keep_model:
            r.pop("model", None)
            r.pop("A", None)
    return folds, probs_full, seen, last


# ==============================================================================
# DATA
# ==============================================================================


def load_data(root):
    recs, _ = TM.load_recordings(root, persons=None,
                                 exclude_codes=NN.EXCLUDE_SETS["no_errors"])
    people = sorted({r.person_id for r in recs})
    mode = T.CALIBRATION_MODE or "off"
    cal, _, _ = T.training_calibrations(root, mode, recs, people)
    built = NN.build_sequences(recs, WINDOW_S, OVERLAP, calibration=cal,
                               report={})
    X, y, uid, persons, labels, meta = built
    return {"X": X, "y": y, "uid": uid, "persons": persons, "labels": labels,
            "meta": meta, "calibration": mode if cal else "off",
            "n_recordings": len(recs)}


# ==============================================================================
# MAIN
# ==============================================================================


def fmt_s(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else \
        f"{s // 60} min {s % 60:02d} s"


def probe(data, device, epochs, protocols, models, seeds):
    """Time a few epochs of every model on one real fold; project the total."""
    import torch
    persons = data["persons"]
    held = sorted(set(persons.tolist()))[0]
    tr, te = np.flatnonzero(persons != held), np.flatnonzero(persons == held)
    X, y, uid = data["X"], data["y"], data["uid"]
    scale, _, _ = NN.fold_scaler(X[tr])
    A, B = scale(X[tr]), scale(X[te])
    n_classes = len(data["labels"])
    n_fold = len(set(persons.tolist()))
    print(f"\nPROBE on a real training fold ({len(tr)} windows)")
    total = 0.0
    for name in models:
        if name == "rocket":
            t0 = time.perf_counter()
            RocketGPU(device=str(device)).fit(A, y[tr], uid[tr])
            per_fold = time.perf_counter() - t0
            print(f"  {name:<17} one fold {per_fold:6.1f} s (no epochs)")
        else:
            net = make_net(name, A.shape[2], n_classes, A.shape[1], 0)
            aug = roll_channels if name == "circ_gru" else None
            fit_net(net, A, y[tr], device, 1, n_classes, aug=aug)       # warm up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fit_net(net, A, y[tr], device, 5, n_classes, aug=aug)
            torch.cuda.synchronize()
            per_epoch = (time.perf_counter() - t0) / 5
            per_fold = per_epoch * epochs
            print(f"  {name:<17} {per_epoch * 1000:7.1f} ms/epoch -> "
                  f"{per_fold:6.1f} s per fold at {epochs} epochs   "
                  f"({nn_arch.count_parameters(net)} parameters)")
        cost = per_fold * n_fold * len(protocols) * len(seeds)
        total += cost
        print(f"  {'':<17} x {n_fold} folds x {len(protocols)} protocol(s) x "
              f"{len(seeds)} seed(s) = {fmt_s(cost)}")
    total *= 1.08                       # monitor passes, loading, saving
    end = datetime.fromtimestamp(time.time() + total)
    print(f"\n  PREDICTED TOTAL: {fmt_s(total)}   (finishes about "
          f"{end:%H:%M} if started now)")
    return total


def main(argv=None):
    T.force_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=NN.DATASET_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    ap.add_argument("--contrast", action="store_true",
                    help="also run the recording-grouped (within-person) folds")
    ap.add_argument("--no-dc", action="store_true",
                    help="skip the DC-only control")
    ap.add_argument("--probe", action="store_true",
                    help="time a few epochs, print the predicted total, exit")
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- refusing to fall back to the "
                         "CPU (train_nn.py's rule).")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    data = load_data(args.root)
    persons = sorted(set(data["persons"].tolist()))
    print(f"data: {data['X'].shape[0]} windows of {data['X'].shape[1]} x "
          f"{data['X'].shape[2]} from {data['n_recordings']} recordings, "
          f"{len(persons)} people ({', '.join(persons)}), "
          f"calibration {data['calibration']}")
    print(f"majority class: "
          f"{max(Counter(data['y'].tolist()).values()) / len(data['y']) * 100:.1f} %")

    protocols = ["loso"] + ([] if args.no_dc else ["dc_only"]) + \
        (["contrast"] if args.contrast else [])
    seeds = list(range(args.seeds))

    if args.probe:
        probe(data, device, args.epochs, protocols, args.models, seeds)
        return 0

    out = args.resume or args.out or os.path.join(
        ROOT, "models", "reports",
        "new_models_test_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    results_path = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["protocol"], r["seed"]))
                except ValueError:
                    pass
    np.savez_compressed(os.path.join(out, "data.npz"), y=data["y"],
                        uid=data["uid"].astype(str),
                        persons=data["persons"].astype(str),
                        labels=np.asarray(data["labels"]))
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"started": datetime.now().isoformat(timespec="seconds"),
                   "epochs": args.epochs, "models": args.models,
                   "protocols": protocols, "seeds": seeds,
                   "window_s": WINDOW_S, "overlap": OVERLAP,
                   "calibration": data["calibration"],
                   "n_windows": int(len(data["y"])), "people": persons,
                   "labels": list(data["labels"]),
                   "titles": TITLES}, fh, indent=1)

    log_path = os.path.join(out, "log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    jobs = [(m, p, s) for s in seeds for p in protocols for m in args.models]
    todo = [j for j in jobs if j not in done]
    log(f"output: {out}")
    log(f"{len(todo)} of {len(jobs)} (model, protocol, seed) runs to do")
    t_start = time.perf_counter()
    for number, (name, kind, seed) in enumerate(todo, 1):
        t0 = time.perf_counter()
        log(f"[{number}/{len(todo)}] {TITLES[name]}  --  {kind}, seed {seed}")
        folds, probs, seen, last = run_protocol(
            name, kind, data, device, args.epochs, seed, log,
            keep_model=(kind == "loso" and seed == 0))
        record = {"model": name, "protocol": kind, "seed": seed,
                  "folds": folds, "seconds": time.perf_counter() - t0,
                  "n_parameters": folds[0]["n_parameters"]}
        if kind == "loso" and seed == 0 and last is not None:
            record["latency_ms"] = latency_ms(name, last["model"],
                                              last["A"], device)
            last.pop("model", None)
            last.pop("A", None)
        np.savez_compressed(
            os.path.join(out, f"oof_{name}_{kind}_s{seed}.npz"),
            probs=probs, seen=seen)
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        accs = [f["acc"] for f in folds]
        elapsed = time.perf_counter() - t_start
        eta = elapsed / number * (len(todo) - number)
        log(f"      -> mean {np.mean(accs) * 100:.1f} % over {len(accs)} folds"
            + (f", {record['latency_ms']:.2f} ms/window"
               if "latency_ms" in record else "")
            + f"   [{fmt_s(elapsed)} elapsed, about {fmt_s(eta)} left]")
        with open(os.path.join(out, "progress.json"), "w") as fh:
            json.dump({"done": number, "total": len(todo),
                       "elapsed_s": elapsed, "eta_s": eta,
                       "updated": datetime.now().isoformat(timespec="seconds")},
                      fh)
    log(f"\nfinished in {fmt_s(time.perf_counter() - t_start)}")
    log(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
================================================================================
 CUDA DEEP-MODEL TRAINER  (1D-CNN / LSTM / CNN-LSTM)
================================================================================

 Companion to `train_model.py`. Same data layer, same validation protocol,
 different model family: this one feeds the WINDOWED TIME SERIES to a neural
 net instead of the hand-crafted feature vector.

 Why a separate file:

   * `train_model.py` is the scikit-learn sweep. sklearn has no GPU backend,
     so bolting torch into it would mean two unrelated execution engines in
     one script.
   * The feature contract differs. `train_model.py` and `LivePredictor` share
     `model_feature_vector()` -- that shared function is what stops training
     and inference from drifting apart. A conv net deliberately does NOT use
     it: it learns its own features from the (win, 8) block. Mixing the two
     contracts in one file would make it very easy to feed the wrong thing to
     the wrong model.
   * It saves to its own path. `models/gesture_model.joblib` stays a sklearn
     payload, because that is what the app's LIVE CLASSIFIER panel can load.
     Overwriting it with a torch checkpoint would break the recorder.

 What is deliberately IDENTICAL to train_model.py, and must stay that way:

   * `MODEL_FS` resampling. The stream is 28-70 Hz and unevenly sampled, so a
     fixed sample count would mean a varying real duration.
   * The labeled span is `label_start_index:label_end_index`, nothing wider.
   * StratifiedGroupKFold grouped by RECORDING UID. Windows overlap, so a
     random split leaks a recording across train and test and inflates the
     score. A whole recording is either in train or in test.
   * Leave-one-subject-out as the honest subject-independent number.
   * Per-recording majority vote alongside per-window accuracy.

 Honest-measurement notes, in the same spirit as CLAUDE.md section 7b:

   * There is NO early stopping on the test fold. Stopping when the held-out
     fold looks best is a way of fitting the test set; epochs are fixed.
   * Input standardisation statistics are computed on the TRAINING fold only
     and applied to the test fold. Computing them over the whole dataset
     would leak the test distribution into training.
   * Absolute amplitude is preserved (log1p, then standardise). CLAUDE.md
     records that baseline subtraction measurably hurt on this stream, so it
     is not done here either.
   * A grid over (model kind x window length) is a selection: the best of N
     scored on the same folds is optimistic. The grid is small on purpose and
     every row is printed.

 Run:

     python train_torch.py                     # CUDA, full grid
     python train_torch.py --quick             # one model, one window
     python train_torch.py --device cpu        # no GPU
     python train_torch.py --epochs 80

================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# The project root (one level up) holds emg_dataset_tool and emg_app.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import emg_dataset_tool as T

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASET_ROOT = T.DATA_FOLDER
OUT_DIR = "models"
CHECKPOINT_NAME = "gesture_model_torch.pt"
REPORT_NAME = "reports"
REPORT_DIR = os.path.join(OUT_DIR, REPORT_NAME)

# Must match train_model.py: a window means a fixed DURATION, not a fixed
# sample count. See resample_uniform() in the data layer.
MODEL_FS = 60.0

# Window lengths to try, in SECONDS. A conv net needs a few more samples than
# a statistics-based feature vector does, so this starts later than the
# sklearn sweep's 0.20 s.
WINDOW_SECONDS = [0.30, 0.40, 0.50]
OVERLAP = 0.75

MODEL_KINDS = ["cnn", "lstm", "cnn_lstm"]

# Same filters as the sklearn trainer.
EXCLUDE_SETS = {
    "all": (),
    "no_errors": ("disconnect", "flat_channel", "empty", "invalid_values"),
    "clean": ("disconnect", "flat_channel", "empty", "invalid_values",
              "no_activity"),
}
MIN_LABEL_WINDOW_SAMPLES = 20
CV_SPLITS = 6
RANDOM_STATE = 0

# Training hyperparameters.
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
HIDDEN = 64


# ==============================================================================
# DATA
# ==============================================================================


def load_recordings(root, persons=None, sessions=None, exclude_codes=()):
    """Load recordings, dropping the ones whose quality codes are excluded.

    Intentionally the same logic as train_model.load_recordings so the two
    trainers see the same dataset.
    """
    recs = T.load_dataset(root, persons=persons, sessions=sessions,
                          stage="prefer_trimmed", prefer="npz")
    kept, dropped = [], []
    for rec in recs:
        codes = {i.code for i in rec.issues}
        if codes & set(exclude_codes):
            dropped.append((rec, sorted(codes & set(exclude_codes))))
            continue
        if (rec.label_end_index - rec.label_start_index) < MIN_LABEL_WINDOW_SAMPLES:
            dropped.append((rec, ["label_window_too_short"]))
            continue
        kept.append(rec)
    return kept, dropped


def build_sequences(recs, window_s, overlap):
    """
    Recordings -> (X, y, groups_uid, groups_person, labels).

    X is (n_windows, win, 8) float32 in DEVICE COUNTS -- unscaled. Scaling is
    fold-local (see fold_standardiser) so that no test-fold statistic can
    reach the training set.
    """
    win = max(4, int(round(window_s * MODEL_FS)))
    step = max(1, int(round(win * (1.0 - overlap))))
    labels = sorted({r.label for r in recs})
    index = {name: i for i, name in enumerate(labels)}

    blocks, ys, uids, people = [], [], [], []
    for rec in recs:
        a, b = rec.label_start_index, rec.label_end_index
        if b - a < MIN_LABEL_WINDOW_SAMPLES:
            continue

        # Fixed-duration grid, exactly as train_model.build_dataset does it.
        seconds = max(1e-6, float(rec.t_rel[b - 1] - rec.t_rel[a]))
        n_out = max(win, int(round(seconds * MODEL_FS)))
        block = T.resample_uniform(rec.emg[a:b], rec.t_rel[a:b], n_out)

        windows = T.sliding_windows(block.astype(np.float32), win, step)
        if windows.shape[0] == 0:
            continue

        # sliding_windows returns a read-only strided VIEW. torch.from_numpy
        # needs owned, contiguous, writeable memory, so copy here once.
        blocks.append(np.ascontiguousarray(windows, dtype=np.float32))
        ys.extend([index[rec.label]] * windows.shape[0])
        uids.extend([rec.uid] * windows.shape[0])
        people.extend([rec.person_id] * windows.shape[0])

    if not blocks:
        return None

    X = np.concatenate(blocks, axis=0)
    return (X,
            np.asarray(ys, dtype=np.int64),
            np.asarray(uids),
            np.asarray(people),
            labels)


def fold_standardiser(X_train):
    """
    log1p + per-channel standardisation, fitted on the TRAINING fold only.

    log1p first because the filtered stream spans baseline ~15 to max-effort
    ~1444 (CLAUDE.md section 2): on a linear scale the strong contractions
    dominate the gradient entirely. Absolute level is kept -- it is the part
    that identifies a gesture on this rectified stream, and CLAUDE.md records
    that removing it measurably hurt.
    """
    logged = np.log1p(np.abs(X_train))
    mean = logged.mean(axis=(0, 1), keepdims=True)
    std = logged.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def apply(X):
        return ((np.log1p(np.abs(X)) - mean) / std).astype(np.float32)

    return apply, mean, std


# ==============================================================================
# MODELS
# ==============================================================================


def build_model(kind, n_channels, n_classes, win, seed=RANDOM_STATE):
    """
    One of the three architectures.

    The seed is taken HERE, not in train_one: weight initialisation happens
    at construction, so seeding afterwards leaves the starting point to
    whatever the global RNG had reached. Two fits with the same seed would
    then differ, which makes a result impossible to reproduce and a
    regression impossible to tell from noise.
    """
    import torch
    import torch.nn as nn

    if seed is not None:
        torch.manual_seed(seed)

    class CNN(nn.Module):
        """1D convolutions over time. Conv1d wants (B, C, L)."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, HIDDEN, kernel_size=3, padding=1),
                nn.BatchNorm1d(HIDDEN), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, n_classes))

        def forward(self, x):                 # x: (B, win, C)
            return self.head(self.features(x.transpose(1, 2)))

    class LSTM(nn.Module):
        """Two stacked LSTM layers; the last time step feeds the classifier."""

        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_channels, HIDDEN, num_layers=2,
                               batch_first=True, dropout=DROPOUT)
            self.head = nn.Sequential(
                nn.Dropout(DROPOUT), nn.Linear(HIDDEN, n_classes))

        def forward(self, x):                 # x: (B, win, C)
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    class CNNLSTM(nn.Module):
        """Convolutions compress time, then an LSTM reads the sequence."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32), nn.ReLU(),
            )
            self.rnn = nn.LSTM(32, HIDDEN, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(DROPOUT), nn.Linear(HIDDEN, n_classes))

        def forward(self, x):                 # x: (B, win, C)
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.rnn(z)
            return self.head(out[:, -1, :])

    if kind == "cnn":
        return CNN()
    if kind == "lstm":
        return LSTM()
    if kind == "cnn_lstm":
        return CNNLSTM()
    raise ValueError(f"unknown model kind {kind!r}")


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ==============================================================================
# TRAIN / PREDICT
# ==============================================================================


def train_one(model, X_train, y_train, device, epochs, class_weights=None,
              seed=RANDOM_STATE, log=None):
    """
    Fit on the training fold. No validation split, no early stopping.

    Early stopping on the held-out fold would be fitting the test set, so the
    epoch count is fixed and declared instead.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = model.to(device)
    model.train()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)

    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32,
                                 device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)

    n = Xt.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * idx.numel()
        scheduler.step()
        if log is not None and (epoch + 1) % max(1, epochs // 4) == 0:
            log.append((epoch + 1, total / max(1, n)))

    return model


def predict_all(model, X, device, batch=512):
    """Class predictions and probabilities for X."""
    import torch

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            chunk = torch.from_numpy(X[start:start + batch]).to(device)
            logits = model(chunk)
            p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(probs)


# ==============================================================================
# EVALUATION  (same protocol as train_model.py)
# ==============================================================================


def evaluate(kind, X, y, groups, groups_person, labels, win, device, epochs,
             n_splits=CV_SPLITS, verbose=False):
    """
    Grouped stratified cross-validation, grouped by RECORDING UID.

    Windows from one recording overlap heavily; letting them straddle the
    split would mean testing on near-copies of training rows.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    counts = Counter(y.tolist())
    n_groups = len(set(groups.tolist()))
    n_splits = int(min(n_splits, min(counts.values()), n_groups))
    if n_splits < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    window_acc, window_f1, fold_acc = [], [], []
    vote_correct = vote_total = 0
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    n_params = 0

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        scale, _, _ = fold_standardiser(X[train_idx])
        Xtr, Xte = scale(X[train_idx]), scale(X[test_idx])

        # Inverse-frequency class weights, computed on the training fold.
        bins = np.bincount(y[train_idx], minlength=len(labels)).astype(float)
        weights = np.where(bins > 0, bins.sum() / (len(labels) * np.maximum(bins, 1)), 1.0)

        model = build_model(kind, X.shape[2], len(labels), win)
        n_params = count_parameters(model)
        log = [] if verbose else None
        model = train_one(model, Xtr, y[train_idx], device, epochs,
                          class_weights=weights, log=log)
        if verbose and log:
            trace = "  ".join(f"ep{e}:{l:.3f}" for e, l in log)
            print(f"      fold {fold + 1}  loss  {trace}")

        pred, _ = predict_all(model, Xte, device)
        acc = accuracy_score(y[test_idx], pred)
        window_acc.append(acc)
        fold_acc.append(acc)
        window_f1.append(f1_score(y[test_idx], pred, average="macro",
                                  zero_division=0))
        cm += confusion_matrix(y[test_idx], pred,
                               labels=list(range(len(labels))))

        # Majority vote per recording -- what the live app effectively does.
        per_rec, truth = defaultdict(list), {}
        for pos, gid in enumerate(groups[test_idx]):
            per_rec[gid].append(pred[pos])
            truth[gid] = y[test_idx][pos]
        for gid, votes in per_rec.items():
            winner = Counter(votes).most_common(1)[0][0]
            vote_total += 1
            vote_correct += int(winner == truth[gid])

    return {
        "n_splits": n_splits,
        "window_accuracy": float(np.mean(window_acc)),
        "window_accuracy_std": float(np.std(window_acc)),
        "window_macro_f1": float(np.mean(window_f1)),
        "recording_vote_accuracy": (vote_correct / vote_total
                                    if vote_total else 0.0),
        "n_recordings_voted": vote_total,
        "fold_accuracies": [float(a) for a in fold_acc],
        "confusion_matrix": cm.tolist(),
        "n_parameters": n_params,
    }


def leave_one_subject_out(kind, X, y, groups_person, labels, win, device,
                          epochs):
    """Subject-independent score. None when there is only one subject."""
    from sklearn.metrics import accuracy_score

    people = sorted(set(groups_person.tolist()))
    if len(people) < 2:
        return None

    scores = {}
    for person in people:
        test = np.flatnonzero(groups_person == person)
        train = np.flatnonzero(groups_person != person)
        if not len(test) or not len(train):
            continue
        if len(set(y[train].tolist())) < 2:
            continue
        scale, _, _ = fold_standardiser(X[train])
        model = build_model(kind, X.shape[2], len(labels), win)
        model = train_one(model, scale(X[train]), y[train], device, epochs)
        pred, _ = predict_all(model, scale(X[test]), device)
        scores[person] = float(accuracy_score(y[test], pred))

    if not scores:
        return None
    return {
        "per_subject": scores,
        "mean": float(np.mean(list(scores.values()))),
        "std": float(np.std(list(scores.values()))),
    }


def print_confusion(cm, labels):
    width = max(len(l) for l in labels) + 1
    print(" " * (width + 2) + " ".join(f"{l[:6]:>6}" for l in labels)
          + "    recall")
    for i, name in enumerate(labels):
        row = cm[i]
        total = row.sum()
        recall = row[i] / total * 100 if total else 0.0
        print(f"  {name:<{width}} " + " ".join(f"{v:>6d}" for v in row)
              + f"   {recall:5.1f} %")


# ==============================================================================
# DEVICE
# ==============================================================================


def resolve_device(requested):
    """
    Pick the torch device, and FAIL LOUDLY rather than fall back silently.

    A silent CPU fallback is the worst outcome: the sweep would run for an
    hour and the operator would never learn the GPU was idle.
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. For CUDA on this machine:\n"
            "  python -m pip install torch==2.9.1+cu128 "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            f"({exc})")

    if requested == "cpu":
        return torch.device("cpu"), {"device": "cpu",
                                     "torch": torch.__version__}

    if not torch.cuda.is_available():
        build = torch.__version__
        hint = ("this is a CPU-only torch build -- reinstall from the CUDA "
                "index" if "+cpu" in build or "+cu" not in build else
                "CUDA build present but no usable GPU/driver was found")
        raise SystemExit(
            f"CUDA was requested but torch.cuda.is_available() is False.\n"
            f"  torch build : {build}\n"
            f"  diagnosis   : {hint}\n"
            f"  install     : python -m pip install torch==2.9.1+cu128 "
            f"--index-url https://download.pytorch.org/whl/cu128\n"
            f"  or run on the CPU with: --device cpu")

    index = 0
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    info = {
        "device": f"cuda:{index}",
        "gpu": name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_mb": int(props.total_memory / 1024 / 1024),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    # Deterministic-ish: cudnn autotuning picks different kernels per run,
    # which changes float accumulation order and therefore the result.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(f"cuda:{index}"), info


def compute_banner(info):
    """
    A box saying GPU or CPU, loudly.

    The whole point of the CUDA path is that it is used, and a run that
    quietly fell back to the CPU looks exactly like one that did not --
    except that it takes twenty times as long. So the answer goes on screen
    in a shape that cannot be missed in a scrolling log, at the start of the
    run and again at the end.
    """
    width = 74
    if info.get("device", "cpu") != "cpu":
        lines = [
            f"COMPUTE:  >>> GPU <<<   {info.get('gpu', 'CUDA device')}",
            f"          CUDA {info.get('cuda_runtime')}  |  "
            f"torch {info.get('torch')}  |  "
            f"{info.get('total_memory_mb')} MB  |  "
            f"capability {info.get('capability')}",
        ]
    else:
        lines = [
            "COMPUTE:  >>> CPU <<<   no GPU in use",
            f"          torch {info.get('torch')}  |  "
            f"{os.cpu_count()} logical cores  |  expect this to be slow",
        ]
    # Size the box to its contents: a long GPU name must not push the right
    # edge out and leave a ragged frame.
    field = max(max(len(line) for line in lines), width - 4)
    rule = "+" + "-" * (field + 2) + "+"
    body = "\n".join(f"| {line:<{field}} |" for line in lines)
    return f"\n{rule}\n{body}\n{rule}"


# ==============================================================================
# MAIN
# ==============================================================================


def main(argv=None):
    T.force_utf8_console()

    parser = argparse.ArgumentParser(
        description="Train a CUDA 1D-CNN / LSTM EMG gesture model")
    parser.add_argument("--root", default=DATASET_ROOT,
                        help="dataset root folder")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="where the checkpoint and reports go")
    parser.add_argument("--model-path", default=None,
                        help="explicit checkpoint path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="cuda (default) or cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true",
                        help="one model, one window length")
    parser.add_argument("--kind", action="append", default=None,
                        choices=MODEL_KINDS, help="restrict the model grid")
    parser.add_argument("--window", action="append", type=float, default=None,
                        help="restrict the window grid, in seconds")
    parser.add_argument("--person", action="append", default=None)
    parser.add_argument("--exclude", default="no_errors",
                        choices=sorted(EXCLUDE_SETS))
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-fold training loss")
    args = parser.parse_args(argv)

    print("=" * 78)
    print(" CUDA DEEP-MODEL TRAINER  --  1D-CNN / LSTM / CNN-LSTM")
    print("=" * 78)

    device, dev_info = resolve_device(args.device)
    print(compute_banner(dev_info))
    print("\nCompute")
    for key, value in dev_info.items():
        print(f"  {key:<14}: {value}")
    print(f"  {'host':<14}: {platform.platform()}")

    # ---- data -------------------------------------------------------------
    recs, dropped = load_recordings(args.root, persons=args.person,
                                    exclude_codes=EXCLUDE_SETS[args.exclude])
    if not recs:
        raise SystemExit(f"No usable recordings under {args.root!r}.")

    classes = sorted({r.label for r in recs})
    people = sorted({r.person_id for r in recs})
    print("\nDataset")
    print(f"  root          : {os.path.abspath(args.root)}")
    print(f"  recordings    : {len(recs)} used, {len(dropped)} dropped "
          f"(filter: {args.exclude})")
    print(f"  classes       : {len(classes)}  {', '.join(classes)}")
    for name in classes:
        print(f"      {name:<12} {sum(1 for r in recs if r.label == name)}")
    print(f"  subjects      : {', '.join(people)}")

    if len(classes) < 2:
        raise SystemExit(
            f"Only one class ({classes[0]}) -- a classifier needs at least "
            f"two. Record another gesture first.")

    chance = max(Counter(r.label for r in recs).values()) / len(recs)
    print(f"  majority-class baseline: {chance * 100:.1f} %  "
          f"(any model must beat this, not 1/n)")

    kinds = args.kind or MODEL_KINDS
    windows = args.window or WINDOW_SECONDS
    if args.quick:
        kinds, windows = kinds[:1], windows[:1]

    print("\nProtocol")
    print(f"  resampled to  : {MODEL_FS:.0f} Hz before windowing")
    print(f"  validation    : StratifiedGroupKFold grouped by RECORDING uid "
          f"({CV_SPLITS} splits max)")
    print(f"  epochs        : {args.epochs} fixed, no early stopping on the "
          f"test fold")
    print(f"  scaling       : log1p + standardise, fitted per training fold")
    print(f"  grid          : {len(kinds)} models x {len(windows)} windows "
          f"= {len(kinds) * len(windows)} candidates")
    if len(kinds) * len(windows) > 1:
        print("  NOTE: picking the best of several candidates on the same "
              "folds makes\n        the winner's score optimistic. Every row "
              "is printed below.")

    # ---- sweep ------------------------------------------------------------
    results = []
    total = len(kinds) * len(windows)
    done = 0
    t_start = time.perf_counter()

    for window_s in windows:
        built = build_sequences(recs, window_s, OVERLAP)
        if built is None:
            print(f"\n  window {window_s:.2f} s -> no usable windows, skipped")
            continue
        X, y, groups, groups_person, labels = built
        win = X.shape[1]
        print(f"\n--- window {window_s:.2f} s  ({win} samples @ {MODEL_FS:.0f} Hz)"
              f"  {X.shape[0]} windows from {len(set(groups.tolist()))} recordings")

        for kind in kinds:
            done += 1
            t0 = time.perf_counter()
            stats = evaluate(kind, X, y, groups, groups_person, labels, win,
                             device, args.epochs, verbose=args.verbose)
            elapsed = time.perf_counter() - t0
            if stats is None:
                print(f"  [{done}/{total}] {kind:<10} not enough groups to "
                      f"cross-validate")
                continue

            stats.update({
                "kind": kind,
                "window_s": window_s,
                "win_samples": win,
                "overlap": OVERLAP,
                "n_windows": int(X.shape[0]),
                "seconds": elapsed,
            })
            results.append(stats)
            print(f"  [{done}/{total}] {kind:<10} "
                  f"window {stats['window_accuracy'] * 100:5.1f} % "
                  f"+-{stats['window_accuracy_std'] * 100:4.1f}   "
                  f"vote {stats['recording_vote_accuracy'] * 100:5.1f} %   "
                  f"f1 {stats['window_macro_f1'] * 100:5.1f} %   "
                  f"{stats['n_parameters']:>7d} params   {elapsed:5.1f} s")

    if not results:
        raise SystemExit("Nothing could be evaluated.")

    # ---- ranking ----------------------------------------------------------
    results.sort(key=lambda r: (-r["recording_vote_accuracy"],
                                -r["window_accuracy"]))
    print("\n" + "=" * 78)
    print(" ALL CANDIDATES  (ranked by per-recording vote, then per-window)")
    print("=" * 78)
    print(f"  {'model':<10} {'win':>6} {'window acc':>12} {'vote acc':>10} "
          f"{'macro f1':>9} {'params':>8}")
    for r in results:
        print(f"  {r['kind']:<10} {r['window_s']:>5.2f}s "
              f"{r['window_accuracy'] * 100:>10.1f} % "
              f"{r['recording_vote_accuracy'] * 100:>8.1f} % "
              f"{r['window_macro_f1'] * 100:>7.1f} % "
              f"{r['n_parameters']:>8d}")

    best = results[0]
    print("\n" + "=" * 78)
    print(f" BEST: {best['kind']}  at {best['window_s']:.2f} s "
          f"({best['win_samples']} samples)")
    print("=" * 78)
    print(f"  per-window accuracy    : {best['window_accuracy'] * 100:.1f} % "
          f"+- {best['window_accuracy_std'] * 100:.1f}")
    print(f"  per-recording vote     : {best['recording_vote_accuracy'] * 100:.1f} % "
          f"over {best['n_recordings_voted']} recordings")
    print(f"  macro F1               : {best['window_macro_f1'] * 100:.1f} %")
    print(f"  folds                  : {best['n_splits']}  "
          f"({', '.join(f'{a * 100:.0f}%' for a in best['fold_accuracies'])})")
    print(f"  majority-class baseline: {chance * 100:.1f} %")
    print(f"  trainable parameters   : {best['n_parameters']}")
    print()
    print_confusion(np.array(best["confusion_matrix"]), classes)

    # ---- subject-independent ---------------------------------------------
    built = build_sequences(recs, best["window_s"], OVERLAP)
    X, y, groups, groups_person, labels = built
    loso = leave_one_subject_out(best["kind"], X, y, groups_person, labels,
                                 X.shape[1], device, args.epochs)
    print()
    if loso:
        print(f"  subject-independent    : {loso['mean'] * 100:.1f} % "
              f"+- {loso['std'] * 100:.1f}")
        for person, score in sorted(loso["per_subject"].items()):
            print(f"      held out {person:<12} {score * 100:5.1f} %")
    else:
        print("  subject-independent    : NOT MEASURABLE -- one subject only.")
        print("                           This model is specific to the person")
        print("                           who recorded it.")

    # ---- final fit + save -------------------------------------------------
    if args.no_save:
        print("\n--no-save: nothing written.")
        print(compute_banner(dev_info))
        print(f"\nTotal time: {time.perf_counter() - t_start:.1f} s")
        return 0

    import torch

    scale, mean, std = fold_standardiser(X)
    bins = np.bincount(y, minlength=len(labels)).astype(float)
    weights = np.where(bins > 0,
                       bins.sum() / (len(labels) * np.maximum(bins, 1)), 1.0)
    model = build_model(best["kind"], X.shape[2], len(labels), X.shape[1])
    model = train_one(model, scale(X), y, device, args.epochs,
                      class_weights=weights)

    out_dir = args.out_dir
    # Derived from --out-dir, not from the module default: a caller that
    # redirects the output must get ALL of it, report included. The same
    # mistake in train_model.py had the GUI overwriting a different model
    # than the one it was using (CLAUDE.md section 13).
    report_dir = os.path.join(out_dir, REPORT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    path = args.model_path or os.path.join(out_dir, CHECKPOINT_NAME)

    payload = {
        "format": "emg_torch_v1",
        "kind": best["kind"],
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": labels,
        "n_channels": int(X.shape[2]),
        "win_samples": int(X.shape[1]),
        "window_s": float(best["window_s"]),
        "overlap": OVERLAP,
        "model_fs": MODEL_FS,
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        # The scaler IS part of the model: inference must reproduce it or the
        # inputs are meaningless.
        "scaler": {"kind": "log1p_standardise",
                   "mean": mean.astype(np.float32),
                   "std": std.astype(np.float32)},
        # Same provenance block train_model.py writes, so either payload can
        # be asked the same question: what was this trained on, and is that
        # still what is on disk?
        "data": T.dataset_fingerprint(args.root),
        "stats": {
            "validation_protocol": "leave-recordings-out "
                                   "(StratifiedGroupKFold on recording uid)",
            "subject_independent": loso,
            "majority_class_baseline": float(chance),
            "n_recordings": len(recs),
            "n_recordings_total": len(recs) + len(dropped),
            "excluded": args.exclude,
            "subjects": people,
            "labels": labels,
            "epochs": args.epochs,
            "early_stopping": False,
            "candidates_evaluated": len(results),
            "selection_bias_note":
                "The winner was chosen as the best of "
                f"{len(results)} candidates scored on the same folds, so its "
                "number is optimistic. Leave-one-subject-out is the figure "
                "that did not pick its own folds.",
            **{k: best[k] for k in (
                "window_accuracy", "window_accuracy_std", "window_macro_f1",
                "recording_vote_accuracy", "n_recordings_voted", "n_splits",
                "fold_accuracies", "confusion_matrix", "n_windows",
                "n_parameters")},
        },
        "device_trained_on": dev_info,
        "created": datetime.now().isoformat(timespec="seconds"),
        # Same key train_model.py uses, so one reader serves both payloads.
        "created_iso": T.iso_time(time.time()),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Archive first, then the working copy -- the same way train_model.py
    # does it. A sweep that takes ten minutes should not be destroyed by the
    # next one, and comparing two checkpoints is how a regression gets found.
    stem = os.path.splitext(os.path.basename(path))[0]
    archive = T.unique_path(
        os.path.join(os.path.dirname(os.path.abspath(path)),
                     f"{stem}_{stamp}.pt"))
    torch.save(payload, archive)
    torch.save(payload, path)
    print(f"\nSaved checkpoint -> {os.path.abspath(path)}")
    print(f"Archived copy    -> {archive}")
    report = os.path.join(report_dir, f"torch_sweep_{stamp}.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"device": dev_info, "chance": chance,
                   "classes": classes, "results": results,
                   "subject_independent": loso}, fh, indent=2)
    print(f"Saved sweep report -> {os.path.abspath(report)}")

    print(compute_banner(dev_info))
    print(f"\nTotal time: {time.perf_counter() - t_start:.1f} s")
    print("\nNOTE: the app's LIVE CLASSIFIER loads the scikit-learn payload at")
    print(f"      {T.MODEL_PATH!r}, not this checkpoint. This file does not")
    print("      change what the recorder predicts with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
================================================================================
 NEURAL NETWORK TRAINER  (CUDA)  --  the SECOND, SEPARATE opinion
================================================================================

 Supersedes `train_torch.py`, which stays on disk as the record of the
 retired 2026-10-14 experiment (CLAUDE.md section 8b) and is not run any
 more. What is new here is not the idea of a neural net -- it is that this
 one is held to the same validation protocol as the scikit-learn sweep, is
 kept strictly out of the deployed answer, and reports its own numbers
 separately so the two can be compared instead of blended.

 --------------------------------------------------------------------------
 THE SEPARATION RULE  (read this before wiring anything together)
 --------------------------------------------------------------------------

 The net writes `models/gesture_nn.pt`. NOTHING fuses it with
 `models/gesture_model.joblib`:

   * `LivePredictor` never opens this file, and the neural engine never
     touches `LivePredictor`;
   * the DECISION TRAIL's soft vote (CLAUDE.md section 8e) is built from the
     sklearn candidates only -- the net is not a member and is never averaged
     into the fusion;
   * the app's LIVE CLASSIFIER answer is unchanged by this file existing. The
     net gets its own START button, its own probability bars and its own
     statistics dialog.

 That is a deliberate design choice, not an implementation shortcut. Two
 independently validated models that disagree are informative; one blended
 number that hides which half was wrong is not. It also keeps the deployment
 story honest: the thing destined for the SBC on the robot is still a 209 KB
 joblib payload that needs neither torch nor CUDA, and
 `tests/test_no_torch.py` still proves it.

 --------------------------------------------------------------------------
 WHAT IS IDENTICAL TO train_model.py, AND MUST STAY SO
 --------------------------------------------------------------------------

   * `MODEL_FS = 60 Hz` resampling of the `label_start_index:label_end_index`
     span. The stream is 28-70 Hz and unevenly sampled, so a fixed sample
     count would be a varying real duration.
   * **window-level dedup** (`train_model.dedupe_windows`). 280 of this
     dataset's recordings are clips tiling a few free takes with ~50 %
     overlap, and uid-grouped CV does not separate them because every clip
     legitimately owns its uid. CLAUDE.md section 8g: this is correctness,
     not a speed-up. The retired trainer did NOT do this, which is one reason
     its numbers were higher than they should have been.
   * **subject-grouped fold selection when two or more wearers are on disk**
     (`train_model.resolve_grouping`, CLAUDE.md section 8h). Choosing a
     winner on recording-grouped folds with several wearers selects for
     within-subject fit and then fails on the next person.
   * **per-wearer calibration** threaded through from `--calibration`
     (CLAUDE.md section 8k), each wearer's windows scaled with that wearer's
     own numbers, labels never consulted.
   * leave-one-subject-out, the majority-class baseline, and the DC-only
     control -- reported next to every number, never instead of it.

 These are imported from `train_model` rather than re-implemented, so there
 is exactly one definition of each and the two trainers cannot drift.

 --------------------------------------------------------------------------
 WHAT IS DELIBERATELY DIFFERENT
 --------------------------------------------------------------------------

   * **No shared feature function.** `model_feature_vector()` is the contract
     binding `train_model.py` to `LivePredictor`; a net learns its own
     features from the `(win, 8)` block and must not be mixed into that path.
     The net's input contract is `T.nn_input_block()` instead -- resample,
     then calibrate, and nothing else -- and `nn_serve.py` calls the same
     function.
   * **Fold-local scaling.** `log1p` + per-channel standardisation fitted on
     the training fold ONLY. Fitting it over the whole dataset would leak the
     test distribution into training. The fitted numbers travel in the
     checkpoint, because at inference the scaler IS part of the model.
   * **No early stopping.** Stopping when the held-out fold looks best is
     fitting the test set. Epochs are fixed and declared.
   * **Fail loudly on the device.** `--device cuda` is the default and a
     missing CUDA raises rather than falling back: a silent CPU run looks
     exactly like a working one except that it takes twenty times as long.

 Run:

     python training/train_nn.py                       # full grid, CUDA
     python training/train_nn.py --quick               # the default arch, one window
     python training/train_nn.py --device cpu
     python training/train_nn.py --kind tcn --kind gru --window 0.5
     python training/train_nn.py --calibration off     # undo section 8k exactly
"""
from __future__ import annotations

import argparse
import json
import math
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
import nn_arch
#  The protocol pieces live in ONE place. Importing them means this trainer
#  cannot dedup, group or filter differently from the sklearn sweep.
import train_model as TM

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASET_ROOT = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
OUT_DIR = os.path.join(T.PROJECT_DIR, "models")
REPORT_DIR = os.path.join(T.PROJECT_DIR, "models", "reports")
CHECKPOINT_NAME = os.path.basename(T.NN_MODEL_PATH)

#  Same grid as the sklearn sweep, for the same reason (see train_model.py).
MODEL_FS = TM.MODEL_FS

#  A net needs a few more samples than a statistics vector, so this starts at
#  0.30 s rather than the sweep's 0.20 s. 0.75 overlap: more windows out of a
#  small dataset, and dedup removes the duplicate coverage that creates.
WINDOW_SECONDS = [0.30, 0.40, 0.50]
OVERLAP = 0.75

#  `tcn` first: it is the default and `--quick` takes the head of this list.
KINDS = list(nn_arch.KINDS)

EXCLUDE_SETS = TM.EXCLUDE_SETS
MIN_LABEL_WINDOW_SAMPLES = TM.MIN_LABEL_WINDOW_SAMPLES
CV_SPLITS = TM.CV_SPLITS
RANDOM_STATE = TM.RANDOM_STATE
DEDUPE_WINDOWS = TM.DEDUPE_WINDOWS
SELECTION_GROUPING = TM.SELECTION_GROUPING

#  Training hyperparameters. Fixed and declared rather than tuned on the test
#  fold; `--epochs` overrides for a quick look.
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.05
HIDDEN = nn_arch.HIDDEN
DROPOUT = nn_arch.DROPOUT


# ==============================================================================
# PROGRESS  (same two-line protocol the GUI already parses)
# ==============================================================================


def emit_progress(done, total, phase="nn sweep", device="cpu", **detail):
    """
    `PROGRESS` + `TRAINEVENT`, exactly as `train_model.emit_progress` does it,
    so the GUI's TRAINING panel needs no second parser. The one difference is
    that `device` is real here instead of hard-coded to cpu: this trainer can
    genuinely use the GPU.
    """
    total = max(1, int(total))
    done = min(int(done), total)
    print(f"PROGRESS {done} {total} {phase}", flush=True)
    event = {"done": done, "total": total, "phase": phase, "device": device,
             "trainer": "nn"}
    event.update({k: v for k, v in detail.items() if v is not None})
    try:
        print("TRAINEVENT " + json.dumps(event, default=str), flush=True)
    except (TypeError, ValueError):
        pass          # a progress line must never be able to kill a sweep


# ==============================================================================
# DATA  --  recordings to (n, win, 8) sequences
# ==============================================================================


def build_sequences(recs, window_s, overlap, dedupe=DEDUPE_WINDOWS,
                    calibration=None, report=None):
    """
    Recordings -> `(X, y, groups_uid, groups_person, labels, meta)`.

    `X` is `(n_windows, win, 8)` float32 on the CALIBRATED scale but
    otherwise in device counts -- unscaled. Standardisation is fold-local
    (see `fold_scaler`) so no test-fold statistic can reach training.

    Two passes, for the same reason `train_model.build_dataset` has two: the
    dedup mask has to be known before any per-window work happens.

    `calibration` is a TABLE, person id -> resolved calibration. Each
    recording is scaled with the numbers of the wearer who produced it, and a
    wearer without one is left uncalibrated rather than borrowing someone
    else's gain. It is applied to the resampled BLOCK rather than to each
    window, which is identical arithmetic -- the map is affine and per
    channel -- for a fraction of the work.
    """
    win = max(4, int(round(window_s * MODEL_FS)))
    step = max(1, int(round(win * (1.0 - overlap))))
    labels = sorted({r.label for r in recs})
    index = {name: i for i, name in enumerate(labels)}

    staged = []
    for rec in recs:
        a, b = rec.label_start_index, rec.label_end_index
        if b - a < MIN_LABEL_WINDOW_SAMPLES:
            continue
        seconds = max(1e-6, float(rec.t_rel[b - 1] - rec.t_rel[a]))
        n_out = max(win, int(round(seconds * MODEL_FS)))
        block = T.resample_uniform(rec.emg[a:b], rec.t_rel[a:b], n_out)
        #  The net's whole input contract, in one call, shared with nn_serve.
        block = T.nn_input_block(block, (calibration or {}).get(rec.person_id))
        windows = T.sliding_windows(block, win, step)
        if windows.shape[0] == 0:
            continue
        staged.append((rec, windows, win, step, n_out))

    if not staged:
        return None

    if dedupe:
        masked = TM.dedupe_windows(staged, keep_report=report)
        #  The same floor the sklearn sweep applies: dedup must never leave a
        #  class too thin to cross-validate. Keeping the duplicates and
        #  saying so beats a run that cannot be evaluated at all.
        before, after = Counter(), Counter()
        for rec, windows, mask in masked:
            before[rec.label] += int(windows.shape[0])
            after[rec.label] += int(mask.sum())
        starved = sorted(lab for lab in before
                         if after[lab] < CV_SPLITS <= before[lab])
        if starved:
            masked = [(rec, w, np.ones(w.shape[0], dtype=bool))
                      for rec, w, _ in masked]
            if report is not None:
                report["dedup_fallback"] = starved
                report["windows_after"] = report.get("windows_before")
    else:
        masked = [(rec, w, np.ones(w.shape[0], dtype=bool))
                  for rec, w, _, _, _ in staged]
        if report is not None:
            total = int(sum(w.shape[0] for _, w, _, _, _ in staged))
            report["windows_before"] = report["windows_after"] = total

    blocks, ys, uids, people = [], [], [], []
    for rec, windows, mask in masked:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        #  `sliding_windows` hands back a read-only strided VIEW.
        #  `torch.from_numpy` needs owned, writeable, contiguous memory, so
        #  the copy happens once, here, for the windows that survived.
        blocks.append(np.ascontiguousarray(windows[idx], dtype=np.float32))
        ys.extend([index[rec.label]] * idx.size)
        uids.extend([rec.uid] * idx.size)
        people.extend([rec.person_id] * idx.size)

    if not blocks:
        return None

    X = np.concatenate(blocks, axis=0)
    meta = {"window_samples": win, "step": step, "window_s": float(window_s),
            "overlap": float(overlap), "deduped": bool(dedupe),
            "calibrated_subjects": sorted(calibration or {})}
    return (X,
            np.asarray(ys, dtype=np.int64),
            np.asarray(uids, dtype=object),
            np.asarray(people, dtype=object),
            labels, meta)


def fold_scaler(X_train):
    """
    `log1p` + per-channel standardisation, fitted on the TRAINING fold only.

    `log1p` first because the filtered stream spans a baseline around 15-40
    to a max-effort 1444 (CLAUDE.md section 2): on a linear scale the strong
    contractions own the gradient. The absolute level survives the transform,
    which is the point -- it is the part that identifies a gesture on this
    rectified stream, and removing it has measurably hurt twice.

    Returns `(apply, mean, std)`. The two arrays go into the checkpoint:
    at inference the scaler is not preprocessing, it is part of the model.

    The two arrays are PER-CHANNEL VECTORS of shape (8,), not (1, 1, 8), and
    that is load-bearing rather than tidiness. A (1, 1, 8) mean broadcasts
    correctly against the trainer's batched `(n, win, 8)` but turns a single
    live `(win, 8)` window into `(1, win, 8)` -- which then reaches conv1d as
    a 4-D tensor and raises, at inference only, in the one code path no
    training run exercises. A (8,) vector broadcasts correctly against both.
    `nn_serve.py` flattens whatever it finds for the same reason.
    """
    logged = np.log1p(np.abs(X_train))
    mean = logged.mean(axis=(0, 1))
    std = logged.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)

    def apply(X):
        return ((np.log1p(np.abs(X)) - mean) / std).astype(np.float32)

    return apply, mean.astype(np.float32), std.astype(np.float32)


def dc_only_sequences(X):
    """
    Every window replaced by its own per-channel MEAN, held constant in time.

    The neural equivalent of `train_model.dc_only_control`: the net keeps its
    full architecture and its full input shape, and loses exactly one thing --
    all temporal structure. What is left is the 8 DC levels.

    CLAUDE.md section 8f made this control a rule, and section 8k added the
    reading of it: a HIGH cross-subject DC-only score means the wearers differ
    in gain, not that gestures were recognised. For a net that claims to read
    how the signal changes in time, this is the control that says whether it
    is doing that or just reading amplitude.
    """
    return np.repeat(X.mean(axis=1, keepdims=True), X.shape[1],
                     axis=1).astype(np.float32)


# ==============================================================================
# DEVICE
# ==============================================================================


def resolve_device(requested):
    """
    Pick the torch device and FAIL LOUDLY rather than fall back silently.

    A silent CPU fallback is the worst outcome: the sweep runs, the operator
    believes the GPU was used, and the only symptom is the clock.
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

    props = torch.cuda.get_device_properties(0)
    info = {"device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
            "capability": f"{props.major}.{props.minor}",
            "total_memory_mb": int(props.total_memory / 1024 / 1024),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}
    #  cudnn autotuning picks different kernels per run, which changes the
    #  float accumulation order and therefore the result.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0"), info


def compute_banner(info):
    """A box saying GPU or CPU, loudly, at the start and again at the end."""
    if info.get("device", "cpu") != "cpu":
        lines = [f"COMPUTE:  >>> GPU <<<   {info.get('gpu', 'CUDA device')}",
                 f"          CUDA {info.get('cuda_runtime')}  |  "
                 f"torch {info.get('torch')}  |  "
                 f"{info.get('total_memory_mb')} MB  |  "
                 f"capability {info.get('capability')}"]
    else:
        lines = ["COMPUTE:  >>> CPU <<<   no GPU in use",
                 f"          torch {info.get('torch')}  |  "
                 f"{os.cpu_count()} logical cores  |  expect this to be slow"]
    field = max(max(len(line) for line in lines), 70)
    rule = "+" + "-" * (field + 2) + "+"
    body = "\n".join(f"| {line:<{field}} |" for line in lines)
    return f"\n{rule}\n{body}\n{rule}"


# ==============================================================================
# TRAIN / PREDICT
# ==============================================================================


def train_one(model, X_train, y_train, device, epochs, n_classes,
              class_weights=None, seed=RANDOM_STATE, log=None,
              augmenter=None, scale_stats=None):
    """
    Fit on the training fold. No validation split, no early stopping.

    `augmenter` (training/nn_augment.Augmenter) switches to the RAW path:
    `X_train` is then the calibrated window in counts, each batch is
    augmented and only then scaled with `scale_stats = (mean, std)` -- either
    per-channel vectors `(C,)` or one row per window `(n, C)` (per-person
    adaptation). With `augmenter=None` this function is exactly what it was:
    `X_train` is already scaled and nothing is touched.

    Early stopping on the held-out fold is fitting the test set, so the epoch
    count is fixed and printed instead. AdamW + cosine decay + a little label
    smoothing is the whole recipe: with a few thousand parameters and a
    thousand windows there is nothing here worth tuning on two subjects.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    model = model.to(device)
    model.train()

    Xt = torch.from_numpy(np.ascontiguousarray(X_train)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y_train)).to(device)

    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32,
                                 device=device)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=LABEL_SMOOTHING)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    n = Xt.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if augmenter is not None:
        if scale_stats is None:
            raise ValueError("augmentation works on raw windows and needs "
                             "scale_stats=(mean, std) to scale them after")
        mean_t = torch.as_tensor(np.asarray(scale_stats[0]),
                                 dtype=torch.float32, device=device)
        std_t = torch.as_tensor(np.asarray(scale_stats[1]),
                                dtype=torch.float32, device=device)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            #  BatchNorm cannot normalise a batch of one, and a trailing
            #  remainder of exactly one sample is common at these sizes.
            if idx.numel() < 2:
                continue
            optimiser.zero_grad(set_to_none=True)
            if augmenter is None:
                xb = Xt[idx]
            else:
                xb = augmenter(Xt[idx])
                m = mean_t[idx][:, None, :] if mean_t.ndim == 2 else mean_t
                sd = std_t[idx][:, None, :] if std_t.ndim == 2 else std_t
                xb = (torch.log1p(xb.abs()) - m) / sd
            loss = criterion(model(xb), yt[idx])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * idx.numel()
        scheduler.step()
        if log is not None and (epoch + 1) % max(1, epochs // 4) == 0:
            log.append((epoch + 1, total / max(1, n)))
    return model


def predict_all(model, X, device, batch=1024):
    """Class predictions and probabilities for X."""
    import torch
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            chunk = torch.from_numpy(
                np.ascontiguousarray(X[start:start + batch])).to(device)
            p = torch.softmax(model(chunk), dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(probs)


# ==============================================================================
# EVALUATION  (the sklearn sweep's protocol, on sequences)
# ==============================================================================


def evaluate(kind, X, y, groups, labels, win, device, epochs,
             split_groups=None, grouping="recording", n_splits=CV_SPLITS,
             hidden=HIDDEN, dropout=None, on_fold=None, verbose=False):
    dropout = DROPOUT if dropout is None else dropout
    """
    Grouped stratified cross-validation, returning the same dict shape
    `train_model.evaluate` returns so one reader serves both trainers.

    `groups` is always the RECORDING id, because that is what the
    per-recording vote aggregates on. `split_groups` is what the FOLDS are
    split on and defaults to `groups`; pass the per-window subject id for a
    subject-independent score. Keeping the two separate is what lets a fold
    hold out a whole person while the vote is still counted per recording --
    passing one for the other would either leak a subject or destroy the
    vote.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    if split_groups is None:
        split_groups = groups
    counts = Counter(y.tolist())
    n_groups = len(set(split_groups.tolist()))
    n_splits = int(min(n_splits, min(counts.values()), n_groups))
    if n_splits < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    folds = list(cv.split(X, y, split_groups))

    window_acc, window_f1, fold_acc = [], [], []
    vote_correct = vote_total = 0
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    n_params = 0

    for number, (train_idx, test_idx) in enumerate(folds):
        scale, _, _ = fold_scaler(X[train_idx])
        bins = np.bincount(y[train_idx], minlength=len(labels)).astype(float)
        weights = np.where(bins > 0,
                           bins.sum() / (len(labels) * np.maximum(bins, 1)),
                           1.0)
        model = nn_arch.build(kind, X.shape[2], len(labels), win,
                              hidden=hidden, dropout=dropout,
                              seed=RANDOM_STATE)
        n_params = nn_arch.count_parameters(model)
        log = [] if verbose else None
        model = train_one(model, scale(X[train_idx]), y[train_idx], device,
                          epochs, len(labels), class_weights=weights, log=log)
        if verbose and log:
            print("      fold %d  loss  %s" % (
                number + 1, "  ".join(f"ep{e}:{l:.3f}" for e, l in log)))

        pred, _ = predict_all(model, scale(X[test_idx]), device)
        acc = accuracy_score(y[test_idx], pred)
        window_acc.append(acc)
        fold_acc.append(acc)
        window_f1.append(f1_score(y[test_idx], pred, average="macro",
                                  zero_division=0))
        cm += confusion_matrix(y[test_idx], pred,
                               labels=list(range(len(labels))))

        per_rec, truth = defaultdict(list), {}
        for pos, gid in enumerate(groups[test_idx]):
            per_rec[gid].append(pred[pos])
            truth[gid] = y[test_idx][pos]
        for gid, votes in per_rec.items():
            vote_total += 1
            vote_correct += int(Counter(votes).most_common(1)[0][0]
                                == truth[gid])
        if on_fold is not None:
            on_fold(number + 1, len(folds))

    return {"n_splits": n_splits, "grouping": grouping,
            "window_accuracy": float(np.mean(window_acc)),
            "window_accuracy_std": float(np.std(window_acc)),
            "window_macro_f1": float(np.mean(window_f1)),
            "recording_vote_accuracy": (vote_correct / vote_total
                                        if vote_total else 0.0),
            "n_recordings_voted": vote_total,
            "fold_accuracies": [float(a) for a in fold_acc],
            "confusion_matrix": cm.tolist(),
            "n_parameters": n_params}


def leave_one_subject_out(kind, X, y, groups_person, labels, win, device,
                          epochs, hidden=HIDDEN, dropout=None):
    """Per-wearer held-out score. None with fewer than two subjects."""
    dropout = DROPOUT if dropout is None else dropout
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
        scale, _, _ = fold_scaler(X[train])
        model = nn_arch.build(kind, X.shape[2], len(labels), win,
                              hidden=hidden, dropout=dropout,
                              seed=RANDOM_STATE)
        model = train_one(model, scale(X[train]), y[train], device, epochs,
                          len(labels))
        pred, _ = predict_all(model, scale(X[test]), device)
        scores[person] = float(accuracy_score(y[test], pred))
    if not scores:
        return None
    return {"per_subject": scores,
            "mean": float(np.mean(list(scores.values()))),
            "std": float(np.std(list(scores.values())))}


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
# MAIN
# ==============================================================================


def main(argv=None):
    global OVERLAP, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE
    global LABEL_SMOOTHING
    T.force_utf8_console()

    parser = argparse.ArgumentParser(
        description="Train the separate CUDA neural-net gesture model")
    parser.add_argument("--root", default=DATASET_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--model-path", default=None,
                        help="explicit checkpoint path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true",
                        help="the default architecture, one window length")
    parser.add_argument("--kind", action="append", default=None,
                        choices=KINDS, help="restrict the architecture grid")
    parser.add_argument("--window", action="append", type=float, default=None,
                        help="restrict the window grid, in seconds")
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--label-smoothing", type=float,
                        default=LABEL_SMOOTHING)
    parser.add_argument("--person", action="append", default=None)
    parser.add_argument("--exclude", default="no_errors",
                        choices=sorted(EXCLUDE_SETS))
    parser.add_argument("--calibration", default=None,
                        choices=sorted(T.CALIBRATION_MODES),
                        help="per-wearer calibration mode (default: the app's "
                             f"CALIBRATION_MODE = {T.CALIBRATION_MODE!r})")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep duplicate-coverage windows (leaks; for "
                             "measuring what dedup is worth)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the DC-only control")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-fold training loss")
    args = parser.parse_args(argv)

    #  The GUI's NEURAL SETUP dialog arrives as flags; the module values the
    #  training code reads are set once here.
    OVERLAP = min(0.9, max(0.0, args.overlap))
    DROPOUT = min(0.8, max(0.0, args.dropout))
    LEARNING_RATE = max(1e-6, args.lr)
    WEIGHT_DECAY = max(0.0, args.weight_decay)
    BATCH_SIZE = max(8, args.batch_size)
    LABEL_SMOOTHING = min(0.5, max(0.0, args.label_smoothing))

    print("=" * 78)
    print(" NEURAL NETWORK TRAINER  --  separate model, separate numbers")
    print("=" * 78)

    device, dev_info = resolve_device(args.device)
    device_tag = "cpu" if dev_info.get("device") == "cpu" else "cuda"
    print(compute_banner(dev_info))
    print("\nCompute")
    for key, value in dev_info.items():
        print(f"  {key:<14}: {value}")
    print(f"  {'host':<14}: {platform.platform()}")

    # ---- data -------------------------------------------------------------
    recs, dropped = TM.load_recordings(
        args.root, persons=args.person,
        exclude_codes=EXCLUDE_SETS[args.exclude])
    if not recs:
        raise SystemExit(f"No usable recordings under {args.root!r}.")

    classes = sorted({r.label for r in recs})
    people = sorted({r.person_id for r in recs})
    if len(classes) < 2:
        print(f"\nOnly one class ({classes[0]}) -- a classifier needs at "
              f"least two. Record another gesture first.")
        return T.TRAIN_EXIT_NEED_MORE_DATA

    chance = max(Counter(r.label for r in recs).values()) / len(recs)
    print("\nDataset")
    print(f"  root          : {os.path.abspath(args.root)}")
    print(f"  recordings    : {len(recs)} used, {len(dropped)} dropped "
          f"(filter: {args.exclude})")
    print(f"  classes       : {len(classes)}  {', '.join(classes)}")
    for name in classes:
        print(f"      {name:<12} {sum(1 for r in recs if r.label == name)}")
    print(f"  subjects      : {', '.join(people)}")
    print(f"  majority-class baseline: {chance * 100:.1f} %  "
          f"(any model must beat this, not 1/n)")

    # ---- calibration ------------------------------------------------------
    requested_mode = (args.calibration if args.calibration is not None
                      else T.CALIBRATION_MODE) or "off"
    #  Same calibrations as train_model.py: each wearer derived from their own
    #  recordings (T.training_calibrations); in memory only.
    cal_table, cal_filled, _cal_used = T.training_calibrations(
        args.root, requested_mode, recs, people)
    for _p, _how in cal_filled.items():
        print(f"  {_p}: {_how}")
    applied_mode = requested_mode if cal_table else "off"
    print("\nCalibration")
    print(f"  requested     : {requested_mode}")
    if requested_mode == "off":
        print("  applied       : off -- features are plain device counts")
    elif not cal_table:
        print("  applied       : off -- NO wearer has a usable calibration.")
        print("                  The checkpoint will record 'off', because "
              "that is the")
        print("                  scale the net was really fitted on.")
    else:
        print(f"  applied       : {applied_mode} for "
              f"{', '.join(sorted(cal_table))}")
        missing = [p for p in people if p not in cal_table]
        if missing:
            print(f"  *** MIXED SCALES: {', '.join(missing)} ha"
                  f"{'s' if len(missing) == 1 else 've'} no usable "
                  f"calibration and will be left in device counts.")
            print("      A feature space where some wearers are calibrated "
                  "and some are not")
            print("      is not one scale. Calibrate everyone, or run with "
                  "--calibration off.")
        print("  NOTE: a calibrated subject-grouped score is NOT zero-shot. "
              "The held-out")
        print("        wearer's own unlabelled signal set their gain, so it "
              "answers 'how")
        print("        well does this work on a new wearer WHO CALIBRATES "
              "FIRST'.")

    # ---- grouping ---------------------------------------------------------
    viable, why = TM.subject_grouping_viable(recs)
    grouping, note = TM.resolve_grouping(len(people) if viable else 1,
                                         SELECTION_GROUPING)
    contrast = "recording" if grouping == "subject" else "subject"
    kinds = args.kind or KINDS
    windows = args.window or WINDOW_SECONDS
    if args.quick:
        kinds, windows = kinds[:1], windows[-1:]

    print("\nProtocol")
    print(f"  resampled to  : {MODEL_FS:.0f} Hz before windowing")
    print(f"  folds         : StratifiedGroupKFold, {note}")
    if not viable and len(people) > 1:
        print(f"                  (subject grouping unavailable: {why})")
    print(f"  dedup         : {'ON' if not args.no_dedupe else 'OFF'} "
          "-- window-level, by source timeline (CLAUDE.md 8g)")
    print(f"  epochs        : {args.epochs} fixed, no early stopping on the "
          "test fold")
    print("  scaling       : log1p + standardise, fitted per training fold")
    print(f"  grid          : {len(kinds)} architecture(s) x {len(windows)} "
          f"window(s) = {len(kinds) * len(windows)} candidates")
    for kind in kinds:
        print(f"      {kind:<9} {nn_arch.DESCRIPTIONS.get(kind, '')}")
    if len(kinds) * len(windows) > 1:
        print("  NOTE: picking the best of several candidates on the same "
              "folds makes the")
        print("        winner's score optimistic. Every row is printed below, "
              "and the")
        print("        leave-one-subject-out figure did not pick its own "
              "folds.")
    print("\n  THIS MODEL DOES NOT VOTE. It is saved to its own file, it is "
          "not a member")
    print("  of the decision panel's soft vote, and the app's LIVE "
          "CLASSIFIER answer is")
    print("  unchanged by it. Separate numbers, separate button, separate "
          "opinion.")

    # ---- sweep ------------------------------------------------------------
    results = []
    total_steps = len(kinds) * len(windows) + 2        # + control + LOSO
    done = 0
    t_start = time.perf_counter()
    emit_progress(0, total_steps, "nn sweep", device_tag,
                  message="loading windows")

    cache = {}
    for window_s in windows:
        report = {}
        built = build_sequences(recs, window_s, OVERLAP,
                               dedupe=not args.no_dedupe,
                               calibration=cal_table, report=report)
        if built is None:
            print(f"\n  window {window_s:.2f} s -> no usable windows, skipped")
            continue
        X, y, groups, groups_person, labels, meta = built
        cache[window_s] = built
        win = X.shape[1]
        print(f"\n--- window {window_s:.2f} s  ({win} samples @ "
              f"{MODEL_FS:.0f} Hz)   {X.shape[0]} windows from "
              f"{len(set(groups.tolist()))} recordings"
              + (f"   (dedup {report.get('windows_before')} -> "
                 f"{report.get('windows_after')})"
                 if report.get("windows_before") else ""))
        split_groups = groups_person if grouping == "subject" else groups

        for kind in kinds:
            done += 1
            t0 = time.perf_counter()
            emit_progress(done - 1, total_steps, "nn sweep", device_tag,
                          model=kind, window_s=window_s, overlap=OVERLAP,
                          representation="sequence",
                          feature_set=f"{win}x{X.shape[2]} samples",
                          n_windows=int(X.shape[0]), n_folds=CV_SPLITS)
            stats = evaluate(kind, X, y, groups, labels, win, device,
                             args.epochs, split_groups=split_groups,
                             grouping=grouping, hidden=args.hidden,
                             verbose=args.verbose)
            elapsed = time.perf_counter() - t0
            if stats is None:
                print(f"  [{done}/{len(kinds) * len(windows)}] {kind:<9} not "
                      "enough groups to cross-validate")
                continue
            stats.update({"kind": kind, "window_s": window_s,
                          "win_samples": win, "overlap": OVERLAP,
                          "n_windows": int(X.shape[0]), "seconds": elapsed,
                          "hidden": args.hidden})
            results.append(stats)
            print(f"  [{done}/{len(kinds) * len(windows)}] {kind:<9} "
                  f"window {stats['window_accuracy'] * 100:5.1f} % "
                  f"+-{stats['window_accuracy_std'] * 100:4.1f}   "
                  f"vote {stats['recording_vote_accuracy'] * 100:5.1f} %   "
                  f"f1 {stats['window_macro_f1'] * 100:5.1f} %   "
                  f"{stats['n_parameters']:>6d} par   {elapsed:5.1f} s")

    if not results:
        raise SystemExit("Nothing could be evaluated.")

    # ---- ranking ----------------------------------------------------------
    results.sort(key=lambda r: (-r["window_accuracy"],
                                -r["recording_vote_accuracy"]))
    print("\n" + "=" * 78)
    print(f" ALL CANDIDATES  (ranked by per-window accuracy, {grouping}"
          "-grouped)")
    print("=" * 78)
    print(f"  {'arch':<9} {'win':>6} {'window acc':>12} {'vote acc':>10} "
          f"{'macro f1':>9} {'params':>8}")
    for r in results:
        print(f"  {r['kind']:<9} {r['window_s']:>5.2f}s "
              f"{r['window_accuracy'] * 100:>10.1f} % "
              f"{r['recording_vote_accuracy'] * 100:>8.1f} % "
              f"{r['window_macro_f1'] * 100:>7.1f} % "
              f"{r['n_parameters']:>8d}")

    best = results[0]
    X, y, groups, groups_person, labels, meta = cache[best["window_s"]]
    win = X.shape[1]
    split_groups = groups_person if grouping == "subject" else groups

    print("\n" + "=" * 78)
    print(f" BEST: {best['kind']}  at {best['window_s']:.2f} s "
          f"({best['win_samples']} samples, {best['n_parameters']} parameters)")
    print("=" * 78)
    print(f"  {nn_arch.DESCRIPTIONS.get(best['kind'], '')}")
    print(f"  grouping               : {grouping}  ({note})")
    print(f"  per-window accuracy    : {best['window_accuracy'] * 100:.1f} % "
          f"+- {best['window_accuracy_std'] * 100:.1f}")
    print(f"  per-recording vote     : "
          f"{best['recording_vote_accuracy'] * 100:.1f} % over "
          f"{best['n_recordings_voted']} recordings")
    print(f"  macro F1               : {best['window_macro_f1'] * 100:.1f} %")
    print(f"  folds                  : {best['n_splits']}  "
          f"({', '.join(f'{a * 100:.0f}%' for a in best['fold_accuracies'])})")
    print(f"  majority-class baseline: {chance * 100:.1f} %")
    print()
    print_confusion(np.array(best["confusion_matrix"]), labels)

    # ---- the other protocol, as a contrast --------------------------------
    print("\n  The OTHER protocol, for contrast (not the selection key):")
    other_key = groups if grouping == "subject" else groups_person
    other = evaluate(best["kind"], X, y, groups, labels, win, device,
                     args.epochs, split_groups=other_key, grouping=contrast,
                     hidden=args.hidden)
    if other is None:
        print(f"      {contrast}-grouped: not measurable on this dataset")
    else:
        gap = (best["window_accuracy"] - other["window_accuracy"]) * 100
        print(f"      {contrast}-grouped: "
              f"{other['window_accuracy'] * 100:.1f} % per window, "
              f"{other['recording_vote_accuracy'] * 100:.1f} % per recording")
        print(f"      gap to the chosen protocol: {gap:+.1f} points"
              + ("   <-- that gap IS the subject-specificity"
                 if abs(gap) > 5 and grouping == "subject" else ""))

    # ---- the DC-only control ----------------------------------------------
    done += 1
    control = None
    if not args.no_control:
        emit_progress(done - 1, total_steps, "DC-only control", device_tag,
                      model=best["kind"], message="temporal structure removed")
        print("\n  DC-only control: the same net, every window replaced by "
              "its channel means")
        control = evaluate(best["kind"], dc_only_sequences(X), y, groups,
                           labels, win, device, args.epochs,
                           split_groups=split_groups, grouping=grouping,
                           hidden=args.hidden)
        if control is None:
            print("      not measurable")
        else:
            gain = (best["window_accuracy"] - control["window_accuracy"]) * 100
            print(f"      DC only  : {control['window_accuracy'] * 100:.1f} % "
                  f"per window ({grouping}-grouped)")
            print(f"      full      : {best['window_accuracy'] * 100:.1f} %")
            print(f"      so the TEMPORAL STRUCTURE is worth {gain:+.1f} "
                  "points")
            if gain < 5:
                print("      *** WARNING: the net is reading amplitude, not "
                      "time. With the")
                print("          8 channel means alone it scores nearly the "
                      "same, so nothing")
                print("          here is evidence that a sequence model was "
                      "needed.")
            if applied_mode != "off" and control["window_accuracy"] > 0.6:
                print("      Reminder (CLAUDE.md 8k): a RISING DC-only score "
                      "is not good news --")
                print("      it means the wearers' levels transfer, not that "
                      "gestures were learnt.")

    # ---- subject-independent ----------------------------------------------
    done += 1
    emit_progress(done - 1, total_steps, "leave-one-subject-out", device_tag,
                  model=best["kind"])
    loso = leave_one_subject_out(best["kind"], X, y, groups_person, labels,
                                 win, device, args.epochs, hidden=args.hidden)
    print()
    if loso:
        print(f"  subject-independent    : {loso['mean'] * 100:.1f} % "
              f"+- {loso['std'] * 100:.1f}")
        for person, score in sorted(loso["per_subject"].items()):
            print(f"      held out {person:<12} {score * 100:5.1f} %")
        if applied_mode != "off":
            print("      (each held-out wearer calibrated with their OWN "
                  "numbers -- see the")
            print("       calibration note above: this is not zero-shot.)")
    else:
        print("  subject-independent    : NOT MEASURABLE -- one subject only.")
        print("                           This net is specific to the person "
              "who recorded it.")

    if args.no_save:
        print("\n--no-save: nothing written.")
        print(compute_banner(dev_info))
        print(f"\nTotal time: {time.perf_counter() - t_start:.1f} s")
        emit_progress(total_steps, total_steps, "done", device_tag,
                      status="completed", message="--no-save, nothing written")
        return 0

    # ---- final fit + save -------------------------------------------------
    import torch

    scale, mean, std = fold_scaler(X)
    bins = np.bincount(y, minlength=len(labels)).astype(float)
    weights = np.where(bins > 0,
                       bins.sum() / (len(labels) * np.maximum(bins, 1)), 1.0)
    model = nn_arch.build(best["kind"], X.shape[2], len(labels), win,
                          hidden=args.hidden, dropout=DROPOUT,
                          seed=RANDOM_STATE)
    model = train_one(model, scale(X), y, device, args.epochs, len(labels),
                      class_weights=weights)
    fit_pred, _ = predict_all(model, scale(X), device)
    fit_accuracy = float((fit_pred == y).mean())

    out_dir = args.out_dir
    #  Derived from --out-dir, not from the module default: a caller that
    #  redirects the output must get ALL of it, report included. The same
    #  mistake in train_model.py had the GUI overwriting a different model
    #  than the one it was using (CLAUDE.md section 13).
    report_dir = os.path.join(out_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    path = args.model_path or os.path.join(out_dir, CHECKPOINT_NAME)

    payload = {
        "format": "emg_nn_v1",
        "kind": best["kind"],
        "architecture": nn_arch.DESCRIPTIONS.get(best["kind"], ""),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": labels,
        "n_channels": int(X.shape[2]),
        "win_samples": int(win),
        "window_s": float(best["window_s"]),
        "overlap": OVERLAP,
        "model_fs": float(MODEL_FS),
        "hidden": int(args.hidden),
        "dropout": float(DROPOUT),
        "emg_mode": T.EMG_MODE,
        #  The scaler IS part of the model: inference must reproduce it
        #  exactly or the inputs are meaningless.
        "scaler": {"kind": "log1p_standardise", "mean": mean, "std": std},
        #  What was APPLIED, not what was asked for. Stamping "affine" on
        #  plain device counts would produce a model the app refuses to load
        #  -- a trainer locking its own output out of its own application.
        "calibration": {"mode": applied_mode, "requested": requested_mode,
                        "subjects": sorted(cal_table),
                        "target": float(T.CALIBRATION_TARGET_LEVEL)},
        #  Same provenance block train_model.py writes, so either payload can
        #  be asked the same question: trained on what, and is that still
        #  what is on disk?
        "data": T.dataset_fingerprint(args.root),
        "stats": {
            "model_name": f"NN-{best['kind']}",
            "trainer": "train_nn.py",
            "validation_protocol":
                f"StratifiedGroupKFold grouped by "
                f"{'SUBJECT' if grouping == 'subject' else 'recording uid'}"
                f" -- {note}",
            "grouping": grouping,
            "grouping_note": note,
            "grouping_contrast": other,
            "subject_independent": loso,
            "dc_only_control": control,
            "majority_class_baseline": float(chance),
            "train_fit_accuracy": fit_accuracy,
            "n_recordings": len(recs),
            "n_recordings_total": len(recs) + len(dropped),
            "n_sessions": len({f"{r.person_id}/{r.session_id}"
                               for r in recs}),
            "excluded": args.exclude,
            "subjects": people,
            "labels": labels,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "early_stopping": False,
            "deduped": not args.no_dedupe,
            "candidates_evaluated": len(results),
            "candidates": [
                {k: v for k, v in r.items() if k != "confusion_matrix"}
                for r in results],
            "architecture_note": nn_arch.DESCRIPTIONS.get(best["kind"], ""),
            "selection_bias_note":
                f"The winner was chosen as the best of {len(results)} "
                "candidates scored on the same folds, so its number is "
                "optimistic. Leave-one-subject-out is the figure that did "
                "not pick its own folds.",
            "separation_note":
                "This net is NOT part of the deployed answer. It is saved to "
                "its own file, it is not a member of the decision panel's "
                "soft vote, and the app's LIVE CLASSIFIER prediction is "
                "produced entirely by the scikit-learn model.",
            "created_iso": T.iso_time(time.time()),
            **{k: best[k] for k in (
                "window_accuracy", "window_accuracy_std", "window_macro_f1",
                "recording_vote_accuracy", "n_recordings_voted", "n_splits",
                "fold_accuracies", "confusion_matrix", "n_windows",
                "n_parameters", "window_s", "overlap", "win_samples",
                "kind")},
        },
        "device_trained_on": dev_info,
        "created": datetime.now().isoformat(timespec="seconds"),
        "created_iso": T.iso_time(time.time()),
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #  Archive first, then the working copy, the same way train_model.py does
    #  it: a sweep should not be destroyed by the next one, and comparing two
    #  checkpoints is how a regression gets found.
    stem = os.path.splitext(os.path.basename(path))[0]
    archive = T.unique_path(os.path.join(
        os.path.dirname(os.path.abspath(path)), f"{stem}_{stamp}.pt"))
    torch.save(payload, archive)
    torch.save(payload, path)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"\nSaved checkpoint -> {os.path.abspath(path)}  ({size_kb:.0f} KB)")
    print(f"Archived copy    -> {archive}")

    report_payload = {
        "trainer": "train_nn.py", "device": dev_info, "chance": chance,
        "classes": labels, "grouping": grouping, "grouping_note": note,
        "calibration": payload["calibration"], "results": results,
        "grouping_contrast": other, "dc_only_control": control,
        "subject_independent": loso, "epochs": args.epochs,
        "deduped": not args.no_dedupe,
    }
    report_file = os.path.join(report_dir, f"nn_sweep_{stamp}.json")
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2, default=str)
    print(f"Saved sweep report -> {os.path.abspath(report_file)}")

    #  A plain-JSON sidecar beside the checkpoint, holding everything except
    #  the weights. The application must be able to answer "is a net trained,
    #  on what, and how well does it do" WITHOUT importing torch and without
    #  starting the inference worker -- a `.pt` is a torch pickle and nothing
    #  else can read it. So NN STATS reads this, and the live worker reports
    #  the same numbers back when it is running; they come from one write.
    sidecar = T.nn_stats_path(path)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in payload.items()
                   if k not in ("state_dict", "scaler")},
                  fh, indent=2, default=str)
    print(f"Saved stats sidecar -> {os.path.abspath(sidecar)}")

    print(compute_banner(dev_info))
    print(f"\nTotal time: {time.perf_counter() - t_start:.1f} s")
    print("\nREMINDER: the app's LIVE CLASSIFIER still loads "
          f"{T.MODEL_PATH!r} and is")
    print("          unaffected by this file. The net has its own START "
          "button and its")
    print("          own statistics; the two are never averaged together.")

    emit_progress(total_steps, total_steps, "done", device_tag,
                  status="completed", model=f"NN-{best['kind']}",
                  artifact=os.path.abspath(path),
                  window_accuracy=best["window_accuracy"],
                  recording_vote_accuracy=best["recording_vote_accuracy"],
                  message=f"NN-{best['kind']} "
                          f"{best['window_accuracy'] * 100:.1f} % per window "
                          f"({grouping}-grouped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

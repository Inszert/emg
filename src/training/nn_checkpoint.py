"""
The neural-net checkpoint format, in ONE place (torch is imported lazily).

`train_nn.py` and `train_zoo.py` both write checkpoints that
`training/nn_serve.py` loads and `emg_app/neural.py` describes. The outer keys
are defined here so the two trainers cannot drift apart; what goes inside
`stats` is the caller's business.

    format        "emg_nn_v1"
    kind          architecture name, rebuilt by nn_arch.build()
    state_dict    the weights
    scaler        log1p_standardise mean/std -- PART OF THE MODEL
    calibration   the per-wearer mode the inputs were built on (what was
                  APPLIED, not what was asked for)
    adaptation    {"mode": "person", ...} when the net was trained on
                  per-person standardised inputs, else {"mode": "off"}; the
                  worker only lets `adapt` statistics reach such a net
    augmentation  names of the augmentations used in training ([] = none)
    data          dataset fingerprint (is the model up to date?)
    stats         scores, protocol, provenance
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import emg_dataset_tool as T
import nn_arch

FORMAT = "emg_nn_v1"


def build_payload(model, *, kind, labels, n_channels, win, window_s, overlap,
                  model_fs, hidden, dropout, mean, std, applied_mode,
                  requested_mode, calibration_subjects, root, stats,
                  device_info, adaptation=None, augmentation=None):
    return {
        "format": FORMAT,
        "kind": kind,
        "architecture": nn_arch.DESCRIPTIONS.get(kind, ""),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": list(labels),
        "n_channels": int(n_channels),
        "win_samples": int(win),
        "window_s": float(window_s),
        "overlap": float(overlap),
        "model_fs": float(model_fs),
        "hidden": int(hidden),
        "dropout": float(dropout),
        "emg_mode": T.EMG_MODE,
        #  The scaler IS part of the model: inference must reproduce it
        #  exactly or the inputs are meaningless.
        "scaler": {"kind": "log1p_standardise", "mean": mean, "std": std},
        #  What was APPLIED, not what was asked for. Stamping "affine" on
        #  plain device counts would produce a model the app refuses to load
        #  -- a trainer locking its own output out of its own application.
        "calibration": {"mode": applied_mode, "requested": requested_mode,
                        "subjects": sorted(calibration_subjects),
                        "target": float(T.CALIBRATION_TARGET_LEVEL)},
        "adaptation": dict(adaptation or {"mode": "off"}),
        "augmentation": list(augmentation or []),
        #  Same provenance block train_model.py writes, so either payload can
        #  be asked the same question: trained on what, and is that still
        #  what is on disk?
        "data": T.dataset_fingerprint(root),
        "stats": stats,
        "device_trained_on": device_info,
        "created": datetime.now().isoformat(timespec="seconds"),
        "created_iso": T.iso_time(time.time()),
    }


def save_payload(payload, path, archive=True):
    """
    Write the checkpoint and its plain-JSON sidecar. Returns
    `(path, archive_path_or_None, sidecar_path)`.

    The sidecar holds everything except the weights and the scaler: the
    application must be able to answer "is a net trained, on what, and how
    well does it do" WITHOUT importing torch (a `.pt` is a torch pickle).
    """
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    archive_path = None
    if archive:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = os.path.splitext(os.path.basename(path))[0]
        archive_path = T.unique_path(os.path.join(
            os.path.dirname(os.path.abspath(path)), f"{stem}_{stamp}.pt"))
        torch.save(payload, archive_path)
    #  Write-then-replace: the app must never see half a checkpoint.
    staging = path + ".partial"
    torch.save(payload, staging)
    os.replace(staging, path)
    sidecar = T.nn_stats_path(path)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in payload.items()
                   if k not in ("state_dict", "scaler")},
                  fh, indent=2, default=str)
    return path, archive_path, sidecar

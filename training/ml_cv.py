"""
Windows, features and leave-one-person-out probabilities for the ML models.
Shared by `tools/ablation_ml.py` and `training/train_zoo.py` so that the two
cannot measure different things. Imports no torch.

`build_windows` returns the deduped windows UNcalibrated, plus the per-window
wearer calibration; calibration is applied inside `TM.feature_vector` (the one
function training and the live predictor share), so `featurize` is the only
place a feature vector is made.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emg_dataset_tool as T                                    # noqa: E402
import train_model as TM                                        # noqa: E402
import train_nn as NN                                           # noqa: E402
from zoo_roster import ML_SETUP                                 # noqa: E402


def build_windows(root, calib_mode, setup=None):
    """Uncalibrated windows + per-window calibration, deduped (train_model's rules)."""
    setup = setup or ML_SETUP
    recs, dropped = TM.load_recordings(
        root, persons=None, exclude_codes=NN.EXCLUDE_SETS["no_errors"])
    people = sorted({r.person_id for r in recs})
    cal, filled, used = T.training_calibrations(root, calib_mode, recs, people)
    win = max(4, int(round(setup["window_s"] * TM.MODEL_FS)))
    step = max(1, int(round(win * (1.0 - setup["overlap"]))))
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
            "labels": labels, "recs": recs, "dropped": dropped,
            "calibration": {"mode": calib_mode if cal else "off",
                            "subjects": sorted(cal), "filled": filled,
                            "records": used},
            "window_samples": win, "step": step}


def featurize(W, cals, setup=None):
    setup = setup or ML_SETUP
    return np.asarray([TM.feature_vector(W[i], setup["representation"],
                                         setup["freq_features"],
                                         setup["feature_set"], cals[i])
                       for i in range(len(W))], dtype=np.float32)


def augmented_features(W, cals, kinds, seed=0, copies=2, setup=None):
    """
    `copies` augmented versions of every window, as feature vectors (same row
    order as `W`). The augmentations act on the raw counts, before the
    wearer's calibration, which `featurize` applies.
    """
    import torch
    import nn_augment
    out = []
    for c in range(int(copies)):
        aug = nn_augment.Augmenter(kinds, seed=int(seed) * 100 + c,
                                   device="cpu")
        with torch.no_grad():
            A = aug(torch.from_numpy(np.ascontiguousarray(W))).numpy()
        out.append(featurize(A.astype(np.float32), cals, setup))
    return out


def prepare_features(D, augment=(), adapt=False, seed=0, copies=2):
    """
    What the classifiers are fitted on for one measurement:
    `(F, extras, table)` -- the clean features (per-person standardised when
    `adapt`), the augmented copies (standardised the same way), and the
    per-person statistics table (None without adaptation).
    """
    from emg_app import adaptation as AD
    F = D["F"]
    extras = (augmented_features(D["W"], D["cals"], augment, seed, copies)
              if augment else [])
    table = None
    if adapt:
        table = AD.person_feature_table(F, D["persons"], seed=seed)
        F = AD.apply_person_features(F, D["persons"], table)
        extras = [AD.apply_person_features(E, D["persons"], table)
                  for E in extras]
    return F, extras, table


def make_pipeline(entry, seed=0, n_jobs=-1):
    """A spec-shaped entry -> an unfitted pipeline (probabilities available)."""
    spec = {"name": entry[0], "kind": entry[1], "scaler": entry[2],
            "params": dict(entry[3])} if isinstance(entry, (list, tuple)) \
        else entry
    pipe = TM._make_pipeline(spec, for_scoring=False)
    clf = pipe.named_steps["clf"]
    params = clf.get_params()
    if "random_state" in params:
        pipe.set_params(clf__random_state=int(seed))
    if "n_jobs" in params:
        pipe.set_params(clf__n_jobs=n_jobs)
    return pipe


def loso_probs(F, y, persons, entry, seed=0, extras=()):
    """
    Out-of-fold probabilities, every person held out in turn. `extras` are
    augmented copies of `F` (same rows): they join the TRAINING rows only.
    """
    probs = np.zeros((len(y), int(y.max()) + 1), dtype=np.float32)
    for held in sorted(set(persons.tolist())):
        tr = np.flatnonzero(persons != held)
        te = np.flatnonzero(persons == held)
        pipe = make_pipeline(entry, seed)
        pipe.fit(np.concatenate([F[tr]] + [E[tr] for E in extras], axis=0),
                 np.concatenate([y[tr]] * (1 + len(extras))))
        p = pipe.predict_proba(F[te])
        classes = pipe.classes_ if hasattr(pipe, "classes_") else \
            pipe.named_steps["clf"].classes_
        probs[np.ix_(te, np.asarray(classes, dtype=int))] = p
    return probs

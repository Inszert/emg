"""
================================================================================
 TRAIN A GESTURE MODEL
================================================================================

 Sweeps preprocessing choices and classifiers over the recorded dataset,
 reports honest cross-validated scores, and saves the winner for live use in
 emg_dataset_tool.py.

 Run:
     python train_model.py                  # full sweep, saves the best model
     python train_model.py --quick          # fewer combinations
     python train_model.py --no-save        # evaluate only
     python train_model.py --person Dano    # restrict to one subject

 VALIDATION PROTOCOL -- the part that makes the numbers mean something
 --------------------------------------------------------------------
 Windows overlap, so two windows from one recording share samples. Splitting
 windows randomly puts near-duplicates on both sides of the split and reports
 an accuracy that does not survive contact with reality.

 This script therefore groups by RECORDING: a whole recording is either in
 train or in test, never both (StratifiedGroupKFold on the recording uid).

 With several subjects it additionally reports leave-one-subject-out, which is
 the number that predicts what happens when somebody new wears the armband.
 With one subject that is impossible and the script says so instead of
 quietly reporting something flattering.

 Two accuracies are reported:
   * per-window   -- one 200 ms slice, no smoothing. The pessimistic number.
   * per-recording-- majority vote over a whole gesture. This is what the live
                     app actually shows you, because it smooths over windows.
================================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

import emg_dataset_tool as T

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.base import clone
    from joblib import Parallel, delayed
    import joblib
except ImportError as exc:                                   # pragma: no cover
    print(f"scikit-learn and joblib are required: {exc}")
    print("    python -m pip install scikit-learn joblib")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASET_ROOT = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
OUT_DIR = os.path.join(T.PROJECT_DIR, "models")
REPORT_DIR = os.path.join(T.PROJECT_DIR, "models", "reports")

# Everything is resampled onto this grid so a window means a fixed DURATION
# regardless of how the BLE link behaved. See resample_uniform().
MODEL_FS = 60.0

# Window lengths to try, in SECONDS (not samples -- the whole point of MODEL_FS).
WINDOW_SECONDS = [0.20, 0.30, 0.40, 0.50]
OVERLAPS = [0.5, 0.75]

# "raw"        -> device counts, absolute amplitude preserved
# "normalized" -> per-window robust (median/MAD) scaling; throws away absolute
#                 amplitude, which is the part that differs most between people
# "envelope"   -> moving-RMS envelope
REPRESENTATIONS = ["raw", "normalized", "envelope"]

INCLUDE_FREQ_FEATURES = [True, False]

# Which feature family to feed the classifier:
#   "classic" -> extract_features(): rms/mav/wl/zc/ssc/mnf/mdf ... designed for
#                raw bipolar EMG. It median-centres each channel, which on this
#                rectified-envelope stream DELETES the activation level.
#   "level"   -> extract_level_features(): per-channel mean/median/p90/std plus
#                `share` (the spatial profile with amplitude divided out).
#   "both"    -> everything.
# Measured Fisher ratio on this dataset: channel level 25.5, median-centred
# 15.1 -- which is why "classic" alone scored barely above chance.
FEATURE_SETS = ["level", "both", "classic"]

# Drop recordings the quality checks flagged. "no_activity" means the
# contraction was too weak to detect, which is a real problem in this dataset.
EXCLUDE_SETS = {
    "all": (),
    "no_errors": ("disconnect", "flat_channel", "empty", "invalid_values"),
    "clean": ("disconnect", "flat_channel", "empty", "invalid_values",
              "no_activity"),
}

MIN_LABEL_WINDOW_SAMPLES = 20        # skip recordings with no usable cue window
CV_SPLITS = 6
RANDOM_STATE = 0

# How many CV folds may be fitted at once, for estimators that are themselves
# single-threaded (SVC, LDA, LogReg). Leave one core for the rest of the
# machine -- the GUI runs the sweep as a subprocess while still streaming EMG.
FOLD_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def emit_progress(done: int, total: int, phase: str = "sweep", **detail) -> None:
    """
    Print a machine-readable progress line.

    Two lines, on purpose:

      PROGRESS <done> <total> <phase>        the original, unchanged
      TRAINEVENT {...json...}                the detailed one

    The plain line stays because a sweep started before this change is still
    running and the GUI must keep understanding it; anything that parses only
    PROGRESS keeps working. The JSON line carries what a progress panel
    actually wants to show -- which model, which preprocessing, which fold --
    and a reader that does not know about it simply ignores it.

    Every field is something that really happened. Nothing here is estimated;
    the GUI derives the ETA from observed completions.
    """
    total = max(1, int(total))
    done = min(int(done), total)
    print(f"PROGRESS {done} {total} {phase}", flush=True)

    event = {"done": done, "total": total, "phase": phase,
             "device": "cpu"}       # sklearn has no GPU backend; see §3
    event.update({k: v for k, v in detail.items() if v is not None})
    try:
        print("TRAINEVENT " + json.dumps(event, default=str), flush=True)
    except (TypeError, ValueError):
        pass          # a progress line must never be able to kill a sweep


def build_models(quick: bool = False):
    """
    The candidate classifiers.

    LDA is first on purpose: it is the long-standing myoelectric-control
    baseline, it has no hyperparameters to overfit, it trains in milliseconds,
    and its shrinkage variant is the standard choice for per-user
    recalibration later. Tree ensembles and SVM-RBF are the usual strong
    contenders at this dataset size. A small MLP stands in for "neural
    network" -- the 1D-CNN/LSTM architectures the literature reports at 96 %+
    need far more data and a higher, steadier sample rate than 72 recordings
    of a 60 Hz envelope stream.
    """
    models = {
        "LDA-shrinkage": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                               shrinkage="auto")),
        ]),
        "LDA-svd": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="svd")),
        ]),
        "SVM-rbf-C10": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=10, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "RandomForest-400": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "ExtraTrees-400": Pipeline([
            ("clf", ExtraTreesClassifier(n_estimators=400, n_jobs=-1,
                                         random_state=RANDOM_STATE)),
        ]),
        "HistGradBoost": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "LogReg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       random_state=RANDOM_STATE)),
        ]),
    }
    if quick:
        return models

    models.update({
        "SVM-rbf-C1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=1, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "SVM-rbf-C100": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=100, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "SVM-rbf-robustscale": Pipeline([
            ("scale", RobustScaler()),
            ("clf", SVC(C=10, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "kNN-5": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance",
                                         n_jobs=-1)),
        ]),
        "kNN-15": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                         n_jobs=-1)),
        ]),
        "MLP-128": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=1200,
                                  early_stopping=True,
                                  random_state=RANDOM_STATE)),
        ]),
        "MLP-256-128": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1200,
                                  early_stopping=True,
                                  random_state=RANDOM_STATE)),
        ]),
        "RandomForest-800-deep": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=800, min_samples_leaf=1,
                                           n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
    })
    return models


# ==============================================================================
# MODELS FROM A SPEC
# ==============================================================================
#
# The spec is edited in the GUI (MODEL SPEC...) and read here. The schema and
# the validation live in emg_dataset_tool, so there is exactly one definition
# of what a spec means; this half only knows how to turn a validated entry
# into a scikit-learn estimator.


def _make_scaler(name: str):
    """A scaler stage, or None for the tree models that do not want one."""
    return {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }.get(name, lambda: None)()


def _weight(value):
    """'none' in the GUI means None in scikit-learn."""
    return None if value in (None, "none", "") else value


def _logreg_solver(penalty: str, wanted: str) -> str:
    """
    A solver that actually supports the chosen penalty.

    lbfgs cannot do L1 or elastic net, and liblinear cannot do elastic net.
    Letting a mismatch through would raise inside the sweep and lose the
    candidate, so "auto" resolves it and an explicit mismatch is corrected
    with a note rather than silently.
    """
    supported = {
        "l2": ("lbfgs", "saga", "liblinear"),
        "l1": ("saga", "liblinear"),
        "elasticnet": ("saga",),
        "none": ("lbfgs", "saga"),
    }[penalty]
    if wanted in supported:
        return wanted
    if wanted != "auto":
        print(f"  spec: solver {wanted!r} cannot do penalty {penalty!r}; "
              f"using {supported[0]!r}")
    return supported[0]


def _svm_probability(for_scoring: bool) -> bool:
    """
    Whether this SVC needs Platt scaling.

    `probability=True` makes libsvm run its own internal 5-fold CV to fit a
    sigmoid for `predict_proba`. Inside the sweep that work is pure waste:
    `evaluate()` only ever calls `predict()`, and `predict()` comes from the
    decision function, not from the sigmoid. The SVM solve itself -- and
    therefore the support vectors and every score -- is unchanged.

    Measured on this machine at the sweep's worst-case size (6300 windows x
    154 features, C=10):

        probability=True    54.26 s     n_support [2892, 2829]
        probability=False    9.52 s     n_support [2892, 2829]
        predict() identical: True  (100.0000 % over 6300 rows)

    So scoring runs without it and the SAVED model is built separately with
    it, because `LivePredictor` and the soft-voting ensemble both need
    `predict_proba`. `tests/test_model_spec.py` pins the identity.
    """
    return not for_scoring


def _make_estimator(kind: str, params: dict, for_scoring: bool = False):
    """
    One classifier, from a validated spec entry.

    `for_scoring` builds the variant used inside cross-validation, where only
    `predict()` is ever called. The only thing it changes is SVC's Platt
    scaling -- see `_svm_probability()` for why that is safe.
    """
    p = dict(params or {})

    if kind == "lda":
        solver = p.get("solver", "lsqr")
        shrinkage = p.get("shrinkage", "auto")
        # svd does not support shrinkage at all; passing it raises.
        if solver == "svd" or shrinkage == "none":
            return LinearDiscriminantAnalysis(solver=solver)
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    if kind == "svm_rbf":
        return SVC(C=float(p.get("C", 10.0)),
                   gamma=p.get("gamma", "scale"),
                   kernel=p.get("kernel", "rbf"),
                   class_weight=_weight(p.get("class_weight")),
                   probability=_svm_probability(for_scoring),
                   random_state=RANDOM_STATE)

    if kind == "svm_linear":
        return SVC(C=float(p.get("C", 1.0)), kernel="linear",
                   class_weight=_weight(p.get("class_weight")),
                   probability=_svm_probability(for_scoring),
                   random_state=RANDOM_STATE)

    if kind in ("random_forest", "extra_trees"):
        cls = (RandomForestClassifier if kind == "random_forest"
               else ExtraTreesClassifier)
        depth = int(p.get("max_depth", 0))
        features = p.get("max_features", "sqrt")
        kwargs = dict(
            n_estimators=int(p.get("n_estimators", 400)),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            # 0 means "unlimited" in the GUI, because a spin box has no None.
            max_depth=None if depth <= 0 else depth,
            class_weight=_weight(p.get("class_weight")),
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        if kind == "random_forest":
            kwargs["max_features"] = None if features == "none" else features
        return cls(**kwargs)

    if kind == "hist_gradient_boost":
        stopping = p.get("early_stopping", "auto")
        return HistGradientBoostingClassifier(
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_iter=int(p.get("max_iter", 200)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
            l2_regularization=float(p.get("l2_regularization", 0.0)),
            early_stopping=("auto" if stopping == "auto"
                            else stopping == "yes"),
            validation_fraction=float(p.get("validation_fraction", 0.1)),
            n_iter_no_change=int(p.get("n_iter_no_change", 10)),
            class_weight=_weight(p.get("class_weight")),
            random_state=RANDOM_STATE)

    if kind == "logreg":
        penalty = p.get("penalty", "l2")
        solver = _logreg_solver(penalty, p.get("solver", "auto"))
        kwargs = dict(C=float(p.get("C", 1.0)),
                      max_iter=int(p.get("max_iter", 2000)),
                      solver=solver,
                      class_weight=_weight(p.get("class_weight")),
                      random_state=RANDOM_STATE)
        # sklearn spells "no penalty" as penalty=None, and l1_ratio is only
        # accepted with elastic net.
        kwargs["penalty"] = None if penalty == "none" else penalty
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = float(p.get("l1_ratio", 0.5))
        return LogisticRegression(**kwargs)

    if kind == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(p.get("n_neighbors", 5)),
            weights=p.get("weights", "distance"), n_jobs=-1)

    if kind == "mlp":
        layers = p.get("hidden_layer_sizes", [128])
        if isinstance(layers, (int, float)):
            layers = [int(layers)]
        batch = int(p.get("batch_size", 0))
        solver = p.get("solver", "adam")
        kwargs = dict(
            hidden_layer_sizes=tuple(int(n) for n in layers),
            alpha=float(p.get("alpha", 0.0001)),
            solver=solver,
            max_iter=int(p.get("max_iter", 1200)),
            # 0 means "auto" in the GUI; sklearn spells it "auto".
            batch_size="auto" if batch <= 0 else batch,
            n_iter_no_change=int(p.get("n_iter_no_change", 10)),
            validation_fraction=float(p.get("validation_fraction", 0.1)),
            random_state=RANDOM_STATE)
        # lbfgs is a full-batch solver: it ignores batching and cannot early
        # stop on a validation slice, and passing those raises a warning
        # storm rather than doing anything.
        if solver != "lbfgs":
            kwargs["learning_rate_init"] = float(
                p.get("learning_rate_init", 0.001))
            kwargs["early_stopping"] = p.get("early_stopping", "yes") == "yes"
        return MLPClassifier(**kwargs)

    raise ValueError(f"unknown model kind: {kind!r}")


def _make_pipeline(entry: dict, for_scoring: bool = False):
    """A spec entry as a Pipeline, with its scaler stage if it has one."""
    estimator = _make_estimator(entry["kind"], entry.get("params"),
                                for_scoring=for_scoring)
    scaler = _make_scaler(entry.get("scaler", "none"))
    steps = ([("scale", scaler)] if scaler is not None else [])
    steps.append(("clf", estimator))
    return Pipeline(steps)


def build_models_from_spec(spec: dict, quick: bool = False,
                           for_scoring: bool = False):
    """
    The candidate classifiers described by `spec`.

    The voting ensemble is added as one more CANDIDATE rather than replacing
    the others, so the sweep scores it against its own members with the same
    protocol. If the vote is worse than its best member, that is worth
    knowing, and `voting.always` is the switch for using it regardless.

    `quick` halves the roster by keeping the enabled entries in spec order --
    the GUI's QUICK button is about finishing sooner, not about a different
    set of models.
    """
    clean, problems = T.validate_model_spec(spec)
    for problem in problems:
        print(f"  spec: {problem}")

    # AUTO mode becomes ordinary candidates, so every combination is scored by
    # the same cross-validation as a hand-written entry and appears as its own
    # row. There is no separate, hidden tuning path.
    clean, note = T.expand_auto_spec(clean, seed=RANDOM_STATE)
    if (clean.get("auto") or {}).get("enabled"):
        print(f"  spec: {note}")
        print("  spec: NOTE -- picking the best of many combinations on the "
              "same folds makes the winner's score optimistic. The honest "
              "number for a tuned model is leave-one-subject-out below.")

    entries = [m for m in clean["models"] if m["enabled"]]
    if quick and len(entries) > 4:
        entries = entries[:4]

    models = {}
    #  The ensemble is built from its OWN copies, always with predict_proba
    #  available. Soft voting picks members by `hasattr(clf, "predict_proba")`,
    #  and an SVC built for scoring does not have it -- so sharing the
    #  scoring pipelines here would silently drop the SVM from the ensemble
    #  and change what is being measured, which is exactly the kind of
    #  invisible experiment change this file exists to avoid.
    voting_pool = {}
    for entry in entries:
        try:
            models[entry["name"]] = _make_pipeline(entry,
                                                   for_scoring=for_scoring)
            voting_pool[entry["name"]] = _make_pipeline(entry,
                                                        for_scoring=False)
        except (ValueError, TypeError) as exc:
            # One bad entry must not abort a sweep that has fourteen good ones.
            print(f"  spec: {entry['name']} could not be built ({exc}); "
                  "skipped")

    voting = clean["voting"]
    if voting["enabled"] and len(models) > 1:
        if voting["mode"] == "soft":
            # Soft voting needs predict_proba from every member.
            members = [(name, pipe) for name, pipe in voting_pool.items()
                       if hasattr(pipe.named_steps["clf"], "predict_proba")]
            dropped = [n for n in voting_pool if n not in dict(members)]
            if dropped:
                print(f"  spec: {', '.join(dropped)} cannot do soft voting "
                      "(no predict_proba); left out of the ensemble")
        else:
            members = list(voting_pool.items())

        if len(members) > 1:
            name = f"Vote-{voting['mode']}-{len(members)}"
            models[name] = VotingClassifier(
                estimators=[(n, p) for n, p in members],
                voting=voting["mode"], n_jobs=None)
            print(f"  spec: {name} over {', '.join(n for n, _ in members)}")

    if not models:                                   # pragma: no cover
        print("  spec: nothing could be built; falling back to the defaults")
        return build_models(quick)
    return models


def resolve_spec(path: str = None):
    """
    The spec to train with, and where it came from.

    No file means the defaults, which reproduce the roster this script has
    always swept -- so an existing setup behaves identically after this
    change.
    """
    if not path:
        return T.default_model_spec(), "built-in defaults"
    spec, problems = T.load_model_spec(path)
    for problem in problems:
        print(f"  spec: {problem}")
    return spec, path


# ==============================================================================
# DATASET BUILDING
# ==============================================================================


def load_recordings(root, persons=None, sessions=None, exclude_codes=()):
    """Load recordings, dropping the ones whose quality codes are excluded."""
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


def feature_vector(window, representation, include_freq, feature_set):
    """
    One window -> one feature vector.

    This function is the contract between training and inference: the live
    predictor calls the very same code path, so a change here cannot silently
    desynchronise the two.
    """
    return T.model_feature_vector(window, representation, feature_set,
                                  include_freq, MODEL_FS)


def feature_names_for(feature_set, include_freq):
    return T.model_feature_names(feature_set, include_freq)


def build_dataset(recs, window_s, overlap, representation, include_freq,
                  feature_set="level"):
    """
    Recordings -> (features, y, groups, labels, meta).

    `groups` is the recording uid, which is the unit the cross-validation
    splits on.
    """
    win = max(4, int(round(window_s * MODEL_FS)))
    step = max(1, int(round(win * (1.0 - overlap))))
    labels = sorted({r.label for r in recs})
    index = {name: i for i, name in enumerate(labels)}

    feats, ys, groups, rec_ids = [], [], [], []
    for rec in recs:
        a, b = rec.label_start_index, rec.label_end_index
        if b - a < MIN_LABEL_WINDOW_SAMPLES:
            continue

        # Fixed-duration grid: the cue window resampled to MODEL_FS.
        seconds = max(1e-6, float(rec.t_rel[b - 1] - rec.t_rel[a]))
        n_out = max(win, int(round(seconds * MODEL_FS)))
        block = T.resample_uniform(rec.emg[a:b], rec.t_rel[a:b], n_out)

        windows = T.sliding_windows(block.astype(np.float32), win, step)
        if windows.shape[0] == 0:
            continue

        for w in windows:
            feats.append(feature_vector(w, representation, include_freq,
                                        feature_set))
            ys.append(index[rec.label])
            groups.append(rec.uid)
            rec_ids.append(rec.uid)

    if not feats:
        return None
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(ys, dtype=int),
            np.asarray(groups, dtype=object),
            labels,
            {"window_samples": win, "step": step, "window_s": window_s})


# ==============================================================================
# EVALUATION
# ==============================================================================


def _estimator_is_parallel(model) -> bool:
    """
    True when the estimator already spreads itself over the cores.

    Forests and boosting run with `n_jobs=-1`; running their folds in
    parallel as well would oversubscribe the CPU and usually ends up slower.
    SVC, LDA and the rest are single-threaded, and those are the ones whose
    folds are worth parallelising.
    """
    clf = getattr(model, "named_steps", {}).get("clf", model)
    n_jobs = getattr(clf, "n_jobs", None)
    if n_jobs not in (None, 1):
        return True
    # A VotingClassifier is a bag of estimators: if any member is parallel,
    # treat the whole thing as parallel.
    for _, member in getattr(clf, "estimators", []) or []:
        inner = getattr(member, "named_steps", {}).get("clf", member)
        if getattr(inner, "n_jobs", None) not in (None, 1):
            return True
    return False


def _run_folds(model, X, y, groups, folds, on_fold=None):
    """
    Fit and predict each fold, returning `(test_idx, predictions)` per fold.

    Each fold gets its OWN clone, so nothing is shared between them: that is
    what makes running them in parallel identical to running them in
    sequence, and it also removes the quiet assumption that `fit()` fully
    resets the estimator.
    """
    def one(train_idx, test_idx):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])
        return test_idx, fitted.predict(X[test_idx])

    workers = 1 if _estimator_is_parallel(model) else min(len(folds),
                                                          FOLD_WORKERS)
    if workers <= 1 or on_fold is not None:
        # Sequential when the estimator is already parallel, and whenever a
        # per-fold callback is attached -- progress must be reported in order.
        out = []
        for i, (train_idx, test_idx) in enumerate(folds):
            out.append(one(train_idx, test_idx))
            if on_fold is not None:
                on_fold(i + 1, len(folds))
        return out

    # THREADS, not processes. Measured on this machine at the sweep's worst
    # case (6307 windows x 154 features, SVC C=10, 6 folds):
    #
    #     sequential   6.17 s
    #     threads      4.66 s   1.32x   identical results
    #     processes    9.99 s   0.62x   identical results
    #
    # Processes lose because Windows has to spawn an interpreter per worker
    # and pickle the fold arrays into it; libsvm releases the GIL for enough
    # of its fit that threads come out ahead. 1.32x is modest, but it is free
    # and provably result-identical.
    return Parallel(n_jobs=workers, prefer="threads")(
        delayed(one)(train_idx, test_idx) for train_idx, test_idx in folds)


def evaluate(model, X, y, groups, labels, n_splits=CV_SPLITS, on_fold=None):
    """
    Grouped stratified cross-validation.

    Returns per-window accuracy/F1, per-recording majority-vote accuracy, the
    summed confusion matrix and the per-fold spread.

    `on_fold(done, total)` is called after each fold when given, for progress
    reporting; passing it forces sequential folds so the counts are ordered.
    """
    counts = Counter(y.tolist())
    n_groups = len(set(groups.tolist()))
    n_splits = int(min(n_splits, min(counts.values()), n_groups))
    if n_splits < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    folds = list(cv.split(X, y, groups))

    results = _run_folds(model, X, y, groups, folds, on_fold=on_fold)

    window_acc, window_f1, fold_acc = [], [], []
    vote_correct = vote_total = 0
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for test_idx, pred in results:
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
    }


def leave_one_subject_out(model, X, y, groups_person, labels):
    """Subject-independent score. None when there is only one subject."""
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
        model.fit(X[train], y[train])
        scores[person] = float(accuracy_score(y[test], model.predict(X[test])))
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
    print(" " * (width + 2) + " ".join(
        f"{(cm[:, j][j] / cm[:, j].sum() * 100 if cm[:, j].sum() else 0):>5.1f}%"
        for j in range(len(labels))) + "   <- precision")


# ==============================================================================
# MAIN
# ==============================================================================


def main(argv=None):
    T.force_utf8_console()
    parser = argparse.ArgumentParser(description="Train an EMG gesture model")
    parser.add_argument("--quick", action="store_true",
                        help="fewer models and window sizes")
    parser.add_argument("--no-save", action="store_true",
                        help="evaluate only, do not write a model")
    parser.add_argument("--person", action="append", default=None,
                        help="restrict to these person ids (repeatable)")
    parser.add_argument("--root", default=DATASET_ROOT,
                        help="dataset root to train on")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="where to write the model and reports")
    parser.add_argument("--model-path", default=None,
                        help="exact file to write the model to; overrides "
                             "--out-dir for the model itself. The GUI passes "
                             "this so its MODEL_PATH and this script can "
                             "never point at different files.")
    parser.add_argument("--spec", default=None,
                        help="model_spec.json describing which classifiers to "
                             "sweep, their hyperparameters and whether they "
                             "vote. Written by MODEL SPEC... in the GUI. "
                             "Without it the built-in roster is used.")
    parser.add_argument("--top", type=int, default=12,
                        help="how many sweep rows to print")
    args = parser.parse_args(argv)

    print("=" * 78)
    print(" EMG GESTURE MODEL TRAINING")
    print("=" * 78)

    # ---- data ---------------------------------------------------------------
    all_recs = T.load_dataset(args.root, persons=args.person,
                              stage="prefer_trimmed", prefer="npz")
    if not all_recs:
        print(f"\nNo recordings found under {args.root}.")
        return 1

    people = sorted({r.person_id for r in all_recs})
    sessions = sorted({(r.person_id, r.session_id) for r in all_recs})
    print(f"\nDataset        : {args.root}")
    print(f"Recordings     : {len(all_recs)}")
    print(f"Subjects       : {len(people)}  {people}")
    print(f"Sessions       : {len(sessions)}")
    print(f"Labels         : {sorted({r.label for r in all_recs})}")
    rates = [r.fs for r in all_recs if r.duration > 0.5]
    print(f"Measured rate  : {min(rates):.0f}-{max(rates):.0f} Hz "
          f"(median {np.median(rates):.0f}); resampling everything to "
          f"{MODEL_FS:.0f} Hz")

    if len(people) < 2:
        print("\n" + "!" * 78)
        print(" ONE SUBJECT ONLY -- this model will be subject-specific.")
        print(" Leave-one-subject-out cannot be computed, so NO claim about")
        print(" generalising to another wearer is made below. Scores are")
        print(" leave-recordings-out within this subject.")
        print("!" * 78)

    # ---- sweep --------------------------------------------------------------
    windows = WINDOW_SECONDS[:2] if args.quick else WINDOW_SECONDS
    overlaps = OVERLAPS[:1] if args.quick else OVERLAPS
    representations = REPRESENTATIONS[:2] if args.quick else REPRESENTATIONS
    freq_options = [True] if args.quick else INCLUDE_FREQ_FEATURES
    feature_sets = FEATURE_SETS[:2] if args.quick else FEATURE_SETS
    exclude_sets = ({"clean": EXCLUDE_SETS["clean"]} if args.quick
                    else EXCLUDE_SETS)
    # A classifier needs something to discriminate BETWEEN. Without this
    # check the sweep runs every model against a single class and prints a
    # wall of "The number of classes has to be greater than one", which says
    # nothing about what to actually do.
    present = sorted({r.label for r in all_recs})
    if len(present) < 2:
        print("\n" + "=" * 78)
        print(" NOT ENOUGH GESTURE CLASSES TO TRAIN")
        print("=" * 78)
        print(f"\nEvery recording on disk has the same label: {present[0]!r}"
              if present else "\nNo labelled recordings were found.")
        print(f"\n  recordings found : {len(all_recs)}")
        print(f"  gesture classes  : {len(present)}  ({', '.join(present)})")
        print("\nA classifier learns to tell gestures APART, so it needs at "
              "least two.\nRecord a second gesture -- FIST is the usual next "
              "one -- and train again.")
        print("\nNothing was written. The existing model, if any, is "
              "untouched.")
        return 2

    spec, spec_source = resolve_spec(args.spec)
    print(f"\nModel spec     : {spec_source}")
    print(f"                 {T.describe_model_spec(spec)}")
    # for_scoring: inside the sweep only predict() is used, so the SVMs skip
    # Platt scaling (5.7x faster, identical predictions -- see
    # _svm_probability). The SAVED model is rebuilt below without this.
    models = build_models_from_spec(spec, args.quick, for_scoring=True)
    print(f"                 candidates: {', '.join(models)}")

    total = (len(windows) * len(overlaps) * len(representations)
             * len(freq_options) * len(feature_sets) * len(exclude_sets)
             * len(models))
    # Stage A probes only a few models per setup; stage B then runs every
    # model on the few best setups. Both are counted up front so the GUI can
    # show one honest percentage instead of restarting at 0 % halfway.
    n_setups = max(1, total // max(1, len(models)))
    n_probes = min(3, len(models))
    n_best = 2 if args.quick else 5
    stage_a_total = n_setups * n_probes
    grand_total = stage_a_total + n_best * len(models)
    emit_progress(0, grand_total, "starting", status="running",
                  n_candidates=grand_total, n_folds=CV_SPLITS,
                  message=(f"{len(models)} candidates x "
                           f"{total // max(1, len(models))} setups"))

    print(f"\nSweep          : {total} combinations "
          f"({len(models)} models x {total // max(1, len(models))} setups)")
    print(f"Validation     : StratifiedGroupKFold grouped by RECORDING "
          f"({CV_SPLITS} folds max)\n")

    # Two stages, because the full cartesian product of setups x models would
    # take hours: first find the best PREPROCESSING with three fast probe
    # models, then run every model on the few best setups.
    probe_names = [n for n in ("LDA-shrinkage", "ExtraTrees-400", "SVM-rbf-C10")
                   if n in models]
    print(f"Stage A        : preprocessing search with {probe_names}")
    print(f"Stage B        : all {len(models)} models on the best setups\n")

    results = []
    rec_cache = {}

    def dataset_for(key):
        """
        Rebuild one setup's feature matrix from its cached recordings.

        Stage B and the final fit each need a handful of setups, so they are
        rebuilt rather than all 432 being held in memory for the whole run.
        `build_dataset` is deterministic, so a rebuilt matrix is identical to
        the one stage A scored.
        """
        exclude_name, window_s, overlap, representation, include_freq, \
            feature_set = key
        recs_for = rec_cache.get(exclude_name)
        if recs_for is None:
            return None
        built = build_dataset(recs_for, window_s, overlap, representation,
                              include_freq, feature_set)
        if built is None:
            return None
        X, y, groups, labels, meta = built
        return X, y, groups, labels, meta, recs_for
    done = 0
    t0 = time.perf_counter()

    for exclude_name, codes in exclude_sets.items():
        recs, dropped = load_recordings(args.root, args.person, None, codes)
        if len(recs) < 12:
            print(f"  [{exclude_name}] only {len(recs)} recordings left, skipping")
            continue
        print(f"  [{exclude_name}] {len(recs)} recordings "
              f"({len(dropped)} dropped)")

        for window_s in windows:
            for overlap in overlaps:
                for representation in representations:
                  for feature_set in feature_sets:
                    for include_freq in freq_options:
                        # The frequency flag only affects the classic family.
                        if feature_set == "level" and not include_freq:
                            continue
                        key = (exclude_name, window_s, overlap, representation,
                               include_freq, feature_set)
                        built = build_dataset(recs, window_s, overlap,
                                              representation, include_freq,
                                              feature_set)
                        if built is None:
                            continue
                        X, y, groups, labels, meta = built
                        # Only the RECORDINGS are kept (three sets of them,
                        # one per exclude filter). Caching every built X here
                        # retained all 432 feature matrices for the whole
                        # sweep -- hundreds of MB that grew monotonically,
                        # when stage B needs five of them and the final fit
                        # needs one. They are rebuilt on demand instead.
                        rec_cache[exclude_name] = recs

                        for model_name in probe_names:
                            model = models[model_name]
                            done += 1
                            try:
                                scores = evaluate(model, X, y, groups, labels)
                            except Exception as exc:
                                print(f"    {model_name} failed: {exc}")
                                continue
                            if scores is None:
                                continue
                            results.append({
                                "stage": "A",
                                "model": model_name,
                                "exclude": exclude_name,
                                "window_s": window_s,
                                "overlap": overlap,
                                "representation": representation,
                                "feature_set": feature_set,
                                "freq_features": include_freq,
                                "n_windows": int(X.shape[0]),
                                "n_features": int(X.shape[1]),
                                "n_recordings": len(recs),
                                **scores,
                            })
                            if done % 5 == 0 or done == grand_total:
                                emit_progress(
                                    done, grand_total, "stage A",
                                    model=model_name,
                                    candidate=done, n_candidates=grand_total,
                                    window_s=window_s, overlap=overlap,
                                    representation=representation,
                                    feature_set=feature_set,
                                    freq_features=include_freq,
                                    exclude=exclude_name,
                                    n_windows=int(X.shape[0]),
                                    n_features=int(X.shape[1]),
                                    n_folds=scores.get("n_splits"),
                                    status="running",
                                    message=(f"{model_name} on {window_s:.2f}s "
                                             f"{representation}/{feature_set}"))
                            if done % 50 == 0:
                                print(f"    ... {done}/{stage_a_total} "
                                      f"({time.perf_counter() - t0:.0f} s)")

    if not results:
        print("\nNothing could be evaluated.")
        return 1

    def setup_key(row):
        return (row["exclude"], row["window_s"], row["overlap"],
                row["representation"], row["freq_features"], row["feature_set"])

    print(f"\nStage A done: {len(results)} probes in "
          f"{time.perf_counter() - t0:.0f} s")

    # Best setups by the probe scores, de-duplicated.
    best_setups, seen = [], set()
    for row in sorted(results, key=lambda r: (r["recording_vote_accuracy"],
                                              r["window_accuracy"]),
                      reverse=True):
        key = setup_key(row)
        if key in seen:
            continue
        seen.add(key)
        best_setups.append(key)
        if len(best_setups) >= n_best:
            break

    print(f"\nStage B: running all {len(models)} models on "
          f"{len(best_setups)} best setup(s)")
    for key in best_setups:
        rebuilt = dataset_for(key)
        if rebuilt is None:
            print(f"  setup {key} could not be rebuilt; skipped")
            continue
        X, y, groups, labels, meta, recs = rebuilt
        print(f"  setup: exclude={key[0]} win={key[1]}s ov={key[2]} "
              f"repr={key[3]} feats={key[5]} -> {X.shape[0]} x {X.shape[1]}")
        for model_name, model in models.items():
            if any(r["model"] == model_name and setup_key(r) == key
                   for r in results):
                continue                      # already measured in stage A
            try:
                scores = evaluate(model, X, y, groups, labels)
            except Exception as exc:
                print(f"    {model_name} failed: {exc}")
                continue
            if scores is None:
                continue
            done += 1
            if done % 5 == 0 or done == grand_total:
                emit_progress(
                    done, grand_total, "stage B",
                    model=model_name,
                    candidate=done, n_candidates=grand_total,
                    window_s=key[1], overlap=key[2], representation=key[3],
                    feature_set=key[5], freq_features=key[4], exclude=key[0],
                    n_windows=int(X.shape[0]), n_features=int(X.shape[1]),
                    n_folds=scores.get("n_splits"), status="running",
                    message=f"{model_name} on the best setups")
            results.append({
                "stage": "B",
                "model": model_name,
                "exclude": key[0],
                "window_s": key[1],
                "overlap": key[2],
                "representation": key[3],
                "freq_features": key[4],
                "feature_set": key[5],
                "n_windows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_recordings": len(recs),
                **scores,
            })

    elapsed = time.perf_counter() - t0
    print(f"\nEvaluated {len(results)} combinations in {elapsed:.0f} s")

    # ---- leaderboard --------------------------------------------------------
    # Rank on the per-recording vote, which is what the app shows, and break
    # ties on the per-window score.
    results.sort(key=lambda r: (r["recording_vote_accuracy"],
                                r["window_accuracy"]), reverse=True)

    print("\n" + "=" * 78)
    print(f" TOP {args.top} COMBINATIONS")
    print("=" * 78)
    header = (f"{'model':<22} {'win':>5} {'ov':>5} {'repr':<11} "
              f"{'feats':<8} {'data':<10} {'window':>8} {'vote':>7}")
    print(header)
    print("-" * 78)
    for row in results[:args.top]:
        print(f"{row['model']:<22} {row['window_s']:>5.2f} "
              f"{row['overlap']:>5.2f} {row['representation']:<11} "
              f"{row['feature_set']:<8} {row['exclude']:<10} "
              f"{row['window_accuracy'] * 100:>6.1f}% "
              f"{row['recording_vote_accuracy'] * 100:>6.1f}%")

    print("\n--- best per model (ranked by per-recording vote) ---")
    best_per_model = {}
    for row in results:
        if row["model"] not in best_per_model:
            best_per_model[row["model"]] = row
    for name, row in sorted(best_per_model.items(),
                            key=lambda kv: -kv[1]["recording_vote_accuracy"]):
        print(f"  {name:<22} window {row['window_accuracy'] * 100:5.1f}% "
              f"+-{row['window_accuracy_std'] * 100:4.1f}   "
              f"vote {row['recording_vote_accuracy'] * 100:5.1f}%   "
              f"f1 {row['window_macro_f1'] * 100:5.1f}%   "
              f"({row['window_s']:.2f}s {row['representation']}/"
              f"{row['feature_set']})")

    print("\n--- effect of each choice (mean per-recording vote) ---")
    for field in ("feature_set", "representation", "window_s", "overlap",
                  "freq_features", "exclude"):
        groups_ = defaultdict(list)
        for row in results:
            groups_[row[field]].append(row["recording_vote_accuracy"])
        print(f"  {field}:")
        for value, scores in sorted(groups_.items(),
                                    key=lambda kv: -np.mean(kv[1])):
            print(f"      {str(value):<12} {np.mean(scores) * 100:5.1f}%  "
                  f"(best {max(scores) * 100:5.1f}%, n={len(scores)})")

    # ---- winner -------------------------------------------------------------
    best = results[0]
    print("\n" + "=" * 78)
    print(" WINNER")
    print("=" * 78)
    for key in ("model", "feature_set", "representation", "window_s",
                "overlap", "freq_features", "exclude", "n_windows",
                "n_features", "n_recordings"):
        print(f"  {key:<18} {best[key]}")
    print(f"  {'per-window acc':<18} {best['window_accuracy'] * 100:.1f}% "
          f"+- {best['window_accuracy_std'] * 100:.1f}")
    print(f"  {'per-recording acc':<18} "
          f"{best['recording_vote_accuracy'] * 100:.1f}% "
          f"({best['n_recordings_voted']} recordings)")
    print(f"  {'macro F1':<18} {best['window_macro_f1'] * 100:.1f}%")
    print(f"  {'folds':<18} "
          + ", ".join(f"{a * 100:.0f}%" for a in best["fold_accuracies"]))

    key = (best["exclude"], best["window_s"], best["overlap"],
           best["representation"], best["freq_features"], best["feature_set"])
    rebuilt = dataset_for(key)
    if rebuilt is None:
        print(f"\nThe winning setup {key} could not be rebuilt; nothing saved.")
        return 1
    X, y, groups, labels, meta, recs = rebuilt

    print(f"\n  confusion matrix (summed over folds, per window):")
    print_confusion(np.array(best["confusion_matrix"]), labels)

    # Also scoring-only: leave_one_subject_out calls fit() and predict().
    model = build_models_from_spec(spec, args.quick,
                                   for_scoring=True)[best["model"]]
    lopo = leave_one_subject_out(
        model, X, y,
        np.array([next(r.person_id for r in recs if r.uid == g)
                  for g in groups], dtype=object),
        labels)
    if lopo:
        print(f"\n  leave-one-subject-out: {lopo['mean'] * 100:.1f}% "
              f"+- {lopo['std'] * 100:.1f}")
        for person, score in sorted(lopo["per_subject"].items()):
            print(f"      holding out {person:<12} {score * 100:5.1f}%")
    else:
        print("\n  leave-one-subject-out: NOT COMPUTABLE (one subject). "
              "This model is subject-specific.")

    # ---- fit on everything and save ----------------------------------------
    out_dir = os.path.abspath(args.out_dir)
    report_dir = os.path.join(out_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    stamp = T.timestamp_slug()
    report = {
        "created_iso": T.iso_time(time.time()),
        "dataset_root": args.root,
        "model_spec": spec,
        "model_spec_source": spec_source,
        "n_recordings_total": len(all_recs),
        "subjects": people,
        "n_sessions": len(sessions),
        "labels": labels,
        "model_fs": MODEL_FS,
        "measured_rate_range": [float(min(rates)), float(max(rates))],
        "validation": {
            "protocol": "StratifiedGroupKFold grouped by recording uid",
            "n_splits": best["n_splits"],
            "note": ("Windows overlap, so grouping by recording is required; "
                     "a random window split would leak and inflate scores."),
            "subject_independent": lopo,
            "subject_independent_note": (
                None if lopo else
                "Only one subject in the dataset: no cross-subject claim can "
                "be made. Collect more people for that."),
        },
        "best": best,
        "best_per_model": {k: v for k, v in best_per_model.items()},
        "sweep": results,
        "elapsed_s": elapsed,
    }
    report_path = T.unique_path(os.path.join(report_dir, f"sweep_{stamp}.json"))
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n  sweep report: {os.path.relpath(report_path, T.PROJECT_DIR)}")

    if args.no_save:
        print("\n--no-save: model not written.")
        return 0

    print("\nFitting the winner on the whole dataset...")
    # No for_scoring here on purpose: this is the model that gets SAVED, and
    # LivePredictor calls predict_proba on it.
    model = build_models_from_spec(spec, args.quick)[best["model"]]
    model.fit(X, y)

    train_pred = model.predict(X)
    payload = {
        "model": model,
        "model_spec": spec,
        "model_spec_source": spec_source,
        "labels": labels,
        "fs": MODEL_FS,
        "window_s": best["window_s"],
        "window_ms": best["window_s"] * 1000.0,
        "representation": best["representation"],
        "features": True,
        "feature_set": best["feature_set"],
        "freq_features": best["freq_features"],
        "window_samples": meta["window_samples"],
        "feature_names": feature_names_for(best["feature_set"],
                                           best["freq_features"]),
        "channel_names": list(T.CHANNEL_NAMES),
        "emg_mode": T.EMG_MODE,
        "created_iso": T.iso_time(time.time()),
        # What this model was trained on, in a form the app can compare
        # against the dataset on disk. Without it "is my model up to date?"
        # is unanswerable, and the GUI says so rather than guessing.
        "data": T.dataset_fingerprint(args.root),
        "stats": {
            "model_name": best["model"],
            "feature_set": best["feature_set"],
            "representation": best["representation"],
            "window_s": best["window_s"],
            "window_accuracy": best["window_accuracy"],
            "window_accuracy_std": best["window_accuracy_std"],
            "window_macro_f1": best["window_macro_f1"],
            "recording_vote_accuracy": best["recording_vote_accuracy"],
            "n_recordings_voted": best["n_recordings_voted"],
            "fold_accuracies": best["fold_accuracies"],
            "confusion_matrix": best["confusion_matrix"],
            "n_windows": best["n_windows"],
            "n_features": best["n_features"],
            "n_recordings": best["n_recordings"],
            "n_recordings_total": len(all_recs),
            "subjects": people,
            "n_sessions": len(sessions),
            "excluded": best["exclude"],
            "overlap": best["overlap"],
            "validation_protocol": "leave-recordings-out (StratifiedGroupKFold)",
            "n_splits": best["n_splits"],
            "subject_independent": lopo,
            "train_fit_accuracy": float(accuracy_score(y, train_pred)),
            "report_file": os.path.relpath(report_path, T.PROJECT_DIR),
            "created_iso": T.iso_time(time.time()),
        },
    }

    model_path = (os.path.abspath(args.model_path) if args.model_path
                  else os.path.join(out_dir, "gesture_model.joblib"))
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    archive = T.unique_path(os.path.join(
        os.path.dirname(model_path), f"{stem}_{stamp}.joblib"))
    joblib.dump(payload, archive)
    # Write-then-replace, so a crash or a kill mid-dump cannot leave a
    # half-written gesture_model.joblib for the app to try to load. The
    # replace is atomic on the same filesystem; the model the app sees is
    # either the old one or the complete new one, never a fragment.
    staging = model_path + ".partial"
    joblib.dump(payload, staging)
    os.replace(staging, model_path)           # the one the app loads
    print(f"  saved: {os.path.relpath(model_path, T.PROJECT_DIR)}")
    print(f"  archive: {os.path.relpath(archive, T.PROJECT_DIR)}")

    emit_progress(grand_total, grand_total, "done", status="completed",
                  model=best["model"], device="cpu",
                  window_accuracy=best["window_accuracy"],
                  recording_vote_accuracy=best["recording_vote_accuracy"],
                  window_macro_f1=best["window_macro_f1"],
                  n_recordings=best["n_recordings"],
                  artifact=os.path.relpath(model_path, T.PROJECT_DIR),
                  report=os.path.relpath(report_path, T.PROJECT_DIR),
                  message=f"saved {best['model']}")

    print("\n  (training-set fit accuracy "
          f"{payload['stats']['train_fit_accuracy'] * 100:.1f}% -- this is NOT "
          "a performance estimate, it is only a sanity check that the model "
          "can represent the data at all.)")

    print("\nTo use it live, set in emg_dataset_tool.py:")
    print("    ENABLE_LIVE_PREDICTION = True")
    print("then run the app and press PREDICT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

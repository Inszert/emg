"""
================================================================================
 TRAIN A GESTURE MODEL
================================================================================

 Sweeps preprocessing choices and classifiers over the recorded dataset,
 reports honest cross-validated scores, and saves the winner for live use in
 emg_dataset_tool.py.

 Run:
     python training/train_model.py                  # full sweep, saves the best model
     python training/train_model.py --quick          # fewer combinations
     python training/train_model.py --no-save        # evaluate only
     python training/train_model.py --person Dano    # restrict to one subject

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
import io
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

# The project root (one level up) holds emg_dataset_tool and emg_app.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
# "bandpass"   -> zero-phase Butterworth, EMG_BANDPASS_HZ[EMG_MODE]. On the
#                 default "filtered" stream that is an 8 Hz LOW-PASS only:
#                 the Myo already band-passed in hardware, and a high-pass
#                 would delete the activation level (CLAUDE.md section 13).
#                 It is a candidate, not a default -- whether smoothing the
#                 envelope helps is measured here like everything else.
REPRESENTATIONS = ["raw", "normalized", "envelope", "bandpass"]

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

# ---- sweep cost ------------------------------------------------------------
# These three turn a sweep that grew to hours into one that fits in minutes.
# What each costs in accuracy is measured, not assumed -- see
# `how_to_fix_long_training.md`.

# Drop windows that re-cover signal another window already covered. This is
# CORRECTNESS first and speed second: 280 of 308 recordings are clips cut from
# the same few free takes, they tile those takes with ~50 % overlap, and 74 of
# them are byte-identical to another clip (docs/EDA.md). Grouping the
# cross-validation by recording uid does not separate them, because every clip
# legitimately owns its uid -- so without this the SAME samples sit in train
# and test. Measured: 5598 -> 2425 windows, and the score falls from 99.7 % to
# 99.0 %, which is the leak being removed rather than accuracy being lost.
DEDUPE_WINDOWS = True

# Fraction of windows the stage A1 SCREEN keeps. The screen only ranks
# preprocessing setups; every score that gets reported or saved is measured at
# 1.0 in stage A2 / stage B.
SCREEN_FRACTION = 0.5

# Cheap probes for the screen. LDA and SVM score a setup in ~0.2-0.3 s each,
# where ExtraTrees-400 costs ~3.4 s no matter how small the data is (per-tree
# overhead, not per-row work).
# Listed in preference order; at most SCREEN_MAX_PROBES of the ones that are
# actually in the spec get used. The extra names are fallbacks for a spec that
# disabled the first choices, NOT four probes to run -- screening with all four
# would double the screen's cost for a ranking that two already give.
SCREEN_PROBES = ("LDA-shrinkage", "SVM-rbf-C10", "LDA-svd", "LogReg")
SCREEN_MAX_PROBES = 2

# How many setups reach the full-data probe, and how many of those are drawn
# at random rather than taken from the screen's ranking.
A2_SHORTLIST = 32
A2_RANDOM = 8

# ---- what the sweep SELECTS on --------------------------------------------
# "auto"      group the folds by SUBJECT as soon as there are two or more,
#             otherwise by recording. This is the honest default: with
#             several wearers, a winner chosen on recording-grouped folds was
#             chosen for within-subject fit, because every subject sits in
#             both train and test. A setup that leans on one person's
#             absolute amplitudes wins those folds and then fails on the next
#             wearer -- which is exactly what collecting more people is for.
# "recording" always group by recording uid (the pre-2026-10-15 behaviour).
# "subject"   always group by subject; fails over to recording with only one.
#
# The per-recording vote is unaffected: it aggregates predictions by
# recording whatever the split was.
SELECTION_GROUPING = "auto"

# How many CV folds may be fitted at once, for estimators that are themselves
# single-threaded (SVC, LDA, LogReg). Leave one core for the rest of the
# machine -- the GUI runs the sweep as a subprocess while still streaming EMG.
FOLD_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# A progress event every this many candidates. Was 5, which on the expensive
# stages is minutes of silence between lines and made the GUI's readout lag; a
# line is a few hundred bytes, so there is no reason to thin them out.
PROGRESS_EVERY = 1


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


def select_recordings(recs, include):
    """
    Keep only the recordings in the spec's `data.include` list of
    "person/session" pairs. An empty list keeps everything.
    """
    if not include:
        return list(recs)
    wanted = set(include)
    return [r for r in recs if f"{r.person_id}/{r.session_id}" in wanted]


def load_recordings(root, persons=None, sessions=None, exclude_codes=(),
                    include=None, exclude_persons=()):
    """
    Load recordings, dropping the ones whose quality codes are excluded.

    `exclude_persons` leaves those people's recordings out entirely (they are
    not reported in `dropped`: they were never candidates). It is an explicit
    argument, never read from the configuration, so the research tools that
    share this loader keep measuring the whole dataset.
    """
    recs = select_recordings(
        T.load_dataset(root, persons=persons, sessions=sessions,
                       stage="prefer_trimmed", prefer="npz"), include)
    if exclude_persons:
        out = set(exclude_persons)
        recs = [r for r in recs if r.person_id not in out]
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


def source_timeline(rec):
    """
    Where this recording's samples sit on the timeline they were cut from.

    Returns `(source_id, start, end)` in whatever unit that source counts in,
    or None when the recording cannot be placed on a shared timeline.

    This exists because most of the dataset is NOT independent repetitions.
    Clips cut from one free take carry `parent_take_uid` plus the exact
    `take_start_index`/`take_end_index` they were copied from, and those spans
    overlap each other heavily -- measured 2.23x more clip-seconds than unique
    signal-seconds. Two clips that overlap in the take share real samples, so
    splitting them across cross-validation folds puts the SAME samples in
    train and test. `dedupe_windows` uses this to stop that.

    Recordings that were never cut from a take are placed on their session's
    wall clock instead, because two `label_on_graph` selections taken from the
    same live stream can overlap in exactly the same way.
    """
    meta = rec.meta or {}
    take = meta.get("parent_take_uid")
    if take:
        try:
            return (f"take:{take}", int(meta["take_start_index"]),
                    int(meta["take_end_index"]))
        except (KeyError, TypeError, ValueError):
            pass
    # Anything else is NOT placed, deliberately.
    #
    # An earlier version fell back to the recording's wall clock. That is a
    # guess, not provenance: tests/test_predict.py writes all 36 of its
    # recordings with the same timestamp array, so every recording of a label
    # looked like it occupied the same instant, collapsed into one bucket,
    # and the sweep had nothing left to evaluate. Any batch-written or
    # imported dataset would behave the same way.
    #
    # `parent_take_uid` with exact sample indices is different in kind -- it
    # is a record of where the samples came from, written by EXTRACT CLIP.
    # Only that is trusted. Unplaced recordings keep all their windows, which
    # is the safe direction: an independent recording is not a duplicate.
    return None


def dedupe_windows(rec_windows, keep_report=None):
    """
    Drop windows that re-cover signal an earlier window already covered.

    `rec_windows` is a list of `(rec, windows, win, step, n_out)`. The result
    is the same list with a boolean mask per entry.

    The rule is a grid, not a pairwise comparison: a window's CENTRE is mapped
    onto its source timeline and quantised to the step size, and the first
    window to claim a bucket keeps it. That is exactly the set of windows you
    would get by windowing each source timeline ONCE at the configured step,
    which is the honest amount of data -- so coverage of the unique signal is
    preserved in full, and only the duplicates go.

    It is deliberately not "drop the overlapping clip": dropping whole clips
    also throws away the parts that overlapped nothing. Measured on this
    dataset, clip-level dropping left 108 of 308 recordings, while this leaves
    every recording contributing its non-duplicate windows.

    Recordings with no placeable source timeline keep all their windows.
    """
    seen = set()
    out = []
    n_in = n_out_kept = 0
    for rec, windows, win, step, n_out in rec_windows:
        n_in += int(windows.shape[0])
        place = source_timeline(rec)
        if place is None or n_out <= 0:
            out.append((rec, windows, np.ones(windows.shape[0], dtype=bool)))
            n_out_kept += int(windows.shape[0])
            continue
        source_id, s0, s1 = place
        span = float(s1 - s0)
        if span <= 0:
            out.append((rec, windows, np.ones(windows.shape[0], dtype=bool)))
            n_out_kept += int(windows.shape[0])
            continue
        # Source units per resampled sample, and per window step.
        per_sample = span / float(n_out)
        bucket_size = max(1e-9, per_sample * float(step))
        mask = np.zeros(windows.shape[0], dtype=bool)
        for i in range(windows.shape[0]):
            centre = s0 + (i * step + win * 0.5) * per_sample
            key = (source_id, rec.label, int(math.floor(centre / bucket_size)))
            if key in seen:
                continue
            seen.add(key)
            mask[i] = True
        out.append((rec, windows, mask))
        n_out_kept += int(mask.sum())
    if keep_report is not None:
        keep_report["windows_before"] = n_in
        keep_report["windows_after"] = n_out_kept
    return out


def feature_vector(window, representation, include_freq, feature_set,
                   calibration=None):
    """
    One window -> one feature vector.

    This function is the contract between training and inference: the live
    predictor calls the very same code path, so a change here cannot silently
    desynchronise the two.

    `calibration` is the RESOLVED calibration of the wearer this window came
    from (T.resolve_calibration), or None. It is passed through rather than
    applied here for the same reason: the one place that applies it is
    T.model_feature_vector, which the live predictor calls too.
    """
    return T.model_feature_vector(window, representation, feature_set,
                                  include_freq, MODEL_FS, calibration)


def feature_names_for(feature_set, include_freq):
    return T.model_feature_names(feature_set, include_freq)


def build_dataset(recs, window_s, overlap, representation, include_freq,
                  feature_set="level", dedupe=True, screen_fraction=1.0,
                  report=None, calibration=None):
    """
    Recordings -> (features, y, groups, labels, meta).

    `groups` is the recording uid, which is the unit the cross-validation
    splits on.

    `dedupe` drops windows that re-cover signal another window already
    covered -- see `dedupe_windows` for why that is correctness, not a
    speed-up: without it, overlapping clips cut from the same take put the
    same samples in train and test.

    `screen_fraction` < 1 keeps only that fraction of the surviving windows,
    evenly spaced per recording. It exists for the preprocessing SCREEN only
    (stage A1), where the job is to rank setups, not to produce a number worth
    quoting. Every score that gets reported is measured at 1.0.

    `calibration` is a TABLE, person id -> resolved calibration
    (T.calibration_table). Each recording's windows are calibrated with the
    numbers of the wearer who produced them, and a wearer with no calibration
    is left uncalibrated rather than borrowing someone else's gain. An empty
    table is the uncalibrated pipeline exactly as it was.
    """
    win = max(4, int(round(window_s * MODEL_FS)))
    step = max(1, int(round(win * (1.0 - overlap))))
    labels = sorted({r.label for r in recs})
    index = {name: i for i, name in enumerate(labels)}

    # Pass 1: resample and window. Measured at 0.03 s for the whole dataset,
    # against 1-3.5 s to featurise it, so the loop is split in two rather than
    # featurising inside it -- that way the dedup mask is known BEFORE the
    # expensive part runs and the dropped windows cost nothing.
    staged = []
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
        staged.append((rec, windows, win, step, n_out))

    if not staged:
        return None

    if dedupe:
        masked = dedupe_windows(staged, keep_report=report)
        # The floor: dedup must never leave a class too thin to cross-
        # validate. If it would, keep the duplicates for this setup and say
        # so -- a measurable leak is better than a sweep that cannot run, and
        # silently returning None here would look like "this setup produced
        # no windows".
        before = Counter()
        after = Counter()
        for rec, windows, mask in masked:
            before[rec.label] += int(windows.shape[0])
            after[rec.label] += int(mask.sum())
        starved = sorted(
            lab for lab in before
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

    # Pass 2: featurise what survived.
    feats, ys, groups = [], [], []
    for rec, windows, mask in masked:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if screen_fraction < 1.0 and idx.size > 1:
            # Evenly spaced, not random: a contiguous random draw would
            # over-sample whichever part of the gesture the RNG favoured.
            n_keep = max(1, int(round(idx.size * screen_fraction)))
            idx = idx[np.linspace(0, idx.size - 1, n_keep).round().astype(int)]
            idx = np.unique(idx)
        cal = (calibration or {}).get(rec.person_id)
        for i in idx:
            feats.append(feature_vector(windows[i], representation,
                                        include_freq, feature_set, cal))
            ys.append(index[rec.label])
            groups.append(rec.uid)

    if not feats:
        return None
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(ys, dtype=int),
            np.asarray(groups, dtype=object),
            labels,
            {"window_samples": win, "step": step, "window_s": window_s,
             "deduped": bool(dedupe),
             "calibrated_subjects": sorted(calibration or {}),
             "screen_fraction": float(screen_fraction)})


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


def evaluate(model, X, y, groups, labels, n_splits=CV_SPLITS, on_fold=None,
             split_groups=None, grouping="recording"):
    """
    Grouped stratified cross-validation.

    Returns per-window accuracy/F1, per-recording majority-vote accuracy, the
    summed confusion matrix and the per-fold spread.

    `groups` is always the RECORDING id -- it is what the per-recording vote
    aggregates on. `split_groups` is what the folds are split on, and
    defaults to `groups`; pass the per-window subject id to get a
    subject-independent score instead. The two are deliberately separate:
    with several wearers the split must hold out a whole person while the
    vote still has to be counted per recording.

    `on_fold(done, total)` is called after each fold when given, for progress
    reporting; passing it forces sequential folds so the counts are ordered.
    """
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
        "grouping": grouping,
        "window_accuracy": float(np.mean(window_acc)),
        "window_accuracy_std": float(np.std(window_acc)),
        "window_macro_f1": float(np.mean(window_f1)),
        "recording_vote_accuracy": (vote_correct / vote_total
                                    if vote_total else 0.0),
        "n_recordings_voted": vote_total,
        "fold_accuracies": [float(a) for a in fold_acc],
        "confusion_matrix": cm.tolist(),
    }


def person_vector(recs, groups):
    """
    Per-window subject id, aligned with `groups`.

    Built through a dict rather than a scan per window: the old inline
    `next(r.person_id for r in recs if r.uid == g)` was O(windows x
    recordings), which is 2.7 million comparisons at today's size and grows
    quadratically as subjects are added.
    """
    person_of = {r.uid: r.person_id for r in recs}
    return np.array([person_of.get(g, "?") for g in groups], dtype=object)


def subject_grouping_viable(recs, min_recordings=3):
    """
    Can the folds really be split by subject? `(ok, reason)`.

    Counting subjects is not enough. A subject-grouped split holds out one
    whole person, so for every fold to be trainable AND scoreable:

      * at least two subjects must carry enough recordings to be a fold at
        all -- a person with one recording is not a fold, it is an outlier;
      * every LABEL must appear for at least two subjects. A label owned by a
        single person vanishes from training in the fold that holds that
        person out, and the score for it is meaningless.

    When this fails the sweep must fall back to recording-grouped folds and
    say so, rather than producing a cross-validation where nothing can be
    evaluated.
    """
    per_subject = defaultdict(set)
    counts = Counter()
    for rec in recs:
        per_subject[rec.person_id].add(rec.label)
        counts[rec.person_id] += 1
    if len(counts) < 2:
        return False, f"only {len(counts)} subject(s) on disk"

    usable = sorted(s for s, n in counts.items() if n >= min_recordings)
    if len(usable) < 2:
        thin = ", ".join(f"{s} ({counts[s]})" for s in sorted(counts))
        return False, (f"only {len(usable)} subject(s) have at least "
                       f"{min_recordings} recordings -- counts: {thin}")

    subjects_per_label = Counter()
    for subject, labels_seen in per_subject.items():
        if counts[subject] < min_recordings:
            continue
        for label in labels_seen:
            subjects_per_label[label] += 1
    all_labels = {r.label for r in recs}
    lonely = sorted(lab for lab in all_labels
                    if subjects_per_label[lab] < 2)
    if lonely:
        return False, (f"{', '.join(lonely)} recorded by fewer than 2 "
                       f"subjects, so that label would vanish from training "
                       f"when its only subject is held out")
    return True, (f"{len(usable)} subjects with >= {min_recordings} "
                  f"recordings each, every label present for at least 2")


def resolve_grouping(n_subjects, requested=None):
    """
    Which grouping the sweep scores on, and a sentence saying why.

    Returns `(grouping, note)` where grouping is "subject" or "recording".
    """
    requested = requested or SELECTION_GROUPING
    if requested == "recording":
        return "recording", ("grouped by recording uid (configured); with "
                             "several subjects this measures within-subject "
                             "fit, not robustness across wearers")
    if n_subjects < 2:
        return "recording", ("grouped by recording uid -- only one subject, "
                             "so a subject-independent split is impossible")
    return "subject", (f"grouped by SUBJECT across {n_subjects} wearers: the "
                       "held-out person's data is never in training, so the "
                       "winner is chosen for robustness across people rather "
                       "than for within-subject fit")


def dc_only_control(model, X, y, groups, labels, feature_names,
                    split_groups=None):
    """
    Re-score the winner using ONLY the 8 per-channel mean levels.

    docs/EDA.md section 3 established that on this dataset a classifier given
    nothing but the 8 channel means scores 0.993, against 0.187 for the
    majority class -- because every active gesture is one 50-second take, so
    "which block of time is this" is enough to name the label. CLAUDE.md
    section 8f therefore makes this control a rule: a high score means nothing
    until you know whether the shape, spectrum and timing bought anything over
    the DC levels alone.

    Reported, never acted on. It does not change what is trained or saved.

    `split_groups` must be the SAME split key the winner was scored on. It is
    a parameter because getting this wrong makes the control unreadable
    rather than merely imprecise: scoring the control recording-grouped while
    the winner is scored subject-grouped compares two different protocols,
    and the subtraction then prints a negative "gain" and trips the
    "adds almost nothing" warning for no reason. That happened -- see the
    caveat in CLAUDE.md section 8 -- which is why the key is passed in now
    instead of defaulting to `groups`.
    """
    cols = [i for i, n in enumerate(feature_names)
            if n.endswith("_mean") and not n.endswith("_log_mean")]
    if len(cols) < 2 or X.shape[1] <= len(cols):
        return None
    scores = evaluate(clone(model), X[:, cols], y, groups, labels,
                      split_groups=split_groups)
    if scores is None:
        return None
    return {"n_features": len(cols),
            "feature_names": [feature_names[i] for i in cols],
            # Which protocol this number is from, stored next to the number,
            # so a report can never be read as comparing like with like when
            # it is not.
            "grouping": "recording" if split_groups is None else "subject",
            "window_accuracy": scores["window_accuracy"],
            "recording_vote_accuracy": scores["recording_vote_accuracy"]}


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


def feature_space_projection(X, y, labels, groups, feature_names,
                             take_of=None, max_points=6000):
    """
    A 2-D PCA of the feature matrix, plus everything needed to place a LIVE
    vector on the same axes.

    PCA and NOT LDA, deliberately. LDA picks the axes that best separate the
    labels, so its scatter looks cleanly separated even when the separation
    comes from electrode drift rather than from the gesture -- it would make
    every model look good by construction, which is exactly what safety rule
    11 forbids. PCA never sees `y`, so what it draws is the structure that is
    actually in the features.

    The returned `mean`/`scale`/`components` are the whole transform, so
    `DecisionPanel.project()` can map one live feature vector into the same
    space without refitting anything and without a second code path.

    `take` is the index of each window's parent take. Colouring by it is how
    the class-take confound in docs/EDA.md section 2 becomes visible: when the
    clusters line up with takes rather than with labels, the model is reading
    the block.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 3 or d < 2:
        return None

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    #  A constant feature has no variance to standardise by. Leaving it at 1
    #  keeps it at zero after centring instead of producing inf.
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = (X - mean) / scale

    #  Economy SVD of the standardised matrix: the right singular vectors ARE
    #  the principal axes, and this avoids forming a d x d covariance matrix.
    try:
        _, sing, vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:                          # pragma: no cover
        return None
    if sing.size < 2:
        return None

    components = vt[:2]
    var = (sing ** 2) / max(1, n - 1)
    total = float(var.sum()) or 1.0
    coords = Z @ components.T

    #  Per-window mean level, by NAME like the trail's `chan` block, so the
    #  view can colour by amplitude -- which is what makes the R4 void
    #  visible. Absent for feature sets that do not carry the channel means.
    wanted = [f"CH{c}_mean" for c in range(1, len(T.CHANNEL_NAMES) + 1)]
    level = None
    names = list(feature_names or [])
    if all(w in names for w in wanted):
        cols = [names.index(w) for w in wanted]
        level = np.asarray(X[:, cols], dtype=np.float32).mean(axis=1)

    take_names, take_idx = [], None
    if take_of:
        order, seen = [], {}
        for gid in groups.tolist():
            src_id = take_of.get(gid) or f"(unplaced) {gid}"
            if src_id not in seen:
                seen[src_id] = len(order)
                order.append(src_id)
        take_names = order
        take_idx = np.array([seen[take_of.get(g) or f"(unplaced) {g}"]
                             for g in groups.tolist()], dtype=np.int16)

    #  The scatter is a diagnostic, not the dataset. Thinning keeps the panel
    #  file small and the plot responsive; it is evenly spaced rather than
    #  random so every recording keeps representation.
    keep = np.arange(n)
    if n > max_points:
        keep = np.unique(np.linspace(0, n - 1, max_points).astype(int))

    return {
        "kind": "pca",
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "components": components.astype(np.float32),
        "coords": coords[keep].astype(np.float32),
        "truth": np.asarray(y, dtype=np.int16)[keep],
        "level": None if level is None else level[keep],
        "take": None if take_idx is None else take_idx[keep],
        "take_names": take_names,
        "rows": keep.astype(np.int32),
        "explained": [float(var[0] / total), float(var[1] / total)],
        "n_points": int(len(keep)),
        "n_windows": int(n),
        "n_features": int(d),
        "note": ("unsupervised PCA of the standardised feature matrix; the "
                 "axes were chosen without seeing the labels, so any "
                 "separation shown is really in the features"),
    }


def build_decision_panel(spec, quick, X, y, groups, labels, best, meta,
                         take_of=None, calibration_mode="off"):
    """
    Every candidate's fitted model plus its out-of-fold decision trail.

    This is what the DECISION TRAIL view in MODEL STATS runs on. It is a
    SIDE file: `gesture_model.joblib` goes on keeping exactly one estimator,
    because that is what ships to the robot, and nothing in the live
    prediction path ever loads this one.

    Two different things come out of here and they must not be confused:

      * `members` -- each candidate fitted on the WHOLE dataset. These are
        what run on a live window, so the view can show what each algorithm
        would have said about the gesture you just made. Fitted on
        everything means their agreement on live data is NOT a performance
        estimate, and the panel is labelled accordingly.
      * `cv` and `trail` -- out-of-fold probabilities from the same grouped
        cross-validation the sweep uses. These are the honest numbers. They
        are kept per window, so the view can be scrubbed through the real
        dataset when nothing is streaming.

    Every candidate is scored on the WINNER's setup rather than on its own
    best one. That is the whole point: one feature vector, seven opinions,
    directly comparable. It also means these numbers are usually lower than
    `best_per_model` in the sweep report, where each model got to pick its
    own preprocessing.
    """
    models = build_models_from_spec(spec, quick, for_scoring=False)
    vote_name = next((n for n in models if n.startswith("Vote-")), None)
    members = [n for n in models if n != vote_name]
    if not members:
        return None

    counts = Counter(y.tolist())
    n_groups = len(set(groups.tolist()))
    n_splits = int(min(CV_SPLITS, min(counts.values()), n_groups))
    if n_splits < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    folds = list(cv.split(X, y, groups))
    order = members + ([vote_name] if vote_name else [])
    n, k = len(y), len(labels)

    proba = {name: np.full((n, k), np.nan, dtype=np.float32) for name in order}
    fold_of = np.full(n, -1, dtype=np.int16)

    for index, (train_idx, test_idx) in enumerate(folds):
        fold_of[test_idx] = index
        for name in order:
            fitted = clone(models[name])
            fitted.fit(X[train_idx], y[train_idx])
            proba[name][test_idx] = fitted.predict_proba(X[test_idx])
        print(f"    trail fold {index + 1}/{len(folds)} done")

    for name in order:
        if np.isnan(proba[name]).any():                    # pragma: no cover
            print(f"    {name} left gaps in the trail; panel not written")
            return None

    #  Soft voting is the unweighted mean of the members' probabilities.
    #  Checked rather than asserted in prose, and the result is stored so the
    #  view can show the arithmetic instead of claiming it.
    fusion_err = None
    if vote_name:
        mean_members = np.mean([proba[m] for m in members], axis=0)
        fusion_err = float(np.max(np.abs(mean_members - proba[vote_name])))
        print(f"    soft vote vs mean of members: max|diff| = {fusion_err:.2e}")

    uid_index, uid_order = {}, []
    for gid in groups.tolist():
        if gid not in uid_index:
            uid_index[gid] = len(uid_order)
            uid_order.append(gid)
    rec_of = np.array([uid_index[g] for g in groups.tolist()], dtype=np.int32)

    cv_stats, recordings = {}, []
    for name in order:
        pred = np.argmax(proba[name], axis=1)
        cm = np.zeros((k, k), dtype=int)
        for truth, guess in zip(y.tolist(), pred.tolist()):
            cm[truth, guess] += 1
        votes, truth_of = defaultdict(list), {}
        for pos, gid in enumerate(groups.tolist()):
            votes[gid].append(int(pred[pos]))
            truth_of[gid] = int(y[pos])
        correct = sum(Counter(v).most_common(1)[0][0] == truth_of[g]
                      for g, v in votes.items())
        cv_stats[name] = {
            "window_accuracy": float(np.mean(pred == y)),
            "recording_accuracy": float(correct / len(votes)),
            "mean_confidence": float(np.mean(np.max(proba[name], axis=1))),
            "confusion_matrix": cm.tolist(),
            "n_windows_wrong": int(np.sum(pred != y)),
        }

    for gid in uid_order:
        rows = np.flatnonzero(groups == gid)
        truth = int(y[rows[0]])
        entry = {"uid": gid, "label": labels[truth], "truth": truth,
                 "n_windows": int(len(rows)), "fold": int(fold_of[rows[0]]),
                 "models": {}}
        for name in order:
            pred = np.argmax(proba[name][rows], axis=1)
            tally = Counter(pred.tolist())
            winner = tally.most_common(1)[0][0]
            entry["models"][name] = {
                "vote": int(winner),
                "n_for_truth": int(tally.get(truth, 0)),
                "correct": bool(winner == truth),
            }
        recordings.append(entry)

    #  The per-channel level, when the winning feature set carries it, so the
    #  idle view can show the input a recorded window presented. Taken by
    #  NAME rather than by position, because the column order differs between
    #  feature sets.
    names = feature_names_for(best["feature_set"], best["freq_features"])
    wanted = [f"CH{c}_mean" for c in range(1, len(T.CHANNEL_NAMES) + 1)]
    chan = None
    if all(w in names for w in wanted):
        cols = [names.index(w) for w in wanted]
        chan = np.asarray(X[:, cols], dtype=np.float32)

    print("    projecting the feature space for MODEL STATS...")
    space = feature_space_projection(X, y, labels, groups, names, take_of)
    if space is None:
        print("      skipped (too few windows or features to project)")
    else:
        print(f"      PCA on {space['n_features']} features: "
              f"PC1 {space['explained'][0] * 100:.1f} %, "
              f"PC2 {space['explained'][1] * 100:.1f} % of the variance, "
              f"{space['n_points']} points drawn")

    print("    fitting each candidate on the whole dataset for live use...")
    fitted_members = {}
    for name in order:
        fitted = clone(models[name])
        fitted.fit(X, y)
        fitted_members[name] = fitted

    return {
        "kind": "decision_panel",
        "created_iso": T.iso_time(time.time()),
        "labels": labels,
        "member_order": members,
        "vote_name": vote_name,
        "vote_members": members if vote_name else [],
        "members": fitted_members,
        "fitted_on": "the whole dataset (not a held-out estimate)",
        "winner": best["model"],
        "setup": {
            "window_s": best["window_s"], "overlap": best["overlap"],
            "representation": best["representation"],
            "feature_set": best["feature_set"],
            "freq_features": best["freq_features"],
            "exclude": best["exclude"],
            "window_samples": meta["window_samples"],
            "model_fs": MODEL_FS,
            #  The members were fitted on features built on this calibration
            #  scale, so a panel is only comparable with a model built the
            #  same way -- matches() checks it.
            "calibration_mode": calibration_mode,
            "n_features": int(X.shape[1]),
            "feature_names": names,
        },
        "fusion": {
            "rule": "unweighted mean of member predict_proba, then argmax",
            "max_abs_diff_vs_mean": fusion_err,
            "verified": bool(fusion_err is not None and fusion_err < 1e-6),
        },
        "protocol": {
            "cv": "StratifiedGroupKFold grouped by recording uid",
            "n_splits": n_splits, "random_state": RANDOM_STATE,
            "note": ("every probability in `trail` is out-of-fold: the "
                     "window's recording was not in the training set of the "
                     "fold that scored it"),
        },
        "cv": cv_stats,
        "recordings": recordings,
        "trail": {
            "proba": {name: proba[name] for name in order},
            "truth": np.asarray(y, dtype=np.int16),
            "fold": fold_of,
            "rec": rec_of,
            "chan": chan,
        },
        "space": space,
        "n_windows": int(n),
        "n_recordings": len(uid_order),
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
    parser.add_argument("--calibration", default=None,
                        choices=list(T.CALIBRATION_MODES),
                        help="apply each wearer's stored per-wearer "
                             "calibration before featurising, in this mode. "
                             "Default: whatever CALIBRATION_MODE is set to in "
                             "emg_dataset_tool.py (\"off\" out of the box). A "
                             "calibrated model can only be used live by a "
                             "wearer who also has a calibration, and the "
                             "payload records the mode so the app can refuse "
                             "rather than predict from the wrong scale.")
    parser.add_argument("--top", type=int, default=12,
                        help="how many sweep rows to print")
    parser.add_argument("--no-panel", action="store_true",
                        help="skip gesture_panel.joblib, the side file that "
                             "lets MODEL STATS show every candidate's live "
                             "decision. The app runs fine without it; only "
                             "the DECISION TRAIL view goes away.")
    args = parser.parse_args(argv)

    print("=" * 78)
    print(" EMG GESTURE MODEL TRAINING")
    print("=" * 78)

    # ---- data ---------------------------------------------------------------
    # The spec is read first because it also says WHICH data to train on and
    # whether calibration is applied to it.
    spec, spec_source = resolve_spec(args.spec)
    data_sel = spec.get("data") or T.default_data_selection()
    include = data_sel["include"]

    all_recs = select_recordings(
        T.load_dataset(args.root, persons=args.person,
                       stage="prefer_trimmed", prefer="npz"), include)
    if not all_recs:
        print(f"\nNo recordings found under {args.root}"
              + (f" for the selected data {include}." if include else "."))
        return 1
    if include:
        print(f"\nTraining data  : {len(include)} selected folder(s) -- "
              f"{', '.join(include)}")

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

    # ---- per-wearer calibration -------------------------------------------
    # Optional, off unless asked for, and resolved ONCE per wearer here so the
    # per-window cost is a subtract and a divide. A wearer with no usable
    # calibration stays uncalibrated: borrowing another person's gain would
    # be worse than applying none.
    cal_requested = args.calibration or T.CALIBRATION_MODE or "off"
    if data_sel["calibration"] != T.DATA_CALIBRATION_DEFAULT:
        # The spec's own choice beats the app-wide setting the GUI passes.
        cal_requested = data_sel["calibration"]
        print(f"\nCalibration    : set by the model spec ({cal_requested})")
    #  Every training wearer is calibrated from their own recordings (one
    #  method, one scale -- T.training_calibrations), falling back to the
    #  saved record, then to the mean of the others. In memory only.
    cal_table, cal_filled, cal_records = T.training_calibrations(
        args.root, cal_requested, all_recs, people)
    cal_missing = [p for p in people if p not in cal_table]
    #  The payload must record what was DONE, not what was asked for. With a
    #  mode set but no usable calibration for anybody, nothing is applied --
    #  and stamping the payload "affine" anyway would make LivePredictor
    #  refuse to load a model whose features are plain device counts. The
    #  requested mode is kept beside it so the discrepancy is visible rather
    #  than silently resolved.
    cal_mode = cal_requested if cal_table else "off"
    print(f"\nCalibration    : mode {cal_requested}")
    if cal_requested != "off" and not cal_table:
        print("                 NO WEARER HAS A USABLE CALIBRATION -- none "
              "was applied, and the")
        print("                 model is recorded as uncalibrated so it can "
              "still be loaded.")
    if cal_requested == "off":
        print("                 not applied -- features are in device counts, "
              "as every earlier run")
    else:
        for person in people:
            record = cal_records.get(person)
            if person in cal_filled:
                print(f"                 {person:<10} {cal_filled[person]}")
            elif person in cal_table:
                print(f"                 {person:<10} "
                      f"{T.describe_calibration(record, cal_mode)}")
            elif record is None:
                print(f"                 {person:<10} NO CALIBRATION -- "
                      f"left in device counts")
            else:
                _ok, _why = T.validate_calibration(record)
                print(f"                 {person:<10} REFUSED -- "
                      f"{_why[0] if _why else 'invalid'}")
        if cal_missing and cal_table:
            print("\n" + "!" * 78)
            print(" MIXED SCALES -- some wearers are calibrated and some are "
                  "not, so the")
            print(" feature space is not one scale. Calibrate everyone or "
                  "nobody before")
            print(f" reading these scores. Uncalibrated: "
                  f"{', '.join(cal_missing)}")
            print("!" * 78)
        if cal_table and len(people) > 1:
            print("\n  NOTE: a calibrated subject-grouped score is NOT "
                  "zero-shot. The held-out")
            print("        wearer's own signal set their gain, so the number "
                  "answers \"how well")
            print("        does this work on a new wearer WHO CALIBRATES "
                  "FIRST\" -- which is a")
            print("        real deployment, but a different question from the "
                  "uncalibrated one.")
            print("        The uncalibrated run is the comparison; "
                  "--calibration off gives it.")

    # ---- sweep --------------------------------------------------------------
    windows = WINDOW_SECONDS[:2] if args.quick else WINDOW_SECONDS
    overlaps = OVERLAPS[:1] if args.quick else OVERLAPS
    representations = REPRESENTATIONS[:2] if args.quick else REPRESENTATIONS
    freq_options = [True] if args.quick else INCLUDE_FREQ_FEATURES
    feature_sets = FEATURE_SETS[:2] if args.quick else FEATURE_SETS
    exclude_sets = ({"clean": EXCLUDE_SETS["clean"]} if args.quick
                    else EXCLUDE_SETS)
    # PINNED SETUP: a spec can name ONE preprocessing setup (written by
    # nn_ml_comb_test.py once a combination has been measured). The sweep
    # then trains exactly that setup instead of re-choosing one the chosen
    # members were never scored on -- and runs in minutes, not a quarter of
    # an hour.
    pinned = spec.get("setup")
    if pinned:
        windows, overlaps = [pinned["window_s"]], [pinned["overlap"]]
        representations = [pinned["representation"]]
        freq_options = [pinned["freq_features"]]
        feature_sets = [pinned["feature_set"]]
        exclude_sets = {pinned["exclude"]: EXCLUDE_SETS[pinned["exclude"]]}
        print(f"\nSetup          : PINNED by the spec -- "
              f"{pinned['representation']}/{pinned['feature_set']}"
              f"{' +freq' if pinned['freq_features'] else ''}, "
              f"{pinned['window_s']:g} s @ {pinned['overlap']:g}, "
              f"data '{pinned['exclude']}'")
        if spec.get("chosen_by"):
            print(f"                 chosen by {spec['chosen_by']}")
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

    print(f"\nModel spec     : {spec_source}")
    print(f"                 {T.describe_model_spec(spec)}")
    # for_scoring: inside the sweep only predict() is used, so the SVMs skip
    # Platt scaling (5.7x faster, identical predictions -- see
    # _svm_probability). The SAVED model is rebuilt below without this.
    models = build_models_from_spec(spec, args.quick, for_scoring=True)
    print(f"                 candidates: {', '.join(models)}")

    # Resolve the exclude filters now, not inside the sweep: two filters that
    # select the same recordings are the same sweep, and the progress total
    # has to know that before it is printed.
    exclude_plan = []
    collapsed_excludes = {}
    _seen_recordings = {}
    for exclude_name, codes in exclude_sets.items():
        recs_f, dropped_f = load_recordings(args.root, args.person, None, codes,
                                            include=include)
        if len(recs_f) < 12:
            print(f"  [{exclude_name}] only {len(recs_f)} recordings left, "
                  f"skipping")
            continue
        fingerprint = frozenset(r.uid for r in recs_f)
        twin = _seen_recordings.get(fingerprint)
        if twin is not None:
            collapsed_excludes[exclude_name] = twin
            print(f"  [{exclude_name}] selects the same {len(recs_f)} "
                  f"recordings as [{twin}] -- not swept again")
            continue
        _seen_recordings[fingerprint] = exclude_name
        print(f"  [{exclude_name}] {len(recs_f)} recordings "
              f"({len(dropped_f)} dropped)")
        exclude_plan.append((exclude_name, recs_f))
    if not exclude_plan:
        print("\nNo exclude filter left enough recordings to train on.")
        return 1

    # The real setup count: the cartesian product MINUS the level/no-freq
    # combination the sweep skips, times the filters that survived folding.
    setups_per_filter = sum(
        1
        for _w in windows for _o in overlaps for _r in representations
        for _fs in feature_sets for _f in freq_options
        if not (_fs == "level" and not _f and True in freq_options))
    n_setups_real = setups_per_filter * len(exclude_plan)
    total = n_setups_real * len(models)
    # Stage A probes only a few models per setup; stage B then runs every
    # model on the few best setups. Both are counted up front so the GUI can
    # show one honest percentage instead of restarting at 0 % halfway.
    n_setups = max(1, n_setups_real)
    # The probes are NAMED, so count them -- min(3, len(models)) guessed, and
    # a spec without ExtraTrees made the guess too high, which is one of the
    # two reasons the bar used to stall below 100 %.
    probe_names = [n for n in ("LDA-shrinkage", "ExtraTrees-400", "SVM-rbf-C10")
                   if n in models]
    if not probe_names:
        # A spec with none of the named probes (e.g. one written by
        # nn_ml_comb_test) would otherwise screen with nothing and stop at
        # "Nothing could be evaluated". Probe with its first real model.
        probe_names = [n for n in models if not n.startswith("Vote")][:1]
    n_probes = max(1, len(probe_names))
    n_best = 2 if args.quick else 5
    # A1 screens every setup with the cheap probes; A2 re-probes only the
    # shortlist with all of them; B runs every model on the best few. The
    # total is built from the same numbers the stages actually use, so the
    # GUI's percentage does not jump when a stage ends.
    n_screen = max(1, min(SCREEN_MAX_PROBES,
                          len([n for n in SCREEN_PROBES if n in models]) or 1))
    n_short = max(n_best, 6 if args.quick else A2_SHORTLIST)
    stage_a_total = n_setups * n_screen
    # Stage B skips the models stage A2 already measured on that setup, and
    # every stage B setup comes from A2 -- so it runs the NON-probe models
    # only, not all of them.
    stage_b_total = n_best * max(1, len(models) - len(probe_names))
    # The tail after stage B: contrast score, DC-only control, final fit,
    # decision panel. Reserved so the bar keeps moving through the slowest,
    # previously silent, part of the run.
    tail_total = 4
    grand_total = (stage_a_total + n_short * n_probes
                   + stage_b_total + tail_total)
    emit_progress(0, grand_total, "starting", status="running",
                  n_candidates=grand_total, n_folds=CV_SPLITS,
                  # The stages in order, with how many units each holds. The
                  # GUI times each stage on its own, because a unit costs very
                  # different amounts in different stages.
                  phases=[["stage A1 screen", stage_a_total],
                          ["stage A2 probe", n_short * n_probes],
                          ["stage B", stage_b_total],
                          ["tail", tail_total]],
                  message=(f"{len(models)} candidates x "
                           f"{total // max(1, len(models))} setups"))

    print(f"\nSweep          : {n_setups} preprocessing setups x "
          f"{len(models)} models = {total} combinations, of which "
          f"{grand_total} are evaluated")
    grouping, grouping_note = resolve_grouping(len(people))
    if grouping == "subject":
        # Counting subjects said yes; check the folds would actually work.
        viable, why = subject_grouping_viable(all_recs)
        if not viable:
            grouping = "recording"
            grouping_note = ("grouped by recording uid -- a subject-"
                             f"independent split is not viable here ({why})")
            print(f"  NOTE: {len(people)} subjects on disk, but splitting by "
                  f"subject is not viable:\n        {why}.\n        Falling "
                  f"back to recording-grouped folds, which is the honest "
                  f"option.\n        This is NOT a cross-wearer score.")
    print(f"Validation     : StratifiedGroupKFold {grouping_note} "
          f"({CV_SPLITS} folds max)")
    if grouping == "subject":
        print("                 (the per-recording vote is still counted per "
              "recording)")
    print()

    # Three stages, because the cost is not spread evenly over them.
    #
    # Measured on this dataset: featurising a setup costs ~1-3.5 s and scales
    # with the window count, LDA and SVM each score one in ~0.2-0.3 s, and
    # ExtraTrees-400 costs ~3.4 s however small the data gets (per-tree
    # overhead, not per-row work). Running the expensive probe on all 360
    # setups therefore spent most of the sweep ranking setups that were never
    # going to win.
    #
    #   A1  SCREEN  every setup, deduped and window-subsampled, cheap probes
    #   A2  PROBE   the shortlist, full data, the expensive probe as well
    #   B           every model on the few best setups, full data
    #
    # Why a shortlist is safe here, measured against the last full-grid sweep
    # (models/reports/sweep_20261014-141319-030.json, where all 360 setups
    # were scored by all three probes): the best per-window accuracy reachable
    # inside a 40-setup shortlist was 99.63 % against 99.65 % for the entire
    # grid, and the per-recording vote was identical. Dozens of setups tie at
    # the top, so paying for all 360 full probes bought 0.02 points.
    #
    # A1 rows are NOT comparable with A2/B rows -- they are measured on fewer
    # windows -- so they are kept out of `results` and reported separately.
    # probe_names was resolved above, with the progress budget.
    screen_names = [n for n in SCREEN_PROBES
                    if n in models][:SCREEN_MAX_PROBES]
    if not screen_names:
        # Nothing cheap in the spec: screen with the first probe and say so,
        # rather than silently screening with a 400-tree forest.
        screen_names = probe_names[:1]
        print(f"Stage A1       : no cheap probe in this spec; screening with "
              f"{screen_names}")
    n_shortlist = max(n_best, 6 if args.quick else A2_SHORTLIST)
    print(f"Stage A1 screen: all setups, {int(SCREEN_FRACTION * 100)}% of "
          f"windows, {screen_names}")
    print(f"Stage A2 probe : best {n_shortlist} setups, full data, "
          f"{probe_names}")
    print(f"Stage B        : all {len(models)} models on the best setups\n")

    print(f"Window dedup   : {'ON' if DEDUPE_WINDOWS else 'OFF'} "
          f"-- overlapping clips share real samples; see dedupe_windows()")

    results = []
    screen_rows = []
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
                              include_freq, feature_set,
                              dedupe=DEDUPE_WINDOWS,
                              calibration=cal_table)
        if built is None:
            return None
        X, y, groups, labels, meta = built
        return X, y, groups, labels, meta, recs_for
    done = 0
    t0 = time.perf_counter()

    # `exclude_plan` was resolved above, with filters that select identical
    # recordings already folded together -- on this dataset "all", "no_errors"
    # and "clean" all keep the same 308, so 360 setups collapse to 120. The
    # axis is not deleted: on a dataset with real quality flags it still does
    # its job, and the report records what was folded.
    for exclude_name, recs in exclude_plan:

        for window_s in windows:
            for overlap in overlaps:
                for representation in representations:
                  for feature_set in feature_sets:
                    for include_freq in freq_options:
                        # The frequency flag only affects the classic family.
                        # A pinned setup may carry only False, so the
                        # level-family copy is skipped only when its True twin
                        # is going to run.
                        if (feature_set == "level" and not include_freq
                                and True in freq_options):
                            continue
                        key = (exclude_name, window_s, overlap, representation,
                               include_freq, feature_set)
                        dedup_report = {}
                        built = build_dataset(
                            recs, window_s, overlap, representation,
                            include_freq, feature_set,
                            dedupe=DEDUPE_WINDOWS,
                            screen_fraction=SCREEN_FRACTION,
                            report=dedup_report,
                            calibration=cal_table)
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

                        split_by = (person_vector(recs, groups)
                                    if grouping == "subject" else None)
                        for model_name in screen_names:
                            model = models[model_name]
                            done += 1
                            try:
                                scores = evaluate(
                                    model, X, y, groups, labels,
                                    split_groups=split_by, grouping=grouping)
                            except Exception as exc:
                                print(f"    {model_name} failed: {exc}")
                                continue
                            if scores is None:
                                continue
                            screen_rows.append({
                                "stage": "A1",
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
                                "screen_fraction": SCREEN_FRACTION,
                                "windows_before_dedup":
                                    dedup_report.get("windows_before"),
                                "windows_after_dedup":
                                    dedup_report.get("windows_after"),
                                **scores,
                            })
                            if done % PROGRESS_EVERY == 0 or done == grand_total:
                                emit_progress(
                                    done, grand_total, "stage A1 screen",
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
                                    message=(f"screening {model_name} on "
                                             f"{window_s:.2f}s "
                                             f"{representation}/"
                                             f"{feature_set}"))
                            if done % 50 == 0:
                                print(f"    ... {done}/{stage_a_total} "
                                      f"({time.perf_counter() - t0:.0f} s)")

    if not screen_rows:
        print("\nNothing could be evaluated.")
        print(f"  Every candidate returned no score with folds {grouping}-"
              f"grouped.")
        if grouping == "subject":
            print("  The usual cause is a subject whose recordings cannot "
                  "form a fold.\n"
                  "  Set SELECTION_GROUPING = \"recording\" in "
                  "train_model.py to score\n"
                  "  within-subject instead, and read CLAUDE.md section 8h "
                  "for what that costs.")
        else:
            print("  The usual cause is too few recordings per gesture for "
                  f"{CV_SPLITS} folds.")
        return 1

    def setup_key(row):
        return (row["exclude"], row["window_s"], row["overlap"],
                row["representation"], row["freq_features"], row["feature_set"])

    screened = {setup_key(r) for r in screen_rows}
    print(f"\nStage A1 done: {len(screen_rows)} screens over "
          f"{len(screened)} setups in {time.perf_counter() - t0:.0f} s")
    if collapsed_excludes:
        print(f"              ({len(collapsed_excludes)} exclude filter(s) "
              f"folded into another: "
              + ", ".join(f"{k}->{v}" for k, v in collapsed_excludes.items())
              + ")")

    # ---- the shortlist -----------------------------------------------------
    # Rank setups by their mean screen score, then add a few seeded-random
    # ones. The random slice is not decoration: measured on the last
    # full-grid sweep, a purely random 40-setup shortlist matched the whole
    # grid's best per-recording vote exactly, while the cheap probes' own
    # top-20 fell 0.28 points short of it. The cheap probes rank setups well
    # but not perfectly, so the shortlist deliberately does not trust them
    # completely.
    per_setup = defaultdict(list)
    for row in screen_rows:
        per_setup[setup_key(row)].append(
            (row["recording_vote_accuracy"], row["window_accuracy"]))
    ranked = sorted(
        per_setup,
        key=lambda k: (float(np.mean([v for v, _ in per_setup[k]])),
                       float(np.mean([w for _, w in per_setup[k]]))),
        reverse=True)

    n_random = 0 if args.quick else A2_RANDOM
    n_top = max(1, n_shortlist - n_random)
    shortlist = list(ranked[:n_top])
    chosen = set(shortlist)
    rest = [k for k in ranked if k not in chosen]
    if n_random and rest:
        rng = np.random.default_rng(RANDOM_STATE)
        picks = rng.choice(len(rest), size=min(n_random, len(rest)),
                           replace=False)
        shortlist += [rest[int(i)] for i in picks]
    print(f"Stage A2: {len(shortlist)} setups of {len(ranked)} "
          f"({n_top} best by screen + {len(shortlist) - n_top} sampled at "
          f"random, seed {RANDOM_STATE})")

    # ---- stage A2: the shortlist, full data, every probe -------------------
    for key in shortlist:
        rebuilt = dataset_for(key)
        if rebuilt is None:
            continue
        X, y, groups, labels, meta, recs_for = rebuilt
        split_by = (person_vector(recs_for, groups)
                    if grouping == "subject" else None)
        for model_name in probe_names:
            done += 1
            try:
                scores = evaluate(models[model_name], X, y, groups, labels,
                                  split_groups=split_by, grouping=grouping)
            except Exception as exc:
                print(f"    {model_name} failed: {exc}")
                continue
            if scores is None:
                continue
            results.append({
                "stage": "A",
                "model": model_name,
                "exclude": key[0],
                "window_s": key[1],
                "overlap": key[2],
                "representation": key[3],
                "freq_features": key[4],
                "feature_set": key[5],
                "n_windows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_recordings": len(recs_for),
                **scores,
            })
            if done % PROGRESS_EVERY == 0 or done == grand_total:
                emit_progress(
                    done, grand_total, "stage A2 probe",
                    model=model_name, candidate=done,
                    n_candidates=grand_total,
                    window_s=key[1], overlap=key[2], representation=key[3],
                    feature_set=key[5], freq_features=key[4], exclude=key[0],
                    n_windows=int(X.shape[0]), n_features=int(X.shape[1]),
                    n_folds=scores.get("n_splits"), status="running",
                    message=f"{model_name} on the shortlist")

    if not results:
        print("\nNothing survived the shortlist probe.")
        return 1

    print(f"Stage A2 done: {len(results)} full probes in "
          f"{time.perf_counter() - t0:.0f} s")

    # Best setups by the FULL-DATA probe scores, de-duplicated.
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
        split_by = (person_vector(recs, groups)
                    if grouping == "subject" else None)
        print(f"  setup: exclude={key[0]} win={key[1]}s ov={key[2]} "
              f"repr={key[3]} feats={key[5]} -> {X.shape[0]} x {X.shape[1]}")
        for model_name, model in models.items():
            if any(r["model"] == model_name and setup_key(r) == key
                   for r in results):
                continue                      # already measured in stage A
            try:
                scores = evaluate(model, X, y, groups, labels,
                                  split_groups=split_by, grouping=grouping)
            except Exception as exc:
                print(f"    {model_name} failed: {exc}")
                continue
            if scores is None:
                continue
            done += 1
            if done % PROGRESS_EVERY == 0 or done == grand_total:
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
    # `voting.always`: the ensemble is saved even if a member scored higher.
    # Documented since the spec existed, but the sweep never honoured it --
    # the winner was always simply the top row.
    if (spec.get("voting") or {}).get("always"):
        votes = [r for r in results if str(r["model"]).startswith("Vote-")]
        if votes:
            results.remove(votes[0])
            results.insert(0, votes[0])
            print(f"\n  voting.always: {votes[0]['model']} is saved "
                  f"regardless of rank.")

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
    emit_progress(done + 1, grand_total, "scoring the winner",
                  status="running", model=best["model"],
                  message="subject-independent and contrast scores")
    persons = person_vector(recs, groups)
    lopo = leave_one_subject_out(model, X, y, persons, labels)

    # The contrast number: whichever grouping did NOT pick the winner. With
    # several subjects the headline is subject-independent, so the
    # within-subject score is shown beside it to make the gap visible --
    # that gap IS the subject-specificity, and hiding it would let a
    # within-subject number be read as a cross-wearer one.
    contrast = None
    if grouping == "subject":
        contrast = evaluate(model, X, y, groups, labels, grouping="recording")
    elif len(people) > 1:
        contrast = evaluate(model, X, y, groups, labels,
                            split_groups=persons, grouping="subject")
    if contrast:
        other = "within-subject" if grouping == "subject" else "subject-independent"
        print(f"\n  {other} contrast ({contrast['grouping']}-grouped): "
              f"{contrast['window_accuracy'] * 100:.1f}% per window, "
              f"{contrast['recording_vote_accuracy'] * 100:.1f}% per recording"
              f"  ({contrast['n_splits']} folds)")
        gap = abs(best["window_accuracy"] - contrast["window_accuracy"])
        if gap > 0.05:
            print(f"     the two protocols disagree by {gap * 100:.1f} points "
                  "-- quote the subject-independent one for anything about a "
                  "NEW wearer.")
    # The control from CLAUDE.md section 8f: can 8 DC levels do this alone?
    emit_progress(done + 2, grand_total, "DC-only control", status="running",
                  model=best["model"],
                  message="re-scoring the winner on 8 channel means")
    feat_names = feature_names_for(best["feature_set"], best["freq_features"])
    majority = (float(max(Counter(y.tolist()).values())) / float(len(y))
                if len(y) else 0.0)
    split_vector = persons if grouping == "subject" else None
    #  Scored on the winner's own split key, not on the recording ids: the
    #  control only means something against the number it is subtracted from.
    control = dc_only_control(model, X, y, groups, labels, feat_names,
                              split_groups=split_vector)
    print(f"\n  majority-class baseline: {majority * 100:.1f}%")
    if control:
        gain = best["window_accuracy"] - control["window_accuracy"]
        print(f"  DC-only control ({control['n_features']} channel means, "
              f"{control['grouping']}-grouped like the winner): "
              f"{control['window_accuracy'] * 100:.1f}% per window, "
              f"{control['recording_vote_accuracy'] * 100:.1f}% per recording")
        print(f"  -> the other {len(feat_names) - control['n_features']} "
              f"features add {gain * 100:+.1f} points.")
        if gain < 0.05:
            print("     READ THIS: shape, spectrum and timing add almost "
                  "nothing. On this\n"
                  "     dataset each active gesture is a single take, so the "
                  "classifier can\n"
                  "     win by recognising WHICH 50-SECOND BLOCK it is looking "
                  "at. Treat the\n"
                  "     score as a dataset artefact, not as gesture "
                  "recognition -- docs/EDA.md.")

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
            "protocol": f"StratifiedGroupKFold {grouping_note}",
            "grouping": grouping,
            "grouping_contrast": contrast,
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
        "screen": screen_rows,
        "elapsed_s": elapsed,
        # How the sweep was cut down, so a fast run can never be mistaken for
        # an exhaustive one. See how_to_fix_long_training.md.
        "cost_controls": {
            "dedupe_windows": DEDUPE_WINDOWS,
            "screen_fraction": SCREEN_FRACTION,
            "screen_probes": list(screen_names),
            "shortlist_size": len(shortlist),
            "shortlist_random": (0 if args.quick else A2_RANDOM),
            "setups_total": len(ranked),
            "setups_fully_probed": len(shortlist),
            "collapsed_excludes": collapsed_excludes,
            "note": ("Stage A1 screened every setup on "
                     f"{int(SCREEN_FRACTION * 100)} % of the deduped windows "
                     "with cheap probes; only the shortlist was probed on full "
                     "data. A1 rows are in 'screen' and are NOT comparable "
                     "with 'sweep' rows."),
        },
        "calibration": {
            "mode": cal_mode,
            "requested": cal_requested,
            "target_level": float(T.CALIBRATION_TARGET_LEVEL),
            "calibrated_subjects": sorted(cal_table),
            "uncalibrated_subjects": cal_missing,
            "records": {p: cal_records.get(p) for p in sorted(cal_table)},
            "protocol_note": (
                "A calibrated subject-grouped score is not zero-shot: the "
                "held-out wearer's own unlabelled signal set their gain. It "
                "answers 'how well does this work on a new wearer who "
                "calibrates first'. Compare against the --calibration off "
                "run for the zero-shot number."
                if cal_mode != "off" and len(people) > 1 else
                "no per-wearer calibration was applied"),
        },
        "confound": {
            "majority_class": majority,
            "dc_only_control": control,
            "warning": ("Each active gesture in this dataset is a single "
                        "continuous take, so label is confounded with "
                        "electrode state and wall-clock block. Compare any "
                        "score against dc_only_control before reading it as "
                        "gesture recognition. See docs/EDA.md."),
        },
    }
    report_path = T.unique_path(os.path.join(report_dir, f"sweep_{stamp}.json"))
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n  sweep report: {os.path.relpath(report_path, T.PROJECT_DIR)}")

    if args.no_save:
        print("\n--no-save: model not written.")
        return 0

    emit_progress(done + 3, grand_total, "final fit", status="running",
                  model=best["model"],
                  message="fitting the winner on the whole dataset")
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
        # Which per-wearer calibration this model's features were built with,
        # and whose. LivePredictor reads this and REFUSES to predict in a mode
        # it cannot reproduce for the current wearer: a model fitted on
        # calibrated features and then fed device counts would not break, it
        # would be confidently wrong, which is worse.
        "calibration": {
            # `mode` is what was APPLIED -- it is what LivePredictor has to
            # reproduce. `requested` is what was asked for, and the two differ
            # only when no wearer had a usable calibration.
            "mode": cal_mode,
            "requested": cal_requested,
            "target_level": float(T.CALIBRATION_TARGET_LEVEL),
            "subjects": sorted(cal_table),
            "uncalibrated_subjects": cal_missing,
            "records": {p: cal_records.get(p) for p in sorted(cal_table)},
            "note": ("the features were built on each wearer's own calibrated "
                     "scale; a new wearer must calibrate before this model "
                     "can be used on them"
                     if cal_mode != "off" else
                     f"no calibration applied -- device counts"
                     + (f" (mode {cal_requested!r} was requested, but no "
                        f"wearer had a usable calibration)"
                        if cal_requested != "off" else "")),
        },
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
            "validation_protocol": (
                "leave-one-subject-out folds (StratifiedGroupKFold grouped by "
                "subject)" if grouping == "subject"
                else "leave-recordings-out (StratifiedGroupKFold)"),
            "selection_grouping": grouping,
            "grouping_contrast": contrast,
            "n_splits": best["n_splits"],
            "subject_independent": lopo,
            "train_fit_accuracy": float(accuracy_score(y, train_pred)),
            "majority_class": majority,
            "dc_only_control": control,
            "deduped_windows": DEDUPE_WINDOWS,
            "calibration_mode": cal_mode,
            "calibrated_subjects": sorted(cal_table),
            "confound_warning": (
                "Each active gesture is one continuous take, so label is "
                "confounded with electrode state and time block; the "
                "DC-only control says how much the real features added."),
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

    # ---- the decision panel -------------------------------------------------
    #  Written AFTER the model, and never allowed to take the run down with
    #  it: this file is a diagnostic view, the model is the deliverable.
    if not args.no_panel:
        print("\nBuilding the decision panel (every candidate on the "
              "winner's setup)...")
        try:
            #  uid -> the take it was cut from, so the scatter can be
            #  coloured by source. This is the confound in docs/EDA.md
            #  section 2 made visible rather than described.
            take_of = {}
            for rec in all_recs:
                placed = source_timeline(rec)
                take_of[rec.uid] = placed[0] if placed else None
            panel = build_decision_panel(spec, args.quick, X, y, groups,
                                         labels, best, meta, take_of,
                                         cal_mode)
        except Exception as exc:                           # pragma: no cover
            panel = None
            print(f"  decision panel failed ({exc}); the model is unaffected.")
        if panel is None:
            print("  no decision panel written.")
        else:
            panel_path = os.path.join(os.path.dirname(model_path),
                                      T.PANEL_FILENAME)
            staging = panel_path + ".partial"
            joblib.dump(panel, staging)
            os.replace(staging, panel_path)
            size_mb = os.path.getsize(panel_path) / 1e6
            print(f"  decision panel: "
                  f"{os.path.relpath(panel_path, T.PROJECT_DIR)} "
                  f"({size_mb:.1f} MB, {len(panel['member_order'])} members"
                  f"{' + ' + panel['vote_name'] if panel['vote_name'] else ''})")
            print("  MODEL STATS -> DECISION TRAIL now shows every "
                  "candidate's live vote.")

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

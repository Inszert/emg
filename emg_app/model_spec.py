"""
The model specification schema and validation (sklearn-free), plus estimator
descriptions.
"""

import json
import os
import random
import re
from typing import Optional

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BG, COL_BORDER, COL_GOOD, COL_PANEL, COL_PANEL_2,
    COL_TEXT, COL_TEXT_DIM, COL_WARN, FS_LG, FS_MD, FS_SM)
from emg_app.utils import ensure_dir
from emg_app.calibration import CALIBRATION_MODES


# ------------------------------------------------------------------------------
# Model specification
# ------------------------------------------------------------------------------
#
# A spec is plain JSON so it can sit next to the model, be diffed, and be
# handed to a subprocess:
#
#     {"version": 1,
#      "voting": {"enabled": true, "mode": "soft", "always": false},
#      "models": [{"name": "...", "kind": "lda", "enabled": true,
#                  "scaler": "standard", "params": {...}}]}
#
# Everything in here is Qt-free and sklearn-free: the GUI edits it, the trainer
# turns it into estimators, and neither needs the other's dependencies.

MODEL_SPEC_VERSION = 1

# Which recordings a training run uses, and whether the wearers' calibration is
# applied to them. Part of the spec because both change what the model IS: a
# model trained on two wearers is a different model from one trained on five,
# and a calibrated model refuses to run for a wearer who has not calibrated.
#
#   "include":     ["Dano/session_01", ...]  -- person/session pairs. EMPTY
#                  means everything on disk (the old behaviour).
#   "calibration": "default" follows CALIBRATION_MODE / --calibration; any of
#                  CALIBRATION_MODES overrides it for training. "off" is the
#                  one that lets a wearer with no calibration use the model.
DATA_CALIBRATION_DEFAULT = "default"


def default_data_selection() -> dict:
    return {"include": [], "calibration": DATA_CALIBRATION_DEFAULT}


def validate_data_selection(raw, problems: list) -> dict:
    """The spec's `data` block, cleaned. Unknown values fall back, never raise."""
    raw = raw if isinstance(raw, dict) else {}
    include = raw.get("include")
    if isinstance(include, str):
        include = [include]
    include = include if isinstance(include, (list, tuple)) else []
    clean = []
    for item in include:
        text = str(item).strip().replace("\\", "/")
        if text.count("/") != 1 or not all(text.split("/")):
            problems.append(f"data.include: {item!r} is not 'person/session'; "
                            "dropped")
            continue
        if text not in clean:
            clean.append(text)
    mode = str(raw.get("calibration", DATA_CALIBRATION_DEFAULT))
    if mode != DATA_CALIBRATION_DEFAULT and mode not in CALIBRATION_MODES:
        problems.append(f"data.calibration {mode!r} is not 'default' or one of "
                        f"{list(CALIBRATION_MODES)}; using 'default'")
        mode = DATA_CALIBRATION_DEFAULT
    return {"include": clean, "calibration": mode}


#  The values train_model.py sweeps, repeated here because this module must
#  not import the trainer (no sklearn in the app).
SPEC_SETUP_REPRESENTATIONS = ("raw", "normalized", "envelope", "bandpass")
SPEC_SETUP_FEATURE_SETS = ("level", "both", "classic")
SPEC_SETUP_EXCLUDES = ("all", "no_errors", "clean")


def validate_setup(raw, problems: list) -> Optional[dict]:
    """
    The spec's optional `setup` block: ONE preprocessing setup to train on
    instead of sweeping them all. None when absent -- the sweep as always.

    It exists so that a combination measured once (nn_ml_comb_test.py) can be
    made the default and REPRODUCED by every later RETRAIN, rather than the
    sweep re-choosing a setup the chosen members were never scored on.
    Anything unusable drops the pin, never the run.
    """
    if not isinstance(raw, dict):
        return None
    try:
        out = {"window_s": round(float(raw["window_s"]), 3),
               "overlap": round(float(raw.get("overlap", 0.75)), 3),
               "representation": str(raw.get("representation", "raw")),
               "freq_features": bool(raw.get("freq_features", False)),
               "feature_set": str(raw.get("feature_set", "level")),
               "exclude": str(raw.get("exclude", "all"))}
    except (KeyError, TypeError, ValueError):
        problems.append("setup: needs at least a numeric window_s; pin "
                        "dropped, the sweep chooses the setup")
        return None
    bad = []
    if not 0.1 <= out["window_s"] <= 2.0:
        bad.append(f"window_s {out['window_s']}")
    if not 0.0 <= out["overlap"] <= 0.9:
        bad.append(f"overlap {out['overlap']}")
    if out["representation"] not in SPEC_SETUP_REPRESENTATIONS:
        bad.append(f"representation {out['representation']!r}")
    if out["feature_set"] not in SPEC_SETUP_FEATURE_SETS:
        bad.append(f"feature_set {out['feature_set']!r}")
    if out["exclude"] not in SPEC_SETUP_EXCLUDES:
        bad.append(f"exclude {out['exclude']!r}")
    if bad:
        problems.append(f"setup: {', '.join(bad)} not usable; pin dropped, "
                        "the sweep chooses the setup")
        return None
    return out


def default_model_spec() -> dict:
    """The roster the sweep has always run, as a spec."""
    return {
        "version": MODEL_SPEC_VERSION,
        "voting": {
            "enabled": bool(cfg.MODEL_VOTING_ENABLED),
            "mode": str(cfg.MODEL_VOTING_MODE),
            "always": bool(cfg.MODEL_VOTING_ALWAYS),
        },
        "auto": {
            "enabled": bool(cfg.MODEL_AUTO_ENABLED),
            "max_per_model": int(cfg.MODEL_AUTO_MAX_PER_MODEL),
        },
        "data": default_data_selection(),
        "models": [
            {"name": name, "kind": kind, "enabled": bool(enabled),
             "scaler": scaler, "params": dict(params)}
            for name, kind, enabled, scaler, params in cfg.DEFAULT_MODEL_SPEC_MODELS
        ],
    }


def model_param_default(kind: str, param: str):
    """The schema default for one hyperparameter."""
    spec = cfg.MODEL_KINDS.get(kind, {}).get("params", {}).get(param)
    if not spec:
        return None
    value = spec.get("default")
    return list(value) if isinstance(value, (list, tuple)) else value


def _coerce_param(rule: dict, value, problems: list, where: str):
    """One hyperparameter, coerced and clamped to its schema rule."""
    kind = rule.get("type")
    default = rule.get("default")

    if kind == "choice":
        allowed = [str(v) for v in rule.get("values", [])]
        text = str(value)
        if text not in allowed:
            problems.append(f"{where}: {text!r} is not one of {allowed}; "
                            f"using {default!r}")
            return default
        return text

    if kind == "int":
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            problems.append(f"{where}: {value!r} is not a number; "
                            f"using {default!r}")
            return default
        low, high = rule.get("min"), rule.get("max")
        if low is not None and number < low:
            problems.append(f"{where}: {number} is below {low}; clamped")
            number = int(low)
        if high is not None and number > high:
            problems.append(f"{where}: {number} is above {high}; clamped")
            number = int(high)
        return number

    if kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{where}: {value!r} is not a number; "
                            f"using {default!r}")
            return default
        low, high = rule.get("min"), rule.get("max")
        if low is not None and number < low:
            problems.append(f"{where}: {number} is below {low}; clamped")
            number = float(low)
        if high is not None and number > high:
            problems.append(f"{where}: {number} is above {high}; clamped")
            number = float(high)
        return number

    if kind == "ints":
        raw = value
        if isinstance(raw, str):
            raw = [part for part in re.split(r"[,\s]+", raw.strip()) if part]
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out = []
        for item in raw:
            try:
                number = int(round(float(item)))
            except (TypeError, ValueError):
                problems.append(f"{where}: {item!r} is not a number; dropped")
                continue
            low, high = rule.get("min"), rule.get("max")
            if low is not None:
                number = max(int(low), number)
            if high is not None:
                number = min(int(high), number)
            out.append(number)
        limit = rule.get("max_items")
        if limit and len(out) > limit:
            problems.append(f"{where}: more than {limit} layers; truncated")
            out = out[:limit]
        if not out:
            problems.append(f"{where}: empty; using {default!r}")
            return list(default or [])
        return out

    return value                                             # unknown rule type


def validate_model_spec(spec: dict):
    """
    Return (clean_spec, problems).

    Never raises and never returns something the trainer cannot build: an
    unknown model kind is dropped, a bad value is coerced or clamped, and a
    spec with nothing enabled falls back to the defaults. `problems` is a list
    of human-readable strings that the GUI shows and the trainer logs.
    """
    problems = []
    spec = spec if isinstance(spec, dict) else {}

    raw_voting = spec.get("voting")
    raw_voting = raw_voting if isinstance(raw_voting, dict) else {}
    mode = str(raw_voting.get("mode", cfg.MODEL_VOTING_MODE))
    if mode not in ("soft", "hard"):
        problems.append(f"voting mode {mode!r} is not 'soft' or 'hard'; "
                        "using 'soft'")
        mode = "soft"
    voting = {
        "enabled": bool(raw_voting.get("enabled", cfg.MODEL_VOTING_ENABLED)),
        "mode": mode,
        "always": bool(raw_voting.get("always", cfg.MODEL_VOTING_ALWAYS)),
    }

    raw_auto = spec.get("auto")
    raw_auto = raw_auto if isinstance(raw_auto, dict) else {}
    try:
        per_model = int(raw_auto.get("max_per_model",
                                     cfg.MODEL_AUTO_MAX_PER_MODEL))
    except (TypeError, ValueError):
        problems.append("auto.max_per_model is not a number; using "
                        f"{cfg.MODEL_AUTO_MAX_PER_MODEL}")
        per_model = cfg.MODEL_AUTO_MAX_PER_MODEL
    if per_model < 1:
        problems.append("auto.max_per_model below 1; using 1")
        per_model = 1
    if per_model > 64:
        problems.append("auto.max_per_model above 64; clamped -- every "
                        "combination costs a full cross-validation")
        per_model = 64
    auto = {"enabled": bool(raw_auto.get("enabled", cfg.MODEL_AUTO_ENABLED)),
            "max_per_model": per_model}

    models = []
    seen_names = set()
    for index, entry in enumerate(spec.get("models") or []):
        if not isinstance(entry, dict):
            problems.append(f"model #{index + 1} is not an object; dropped")
            continue
        kind = str(entry.get("kind", ""))
        if kind not in cfg.MODEL_KINDS:
            problems.append(
                f"model #{index + 1}: unknown kind {kind!r}; dropped "
                f"(known: {', '.join(sorted(cfg.MODEL_KINDS))})")
            continue

        schema = cfg.MODEL_KINDS[kind]
        name = str(entry.get("name") or kind).strip() or kind
        if name in seen_names:
            # Two candidates with one name would collide in the results table
            # and in the saved report, so make it unique rather than drop it.
            suffix = 2
            while f"{name}-{suffix}" in seen_names:
                suffix += 1
            problems.append(f"duplicate model name {name!r}; "
                            f"renamed to {name}-{suffix}")
            name = f"{name}-{suffix}"
        seen_names.add(name)

        scaler = str(entry.get("scaler", schema.get("scaler", "none")))
        if scaler not in cfg.MODEL_SCALERS:
            problems.append(f"{name}: unknown scaler {scaler!r}; "
                            f"using {schema.get('scaler', 'none')!r}")
            scaler = schema.get("scaler", "none")

        given = entry.get("params")
        given = given if isinstance(given, dict) else {}
        params = {}
        for param, rule in schema["params"].items():
            if param in given:
                params[param] = _coerce_param(
                    rule, given[param], problems, f"{name}.{param}")
            else:
                default = rule.get("default")
                params[param] = (list(default)
                                 if isinstance(default, (list, tuple))
                                 else default)
        for unknown in sorted(set(given) - set(schema["params"])):
            problems.append(f"{name}: {unknown!r} is not a parameter of "
                            f"{kind}; ignored")

        models.append({"name": name, "kind": kind,
                       "enabled": bool(entry.get("enabled", True)),
                       "scaler": scaler, "params": params,
                       # Per-model AUTO: tune only the ones worth tuning.
                       "auto": bool(entry.get("auto", True))})

    if not models:
        problems.append("no usable models in the spec; using the defaults")
        fallback = default_model_spec()
        fallback["data"] = validate_data_selection(spec.get("data"), problems)
        return fallback, problems
    if not any(m["enabled"] for m in models):
        problems.append("every model was switched off; enabling the first one "
                        "so there is something to train")
        models[0]["enabled"] = True

    clean = {"version": MODEL_SPEC_VERSION, "voting": voting,
             "auto": auto,
             "data": validate_data_selection(spec.get("data"), problems),
             "models": models}
    setup = validate_setup(spec.get("setup"), problems)
    if setup is not None:
        clean["setup"] = setup
    if spec.get("chosen_by"):
        clean["chosen_by"] = str(spec["chosen_by"])
    return clean, problems


def model_search_space(kind: str) -> dict:
    """What AUTO mode will try for one kind. {} = nothing to tune."""
    return {param: list(values)
            for param, values in cfg.MODEL_SEARCH_SPACES.get(kind, {}).items()
            if param in cfg.MODEL_KINDS.get(kind, {}).get("params", {})}


def auto_combinations(entry: dict, limit: int, seed: int = 0):
    """
    The parameter combinations AUTO mode will try for one model.

    The full grid is sampled down to `limit` with a seeded shuffle, so a
    rerun produces the same candidates -- a sweep you cannot reproduce is not
    a result. The model's own configured values are always combination #1, so
    turning AUTO on can never lose the setting you had.
    """
    space = model_search_space(entry["kind"])
    if not space or limit <= 1:
        return [dict(entry.get("params") or {})]

    base = dict(entry.get("params") or {})
    grid = [dict(base)]
    for param, values in sorted(space.items()):
        grown = []
        for partial in grid:
            for value in values:
                candidate = dict(partial)
                candidate[param] = (list(value)
                                    if isinstance(value, (list, tuple))
                                    else value)
                grown.append(candidate)
        grid = grown

    # De-duplicate, keeping the configured one first.
    seen, unique = set(), []
    for candidate in [base] + grid:
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    if len(unique) <= limit:
        return unique
    rest = unique[1:]
    random.Random(seed).shuffle(rest)
    return [unique[0]] + rest[:limit - 1]


def expand_auto_spec(spec: dict, seed: int = 0):
    """
    Turn AUTO mode into ordinary candidates, and return (spec, note).

    Every combination becomes its own named entry, so it is trained and scored
    by exactly the same cross-validation as a hand-written one and shows up as
    its own row in the results. There is no second, hidden evaluation path --
    which is the point: a tuned model has to earn its score the same way.

    Caller beware, and the GUI says so: choosing the best of many
    combinations on the same folds makes the WINNER's score optimistic. More
    combinations, more optimism. The honest number for a tuned pipeline is a
    fresh subject, which is what leave-one-subject-out reports.
    """
    clean, _ = validate_model_spec(spec)
    auto = clean.get("auto") or {}
    if not auto.get("enabled"):
        return clean, "auto mode off"

    limit = int(auto.get("max_per_model", cfg.MODEL_AUTO_MAX_PER_MODEL))
    expanded, grown = [], 0
    for entry in clean["models"]:
        if not entry["enabled"] or not entry.get("auto", True):
            expanded.append(entry)
            continue

        combos = auto_combinations(entry, limit, seed=seed)
        if len(combos) <= 1:
            expanded.append(entry)
            continue

        space = model_search_space(entry["kind"])
        for index, params in enumerate(combos):
            # Name each variant by what actually differs, so the results
            # table is readable instead of being full of "RF-3".
            bits = []
            for param in sorted(space):
                value = params.get(param)
                if isinstance(value, list):
                    value = "x".join(str(v) for v in value)
                bits.append(f"{param}={value}")
            expanded.append({
                "name": f"{entry['name']}[{', '.join(bits)}]" if bits
                        else f"{entry['name']}#{index + 1}",
                "kind": entry["kind"], "enabled": True,
                "scaler": entry["scaler"], "params": params,
                "auto": False,
                "auto_from": entry["name"],
            })
            grown += 1

    clean = dict(clean)
    clean["models"] = expanded
    final, _ = validate_model_spec(clean)
    # validate_model_spec drops keys it does not know, so put the provenance
    # back: the report needs to say which entry a variant came from.
    by_name = {m["name"]: m for m in expanded}
    for model in final["models"]:
        source = by_name.get(model["name"], {}).get("auto_from")
        if source:
            model["auto_from"] = source
    return final, (f"auto mode: {grown} combination(s) generated, "
                   f"{len([m for m in final['models'] if m['enabled']])} "
                   f"candidates in total")


def enabled_models(spec: dict):
    """The entries the sweep will actually train, in order."""
    clean, _ = validate_model_spec(spec)
    return [m for m in clean["models"] if m["enabled"]]


def model_spec_path(model_path: str = None) -> str:
    """`models/model_spec.json` -- next to the model it describes."""
    target = model_path or cfg.MODEL_PATH
    if not os.path.isabs(target):
        target = os.path.join(cfg.PROJECT_DIR, target)
    return os.path.join(os.path.dirname(os.path.abspath(target)),
                        cfg.MODEL_SPEC_NAME)


def panel_path(model_path: str = None) -> str:
    """`models/gesture_panel.joblib` -- next to the model it explains."""
    target = model_path or cfg.MODEL_PATH
    if not os.path.isabs(target):
        target = os.path.join(cfg.PROJECT_DIR, target)
    return os.path.join(os.path.dirname(os.path.abspath(target)),
                        cfg.PANEL_FILENAME)


def save_model_spec(path: str, spec: dict) -> str:
    """
    Write a spec, validated first.

    Overwrites deliberately: unlike a recording, a spec is a setting, and
    keeping fifteen numbered copies of it would make it impossible to tell
    which one the trainer will read. The file it produces is copied into every
    model's payload, so the provenance of a trained model is not lost.
    """
    clean, problems = validate_model_spec(spec)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2, ensure_ascii=False, sort_keys=False)
    os.replace(temp, path)
    return path


def load_model_spec(path: str = None):
    """
    Read a spec from disk. Returns (spec, problems).

    A missing file is not an error -- it means "never edited", and the
    defaults are the right answer.
    """
    path = path or model_spec_path()
    if not os.path.exists(path):
        return default_model_spec(), []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return default_model_spec(), [f"could not read {os.path.basename(path)}"
                                      f" ({exc}); using the defaults"]
    return validate_model_spec(payload)


# Which constructor arguments are worth showing per class. Printing every
# sklearn default would bury the three numbers that matter.
ESTIMATOR_KEY_PARAMS = {
    "LinearDiscriminantAnalysis": ("solver", "shrinkage"),
    "SVC": ("kernel", "C", "gamma", "class_weight"),
    "RandomForestClassifier": ("n_estimators", "max_depth", "min_samples_leaf",
                               "max_features", "class_weight"),
    "ExtraTreesClassifier": ("n_estimators", "max_depth", "min_samples_leaf",
                             "class_weight"),
    "HistGradientBoostingClassifier": ("learning_rate", "max_iter",
                                       "max_leaf_nodes", "l2_regularization",
                                       "early_stopping", "class_weight"),
    "LogisticRegression": ("penalty", "C", "l1_ratio", "solver", "max_iter",
                           "class_weight"),
    "KNeighborsClassifier": ("n_neighbors", "weights"),
    "MLPClassifier": ("hidden_layer_sizes", "alpha", "solver", "batch_size",
                      "learning_rate_init", "early_stopping", "max_iter"),
    "StandardScaler": (),
    "RobustScaler": (),
    "MinMaxScaler": (),
}


def _estimator_params(estimator) -> str:
    """The interesting constructor arguments of one estimator, as text."""
    name = type(estimator).__name__
    wanted = ESTIMATOR_KEY_PARAMS.get(name)
    try:
        params = estimator.get_params(deep=False)
    except (AttributeError, TypeError):                      # pragma: no cover
        return ""
    if wanted is None:
        # An unknown class: show everything short rather than nothing, so a
        # model trained by a future version is still readable here.
        wanted = tuple(sorted(params))
    bits = []
    for key in wanted:
        if key not in params:
            continue
        value = params[key]
        if value is None and key == "class_weight":
            continue                                 # the default, not news
        if isinstance(value, float):
            text = f"{value:g}"
        elif isinstance(value, (tuple, list)):
            text = "x".join(str(v) for v in value)
        else:
            text = str(value)
        bits.append(f"{key}={text}")
    return ", ".join(bits)


def serialise_estimator(estimator, depth: int = 0) -> None:
    """
    Make a fitted estimator predict on ONE thread, everywhere inside it.

    A forest saved with `n_jobs=-1` stands up a joblib worker pool on every
    predict call. For the many rows of a sweep that pays; for the single row
    of a live window it is nearly all overhead -- measured 2026-10-22 on the
    deployed 500-tree forests: 64 ms per window at -1, 20 ms at 1, and the
    four-model vote 148 ms against a 150 ms prediction interval. `n_jobs`
    changes only how a fitted model is evaluated, never what it answers.
    Used by LivePredictor and by the DECISION TRAIL panel.
    """
    if estimator is None or depth > 4:
        return
    if getattr(estimator, "n_jobs", None) not in (None, 1):
        try:
            estimator.n_jobs = 1
        except Exception:                                  # pragma: no cover
            pass
    for step in (getattr(estimator, "named_steps", None) or {}).values():
        serialise_estimator(step, depth + 1)
    #  VotingClassifier: `estimators_` is the fitted list, and it is the one
    #  that runs; `estimators` holds (name, estimator) pairs.
    for member in getattr(estimator, "estimators_", None) or []:
        serialise_estimator(member, depth + 1)
    for entry in getattr(estimator, "estimators", None) or []:
        serialise_estimator(entry[1] if isinstance(entry, tuple) else entry,
                            depth + 1)


def _learned_numbers(obj, _seen=None) -> int:
    """
    How many numbers a fitted object stores to make its predictions.

    Counted off the object, like describe_estimator: every fitted array
    (public attributes ending in `_`), recursing into fitted sub-estimators.
    A decision tree counts as nodes x (feature + threshold + class values),
    which is what it actually looks up. Comparable in spirit with a net's
    parameter count, not identical: a tree node is a lookup, a weight is a
    multiply.
    """
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    tree = getattr(obj, "tree_", None)
    if tree is not None and hasattr(tree, "node_count"):
        value = getattr(tree, "value", None)
        width = int(np.prod(value.shape[1:])) if value is not None else 1
        return int(tree.node_count) * (2 + width)
    total = 0
    for key, value in vars(obj).items():
        if not key.endswith("_") or key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                total += int(value.size)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, np.ndarray) and item.dtype != object:
                    total += int(item.size)
                elif hasattr(item, "__dict__"):
                    total += _learned_numbers(item, _seen)
        elif hasattr(value, "get_params"):
            total += _learned_numbers(value, _seen)
    for _name, step in (getattr(obj, "steps", None) or []):
        if hasattr(step, "__dict__"):
            total += _learned_numbers(step, _seen)
    # HistGradientBoosting keeps its trees in a private list of predictors.
    for stage in (getattr(obj, "_predictors", None) or []):
        for predictor in stage:
            nodes = getattr(predictor, "nodes", None)
            if nodes is not None:
                total += int(len(nodes)) * 3
    return total


def _size_detail(estimator) -> str:
    """One line on what makes up an estimator's size."""
    final = estimator
    while getattr(final, "steps", None):
        final = final.steps[-1][1]
    trees = getattr(final, "estimators_", None)
    if trees and all(hasattr(t, "tree_") for t in trees):
        nodes = sum(int(t.tree_.node_count) for t in trees)
        return f"{len(trees)} trees, {nodes:,} nodes"
    vectors = getattr(final, "support_vectors_", None)
    if vectors is not None:
        return f"{vectors.shape[0]:,} support vectors x {vectors.shape[1]}"
    coef = getattr(final, "coef_", None)
    if coef is not None:
        return f"weights {' x '.join(str(d) for d in coef.shape)}"
    fit_x = getattr(final, "_fit_X", None)
    if fit_x is not None:
        return f"stores {fit_x.shape[0]:,} training rows"
    layers = getattr(final, "coefs_", None)
    if layers:
        return "layers " + " -> ".join(
            [str(layers[0].shape[0])] + [str(c.shape[1]) for c in layers])
    return type(final).__name__


def estimator_size(estimator) -> dict:
    """
    {"total": learned numbers, "parts": [(name, class, numbers, detail)]}

    A VotingClassifier is broken down per member, anything else is one part.
    Qt-free and sklearn-free: it reads attributes, it imports nothing.
    """
    if estimator is None:
        return {"total": 0, "parts": []}
    members = getattr(estimator, "estimators_", None)
    if members is not None and type(estimator).__name__.startswith("Voting"):
        names = [n for n, _ in (getattr(estimator, "estimators", None) or [])]
        names += [f"member {i + 1}" for i in range(len(names), len(members))]
        parts = []
        for name, member in zip(names, members):
            final = member
            while getattr(final, "steps", None):
                final = final.steps[-1][1]
            parts.append((name, type(final).__name__,
                          _learned_numbers(member), _size_detail(member)))
        return {"total": sum(p[2] for p in parts), "parts": parts}
    final = estimator
    while getattr(final, "steps", None):
        final = final.steps[-1][1]
    numbers = _learned_numbers(estimator)
    return {"total": numbers,
            "parts": [(type(final).__name__, type(final).__name__, numbers,
                       _size_detail(estimator))]}


def file_size_text(path: str) -> str:
    """`18.7 MB` / `25 KB` for a file on disk, `?` when it is not there."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "?"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{max(1, round(size / 1024))} KB"


def describe_estimator(estimator, indent: int = 0, max_depth: int = 4):
    """
    The real structure of a fitted estimator, as lines of text.

    Read off the OBJECT, not off the spec that was meant to produce it: this
    is what will actually run on the next prediction, so if the two ever
    disagree, this is the truth. A Pipeline becomes its ordered stages, a
    VotingClassifier becomes its members plus how their outputs are combined,
    and anything else becomes one line.
    """
    pad = "    " * indent
    if estimator is None:
        return [f"{pad}(no model)"]
    if indent > max_depth:                                   # pragma: no cover
        return [f"{pad}..."]

    name = type(estimator).__name__
    lines = []

    steps = getattr(estimator, "steps", None)
    if steps is not None:                                    # a Pipeline
        lines.append(f"{pad}Pipeline  ({len(steps)} stages, in order)")
        for position, (step_name, step) in enumerate(steps, start=1):
            arrow = "->" if position > 1 else "  "
            lines.append(f"{pad}  {arrow} [{position}] {step_name}: "
                         f"{type(step).__name__}")
            detail = _estimator_params(step)
            if detail:
                lines.append(f"{pad}         {detail}")
            if getattr(step, "steps", None) or getattr(step, "estimators", None):
                lines.extend(describe_estimator(step, indent + 3, max_depth))
        return lines

    members = getattr(estimator, "estimators", None)
    if members is not None and name.startswith("Voting"):
        mode = getattr(estimator, "voting", "?")
        weights = getattr(estimator, "weights", None)
        lines.append(f"{pad}{name}   --   {len(members)} models, "
                     f"{mode} voting")
        if mode == "soft":
            lines.append(f"{pad}  combines: average of each model's "
                         "predicted probabilities, highest average wins")
        else:
            lines.append(f"{pad}  combines: one vote per model, "
                         "most votes wins (ties go to the first label)")
        lines.append(f"{pad}  weights : "
                     + ("equal" if not weights
                        else ", ".join(f"{w:g}" for w in weights)))
        for position, item in enumerate(members, start=1):
            member_name, member = (item if isinstance(item, tuple)
                                   else (f"model{position}", item))
            lines.append(f"{pad}  member {position}: {member_name}")
            lines.extend(describe_estimator(member, indent + 2, max_depth))
        return lines

    lines.append(f"{pad}{name}")
    detail = _estimator_params(estimator)
    if detail:
        lines.append(f"{pad}  {detail}")
    return lines


def _html_escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def estimator_html(estimator, depth: int = 0) -> str:
    """
    A fitted estimator drawn as the chain of stages it actually is.

    The text version (describe_estimator) is exact but reads as a wall; this
    is the same information laid out so the SHAPE is visible at a glance --
    what goes in, what each stage does, and, for an ensemble, that several
    models run in parallel and how their answers are merged.

    Read off the object, like its text counterpart: this is what will run.
    """
    if estimator is None:
        return f'<p style="color:{COL_TEXT_DIM}">No model loaded.</p>'

    name = type(estimator).__name__

    steps = getattr(estimator, "steps", None)
    if steps is not None:
        rows = []
        for position, (step_name, step) in enumerate(steps, start=1):
            detail = _estimator_params(step)
            inner = ""
            if getattr(step, "steps", None) or getattr(step, "estimators", None):
                inner = estimator_html(step, depth + 1)
            rows.append(
                f'<tr>'
                f'<td style="padding:6px 10px;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px;white-space:nowrap">step {position}</td>'
                f'<td style="padding:6px 10px">'
                f'<b style="color:{COL_ACCENT}">{_html_escape(type(step).__name__)}</b>'
                f'<span style="color:{COL_TEXT_DIM}"> &nbsp;({_html_escape(step_name)})</span>'
                + (f'<br><span style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
                   f'{_html_escape(detail)}</span>' if detail else "")
                + inner +
                f'</td></tr>'
            )
        return (f'<table width="100%" cellspacing="0" cellpadding="0" '
                f'style="background:{COL_PANEL};border:1px solid {COL_BORDER}">'
                + "".join(rows) + '</table>')

    members = getattr(estimator, "estimators", None)
    if members is not None and name.startswith("Voting"):
        mode = getattr(estimator, "voting", "?")
        weights = getattr(estimator, "weights", None)
        combine = ("every model outputs a probability per gesture; these are "
                   "averaged and the highest average wins"
                   if mode == "soft" else
                   "every model names one gesture; the most-named wins")
        cards = []
        for position, item in enumerate(members, start=1):
            member_name, member = (item if isinstance(item, tuple)
                                   else (f"model{position}", item))
            cards.append(
                f'<div style="margin:6px 0;padding:8px;'
                f'background:{COL_PANEL};border:1px solid {COL_BORDER}">'
                f'<div style="color:{COL_ACCENT};font-weight:700">'
                f'{position}. {_html_escape(member_name)}</div>'
                + estimator_html(member, depth + 1) + '</div>')
        return (
            f'<div style="padding:10px;background:{COL_PANEL_2};'
            f'border:1px solid {COL_ACCENT}">'
            f'<div style="font-size:{FS_LG}px;font-weight:700;color:{COL_TEXT}">'
            f'{len(members)} models vote &nbsp;&middot;&nbsp; '
            f'{_html_escape(mode)}</div>'
            f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px;'
            f'padding:4px 0 2px 0">How they are combined: {combine}.</div>'
            f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
            f'Weights: {"equal" if not weights else _html_escape(weights)}</div>'
            + "".join(cards) + '</div>')

    detail = _estimator_params(estimator)
    return (f'<div style="padding:8px;background:{COL_PANEL};'
            f'border:1px solid {COL_BORDER}">'
            f'<b style="color:{COL_ACCENT}">{_html_escape(name)}</b>'
            + (f'<br><span style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
               f'{_html_escape(detail)}</span>' if detail else "")
            + '</div>')


def model_report_html(predictor, stats: dict) -> str:
    """The whole 'what is this model' report, as one styled document."""
    stats = dict(stats or {})

    def section(title, body):
        return (f'<div style="margin:16px 0 6px 0;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px;font-weight:700;letter-spacing:1px">'
                f'{title}</div>{body}')

    def row(key, value):
        return (f'<tr><td style="padding:3px 12px 3px 0;color:{COL_TEXT_DIM};'
                f'white-space:nowrap">{_html_escape(key)}</td>'
                f'<td style="padding:3px 0;color:{COL_TEXT}">'
                f'{_html_escape(value)}</td></tr>')

    parts = [f'<body style="background:{COL_BG};color:{COL_TEXT};'
             f'font-family:Segoe UI;font-size:{FS_MD}px">']

    # -- what goes in ---------------------------------------------------------
    parts.append(section("WHAT GOES IN", (
        f'<table cellspacing="0">'
        + row("Window", f"{stats.get('window_s', 0) * 1000:.0f} ms of all "
                        f"{stats.get('n_channels', cfg.N_CHANNELS)} channels")
        + row("Resampled to", f"{stats.get('fs', predictor.fs):g} Hz")
        + row("Features", f"{stats.get('n_features', '?')} "
                          f"({stats.get('feature_set', '?')} set, "
                          f"{stats.get('representation', '?')})")
        + row("Outputs", ", ".join(predictor.labels or
                                   stats.get("labels", []) or ["?"]))
        + row("Calibration",
              (f"{stats.get('calibration_mode')} -- per-wearer, fitted on "
               + ", ".join(stats.get("calibrated_subjects") or ["?"])
               + "; a new wearer must calibrate first")
              if (stats.get("calibration_mode") or "off") != "off"
              else "off -- device counts, no per-wearer scaling")
        + '</table>')))

    # -- how it decides -------------------------------------------------------
    parts.append(section("HOW IT DECIDES", estimator_html(predictor.model)))

    # -- what it beat ---------------------------------------------------------
    spec = stats.get("model_spec")
    if isinstance(spec, dict):
        clean, _ = validate_model_spec(spec)
        vote = clean.get("voting") or {}
        auto = clean.get("auto") or {}
        candidates = [m for m in clean["models"] if m.get("enabled")]
        winner = stats.get("model_name")
        lines = []
        for model in candidates:
            is_winner = model["name"] == winner
            params = ", ".join(f"{k}={v}" for k, v in
                               sorted((model.get("params") or {}).items()))
            lines.append(
                f'<tr style="background:{COL_PANEL_2 if is_winner else "transparent"}">'
                f'<td style="padding:4px 10px;'
                f'color:{COL_GOOD if is_winner else COL_TEXT}">'
                + ("&#9733; " if is_winner else "&nbsp;&nbsp;&nbsp;")
                + f'{_html_escape(model["name"])}</td>'
                f'<td style="padding:4px 10px;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px">{_html_escape(params)}</td></tr>')
        parts.append(section(
            f"WHAT IT BEAT &nbsp;({len(candidates)} candidates, "
            f"same cross-validation)",
            f'<table width="100%" cellspacing="0">' + "".join(lines)
            + '</table>'
            + f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px;'
              f'padding-top:6px">Voting: '
            + (f'{vote.get("mode")}' if vote.get("enabled") else "off")
            + f' &nbsp;&middot;&nbsp; Auto mode: '
            + (f'on, up to {auto.get("max_per_model")} per model'
               if auto.get("enabled") else "off") + '</div>'))

        if auto.get("enabled"):
            parts.append(
                f'<div style="margin-top:10px;padding:10px;'
                f'border-left:3px solid {COL_WARN};background:{COL_PANEL}">'
                f'<b style="color:{COL_WARN}">Auto mode was on.</b> '
                f'<span style="color:{COL_TEXT_DIM}">The winner was chosen '
                f'from {len(candidates)} candidates scored on the same folds, '
                f'so the headline accuracy is optimistic for this model. The '
                f'leave-one-subject-out figure below did not get to pick its '
                f'own folds.</span></div>')

    parts.append('</body>')
    return "".join(parts)


def describe_model_spec(spec: dict) -> str:
    """One line for the GUI and the log."""
    clean, _ = validate_model_spec(spec)
    on = [m for m in clean["models"] if m["enabled"]]
    voting = clean["voting"]
    text = f"{len(on)} of {len(clean['models'])} models"
    auto = clean.get("auto") or {}
    if auto.get("enabled"):
        text += f" x up to {auto.get('max_per_model')} auto combos"
    if voting["enabled"] and len(on) > 1:
        text += f" + {voting['mode']} vote"
        if voting["always"]:
            text += " (always used)"
        else:
            text += " (competing)"
    setup = clean.get("setup")
    if setup:
        text += (f"; setup pinned: {setup['representation']}/"
                 f"{setup['feature_set']} {setup['window_s']:g} s")
    return text

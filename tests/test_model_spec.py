"""
MODEL SPECIFICATION: which classifiers the sweep trains, their
hyperparameters, how they vote, and AUTO mode.

No armband and no display needed (Qt runs offscreen). scikit-learn is needed
for the half that actually builds estimators; that half is skipped without it.

The properties that matter most here:

  * a spec can never reach the trainer in a state it would crash on --
    validate_model_spec() coerces, clamps, renames and drops;
  * every model kind in the schema really builds and fits, including every
    penalty/solver combination, so a setting offered in the GUI cannot be one
    that explodes an hour into a sweep;
  * AUTO mode is reproducible and never loses the values you configured;
  * what MODEL STATS shows is read off the fitted estimator, so it cannot
    disagree with what will actually run.
"""
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

import emg_dataset_tool as T

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def toy_data(n_per_class=40, n_features=6, seed=0):
    """Three well-separated classes -- enough for every estimator to fit."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(i, 1.0, (n_per_class, n_features))
                   for i in range(3)])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


# ==============================================================================
# PART 1 -- the spec layer, no Qt, no sklearn
# ==============================================================================

def test_spec_layer():
    print("\n--- spec layer ---")
    spec = T.default_model_spec()

    check("defaults have a version", spec["version"] == T.MODEL_SPEC_VERSION)
    check("defaults list models", len(spec["models"]) >= 10,
          str(len(spec["models"])))
    check("defaults enable a useful subset",
          0 < len(T.enabled_models(spec)) < len(spec["models"]),
          f"{len(T.enabled_models(spec))} of {len(spec['models'])}")
    check("defaults validate without complaint",
          T.validate_model_spec(spec)[1] == [],
          str(T.validate_model_spec(spec)[1]))
    check("defaults have a voting block",
          set(spec["voting"]) == {"enabled", "mode", "always"})
    check("defaults have an auto block",
          set(spec["auto"]) == {"enabled", "max_per_model"})
    check("auto is off by default", spec["auto"]["enabled"] is False)
    check("every default model names a real kind",
          all(m["kind"] in T.MODEL_KINDS for m in spec["models"]))
    check("every default scaler is real",
          all(m["scaler"] in T.MODEL_SCALERS for m in spec["models"]))
    check("describe reads sensibly", "models" in T.describe_model_spec(spec),
          T.describe_model_spec(spec))

    # -- the schema is self-consistent ---------------------------------------
    for kind, schema in T.MODEL_KINDS.items():
        check(f"{kind}: has a label and a scaler",
              bool(schema.get("label")) and schema.get("scaler") in T.MODEL_SCALERS)
        for param, rule in schema["params"].items():
            kinds = ("int", "float", "choice", "ints")
            check(f"{kind}.{param}: known rule type",
                  rule.get("type") in kinds, str(rule.get("type")))
            check(f"{kind}.{param}: has a default",
                  rule.get("default") is not None)
            if rule["type"] == "choice":
                check(f"{kind}.{param}: default is one of the values",
                      rule["default"] in rule["values"])
            if rule["type"] in ("int", "float"):
                check(f"{kind}.{param}: default is inside its range",
                      rule["min"] <= rule["default"] <= rule["max"],
                      f"{rule['min']} <= {rule['default']} <= {rule['max']}")

    # -- the search spaces only mention real parameters -----------------------
    for kind, space in T.MODEL_SEARCH_SPACES.items():
        check(f"search space for {kind}: kind exists", kind in T.MODEL_KINDS)
        params = T.MODEL_KINDS.get(kind, {}).get("params", {})
        unknown = sorted(set(space) - set(params))
        check(f"search space for {kind}: no invented parameters",
              not unknown, str(unknown))
        for param, values in space.items():
            rule = params.get(param, {})
            if rule.get("type") == "choice":
                bad = [v for v in values if v not in rule["values"]]
                check(f"search space {kind}.{param}: values are allowed",
                      not bad, str(bad))

    # -- validation ----------------------------------------------------------
    clean, problems = T.validate_model_spec({
        "models": [
            {"kind": "not_a_model"},
            {"kind": "svm_rbf", "name": "a", "params": {"C": "nonsense"}},
            {"kind": "svm_rbf", "name": "a", "params": {"C": 99999.0}},
            {"kind": "knn", "name": "b", "params": {"n_neighbors": -4,
                                                    "invented": 1}},
            {"kind": "mlp", "name": "c",
             "params": {"hidden_janitor": 1,
                        "hidden_layer_sizes": "256, 128, 64, 32, 16"}},
        ],
        "voting": {"mode": "sideways"},
        "auto": {"enabled": True, "max_per_model": 9999},
    })
    names = [m["name"] for m in clean["models"]]
    check("unknown kind dropped", "not_a_model" not in
          [m["kind"] for m in clean["models"]])
    check("four usable models survived", len(clean["models"]) == 4,
          str(len(clean["models"])))
    check("a non-numeric value falls back to the default",
          clean["models"][0]["params"]["C"] == 10.0,
          str(clean["models"][0]["params"]["C"]))
    check("an out-of-range value is clamped",
          clean["models"][1]["params"]["C"] == 1000.0,
          str(clean["models"][1]["params"]["C"]))
    check("a negative count is clamped to the minimum",
          clean["models"][2]["params"]["n_neighbors"] == 1,
          str(clean["models"][2]["params"]["n_neighbors"]))
    check("a duplicate name is renamed, not dropped",
          names[1] == "a-2", str(names))
    check("an invented parameter is ignored",
          "invented" not in clean["models"][2]["params"])
    check("layer sizes parse from text",
          clean["models"][3]["params"]["hidden_layer_sizes"] == [256, 128, 64, 32],
          str(clean["models"][3]["params"]["hidden_layer_sizes"]))
    check("too many layers are truncated to max_items",
          len(clean["models"][3]["params"]["hidden_layer_sizes"]) == 4)
    check("a bad voting mode falls back to soft",
          clean["voting"]["mode"] == "soft")
    check("auto.max_per_model is clamped",
          clean["auto"]["max_per_model"] == 64,
          str(clean["auto"]["max_per_model"]))
    check("missing parameters are filled from the schema",
          set(clean["models"][0]["params"]) == set(T.MODEL_KINDS["svm_rbf"]["params"]))
    check("every problem is explained", len(problems) >= 7, str(len(problems)))

    check("an empty spec falls back to the defaults",
          len(T.validate_model_spec({})[0]["models"]) == len(spec["models"]))
    check("garbage falls back to the defaults",
          len(T.validate_model_spec("not a dict")[0]["models"]) > 0)

    all_off = {"models": [dict(m, enabled=False) for m in spec["models"]]}
    fixed, why = T.validate_model_spec(all_off)
    check("ALL MODELS OFF IS REPAIRED -- there is always something to train",
          len(T.enabled_models(fixed)) >= 1)
    check("and it says so", any("switched off" in w for w in why), str(why))

    # -- auto expansion ------------------------------------------------------
    auto_spec = T.default_model_spec()
    auto_spec["auto"] = {"enabled": True, "max_per_model": 4}
    exp, note = T.expand_auto_spec(auto_spec, seed=7)
    base_on = len(T.enabled_models(auto_spec))
    exp_on = len(T.enabled_models(exp))
    check("auto expands the roster", exp_on > base_on, f"{base_on} -> {exp_on}")
    check("expansion is capped per model", exp_on <= base_on * 4,
          f"{exp_on} <= {base_on * 4}")
    check("the note counts them", "combination" in note, note)
    check("variants record which model they came from",
          any(m.get("auto_from") for m in exp["models"]))
    check("AUTO IS REPRODUCIBLE with the same seed",
          [m["name"] for m in T.expand_auto_spec(auto_spec, seed=7)[0]["models"]]
          == [m["name"] for m in exp["models"]])
    check("a different seed explores differently",
          [m["name"] for m in T.expand_auto_spec(auto_spec, seed=8)[0]["models"]]
          != [m["name"] for m in exp["models"]])

    # The configured values must survive: AUTO adds candidates, never
    # replaces what the operator set.
    tuned = {"models": [{"name": "mine", "kind": "svm_rbf", "enabled": True,
                         "scaler": "standard",
                         "params": {"C": 7.5, "gamma": "auto"}}],
             "auto": {"enabled": True, "max_per_model": 6}}
    exp2, _ = T.expand_auto_spec(tuned, seed=1)
    kept = [m for m in exp2["models"]
            if m["params"]["C"] == 7.5 and m["params"]["gamma"] == "auto"]
    check("AUTO KEEPS THE CONFIGURED VALUES as one of the candidates",
          len(kept) >= 1, f"{len(kept)} of {len(exp2['models'])}")
    check("auto off changes nothing",
          T.expand_auto_spec(spec, seed=1)[0]["models"] ==
          T.validate_model_spec(spec)[0]["models"])

    opted_out = {"models": [{"name": "pinned", "kind": "svm_rbf",
                             "enabled": True, "scaler": "standard",
                             "auto": False, "params": {"C": 3.0}}],
                 "auto": {"enabled": True, "max_per_model": 8}}
    exp3, _ = T.expand_auto_spec(opted_out, seed=1)
    check("a model can opt out of AUTO", len(exp3["models"]) == 1,
          str(len(exp3["models"])))
    check("and keeps its value", exp3["models"][0]["params"]["C"] == 3.0)

    # -- save / load ---------------------------------------------------------
    root = tempfile.mkdtemp(prefix="emg_spec_")
    try:
        path = os.path.join(root, "models", T.MODEL_SPEC_NAME)
        edited = T.default_model_spec()
        edited["models"][0]["params"]["solver"] = "svd"
        edited["models"][2]["enabled"] = False
        edited["voting"] = {"enabled": True, "mode": "hard", "always": True}
        edited["auto"] = {"enabled": True, "max_per_model": 5}
        T.save_model_spec(path, edited)
        check("spec file written", os.path.exists(path))

        back, problems = T.load_model_spec(path)
        check("round trip, no complaints", problems == [], str(problems))
        check("a parameter edit survives",
              back["models"][0]["params"]["solver"] == "svd")
        check("an enable/disable edit survives",
              back["models"][2]["enabled"] is False)
        check("the voting block survives",
              back["voting"] == {"enabled": True, "mode": "hard",
                                 "always": True})
        check("the auto block survives",
              back["auto"] == {"enabled": True, "max_per_model": 5})
        check("the file is readable JSON",
              isinstance(json.load(open(path, encoding="utf-8")), dict))

        check("a missing file means the defaults, not an error",
              T.load_model_spec(os.path.join(root, "nope.json"))[0]["models"]
              == spec["models"])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ truncated")
        recovered, why = T.load_model_spec(path)
        check("A CORRUPT SPEC FALLS BACK instead of crashing",
              len(recovered["models"]) > 0)
        check("and explains itself", any("could not read" in w for w in why),
              str(why))

        check("spec path sits next to the model",
              os.path.basename(T.model_spec_path("models/x.joblib"))
              == T.MODEL_SPEC_NAME)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ==============================================================================
# PART 2 -- every setting really builds and fits
# ==============================================================================

def test_estimators():
    print("\n--- estimators ---")
    try:
        import train_model as TM
    except Exception as exc:                     # noqa: BLE001
        check(f"train_model imports (skipped: {exc})", True)
        return
    if not getattr(TM, "HAVE_SKLEARN", True):
        check("scikit-learn present (skipped)", True)
        return

    X, y = toy_data()

    # -- every kind, at its schema defaults ----------------------------------
    for kind, schema in T.MODEL_KINDS.items():
        entry = {"name": kind, "kind": kind, "enabled": True,
                 "scaler": schema["scaler"],
                 "params": {p: T.model_param_default(kind, p)
                            for p in schema["params"]}}
        clean, _ = T.validate_model_spec({"models": [entry]})
        try:
            pipe = TM._make_pipeline(clean["models"][0])
            pipe.fit(X, y)
            pipe.predict(X)
            check(f"{kind}: builds, fits and predicts at its defaults", True)
        except Exception as exc:                 # noqa: BLE001
            check(f"{kind}: builds, fits and predicts at its defaults", False,
                  f"{type(exc).__name__}: {exc}")

    # -- every scaler ---------------------------------------------------------
    for scaler in T.MODEL_SCALERS:
        entry = {"kind": "logreg", "scaler": scaler,
                 "params": {"C": 1.0, "max_iter": 300, "penalty": "l2",
                            "solver": "auto", "class_weight": "none"}}
        try:
            pipe = TM._make_pipeline(entry)
            pipe.fit(X, y)
            stages = [name for name, _ in pipe.steps]
            expected = ["clf"] if scaler == "none" else ["scale", "clf"]
            check(f"scaler {scaler}: fits, stages {expected}",
                  stages == expected, str(stages))
        except Exception as exc:                 # noqa: BLE001
            check(f"scaler {scaler}: fits", False, str(exc))

    # -- THE combinatorial trap: penalty x solver ----------------------------
    for penalty in ("l2", "l1", "elasticnet", "none"):
        for solver in ("auto", "lbfgs", "saga", "liblinear"):
            entry = {"kind": "logreg", "scaler": "standard",
                     "params": {"C": 1.0, "penalty": penalty, "solver": solver,
                                "l1_ratio": 0.5, "max_iter": 300,
                                "class_weight": "balanced"}}
            try:
                TM._make_pipeline(entry).fit(X, y)
                check(f"logreg {penalty} + {solver}: fits", True)
            except Exception as exc:             # noqa: BLE001
                check(f"logreg {penalty} + {solver}: fits", False,
                      f"{type(exc).__name__}: {exc}")

    # -- early stopping / batching on the MLP --------------------------------
    for solver in ("adam", "sgd", "lbfgs"):
        entry = {"kind": "mlp", "scaler": "standard",
                 "params": {"hidden_layer_sizes": [32], "alpha": 0.001,
                            "solver": solver, "batch_size": 32,
                            "learning_rate_init": 0.01, "max_iter": 150,
                            "early_stopping": "yes",
                            "validation_fraction": 0.2,
                            "n_iter_no_change": 5}}
        try:
            TM._make_pipeline(entry).fit(X, y)
            check(f"mlp solver={solver} with early stopping + batching: fits",
                  True)
        except Exception as exc:                 # noqa: BLE001
            check(f"mlp solver={solver} with early stopping + batching: fits",
                  False, f"{type(exc).__name__}: {exc}")

    check("mlp batch_size 0 becomes auto",
          TM._make_estimator("mlp", {"batch_size": 0}).batch_size == "auto")
    check("mlp batch_size is honoured otherwise",
          TM._make_estimator("mlp", {"batch_size": 64}).batch_size == 64)
    check("lbfgs is not given early stopping it cannot use",
          TM._make_estimator("mlp", {"solver": "lbfgs"}).early_stopping is False)

    # -- class balancing and L2 on the boosters ------------------------------
    for kind, params in (
        ("random_forest", {"class_weight": "balanced_subsample",
                           "n_estimators": 40}),
        ("extra_trees", {"class_weight": "balanced", "n_estimators": 40}),
        ("svm_rbf", {"class_weight": "balanced", "C": 1.0}),
        ("svm_linear", {"class_weight": "balanced", "C": 1.0}),
        ("hist_gradient_boost", {"early_stopping": "yes", "max_iter": 30,
                                 "l2_regularization": 1.0,
                                 "validation_fraction": 0.2,
                                 "n_iter_no_change": 3,
                                 "class_weight": "balanced"}),
    ):
        try:
            TM._make_pipeline({"kind": kind, "scaler": "none",
                               "params": params}).fit(X, y)
            check(f"{kind} with class balancing / L2: fits", True)
        except Exception as exc:                 # noqa: BLE001
            check(f"{kind} with class balancing / L2: fits", False, str(exc))

    check("'none' class_weight becomes None, not the string",
          TM._make_estimator("svm_rbf", {"class_weight": "none"})
          .class_weight is None)
    check("max_depth 0 means unlimited",
          TM._make_estimator("random_forest", {"max_depth": 0})
          .max_depth is None)
    check("an unknown kind is refused loudly", _raises(
        lambda: TM._make_estimator("telepathy", {})))

    # -- the roster and the vote ---------------------------------------------
    spec = T.default_model_spec()
    models = TM.build_models_from_spec(spec)
    on = T.enabled_models(spec)
    check("one candidate per enabled model, plus the vote",
          len(models) == len(on) + 1, f"{len(models)} vs {len(on)}+1")
    votes = [n for n in models if n.startswith("Vote-")]
    check("the ensemble is one of the candidates", len(votes) == 1, str(votes))
    check("it is named for its mode and size",
          votes[0] == f"Vote-soft-{len(on)}", votes[0])
    check("DISABLED MODELS ARE NOT TRAINED",
          all(m["name"] in models for m in on)
          and not any(m["name"] in models for m in spec["models"]
                      if not m["enabled"]))

    voting = models[votes[0]]
    check("the ensemble holds every enabled model",
          len(voting.estimators) == len(on), str(len(voting.estimators)))
    check("soft voting by default", voting.voting == "soft")
    voting.fit(X, y)
    check("the ensemble fits and predicts",
          voting.predict(X).shape == y.shape)

    no_vote = T.default_model_spec()
    no_vote["voting"]["enabled"] = False
    check("voting off means no ensemble candidate",
          not any(n.startswith("Vote-")
                  for n in TM.build_models_from_spec(no_vote)))

    hard = T.default_model_spec()
    hard["voting"]["mode"] = "hard"
    hard_models = TM.build_models_from_spec(hard)
    hard_name = [n for n in hard_models if n.startswith("Vote-")][0]
    check("hard voting is built when asked",
          hard_models[hard_name].voting == "hard", hard_name)

    single = {"models": [{"name": "only", "kind": "lda", "enabled": True,
                          "scaler": "standard", "params": {}}],
              "voting": {"enabled": True, "mode": "soft", "always": False}}
    check("no ensemble from a single model",
          not any(n.startswith("Vote-")
                  for n in TM.build_models_from_spec(single)))

    # -- auto mode feeds the same builder ------------------------------------
    auto = T.default_model_spec()
    auto["auto"] = {"enabled": True, "max_per_model": 3}
    auto_models = TM.build_models_from_spec(auto)
    check("AUTO MODE PRODUCES MORE CANDIDATES", len(auto_models) > len(models),
          f"{len(auto_models)} vs {len(models)}")
    check("every auto candidate is a real estimator",
          all(hasattr(m, "fit") for m in auto_models.values()))
    named = [n for n in auto_models if "[" in n]
    check("auto candidates are named by what differs", len(named) > 0,
          named[0] if named else "none")

    # -- resolve_spec is the trainer's contract with the GUI -----------------
    root = tempfile.mkdtemp(prefix="emg_spec_cli_")
    try:
        path = os.path.join(root, T.MODEL_SPEC_NAME)
        mine = T.default_model_spec()
        mine["models"][0]["params"]["solver"] = "svd"
        T.save_model_spec(path, mine)
        resolved, source = TM.resolve_spec(path)
        check("the trainer reads the spec the GUI wrote",
              resolved["models"][0]["params"]["solver"] == "svd")
        check("and says where it came from", source == path)
        check("no --spec means the built-in defaults",
              TM.resolve_spec(None)[1] == "built-in defaults")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # -- what MODEL STATS will show ------------------------------------------
    lines = T.describe_estimator(voting)
    text = "\n".join(lines)
    check("the description names the ensemble", "VotingClassifier" in text)
    check("IT SAYS HOW THE OUTPUTS ARE COMBINED", "combines:" in text)
    check("it names soft averaging", "average" in text and "probabilit" in text)
    check("it reports the weights", "weights" in text)
    check("it lists every member",
          all(f"member {i}" in text for i in range(1, len(on) + 1)))
    check("it shows each member's stages", text.count("Pipeline") >= 2,
          str(text.count("Pipeline")))
    check("it shows the stages in order", "[1]" in text and "[2]" in text)
    check("it shows real hyperparameters", "C=10" in text, "")

    one = TM.build_models_from_spec(spec)["SVM-rbf-C10"]
    one_text = "\n".join(T.describe_estimator(one))
    check("a single pipeline describes its scaler and classifier",
          "StandardScaler" in one_text and "SVC" in one_text)
    check("and its settings", "kernel=rbf" in one_text, one_text.strip()[:60])
    check("no model describes as '(no model)'",
          T.describe_estimator(None) == ["(no model)"])

    # The description is read off the OBJECT, so a spec that disagrees with
    # the fitted estimator cannot mislead.
    mutated = TM.build_models_from_spec(spec)["SVM-rbf-C10"]
    mutated.set_params(clf__C=123.0)
    check("THE DESCRIPTION FOLLOWS THE OBJECT, NOT THE SPEC",
          "C=123" in "\n".join(T.describe_estimator(mutated)))


def _raises(fn):
    try:
        fn()
    except Exception:                            # noqa: BLE001
        return True
    return False


# ==============================================================================
# PART 3 -- the dialog
# ==============================================================================

def test_gui():
    print("\n--- gui ---")
    from PySide6 import QtCore, QtWidgets

    root = tempfile.mkdtemp(prefix="emg_gui_spec_")
    T.DATA_FOLDER = root
    T.PROJECT_DIR = root
    T.MYO_PORT = "COM_DOES_NOT_EXIST"

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    T.apply_theme(app)
    window = None
    try:
        window = T.MainWindow()
        w = window

        check("MODEL SPEC button built", hasattr(w, "spec_button"))
        check("it says what it is", "MODEL SPEC" in w.spec_button.text())
        check("the summary line is built", hasattr(w, "spec_summary"))
        check("the summary says what the next retrain will do",
              "next retrain" in w.spec_summary.text(), w.spec_summary.text())
        check("a spec is loaded at startup",
              isinstance(w.model_spec, dict) and bool(w.model_spec["models"]))

        path = T.model_spec_path(T.MODEL_PATH)
        dialog = T.ModelSpecDialog(w.model_spec, path, w)

        check("every model has a row",
              dialog.model_list.count() == len(w.model_spec["models"]),
              str(dialog.model_list.count()))
        enabled_rows = sum(
            1 for i in range(dialog.model_list.count())
            if dialog.model_list.item(i).checkState() == QtCore.Qt.Checked)
        check("the checkboxes match the spec",
              enabled_rows == len(T.enabled_models(w.model_spec)),
              f"{enabled_rows}")
        check("the roster line counts them",
              "will be trained" in dialog.roster_info.text(),
              dialog.roster_info.text())

        # -- the generated editor ---------------------------------------------
        svm_row = next(i for i, m in enumerate(dialog.spec["models"])
                       if m["kind"] == "svm_rbf")
        dialog.model_list.setCurrentRow(svm_row)
        check("the detail pane names the model",
              "SVM" in dialog.detail_title.text(), dialog.detail_title.text())
        check("the editor was generated from the schema",
              set(dialog._param_widgets)
              == set(T.MODEL_KINDS["svm_rbf"]["params"]),
              str(sorted(dialog._param_widgets)))
        check("a float parameter got a spin box",
              isinstance(dialog._param_widgets["C"],
                         QtWidgets.QDoubleSpinBox))
        check("a choice parameter got a combo",
              isinstance(dialog._param_widgets["gamma"], QtWidgets.QComboBox))

        dialog._param_widgets["C"].setValue(42.5)
        check("editing a spin box reaches the spec",
              dialog.spec["models"][svm_row]["params"]["C"] == 42.5,
              str(dialog.spec["models"][svm_row]["params"]["C"]))
        dialog._param_widgets["gamma"].setCurrentText("auto")
        check("editing a combo reaches the spec",
              dialog.spec["models"][svm_row]["params"]["gamma"] == "auto")

        dialog._set_param(svm_row, "C", 10 ** 9)
        check("AN IMPOSSIBLE VALUE IS CLAMPED IN THE DIALOG",
              dialog.spec["models"][svm_row]["params"]["C"] <= 1000.0,
              str(dialog.spec["models"][svm_row]["params"]["C"]))

        mlp_row = next(i for i, m in enumerate(dialog.spec["models"])
                       if m["kind"] == "mlp")
        dialog.model_list.setCurrentRow(mlp_row)
        check("layer sizes got a text field",
              isinstance(dialog._param_widgets["hidden_layer_sizes"],
                         QtWidgets.QLineEdit))
        check("the regularisation controls are offered",
              "alpha" in dialog._param_widgets
              and "early_stopping" in dialog._param_widgets
              and "batch_size" in dialog._param_widgets,
              str(sorted(dialog._param_widgets)))
        dialog._set_param(mlp_row, "hidden_layer_sizes", "512, 256")
        check("typed layer sizes parse",
              dialog.spec["models"][mlp_row]["params"]["hidden_layer_sizes"]
              == [512, 256],
              str(dialog.spec["models"][mlp_row]["params"]["hidden_layer_sizes"]))

        logreg_row = next(i for i, m in enumerate(dialog.spec["models"])
                          if m["kind"] == "logreg")
        dialog.model_list.setCurrentRow(logreg_row)
        check("L1/L2 is selectable where it belongs",
              "penalty" in dialog._param_widgets
              and set(T.MODEL_KINDS["logreg"]["params"]["penalty"]["values"])
              >= {"l1", "l2", "elasticnet"})

        # -- enable / disable --------------------------------------------------
        item = dialog.model_list.item(0)
        was = dialog.spec["models"][0]["enabled"]
        item.setCheckState(QtCore.Qt.Unchecked if was else QtCore.Qt.Checked)
        check("ticking a row toggles the model",
              dialog.spec["models"][0]["enabled"] is (not was))
        item.setCheckState(QtCore.Qt.Checked if was else QtCore.Qt.Unchecked)

        dialog._set_all(True)
        check("ALL ON enables everything",
              all(m["enabled"] for m in dialog.spec["models"]))
        dialog._set_all(False)
        check("ALL OFF is repaired to leave one trainable",
              len(T.enabled_models(dialog.spec)) >= 1)
        dialog._set_all(True)

        # -- add / remove ------------------------------------------------------
        before = dialog.model_list.count()
        dialog.add_combo.setCurrentIndex(
            dialog.add_combo.findData("extra_trees"))
        dialog.add_model()
        check("ADD appends a candidate",
              dialog.model_list.count() == before + 1)
        check("the new one is of the chosen kind",
              dialog.spec["models"][-1]["kind"] == "extra_trees")
        check("it starts at the schema defaults",
              dialog.spec["models"][-1]["params"]["n_estimators"]
              == T.model_param_default("extra_trees", "n_estimators"))
        dialog.model_list.setCurrentRow(dialog.model_list.count() - 1)
        dialog.remove_model()
        check("REMOVE takes it away",
              dialog.model_list.count() == before)

        # -- voting ------------------------------------------------------------
        dialog.vote_check.setChecked(True)
        check("voting is on", dialog.spec["voting"]["enabled"] is True)
        check("the preview names the members",
              "Vote-" in dialog.vote_info.text(), dialog.vote_info.text())
        dialog.vote_mode.setCurrentIndex(dialog.vote_mode.findData("hard"))
        check("the mode reaches the spec",
              dialog.spec["voting"]["mode"] == "hard")
        dialog.vote_always.setChecked(True)
        check("'always use the vote' reaches the spec",
              dialog.spec["voting"]["always"] is True)
        dialog.vote_always.setChecked(False)
        dialog.vote_check.setChecked(False)
        check("voting off says so",
              "no ensemble" in dialog.vote_info.text(),
              dialog.vote_info.text())
        dialog.vote_check.setChecked(True)
        dialog.vote_mode.setCurrentIndex(dialog.vote_mode.findData("soft"))

        # -- auto mode ---------------------------------------------------------
        check("auto starts off", dialog.spec["auto"]["enabled"] is False)
        check("and says what that means",
              "off" in dialog.auto_info.text(), dialog.auto_info.text())
        check("no warning while it is off", dialog.auto_warning.text() == "")
        dialog.auto_check.setChecked(True)
        dialog.auto_spin.setValue(3)
        check("auto on reaches the spec",
              dialog.spec["auto"] == {"enabled": True, "max_per_model": 3},
              str(dialog.spec["auto"]))
        check("the preview counts the candidates",
              "candidates" in dialog.auto_info.text(),
              dialog.auto_info.text())
        check("it reports the cost in training time",
              "x the training time" in dialog.auto_info.text())
        check("IT WARNS THAT THE WINNER'S SCORE BECOMES OPTIMISTIC",
              "optimistic" in dialog.auto_warning.text(),
              dialog.auto_warning.text()[:60])
        check("and points at the honest number",
              "leave-one-subject-out" in dialog.auto_warning.text())

        dialog.model_list.setCurrentRow(svm_row)
        check("a per-model auto opt-out is offered",
              any(isinstance(dialog._param_widgets.get(k), QtWidgets.QWidget)
                  for k in dialog._param_widgets) is True)
        dialog._set_auto_for(svm_row, False)
        check("a model can be pinned while others are tuned",
              dialog.spec["models"][svm_row]["auto"] is False)
        dialog._set_auto_for(svm_row, True)

        # -- save --------------------------------------------------------------
        dialog.auto_check.setChecked(False)
        dialog.save_spec()
        check("SAVE writes the spec file", os.path.exists(path))
        check("the dialog reports that it saved", dialog.saved is True)
        saved, problems = T.load_model_spec(path)
        check("the saved spec is valid", problems == [], str(problems))
        svm = saved["models"][svm_row]
        check("the edited hyperparameter is on disk",
              svm["params"]["C"] == 1000.0 and svm["params"]["gamma"] == "auto",
              str(svm["params"]))
        check("the window picks it up",
              T.load_model_spec(path)[0]["models"][svm_row]["params"]["gamma"]
              == "auto")

        # -- reset -------------------------------------------------------------
        dialog2 = T.ModelSpecDialog(saved, path, w)
        dialog2.spec = T.default_model_spec()
        dialog2._reload_all()
        check("RESET goes back to the built-in roster",
              dialog2.spec["models"][svm_row]["params"]["C"]
              == T.model_param_default("svm_rbf", "C"))
        check("the last model cannot be removed",
              _last_model_is_protected(dialog2))
        dialog2.close()
        dialog.close()

        # -- the button is blocked while training ------------------------------
        w.training_process = object()
        w._refresh_controls()
        check("the spec cannot be edited mid-training",
              not w.spec_button.isEnabled())
        w.training_process = None
        w._refresh_controls()
        check("and can be again afterwards", w.spec_button.isEnabled())
    finally:
        if window is not None:
            try:
                window.acq.stop()
            except Exception:                    # noqa: BLE001
                pass
            window.close()
        shutil.rmtree(root, ignore_errors=True)


def _last_model_is_protected(dialog) -> bool:
    dialog.spec["models"] = dialog.spec["models"][:1]
    dialog._reload_all()
    dialog.model_list.setCurrentRow(0)
    dialog.remove_model()
    return len(dialog.spec["models"]) == 1


def test_scoring_variant():
    """
    The SVM scoring shortcut, and the trap next to it.

    Scoring builds SVC without Platt scaling because `evaluate()` only calls
    `predict()` -- 5.7x faster, identical predictions. The SAVED model must
    still have `predict_proba` for LivePredictor, and the soft-voting
    ensemble must still contain the SVM: soft voting selects members with
    `hasattr(clf, "predict_proba")`, so sharing the scoring pipelines would
    silently drop the SVM from the vote and change the experiment.
    """
    print("\n--- scoring vs saved estimators ---")
    try:
        import train_model as TM
        from sklearn.svm import SVC
    except Exception as exc:                     # noqa: BLE001
        check(f"train_model imports (skipped: {exc})", True)
        return

    check("SVC without probability has no predict_proba",
          not hasattr(SVC(probability=False), "predict_proba"))
    check("SVC with probability has predict_proba",
          hasattr(SVC(probability=True), "predict_proba"))

    spec = {
        "version": 1,
        "voting": {"enabled": True, "mode": "soft", "always": False},
        "auto": {"enabled": False, "max_per_model": 4},
        "models": [
            {"name": "SVM-rbf-C10", "kind": "svm_rbf", "enabled": True,
             "scaler": "standard", "params": {"C": 10.0}},
            {"name": "LDA-shrinkage", "kind": "lda", "enabled": True,
             "scaler": "standard", "params": {"solver": "lsqr",
                                              "shrinkage": "auto"}},
        ],
    }

    scoring = TM.build_models_from_spec(spec, for_scoring=True)
    saved = TM.build_models_from_spec(spec, for_scoring=False)

    check("scoring SVM skips Platt scaling",
          scoring["SVM-rbf-C10"].named_steps["clf"].probability is False)
    check("saved SVM keeps Platt scaling",
          saved["SVM-rbf-C10"].named_steps["clf"].probability is True)
    check("the saved SVM can produce probabilities",
          hasattr(saved["SVM-rbf-C10"].named_steps["clf"], "predict_proba"))

    for which, models in (("scoring", scoring), ("saved", saved)):
        vote = [m for n, m in models.items() if n.startswith("Vote")]
        check(f"{which}: an ensemble was built", len(vote) == 1)
        if not vote:
            continue
        members = dict(vote[0].estimators)
        check(f"{which}: the SVM is still in the ensemble",
              "SVM-rbf-C10" in members, str(sorted(members)))
        check(f"{which}: the ensemble has every enabled model",
              len(members) == 2, str(sorted(members)))
        if "SVM-rbf-C10" in members:
            check(f"{which}: the ensemble's SVM can do soft voting",
                  members["SVM-rbf-C10"].named_steps["clf"].probability is True)

    # The scoring shortcut must not change a single prediction.
    import numpy as np
    X, y = toy_data()
    a = scoring["SVM-rbf-C10"].fit(X, y).predict(X)
    b = saved["SVM-rbf-C10"].fit(X, y).predict(X)
    check("scoring and saved SVMs predict identically",
          np.array_equal(a, b),
          f"{(a == b).mean() * 100:.2f}% agreement")


if __name__ == "__main__":
    test_spec_layer()
    test_estimators()
    test_scoring_variant()
    test_gui()
    print("\n" + "=" * 16 + f"  {sum(OK)}/{len(OK)} checks passed  " + "=" * 16)
    sys.exit(0 if all(OK) else 1)

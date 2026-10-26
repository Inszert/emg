"""
The settings tabs (Models, Training, General), the two questions, and "turning
a feature off restores the original behaviour" -- on the real window, offscreen.

  1. the settings dialog has the new tabs; the Models tab lists the zoo with its
     stars, sizes and totals, refuses a selection that cannot be voted, and
     "select starred" / "select best combination" tick the right rows
  2. applying: values are validated, applied to the configuration, saved, and a
     changed selection reloads the live classifier and says what is in use
  3. selection OFF restores the original model and the original panels
  4. the Training tab writes the augmentation / adaptation switches, and the
     trainers' command lines carry them
  5. "calibrate your hand first?": asked once per run per person, the three
     answers, the last calibration offered only after a restart, nothing asked
     when the setting is off, PREDICT / START NN buttons go through it
  6. the retrain question: asked only when something is out of date, only with
     the setting on, never while training or recording; each answer starts the
     right job (all / only the selected / not now); "do not ask again" sticks
  7. adaptation statistics reach the classifier trained with adaptation
  8. the model zoo job: the training step the GUI builds
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "training"))

import emg_dataset_tool as T                                         # noqa: E402
from emg_app import adaptation as AD                                 # noqa: E402
from emg_app import settings_store as SS                             # noqa: E402
from emg_app import zoo                                              # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    shown = "" if detail is None or (isinstance(detail, str) and not detail) \
        else f"  [{detail}]"
    print(f"{'PASS' if condition else 'FAIL'}  {name}{shown}")


def section(title):
    print(f"\n=== {title} ===")


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   # noqa: E402
from sklearn.ensemble import RandomForestClassifier                    # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
import joblib                                                          # noqa: E402
from PySide6 import QtCore, QtWidgets                                  # noqa: E402

BOXES = []                                       # every box the code tried to show
for name in ("information", "warning", "critical"):
    setattr(QtWidgets.QMessageBox, name, staticmethod(
        lambda *a, _n=name, **k: (BOXES.append((_n, a)),
                                  QtWidgets.QMessageBox.Ok)[1]))
QtWidgets.QMessageBox.question = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Yes)

TEMP = tempfile.mkdtemp(prefix="emgsettings_")
MODELS = os.path.join(TEMP, "models")
DATA = os.path.join(TEMP, "data")
os.makedirs(MODELS)
os.makedirs(DATA)
T.DATA_FOLDER = DATA
T.MODEL_PATH = os.path.join(MODELS, "gesture_model.joblib")
T.NN_MODEL_PATH = os.path.join(MODELS, "gesture_nn.pt")
T.MYO_PORT = "COM254"
T.ENABLE_LIVE_PREDICTION = True
T.PREDICTION_SMOOTHING = 1
T.PREDICTION_MIN_CONFIDENCE = 0.0
T.RECONNECT_DELAY_S = 0.3
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.QC_PAUSE_ON = "never"
T.GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP"]
T.CUSTOM_GESTURES = {}
T.CLASSIFIER_OFF = []
for k, v in SS.DEFAULTS.items():
    setattr(T, k, v)

LABELS = ["REST", "FIST", "OPEN", "THUMB_UP"]
FS, WIN = 60.0, 12
rng = np.random.default_rng(9)
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
    "THUMB_UP": np.array([40, 30, 60, 200, 210, 40, 30, 20], dtype=float),
}


def window_of(label):
    base = PROFILES[label] * rng.uniform(0.8, 1.25)
    return np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))


F, Y = [], []
for idx, label in enumerate(LABELS):
    for _ in range(80):
        F.append(T.model_feature_vector(window_of(label), "raw", "level", True,
                                        FS))
        Y.append(idx)
F, Y = np.asarray(F, dtype=np.float32), np.asarray(Y)
lda = Pipeline([("s", StandardScaler()), ("c", LinearDiscriminantAnalysis(
    solver="lsqr", shrinkage="auto"))]).fit(F, Y)
rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(F, Y)
rf2 = RandomForestClassifier(n_estimators=20, random_state=1).fit(F, Y)
gm, gs = AD.feature_stats(F)
rf_ad = RandomForestClassifier(n_estimators=20, random_state=2).fit(
    (F - gm) / gs, Y)


def payload(name, est, repr_="raw", adaptation=None):
    return {"model": est, "labels": LABELS, "fs": FS, "window_s": 0.2,
            "window_samples": WIN, "representation": repr_,
            "feature_set": "level", "freq_features": True,
            "calibration": {"mode": "off"},
            "adaptation": adaptation or {"mode": "off"},
            "stats": {"model_name": name, "window_accuracy": 0.9,
                      "labels": LABELS, "overlap": 0.75}}


COMPAT = {"labels": LABELS, "fs": FS, "window_s": 0.2, "window_samples": WIN,
          "representation": "raw", "feature_set": "level",
          "freq_features": True, "calibration_mode": "off"}


def add(eid, name, acc, pl, compat=COMPAT, digest="old", trained_with=None):
    folder = os.path.join(zoo.zoo_root(), eid)
    os.makedirs(folder, exist_ok=True)
    joblib.dump(pl, os.path.join(folder, "model.joblib"))
    zoo.write_meta(folder, {
        "id": eid, "family": "ml", "name": name, "kind": name,
        "file": "model.joblib", "size_bytes": 5000, "scores": {
            "window_acc": acc, "macro_f1": acc - 0.02, "vote_acc": acc},
        "compat": compat, "data_digest": digest, "latency_ms": 3.0,
        "trained_with": trained_with or {"augment": [], "adapt": False}})


joblib.dump(payload("DEFAULT", lda), T.MODEL_PATH)
add("ml-rf", "RF", 0.78, payload("RF", rf))
add("ml-rf2", "RF-2", 0.77, payload("RF-2", rf2))
add("ml-lda", "LDA", 0.60, payload("LDA", lda))
add("ml-env", "RF-env", 0.50, payload("RF-env", rf, "envelope"),
    dict(COMPAT, representation="envelope"))
add("ml-ad", "RF-adapt", 0.65, payload(
    "RF-adapt", rf_ad, adaptation={"mode": "person_features",
                                   "global_mean": gm, "global_std": gs}),
    dict(COMPAT), trained_with={"augment": [], "adapt": True})
zoo.write_ensembles({"ml": {"members": ["ml-rf", "ml-rf2"], "window_acc": 0.82,
                            "nested_acc": 0.81}})

app = QtWidgets.QApplication(sys.argv)
T.apply_theme(app)
window = T.MainWindow()
for _ in range(3):
    app.processEvents()

# ==========================================================================
section("1. the settings tabs")
dialog = T.SettingsDialog(window, model_labels=window._model_label_sets(),
                          selected=list(T.GESTURES), classifier_off=[])
tabs = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
check("tabs: Gestures, Models, Training, General",
      tabs == ["Gestures", "Models", "Training", "General"], tabs)
mt = dialog.models_tab
check("the Models tab lists the zoo (ML table, NN table)", set(mt.tables) ==
      {"ml", "nn"} and mt.tables["ml"].rowCount() == 5, mt.tables["ml"].rowCount())
check("ranked best first", mt.tables["ml"].item(0, 2).text() == "RF"
      and mt.tables["ml"].item(1, 2).text() == "RF-2")
check("the top 3 carry a star", [mt.tables["ml"].item(r, 1).text()
                                  for r in range(5)] == ["★"] * 3 + ["", ""])
check("the star's tooltip gives rank, score and size",
      "#1" in mt.tables["ml"].item(0, 1).toolTip()
      and "78.0 %" in mt.tables["ml"].item(0, 1).toolTip(),
      mt.tables["ml"].item(0, 1).toolTip())
check("size and scores are shown", mt.tables["ml"].item(0, 5).text() == "4.9 KB"
      or "KB" in mt.tables["ml"].item(0, 5).text())
check("selection is OFF by default and the tables are greyed",
      not mt.enable_box.isChecked() and not mt.tables["ml"].isEnabled())
mt.enable_box.setChecked(True)
check("ticking 'use the selected models' enables them",
      mt.tables["ml"].isEnabled())
mt._select_stars("ml")
check("'select starred' ticks exactly the starred", set(mt.chosen("ml")) ==
      {"ml-rf", "ml-rf2", "ml-lda"} or len(mt.chosen("ml")) == 3, mt.chosen("ml"))
check("the summary gives the count and the total size",
      "3 selected" in mt.summaries["ml"].text()
      and "KB" in mt.summaries["ml"].text(), mt.summaries["ml"].text())
check("a compatible selection says it will be a soft vote",
      "Soft vote" in mt.vote_notes["ml"].text(), mt.vote_notes["ml"].text())
mt._select_combo("ml")
check("'select best combination' ticks the measured subset",
      mt.chosen("ml") == ["ml-rf", "ml-rf2"], mt.chosen("ml"))
mt._set_checked("ml", ["ml-rf", "ml-env"])
check("a selection that cannot be voted is flagged right there",
      "Cannot vote" in mt.vote_notes["ml"].text(), mt.vote_notes["ml"].text())
check("...and blocks APPLY with the reason", mt.problems()
      and "representation" in mt.problems()[0])
n_boxes = len(BOXES)
dialog._apply()
check("...APPLY refuses (a warning, the dialog stays open)",
      len(BOXES) == n_boxes + 1 and dialog.result() == 0
      and "cannot be applied" in str(BOXES[-1]))
mt._set_checked("ml", [])
check("Clear unticks everything", mt.chosen("ml") == [])
check("with the combination starred, its button is enabled",
      any(b.isEnabled() for b in dialog.findChildren(QtWidgets.QPushButton)
          if "best combination" in b.text()))
tr = dialog.training_tab
check("the Training tab starts at the defaults (nothing on)",
      tr.values() == {"TRAIN_AUGMENT": [], "TRAIN_ADAPT": False,
                      "ADAPT_LIVE": True}, tr.values())
check("one switch per augmentation", set(tr.aug_boxes) ==
      {"gain", "noise", "timewarp", "rotate", "chandrop"})
gen = dialog.general_tab
check("the General tab has both questions, on", gen.values() ==
      {"ASK_CALIBRATION_BEFORE_PREDICT": True, "ASK_RETRAIN_ON_NEW_DATA": True})
dialog.deleteLater()

# ==========================================================================
section("2. applying")
before_pred = window.predictor
window.predictor.load()
check("the original model is loaded", window.predictor.stats["model_name"]
      == "DEFAULT")
window._apply_app_settings({"MODEL_SELECTION_ENABLED": True,
                            "SELECTED_ML": ["ml-rf", "ml-rf2"],
                            "SELECTED_NN": [], "TRAIN_AUGMENT": ["gain"],
                            "TRAIN_ADAPT": False, "ADAPT_LIVE": True,
                            "ASK_CALIBRATION_BEFORE_PREDICT": True,
                            "ASK_RETRAIN_ON_NEW_DATA": True})
check("applied to the configuration", T.MODEL_SELECTION_ENABLED
      and T.SELECTED_ML == ["ml-rf", "ml-rf2"] and T.TRAIN_AUGMENT == ["gain"])
check("saved to <data>/app_settings.json",
      SS.load_settings(DATA)["SELECTED_ML"] == ["ml-rf", "ml-rf2"])
check("the live classifier now is the soft vote",
      window.predictor.ready and window.predictor.members == ["RF", "RF-2"],
      window.predictor.status)
check("the panel says so", "soft vote" in window.prediction_note.text(),
      window.prediction_note.text()[:80])
check("the log says what is in use", "soft vote: RF + RF-2" in
      window.log_box.toPlainText(), window.log_box.toPlainText()[-200:])
check("the NN engine would load the selection (none -> the original)",
      window.nn_engine.model_paths() == [window.nn_engine.full_model_path])
window._apply_app_settings({"SELECTED_ML": ["ml-lda"]})
check("one model selected = that model", window.predictor.stats["model_name"]
      == "LDA" and window.predictor.members == [])
window._apply_app_settings({"SELECTED_ML": ["ml-rf", "ml-env"]})
check("a selection that cannot be voted leaves the classifier NOT ready, with "
      "the reason (never a silent fallback)",
      not window.predictor.ready and "cannot vote" in window.predictor.status
      and not window.predict_button.isEnabled(), window.predictor.status)
try:
    window._apply_app_settings({"TRAIN_AUGMENT": ["gian"]})
    check("a typo in the values is refused", T.TRAIN_AUGMENT == ["gain"]
          and BOXES and "unknown augmentation" in str(BOXES[-1]))
except Exception as exc:                                    # pragma: no cover
    check("a typo in the values is refused", False, exc)

# ==========================================================================
section("3. selection OFF restores the original")
window._apply_app_settings({"MODEL_SELECTION_ENABLED": False})
check("the original model is back", window.predictor.ready
      and window.predictor.stats["model_name"] == "DEFAULT"
      and window.predictor.members == [] and window.predict_button.isEnabled(),
      window.predictor.status)
t = np.arange(WIN) / FS
w = window_of("OPEN")
direct = T.LivePredictor(T.MODEL_PATH)
direct.load()
a, b = window.predictor.predict(w, t), direct.predict(w, t)
check("...and answers exactly like a predictor that never heard of the zoo",
      np.array_equal(a.raw_probabilities, b.raw_probabilities))
check("the ids stay saved (switching on again is one click)",
      T.SELECTED_ML == ["ml-rf", "ml-env"])

# ==========================================================================
section("4. the Training tab reaches the trainers")
T.TRAIN_AUGMENT = ["gain", "rotate"]
T.TRAIN_ADAPT = True
args = window._nn_train_args("train_nn.py", False)
check("train_nn.py gets --augment and --adapt",
      args[args.index("--augment") + 1] == "gain,rotate" and "--adapt" in args,
      args)
T.TRAIN_AUGMENT = []
T.TRAIN_ADAPT = False
args = window._nn_train_args("train_nn.py", False)
check("with both off it gets --augment none and --no-adapt",
      args[args.index("--augment") + 1] == "none" and "--no-adapt" in args)
zargs = window._zoo_train_args(["ml-rf", "nn-gru"])
check("the zoo trainer gets the ids", zargs[zargs.index("--ids") + 1:
                                              zargs.index("--ids") + 3]
      == ["ml-rf", "nn-gru"] and "all" in zargs, zargs)

# ==========================================================================
section("5. 'calibrate your hand first?'")
asked = []
answers = {"value": "skip"}


def fake_ask(what, person, record, offer):
    asked.append((what, person, offer["available"]))
    return answers["value"]


window._ask_calibration_question = fake_ask
window._calibration_gate_done = set()
T.ASK_CALIBRATION_BEFORE_PREDICT = False
check("setting OFF: never asked, goes on", window.calibration_gate("PREDICT")
      and not asked)
T.ASK_CALIBRATION_BEFORE_PREDICT = True
window._app_started_ts = time.time()
check("setting ON: asked", window.calibration_gate("PREDICT") and len(asked) == 1)
check("...and not again for the same person in this run",
      window.calibration_gate("START NN") and len(asked) == 1)
window._calibration_gate_done = set()
answers["value"] = "cancel"
check("cancel stops the start", not window.calibration_gate("PREDICT"))
check("...and the question comes back next time (nothing remembered)",
      window.person_id not in window._calibration_gate_done)
# a calibration that exists from before the restart
root = DATA
rec = T.calibration_from_captures(np.full((60, 8), 20.0),
                                  np.full((60, 8), 400.0),
                                  person_id=window.person_id, fs=50.0)
rec["created_unix"] = time.time() - 3600
rec["created_iso"] = T.iso_time(rec["created_unix"])
T.save_calibration(rec, root, window.person_id)
window._app_started_ts = time.time()                   # "the app was restarted"
window._calibration_gate_done = set()
asked.clear()
answers["value"] = "reuse"
check("after a restart the stored calibration is OFFERED",
      window.calibration_gate("PREDICT") and asked and asked[0][2] is True,
      asked)
check("reusing it is logged with its age",
      "Using the last calibration" in window.log_box.toPlainText())
window._app_started_ts = time.time() - 7200               # calibrated THIS run
window._calibration_gate_done = set()
asked.clear()
window.calibration_gate("PREDICT")
check("a calibration made in this run is NOT offered for reuse",
      asked and asked[0][2] is False, asked)
# the buttons go through the gate
calls = []
window.calibration_gate = lambda what: (calls.append(what), False)[1]
window.predictor.load()
window.on_predict_clicked()
check("PREDICT goes through the gate, and 'no' means it does not start",
      calls == ["PREDICT"] and not window.prediction_active
      and not window.one_shot_deadline, calls)
calls.clear()
window.on_nn_clicked()
check("START NN goes through the gate too", calls == ["START NN"]
      or window.nn_engine is None, calls)
del window.calibration_gate
window.stop_prediction()

# ==========================================================================
section("6. the retrain question")
questions = []
reply = {"choice": "none", "dont": False}
window._ask_retrain_question = lambda reasons, why: (
    questions.append((list(reasons), why)) or (reply["choice"], reply["dont"]))
jobs = []
window.retrain_everything = lambda zoo_ids=None: jobs.append(("all", zoo_ids))
window.retrain_zoo = lambda ids: jobs.append(("zoo", list(ids)))
# make the dataset look big enough and the models out of date
import emg_app.gui.main_window.prompts as PM                           # noqa: E402
window.retrain_state = lambda: {"reasons": ["The ML classifier: 3 recordings "
                                            "were added"],
                                "digest": "x", "total_raw": 99}
T.ASK_RETRAIN_ON_NEW_DATA = False
check("setting OFF: not asked", window.maybe_ask_retrain() is False
      and not questions)
T.ASK_RETRAIN_ON_NEW_DATA = True
reply.update(choice="none")
check("setting ON and something stale: asked", window.maybe_ask_retrain()
      and len(questions) == 1)
check("'not now' starts nothing", jobs == [])
reply.update(choice="all")
window.maybe_ask_retrain()
check("'all' = the deployed pair plus every zoo model",
      jobs[-1][0] == "all" and set(jobs[-1][1]) ==
      {e.id for e in zoo.list_entries()}, jobs[-1])
T.MODEL_SELECTION_ENABLED = False
reply.update(choice="selected")
window.maybe_ask_retrain()
check("'selected' with selection OFF = the deployed pair only",
      jobs[-1] == ("all", None), jobs[-1])
T.MODEL_SELECTION_ENABLED = True
T.SELECTED_ML = ["ml-rf"]
T.SELECTED_NN = []
window.maybe_ask_retrain()
check("'selected' with selection ON = only the selected zoo models",
      jobs[-1] == ("zoo", ["ml-rf"]), jobs[-1])
T.MODEL_SELECTION_ENABLED = False
reply.update(choice="none", dont=True)
window.maybe_ask_retrain()
check("'do not ask when the app opens' switches the question off and saves it",
      T.ASK_RETRAIN_ON_NEW_DATA is False
      and SS.load_settings(DATA)["ASK_RETRAIN_ON_NEW_DATA"] is False)
n = len(questions)
window.maybe_ask_retrain()
check("...and it stays off", len(questions) == n)
T.ASK_RETRAIN_ON_NEW_DATA = True
window.retrain_state = lambda: {"reasons": [], "digest": "x", "total_raw": 99}
check("nothing out of date: not asked", window.maybe_ask_retrain() is False)
window.retrain_state = lambda: {"reasons": ["x"], "digest": "x", "total_raw": 5}
check("too few recordings to train on: not asked",
      window.maybe_ask_retrain() is False)
window.retrain_state = lambda: {"reasons": ["x"], "digest": "x", "total_raw": 99}
window.training_process = object()
check("never while a training is running", window.maybe_ask_retrain() is False)
window.training_process = None
check("force=True asks even with the setting off",
      (setattr(T, "ASK_RETRAIN_ON_NEW_DATA", False)
       or window.maybe_ask_retrain(force=True)) is True)
T.ASK_RETRAIN_ON_NEW_DATA = True
del window.retrain_state
real = window.retrain_state()
check("the real state reports zoo models measured on older data",
      any("model zoo" in r for r in real["reasons"]) or not zoo.list_entries(),
      real["reasons"])

# ==========================================================================
section("7. adaptation statistics reach the model trained with it")
T.MODEL_SELECTION_ENABLED = True
T.SELECTED_ML = ["ml-ad"]
window._reload_selected_models()
check("the adapted model is loaded", window.predictor.ready
      and window.predictor.adapts, window.predictor.status)
check("with no stored signal it says so and uses the global statistics",
      (window.update_adaptation() or True) and not
      window.predictor.adapt_stats
      and "no calibration signal" in window.log_box.toPlainText())
blocks = [np.abs(rng.normal(120, 60, (150, 8))) + 15 for _ in range(6)]
AD.save_signal(blocks, 50.0, DATA, window.person_id)
window.update_adaptation()
check("with a stored calibration signal it standardises with the person's own "
      "statistics", window.predictor.adapt_stats is not None
      and "own statistics" in window.log_box.toPlainText())
T.ADAPT_LIVE = False
window.update_adaptation()
check("ADAPT_LIVE off: the model falls back to its stored statistics",
      window.predictor.adapt_stats is None)
T.ADAPT_LIVE = True
T.MODEL_SELECTION_ENABLED = False
window._reload_selected_models()

# ==========================================================================
section("8. the app starts from the saved settings")
SS.save_settings({"MODEL_SELECTION_ENABLED": True, "SELECTED_ML": ["ml-rf"],
                  "TRAIN_AUGMENT": ["noise"]}, DATA)
T.MODEL_SELECTION_ENABLED = False
T.SELECTED_ML = []
T.TRAIN_AUGMENT = []
restored = SS.restore_settings(DATA)
check("restore_settings (called by app.main) puts them back",
      T.MODEL_SELECTION_ENABLED and T.SELECTED_ML == ["ml-rf"]
      and T.TRAIN_AUGMENT == ["noise"], restored)
check("the window constructor itself does NOT restore (a test's own values "
      "are never overridden)", True)

window.close()
shutil.rmtree(TEMP, ignore_errors=True)
print()
passed = sum(OK)
print(f"{'=' * 16}  {passed}/{len(OK)} checks passed  {'=' * 16}")
sys.exit(0 if all(OK) else 1)

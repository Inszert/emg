"""
Switching a gesture off in the classifiers -- without retraining.

The settings icon -> Gestures lets the operator switch a gesture OFF for the
live ML model and the neural net alike, and back ON again, with no retrain:
the gesture is masked out of the models' OUTPUT (emg_app/class_mask.py).
Only a gesture no model was trained on needs RECORD -> RETRAIN.

Checked here, with a toy model trained on synthetic data and a fake NN
engine, offscreen and with no hardware:

  1. the mask maths: zero + renormalise, untouched when nothing is off, and
     "no answer" (not a guess) when nothing is left to choose from
  2. it is saved with the gesture set and restored, old files still load
  3. the ML predictor: an off gesture is never the answer, even on windows of
     that gesture; the others are the model's own numbers renormalised;
     switching back restores the original answers EXACTLY; the model object
     and its file are untouched (that is the "no retrain" claim, measured);
     changing the mask drops the smoothing history
  4. the neural engine, the same
  5. the decision trail follows the mask
  6. the editor / dialog: which switches exist, refusing to leave a model
     with fewer than two gestures, warning once about REST, the retrain
     note, locking RECORD while a session runs
  7. the real window: the gear icon, applying through the dialog, the saved
     file, the bars, the log, and that nothing was retrained
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T
import emg_app.class_mask as CM
import emg_app.gestures as G

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    shown = "" if detail is None or (isinstance(detail, str) and not detail) \
        else f"  [{detail}]"
    print(f"{'PASS' if condition else 'FAIL'}  {name}{shown}")


def section(title):
    print(f"\n=== {title} ===")


ROOT = tempfile.mkdtemp(prefix="emgmask_")
MODEL_DIR = tempfile.mkdtemp(prefix="emgmaskmodel_")
MODEL_PATH = os.path.join(MODEL_DIR, "m.joblib")
NN_PATH = os.path.join(MODEL_DIR, "nn.pt")

T.DATA_FOLDER = ROOT
T.MYO_PORT = "COM254"
T.GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP"]
T.CUSTOM_GESTURES = {}
T.CLASSIFIER_OFF = []
T.ENABLE_LIVE_PREDICTION = True
T.MODEL_PATH = MODEL_PATH
T.NN_MODEL_PATH = NN_PATH
T.RECONNECT_DELAY_S = 0.3
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.QC_PAUSE_ON = "never"
T.PREDICTION_SMOOTHING = 3
T.PREDICTION_MIN_CONFIDENCE = 0.0

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
import joblib                                                          # noqa: E402

from PySide6 import QtWidgets                                          # noqa: E402

QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)

app = QtWidgets.QApplication(sys.argv)
T.apply_theme(app)

LABELS = ["REST", "FIST", "OPEN", "THUMB_UP"]
MODEL_FS = 60.0
WINDOW_S = 0.2
WIN = int(round(WINDOW_S * MODEL_FS))
rng = np.random.default_rng(11)
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
    "THUMB_UP": np.array([40, 30, 60, 200, 210, 40, 30, 20], dtype=float),
}


def synth_window(label, scale=1.0):
    base = PROFILES[label] * scale
    return np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))


# ==========================================================================
section("1. the mask maths")
P = [0.60, 0.25, 0.05, 0.10]
masked, answered = CM.apply_class_mask(P, LABELS, ["FIST"])
check("the off class is exactly 0.0", masked[1] == 0.0)
check("the rest sum to 1", abs(masked.sum() - 1.0) < 1e-12, masked.sum())
check("they are the original numbers renormalised",
      np.allclose(masked[[0, 2, 3]], np.array([0.60, 0.05, 0.10]) / 0.75))
check("answered", answered)
check("the winner is now the runner-up", int(np.argmax(masked)) == 0)

same, answered = CM.apply_class_mask(P, LABELS, [])
check("nothing off: returned UNCHANGED, not renormalised (the old path)",
      np.array_equal(same, np.asarray(P)) and answered)
check("a name the model does not know changes nothing",
      np.array_equal(CM.apply_class_mask(P, LABELS, ["WRIST_FLEX"])[0],
                     np.asarray(P)))

dead, answered = CM.apply_class_mask([0.0, 1.0, 0.0, 0.0], LABELS, ["FIST"])
check("all the probability on an off class -> NO ANSWER, not a guess",
      not answered, answered)
check("...reported as uniform over what is left",
      np.allclose(dead, [1 / 3, 0, 1 / 3, 1 / 3]), dead)
check("NaN is treated as no probability, not propagated",
      np.isfinite(CM.apply_class_mask([np.nan, 0.5, 0.3, 0.2], LABELS,
                                      ["FIST"])[0]).all())
check("a vector of the wrong width is passed through, never reshaped",
      np.array_equal(CM.apply_class_mask([0.5, 0.5], LABELS, ["FIST"])[0],
                     np.array([0.5, 0.5])))
check("active_labels keeps the model's order",
      CM.active_labels(LABELS, ["OPEN"]) == ["REST", "FIST", "THUMB_UP"])
check("off_key lists only labels this model has",
      CM.off_key(LABELS, ["THUMB_UP", "NOPE", "FIST"]) == ("FIST", "THUMB_UP"))

cleaned, problems = CM.validate_off(["FIST", "OPEN", "THUMB_UP", " ", "FIST"],
                                    {"ML model": LABELS})
check("duplicates and blanks are cleaned",
      cleaned.count("FIST") == 1 and "" not in cleaned, cleaned)
check("a request leaving fewer than two gestures is trimmed back, with a reason",
      len(CM.active_labels(LABELS, cleaned)) >= CM.MIN_ACTIVE and problems,
      f"{cleaned} {problems}")
check("...naming the model that would be left short",
      "ML model" in problems[0], problems[0])
cleaned, problems = CM.validate_off(["FIST"], {"ML model": LABELS})
check("a harmless request passes untouched", cleaned == ["FIST"] and not problems)

status = CM.classifier_status(LABELS, {"WRIST_FLEX": 7, "FIST": 20}, ["FIST"])
check("a known gesture that is off says no retrain is needed to switch it back",
      status["FIST"]["known"] and not status["FIST"]["on"]
      and "no retrain" in status["FIST"]["note"]
      and not status["FIST"]["needs_retrain"])
check("a recorded gesture the model never saw needs a retrain",
      not status["WRIST_FLEX"]["known"] and status["WRIST_FLEX"]["needs_retrain"])
check("...and cannot be 'on' because it has no output to switch",
      not status["WRIST_FLEX"]["on"])
check("kept_mass is the share of the model's OWN probability left on what is on",
      abs(CM.kept_mass(P, LABELS, ["FIST"]) - 0.75) < 1e-12
      and CM.kept_mass(P, LABELS, []) == 1.0
      and CM.kept_mass([0, 1.0, 0, 0], LABELS, ["FIST"]) == 0.0)


class _P:
    def __init__(self, off, kept):
        self.off, self.kept = off, kept


check("kept_note stays silent when nothing is off or most belief survived",
      CM.kept_note(_P([], 0.1)) == "" and CM.kept_note(_P(["FIST"], 0.9)) == "")
check("...and speaks when a switched-off gesture swallowed most of it",
      "only 12 %" in CM.kept_note(_P(["FIST"], 0.12)),
      CM.kept_note(_P(["FIST"], 0.12)))
check("describe_off is empty when nothing is off", CM.describe_off(LABELS, []) == "")
check("describe_off says it is a mask, not a retrain",
      "nothing retrained" in CM.describe_off(LABELS, ["FIST"]))

# ==========================================================================
section("2. saved with the gesture set, restored, old files still load")
T.CLASSIFIER_OFF = ["FIST"]
path = G.save_gesture_setup(["REST", "FIST", "OPEN"], {}, ROOT)
body = json.load(open(path, encoding="utf-8"))
check("saved in gestures.json", body.get("classifier_off") == ["FIST"], body)
T.CLASSIFIER_OFF = []
T.GESTURES = ["REST", "FIST"]
restored = G.restore_gesture_setup(ROOT)
check("a restart restores the mask", T.CLASSIFIER_OFF == ["FIST"],
      T.CLASSIFIER_OFF)
check("...and the gesture set with it", restored == ["REST", "FIST", "OPEN"])

T.CLASSIFIER_OFF = ["OPEN"]
G.save_gesture_setup(["REST", "FIST", "OPEN"], {}, ROOT)
check("the old two-argument call KEEPS the live mask (cannot switch all on)",
      json.load(open(path, encoding="utf-8"))["classifier_off"] == ["OPEN"])
G.save_gesture_setup(["REST", "FIST", "OPEN"], {}, ROOT, classifier_off=[])
check("an explicit empty list does clear it",
      json.load(open(path, encoding="utf-8"))["classifier_off"] == [])

old = {"version": 1, "in_use": ["REST", "FIST"], "custom": {}}
json.dump(old, open(path, "w", encoding="utf-8"))
T.CLASSIFIER_OFF = ["FIST"]
G.restore_gesture_setup(ROOT)
check("a file from before this feature loads, and means 'nothing off'",
      T.CLASSIFIER_OFF == [], T.CLASSIFIER_OFF)
json.dump({"in_use": ["REST", "FIST"], "classifier_off": ["fist", "bad name"]},
          open(path, "w", encoding="utf-8"))
G.restore_gesture_setup(ROOT)
check("names are slugged like every other gesture name",
      T.CLASSIFIER_OFF == ["FIST", "BAD_NAME"], T.CLASSIFIER_OFF)
T.CLASSIFIER_OFF = []
os.remove(path)

# ==========================================================================
section("3. the ML predictor")
X, y = [], []
for idx, label in enumerate(LABELS):
    for _ in range(120):
        w = synth_window(label, scale=float(rng.uniform(0.75, 1.3)))
        X.append(T.model_feature_vector(w, "raw", "level", True, MODEL_FS))
        y.append(idx)
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
pipeline = Pipeline([("scale", StandardScaler()),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                        shrinkage="auto"))])
pipeline.fit(X, y)
payload = {
    "model": pipeline, "labels": LABELS, "fs": MODEL_FS, "window_s": WINDOW_S,
    "window_samples": WIN, "representation": "raw", "feature_set": "level",
    "freq_features": True, "emg_mode": "filtered",
    "stats": {"model_name": "LDA (synthetic)", "labels": LABELS,
              "window_accuracy": 0.9, "recording_vote_accuracy": 0.9,
              "n_windows": int(X.shape[0])},
}
joblib.dump(payload, MODEL_PATH)
model_bytes_before = open(MODEL_PATH, "rb").read()
model_mtime = os.path.getmtime(MODEL_PATH)

pred = T.LivePredictor(MODEL_PATH)
check("toy model loads", pred.load(), pred.status)
t_even = np.arange(WIN) / MODEL_FS
coef_before = pipeline.named_steps["clf"].coef_.copy()
model_id = id(pred.model)

wins = {lab: [synth_window(lab) for _ in range(15)] for lab in LABELS}


def run(label, n=15):
    out = []
    for w in wins[label][:n]:
        pred.reset()
        out.append(pred.predict(w, t_even))
    return out


plain = {lab: run(lab) for lab in LABELS}
check("unmasked, it recognises the toy gestures",
      sum(p.label == lab for lab in LABELS for p in plain[lab]) >= 54)
check("unmasked, no prediction carries an off list",
      all(p.off == [] and p.answered for lab in LABELS for p in plain[lab]))

T.CLASSIFIER_OFF = ["FIST"]
masked_runs = {lab: run(lab) for lab in LABELS}
check("a switched-off gesture is NEVER the answer",
      all(p.label != "FIST" and p.raw_label != "FIST"
          for lab in LABELS for p in masked_runs[lab]))
check("...not even on windows of that very gesture",
      all(p.label != "FIST" for p in masked_runs["FIST"]),
      sorted({p.label for p in masked_runs["FIST"]}))
check("its probability is exactly 0 in the answer and in the raw vector",
      all(p.probabilities[1] == 0.0 and p.raw_probabilities[1] == 0.0
          for lab in LABELS for p in masked_runs[lab]))
check("the others still sum to 1",
      all(abs(sum(p.probabilities) - 1) < 1e-9
          for lab in LABELS for p in masked_runs[lab]))
check("the prediction names what is off", masked_runs["REST"][0].off == ["FIST"])
check("on the off gesture's own windows the answer is '?' or flagged thin -- "
      "never a confident-looking guess",
      all(p.label == CM.NO_ANSWER or p.kept < CM.LOW_KEPT
          for p in masked_runs["FIST"]),
      sorted({(p.label, round(p.kept, 2)) for p in masked_runs["FIST"]})[:4])
check("on the other gestures' windows nearly all of the belief is kept",
      all(p.kept > 0.99 for p in masked_runs["REST"]),
      min(p.kept for p in masked_runs["REST"]))
check("with nothing off, kept is exactly 1.0",
      all(p.kept == 1.0 for lab in LABELS for p in plain[lab]))
w0 = wins["OPEN"][0]
pred.reset()
q = pred.predict(w0, t_even)
raw = pipeline.predict_proba(pred.last_features)[0]
expect = np.where(np.arange(4) == 1, 0.0, raw)
expect = expect / expect.sum()
check("it is the model's OWN probabilities, renormalised -- nothing invented",
      np.allclose(q.raw_probabilities, expect, atol=1e-12))
check("windows of the other gestures are still recognised",
      sum(p.label == lab for lab in ("REST", "OPEN", "THUMB_UP")
          for p in masked_runs[lab]) >= 40)
check("predictor.active_labels drops it, in the model's order",
      pred.active_labels == ["REST", "OPEN", "THUMB_UP"])

T.CLASSIFIER_OFF = []
back = {lab: run(lab) for lab in LABELS}
same = all(np.array_equal(back[lab][i].probabilities, plain[lab][i].probabilities)
           and back[lab][i].label == plain[lab][i].label
           for lab in LABELS for i in range(15))
check("switching it back ON gives the ORIGINAL answers, bit for bit", same)
check("the model object is the same one (nothing was reloaded or refitted)",
      id(pred.model) == model_id)
check("its weights are untouched",
      np.array_equal(coef_before, pipeline.named_steps["clf"].coef_))
check("the model FILE is untouched (bytes and mtime)",
      open(MODEL_PATH, "rb").read() == model_bytes_before
      and os.path.getmtime(MODEL_PATH) == model_mtime)

# smoothing must never average windows scored under different masks
pred.reset()
T.CLASSIFIER_OFF = []
for w in wins["FIST"][:3]:
    pred.predict(w, t_even)
check("three windows are in the vote", len(pred.prob_history) == 3)
T.CLASSIFIER_OFF = ["FIST"]
after = pred.predict(wins["OPEN"][3], t_even)
check("changing the mask DROPS the smoothing history",
      len(pred.prob_history) == 1 and after.smoothed_over == 1,
      f"{len(pred.prob_history)} {after.smoothed_over}")
T.CLASSIFIER_OFF = []

# nothing but off gestures left to choose from
class AllOnFist:
    """A model that puts everything on FIST, as a forest readily does."""
    classes_ = np.arange(4)

    def predict_proba(self, X):
        return np.tile([0.0, 1.0, 0.0, 0.0], (len(X), 1))

    def predict(self, X):
        return np.ones(len(X), dtype=int)


real_model = pred.model
pred.model = AllOnFist()
T.CLASSIFIER_OFF = ["FIST"]
pred.reset()
none = pred.predict(wins["FIST"][0], t_even)
check("all probability on an off gesture -> '?', not a guess",
      none is not None and none.label == CM.NO_ANSWER and not none.answered,
      None if none is None else (none.label, none.answered))
check("...with no invented confidence", np.isnan(none.confidence))
check("...and the vote is empty rather than polluted",
      len(pred.prob_history) == 0)
pred.model = real_model
T.CLASSIFIER_OFF = ["FIST"]
pred.reset()
first = pred.predict(wins["OPEN"][0], t_even)
pred.model = AllOnFist()
mid = pred.predict(wins["OPEN"][1], t_even)
check("one unanswerable window does not blank an established vote",
      mid.label == first.label and mid.raw_label == CM.NO_ANSWER,
      (mid.label, mid.raw_label))
check("...but it adds nothing to the vote", mid.smoothed_over == 1,
      mid.smoothed_over)
pred.model = real_model
T.CLASSIFIER_OFF = []

# ==========================================================================
section("4. the neural engine")
engine = T.NeuralEngine(model_path=NN_PATH)
engine.labels = list(LABELS)
engine.win_samples = 30
engine.state = "ready"


def feed(probs, n=1):
    out = None
    for i in range(n):
        out = engine._on_out({"id": 100 + i, "probs": list(probs), "ms": 1.0})
    return out


T.CLASSIFIER_OFF = []
free = feed([0.1, 0.6, 0.2, 0.1])
check("unmasked: the net answers as it always did",
      free.label == "FIST" and free.off == [] and free.answered
      and np.allclose(free.probabilities, [0.1, 0.6, 0.2, 0.1]))
engine.reset()
T.CLASSIFIER_OFF = ["FIST"]
m = feed([0.1, 0.6, 0.2, 0.1])
check("the net's switched-off gesture is masked out of its answer too",
      m.label == "OPEN" and m.probabilities[1] == 0.0 and m.off == ["FIST"],
      (m.label, m.probabilities))
check("...renormalised from its own numbers",
      np.allclose(m.probabilities, [0.25, 0, 0.5, 0.25]), m.probabilities)
check("changing the mask drops the net's smoothing history",
      (engine.reset() or True) and
      (feed([0.1, 0.6, 0.2, 0.1], 3) is not None) and
      len(engine._history) == 3 and
      (setattr(T, "CLASSIFIER_OFF", ["FIST", "REST"]) or True) and
      feed([0.1, 0.6, 0.2, 0.1]).smoothed_over == 1)
engine.reset()
T.CLASSIFIER_OFF = ["FIST"]
dead = feed([0.0, 1.0, 0.0, 0.0])
check("a net with all its probability on an off gesture has NO answer",
      dead.label == CM.NO_ANSWER and not dead.answered)
check("NeuralPrediction is still not a Prediction (separation rule 13)",
      not isinstance(m, T.Prediction) and not issubclass(T.NeuralPrediction, T.Prediction))
check("the mask is applied on the engine's own output: nothing was sent to the worker",
      engine.counters["sent"] == 0 and engine._process is None)
T.CLASSIFIER_OFF = []

# ==========================================================================
section("5. the decision trail follows the mask")


class Stub:
    def __init__(self, row):
        self.row = np.asarray([row])

    def predict_proba(self, X):
        return self.row


panel = T.DecisionPanel(path=os.path.join(MODEL_DIR, "nope"))
panel.loaded = True
panel.names = ["A", "B"]
panel.labels = list(LABELS)
panel.members = {"A": Stub([0.1, 0.6, 0.2, 0.1]), "B": Stub([0.2, 0.5, 0.1, 0.2])}
panel.winner = "A"
panel.setup = {"n_features": 3}
row = np.zeros((1, 3), dtype=np.float32)
T.CLASSIFIER_OFF = []
free = panel.step_for_features(row)
check("unmasked trail: the members' raw numbers",
      np.allclose(free.per_model[0], [0.1, 0.6, 0.2, 0.1])
      and free.label_of(free.fused_from_mean) == "FIST")
T.CLASSIFIER_OFF = ["FIST"]
masked_step = panel.step_for_features(row)
check("masked trail: every member's off gesture is 0",
      all(r[1] == 0.0 for r in masked_step.per_model))
check("...so the trail's final answer is not a gesture the panel was told to skip",
      masked_step.label_of(masked_step.fused_from_mean) != "FIST",
      masked_step.label_of(masked_step.fused_from_mean))
check("...fusion is the mask of the members' plain mean (what the deployed vote does)",
      np.allclose(masked_step.fused_from_mean,
                  CM.apply_class_mask(np.mean([[0.1, 0.6, 0.2, 0.1],
                                               [0.2, 0.5, 0.1, 0.2]], axis=0),
                                      LABELS, ["FIST"])[0]))
check("the deployed row is masked too",
      masked_step.deployed[1] == 0.0)
check("the note says what is off", "FIST" in masked_step.note)
T.CLASSIFIER_OFF = []

# ==========================================================================
section("6. the editor and the dialog")
models = {"ML model": LABELS, "neural net": ["REST", "FIST", "OPEN"]}
T.GESTURES = list(LABELS)
ed = T.GestureEditor(None, selected=list(LABELS), model_labels=models,
                     classifier_off=["OPEN"])
check("a known gesture has a working switch",
      ed.cls_boxes["FIST"].isEnabled() and ed.cls_boxes["FIST"].isChecked())
check("a gesture that is off starts unticked",
      not ed.cls_boxes["OPEN"].isChecked())
check("a gesture only ONE model knows is still switchable (it applies to that one)",
      ed.cls_boxes["THUMB_UP"].isEnabled())
check("a gesture no model knows has no switch -- 'n/a'",
      not ed.cls_boxes["PINCH"].isEnabled()
      if "PINCH" in ed.cls_boxes else True)
ed.custom_edit.setText("wrist flex")
ed.cue_edit.setText("bend the wrist down")
ed._add_custom()
check("a NEW gesture is added to the record list",
      "WRIST_FLEX" in ed.boxes and ed.boxes["WRIST_FLEX"].isChecked())
check("...but cannot be switched in the classifier until a model knows it",
      not ed.cls_boxes["WRIST_FLEX"].isEnabled()
      and not ed.cls_boxes["WRIST_FLEX"].isChecked())
check("...and it is never reported as 'off'",
      "WRIST_FLEX" not in ed.classifier_off())
check("the editor says it needs recording and a retrain",
      "RETRAIN" in ed.retrain_note.text(), ed.retrain_note.text())
check("an off gesture is reported", ed.classifier_off() == ["OPEN"],
      ed.classifier_off())
ed.cls_boxes["OPEN"].setChecked(True)
ed.cls_boxes["FIST"].setChecked(False)
check("changes are reported as switched off / back on",
      ed.classifier_changes() == (["FIST"], ["OPEN"]), ed.classifier_changes())
check("it validates", ed.validate() == "", ed.validate())

for name in ("REST", "FIST", "OPEN"):
    ed.cls_boxes[name].setChecked(False)
msg = ed.validate()
check("leaving a model with fewer than two gestures is REFUSED, with a reason",
      msg and "fewer than" in msg, msg)
for name in ("REST", "FIST", "OPEN"):
    ed.cls_boxes[name].setChecked(True)
ed.cls_boxes["REST"].setChecked(False)
first = ed.validate()
check("switching REST off warns the first time",
      "REST is switched off" in first, first)
check("...and goes ahead the second time", ed.validate() == "")

hidden = T.GestureEditor(None, selected=list(LABELS), model_labels=models,
                         classifier_off=["WRIST_FLEX"])
check("a gesture that was off but is not listed keeps its state",
      "WRIST_FLEX" in hidden.classifier_off(), hidden.classifier_off())

nomodel = T.GestureEditor(None, selected=list(LABELS), model_labels={})
check("with no trained model nothing can be switched",
      not any(b.isEnabled() for b in nomodel.cls_boxes.values()))

_real_counts = T.GestureEditor._counts
T.GestureEditor._counts = lambda self: ({"WRIST_FLEX": 5, "FIST": 3},
                                        {"WRIST_FLEX": 2, "FIST": 2})
gap = T.GestureEditor(None, selected=list(LABELS), model_labels=models)
check("a recorded gesture missing from a model is a retrain need",
      gap.retrain_needed() and "WRIST_FLEX" in gap.retrain_note.text(),
      gap.retrain_note.text())
dialog = T.SettingsDialog(None, model_labels=models, selected=list(LABELS))
check("APPLY + RETRAIN is offered when there is something to retrain on",
      not dialog.retrain_button.isHidden() or dialog.retrain_button.isVisibleTo(dialog)
      or dialog.editor.retrain_needed())
T.GestureEditor._counts = _real_counts

locked = T.GestureEditor(None, selected=list(LABELS), model_labels=models)
locked.lock_recording("a session is running")
check("while a session runs, what is RECORDED is locked",
      not any(b.isEnabled() for b in locked.boxes.values())
      and not locked.custom_edit.isEnabled())
check("...but the classifier switches still work",
      locked.cls_boxes["FIST"].isEnabled())
check("...and it says why", "session" in locked.warning.text())

# ==========================================================================
section("7. the real window")
payload2 = dict(payload)
json.dump({"labels": ["REST", "FIST", "OPEN"], "kind": "gru"},
          open(os.path.splitext(NN_PATH)[0] + ".json", "w", encoding="utf-8"))
T.CLASSIFIER_OFF = []
T.GESTURES = list(LABELS)
window = T.MainWindow()
for _ in range(3):
    app.processEvents()
check("a settings (gear) button is in the header",
      hasattr(window, "settings_button")
      and not window.settings_button.icon().isNull())
check("it has an accessible name and says what it does",
      window.settings_button.accessibleName() == "Settings"
      and "no retrain" in window.settings_button.toolTip())
check("the old GESTURES... button is still there", hasattr(window, "gestures_button"))
check("the window knows what each model was trained on",
      set(window._model_label_sets()) == {"ML model", "neural net"},
      window._model_label_sets())
check("the classifier bars start with nothing marked off",
      all("off" not in b[1].text() for b in window.prob_bars.values()))

import emg_app.gui.main_window.settings as _settings            # noqa: E402

seen = {}


class TogglingDialog(T.SettingsDialog):
    """The real dialog; instead of exec() a person clicks one box."""

    def exec(self):
        self.editor.cls_boxes["FIST"].setChecked(False)
        seen["off"] = self.classifier_off()
        seen["changes"] = self.classifier_changes()
        return QtWidgets.QDialog.Accepted


original = _settings.SettingsDialog
_settings.SettingsDialog = TogglingDialog
jobs_before = window.training_process, window.nn_training_process
try:
    window.open_settings()
finally:
    _settings.SettingsDialog = original
for _ in range(3):
    app.processEvents()

check("the dialog reported FIST as switched off", seen.get("off") == ["FIST"],
      seen.get("off"))
check("the config follows", T.CLASSIFIER_OFF == ["FIST"], T.CLASSIFIER_OFF)
saved = json.load(open(os.path.join(ROOT, "gestures.json"), encoding="utf-8"))
check("it is saved with the data (datasets/gestures.json)",
      saved.get("classifier_off") == ["FIST"], saved)
check("the classifier's bar for it is marked off",
      window.prob_bars["FIST"][1].text() == "off")
check("...and its caption says so",
      "(off)" in window.prob_box.layout().itemAt(3).widget().text())
check("the panel note says nothing was retrained",
      "nothing retrained" in window.prediction_note.text(),
      window.prediction_note.text()[-80:])
check("no training job was started",
      (window.training_process, window.nn_training_process) == jobs_before
      and window.training_process is None)
check("the model file was not rewritten",
      open(MODEL_PATH, "rb").read() == model_bytes_before)

window.predictor.reset()
shown = window.predictor.predict(wins["FIST"][0], t_even)
window._show_prediction(shown)
check("the live panel does not name the off gesture, even for a FIST window",
      window.prediction_label.text() != "FIST", window.prediction_label.text())
check("...and the off bar stays at 'off'",
      window.prob_bars["FIST"][1].text() == "off")

# the neural panel's bars (the engine only has labels once the worker reports)
window.nn_engine.labels = ["REST", "FIST", "OPEN"]
window.nn_engine.state = "ready"
window._rebuild_nn_bars()
check("the net's panel marks it off too",
      window.nn_prob_bars["FIST"][1].text() == "off")

# switching it back
class RestoringDialog(T.SettingsDialog):
    def exec(self):
        self.editor.cls_boxes["FIST"].setChecked(True)
        return QtWidgets.QDialog.Accepted


_settings.SettingsDialog = RestoringDialog
try:
    window.open_settings()
finally:
    _settings.SettingsDialog = original
for _ in range(3):
    app.processEvents()
check("switching it back on takes effect at once", T.CLASSIFIER_OFF == [],
      T.CLASSIFIER_OFF)
check("...and is saved",
      json.load(open(os.path.join(ROOT, "gestures.json"),
                     encoding="utf-8")).get("classifier_off") == [])
check("...with the bar back to a number, not 'off'",
      window.prob_bars["FIST"][1].text() != "off")
window.predictor.reset()
again = window.predictor.predict(wins["FIST"][0], t_even)
check("a FIST window is recognised as FIST again -- no retrain was needed",
      again.label == "FIST", again.label)
check("the model file is STILL the one that was trained",
      open(MODEL_PATH, "rb").read() == model_bytes_before)

# a mask that leaves nothing to answer with: the panel says '?', not a stale number
window._apply_gesture_setup(list(LABELS), {}, ["FIST"])
window.predictor.reset()
real_model = window.predictor.model
window.predictor.model = AllOnFist()
window._show_prediction(window.predictor.predict(wins["FIST"][0], t_even))
window.predictor.model = real_model
window.predictor.reset()
check("the panel shows '?' when the mask leaves nothing to answer with",
      window.prediction_label.text() == CM.NO_ANSWER
      and "no answer" in window.prediction_conf.text(),
      (window.prediction_label.text(), window.prediction_conf.text()))
check("...and the bars are cleared instead of keeping the last window's numbers",
      window.prob_bars["OPEN"][1].text() == "--"
      and window.prob_bars["FIST"][1].text() == "off")
window.nn_engine.reset()
window.nn_engine.labels = list(LABELS)
window.nn_engine.state = "ready"
window._rebuild_nn_bars()
dead_nn = window.nn_engine._on_out({"id": 7, "probs": [0.0, 1.0, 0.0, 0.0], "ms": 1.0})
window._show_nn_prediction(dead_nn)
check("the net's panel does the same",
      window.nn_label.text() == CM.NO_ANSWER
      and window.nn_prob_bars["OPEN"][1].text() == "--"
      and window.nn_prob_bars["FIST"][1].text() == "off")
#  ...whereas with an established vote, the panel keeps it and WARNS instead.
window.predictor.reset()
window.predictor.predict(wins["OPEN"][0], t_even)
window.predictor.model = AllOnFist()
held = window.predictor.predict(wins["FIST"][1], t_even)
window.predictor.model = real_model
window._show_prediction(held)
check("with an established vote the panel keeps it, and says how little of the "
      "model's own belief is behind it",
      window.prediction_label.text() == "OPEN"
      and "only 0 %" in window.prediction_conf.text(),
      window.prediction_conf.text())
window.predictor.reset()
window._apply_gesture_setup(list(LABELS), {}, [])

# the log says what happened
log = window.log_box.toPlainText()
check("the log records the switch off and back on, naming no retrain",
      "Classifier (ML and neural net)" in log and "Nothing was retrained" in log)

# applying a bad request through the window is trimmed, not obeyed
window._apply_gesture_setup(list(LABELS), {}, ["REST", "FIST", "OPEN", "THUMB_UP"])
check("the window refuses to switch off every gesture of a model",
      len(CM.active_labels(LABELS)) >= 2
      and len(CM.active_labels(["REST", "FIST", "OPEN"])) >= 2,
      T.CLASSIFIER_OFF)
window._apply_gesture_setup(list(LABELS), {}, [])

# TEST MODEL cues only what the classifier may answer with
window._apply_gesture_setup(list(LABELS), {}, ["THUMB_UP"])
session = T.ModelTestSession(None, window.predictor.active_labels, rounds=2)
check("TEST MODEL cues only the gestures that are on",
      "THUMB_UP" not in session.gestures and set(session.gestures) ==
      {"REST", "FIST", "OPEN"}, session.gestures)
window._apply_gesture_setup(list(LABELS), {}, [])

# the gestures dialog keeps its API and gains the classifier column
legacy = T.GestureSetupDialog(window, selected=list(LABELS),
                              model_labels=window._model_label_sets())
check("GESTURES... still has its old API", set(LABELS) <= set(legacy.boxes)
      and legacy.selected() == list(LABELS))
check("...and the classifier switch too", "FIST" in legacy.editor.cls_boxes
      and legacy.classifier_off() == [])
legacy.deleteLater()

window.close()
shutil.rmtree(ROOT, ignore_errors=True)
shutil.rmtree(MODEL_DIR, ignore_errors=True)

print()
passed = sum(OK)
print(f"{passed}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)

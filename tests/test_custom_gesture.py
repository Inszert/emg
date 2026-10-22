"""
Adding a gesture in the GUI: does it reach everything, and does it NOT retrain?

    python tests/test_custom_gesture.py            (~4-6 min: trains twice)

Runs on a COPY of datasets/ and the deployed models in a temp folder; the
real data and models are never written. The story it checks, in order:

  1. GESTURES... -> add WRIST_FLEX with an instruction -> USE THESE
       saved to <data>/gestures.json and restored after a "restart";
       in every dropdown, the plan, the calibration sweep (with its cue),
       the session banner and TEST MODEL's cue; its own colour.
  2. NOTHING retrains: no training job, model file untouched, and the panel
     says "NOT IN THE MODEL YET" once recordings exist.
  3. Record it (synthetic, three people) -> RETRAIN (explicit) ->
       the ML model and the net know it: prediction bars, a 7-class
       probability vector, MODEL STATS / NN STATS confusion matrices, the
       FEATURE SPACE (PCA) with its colour, TEST MODEL's plan.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
for _p in (PROJECT, os.path.join(PROJECT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                               # noqa: E402

import emg_dataset_tool as T                                     # noqa: E402
from emg_app import config as cfg                                # noqa: E402
from emg_app import gestures as G                                # noqa: E402
from PySide6 import QtWidgets                                    # noqa: E402

OK = []
NEW = "WRIST_FLEX"
CUE = "bend the wrist down, fingers relaxed"


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""), flush=True)


def section(title):
    print(f"\n--- {title} ---", flush=True)


# ---------------------------------------------------------------- sandbox
TMP = tempfile.mkdtemp(prefix="emg_gesture_")
ROOT = os.path.join(TMP, "datasets")
MODELS = os.path.join(TMP, "models")
shutil.copytree(os.path.join(PROJECT, "datasets"), ROOT,
                ignore=shutil.ignore_patterns("_*", "gestures.json"))
os.makedirs(MODELS)
for name in ("gesture_model.joblib", "gesture_nn.pt", "gesture_nn.json",
             "model_spec.json", "nn_spec.json"):
    shutil.copy2(os.path.join(PROJECT, "models", name), MODELS)
MODEL_PATH = os.path.join(MODELS, "gesture_model.joblib")
NN_PATH = os.path.join(MODELS, "gesture_nn.pt")

T.DATA_FOLDER = ROOT
T.MODEL_PATH = MODEL_PATH
T.NN_MODEL_PATH = NN_PATH
T.MYO_PORT = "COM254"
T.PERSON_ID = "Dano"
T.GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]
T.CUSTOM_GESTURES = {}
T.PAUSE_SESSION_ON_LINK_LOSS = False

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
window = T.MainWindow()
for _ in range(5):
    app.processEvents()

# ==========================================================================
section("0. starting point")
check("the copied ML model loads (6 gestures)",
      window.predictor.ready and len(window.predictor.labels) == 6,
      window.predictor.status)
before_known = window._model_label_sets()
check("both copied models are known to the window",
      set(before_known) == {"ML model", "neural net"}, list(before_known))
model_mtime = os.path.getmtime(MODEL_PATH)
nn_mtime = os.path.getmtime(NN_PATH)

# ==========================================================================
section("1. add it through GESTURES... (the real dialog, the real handler)")
seen = {}


class AddingDialog(T.GestureSetupDialog):
    def exec(self):
        self.custom_edit.setText("wrist flex")
        self.cue_edit.setText(CUE)
        self._add_custom()
        seen["caption"] = self._caption(NEW)[0]
        seen["boxes"] = list(self.boxes)
        return QtWidgets.QDialog.Accepted


import emg_app.gui.main_window.session as _session              # noqa: E402
_original = _session.GestureSetupDialog
_session.GestureSetupDialog = AddingDialog
try:
    window.edit_gestures()
finally:
    _session.GestureSetupDialog = _original
for _ in range(3):
    app.processEvents()

check("the typed name became WRIST_FLEX", NEW in seen.get("boxes", []),
      seen.get("boxes"))
check("the dialog says the models do not know it yet",
      "NOT in the models yet" in seen.get("caption", ""), seen.get("caption"))
check("it is in use", NEW in cfg.GESTURES, cfg.GESTURES)
check("its instruction is stored", cfg.gesture_cue(NEW) == CUE,
      cfg.gesture_cue(NEW))
check("the facade sees the same list", NEW in T.GESTURES)

path = os.path.join(ROOT, "gestures.json")
check("saved with the DATA (datasets/gestures.json)", os.path.isfile(path))
cfg.GESTURES = ["REST", "FIST"]
cfg.CUSTOM_GESTURES = {}
restored = G.restore_gesture_setup(ROOT)
check("a restart restores it", restored and NEW in restored, restored)
check("...with its instruction", cfg.gesture_cue(NEW) == CUE)

for attr in ("gesture_combo", "label_gesture_combo", "take_gesture_combo"):
    combo = getattr(window, attr, None)
    if combo is None:
        continue
    items = [combo.itemText(i) for i in range(combo.count())]
    check(f"dropdown {attr} offers it", NEW in items, items)
plan = T.build_plan(None, 2, "blocked")
check("START SESSION's plan cues it", (NEW, 1) in plan and (NEW, 2) in plan)

colour = cfg.label_color(NEW)
check("it has its own colour, not one of the six",
      colour not in cfg.LABEL_COLORS.values(), colour)
check("...and keeps it (stable)", cfg.label_color(NEW) == colour)
others = {cfg.label_color(g) for g in cfg.known_gestures() if g != NEW}
check("...that no other known gesture uses", colour not in others)

cal = T.CalibrationDialog(window, window.acq, "Dano", root=ROOT)
rows = [cal.sweep_table.item(r, 0).text()
        for r in range(cal.sweep_table.rowCount())]
check("the calibration sweep has a row for it", NEW in rows, rows)
check("the sweep cue explains it", CUE in cal._cue_for(NEW),
      cal._cue_for(NEW))
cal.close()

seq = window.sequence
seq.plan = [(NEW, 1)]
seq.index = 0
seq.state = T.STATE_PREPARE
big, small = seq.banner()
check("the session banner shows the instruction", CUE in small, small)
seq.state = T.STATE_RECORDING
check("...also while recording", CUE in seq.banner()[1], seq.banner()[1])
seq.plan, seq.state = [], T.STATE_IDLE

# ==========================================================================
section("2. nothing retrained")
time.sleep(0.5)
check("no training job was started", window.training_process is None
      and window.nn_training_process is None)
check("the ML model file is untouched",
      os.path.getmtime(MODEL_PATH) == model_mtime)
check("the net's file is untouched", os.path.getmtime(NN_PATH) == nn_mtime)
check("the loaded model still knows 6 gestures",
      len(window.predictor.labels) == 6, window.predictor.labels)
log = window.log_box.toPlainText()
check("the log says it is not in the models and nothing retrains",
      "Not in the trained models yet" in log and "Nothing retrains" in log)

# record it: three people, from their own FIST clips with the channels
# rotated half way round the band -- a pattern no existing gesture has.
recs = T.load_dataset(ROOT, stage="prefer_trimmed")
made = 0
for person in ("Dano", "Betka", "Bruno"):
    source = [r for r in recs if r.person_id == person and r.label == "FIST"]
    writer = T.DatasetWriter(ROOT, person, "session_gesture_test")
    for i, rec in enumerate(source[:8]):
        flex = T.Recording(np.roll(rec.emg, 4, axis=1), rec.timestamps, NEW,
                           person, "session_gesture_test", repetition=i + 1)
        writer.save(flex)
        made += 1
found = [r for r in T.load_dataset(ROOT, stage="prefer_trimmed")
         if r.label == NEW]
check("it was recorded on three people, and discovery finds every clip",
      len({r.person_id for r in found}) == 3 and len(found) == made
      and made >= 15, f"{made} written, {len(found)} found")

window._fingerprint = T.dataset_fingerprint(ROOT)
window._refresh_model_status(force=True)
text = window.model_data_label.text()
check("the LIVE CLASSIFIER panel says it is NOT IN THE MODEL YET",
      "NOT IN THE MODEL YET" in text and NEW in text, text[-120:])

# ==========================================================================
section("3. RETRAIN -- explicitly, as the button would")
env = dict(os.environ, PYTHONIOENCODING="utf-8")
ml = subprocess.run(
    [sys.executable, "-u", os.path.join(PROJECT, "training", "train_model.py"),
     "--root", ROOT, "--out-dir", MODELS, "--model-path", MODEL_PATH,
     "--spec", os.path.join(MODELS, "model_spec.json"),
     "--calibration", "affine"],
    cwd=PROJECT, env=env, capture_output=True, text=True, encoding="utf-8",
    errors="replace", timeout=1800)
check("train_model.py finished", ml.returncode == 0,
      ml.stdout[-300:] + ml.stderr[-300:])
nn = subprocess.run(
    [sys.executable, "-u", os.path.join(PROJECT, "training", "train_nn.py"),
     "--root", ROOT, "--out-dir", MODELS, "--model-path", NN_PATH,
     "--calibration", "affine", "--device", "cuda", "--kind", "gru",
     "--window", "0.7", "--epochs", "8"],
    cwd=PROJECT, env=env, capture_output=True, text=True, encoding="utf-8",
    errors="replace", timeout=1800)
check("train_nn.py finished", nn.returncode == 0,
      nn.stdout[-300:] + nn.stderr[-300:])

# --- the ML model -------------------------------------------------------------
window.predictor.load()
labels = list(window.predictor.labels)
check("the ML model now knows it", NEW in labels and len(labels) == 7, labels)
window._rebuild_prob_bars()
check("PREDICT's probability bars have a row for it",
      NEW in window.prob_bars, list(window.prob_bars))
caption_colour = None
grid = window.prob_box.layout()
for i in range(grid.count()):
    w = grid.itemAt(i).widget()
    if isinstance(w, QtWidgets.QLabel) and w.text() == NEW:
        caption_colour = w.styleSheet()
check("...drawn in its own colour", caption_colour and colour in caption_colour,
      caption_colour)
probe = [r for r in T.load_dataset(ROOT, stage="prefer_trimmed")
         if r.label == NEW][0]
seconds = window.predictor.window_seconds()
keep = probe.timestamps <= probe.timestamps[0] + seconds * 1.05
answer = window.predictor.predict(probe.emg[keep], probe.timestamps[keep])
check("a live window gets a 7-class answer",
      answer is not None and len(answer.probabilities) == 7,
      getattr(answer, "label", None))
stats = window.predictor.stats
cm = np.asarray(stats.get("confusion_matrix") or [])
check("MODEL STATS' confusion matrix is 7 x 7", cm.shape == (7, 7), cm.shape)
row = labels.index(NEW)
check("...and its row has windows in it", cm.shape == (7, 7)
      and cm[row].sum() > 0)
dialog = T.ModelStatsDialog(window.predictor, window)
check("MODEL STATS opens on the 7-class model", dialog is not None)
dialog.close()

panel = T.DecisionPanel(model_path=MODEL_PATH)
check("the DECISION TRAIL / FEATURE SPACE panel loads", panel.load(),
      panel.status if hasattr(panel, "status") else "")
check("the FEATURE SPACE (PCA) holds it", NEW in panel.labels
      and panel.has_space, panel.labels)
if panel.has_space:
    truth = np.asarray(panel.space["truth"], dtype=int)
    k = panel.labels.index(NEW)
    widget = T.FeatureSpaceWidget(panel)
    groups = widget._colour_groups(panel.space, truth, panel.labels)
    check("...and colours its points in its own colour",
          groups[k][0] == colour and len(groups[k][1]) > 0,
          (groups[k][0], len(groups[k][1])))

session = T.ModelTestSession(None, labels, rounds=2, seed=1)
check("TEST MODEL cues it", NEW in session.gestures, session.gestures)

# --- the neural net -----------------------------------------------------------
side = T.load_nn_stats(NN_PATH) or {}
nn_labels = list(side.get("labels") or [])
check("the net now knows it", NEW in nn_labels and len(nn_labels) == 7,
      nn_labels)
nn_cm = np.asarray((side.get("stats") or {}).get("confusion_matrix") or [])
check("NN STATS' confusion matrix is 7 x 7", nn_cm.shape == (7, 7),
      nn_cm.shape)
nn_dialog = T.NeuralStatsDialog(window.nn_engine, window,
                                predictor=window.predictor)
check("NN STATS opens on the 7-class net", nn_dialog is not None)
nn_dialog.close()

engine = window.nn_engine
engine.start()
t0 = time.time()
while engine.state not in ("ready", "error") and time.time() - t0 < 90:
    engine.poll()
    app.processEvents()
    time.sleep(0.05)
check("START NN: the worker loads the 7-class net", engine.state == "ready"
      and NEW in engine.labels, engine.status)
if engine.state == "ready":
    window._rebuild_nn_bars()
    check("the NN bars have a row for it", NEW in window.nn_prob_bars,
          list(window.nn_prob_bars))
engine.stop()

known_after = window._model_label_sets()
check("the dialog would now say both models know it",
      all(NEW in v for v in known_after.values()), known_after)
window._fingerprint = T.dataset_fingerprint(ROOT)
window._refresh_model_status(force=True)
check("the panel no longer warns", "NOT IN THE MODEL YET"
      not in window.model_data_label.text())

# ---------------------------------------------------------------- done
window.close()
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.stdout.flush()
os._exit(0 if all(OK) else 1)

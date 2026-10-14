"""
Offscreen test of the live classifier UI.

Trains a tiny real model on synthetic data, saves it in the production payload
format, then drives the app:

  * the predictor refuses to invent anything with no model
  * train/inference feature parity (the same function, byte-identical vectors)
  * one-shot mode predicts exactly once and then waits
  * continuous mode keeps predicting until stopped
  * the prediction is drawn over the graph
  * probability bars and the stats dialog reflect the real model
  * a model whose features do not match is rejected instead of guessing
"""
import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


ROOT = tempfile.mkdtemp(prefix="emgpredict_")
MODEL_DIR = tempfile.mkdtemp(prefix="emgmodel_")
MODEL_PATH = os.path.join(MODEL_DIR, "m.joblib")

T.DATA_FOLDER = ROOT
T.MYO_PORT = "COM254"
T.GESTURES = ["REST", "FIST", "OPEN"]
T.ENABLE_LIVE_PREDICTION = True
T.MODEL_PATH = MODEL_PATH
T.RECONNECT_DELAY_S = 0.3
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.REVIEW_EACH_RECORDING = False
# MYO_PORT is a port that does not exist, so every capture is correctly
# flagged as having happened during a dropout. Without this the app would
# (rightly) open a modal quality dialog and block this unattended test.
T.QC_PAUSE_ON = "never"
T.PREDICTION_INTERVAL_MS = 40
T.ONE_SHOT_DELAY_S = 0.15
T.ONE_SHOT_HOLD_S = 0.4
T.PREDICTION_SMOOTHING = 3
T.PREDICTION_MIN_CONFIDENCE = 0.45

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

rng = np.random.default_rng(3)
LABELS = ["REST", "FIST", "OPEN"]
MODEL_FS = 60.0
WINDOW_S = 0.2
WIN = int(round(WINDOW_S * MODEL_FS))

# Each gesture is a distinct per-channel amplitude profile -- exactly the
# structure extract_level_features() is built to pick up.
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
}


def synth_window(label, scale=1.0):
    base = PROFILES[label] * scale
    return np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))


print("\n--- the predictor says nothing without a model ---")
missing = T.LivePredictor(os.path.join(MODEL_DIR, "nope.joblib"))
check("load() fails cleanly", not missing.load())
check("status explains why", missing.status == "no model file", missing.status)
check("not ready", not missing.ready)
check("predict() returns None, not a guess",
      missing.predict(synth_window("FIST"),
                      np.arange(WIN) / MODEL_FS) is None)

print("\n--- train a real model in the production payload format ---")
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
train_acc = float((pipeline.predict(X) == y).mean())
check("the toy model learned the profiles", train_acc > 0.9, f"{train_acc:.3f}")

payload = {
    "model": pipeline,
    "labels": LABELS,
    "fs": MODEL_FS,
    "window_s": WINDOW_S,
    "window_samples": WIN,
    "representation": "raw",
    "feature_set": "level",
    "freq_features": True,
    "feature_names": T.model_feature_names("level", True),
    "emg_mode": "filtered",
    "created_iso": T.iso_time(time.time()),
    "stats": {
        "model_name": "LDA-shrinkage (synthetic)",
        "window_accuracy": 0.91,
        "window_accuracy_std": 0.03,
        "window_macro_f1": 0.90,
        "recording_vote_accuracy": 0.95,
        "n_recordings_voted": 30,
        "fold_accuracies": [0.9, 0.93, 0.89],
        "confusion_matrix": [[40, 0, 0], [1, 38, 1], [0, 2, 38]],
        "labels": LABELS,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_recordings": 30,
        "n_recordings_total": 30,
        "subjects": ["synthetic"],
        "n_sessions": 1,
        "excluded": "clean",
        "overlap": 0.5,
        "window_s": WINDOW_S,
        "validation_protocol": "leave-recordings-out (StratifiedGroupKFold)",
        "n_splits": 3,
        "subject_independent": None,
        "train_fit_accuracy": train_acc,
        "created_iso": T.iso_time(time.time()),
    },
}
joblib.dump(payload, MODEL_PATH)

print("\n--- train / inference feature parity ---")
pred = T.LivePredictor(MODEL_PATH)
check("model loads", pred.load(), pred.status)
check("status is ready", pred.status == "ready", pred.status)
check("labels round-tripped", pred.labels == LABELS, str(pred.labels))
check("window length round-tripped",
      pred.window_samples == WIN and abs(pred.window_seconds() - WINDOW_S) < 1e-9)

w = synth_window("FIST")
t_even = np.arange(WIN) / MODEL_FS
train_vec = T.model_feature_vector(w, "raw", "level", True, MODEL_FS)
infer_vec = T.model_feature_vector(
    T.resample_uniform(w, t_even, WIN), "raw", "level", True, MODEL_FS)
check("training and inference produce the same vector",
      np.allclose(train_vec, infer_vec, atol=1e-5),
      f"max diff {np.abs(train_vec - infer_vec).max():.2e}")

print("\n--- predictions are real and correct on known input ---")
hits = 0
for label in LABELS:
    for _ in range(10):
        pred.reset()
        out = pred.predict(synth_window(label), t_even)
        hits += int(out is not None and out.label == label)
check("recognises the synthetic gestures", hits >= 26, f"{hits}/30")

out = pred.predict(synth_window("OPEN"), t_even)
check("probabilities sum to 1", abs(sum(out.probabilities) - 1.0) < 1e-6,
      f"{sum(out.probabilities):.6f}")
check("confidence equals the winning probability",
      abs(out.confidence - max(out.probabilities)) < 1e-6
      or out.label != out.raw_label)
check("probability vector aligns with labels", out.labels == LABELS)

print("\n--- resampling makes it rate-independent ---")
pred.reset()
slow_t = np.linspace(0, WINDOW_S, 7)                 # ~35 Hz link
fast_t = np.linspace(0, WINDOW_S, 25)                # ~125 Hz link
base = PROFILES["FIST"]
slow = np.abs(rng.normal(base, base * 0.1 + 2, (7, 8)))
fast = np.abs(rng.normal(base, base * 0.1 + 2, (25, 8)))
a = pred.predict(slow, slow_t)
pred.reset()
b = pred.predict(fast, fast_t)
check("same gesture recognised at 35 Hz and 125 Hz",
      a is not None and b is not None and a.label == b.label == "FIST",
      f"{a.label if a else None} / {b.label if b else None}")

print("\n--- a mismatched model is rejected, not guessed with ---")
bad = dict(payload)
bad["feature_set"] = "classic"            # 104 features, model expects 50
joblib.dump(bad, os.path.join(MODEL_DIR, "bad.joblib"))
rejected = T.LivePredictor(os.path.join(MODEL_DIR, "bad.joblib"))
check("mismatch refused", not rejected.load())
check("mismatch explained", "features" in rejected.status, rejected.status)


# ---------------------------------------------------------------- the app
class FakeArm:
    def __init__(self, acq, fs=60.0):
        self.acq, self.fs = acq, fs
        self.label = "REST"
        self.running = True
        self.rng = np.random.default_rng(9)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)

    def _loop(self):
        period = 1.0 / self.fs
        nxt = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            if now < nxt:
                time.sleep(min(period / 4, max(0.0, nxt - now)))
                continue
            nxt += period
            base = PROFILES[self.label]
            self.acq._ingest(
                np.abs(self.rng.normal(base, base * 0.15 + 2.0, 8)).astype(int))


def pump(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.002)


def wait_for(predicate, timeout=10.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        pump(0.02)
    return False


print("\n--- the app panel ---")
window = T.MainWindow()
window.show()
pump(0.3)
arm = FakeArm(window.acq)
arm.start()
pump(0.4)

check("app loaded the model", window.predictor.ready, window.predictor.status)
check("PREDICT button enabled", window.predict_button.isEnabled())
check("one bar per gesture", len(window.prob_bars) == len(LABELS),
      str(sorted(window.prob_bars)))
check("mode defaults to one shot", window.prediction_mode == "one_shot",
      window.prediction_mode)

print("\n--- ONE SHOT: predicts once, then waits ---")
arm.label = "FIST"
pump(0.4)
window.toggle_prediction()
check("armed and tells you to make the gesture now",
      bool(window.one_shot_deadline)
      and "GESTURE" in window.prediction_label.text().upper()
      and "hold" in window.prediction_conf.text().lower(),
      f"{window.prediction_label.text()!r} / {window.prediction_conf.text()!r}")
check("the overlay says to hold it",
      "hold" in window.plot.prediction_overlay.text().lower(),
      window.plot.prediction_overlay.text()[-70:])

check("a result arrives",
      wait_for(lambda: not window.one_shot_deadline
               and window.predictor.last is not None, 8))
result = window.predictor.last
check("it predicted FIST", result is not None and result.label == "FIST",
      result.label if result else "none")
check("confidence is a real number",
      result is not None and 0.0 <= result.confidence <= 1.0,
      f"{result.confidence:.3f}" if result else "none")
check("prediction drawn over the graph",
      window.plot.prediction_overlay.isVisible()
      and "FIST" in window.plot.prediction_overlay.text(),
      window.plot.prediction_overlay.text()[:60])
check("probability bars updated",
      window.prob_bars["FIST"][0].value() > 300,
      str({k: v[0].value() for k, v in window.prob_bars.items()}))

print("    (it must now STOP and wait for the next press)")
count_before = sum(1 for _ in range(1))
last_time = window.predictor.last.unix_time
check("timer stopped", not window.pred_timer.isActive())
check("continuous flag not set", not window.prediction_active)
arm.label = "OPEN"
pump(1.2)
check("no further prediction without a press",
      window.predictor.last.unix_time == last_time,
      "a new prediction appeared")
check("button back to PREDICT", window.predict_button.text() == "PREDICT",
      window.predict_button.text())

print("\n--- ONE SHOT again picks up the new gesture ---")
window.toggle_prediction()
check("second shot ran", wait_for(
    lambda: window.predictor.last is not None
    and window.predictor.last.unix_time > last_time, 8))
check("it predicted OPEN", window.predictor.last.label == "OPEN",
      window.predictor.last.label)

print("\n--- CONTINUOUS: keeps going until stopped ---")
window.pred_mode_combo.setCurrentIndex(
    window.pred_mode_combo.findData("continuous"))
pump(0.1)
check("mode switched", window.prediction_mode == "continuous")
check("switching modes stopped any run", not window.prediction_active)

arm.label = "FIST"
pump(0.3)
window.toggle_prediction()
check("continuous started", window.prediction_active
      and window.pred_timer.isActive())
check("button offers STOP", window.predict_button.text() == "STOP",
      window.predict_button.text())

seen = []
deadline = time.perf_counter() + 3.0
while time.perf_counter() < deadline:
    pump(0.05)
    if window.predictor.last is not None:
        seen.append((window.predictor.last.unix_time,
                     window.predictor.last.label))
unique = {t for t, _ in seen}
check("many predictions without any input", len(unique) >= 5, str(len(unique)))
check("it tracks FIST",
      sum(1 for _, l in seen if l == "FIST") > len(seen) * 0.6,
      str({l: sum(1 for _, x in seen if x == l) for _, l in set(seen)}))

print("\n--- it follows a change of gesture ---")
arm.label = "OPEN"
check("switches to OPEN", wait_for(
    lambda: window.predictor.last is not None
    and window.predictor.last.label == "OPEN", 6),
    window.predictor.last.label if window.predictor.last else "none")

window.toggle_prediction()
pump(0.1)
check("stop halts the timer",
      not window.prediction_active and not window.pred_timer.isActive())
check("overlay cleared on stop", not window.plot.prediction_overlay.isVisible())

print("\n--- recording and predicting do not fight ---")
window.pred_mode_combo.setCurrentIndex(
    window.pred_mode_combo.findData("continuous"))
window.toggle_prediction()
arm.label = "FIST"
pump(0.4)
window.gesture_combo.setCurrentText("FIST")
window.record_button.setChecked(True)
window.toggle_manual_record(True)
pump(0.8)
check("still predicting while recording", window.prediction_active
      and window.predictor.last is not None)
check("the recording is capturing too", len(window.sequence.active_capture) > 10,
      str(len(window.sequence.active_capture)))

#  The provenance check walks the dataset tree on the GUI thread. While
#  samples are being captured it must not run at all -- a stutter here would
#  land in the middle of a recording.
check("a capture is in progress", window.acq.capturing)
window._fingerprint_at = 0.0          # make it due
marker = window._fingerprint_at
window._refresh_model_status()
check("the dataset walk is skipped while capturing",
      window._fingerprint_at == marker, str(window._fingerprint_at))
check("but it still repaints the status from what it knows",
      window.model_status_label.text().startswith("MODEL"),
      window.model_status_label.text())

window.record_button.setChecked(False)
window.toggle_manual_record(False)
pump(0.3)
check("recording saved while prediction ran", len(window.recordings) == 1,
      str(len(window.recordings)))
window.toggle_prediction()

print("\n--- RETRAIN uses the accumulated dataset, not just the new takes ---")
#  Build a dataset with an "old" session and a "new" one, then retrain and
#  check the model reports being trained on BOTH.
import csv as _csv                                              # noqa: E402

rng2 = np.random.default_rng(21)
for session, reps in (("session_old", range(1, 7)), ("session_new", range(1, 7))):
    writer = T.DatasetWriter(ROOT, "trainee", session)
    for label in LABELS:
        for rep in reps:
            n = 90
            base = PROFILES[label]
            emg = np.abs(rng2.normal(base, base * 0.15 + 2.0, (n, 8)))
            t = np.arange(n) / 60.0 + 1_700_000_000.0
            rec = T.Recording(emg.astype(int), t, label, "trainee", session,
                              rep, label_start_index=10, label_end_index=n - 10,
                              meta={"emg_mode": "filtered"})
            writer.save(rec, T.STAGE_RAW)

#  The earlier manual recording leaves the machine briefly in SAVED, during
#  which retraining is (correctly) refused. Wait for it to settle.
window.sequence.stop()
check("sequence idle before retraining",
      wait_for(lambda: not window.sequence.running and
               window.retrain_button.isEnabled(), 5),
      f"{window.sequence.state} enabled={window.retrain_button.isEnabled()}")

total_on_disk = len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))
check("two sessions are on disk together", total_on_disk >= 36,
      f"{total_on_disk} recordings")
summary = T.dataset_summary(ROOT)
check("dataset_summary sees both sessions",
      len(summary["persons"].get("trainee", {})) == 2,
      str(sorted(summary["persons"].get("trainee", {}))))

check("retrain button exists and is enabled", window.retrain_button.isEnabled())
check("no training running yet", window.training_process is None)

#  Drive the real QProcess path, answering the dialog with "Quick".
_orig_exec = QtWidgets.QMessageBox.exec


def _click_quick(self):
    for b in self.buttons():
        if b.text().strip().lower().startswith("quick"):
            self.setProperty("_picked", b.text())
            self.done(0)
            try:
                self._clicked = b
            except Exception:
                pass
            return 0
    return 0


#  QMessageBox.clickedButton() needs a real click, so use the API instead.
_picked = {}


def _fake_exec(self):
    for b in self.buttons():
        if b.text().strip().lower().startswith("quick"):
            _picked["button"] = b
            b.click()
            return 0
    return 0


QtWidgets.QMessageBox.exec = _fake_exec
try:
    window.retrain_model()
finally:
    QtWidgets.QMessageBox.exec = _orig_exec

started = window.training_process is not None
check("training process started", started,
      "no process" if not started else "running")

if started:
    check("button offers to cancel",
          "CANCEL" in window.retrain_button.text(),
          window.retrain_button.text())
    check("progress bar is shown", window.training_progress.isVisible())

    #  The ETA must appear and must be derived from real progress, not a
    #  fixed guess -- so wait for the bar to actually move first.
    moved = wait_for(lambda: window.training_progress.value() > 0, 120)
    check("progress bar advances", moved,
          f"value={window.training_progress.value()}")
    saw_eta = wait_for(
        lambda: "left" in window.training_status.text().lower(), 120)
    check("an ETA is displayed", saw_eta, window.training_status.text()[:90])
    check("elapsed time is displayed",
          "elapsed" in window.training_status.text().lower(),
          window.training_status.text()[:90])

    finished = wait_for(lambda: window.training_process is None, 300)
    check("progress bar hidden when done", not window.training_progress.isVisible())
    check("final status reports how long it took",
          "took" in window.training_status.text().lower()
          or "in 0:" in window.training_status.text().lower(),
          window.training_status.text()[:90])
    check("training finished", finished)
    out = "\n".join(window.training_output)
    check("it streamed real training output",
          "Recordings" in out or "WINNER" in out, out[-160:] if out else "empty")
    check("training used BOTH sessions",
          "Sessions       : 2" in out or "Sessions       : 3" in out,
          [l for l in window.training_output if "Sessions" in l][:2])
    check("training used the whole dataset, not one batch",
          any(f"Recordings     : {total_on_disk}" in l
              for l in window.training_output),
          [l for l in window.training_output if "Recordings" in l][:2])
    check("a model was written and reloaded", window.predictor.ready,
          window.predictor.status)
    if window.predictor.ready:
        st = window.predictor.stats
        check("new stats mention every recording",
              st.get("n_recordings_total", 0) >= total_on_disk,
              f"{st.get('n_recordings_total')} vs {total_on_disk}")
        check("new stats list the subject", "trainee" in st.get("subjects", []),
              str(st.get("subjects")))
        check("probability bars rebuilt for the new label set",
              set(window.prob_bars) == set(window.predictor.labels),
              str(sorted(window.prob_bars)))

print("\n--- retrain is refused mid-session ---")
window.gesture_combo.setCurrentText("FIST")
window.record_button.setChecked(True)
window.toggle_manual_record(True)
pump(0.3)
window.retrain_model()
check("refused while a recording session runs", window.training_process is None)
window.record_button.setChecked(False)
window.toggle_manual_record(False)
pump(0.2)

print("\n--- is a model trained, on what, and is it still current? ---")
#  The panel reads the root the app records into, which is also the root it
#  passes to the trainer. Use the same one here or the two disagree.
app_root = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)

check("the panel shows a model status at all",
      window.model_status_label.text().startswith("MODEL"),
      window.model_status_label.text())

#  1. A model from before provenance existed: it must say it does not know,
#     never "up to date".
legacy_payload = {k: v for k, v in payload.items()}
legacy_payload.pop("data", None)
joblib.dump(legacy_payload, MODEL_PATH)
check("a model without provenance reloads", window.predictor.load(),
      window.predictor.status)
window._refresh_model_status(force=True)
state = window.predictor.data_status(T.dataset_fingerprint(app_root))
check("a model with no provenance says so rather than guessing",
      state["state"] == "unknown", state["headline"])
check("it is never reported as up to date",
      "UP TO DATE" not in window.model_status_label.text(),
      window.model_status_label.text())
check("the headline still names when it was trained",
      "TRAINED" in state["headline"], state["headline"])
check("the panel admits the training data is not recorded",
      "not recorded" in window.model_data_label.text(),
      window.model_data_label.text())

#  2. A model that does record its training data, matching what is on disk.
fingerprint = T.dataset_fingerprint(app_root)
current_payload = dict(payload)
current_payload["data"] = dict(fingerprint)
joblib.dump(current_payload, MODEL_PATH)
check("model with provenance reloads", window.predictor.load(),
      window.predictor.status)
window._refresh_model_status(force=True)
check("a model trained on exactly what is on disk reads as up to date",
      "UP TO DATE" in window.model_status_label.text(),
      window.model_status_label.text())
check("the panel names the data it was trained on",
      "recordings" in window.model_data_label.text(),
      window.model_data_label.text())
check("the panel names the subjects",
      "trainee" in window.model_data_label.text(),
      window.model_data_label.text())

#  3. Record one more, and the same model must report itself out of date.
extra_writer = T.DatasetWriter(app_root, "trainee", "session_new")
extra = T.Recording(
    np.abs(rng2.normal(PROFILES["FIST"], 10.0, (90, 8))).astype(int),
    np.arange(90) / 60.0 + 1_700_000_000.0, "FIST", "trainee",
    "session_new", 999, label_start_index=10, label_end_index=80,
    meta={"emg_mode": "filtered"})
extra_writer.save(extra, T.STAGE_RAW)
window._refresh_model_status(force=True)
check("one new recording makes the model out of date",
      "OUT OF DATE" in window.model_status_label.text(),
      window.model_status_label.text())
check("it says how many recordings are new",
      "1 recording(s) have been added"
      in window.model_status_label.toolTip(),
      " ".join(window.model_status_label.toolTip().split())[:90])
check("the model itself still works while out of date",
      window.predictor.ready)

print("\n--- the training progress panel ---")
#  The panel is driven entirely by TRAINEVENT lines from the trainer. Feed it
#  the real shapes and check it reports them rather than inventing anything.
window.training_started_at = time.perf_counter() - 60.0
window._reset_training_panel("train_model.py -- 119 recordings (full sweep)")
check("panel starts queued, not at a stale percentage",
      window.train_fields["status"].text() == "queued"
      and window.train_bar.value() == 0,
      window.train_fields["status"].text())
check("panel names the job",
      "119 recordings" in window.train_job_label.text(),
      window.train_job_label.text())

window._on_train_event({
    "done": 120, "total": 1336, "phase": "stage A", "device": "cpu",
    "model": "SVM-rbf-C10", "window_s": 0.2, "overlap": 0.75,
    "representation": "raw", "feature_set": "both", "n_windows": 6307,
    "n_features": 154, "n_folds": 6, "status": "running",
    "message": "SVM-rbf-C10 on 0.20s raw/both"})
check("progress bar follows real completions",
      abs(window.train_bar.value() / 1000 - 120 / 1336) < 0.002,
      str(window.train_bar.value()))
check("status shows running", window.train_fields["status"].text() == "running")
check("stage is shown", window.train_fields["stage"].text() == "stage A")
check("algorithm is shown",
      window.train_fields["model"].text() == "SVM-rbf-C10")
check("candidate count is real",
      window.train_fields["candidate"].text() == "120 / 1336",
      window.train_fields["candidate"].text())
check("preprocessing is spelled out",
      "0.20s" in window.train_fields["setup"].text()
      and "raw/both" in window.train_fields["setup"].text()
      and "6307 windows" in window.train_fields["setup"].text(),
      window.train_fields["setup"].text())
check("fold count is shown", "6" in window.train_fields["folds"].text(),
      window.train_fields["folds"].text())
check("device is reported as CPU",
      window.train_fields["device"].text() == "CPU",
      window.train_fields["device"].text())
check("elapsed is real time, not invented",
      window.train_fields["elapsed"].text() not in ("--", ""),
      window.train_fields["elapsed"].text())
check("an ETA appears once there is enough work to estimate from",
      "about" in window.train_fields["eta"].text(),
      window.train_fields["eta"].text())

#  Too early to estimate: it must say so rather than guess.
window._reset_training_panel("job")
window._on_train_event({"done": 1, "total": 1336, "phase": "stage A",
                        "status": "running"})
check("no ETA is invented from one completed unit",
      window.train_fields["eta"].text() == "estimating...",
      window.train_fields["eta"].text())

window._on_train_event({
    "done": 1336, "total": 1336, "phase": "done", "status": "completed",
    "model": "LDA-shrinkage", "device": "cpu", "window_accuracy": 1.0,
    "recording_vote_accuracy": 1.0, "window_macro_f1": 1.0,
    "n_recordings": 119, "artifact": "models/gesture_model.joblib",
    "message": "saved LDA-shrinkage"})
check("completion is reported", window.train_fields["status"].text() == "completed")
check("the bar reaches 100%", window.train_bar.value() == 1000,
      str(window.train_bar.value()))
check("the saved artifact is named",
      "gesture_model.joblib" in window.train_fields["artifact"].text(),
      window.train_fields["artifact"].text())
check("final metrics are shown",
      "per gesture" in window.train_message.text(),
      window.train_message.text())

#  A malformed event must not take the output pump down.
before = window.train_fields["status"].text()
try:
    window._on_train_event({"done": "nonsense"})
    survived = True
except Exception:
    survived = False
check("a malformed event does not raise", survived)

print("\n--- model stats dialog ---")
dialog = T.ModelStatsDialog(window.predictor)
check("stats dialog built", dialog is not None)
text = dialog.findChildren(QtWidgets.QPlainTextEdit)
joined = " ".join(t.toPlainText() for t in text)
check("shows the validation protocol",
      "StratifiedGroupKFold" in joined or "leave-recordings-out" in joined,
      joined[:70])
check("warns about the single subject",
      "NOT MEASURABLE" in joined or "one subject" in joined)
check("shows the preprocessing", "level" in joined, joined[-120:])
tables = dialog.findChildren(QtWidgets.QTableWidget)
check("per-class and confusion tables present", len(tables) >= 2,
      str(len(tables)))
dialog.close()

print("\n--- shutdown ---")
arm.stop()
window.close()
pump(0.4)
check("prediction timer stopped on close", not window.pred_timer.isActive())
check("acquisition stopped", not window.acq.is_alive())

shutil.rmtree(ROOT, ignore_errors=True)
shutil.rmtree(MODEL_DIR, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

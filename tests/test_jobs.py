# -*- coding: utf-8 -*-
"""
Checks for background training jobs and the live TEST MODEL score.

No hardware, no armband, no trained model needed. Four subjects:

  1. the RUNNER -- train_job.py runs steps in order, marks each in the log,
     stops the chain at the first failure, and dies quickly when cancelled.
     And the point of the whole thing: a job outlives the process that
     started it.

  2. the WINDOW -- a MainWindow reattaches to a job that was already running
     when it opened, routes a chain's log to the right step, finishes each
     step through the ordinary handlers, and reports a job that finished
     while no window was open exactly once.

  3. the SCORE -- ModelTestSession cues an interleaved plan, never scores the
     settle period, and its three numbers (window / smoothed / trial) come
     from matrices whose metrics can be recomputed from them.

  4. the DIALOG -- TEST MODEL runs end to end against a fake armband and a
     fake model, appends one line to the wearer's history, and says when the
     sitting is in the training data.

Run:  python tests\\test_jobs.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "training"))   # the trainers

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T                                    # noqa: E402

PASS = 0
FAIL = 0
TMP = []
PY = sys.executable


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def tmpdir(prefix="emgjob_"):
    path = tempfile.mkdtemp(prefix=prefix)
    TMP.append(path)
    return path


def step(name, code=0, secs=0.05, n=3):
    """A stand-in trainer: prints PROGRESS lines, sleeps, exits with code."""
    body = (f"import time, sys\n"
            f"for i in range({n}):\n"
            f"    print('PROGRESS', i + 1, {n}, '{name}', flush=True)\n"
            f"    time.sleep({secs})\n"
            f"sys.exit({code})\n")
    return {"name": name, "argv": [PY, "-c", body]}


def wait_status(job_dir, seconds=30):
    t0 = time.time()
    while time.time() - t0 < seconds:
        st = T.read_job_status(job_dir)
        if st and st.get("state") != "running":
            return st
        time.sleep(0.1)
    return T.read_job_status(job_dir)


def log_lines(job_dir):
    return T.JobLogTail(os.path.join(job_dir, "log.txt")).read_lines(
        final=True)


# ==============================================================================
section("1. the runner -- order, markers, failure, cancel, and survival")
# ==============================================================================
models = tmpdir()

job = T.start_training_job(models, "all", [step("ml"), step("nn")])
st = wait_status(job)
lines = log_lines(job)
markers = [l for l in lines if l.startswith("JOBSTEP")]
check("a two-step chain finishes with exit 0",
      st and st["state"] == "finished" and st["exit_code"] == 0, str(st))
check("each step is bracketed by JOBSTEP / JOBSTEPEND, in order",
      markers == ["JOBSTEP 1 2 ml", "JOBSTEPEND ml 0",
                  "JOBSTEP 2 2 nn", "JOBSTEPEND nn 0"], str(markers))
check("both steps' own output lands in the one log",
      sum(l.startswith("PROGRESS") for l in lines) == 6)
check("the runner is gone afterwards", not T.pid_alive(st["pid"]))
check("a finished job nobody watched is reported as unseen",
      job in T.unseen_finished_jobs(models))
T.mark_job_seen(job)
check("...and only once", job not in T.unseen_finished_jobs(models))

job = T.start_training_job(models, "all", [step("ml", code=3), step("nn")])
st = wait_status(job)
check("a failing step stops the chain -- the net never trains on a dataset "
      "the classifier refused",
      st["exit_code"] == 3 and st["steps"][1]["exit_code"] is None
      and not any("JOBSTEP 2" in l for l in log_lines(job)), str(st))
T.mark_job_seen(job)

job = T.start_training_job(models, "ml", [step("ml", secs=5)])
time.sleep(1.5)
check("a running job is what a new window would reattach to",
      T.active_training_job(models) == job)
t0 = time.time()
T.cancel_training_job(job)
st = wait_status(job)
check("CANCEL stops it within seconds, and says it was cancelled",
      st["state"] == "cancelled" and time.time() - t0 < 6,
      f"{st['state']} after {time.time() - t0:.1f} s")
check("nothing is live afterwards", T.active_training_job(models) is None)
T.mark_job_seen(job)

check("pid_alive of a dead pid is False -- and on Windows it must not "
      "signal it: os.kill(pid, 0) there TERMINATES the process",
      not T.pid_alive(999999) and T.pid_alive(os.getpid()))

#  THE POINT: the job outlives whatever started it.
starter = (
    "import os, sys\n"
    f"sys.path.insert(0, {PROJECT!r})\n"
    "import emg_dataset_tool as T\n"
    f"j = T.start_training_job({models!r}, 'ml', "
    f"[{{'name': 'ml', 'argv': [{PY!r}, '-c', "
    f"'import time\\nfor i in range(6): print(\"PROGRESS\", i + 1, 6, "
    f"flush=True); time.sleep(0.5)']}}])\n"
    "print(j, flush=True)\n"
    "os._exit(0)\n")
done = subprocess.run([PY, "-c", starter], capture_output=True, text=True,
                      timeout=60)
survivor = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
time.sleep(0.8)
check("a job is still running after the process that started it has exited",
      survivor and T.job_is_live(survivor), done.stderr[-300:])
st = wait_status(survivor) if survivor else None
check("...and finishes on its own",
      st and st["state"] == "finished" and st["exit_code"] == 0, str(st))
if survivor:
    T.mark_job_seen(survivor)

#  THE BUG THIS PINS: on Windows os.replace onto a file another process has
#  open fails with "Access is denied". The GUI reads status.json three times
#  a second, so an unretried replace crashed the runner now and then and left
#  "running" in the status forever. Hammer it with a reader and count.
import threading                                                # noqa: E402
import train_job                                                # noqa: E402

race = os.path.join(tmpdir(), "status.json")
train_job.write_status(race, {"i": -1})
stop_reader = []


def _reader():
    while not stop_reader:
        try:
            with open(race, encoding="utf-8") as fh:
                fh.read()
        except OSError:
            pass


reader = threading.Thread(target=_reader)
reader.start()
errors = 0
for i in range(300):
    try:
        train_job.write_status(race, {"i": i})
    except OSError:
        errors += 1
stop_reader.append(True)
reader.join()
check("status writes survive a reader polling the file -- every one lands",
      errors == 0 and json.load(open(race, encoding="utf-8"))["i"] == 299,
      f"{errors} failed")

#  The log is read while it is written; half a line must never be parsed.
partial = os.path.join(tmpdir(), "log.txt")
with open(partial, "wb") as fh:
    fh.write(b"PROGRESS 1 3 x\nTRAINEVENT {\"a\": ")
tail = T.JobLogTail(partial)
first = tail.read_lines()
with open(partial, "ab") as fh:
    fh.write(b"1}\n")
second = tail.read_lines()
check("a half-written line is held until its newline arrives",
      first == ["PROGRESS 1 3 x"] and second == ['TRAINEVENT {"a": 1}'],
      f"{first} / {second}")


# ==============================================================================
section("2. the window -- reattach, route a chain, report what it missed")
# ==============================================================================
from PySide6 import QtCore, QtWidgets                           # noqa: E402

QtWidgets.QMessageBox.information = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.warning = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Ok)
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

root = tmpdir("emgjobgui_")
T.DATA_FOLDER = os.path.join(root, "datasets")
T.MODEL_PATH = os.path.join(root, "models", "gesture_model.joblib")
os.makedirs(os.path.dirname(T.MODEL_PATH), exist_ok=True)
T.MYO_PORT = "COM254"
T.PERSON_ID = "Tester"
T.SESSION_ID = "session_09"


def pump(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.02)


def pump_until(cond, seconds=30):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


models_dir = os.path.dirname(os.path.abspath(T.MODEL_PATH))
#  A job that finished before the window existed, and one still running.
missed = T.start_training_job(models_dir, "ml", [step("ml")])
wait_status(missed)
running = T.start_training_job(models_dir, "all",
                               [step("ml", secs=0.4, n=4),
                                step("nn", secs=0.4, n=4)])
time.sleep(0.5)

window = T.MainWindow()
check("the window reattaches to the job that was already running",
      pump_until(lambda: window._job is not None, 10)
      and window._job.job_dir == running)
check("the chain shows as running, with its cancel button",
      window._chain_job is not None
      and "CANCEL" in window.retrain_all_button.text(),
      window.retrain_all_button.text())
check("the run that finished while no window was open is reported",
      "While the app was closed" in window.log_box.toPlainText())
check("...and marked seen, so the next window does not report it again",
      missed not in T.unseen_finished_jobs(models_dir))

saw_nn = pump_until(lambda: window.nn_training_process is not None, 20)
check("step 2 hands the job to the net's slot -- the classifier slot is "
      "free again", saw_nn and window.training_process is None)
check("the job is released when the runner ends",
      pump_until(lambda: window._job is None, 30))
check("both slots are empty and the chain is over",
      window.training_process is None and window.nn_training_process is None
      and window._chain_job is None)
check("the ML + NN button is a start button again",
      window.retrain_all_button.text() == "RETRAIN ML + NEURAL NET",
      window.retrain_all_button.text())
check("progress lines from the log reached the panel",
      any("ml" in l or "nn" in l for l in window.training_output)
      or window.training_progress.value() >= 0)

#  A runner killed from outside must not leave the panel at "training".
dead = T.start_training_job(models_dir, "ml", [step("ml", secs=20)])
time.sleep(1.0)
watcher = window._attach_training_job(dead)
status = T.read_job_status(dead)
subprocess.run(["taskkill", "/T", "/F", "/PID", str(status["pid"])],
               capture_output=True)
check("a runner that dies without a word is noticed and the slot released",
      pump_until(lambda: window._job is None, 15))
window.close()
window.deleteLater()
pump(0.2)


# ==============================================================================
section("3. the score -- plan, settle, and three numbers from three matrices")
# ==============================================================================
labels = ["FIST", "OPEN", "REST"]
s = T.ModelTestSession(None, labels, rounds=4, seed=7, prepare_s=1.0,
                       hold_s=2.0, settle_s=0.5)
check("every gesture appears once per round",
      sorted(s.plan) == sorted(labels * 4), str(s.plan))
check("never the same gesture twice in a row -- that would not be a new "
      "trial", all(a != b for a, b in zip(s.plan, s.plan[1:])), str(s.plan))

#  OPEN is always read as FIST; the others are right.
answer = {"FIST": "FIST", "OPEN": "FIST", "REST": "REST"}
t = 0.0
s.start(t)
resets = 0
scored_in_settle = 0
while not s.finished:
    t += 0.05
    if s.tick(t):
        resets += 1
    if s.state == s.HOLD and t < s.hold_started + s.settle_s and s.scoring(t):
        scored_in_settle += 1
    if s.scoring(t):
        s.add(answer[s.current], answer[s.current])
r = s.summary()
check("tick() says when each hold begins, so smoothing can be reset",
      resets == len(s.plan), f"{resets} vs {len(s.plan)}")
check("nothing is scored during the settle period", scored_in_settle == 0)
check("every trial was scored", r["trials_scored"] == len(s.plan))
check("per-trial accuracy is 2 of 3 gestures right",
      abs(r["per_trial"]["accuracy"] - 2 / 3) < 1e-9,
      str(r["per_trial"]["accuracy"]))
check("the confusion matrix puts every OPEN trial under FIST",
      r["confusion_trial"][1] == [4, 0, 0], str(r["confusion_trial"]))
m = T.confusion_metrics(r["confusion_window"], labels)
check("the metrics shown are recomputable from the matrix shown",
      m == r["per_window"] or (abs(m["macro_f1"]
                                   - r["per_window"]["macro_f1"]) < 1e-12))
check("macro F1 matches the textbook value for this matrix",
      abs(r["per_trial"]["macro_f1"] - np.mean([2 * 1 * .5 / 1.5, 0, 1]))
      < 1e-9, str(r["per_trial"]["macro_f1"]))
empty = T.confusion_metrics(np.zeros((3, 3)), labels)
check("an empty matrix gives NaN, not a crash or a fake 0",
      np.isnan(empty["accuracy"]) and np.isnan(empty["macro_f1"]))
early = T.ModelTestSession(None, labels, rounds=2, seed=1)
early.start(0.0)
early.tick(5.0)
early.stop()
check("a test stopped early reports how far it got, not the whole plan",
      early.summary()["trials_reached"] == 1,
      str(early.summary()["trials_reached"]))

hist_root = tmpdir()
T.save_model_test(hist_root, "Tester", {"per_trial": {"accuracy":
                                                       float("nan")}})
T.save_model_test(hist_root, "Tester", r)
back = T.load_model_tests(hist_root, "Tester")
check("results append to the wearer's history and read back",
      len(back) == 2 and back[1]["trials_scored"] == r["trials_scored"])
check("NaN is stored as null, so the history stays valid JSON",
      back[0]["per_trial"]["accuracy"] is None)


# ==============================================================================
section("4. the dialog -- TEST MODEL end to end on a fake band and model")
# ==============================================================================
class _FakeRing:
    def __init__(self):
        self.level = np.full(8, 20.0)

    def snapshot(self, n):
        t = np.arange(n) / 50.0 + time.perf_counter()
        return np.tile(self.level, (n, 1)), t


class _FakeAcq:
    state = "connected"

    def __init__(self):
        self.ring = _FakeRing()
        self.age = 0.05

    def stats(self):
        return {"rate": 50.0, "last_sample_age": self.age}


class _FakePredictor:
    """Answers OPEN for everything: a model that is always wrong but one."""

    labels = ["FIST", "OPEN", "REST"]
    ready = True
    status = "ready"
    stats = {"model_name": "Fake", "window_accuracy": 0.5,
             "recording_vote_accuracy": 0.6}
    full_path = "fake.joblib"

    def __init__(self, sessions):
        self.payload_meta = {"data": {"subjects": ["Tester"],
                                      "sessions": sessions}}
        self.resets = 0

    def window_seconds(self):
        return 0.4

    def reset(self):
        self.resets += 1

    def predict(self, emg, t):
        return T.Prediction(label="OPEN", confidence=0.9, raw_label="OPEN",
                            raw_confidence=0.9, probabilities=[0, .9, .1],
                            raw_probabilities=[0, .9, .1],
                            labels=self.labels, n_samples=len(emg),
                            smoothed_over=1, unix_time=time.time())


class _FakeWindow(QtWidgets.QWidget):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.acq = _FakeAcq()
        self.logged = []
        self.model_test_dialog = None

    def stop_prediction(self):
        pass

    def _append_log(self, text):
        self.logged.append(text)


T.DATA_FOLDER = os.path.join(root, "datasets")
T.PROJECT_DIR = root
T.MODEL_TEST_PREPARE_S = 0.1
T.MODEL_TEST_SETTLE_S = 0.1
T.PREDICTION_INTERVAL_MS = 30

in_data = _FakeWindow(_FakePredictor(["Tester/session_09"]))
dlg = T.ModelTestDialog(in_data)
check("a sitting the model trained on is flagged as optimistic",
      "IN this model's training data" in dlg.honesty.text(),
      dlg.honesty.text())
dlg.close()

fresh = _FakeWindow(_FakePredictor(["Tester/session_01"]))
dlg = T.ModelTestDialog(fresh)
check("a new sitting of a known wearer is named as such",
      "NEW sitting" in dlg.honesty.text(), dlg.honesty.text())
dlg.rounds_spin.setValue(1)
dlg.hold_spin.setValue(2.0)
dlg.toggle()
check("the test starts, and the button becomes STOP",
      dlg.session is not None and dlg.start_button.text() == "STOP")
finished = pump_until(lambda: dlg.session.finished, 20)
check("it runs to the end on its own", finished)
check("the smoothing is reset at every hold",
      fresh.predictor.resets >= len(dlg.session.plan))
res = dlg.result or {}
check("a result was produced and scored only OPEN right",
      res and abs(res["per_trial"]["accuracy"] - 1 / 3) < 1e-9,
      str(res.get("per_trial")))
check("it is appended to the wearer's model_tests.jsonl",
      len(T.load_model_tests(T.DATA_FOLDER, "Tester")) == 1)
check("the result names the model and whether the wearer was trained on",
      res.get("model", {}).get("name") == "Fake"
      and res.get("wearer_in_training_data") is True
      and res.get("sitting_in_training_data") is False)
check("the results view shows the row-percentage confusion matrix",
      "% of each row" in dlg.results.toPlainText()
      or "of each row" in dlg.results.toHtml())

dlg.toggle()
fresh.acq.age = 9.0
check("a stream that stops mid-test ends it, partial result kept",
      pump_until(lambda: dlg.session.finished, 5)
      and "stopped" in dlg.cue.text())
dlg.close()

# ------------------------------------------------------------------------------
for path in TMP:
    shutil.rmtree(path, ignore_errors=True)
print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)

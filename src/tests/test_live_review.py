"""
The review step against the REAL armband.

Confirms that with live EMG the graph freezes on the take, the proposed cut is
drawn from real activity detection, ACCEPT writes exactly the banded slice of
real samples while the raw files stay byte-identical, and REDO repeats the
same repetition.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


ROOT = tempfile.mkdtemp(prefix="emglivereview_")
T.DATA_FOLDER = ROOT
T.GESTURES = ["FIST"]
T.REPETITIONS = 2
T.GESTURE_ORDER = "blocked"
T.REST_TIME = 1.0
T.PREPARE_TIME = 0.5
T.COUNTDOWN_TIME = 2.0
T.RECORD_TIME = 1.5
T.PAUSE_TIME = 0.3
T.SCAN_TIMEOUT_S = 12.0
T.QC_PAUSE_ON = "never"
T.REVIEW_EACH_RECORDING = True
T.REVIEW_SHOW_TRIM = True
T.REVIEW_SAVE_TRIM_ON_ACCEPT = True
T.REVIEW_QUARANTINE_ON_REDO = True
T.COUNTDOWN_OVERLAY = "corner"

from PySide6 import QtWidgets                                  # noqa: E402

QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)

app = QtWidgets.QApplication(sys.argv)
T.apply_theme(app)


def pump(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.002)


def wait_for(predicate, timeout=60.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        pump(0.02)
    return False


window = T.MainWindow()
window.show()

check("connected to the real Myo",
      wait_for(lambda: window.acq.connected, 70),
      f"{window.acq.state}: {window.acq.state_message}")
if not window.acq.connected:
    window.close()
    shutil.rmtree(ROOT, ignore_errors=True)
    sys.exit(1)

pump(1.5)
check("live EMG reaching the GUI", window.acq.stats()["samples"] > 30,
      str(window.acq.stats()["samples"]))

window.person_edit.setText("live_review")
window.session_edit.setText("session_01")
window._on_identity_changed()
window.reps_spin.setValue(2)
window._refresh_plan_info()

print("\n" + "=" * 62)
print("  Follow the cues: clench a FIST when it says RECORD NOW.")
print("  Two repetitions. No input needed -- the test accepts for you.")
print("=" * 62 + "\n")

window.start_session()
check("session started", window.sequence.running)

check("reached REVIEW with a real take",
      wait_for(lambda: window.sequence.reviewing, 70), window.sequence.state)

rec = window.review_recording
check("review holds a real recording",
      rec is not None and rec.n_samples > 30,
      str(rec.n_samples) if rec else "none")
check("plot frozen on the take", window.plot.review_mode)
check("trim band shown", window.plot.trim_regions[0].isVisible())
check("curves show the whole take",
      len(window.plot.curves[0].getData()[0]) == rec.n_samples,
      f"{len(window.plot.curves[0].getData()[0])} vs {rec.n_samples}")

print(f"  take: {rec.n_samples} samples, {rec.duration:.2f} s @ {rec.fs:.1f} Hz")
print(f"  peak per channel: {rec.emg.max(axis=0).tolist()}")
detected = rec.activity is not None and rec.activity.found
print(f"  activity detected: {detected}"
      + (f"  region {rec.activity.start}..{rec.activity.end}"
         f"  baseline {rec.activity.baseline:.1f}"
         f"  threshold {rec.activity.threshold_high:.1f}" if detected else ""))

start, end = window.plot.trim_indices(rec.n_samples)
if detected:
    want = (int(np.clip(rec.activity.start, 0, rec.n_samples - 1)),
            int(min(rec.activity.end, rec.n_samples)))
else:
    want = (0, rec.n_samples)
check("band matches the detector on real data",
      abs(start - want[0]) <= 1 and abs(end - want[1]) <= 1,
      f"band={start}..{end} detector={want[0]}..{want[1]}")
check("review read-out populated", "keeping" in window.review_info.text(),
      window.review_info.text()[:80])

print("\n--- live stream keeps running behind the review ---")
before = window.acq.stats()["samples"]
pump(1.0)
check("acquisition never paused for the review",
      window.acq.stats()["samples"] > before,
      f"{before} -> {window.acq.stats()['samples']}")

print("\n--- drag the band, then ACCEPT ---")
t = rec.t_rel
lo = float(t[max(1, rec.n_samples // 4)])
hi = float(t[min(rec.n_samples - 1, (rec.n_samples * 3) // 4)])
window.plot.set_trim_bounds(lo, hi)
pump(0.1)
start, end = window.plot.trim_indices(rec.n_samples)
expected = end - start
check("band follows the drag on real data", 2 < expected < rec.n_samples,
      f"keeping {expected} of {rec.n_samples}")

raw_before = rec.emg.copy()
raw_files = {os.path.join(window.writer.root, p): None
             for k, p in rec.files.items() if k.startswith("raw:")}
raw_bytes = {p: open(p, "rb").read() for p in raw_files if os.path.exists(p)}

window.accept_review()
pump(0.3)

trimmed_paths = T.discover_recordings(ROOT, stage=T.STAGE_TRIMMED)
check("trimmed copy written", len(trimmed_paths) == 1, str(len(trimmed_paths)))
if trimmed_paths:
    saved = T.load_recording(trimmed_paths[0])
    check("trimmed file is exactly the band", saved.n_samples == expected,
          f"{saved.n_samples} vs {expected}")
    check("trimmed content is the right slice of real raw data",
          np.array_equal(saved.emg, raw_before[start:end]))
check("RAW FILES byte-identical after accepting",
      all(open(p, "rb").read() == b for p, b in raw_bytes.items()),
      f"{len(raw_bytes)} file(s)")
check("advanced past the first repetition", window.sequence.index == 1,
      str(window.sequence.index))

print("\n--- REDO the second take ---")
check("reached REVIEW for repetition 2",
      wait_for(lambda: window.sequence.reviewing, 70), window.sequence.state)
rejected = window.review_recording
item = window.sequence.current
raw_before_count = len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))

window.redo_review()
pump(0.3)

check("index unchanged after REDO", window.sequence.index == 1,
      str(window.sequence.index))
check("same repetition queued again", window.sequence.current == item,
      f"{window.sequence.current} vs {item}")
deleted = os.path.join(window.writer.session_dir, "_deleted")
check("rejected real take moved to _deleted",
      os.path.isdir(deleted) and len(os.listdir(deleted)) >= 1,
      str(len(os.listdir(deleted))) if os.path.isdir(deleted) else "missing")
check("rejected take dropped from the dataset",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == raw_before_count - 1)

print("\n--- the retry records again, then accept to finish ---")
check("retry reached REVIEW",
      wait_for(lambda: window.sequence.reviewing, 70), window.sequence.state)
retry = window.review_recording
check("retry is a new recording of the same item",
      retry is not rejected and (retry.label, retry.repetition) == item,
      f"{(retry.label, retry.repetition)} vs {item}")

guard = 0
while window.sequence.running and guard < 10:
    if window.sequence.reviewing:
        window.accept_review()
        guard += 1
    pump(0.05)

check("session completed", window.sequence.state == T.STATE_DONE,
      window.sequence.state)
check("two accepted recordings on disk",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == 2,
      str(len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))))

reloaded = T.load_dataset(ROOT, stage="prefer_trimmed")
check("dataset reloads", len(reloaded) == 2, str(len(reloaded)))
X, y, groups, meta = T.make_windows(reloaded, fs=reloaded[0].fs)
check("windows build from the reviewed data", X.shape[0] > 0, str(X.shape))

window.close()
pump(0.5)
check("acquisition stopped", not window.acq.is_alive())

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

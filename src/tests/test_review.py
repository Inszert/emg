"""
Offscreen test of the operator-paced review step:

  * the sequence STOPS at REVIEW and does not advance on its own
  * the graph freezes on the take just captured, with a draggable trim band
  * the countdown overlay does not cover the traces ("corner" placement)
  * ACCEPT writes the trimmed copy for the band and moves on
  * dragging the band changes what gets written
  * REDO quarantines the rejected take and repeats the SAME repetition
  * the raw file is never modified by any of it
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


ROOT = tempfile.mkdtemp(prefix="emgreview_")
T.DATA_FOLDER = ROOT
T.MYO_PORT = "COM254"
T.GESTURES = ["FIST", "OPEN"]
T.REPETITIONS = 2
T.GESTURE_ORDER = "blocked"
T.REST_TIME = 0.05
T.PREPARE_TIME = 0.05
T.COUNTDOWN_TIME = 0.10
T.RECORD_TIME = 0.40
T.PAUSE_TIME = 0.05
T.PRE_GESTURE_MS = 60.0
T.POST_GESTURE_MS = 60.0
T.RECONNECT_DELAY_S = 0.3
T.QC_PAUSE_ON = "never"
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.REVIEW_EACH_RECORDING = True
T.REVIEW_SHOW_TRIM = True
T.REVIEW_SAVE_TRIM_ON_ACCEPT = True
T.REVIEW_QUARANTINE_ON_REDO = True
T.COUNTDOWN_OVERLAY = "corner"

from PySide6 import QtCore, QtWidgets                          # noqa: E402

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


def wait_for(predicate, timeout=20.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        pump(0.02)
    return False


class FakeArm:
    """Synthetic samples through the real _ingest path, with a burst on cue."""

    def __init__(self, acq, fs=50.0):
        self.acq, self.fs = acq, fs
        self.active = 0.0
        self.running = True
        self.rng = np.random.default_rng(11)
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
            base = np.abs(self.rng.normal(0, 6, 8)) + 5
            if self.active:
                base += self.rng.random(8) * self.active
            self.acq._ingest(np.clip(base, 0, 1023).astype(int))


window = T.MainWindow()
window.show()
pump(0.3)
arm = FakeArm(window.acq)
arm.start()
pump(0.5)

# Burst only while RECORDING, so there is a real gesture to find.
driver = QtCore.QTimer()
driver.timeout.connect(
    lambda: setattr(arm, "active",
                    600.0 if window.sequence.state == T.STATE_RECORDING else 0.0)
)
driver.start(10)

print("\n--- countdown overlay must not cover the traces ---")
# The banner timer clears the overlay whenever the state is not COUNTDOWN,
# so pause it while checking the overlay geometry directly.
window.seq_timer.stop()
pump(0.05)
window.plot.set_overlay("3", "rgba(251,146,60,220)")
pump(0.05)
plot_rect = window.plot.glw.rect()
badge = window.plot.overlay.geometry()
check("overlay is visible", window.plot.overlay.isVisible())
check("overlay covers only a small part of the canvas",
      badge.width() * badge.height() < plot_rect.width() * plot_rect.height() * 0.15,
      f"badge {badge.width()}x{badge.height()} vs canvas "
      f"{plot_rect.width()}x{plot_rect.height()}")
check("overlay sits at the top of the canvas",
      badge.top() < plot_rect.height() * 0.25, str(badge.top()))

T.COUNTDOWN_OVERLAY = "center"
window.plot.set_overlay("3")
pump(0.05)
centred = window.plot.overlay.geometry()
check("center mode really does cover the canvas",
      centred.width() >= plot_rect.width(), f"{centred.width()}")
T.COUNTDOWN_OVERLAY = "corner"

T.COUNTDOWN_OVERLAY = "off"
window.plot.set_overlay("3")
pump(0.05)
check("off mode draws nothing", not window.plot.overlay.isVisible())
T.COUNTDOWN_OVERLAY = "corner"
window.plot.set_overlay("")
window.seq_timer.start(20)

print("\n--- the session stops and waits at REVIEW ---")
window.person_edit.setText("reviewer")
window.session_edit.setText("session_01")
window._on_identity_changed()
window.reps_spin.setValue(2)
window._refresh_plan_info()

plan = list(T.build_plan())
check("plan is 4 recordings", len(plan) == 4, str(len(plan)))

window.start_session()
check("reached REVIEW", wait_for(lambda: window.sequence.reviewing, 25),
      window.sequence.state)
check("one recording captured", len(window.recordings) == 1,
      str(len(window.recordings)))

# The key property: it must NOT advance by itself.
state_before = window.sequence.state
index_before = window.sequence.index
pump(2.5)
check("REVIEW does not time out",
      window.sequence.state == state_before
      and window.sequence.index == index_before,
      f"{window.sequence.state} index={window.sequence.index}")
check("still only one recording after waiting",
      len(window.recordings) == 1, str(len(window.recordings)))

print("\n--- the graph is frozen on the take, with a trim band ---")
rec = window.review_recording
check("review recording is set", rec is not None)
check("plot is in review mode", window.plot.review_mode)
check("trim band visible", window.plot.trim_regions[0].isVisible())
check("all 8 channels show the take",
      all(window.plot.curves[i].getData()[0] is not None
          and len(window.plot.curves[i].getData()[0]) == rec.n_samples
          for i in range(8)),
      str([len(window.plot.curves[i].getData()[0]) for i in range(8)]))

x_min, x_max = window.plot.plots[0].viewRange()[0]
check("x axis shows the recording, not the live window",
      x_max > 0 and x_min >= -0.2, f"[{x_min:.2f}, {x_max:.2f}]")

start, end = window.plot.trim_indices(rec.n_samples)
check("trim band is a sane sub-range",
      0 <= start < end <= rec.n_samples, f"{start}..{end}/{rec.n_samples}")
#  The band must be exactly what the detector proposed (clipped to the
#  recording), or the full range when it found nothing -- it must never
#  silently drop data because detection failed. On these short synthetic
#  takes the configured padding often expands the region to the full range,
#  which is correct and is why this compares against the detector rather
#  than just asserting the band is narrower.
detected = rec.activity is not None and rec.activity.found
if detected:
    want = (int(np.clip(rec.activity.start, 0, rec.n_samples - 1)),
            int(min(rec.activity.end, rec.n_samples)))
else:
    want = (0, rec.n_samples)
check("band matches what the detector proposed",
      abs(start - want[0]) <= 1 and abs(end - want[1]) <= 1,
      f"found={detected} band={start}..{end} detector={want[0]}..{want[1]}")
check("review read-out populated", "keeping" in window.review_info.text(),
      window.review_info.text()[:70])

print("\n--- the live stream keeps running behind the review ---")
samples_before = window.acq.stats()["samples"]
pump(0.6)
check("acquisition never paused for the review",
      window.acq.stats()["samples"] > samples_before,
      f"{samples_before} -> {window.acq.stats()['samples']}")
check("plot still frozen on the take", window.plot.review_mode)

print("\n--- dragging the band changes what will be kept ---")
t = rec.t_rel
window.plot.set_trim_bounds(float(t[2]), float(t[max(3, rec.n_samples // 2)]))
pump(0.1)
dragged = window.plot.trim_indices(rec.n_samples)
check("band follows the drag", dragged != (start, end), str(dragged))
check("every channel band moved together",
      len({tuple(np.round(r.getRegion(), 6)) for r in window.plot.trim_regions}) == 1)
window.plot._on_trim_region_changed(window.plot.trim_regions[0])
pump(0.05)
check("read-out tracks the drag",
      f"keeping {dragged[1] - dragged[0]} " in window.review_info.text(),
      window.review_info.text()[:70])

print("\n--- ACCEPT writes the trimmed copy and advances ---")
raw_before = rec.emg.copy()
raw_files = [os.path.join(window.writer.root, p)
             for k, p in rec.files.items() if k.startswith("raw:")]
raw_bytes = {p: open(p, "rb").read() for p in raw_files if os.path.exists(p)}
expected_kept = dragged[1] - dragged[0]

window.accept_review()
pump(0.2)

check("index advanced", window.sequence.index == index_before + 1,
      str(window.sequence.index))
check("left review mode", not window.plot.review_mode)
check("review read-out cleared", window.review_info.text() == "")

trimmed_paths = T.discover_recordings(ROOT, stage=T.STAGE_TRIMMED)
check("a trimmed file was written", len(trimmed_paths) == 1, str(len(trimmed_paths)))
if trimmed_paths:
    saved = T.load_recording(trimmed_paths[0])
    check("trimmed file matches the band exactly",
          saved.n_samples == expected_kept,
          f"{saved.n_samples} vs {expected_kept}")
    check("trimmed content is the right slice of raw",
          np.array_equal(saved.emg, raw_before[dragged[0]:dragged[1]]))
    check("trim provenance recorded as operator review",
          saved.meta.get("trim_method") == "review"
          and saved.meta.get("trim_reviewed_by_operator") is True,
          str(saved.meta.get("trim_method")))

check("RAW UNCHANGED in memory", np.array_equal(rec.emg, raw_before))
check("RAW FILES byte-identical on disk",
      all(open(p, "rb").read() == b for p, b in raw_bytes.items()),
      f"{len(raw_bytes)} file(s) checked")
check("table marked as trimmed",
      window.table.item(0, 5) is not None and window.table.item(0, 5).text() == "yes",
      window.table.item(0, 5).text() if window.table.item(0, 5) else "none")

print("\n--- REDO quarantines the take and repeats the SAME repetition ---")
check("reached the next REVIEW", wait_for(lambda: window.sequence.reviewing, 25),
      window.sequence.state)
rejected = window.review_recording
item_before = window.sequence.current
index_before = window.sequence.index
rows_before = window.table.rowCount()
raw_count_before = len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))

window.redo_review()
pump(0.2)

check("index did NOT advance", window.sequence.index == index_before,
      str(window.sequence.index))
check("same repetition queued again", window.sequence.current == item_before,
      f"{window.sequence.current} vs {item_before}")
check("row removed from the list", window.table.rowCount() == rows_before - 1)
check("recording removed from memory", rejected not in window.recordings)

deleted_dir = os.path.join(window.writer.session_dir, "_deleted")
check("rejected files moved to _deleted, not erased",
      os.path.isdir(deleted_dir) and len(os.listdir(deleted_dir)) >= 1,
      str(len(os.listdir(deleted_dir))) if os.path.isdir(deleted_dir) else "missing")
check("rejected take no longer counted in the dataset",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == raw_count_before - 1,
      f"{raw_count_before} -> {len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))}")
check("row numbers renumbered",
      all(window.table.item(i, 0).text() == str(i + 1)
          for i in range(window.table.rowCount())))

print("\n--- the retried repetition is recorded again ---")
check("back to REVIEW for the retry",
      wait_for(lambda: window.sequence.reviewing, 25), window.sequence.state)
retry = window.review_recording
check("retry has the same label and repetition",
      (retry.label, retry.repetition) == item_before,
      f"{(retry.label, retry.repetition)} vs {item_before}")
check("retry is a different recording", retry is not rejected)

print("\n--- finish the plan by accepting everything ---")
guard = 0
while window.sequence.running and guard < 40:
    if window.sequence.reviewing:
        window.accept_review()
        guard += 1
    pump(0.05)

check("session completed", window.sequence.state == T.STATE_DONE,
      window.sequence.state)
check("collected the whole plan", len(window.recordings) == len(plan),
      f"{len(window.recordings)}/{len(plan)}")
check("raw files for the whole plan",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == len(plan),
      str(len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))))

print("\n--- fully automatic mode still works (review off) ---")
T.REVIEW_EACH_RECORDING = False
window.recordings.clear()
window.table.setRowCount(0)
window.reps_spin.setValue(1)
window._refresh_plan_info()
window.start_session()
auto_plan = len(window.sequence.plan)
deadline = time.perf_counter() + 40
while window.sequence.running and time.perf_counter() < deadline:
    pump(0.05)
check("automatic session ran without any input",
      window.sequence.state == T.STATE_DONE, window.sequence.state)
check("automatic session collected everything",
      len(window.recordings) == auto_plan,
      f"{len(window.recordings)}/{auto_plan}")

driver.stop()
arm.stop()
window.close()
pump(0.4)
check("acquisition stopped", not window.acq.is_alive())

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

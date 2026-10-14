"""
Offscreen test of LABEL ON GRAPH (free recording).

The property that matters most: a selection is anchored in ABSOLUTE time. The
x axis is "seconds relative to now", so a selection stored in axis coordinates
would slide along the signal as time passed and label the wrong samples. This
checks that it does not.
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


ROOT = tempfile.mkdtemp(prefix="emglabel_")
T.DATA_FOLDER = ROOT
T.MYO_PORT = "COM254"
T.GESTURES = ["REST", "FIST", "OPEN"]
T.PERSON_ID = "labeller"
T.SESSION_ID = "session_01"
T.RECONNECT_DELAY_S = 0.3
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.QC_PAUSE_ON = "never"
T.MODEL_PATH = os.path.join(ROOT, "no_model.joblib")
T.PLOT_WINDOW_S = 30.0
T.RING_BUFFER_SECONDS = 60.0
T.LABEL_MIN_SELECTION_S = 0.25
T.LABEL_DEFAULT_SELECTION_S = 2.0
T.GUI_FPS = 60

from PySide6 import QtWidgets                                  # noqa: E402

QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)

app = QtWidgets.QApplication(sys.argv)
T.apply_theme(app)

PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
}


class FakeArm:
    def __init__(self, acq, fs=60.0):
        self.acq, self.fs = acq, fs
        self.label = "REST"
        self.running = True
        self.rng = np.random.default_rng(5)
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
                np.abs(self.rng.normal(base, base * 0.12 + 1.5, 8)).astype(int))


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


window = T.MainWindow()
window.show()
pump(0.3)
arm = FakeArm(window.acq)
arm.start()
check("samples flowing", wait_for(lambda: window.acq.ring.total > 120, 10),
      str(window.acq.ring.total))

print("\n--- the graph window is 30 s ---")
check("plot window defaults to 30 s", abs(window.plot.window_s - 30.0) < 1e-6,
      str(window.plot.window_s))

# Panning is confined to the live axis: nothing exists after "now", and
# nothing before the start of the ring buffer. The bound is the ring's
# CAPACITY (_apply_live_limits), not how much has been recorded so far --
# so a fresh session can pan back over empty time inside the buffer, but
# never outside it.
#  SampleRing has no span helper -- the span is what the timestamps say.
_ring_emg, _ring_t = window.acq.ring.snapshot()
history = float(_ring_t[-1] - _ring_t[0]) if len(_ring_t) > 1 else 0.0
oldest = -float(T.RING_BUFFER_SECONDS) * 1.05
x_min, x_max = window.plot.plots[0].viewRange()[0]
check("x axis ends at now", x_max >= -0.5, f"{x_max:.1f}")
check("x axis never reaches further back than the buffer holds",
      x_min >= oldest - 0.5,
      f"axis {x_min:.1f}s vs buffer {oldest:.1f}s, {history:.1f}s recorded")
limits = window.plot.plots[0].getViewBox().state["limits"]["xLimits"]
check("panning cannot reach outside the buffer",
      limits[0] >= oldest - 1e-6 and limits[1] <= T.PLOT_FUTURE_MARGIN_S + 1e-6,
      f"limits {limits[0]:.1f}..{limits[1]:.1f}s")
check("the window setting is still 30 s regardless",
      abs(window.plot.window_s - 30.0) < 1e-6)
check("ring holds more than the window",
      T.RING_BUFFER_SECONDS > T.PLOT_WINDOW_S)

print("\n--- label mode ---")
check("label mode starts off", not window.plot.label_mode)
check("select/add disabled until label mode is on",
      not window.select_button.isEnabled()
      and not window.add_label_button.isEnabled())

window.label_mode_button.setChecked(True)
pump(0.2)
check("label mode on", window.plot.label_mode)
check("button renames to stop",
      "STOP" in window.label_mode_button.text(), window.label_mode_button.text())
check("select enabled", window.select_button.isEnabled())

print("\n--- the selection is anchored in ABSOLUTE time ---")
arm.label = "FIST"
pump(2.5)
window.arm_selection()
pump(0.1)
bounds = window.plot.selection_bounds()
check("a selection exists", bounds is not None)
check("selection is about the default width",
      abs((bounds[1] - bounds[0]) - T.LABEL_DEFAULT_SELECTION_S) < 0.25,
      f"{bounds[1] - bounds[0]:.2f} s")
check("selection regions visible on all 8 channels",
      all(r.isVisible() for r in window.plot.select_regions))

region_before = window.plot.select_regions[0].getRegion()
pump(1.2)
bounds_after = window.plot.selection_bounds()
region_after = window.plot.select_regions[0].getRegion()

check("ABSOLUTE bounds did not drift",
      abs(bounds_after[0] - bounds[0]) < 0.02
      and abs(bounds_after[1] - bounds[1]) < 0.02,
      f"{bounds[0]:.3f}->{bounds_after[0]:.3f}")
check("but it moved LEFT on the axis as time passed",
      region_after[0] < region_before[0] - 0.8,
      f"{region_before[0]:.2f} -> {region_after[0]:.2f}")
check("the box still covers the same duration",
      abs((region_after[1] - region_after[0])
          - (region_before[1] - region_before[0])) < 0.05)

print("\n--- too short is refused ---")
window.plot.set_trim_bounds(0, 0)          # unrelated; just proving no crash
_, t_now = window.acq.ring.latest()
window.plot.arm_selection(t_now, 0.05)
pump(0.1)
before_count = len(window.recordings)
window.add_selection_as_label()
pump(0.2)
check("a 0.05 s selection is refused",
      len(window.recordings) == before_count, str(len(window.recordings)))

print("\n--- ADD saves exactly the boxed signal ---")
window.label_gesture_combo.setCurrentText("FIST")
arm.label = "FIST"
pump(2.2)
_, t_now = window.acq.ring.latest()
window.plot.arm_selection(t_now, 1.5)
pump(0.15)
sel_start, sel_end = window.plot.selection_bounds()
expected_emg, expected_t = window._slice_ring(sel_start, sel_end)

window.add_selection_as_label()
pump(0.3)

check("one recording added", len(window.recordings) == before_count + 1,
      str(len(window.recordings)))
rec = window.recordings[-1]
check("labelled with the chosen gesture", rec.label == "FIST", rec.label)
check("samples match the selection exactly",
      np.array_equal(rec.emg, expected_emg),
      f"{rec.n_samples} vs {expected_emg.shape[0]}")
check("timestamps lie inside the selected range",
      rec.timestamps[0] >= sel_start - 1e-6
      and rec.timestamps[-1] <= sel_end + 1e-6)
check("the whole selection is the label window",
      rec.label_start_index == 0 and rec.label_end_index == rec.n_samples,
      f"{rec.label_start_index}..{rec.label_end_index}/{rec.n_samples}")
check("provenance records that a human selected it",
      rec.meta.get("capture_style") == "label_on_graph"
      and rec.meta.get("selected_by_operator") is True,
      str(rec.meta.get("capture_style")))
check("written to disk",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == 1,
      str(len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))))
check("selection cleared after adding",
      window.plot.selection_bounds() is None)

print("\n--- the labelled band stays on the graph ---")
check("a mark was added", len(window.plot.marks) == 1, str(len(window.plot.marks)))
mark = window.plot.marks[0]
check("mark remembers absolute time and label",
      abs(mark["start"] - sel_start) < 1e-6 and mark["label"] == "FIST"
      and mark["uid"] == rec.uid)
check("mark is drawn", wait_for(
    lambda: window.plot.mark_pool[0][0].isVisible(), 3))
band_before = window.plot.mark_pool[0][0].getRegion()
pump(1.2)
band_after = window.plot.mark_pool[0][0].getRegion()
check("the band scrolls left with the signal",
      band_after[0] < band_before[0] - 0.8,
      f"{band_before[0]:.2f} -> {band_after[0]:.2f}")
check("mark caption shows the gesture",
      window.plot.mark_labels[0].toPlainText() == "FIST",
      window.plot.mark_labels[0].toPlainText())

print("\n--- several gestures, each its own colour ---")
for gesture in ("OPEN", "REST", "FIST"):
    window.label_gesture_combo.setCurrentText(gesture)
    arm.label = gesture
    pump(1.4)
    _, t_now = window.acq.ring.latest()
    window.plot.arm_selection(t_now, 1.0)
    pump(0.12)
    window.add_selection_as_label()
    pump(0.25)

check("four marks now", len(window.plot.marks) == 4, str(len(window.plot.marks)))
check("four recordings on disk",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == 4,
      str(len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))))
labels_seen = [m["label"] for m in window.plot.marks]
check("labels recorded in order", labels_seen == ["FIST", "OPEN", "REST", "FIST"],
      str(labels_seen))
check("per-gesture colours differ",
      len({T.LABEL_COLORS[l] for l in set(labels_seen)}) == 3,
      str({l: T.LABEL_COLORS[l] for l in set(labels_seen)}))
check("counts shown in the panel",
      "FIST 2" in window.label_counts.text(), window.label_counts.text())

print("\n--- the saved data is usable for training ---")
recs = T.load_dataset(ROOT, stage=T.STAGE_RAW)
check("reloads from disk", len(recs) == 4, str(len(recs)))
check("reloaded data is bit-exact",
      any(np.array_equal(r.emg, rec.emg) for r in recs))
X, y, groups, meta = T.make_windows(recs, window_ms=200, overlap=0.5)
check("windows build from hand-labelled data", X.shape[0] > 0, str(X.shape))
feats, names = T.make_feature_matrix(X, meta["fs"])
check("features finite", np.isfinite(feats).all())

print("\n--- UNDO quarantines, never erases ---")
marks_before = len(window.plot.marks)
rows_before = window.table.rowCount()
on_disk_before = len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))
window.undo_last_label()
pump(0.25)
check("mark removed", len(window.plot.marks) == marks_before - 1)
check("row removed", window.table.rowCount() == rows_before - 1)
check("dropped from the dataset",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) == on_disk_before - 1)
deleted = os.path.join(window.writer.session_dir, "_deleted")
check("files moved to _deleted, not erased",
      os.path.isdir(deleted) and len(os.listdir(deleted)) >= 1,
      str(len(os.listdir(deleted))) if os.path.isdir(deleted) else "missing")

print("\n--- label mode and the cued session do not mix ---")
window.label_mode_button.setChecked(False)
pump(0.15)
check("label mode off clears the selection",
      not window.plot.label_mode and window.plot.selection_bounds() is None)

window.reps_spin.setValue(1)
window._refresh_plan_info()
window.start_session()
pump(0.2)
window.label_mode_button.setChecked(True)
pump(0.2)
check("label mode refused during a cued session",
      not window.plot.label_mode and not window.label_mode_button.isChecked())
window.stop_session()
pump(0.2)

print("\n--- shutdown ---")
arm.stop()
window.close()
pump(0.4)
check("acquisition stopped", not window.acq.is_alive())

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

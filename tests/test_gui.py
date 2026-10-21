"""
Offscreen GUI test: builds the real MainWindow, feeds it synthetic EMG through
the real acquisition sample path, runs a compressed automatic session, and
exercises trim / graph export / delete / clear / dataset export.

No armband needed. MYO_PORT points at a port that does not exist, so the
reconnect loop is exercised at the same time.
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


# ---------------------------------------------------------------- configuration
ROOT = tempfile.mkdtemp(prefix="emggui_")
T.DATA_FOLDER = ROOT
T.MYO_PORT = "COM254"                 # deliberately absent -> reconnect path
T.GESTURES = ["REST", "FIST", "OPEN"]
T.REPETITIONS = 2
T.GESTURE_ORDER = "blocked"
T.REST_TIME = 0.05
T.PREPARE_TIME = 0.05
T.COUNTDOWN_TIME = 0.30
T.RECORD_TIME = 0.30
T.PAUSE_TIME = 0.05
T.PRE_GESTURE_MS = 160.0
T.POST_GESTURE_MS = 160.0
T.RECONNECT_DELAY_S = 0.3
T.RECONNECT_MAX_DELAY_S = 0.5
# Point at a model that does not exist on purpose: this suite verifies the
# panel stays honest with NO model, regardless of whether the person running
# the tests happens to have trained one. The with-model path is covered by
# tests/test_predict.py.
T.MODEL_PATH = os.path.join(ROOT, "no_such_model.joblib")
T.QC_PAUSE_ON = "never"               # no modal dialogs mid-session
T.PAUSE_SESSION_ON_LINK_LOSS = False  # the fake arm supplies the samples
T.REVIEW_EACH_RECORDING = False      # this suite tests the hands-off path
T.AUTO_TRIM = True
T.AUTO_TRIM_WRITE_FILE = True
T.GUI_FPS = 60

from PySide6 import QtCore, QtWidgets          # noqa: E402

# Silence modal dialogs so the test can run unattended.
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


# ------------------------------------------------------------------ fake armband
class FakeArm:
    """
    Pushes synthetic samples through MyoAcquisition._ingest -- the exact same
    entry point the real Myo callback uses, so ring buffer, capture, rate
    estimation and dropped-packet counting are all really exercised.
    """

    def __init__(self, acq, fs=50.0):
        self.acq = acq
        self.fs = fs
        self.active = 0.0
        self.running = True
        self.rng = np.random.default_rng(7)
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
            base = np.abs(self.rng.normal(0, 7, 8)) + 6
            if self.active:
                base += self.rng.random(8) * self.active
            self.acq._ingest(np.clip(base, 0, 1023).astype(int))


print("\n--- window construction ---")
window = T.MainWindow()
#  Read before pumping: the compute probe is scheduled 400 ms after startup,
#  and until it answers the badge must claim neither GPU nor CPU.
check("compute badge starts as unknown, not as CPU and not as GPU",
      "checking" in window.compute_label.text().lower(),
      window.compute_label.text())
check("status bar chip starts as unknown too",
      "?" in window.status_compute.text(), window.status_compute.text())
window.show()
pump(0.4)
check("window created", window.isVisible())
check("plot has 8 channels", len(window.plot.plots) == 8)
check("acquisition thread alive", window.acq.is_alive())

arm = FakeArm(window.acq)
arm.start()
pump(0.8)

stats = window.acq.stats()
check("samples flowing", stats["samples"] > 10, str(stats["samples"]))
check("rate estimated near 50 Hz", 30 < stats["rate"] < 90, f"{stats['rate']:.1f}")
check("ring buffer bounded", window.acq.ring.total > 0)

print("\n--- reconnect behaviour (no dongle on COM254) ---")
check("connection state is not CONNECTED",
      window.acq.state in (T.CONN_RECONNECTING, T.CONN_CONNECTING, T.CONN_ERROR),
      window.acq.state)
pump(1.2)
check("reconnect attempts counted", window.acq.stats()["reconnects"] >= 1,
      str(window.acq.stats()["reconnects"]))
check("GUI still responsive after link failure", window.isVisible())
check("log recorded the failure", "error" in window.log_box.toPlainText().lower())

print("\n--- graph controls ---")
window.window_spin.setValue(2.0)
pump(0.1)
check("time window applied", abs(window.plot.window_s - 2.0) < 1e-6)

window.autoscale_check.setChecked(False)
window.scale_slider.setValue(600)
pump(0.1)
check("manual scale applied",
      not window.plot.autoscale and window.plot.manual_scale == 600.0)
window.autoscale_check.setChecked(True)

window.envelope_check.setChecked(True)
pump(0.2)
check("envelope enabled", window.plot.show_envelope
      and window.plot.env_curves[0].isVisible())
window.envelope_check.setChecked(False)

window.channel_checks[3].setChecked(False)
pump(0.1)
check("channel hidden", not window.plot.plots[3].isVisible())
window.channel_checks[3].setChecked(True)
pump(0.1)
check("channel shown again", window.plot.plots[3].isVisible())

window.freeze_button.setChecked(True)
pump(0.1)
check("graph frozen", window.plot.frozen)
window.freeze_button.setChecked(False)
window.plot.reset_view()
check("reset view did not raise", True)

last, mins, maxs, rms = window.plot.channel_stats()
check("channel readouts populated", float(maxs.max()) > 0, f"max={maxs.max():.0f}")

print("\n--- automatic recording session ---")
window.person_edit.setText("person_07")
window.session_edit.setText("session_03")
window._on_identity_changed()
check("subject applied", (window.person_id, window.session_id)
      == ("person_07", "session_03"))

window.reps_spin.setValue(2)
window.order_combo.setCurrentText("blocked")
window._refresh_plan_info()

# ---- the PLAN panel's new controls -----------------------------------------
check("the hold-time spin exists", hasattr(window, "hold_spin"))
check("it starts at the configured hold time",
      abs(window.hold_spin.value() - T.RECORD_TIME) < 1e-9,
      str(window.hold_spin.value()))
window.hold_spin.setValue(2.0)
window._refresh_plan_info()
check("editing it mirrors into the config, so the stored metadata matches",
      abs(T.RECORD_TIME - 2.0) < 1e-9, str(T.RECORD_TIME))
window.hold_spin.setValue(2.5)
window._refresh_plan_info()

check("all three orders are offered",
      {window.order_combo.itemText(i)
       for i in range(window.order_combo.count())}
      == {"interleaved", "blocked", "random"})
tip = window.order_combo.toolTip()
check("the tooltip explains BLOCKED, not only the default",
      "blocked" in tip.lower() and "back to back" in tip.lower())
check("...and says why interleaved is the honest one",
      "drift" in tip.lower() or "fatigue" in tip.lower())

# ---- the gesture set -------------------------------------------------------
check("the GESTURES button exists", hasattr(window, "gestures_button"))
dialog = T.GestureSetupDialog(window, selected=list(T.GESTURES))
check("the dialog offers the optional gestures as well as the ones in use",
      set(dialog.boxes) >= set(T.GESTURES) | set(T.OPTIONAL_GESTURES),
      ", ".join(dialog.boxes))
check("only the in-use ones start checked",
      dialog.selected() == list(T.GESTURES), ", ".join(dialog.selected()))
dialog.boxes["PINCH"].setChecked(True)
check("ticking an optional one adds it", "PINCH" in dialog.selected())
dialog.custom_edit.setText("wrist flex")
dialog._add_custom()
check("a custom name is accepted and slugged to upper case",
      "WRIST_FLEX" in dialog.selected(), ", ".join(dialog.selected()[-2:]))
for name in list(dialog.boxes):
    dialog.boxes[name].setChecked(name == "FIST")
dialog._accept()
check("fewer than two gestures is refused, with a reason",
      dialog.isVisible() is False or "at least two" in dialog.warning.text(),
      dialog.warning.text()[:60])
dialog.deleteLater()

#  Applying a new set has to reach every combo that hands out a label, or
#  they would keep offering labels the plan no longer contains.
before_gestures = list(T.GESTURES)
window._set_global("GESTURES", before_gestures + ["PINCH"])
for combo in (window.gesture_combo, window.label_gesture_combo,
              window.take_gesture_combo):
    combo.clear()
    combo.addItems(T.GESTURES)
window._refresh_plan_info()
check("the manual-gesture combo follows the label set",
      "PINCH" in [window.gesture_combo.itemText(i)
                  for i in range(window.gesture_combo.count())])
#  Counted from the set actually in force -- an earlier section of this
#  suite narrows GESTURES, so a hardcoded number here would be asserting
#  the test order rather than the behaviour.
check("the plan line counts the new gesture",
      "%d gestures" % len(T.GESTURES) in window.plan_label.text(),
      window.plan_label.text()[:50])
window._set_global("GESTURES", before_gestures)
for combo in (window.gesture_combo, window.label_gesture_combo,
              window.take_gesture_combo):
    combo.clear()
    combo.addItems(T.GESTURES)
window._refresh_plan_info()
check("and switching it back off restores the set",
      list(T.GESTURES) == before_gestures)

seen_states = set()


def watch():
    seen_states.add(window.sequence.state)
    # Drive the fake muscle in step with the state machine.
    arm.active = 500.0 if window.sequence.state == T.STATE_RECORDING else 0.0


watcher = QtCore.QTimer()
watcher.timeout.connect(watch)
watcher.start(10)

window.start_session()
check("session started", window.sequence.running)
expected = len(T.build_plan())
check("plan built", expected == 6, str(expected))

deadline = time.perf_counter() + 60
while window.sequence.running and time.perf_counter() < deadline:
    pump(0.05)
watcher.stop()

check("session reached DONE", window.sequence.state == T.STATE_DONE,
      window.sequence.state)
check("all repetitions recorded", len(window.recordings) == expected,
      f"{len(window.recordings)}/{expected}")
for state in (T.STATE_REST, T.STATE_PREPARE, T.STATE_COUNTDOWN,
              T.STATE_RECORDING, T.STATE_STOPPING, T.STATE_SAVED):
    check(f"banner state {state} occurred", state in seen_states)
check("table rows match recordings",
      window.table.rowCount() == len(window.recordings))

print("\n--- saved files ---")
rec = window.recordings[-1]
check("recording has samples", rec.n_samples > 5, str(rec.n_samples))
check("label window inside the capture",
      0 < rec.label_start_index < rec.label_end_index <= rec.n_samples,
      f"{rec.label_start_index}..{rec.label_end_index}/{rec.n_samples}")
check("pre-roll captured before the cue", rec.label_start_index >= 1,
      str(rec.label_start_index))

raw_csv = [v for k, v in rec.files.items() if k == "raw:csv"]
check("raw csv path recorded", bool(raw_csv), str(list(rec.files)))
if raw_csv:
    full = os.path.join(window.writer.root, raw_csv[0])
    check("raw csv exists on disk", os.path.exists(full))

found_raw = T.discover_recordings(ROOT, stage=T.STAGE_RAW)
found_trim = T.discover_recordings(ROOT, stage=T.STAGE_TRIMMED)
check("raw files discovered", len(found_raw) == expected,
      f"{len(found_raw)}/{expected}")
check("auto-trim wrote trimmed files", len(found_trim) >= 1, str(len(found_trim)))
check("session_config.json written",
      os.path.exists(os.path.join(window.writer.session_dir, "session_config.json")))
check("manifest written", len(window.writer.read_manifest()) >= expected,
      str(len(window.writer.read_manifest())))

folders = sorted(d for d in os.listdir(window.writer.session_dir)
                 if os.path.isdir(os.path.join(window.writer.session_dir, d)))
check("one folder per gesture", set(folders) >= {"FIST", "OPEN", "REST"}, str(folders))

print("\n--- manual recording ---")
window.gesture_combo.setCurrentText("FIST")
arm.active = 500.0
window.record_button.setChecked(True)
window.toggle_manual_record(True)
pump(0.6)
check("manual recording active", window.sequence.state == T.STATE_RECORDING)
window.record_button.setChecked(False)
window.toggle_manual_record(False)
arm.active = 0.0
pump(0.3)
check("manual recording stored", len(window.recordings) == expected + 1,
      str(len(window.recordings)))

print("\n--- trim dialog ---")
target = next((r for r in window.recordings if r.label == "FIST"), None)
dialog = T.TrimDialog(target, window.writer)
check("trim dialog built", dialog.rec is target)
check("trim dialog has 8 channel plots + envelope", len(dialog.plots) == 8)
before = dialog.start_spin.value(), dialog.end_spin.value()
dialog.apply_full()
check("full range selects everything",
      dialog.selection_indices() == (0, target.n_samples),
      str(dialog.selection_indices()))
span = float(dialog.t[-1])
dialog._set_selection(span * 0.2, span * 0.6)
start, end = dialog.selection_indices()
check("manual selection narrowed", 0 < start < end < target.n_samples,
      f"{start}..{end}")
raw_before = target.emg.copy()
n_trim_before = len(T.discover_recordings(ROOT, stage=T.STAGE_TRIMMED))
dialog.save_trimmed()
check("trimmed copy produced", dialog.result_recording is not None)
if dialog.result_recording is not None:
    check("trimmed is shorter",
          dialog.result_recording.n_samples == end - start,
          f"{dialog.result_recording.n_samples} vs {end - start}")
check("ORIGINAL RAW UNCHANGED in memory", np.array_equal(target.emg, raw_before))
check("new trimmed file written",
      len(T.discover_recordings(ROOT, stage=T.STAGE_TRIMMED)) == n_trim_before + 1)
dialog.apply_auto()
check("auto-detect in dialog did not raise", True)
dialog.preview_selection()
dialog.close()

print("\n--- graph export ---")
png = os.path.join(ROOT, "graph_test.png")
svg = os.path.join(ROOT, "graph_test.svg")
try:
    png_out = window.plot.export_image(png)
    check("PNG exported", os.path.exists(png_out)
          and os.path.getsize(png_out) > 2000, f"{os.path.getsize(png_out)} bytes")
except Exception as exc:
    check("PNG exported", False, str(exc))
try:
    svg_out = window.plot.export_image(svg)
    check("SVG exported", os.path.exists(svg_out)
          and os.path.getsize(svg_out) > 1000, f"{os.path.getsize(svg_out)} bytes")
except Exception as exc:
    check("SVG exported", False, str(exc))

window._update_plot_title()
title = window.plot.title_item.text
check("graph caption carries person/session/gesture",
      "person_07" in str(title) and "session_03" in str(title), str(title)[:90])

print("\n--- delete last / clear ---")
count_before = len(T.discover_recordings(ROOT, stage=T.STAGE_RAW))
n_rows = window.table.rowCount()
window.delete_last()
pump(0.1)
check("row removed from the list", window.table.rowCount() == n_rows - 1)
deleted_dir = os.path.join(window.writer.session_dir, "_deleted")
check("files moved to _deleted, not erased",
      os.path.isdir(deleted_dir) and len(os.listdir(deleted_dir)) >= 1,
      str(os.listdir(deleted_dir)[:3]) if os.path.isdir(deleted_dir) else "missing")
check("dataset no longer lists the deleted recording",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) < count_before)

print("\n--- auto-trim all ---")
window.auto_trim_all()
pump(0.2)
check("auto-trim all completed", True)

print("\n--- dataset export ---")
window.export_dataset()
deadline = time.perf_counter() + 40
while window.export_worker is not None and time.perf_counter() < deadline:
    pump(0.05)
exports = os.path.join(T.PROJECT_DIR, "exports")
bundles = ([f for f in os.listdir(exports) if f.endswith(".npz")]
           if os.path.isdir(exports) else [])
check("export bundle written", bool(bundles), str(bundles[:2]))
if bundles:
    newest = max((os.path.join(exports, f) for f in bundles), key=os.path.getmtime)
    with np.load(newest, allow_pickle=False) as data:
        check("bundle has X/y/groups",
              {"X", "y", "groups", "label_names", "features"} <= set(data.files),
              str(sorted(data.files)))
        check("bundle windows non-empty", data["X"].shape[0] > 0, str(data["X"].shape))
        check("bundle groups are person ids",
              "person_07" in set(np.asarray(data["groups"]).tolist()))
    check("export index csv written",
          os.path.exists(os.path.splitext(newest)[0] + "_index.csv"))
    check("export summary json written",
          os.path.exists(os.path.splitext(newest)[0] + "_summary.json"))
    os.remove(newest)
    for suffix in ("_index.csv", "_summary.json"):
        extra = os.path.splitext(newest)[0] + suffix
        if os.path.exists(extra):
            os.remove(extra)

window.clear_list()
pump(0.1)
check("list cleared", window.table.rowCount() == 0 and not window.recordings)
check("files on disk survive CLEAR",
      len(T.discover_recordings(ROOT, stage=T.STAGE_RAW)) > 0)

print("\n--- prediction panel is honest with no model ---")
check("no model loaded", not window.predictor.ready, window.predictor.status)
check("panel shows no fake prediction",
      window.prediction_label.text() == "--", window.prediction_label.text())
check("no fake confidence either",
      window.prediction_conf.text() == "", window.prediction_conf.text())
check("panel explains why",
      "model" in window.prediction_note.text().lower(),
      window.prediction_note.text()[:70])
check("PREDICT button disabled", not window.predict_button.isEnabled())
check("no probability bars without a model", not window.prob_bars,
      str(sorted(window.prob_bars)))
window.toggle_prediction()
pump(0.2)
check("pressing PREDICT does nothing without a model",
      not window.prediction_active and not window.one_shot_deadline
      and window.predictor.last is None)

print("\n--- the GPU/CPU badge ---")
#  It must never claim a device it has not confirmed, so before the probe
#  answers it reads as "checking", not as either answer.
check("badge exists", window.compute_label is not None)
check("the startup probe has answered by now",
      window.compute_info.get("probed") is True, str(window.compute_info))

#  Run it again explicitly, to check the whole path rather than just its
#  result: a probe is started, it is a subprocess, and it reports back.
window.start_compute_probe()
check("the probe runs as a subprocess", window.compute_process is not None)
check("the recorder itself did not import torch", "torch" not in sys.modules)
deadline = time.perf_counter() + 60
while window.compute_process is not None and time.perf_counter() < deadline:
    pump(0.2)
probed = window.compute_process is None
check("the probe answers within a minute", probed,
      str(window.compute_info))
#  A stale timeout must not kill a LATER probe. singleShot cannot be
#  cancelled, so a probe that answered in 8 s still has a timer due at 30 s;
#  without an identity check it killed whichever probe was running when that
#  landed and reported "did not answer" on a machine with a working GPU.
#  Simulated directly rather than by waiting 30 s.
good = dict(window.compute_info)
window.start_compute_probe()
live = window.compute_process
window._compute_probe_timeout(object())          # a timer from an old probe
check("a stale probe timeout does not kill the current probe",
      window.compute_process is live, str(window.compute_info.get("reason")))
deadline = time.perf_counter() + 60
while window.compute_process is not None and time.perf_counter() < deadline:
    pump(0.2)
check("...and that probe still answers for itself",
      window.compute_info.get("probed") is True,
      str(window.compute_info.get("reason")))
check("the probe's OWN timeout still fires when it is the current one",
      (window.start_compute_probe(),
       window._compute_probe_timeout(window.compute_process),
       window.compute_info.get("failed") is True)[2],
      str(window.compute_info.get("reason")))
deadline = time.perf_counter() + 30
while window.compute_process is not None and time.perf_counter() < deadline:
    pump(0.2)
window.start_compute_probe()
deadline = time.perf_counter() + 60
while window.compute_process is not None and time.perf_counter() < deadline:
    pump(0.2)
probed = window.compute_process is None

if probed:
    expected = T.detect_compute_device()
    check("the badge agrees with a direct probe",
          window.compute_info.get("cuda") == expected.get("cuda"),
          f"{window.compute_info.get('cuda')} vs {expected.get('cuda')}")
    text = window.compute_label.text()
    check("the badge says GPU or CPU, in words",
          "GPU READY" in text or "CPU ONLY" in text, text)
    check("the badge names the GPU when there is one",
          ("GPU READY" in text) == bool(expected.get("cuda")), text)
    check("the badge has the explanatory tooltip",
          "CPU" in window.compute_label.toolTip(),
          " ".join(window.compute_label.toolTip().split())[:70])
    #  The panel folds away; the status bar does not.
    chip = window.status_compute.text()
    check("the status bar chip agrees with the badge",
          ("GPU" in chip) == bool(expected.get("cuda")), chip)
    check("the status bar chip is no longer unknown", "?" not in chip, chip)
    check("the status bar chip explains itself on hover",
          "CPU" in window.status_compute.toolTip(),
          " ".join(window.status_compute.toolTip().split())[:70])
    check("the device went into the log",
          "Compute device" in window.log_box.toPlainText(),
          window.log_box.toPlainText()[-70:])
check("the recorder still has not imported torch", "torch" not in sys.modules)

print("\n--- shutdown ---")
arm.stop()
window.close()
pump(0.5)
check("acquisition thread stopped", not window.acq.is_alive())
check("the probe process is cleaned up on close",
      window.compute_process is None)

shutil.rmtree(ROOT, ignore_errors=True)
if os.path.isdir(os.path.join(T.PROJECT_DIR, "exports")) and \
        not os.listdir(os.path.join(T.PROJECT_DIR, "exports")):
    os.rmdir(os.path.join(T.PROJECT_DIR, "exports"))

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

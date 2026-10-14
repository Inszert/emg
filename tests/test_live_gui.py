"""Full application against the real armband: a short real automatic session."""
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


ROOT = tempfile.mkdtemp(prefix="emglivegui_")
T.DATA_FOLDER = ROOT
T.GESTURES = ["REST", "FIST"]
T.REPETITIONS = 2
T.GESTURE_ORDER = "blocked"
T.REST_TIME = 1.0
T.PREPARE_TIME = 0.5
T.COUNTDOWN_TIME = 2.0
T.RECORD_TIME = 1.5
T.PAUSE_TIME = 0.5
T.QC_PAUSE_ON = "never"
T.REVIEW_EACH_RECORDING = False      # this suite tests the hands-off path
T.AUTO_TRIM = True
T.AUTO_TRIM_WRITE_FILE = True

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


print("\n--- launching the real application ---")
window = T.MainWindow()
window.show()

deadline = time.perf_counter() + 45
while not window.acq.connected and time.perf_counter() < deadline:
    pump(0.1)

check("GUI connected to the real Myo", window.acq.connected,
      f"{window.acq.state}: {window.acq.state_message}")
if not window.acq.connected:
    window.close()
    shutil.rmtree(ROOT, ignore_errors=True)
    sys.exit(1)

check("connection pill shows CONNECTED", "CONNECTED" in window.conn_pill.text(),
      window.conn_pill.text())

pump(2.0)
stats = window.acq.stats()
check("live samples reaching the GUI", stats["samples"] > 40, str(stats["samples"]))
check("GUI rate readout plausible", 20 < stats["rate"] < 260,
      f"{stats['rate']:.1f} Hz")

last, mins, maxs, rms = window.plot.channel_stats()
check("plot rendered real data", float(maxs.max()) > 0,
      f"max={maxs.max():.0f} rms={rms.round(1).tolist()}")
check("status bar updated", "Hz" in window.status_rate.text(),
      window.status_rate.text())

print("\n--- real automatic session (REST x2, FIST x2) ---")
window.person_edit.setText("live_gui")
window.session_edit.setText("session_01")
window._on_identity_changed()
window.reps_spin.setValue(2)
window._refresh_plan_info()

window.start_session()
check("session running", window.sequence.running)

deadline = time.perf_counter() + 90
while window.sequence.running and time.perf_counter() < deadline:
    pump(0.05)

check("session completed", window.sequence.state == T.STATE_DONE,
      window.sequence.state)
check("4 recordings collected", len(window.recordings) == 4,
      str(len(window.recordings)))

if window.recordings:
    durations = [r.duration for r in window.recordings]
    rates = [r.fs for r in window.recordings]
    samples = [r.n_samples for r in window.recordings]
    print(f"  samples:   {samples}")
    print(f"  durations: {[round(d, 2) for d in durations]}")
    print(f"  rates:     {[round(r, 1) for r in rates]}")

    check("every recording has real samples", all(s > 30 for s in samples),
          str(samples))
    # A recording cut short by a real link dropout is legitimately shorter --
    # it must be FLAGGED rather than silently accepted at full length.
    short_ok = [
        (1.6 < r.duration < 3.0) or any(i.code in ("missing_samples", "gap",
                                                   "disconnect")
                                        for i in r.issues)
        for r in window.recordings
    ]
    check("durations cover record + pre/post roll, or are flagged",
          all(short_ok),
          str([(round(r.duration, 2), [i.code for i in r.issues])
               for r in window.recordings]))
    check("measured rates near 50 Hz", all(35 < r < 70 for r in rates),
          str([round(r, 1) for r in rates]))
    check("label windows inside the captures",
          all(0 < r.label_start_index < r.label_end_index <= r.n_samples
              for r in window.recordings))
    check("no quality errors",
          not any(r.worst_issue_level == "error" for r in window.recordings),
          str([[i.code for i in r.issues] for r in window.recordings]))
    check("timestamps strictly increasing in every recording",
          all(bool(np.all(np.diff(r.timestamps) > 0)) for r in window.recordings))

found = T.discover_recordings(ROOT, stage=T.STAGE_RAW)
check("raw files on disk", len(found) == 4, str(len(found)))
folders = sorted(d for d in os.listdir(window.writer.session_dir)
                 if os.path.isdir(os.path.join(window.writer.session_dir, d)))
check("gesture folders created", set(folders) >= {"FIST", "REST"}, str(folders))
check("manifest complete", len(window.writer.read_manifest()) >= 4,
      str(len(window.writer.read_manifest())))

reloaded = T.load_dataset(ROOT, stage=T.STAGE_RAW)
check("reloaded from disk", len(reloaded) == 4, str(len(reloaded)))
if reloaded:
    original = {r.uid: r for r in window.recordings}
    exact = all(np.array_equal(r.emg, original[r.uid].emg)
                for r in reloaded if r.uid in original)
    check("reloaded data is bit-exact", exact)

print("\n--- graph export from the live app ---")
png = window.plot.export_image(os.path.join(ROOT, "live.png"))
check("PNG exported", os.path.getsize(png) > 2000, f"{os.path.getsize(png)} bytes")
svg = window.plot.export_image(os.path.join(ROOT, "live.svg"))
check("SVG exported", os.path.getsize(svg) > 1000, f"{os.path.getsize(svg)} bytes")

print("\n--- trim a real recording ---")
fist = next((r for r in window.recordings if r.label == "FIST"), None)
if fist is not None:
    dialog = T.TrimDialog(fist, window.writer)
    raw_before = fist.emg.copy()
    span = float(dialog.t[-1])
    dialog._set_selection(span * 0.25, span * 0.75)
    dialog.save_trimmed()
    check("trim produced a new recording", dialog.result_recording is not None)
    check("RAW UNCHANGED after trimming", np.array_equal(fist.emg, raw_before))
    dialog.close()

print("\n--- shutdown ---")
window.close()
pump(0.6)
check("acquisition stopped", not window.acq.is_alive())

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

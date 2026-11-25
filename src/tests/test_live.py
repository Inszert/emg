"""
LIVE hardware test against the real Myo armband on COM3.

Exercises the production code paths: MyoAcquisition connect -> stream ->
capture -> Recording -> DatasetWriter -> reload, plus a forced reconnect.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import emg_dataset_tool as T

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


T.RECONNECT_DELAY_S = 1.0
T.RECONNECT_MAX_DELAY_S = 2.0
T.SCAN_TIMEOUT_S = 10.0          # retry a dead scan quickly during the test
T.VIBRATE_CUES = True

states = []
acq = T.MyoAcquisition(on_state=lambda s, m: states.append((s, m)))

print("\n--- connecting to the real armband ---")
acq.start()

deadline = time.perf_counter() + 45
while not acq.connected and time.perf_counter() < deadline:
    time.sleep(0.2)

check("connected to the Myo", acq.connected, f"{acq.state}: {acq.state_message}")
if not acq.connected:
    print("\nCould not connect. Is the armband awake and the dongle plugged in?")
    acq.stop()
    sys.exit(1)

print("\n--- streaming ---")
start_samples = acq.stats()["samples"]
time.sleep(6.0)
stats = acq.stats()
got = stats["samples"] - start_samples
measured = stats["rate"]

check("samples arriving", got > 50, f"{got} samples in 6 s")
check("measured rate is plausible", 20 < measured < 260, f"{measured:.1f} Hz")
check("no gross packet loss", stats["dropped"] < got * 0.2,
      f"dropped {stats['dropped']} of {got}")

emg, ts = acq.ring.snapshot(200)
check("ring buffer filled", emg.shape[0] > 50, str(emg.shape))
check("8 channels", emg.shape[1] == 8)
lo, hi = T.emg_range()
check("values inside the device range",
      bool((emg >= lo).all() and (emg <= hi).all()),
      f"min={emg.min()} max={emg.max()} expected [{lo},{hi}]")
check("timestamps monotonic", bool(np.all(np.diff(ts) > 0)))
check("channels are not all identical",
      len(set(emg.std(axis=0).round(3).tolist())) > 1,
      str(emg.std(axis=0).round(1).tolist()))

print(f"\n  nominal rate for mode '{T.EMG_MODE}': {T.nominal_rate():.0f} Hz")
print(f"  MEASURED rate: {measured:.1f} Hz")
print(f"  per-channel mean: {emg.mean(axis=0).round(1).tolist()}")
print(f"  per-channel std : {emg.std(axis=0).round(1).tolist()}")

print("\n--- haptic cue ---")
acq.vibrate(1)
check("vibrate() did not raise", True)

print("\n--- real recording through the capture path ---")
print("  (flex your arm now if you want a visible burst)")
capture = acq.start_capture("FIST", 1, meta={"emg_mode": T.EMG_MODE,
                                             "notes": "live hardware test"})
check("capture is active", acq.capturing)
time.sleep(1.0)
capture.mark_label_start()
time.sleep(2.0)
capture.mark_label_end()
time.sleep(0.5)
returned = acq.stop_capture()

check("stop_capture returned the capture", returned is capture)
check("capture no longer active", not acq.capturing)
check("capture collected samples", len(capture) > 50, str(len(capture)))
check("no overflow", not capture.overflowed)
check("no disconnect during the capture", not capture.had_disconnect)

rec = capture.to_recording("live_test", "session_01")
check("Recording built", rec.n_samples == len(capture), str(rec.n_samples))
check("label window marked",
      0 < rec.label_start_index < rec.label_end_index <= rec.n_samples,
      f"{rec.label_start_index}..{rec.label_end_index}/{rec.n_samples}")
check("measured fs on the recording", 20 < rec.fs < 260, f"{rec.fs:.1f} Hz")
check("duration about 3.5 s", 2.8 < rec.duration < 4.5, f"{rec.duration:.2f} s")

issues = rec.run_quality_checks(expected_seconds=3.5)
print("  quality findings: " + (", ".join(f"{i.level}:{i.code}" for i in issues)
                                or "none"))
check("no quality ERROR on live data",
      not any(i.level == "error" for i in issues),
      ";".join(i.code for i in issues if i.level == "error"))

activity = rec.detect_activity()
print(f"  activity detection: found={activity.found} "
      f"{activity.start}..{activity.end} baseline={activity.baseline:.1f} "
      f"high={activity.threshold_high:.1f}")

print("\n--- saving and reloading real data ---")
root = tempfile.mkdtemp(prefix="emglive_")
try:
    writer = T.DatasetWriter(root, "live_test", "session_01")
    writer.write_session_config()
    written = writer.save(rec, T.STAGE_RAW)
    check("csv written", os.path.exists(written["csv"]))
    check("npz written", os.path.exists(written["npz"]))

    back = T.load_recording(written["npz"])
    check("round trip is bit-exact", np.array_equal(back.emg, rec.emg))
    back_csv = T.load_recording(written["csv"])
    check("csv round trip is bit-exact", np.array_equal(back_csv.emg, rec.emg))

    trimmed = rec.auto_trimmed()
    if trimmed is not None:
        writer.save(trimmed, T.STAGE_TRIMMED)
        check("trimmed copy saved alongside raw",
              len(T.discover_recordings(root, stage=T.STAGE_TRIMMED)) == 1)
        check("raw still present",
              len(T.discover_recordings(root, stage=T.STAGE_RAW)) == 1)
    else:
        print("  (no activity detected -- arm was probably relaxed; "
              "not counted as a failure)")

    recs = T.load_dataset(root, stage="prefer_trimmed")
    X, y, groups, meta = T.make_windows(recs, fs=rec.fs)
    check("windows built from live data", X.shape[0] > 0, str(X.shape))
    feats, names = T.make_feature_matrix(X, meta["fs"])
    check("features finite on live data", np.isfinite(feats).all())
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\n--- forced reconnect (closing the serial port underneath) ---")
before = acq.stats()["reconnects"]
with acq._myo_lock:
    myo = acq._myo
try:
    myo.bt.ser.close()          # simulate the dongle being yanked
except Exception as exc:
    print(f"  (close raised {exc})")

# Generous window: a Myo that has dozed off needs a scan timeout and at least
# one more attempt, and it only starts advertising again once it is moved.
print("  (keep the armband moving so it stays awake)")
deadline = time.perf_counter() + 75
saw_reconnecting = False
while time.perf_counter() < deadline:
    if acq.state == T.CONN_RECONNECTING:
        saw_reconnecting = True
    if saw_reconnecting and acq.connected:
        break
    time.sleep(0.2)

check("entered RECONNECTING", saw_reconnecting, acq.state)
check("reconnect counter advanced", acq.stats()["reconnects"] > before,
      f"{before} -> {acq.stats()['reconnects']}")
check("recovered to CONNECTED", acq.connected, f"{acq.state}: {acq.state_message}")
check("a stuck scan never hangs silently",
      acq.connected or "asleep" in acq.state_message,
      acq.state_message)

if acq.connected:
    n0 = acq.stats()["samples"]
    time.sleep(3.0)
    check("streaming again after reconnect", acq.stats()["samples"] - n0 > 20,
          f"{acq.stats()['samples'] - n0} samples")

print("\n--- shutdown ---")
acq.stop()
acq.join(timeout=6.0)
check("thread stopped cleanly", not acq.is_alive())

print("\nstate transitions seen:")
for state, message in states:
    print(f"  {state:<14} {message}")

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

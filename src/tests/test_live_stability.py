"""
LINK STABILITY test against the real armband.

Streams for a while and measures how well the connection actually holds, so
"it keeps disconnecting" becomes a number instead of an impression.

    python tests\test_live_stability.py            # 120 s
    python tests\test_live_stability.py 300        # 5 minutes
    python tests\test_live_stability.py 120 --baseline   # OLD settings

Wear the armband and use it normally while this runs. `--baseline` reverts to
the original settings (sleep enabled, 640 ms supervision timeout, scan on
reconnect) so you can compare the two directly.
"""
import sys
import time
from collections import Counter

import numpy as np

import emg_dataset_tool as T

DURATION = 120.0
BASELINE = "--baseline" in sys.argv
for arg in sys.argv[1:]:
    try:
        DURATION = float(arg)
        break
    except ValueError:
        continue

if BASELINE:
    print(">>> BASELINE MODE: original settings, for comparison")
    T.MYO_NEVER_SLEEP = False
    T.MYO_CONN_TIMEOUT_MS = 640
    T.MYO_FAST_RECONNECT = False
    T.STALL_TIMEOUT_S = 3.0

print(f"\nsettings: never_sleep={T.MYO_NEVER_SLEEP}  "
      f"supervision={T.MYO_CONN_TIMEOUT_MS} ms  "
      f"fast_reconnect={T.MYO_FAST_RECONNECT}  "
      f"stall_timeout={T.STALL_TIMEOUT_S} s")

states = []
logs = []
acq = T.MyoAcquisition(
    on_state=lambda s, m: states.append((time.time(), s, m)),
    on_log=lambda text: logs.append((time.time(), text)),
)
acq.start()

print("\nconnecting...")
deadline = time.perf_counter() + 60
while not acq.connected and time.perf_counter() < deadline:
    time.sleep(0.2)

if not acq.connected:
    print(f"FAILED to connect: {acq.state} {acq.state_message}")
    acq.stop()
    sys.exit(1)

connect_time = time.perf_counter()
print(f"connected after {connect_time - (deadline - 60):.1f} s")
print(f"\n{'=' * 64}")
print(f"  WEAR THE ARMBAND AND USE IT NORMALLY FOR {DURATION:.0f} SECONDS")
print(f"  Move around, flex, make gestures. Do not take it off.")
print(f"{'=' * 64}\n")

start = time.perf_counter()
samples_start = acq.stats()["samples"]
last_report = start
gap_times = []
last_count = samples_start
last_seen = start

while time.perf_counter() - start < DURATION:
    time.sleep(0.25)
    now = time.perf_counter()
    stats = acq.stats()

    if stats["samples"] > last_count:
        last_count = stats["samples"]
        last_seen = now
    elif now - last_seen > 1.0:
        gap_times.append((now - last_seen, now - start))
        last_seen = now

    if now - last_report >= 15.0:
        last_report = now
        elapsed = now - start
        print(f"  {elapsed:5.0f} s   {stats['rate']:5.1f} Hz   "
              f"samples {stats['samples'] - samples_start:>7,}   "
              f"reconnects {stats['reconnects']}   "
              f"state {acq.state}")

elapsed = time.perf_counter() - start
stats = acq.stats()
got = stats["samples"] - samples_start
acq.stop()
acq.join(timeout=6)

# ---------------------------------------------------------------- results
reconnects = sum(1 for _, s, _ in states if s == T.CONN_RECONNECTING)
connects = sum(1 for _, s, _ in states if s == T.CONN_CONNECTED)
disconnected_for = 0.0
last_time = None
last_state = None
for when, state, _ in states:
    if last_state is not None and last_state != T.CONN_CONNECTED:
        disconnected_for += when - last_time
    last_time, last_state = when, state

expected = T.nominal_rate() * elapsed
print(f"\n{'=' * 64}")
print(f"  RESULTS over {elapsed:.0f} s")
print(f"{'=' * 64}")
print(f"  samples received   : {got:,}")
print(f"  effective rate     : {got / elapsed:.1f} Hz")
print(f"  DROPOUTS           : {reconnects}")
print(f"  reconnects logged  : {stats['reconnects']}")
print(f"  time disconnected  : {disconnected_for:.1f} s "
      f"({disconnected_for / elapsed * 100:.1f} % of the run)")
print(f"  uptime             : {100 - disconnected_for / elapsed * 100:.1f} %")
print(f"  gaps over 1 s      : {len(gap_times)}")
if gap_times:
    worst = max(g for g, _ in gap_times)
    print(f"  worst gap          : {worst:.1f} s")
    for g, at in gap_times[:8]:
        print(f"      {g:5.1f} s gap at t+{at:.0f} s")

print("\n  state transitions:")
for when, state, message in states:
    print(f"    t+{when - states[0][0]:6.1f} s  {state:<14} {message[:70]}")

sleep_logs = [t for _, t in logs if "sleep" in t.lower()]
print(f"\n  sleep mode: "
      + (sleep_logs[0] if sleep_logs else "not reported"))

verdict = "GOOD" if reconnects == 0 else "DROPPED" if reconnects < 3 else "BAD"
print(f"\n  VERDICT: {verdict}  "
      + ("link held for the whole run" if reconnects == 0
         else f"{reconnects} dropout(s)"))
print(f"\n  Compare with:  python tests\\test_live_stability.py "
      f"{DURATION:.0f} --baseline\n")
sys.exit(0 if reconnects == 0 else 1)

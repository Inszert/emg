"""
The blocked-port recovery, without hardware.

Covers: recognising a "port is held" error, choosing ONLY this project's
port-holding scripts (never the recorder, never another program), and ending
them through an injected function. The real psutil path is exercised by a
harmless check that a non-matching process is never selected.

Run:  python tests/test_port_guard.py     (no armband, no COM port needed)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from emg_app.myo import port_guard as PG   # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# -- recognising the error ----------------------------------------------------
class WinErr(OSError):
    winerror = 5


check("'Access is denied' is a blocked port",
      PG.is_blocked_port_error(OSError("could not open port 'COM3': "
                                       "PermissionError(13, 'Access is denied')")))
check("winerror 5 is a blocked port", PG.is_blocked_port_error(WinErr("x")))
check("a missing port is NOT a blocked port (no port to clear)",
      not PG.is_blocked_port_error(
          OSError("could not open port 'COM254': FileNotFoundError(2, "
                  "'The system cannot find the file specified.')")))
check("an unrelated error is not treated as blocked",
      not PG.is_blocked_port_error(ValueError("Myo dongle not found!")))

# -- choosing the holders -----------------------------------------------------
MYPID = os.getpid()          # clear_blocked_port() excludes the real own PID
procs = [
    {"pid": 10, "name": "python.exe",
     "cmdline": ["python", r"C:\x\tests\test_myo.py"]},          # ends
    {"pid": 11, "name": "python.exe",
     "cmdline": ["python", r"C:\x\legacy\myo.py"]},              # ends
    {"pid": 12, "name": "python.exe",
     "cmdline": ["python", "emg_dataset_tool.py"]},              # the recorder
    {"pid": 13, "name": "python.exe",
     "cmdline": ["python", r"C:\x\emg_app\myo\myo_raw.py"]},     # not a script
    {"pid": 14, "name": "arduino.exe",
     "cmdline": ["arduino", "test_myo.py"]},                     # not python
    {"pid": MYPID, "name": "python.exe",
     "cmdline": ["python", "tests/test_myo.py"]},                # ourselves
    {"pid": 1000, "name": "python.exe",
     "cmdline": ["python", "tests/test_myo.py"]},                # a stale copy
    {"pid": 15, "name": "python.exe",
     "cmdline": ["python", r"C:\x\tests\test_live_gui.py"]},     # ends
]
chosen = sorted(p["pid"] for p in PG.select_port_holders(procs, MYPID))
check("only this project's port scripts are chosen",
      chosen == [10, 11, 15, 1000], str(chosen))
check("the recorder is never chosen, even when it runs as python",
      12 not in chosen)
check("a process that is not Python is never chosen", 14 not in chosen)
check("the current process is never chosen", MYPID not in chosen)

# -- ending them ----------------------------------------------------------------
ended_pids = []
log = []
got = PG.clear_blocked_port(log=log.append, procs=procs,
                            end=lambda pid: (ended_pids.append(pid), True)[1])
check("clear_blocked_port ends exactly the chosen processes",
      sorted(ended_pids) == [10, 11, 15, 1000], str(ended_pids))
check("and reports each one it ended",
      len(got) == 4 and len(log) == 4 and "holding the port" in log[0])
got_none = PG.clear_blocked_port(log=log.append, procs=[],
                                 end=lambda pid: True)
check("nothing to end -> nothing ended (the caller re-raises)", got_none == [])

# -- the real process list is read without error --------------------------------
real = PG._list_processes()
check("the real process list can be read", isinstance(real, list))
check("the real list never selects this test process",
      all(p["pid"] != os.getpid()
          for p in PG.select_port_holders(real, os.getpid())))

print()
print(f"================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

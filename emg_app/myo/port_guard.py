"""
Clear a dongle COM port that another of THIS PROJECT's processes is holding.

Windows gives a COM port to one process at a time. When a test script or the
legacy demo (`test_myo.py`, `myo.py`, `test_live*.py`) is still running, opening
the port fails with "Access is denied", and the recorder used to retry forever.
`clear_blocked_port()` ends those project processes -- and only those -- so the
recorder can take the port, which is what doing it by hand in a shell did.

What it will NOT touch:
  * any process that is not a Python process running one of
    `PORT_HOLDER_SCRIPTS` (so not another program that uses a COM port);
  * the recorder itself (`emg_dataset_tool.py`), including another copy of it:
    closing the user's own application is not something to do silently;
  * the current process.

Qt-free and importable headless. psutil is imported lazily; without it nothing
is cleared and the caller is told so.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, List, Optional

#  Scripts that open the armband's serial port and are safe to end.
PORT_HOLDER_SCRIPTS = ("test_myo.py", "myo.py", "test_live.py",
                       "test_live_gui.py", "test_live_review.py")


def is_blocked_port_error(exc: BaseException) -> bool:
    """True when opening the serial port failed because someone else holds it."""
    text = str(exc).lower()
    if "access is denied" in text or "permission" in text:
        return True
    return getattr(exc, "winerror", None) == 5 or getattr(exc, "errno", None) == 13


def _is_project_holder(name: str, cmdline: List[str]) -> bool:
    if "python" not in (name or "").lower():
        return False
    joined = " ".join(cmdline or [])
    if "emg_dataset_tool" in joined:
        return False                      # never the recorder itself
    return any(os.path.basename(part) in PORT_HOLDER_SCRIPTS
               for part in (cmdline or []))


def select_port_holders(procs: Iterable[dict], own_pid: int) -> List[dict]:
    """
    From `{"pid", "name", "cmdline"}` records, the ones to end. Pure: tests pass
    their own list.
    """
    out = []
    for p in procs:
        pid = p.get("pid")
        if pid is None or pid == own_pid:
            continue
        if _is_project_holder(p.get("name", ""), p.get("cmdline") or []):
            out.append(p)
    return out


def _list_processes() -> List[dict]:
    try:
        import psutil
    except ImportError:
        return []
    procs = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        info = proc.info
        procs.append({"pid": info.get("pid"), "name": info.get("name") or "",
                      "cmdline": info.get("cmdline") or []})
    return procs


def _end(pid: int) -> bool:
    import psutil
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except psutil.TimeoutExpired:
            proc.kill()
        return True
    except psutil.Error:
        return False


def clear_blocked_port(log: Optional[Callable[[str], None]] = None,
                       procs: Optional[Iterable[dict]] = None,
                       end: Optional[Callable[[int], bool]] = None) -> List[dict]:
    """
    End the project processes that hold a COM port. Returns the records that
    were ended (empty when none was found or psutil is missing). `procs` and
    `end` are injection points for the tests.
    """
    say = log or (lambda _msg: None)
    candidates = select_port_holders(
        _list_processes() if procs is None else procs, os.getpid())
    if not candidates:
        return []
    ender = end or _end
    ended = []
    for p in candidates:
        cmd = " ".join(p.get("cmdline") or [])
        if ender(p["pid"]):
            ended.append(p)
            say(f"Ended process {p['pid']} that was holding the port: {cmd}")
    return ended

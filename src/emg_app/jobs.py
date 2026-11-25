"""
Background training jobs (CLAUDE.md 8m): start, status, cancel, log tail.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from typing import List, Optional

from emg_app import config as cfg


# ------------------------------------------------------------------------------
# Background training jobs  (outlive the window that started them)
# ------------------------------------------------------------------------------
# A QProcess child dies with the window that owns it, so a sweep started from
# the GUI used to stop the moment the GUI closed. Training now runs through
# `train_job.py`, started DETACHED, and the GUI and the job only ever talk
# through files in one folder per job:
#
#     <models>/jobs/<id>/job.json     what to run (written here)
#                       /log.txt      merged trainer output + JOBSTEP lines
#                       /status.json  written by the runner, polled here
#                       /cancel       appears when the GUI asks it to stop
#                       /seen         the GUI has reported the outcome
#
# A window opened later finds a running job with `active_training_job()` and
# reads the log from the start, so its progress panel catches up exactly as if
# it had been open all along.

TRAINING_JOBS_DIRNAME = "jobs"
TRAINING_JOB_SCRIPT = os.path.join("training", "train_job.py")


def training_jobs_dir(models_dir: str) -> str:
    return os.path.join(models_dir, TRAINING_JOBS_DIRNAME)


def pid_alive(pid) -> bool:
    """
    Is that process still running? Never signals it.

    `os.kill(pid, 0)` is the POSIX idiom and on Windows it is a bug: there it
    calls TerminateProcess with exit code 0, i.e. it KILLS the process it was
    asked about. So Windows asks the kernel instead.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259                        # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:                                                    # pragma: no cover
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_job_status(job_dir: str) -> Optional[dict]:
    """status.json, or None while the runner has not written it yet."""
    try:
        with open(os.path.join(job_dir, "status.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def job_is_live(job_dir: str) -> bool:
    """Running according to its status AND its runner still exists. A runner
    killed from Task Manager leaves "running" behind forever otherwise."""
    status = read_job_status(job_dir)
    return bool(status and status.get("state") == "running"
                and pid_alive(status.get("pid")))


def start_training_job(models_dir: str, kind: str, steps: List[dict],
                       cwd: str = None, python: str = None) -> str:
    """
    Write a job and start `train_job.py` on it, detached. Returns the job
    folder. `steps` is `[{"name": "ml"|"nn", "argv": [...]}, ...]`, run in
    order, stopping at the first that fails.

    Detached means: no console, its own process group, and -- where the
    parent is inside a Windows job object, as it is under some terminals --
    broken away from that job, because a job object set to kill-on-close
    would otherwise take the trainer down with the window anyway.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = os.path.join(training_jobs_dir(models_dir),
                           f"{stamp}_{kind}_{uuid.uuid4().hex[:6]}")
    os.makedirs(job_dir, exist_ok=True)
    job = {"id": os.path.basename(job_dir), "kind": kind,
           "created_unix": time.time(), "cwd": cwd or cfg.PROJECT_DIR,
           "steps": steps}
    with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=2)

    python = python or sys.executable
    #  pythonw.exe has no stdout at all; the runner redirects its children
    #  into the log, so it can be either -- but a GUI started with pythonw
    #  must not hand the runner a console it then flashes on screen.
    argv = [python, "-u", os.path.join(cfg.PROJECT_DIR, TRAINING_JOB_SCRIPT),
            "--job", os.path.join(job_dir, "job.json")]
    kwargs = dict(cwd=cwd or cfg.PROJECT_DIR, stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  close_fds=True)
    if os.name == "nt":
        base = (subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            subprocess.Popen(argv, creationflags=base
                             | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:
            #  The enclosing job does not allow breakaway. Still detached
            #  from the window; only a kill-on-close job could stop it now.
            subprocess.Popen(argv, creationflags=base, **kwargs)
    else:                                                   # pragma: no cover
        subprocess.Popen(argv, start_new_session=True, **kwargs)
    return job_dir


def cancel_training_job(job_dir: str) -> None:
    """Ask the runner to stop. It kills the current step's process tree."""
    try:
        with open(os.path.join(job_dir, "cancel"), "w", encoding="utf-8") as fh:
            fh.write(str(time.time()))
    except OSError:                                         # pragma: no cover
        pass


def list_training_jobs(models_dir: str) -> List[str]:
    """Job folders, newest first."""
    root = training_jobs_dir(models_dir)
    try:
        names = sorted((n for n in os.listdir(root)
                        if os.path.isdir(os.path.join(root, n))), reverse=True)
    except OSError:
        return []
    return [os.path.join(root, n) for n in names]


def active_training_job(models_dir: str) -> Optional[str]:
    """The job a freshly opened window should reattach to, if any."""
    for job_dir in list_training_jobs(models_dir):
        if job_is_live(job_dir):
            return job_dir
    return None


def unseen_finished_jobs(models_dir: str) -> List[str]:
    """Jobs that finished while no window was watching, oldest first."""
    out = []
    for job_dir in list_training_jobs(models_dir):
        status = read_job_status(job_dir)
        if (status and status.get("state") in ("finished", "cancelled")
                and not os.path.exists(os.path.join(job_dir, "seen"))):
            out.append(job_dir)
    return out[::-1]


def mark_job_seen(job_dir: str) -> None:
    try:
        with open(os.path.join(job_dir, "seen"), "w", encoding="utf-8") as fh:
            fh.write(str(time.time()))
    except OSError:                                         # pragma: no cover
        pass


class JobLogTail:
    """
    Read a job's log incrementally, handing out COMPLETE lines only.

    The runner writes the log while this reads it, so a read can end half
    way through a `PROGRESS` or `TRAINEVENT` line; parsing that half would
    lose the event. The partial line is held until its newline arrives.
    """

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self._partial = b""

    def read_lines(self, final: bool = False) -> List[str]:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            chunk = b""
        self.offset += len(chunk)
        data = self._partial + chunk
        if final:
            complete, self._partial = data, b""
        else:
            cut = data.rfind(b"\n")
            complete, self._partial = ((data[:cut + 1], data[cut + 1:])
                                       if cut >= 0 else (b"", data))
        return [line.rstrip("\r")
                for line in complete.decode("utf-8", errors="replace")
                .split("\n") if line.strip()]

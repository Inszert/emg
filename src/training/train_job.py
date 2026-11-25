"""
train_job.py -- run one or more trainers in the background, independent of
the GUI, so that closing the window does not stop a training run.

    python training/train_job.py --job models/jobs/<id>/job.json

The GUI writes `job.json` and starts this script DETACHED. From then on the
two only talk through files in the job folder:

    job.json      what to run: {"steps": [{"name", "argv"}], "cwd"}
    log.txt       every step's merged stdout/stderr, plus JOBSTEP lines
    status.json   {"state", "pid", "step", "steps": [...], "exit_code", ...}
    cancel        appears when the GUI asks to stop; the runner kills the
                  current step's whole process tree and exits

That is the whole reason for the indirection. A QProcess child is owned by
the window that started it and dies with it; a file-based job outlives the
window, and a window opened later finds it again by reading status.json and
picks the log up where it is (MainWindow._resume_training_job).

Steps run strictly one after another -- the "one trainer at a time" rule
(CLAUDE.md 8L) holds inside a chain exactly as it does between buttons -- and
a step that fails stops the chain: there is no point sweeping the net on a
dataset the classifier just refused.

Log markers, one per line, which the GUI routes on:

    JOBSTEP <i> <n> <name>        step i of n is starting
    JOBSTEPEND <name> <exit>      it finished, with that exit code

No third-party imports: this has to start in a fraction of a second and must
not care which trainer it is running.
"""
import argparse
import json
import os
import subprocess
import sys
import time

POLL_S = 0.3


def write_status(path: str, status: dict, patience_s: float = 15.0) -> None:
    """
    Atomic: the GUI polls this file and must never read half of it.

    And retried, because on Windows `os.replace` onto a file that another
    process has open -- the GUI reading it, three times a second -- fails
    with "Access is denied". Measured: 1712 failures in 2000 writes against
    a reader in a tight loop. Unretried, one unlucky collision crashed the
    runner and left "running" in the status for good.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    deadline = time.time() + patience_s
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.time() > deadline:
                raise
            time.sleep(0.02)


def kill_tree(proc: subprocess.Popen) -> None:
    """The trainer AND its joblib workers -- killing only the parent leaves
    the pool running and the CPU saturated for nobody."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:                                                  # pragma: no cover
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:                      # pragma: no cover
        proc.kill()


def run(job_path: str) -> int:
    folder = os.path.dirname(os.path.abspath(job_path))
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    steps = job["steps"]
    cwd = job.get("cwd") or os.getcwd()
    log_path = os.path.join(folder, "log.txt")
    status_path = os.path.join(folder, "status.json")
    cancel_path = os.path.join(folder, "cancel")

    status = {
        "id": job.get("id", os.path.basename(folder)),
        "kind": job.get("kind", ""),
        "pid": os.getpid(),
        "state": "running",
        "step": 0,
        "steps": [{"name": s["name"], "exit_code": None} for s in steps],
        "exit_code": None,
        "started_unix": time.time(),
        "finished_unix": None,
    }
    write_status(status_path, status)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    flags = (subprocess.CREATE_NO_WINDOW
             if os.name == "nt" else 0)

    exit_code = 0
    with open(log_path, "ab", buffering=0) as log:
        for index, step in enumerate(steps):
            if os.path.exists(cancel_path):
                status["state"] = "cancelled"
                break
            status["step"] = index
            write_status(status_path, status)
            log.write(f"JOBSTEP {index + 1} {len(steps)} {step['name']}\n"
                      .encode("utf-8"))
            proc = subprocess.Popen(step["argv"], cwd=cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,
                                    creationflags=flags)
            cancelled = False
            while proc.poll() is None:
                if os.path.exists(cancel_path):
                    kill_tree(proc)
                    cancelled = True
                    break
                time.sleep(POLL_S)
            code = proc.wait()
            status["steps"][index]["exit_code"] = code
            log.write(f"JOBSTEPEND {step['name']} {code}\n".encode("utf-8"))
            if cancelled:
                status["state"] = "cancelled"
                exit_code = code if code else 1
                break
            if code != 0:
                exit_code = code
                break

    if status["state"] != "cancelled":
        status["state"] = "finished"
    status["exit_code"] = exit_code
    status["finished_unix"] = time.time()
    write_status(status_path, status)
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", required=True, help="path to job.json")
    args = parser.parse_args(argv)
    try:
        return run(args.job)
    except Exception as exc:
        #  Whatever went wrong, never leave "running" behind: say it failed,
        #  and put the reason where the GUI reads output.
        folder = os.path.dirname(os.path.abspath(args.job))
        try:
            with open(os.path.join(folder, "log.txt"), "ab") as log:
                log.write(f"\ntrain_job.py failed: {exc!r}\n".encode("utf-8"))
            status_path = os.path.join(folder, "status.json")
            try:
                with open(status_path, encoding="utf-8") as fh:
                    status = json.load(fh)
            except (OSError, ValueError):
                status = {"pid": os.getpid()}
            status.update(state="finished", exit_code=status.get(
                "exit_code") or 1, finished_unix=time.time(),
                error=repr(exc))
            write_status(status_path, status)
        except Exception:                                  # pragma: no cover
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

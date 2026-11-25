"""
Background QObjects: dataset export and the training job watcher.
"""

import csv
import json
import os
import time
import traceback
from typing import Optional

import numpy as np

from emg_app import config as cfg
from emg_app.utils import ensure_dir, iso_time, timestamp_slug, unique_path
from emg_app.recording import recording_config_snapshot
from emg_app.dataset import load_dataset, make_feature_matrix, make_windows
from emg_app.jobs import (
    JobLogTail, cancel_training_job, mark_job_seen, pid_alive,
    read_job_status)
from emg_app.gui.qt import QtCore


# ------------------------------------------------------------------------------
# Background workers
# ------------------------------------------------------------------------------


class DatasetExportWorker(QtCore.QThread):
    """
    Builds a combined, ML-ready bundle from everything on disk.

    Runs off the GUI thread because it walks the whole dataset tree. It only
    READS the dataset -- export can never damage a recording.
    """
    progress = QtCore.Signal(str)
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, root: str, out_dir: str, stage: str = "prefer_trimmed",
                 parent=None):
        super().__init__(parent)
        self.root = root
        self.out_dir = out_dir
        self.stage = stage

    def run(self) -> None:
        try:
            self.progress.emit("Scanning the dataset...")
            recs = load_dataset(self.root, stage=self.stage, prefer="npz")
            if not recs:
                self.failed.emit("No recordings found to export.")
                return

            self.progress.emit(f"Windowing {len(recs)} recordings...")
            X, y, groups, meta = make_windows(recs)
            if X.shape[0] == 0:
                self.failed.emit(
                    "No windows could be built. WINDOW_MS is probably longer "
                    "than the recordings."
                )
                return

            self.progress.emit(f"Extracting features for {X.shape[0]} windows...")
            features, names = make_feature_matrix(X, meta["fs"])

            ensure_dir(self.out_dir)
            stamp = timestamp_slug()
            bundle = unique_path(os.path.join(self.out_dir, f"dataset_{stamp}.npz"))

            self.progress.emit("Writing the bundle...")
            np.savez_compressed(
                bundle,
                X=X.astype(np.float32),
                y=y.astype(np.int16),
                groups=np.asarray(groups, dtype=str),
                label_names=np.asarray(meta["labels"], dtype=str),
                features=features,
                feature_names=np.asarray(names, dtype=str),
                window_samples=np.array(meta["window_samples"]),
                step=np.array(meta["step"]),
                fs=np.array(meta["fs"]),
                channel_names=np.asarray(cfg.CHANNEL_NAMES, dtype=str),
            )

            index_path = os.path.splitext(bundle)[0] + "_index.csv"
            with open(index_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["uid", "person_id", "session_id", "label",
                                "repetition", "window_index", "source"],
                )
                writer.writeheader()
                writer.writerows(meta["records"])

            summary_path = os.path.splitext(bundle)[0] + "_summary.json"
            persons = sorted(set(np.asarray(groups, dtype=str).tolist()))
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "created_iso": iso_time(time.time()),
                    "stage": self.stage,
                    "n_recordings": len(recs),
                    "n_windows": int(X.shape[0]),
                    "window_samples": int(meta["window_samples"]),
                    "step": int(meta["step"]),
                    "fs": float(meta["fs"]),
                    "labels": meta["labels"],
                    "persons": persons,
                    "windows_per_label": {
                        name: int((y == i).sum())
                        for i, name in enumerate(meta["labels"])
                    },
                    "windows_per_person": {
                        p: int((np.asarray(groups, dtype=str) == p).sum())
                        for p in persons
                    },
                    "validation_note": (
                        "Use the groups array for leave-one-person-out or "
                        "GroupKFold. Do NOT split windows randomly: windows "
                        "from one recording overlap, so a random split leaks "
                        "the test set into training."
                    ),
                    "config": recording_config_snapshot(),
                }, fh, indent=2, default=str)

            self.done.emit(
                f"Exported {X.shape[0]} windows from {len(recs)} recordings "
                f"({len(persons)} person(s)).\n\n{bundle}"
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TrainingJobWatcher(QtCore.QObject):
    """
    The window's view of one background training job (see SECTION 5b).

    It replaces the QProcess the GUI used to own. It reads the job's log on a
    timer and emits complete lines, and emits `finished` once the runner says
    it is done -- or once the runner has demonstrably died without saying so,
    so that a killed runner cannot leave the panel at "training" forever.
    Stopping the watcher (`detach`) does NOT stop the job; `kill` does.
    """

    lines = QtCore.Signal(list)
    finished = QtCore.Signal(int, object)

    #  The runner writes status.json within milliseconds of starting. If
    #  there is still none after this long, it never started.
    START_GRACE_S = 30.0

    def __init__(self, job_dir: str, parent=None):
        super().__init__(parent)
        self.job_dir = job_dir
        self.tail = JobLogTail(os.path.join(job_dir, "log.txt"))
        self.created = time.time()
        try:
            with open(os.path.join(job_dir, "job.json"),
                      encoding="utf-8") as fh:
                self.job = json.load(fh)
        except (OSError, ValueError):
            self.job = {}
        self.kind = self.job.get("kind", "")
        self.done = False
        self._seen_status = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.poll)

    def start(self) -> None:
        self.timer.start()

    def status(self) -> Optional[dict]:
        return read_job_status(self.job_dir)

    def started_unix(self) -> float:
        status = self.status() or {}
        return float(status.get("started_unix")
                     or self.job.get("created_unix") or self.created)

    def poll(self) -> None:
        if self.done:
            return
        batch = self.tail.read_lines()
        if batch:
            self.lines.emit(batch)
        status = self.status()
        if status is None:
            #  A read can also fail for a moment while the runner replaces
            #  the file. Only a status NEVER seen means it did not start.
            if (not self._seen_status
                    and time.time() - self.created > self.START_GRACE_S):
                self._finish(-1, "the training runner never started")
            return
        self._seen_status = True
        state = status.get("state")
        if state in ("finished", "cancelled"):
            rest = self.tail.read_lines(final=True)
            if rest:
                self.lines.emit(rest)
            code = status.get("exit_code")
            self._finish(int(code) if code is not None else -1, state)
        elif not pid_alive(status.get("pid")):
            #  Re-read once: the runner may have finished between the two.
            status = self.status() or {}
            if status.get("state") == "running":
                rest = self.tail.read_lines(final=True)
                if rest:
                    self.lines.emit(rest)
                self._finish(-1, "the training runner died")

    def _finish(self, code: int, why: str) -> None:
        self.done = True
        self.timer.stop()
        mark_job_seen(self.job_dir)
        self.finished.emit(code, why)

    def kill(self) -> None:
        """Ask the job to stop. `finished` follows once it has."""
        cancel_training_job(self.job_dir)

    def detach(self) -> None:
        """Stop watching. The job carries on."""
        self.timer.stop()

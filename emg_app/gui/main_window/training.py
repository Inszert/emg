"""
RETRAIN, RETRAIN ML + NEURAL NET, the training panel and the background jobs
it watches (CLAUDE.md 8m).

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import List

import numpy as np

from emg_app import config as cfg
from emg_app.dataset import dataset_summary, describe_training_data
from emg_app.gui.dialogs.model_spec import ModelSpecDialog
from emg_app.gui.qt import QtWidgets
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_GOOD, COL_TEXT, COL_TEXT_DIM, COL_WARN, FS_SM)
from emg_app.gui.widgets import make_label
from emg_app.gui.workers import TrainingJobWatcher
from emg_app.jobs import (
    active_training_job, mark_job_seen, read_job_status, start_training_job,
    unseen_finished_jobs)
from emg_app.model_spec import (
    describe_model_spec, enabled_models, load_model_spec, model_spec_path,
    save_model_spec)
from emg_app.utils import fmt_duration


class TrainingMixin:
    """MainWindow -- RETRAIN, RETRAIN ML + NEURAL NET, the training panel and the background jobs it watches (CLAUDE.md 8m)."""
    def _group_training(self):
        """
        What the trainer is doing right now.

        Everything here comes from `TRAINEVENT` lines the trainer subprocess
        prints as it finishes real work -- no field is estimated except the
        ETA, which is derived from completions already observed and says so.
        With nothing running it reads "idle", not a stale percentage.
        """
        group = QtWidgets.QGroupBox("TRAINING")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.train_job_label = make_label("idle", FS_SM, True, COL_TEXT_DIM)
        self.train_job_label.setWordWrap(True)
        layout.addWidget(self.train_job_label)

        self.train_bar = QtWidgets.QProgressBar()
        self.train_bar.setRange(0, 1000)
        self.train_bar.setTextVisible(True)
        self.train_bar.setFormat("%p%")
        self.train_bar.setFixedHeight(18)
        layout.addWidget(self.train_bar)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.train_fields = {}
        for row, (key, caption) in enumerate((
                ("status", "status"),
                ("stage", "stage"),
                ("model", "algorithm"),
                ("setup", "preprocessing"),
                ("candidate", "candidate"),
                ("folds", "folds"),
                ("device", "device"),
                ("elapsed", "elapsed"),
                ("eta", "remaining"),
                ("artifact", "saved to"),
        )):
            grid.addWidget(make_label(caption, FS_SM, False, COL_TEXT_DIM),
                           row, 0)
            value = make_label("--", FS_SM, False, COL_TEXT)
            value.setWordWrap(True)
            grid.addWidget(value, row, 1)
            self.train_fields[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.train_message = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.train_message.setWordWrap(True)
        layout.addWidget(self.train_message)
        return group

    # -- training progress ----------------------------------------------------

    def _reset_training_panel(self, job: str = "idle") -> None:
        if not hasattr(self, "train_fields"):
            return
        self.train_job_label.setText(job)
        self.train_bar.setValue(0)
        self.train_bar.setFormat("%p%")
        for value in self.train_fields.values():
            value.setText("--")
        self.train_message.setText("")
        self._set_train_field("status", "queued", COL_WARN)
        self._train_stages = []
        self._train_stage_clock = {}
        self._train_eta_snapshot = None
        self._train_live = False

    def _estimate_training_remaining(self, event: dict, done: int, total: int,
                                     now: float):
        """
        (seconds_left, is_lower_bound), or None while there is nothing to go on.

        A unit of work costs very different amounts in different stages (the
        A1 screen is cheap probes on half the windows; stage B fits every
        model), so a single done/total rate is wrong in a predictable way: it
        is optimistic early and then jumps. Each stage is timed on its own:

          * the CURRENT stage's remaining units use the rate measured inside
            that stage -- same kind of work, so that part is trustworthy;
          * LATER stages have not run, so their cost is unknown. They are
            counted at the slowest rate seen so far, which is a floor because
            the stages get more expensive, and the result is flagged as a
            lower bound ("at least") rather than presented as a prediction.
        """
        stages = self._train_stages
        if not stages:
            return None
        names = [name for name, _units in stages]
        phase = str(event.get("phase", ""))
        current = phase if phase in names else "tail"
        if current not in names:
            return None
        index = names.index(current)

        clock = self._train_stage_clock.setdefault(
            current, [now, done, now, done])
        clock[2], clock[3] = now, done

        rates = {}
        for name, (t0, d0, t1, d1) in self._train_stage_clock.items():
            if d1 - d0 >= 2 and t1 > t0:
                rates[name] = (t1 - t0) / (d1 - d0)
        if not rates:
            return None

        before = sum(units for _n, units in stages[:index])
        left_here = max(0, stages[index][1] - max(0, done - before))
        rate_here = rates.get(current)
        slowest = max(rates.values())
        remaining = left_here * (rate_here if rate_here else slowest)
        later = sum(units for _n, units in stages[index + 1:])
        remaining += later * slowest
        return remaining, later > 0

    def _tick_training_clock(self) -> None:
        """Once a second: keep elapsed running and count the ETA down."""
        if not self._train_live or not hasattr(self, "train_fields"):
            return
        now = time.perf_counter()
        self._set_train_field("elapsed",
                              fmt_duration(now - self.training_started_at))
        snap = self._train_eta_snapshot
        if snap is None:
            return
        taken_at, seconds, lower = snap
        left = max(0.0, seconds - (now - taken_at))
        text = f"{'at least ' if lower else 'about '}{fmt_duration(left)}"
        self._set_train_field("eta", text)
        self.train_bar.setFormat(f"%p%  ({fmt_duration(left)} left)")

    def _set_train_field(self, key: str, text: str, colour: str = None) -> None:
        widget = getattr(self, "train_fields", {}).get(key)
        if widget is None:
            return
        widget.setText(text)
        widget.setStyleSheet(
            f"color: {colour or COL_TEXT}; background: transparent;")

    def _on_train_event(self, event: dict) -> None:
        """One TRAINEVENT line from the trainer, straight onto the panel."""
        if not hasattr(self, "train_fields"):
            return
        # Every number here came off a pipe, so none of it is trusted: a
        # malformed event should leave the panel as it was, not raise inside
        # the output pump.
        def number(key, fallback=0):
            try:
                return int(float(event.get(key, fallback)))
            except (TypeError, ValueError):
                return fallback

        done = number("done", 0)
        total = max(1, number("total", 1))
        fraction = min(1.0, max(0.0, done / total))
        self.train_bar.setValue(int(fraction * 1000))

        status = str(event.get("status", "running"))
        colour = {"completed": COL_GOOD, "failed": COL_BAD,
                  "cancelled": COL_WARN, "running": COL_ACCENT}.get(
                      status, COL_TEXT)
        self._set_train_field("status", status, colour)
        self._set_train_field("stage", str(event.get("phase", "--")))
        self._set_train_field("model", str(event.get("model", "--")))
        self._set_train_field("candidate", f"{done} / {total}")
        self._set_train_field(
            "device", str(event.get("device", "cpu")).upper(),
            COL_GOOD if str(event.get("device")) != "cpu" else COL_TEXT)

        try:
            window_s = float(event["window_s"])
        except (KeyError, TypeError, ValueError):
            window_s = None
        if window_s is not None:
            self._set_train_field(
                "setup",
                f"{window_s:.2f}s  ov {event.get('overlap', '?')}  "
                f"{event.get('representation', '?')}/"
                f"{event.get('feature_set', '?')}  "
                f"({event.get('n_windows', '?')} windows x "
                f"{event.get('n_features', '?')} feat)")
        if event.get("n_folds") is not None:
            self._set_train_field("folds", f"{event['n_folds']} per candidate")

        elapsed = time.perf_counter() - self.training_started_at
        self._set_train_field("elapsed", fmt_duration(elapsed))
        # ETA from the rate actually achieved, never a fixed guess, and only
        # once enough work is done for the rate to mean anything.
        if isinstance(event.get("phases"), list):
            try:
                self._train_stages = [(str(n), int(u))
                                      for n, u in event["phases"]]
            except (TypeError, ValueError):
                self._train_stages = []
            self._train_stage_clock = {}
        self._train_live = status == "running"
        if status != "running":
            self._train_eta_snapshot = None
        else:
            now = time.perf_counter()
            estimate = self._estimate_training_remaining(event, done, total, now)
            if estimate is None and not self._train_stages:
                # A trainer that does not announce its stages (the neural-net
                # one): the plain rate so far, once there is enough of it.
                if done >= 5 and fraction > 0.01:
                    estimate = (max(0.0, elapsed / fraction - elapsed), False)
            if estimate is None:
                self._train_eta_snapshot = None
                self._set_train_field("eta", "estimating...")
            else:
                self._train_eta_snapshot = (now, estimate[0], estimate[1])
                self._tick_training_clock()

        if event.get("artifact"):
            self._set_train_field("artifact", str(event["artifact"]), COL_GOOD)
        if event.get("message"):
            self.train_message.setText(str(event["message"]))

        if status == "completed":
            self._set_train_field("eta", "done")
            self.train_bar.setFormat("%p%  complete")
            try:
                vote = float(event["recording_vote_accuracy"])
            except (KeyError, TypeError, ValueError):
                vote = None
            if vote is not None:
                self.train_message.setText(
                    f"{event.get('model', 'model')}: "
                    f"{vote * 100:.1f} % per gesture, "
                    f"{float(event.get('window_accuracy') or 0) * 100:.1f} "
                    "% per window")

    def _refresh_spec_summary(self) -> None:
        """The one-line 'what the next retrain will train' readout."""
        if not hasattr(self, "spec_summary"):
            return
        text = describe_model_spec(self.model_spec)
        if self._spec_problems:
            text += "   (spec file was corrected on load)"
        self.spec_summary.setText("next retrain: " + text)
        self.spec_summary.setStyleSheet(
            f"color: {COL_WARN if self._spec_problems else COL_TEXT_DIM};"
            " background: transparent;")

    def edit_model_spec(self) -> None:
        """
        Open the model specification editor.

        Changing the spec only affects the NEXT retrain -- the model currently
        loaded was produced by whatever spec was in force when it was
        trained, which is recorded inside it.
        """
        path = model_spec_path(self.predictor.path or cfg.MODEL_PATH)
        dialog = ModelSpecDialog(self.model_spec, path, self)
        dialog.exec()
        if not dialog.saved:
            return

        self.model_spec, self._spec_problems = load_model_spec(path)
        self._refresh_spec_summary()
        self._append_log(
            f"Model spec saved: {describe_model_spec(self.model_spec)}.")
        enabled = enabled_models(self.model_spec)
        self._append_log(
            "  will train: " + ", ".join(m["name"] for m in enabled))
        if self.predictor.model is not None:
            self._append_log(
                "  the loaded model is unchanged -- retrain to apply this.")

    def retrain_model(self, confirm: bool = True, quick: bool = False) -> None:
        """
        Retrain on everything recorded so far.

        `confirm=False` skips the "are you sure" dialog and uses `quick`
        directly. That is for `--train` on the command line, where asking the
        question again would be asking it twice; every other guard (a session
        running, too little data, a training already going) still applies.

        train_model.py reads the whole dataset tree, so a retrain always uses
        the new recordings PLUS every earlier session and subject. There is no
        mode that trains on just the latest batch -- that would throw away
        most of the data and is never what you want.

        It runs as a separate process: scikit-learn with n_jobs=-1 would
        otherwise fight the acquisition thread for the CPU, and a crash in
        training must not take the recorder down with it.
        """
        if self.training_process is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Training in progress",
                "Training is already running. Cancel it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self._training_cancelled = True
                self.training_process.kill()
            return

        if self.nn_training_process is not None:
            #  The button is disabled while the net trains, but `--train` on
            #  the command line reaches this method directly.
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                "The neural net is still training. One trainer at a time -- "
                "they would fight over the machine.")
            return

        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                "Finish or stop the recording session first.\n\n"
                "Training saturates the CPU and would disturb the live "
                "stream, which could corrupt the recordings in progress.",
            )
            return

        script = os.path.join(cfg.PROJECT_DIR, "training", "train_model.py")
        if not os.path.exists(script):
            QtWidgets.QMessageBox.warning(
                self, "Retrain", f"train_model.py not found at:\n{script}")
            return

        summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        if summary["total_raw"] < 12:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                f"Only {summary['total_raw']} recordings on disk. Collect "
                "more before training -- a handful cannot be validated.",
            )
            return

        people = summary["persons"]
        per_person = "\n".join(
            f"    {person}: " + ", ".join(
                f"{session} ({sum(counts.values())})"
                for session, counts in sorted(people[person].items()))
            for person in sorted(people))

        box = QtWidgets.QMessageBox(self) if confirm else None
        if box is None:
            self._append_log(
                f"Retrain requested on the command line "
                f"({'quick' if quick else 'full sweep'}).")
        else:
            box.setWindowTitle("Retrain on all data")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText(
                f"Retrain using ALL {summary['total_raw']} recordings on disk."
            )
        if box is not None:
            box.setInformativeText(
                f"{len(people)} subject(s):\n{per_person}\n\n"
                "Everything is used together -- the recordings from this run "
                "are added to the earlier ones, not trained on separately.\n\n"
                "Quick: a reduced grid, about a minute.\n"
                # Named from the spec rather than hard-coded: the GUI always
                # passes --spec, so "21 models" stopped being true the moment
                # MODEL SPEC could switch them off.
                # "every setting" was true of the old exhaustive sweep,
                # which measured 3 h 16 m. It now SCREENS every setting
                # cheaply and fully probes only a shortlist -- measured at
                # 0.02 points of accuracy. Saying "every setting" would
                # overstate it; saying "a while" would overstate the wait.
                f"Full:  screens every setting, then fully probes the best "
                f"few on all {len(enabled_models(self.model_spec))} enabled "
                "model(s) -- roughly 15 minutes.\n\n"
                "The current model is archived, not overwritten.\n\n"
                # Said here, before a long sweep, rather than leaving the GPU
                # badge above to imply something it does not mean.
                f"Runs on: CPU ({os.cpu_count()} logical cores) -- "
                "scikit-learn has no GPU backend, and there is no GPU build "
                "of these algorithms for Windows/Python 3.14."
                # Asked out loud the first time somebody saw the two panels,
                # so it is answered in the dialog rather than only in the
                # documentation: this button trains ONE of the two models.
                + ("\n\nThis does NOT retrain the neural network. That is a "
                   "separate model with its own TRAIN NEURAL NET button; "
                   "after this finishes it will read OUT OF DATE until you "
                   "train it too."
                   if (self.nn_engine is not None
                       and self.nn_engine.trained()) else "")
            )
            quick_button = box.addButton("Quick",
                                         QtWidgets.QMessageBox.AcceptRole)
            full_button = box.addButton("Full sweep",
                                        QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(quick_button)
            box.exec()

            clicked = box.clickedButton()
            if clicked not in (quick_button, full_button):
                return
            quick = clicked is quick_button

        self.stop_prediction()
        # Both: the detailed panel is in TRAINING, and the original bar and
        # status line are in LIVE CLASSIFIER next to the RETRAIN button that
        # was just pressed.
        self.expand_section("prediction")
        self.expand_section("training")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel(
            f"train_model.py  --  {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)"))
        self.training_output = []
        args = self._ml_train_args(script, quick)
        self._launch_training_job("ml", [{"name": "ml",
                                          "argv": [sys.executable] + args}])
        self._after_ml_launch(summary, quick)

    def _ml_train_args(self, script: str, quick: bool) -> List[str]:
        """train_model.py's command line, exactly as the GUI is configured."""
        # Pass the paths the APP is configured with, rather than letting
        # train_model.py fall back to its own defaults -- otherwise a
        # changed DATA_FOLDER or MODEL_PATH would silently train on, and
        # overwrite, something else entirely.
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        out_dir = os.path.dirname(os.path.abspath(
            self.predictor.full_path)) or cfg.PROJECT_DIR
        args = ["-u", script, "--root", root, "--out-dir", out_dir,
                "--model-path", self.predictor.full_path,
                # Same reason as --root and --model-path: the subprocess must
                # calibrate the way the GUI is set to, not the way its own
                # module default happens to read.
                "--calibration", cfg.CALIBRATION_MODE]

        # Write the spec out first so the subprocess reads exactly what the
        # GUI is showing, even if it was never saved from the dialog.
        spec_file = model_spec_path(self.predictor.full_path)
        try:
            save_model_spec(spec_file, self.model_spec)
            args += ["--spec", spec_file]
            self._append_log(
                f"Training with {describe_model_spec(self.model_spec)}.")
        except OSError as exc:
            # Not fatal: without --spec the trainer uses its built-in roster,
            # which is the old behaviour. Say so rather than failing silently.
            self._append_log(
                f"could not write the model spec ({exc}); training with the "
                "built-in roster instead")

        if quick:
            args.append("--quick")
        return args

    def _after_ml_launch(self, summary: dict, quick: bool) -> None:
        self.retrain_button.setText("CANCEL TRAINING")
        self.training_started_at = time.perf_counter()
        self.training_phase_text = (
            f"starting on {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)") + "...")
        self.training_progress.setValue(0)
        self.training_progress.setFormat("%p%")
        self.training_progress.setVisible(True)
        self.training_status.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self._refresh_training_status()
        self._append_log(
            f"Retraining started on all {summary['total_raw']} recordings "
            f"({'quick' if quick else 'full sweep'}), on the CPU."
        )

    # -- background jobs -------------------------------------------------------
    #
    #  Every trainer the GUI starts runs as a job (train_job.py, detached), so
    #  closing the window no longer kills it. The job's log is routed to the
    #  same two line handlers the QProcess pumps used to feed, and each step's
    #  end (JOBSTEPEND in the log) calls the same two finish handlers.

    def _models_dir(self) -> str:
        return (os.path.dirname(os.path.abspath(self.predictor.full_path))
                or cfg.PROJECT_DIR)

    def _launch_training_job(self, kind: str, steps: List[dict]) -> None:
        job_dir = start_training_job(self._models_dir(), kind, steps,
                                     cwd=cfg.PROJECT_DIR)
        self._attach_training_job(job_dir)
        self._append_log(
            f"Training runs in the background (job {os.path.basename(job_dir)})"
            f" -- closing the app will not stop it.")

    def _attach_training_job(self, job_dir: str) -> "TrainingJobWatcher":
        watcher = TrainingJobWatcher(job_dir, self)
        self._job = watcher
        self._job_step = ""
        self._job_steps_ended = set()
        first = (watcher.job.get("steps") or [{}])[0].get("name", "ml")
        if watcher.kind == "all":
            self._chain_job = watcher
        if first == "nn":
            self.nn_training_process = watcher
        else:
            self.training_process = watcher
        watcher.lines.connect(self._on_job_lines)
        watcher.finished.connect(self._on_job_finished)
        watcher.start()
        return watcher

    def _on_job_lines(self, lines: List[str]) -> None:
        """Route a batch of log lines to the handler of the step running."""
        run: List[str] = []

        def flush():
            if run:
                handler = {"nn": self._on_nn_training_lines,
                           "zoo": self._on_zoo_lines}.get(
                    self._job_step, self._on_training_lines)
                handler(list(run))
                run.clear()

        for line in lines:
            if line.startswith("JOBSTEP "):
                flush()
                parts = line.split()
                self._job_step = parts[3] if len(parts) > 3 else ""
                if (self._job_step == "nn" and self._job is not None
                        and self._job.kind == "all"):
                    self._begin_chained_nn_step()
                elif self._job_step == "zoo":
                    self._begin_zoo_step()
                continue
            if line.startswith("JOBSTEPEND "):
                flush()
                parts = line.split()
                name = parts[1] if len(parts) > 1 else ""
                try:
                    code = int(parts[2])
                except (IndexError, ValueError):
                    code = -1
                self._job_steps_ended.add(name)
                self._end_job_step(name, code)
                continue
            run.append(line)
        flush()

    def _end_job_step(self, name: str, code: int) -> None:
        chained = self._job is not None and self._job.kind == "all"
        if name == "nn":
            self._on_nn_training_finished(code)
            return
        if name == "zoo":
            self._on_zoo_finished(code, chained)
            return
        self._on_training_finished(code, announce=not chained)
        if chained and code == 0:
            self._chain_ml_note = getattr(self, "_last_ml_result", "")

    def _begin_chained_nn_step(self) -> None:
        """The classifier finished in an ML + NN chain; the net starts now."""
        self.training_process = None
        self.nn_training_process = self._job
        if self.nn_engine is not None:
            self.stop_neural()
        self.expand_section("neural")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel("train_nn.py  --  chained after the "
                                   "classifier")
        self.training_output = []
        self.training_phase_text = "neural net: starting (step 2 of 2)..."
        self.training_progress.setValue(0)
        self.training_progress.setFormat("%p%")
        self.training_progress.setVisible(True)
        self.training_status.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self._refresh_training_status()
        self._refresh_neural_panel()
        self._append_log("Classifier done -- the neural net is training now "
                         "(step 2 of 2).")

    def _on_job_finished(self, code: int, why) -> None:
        """
        The runner is done. Normally every step already reported through
        JOBSTEPEND; this only finishes a step that never did -- a runner that
        died, or a job cancelled before its step began.
        """
        watcher = self._job
        steps = [s.get("name", "ml") for s in
                 ((watcher.job.get("steps") if watcher else None) or [])]
        pending = self._job_step or (steps[0] if steps else "ml")
        if pending not in self._job_steps_ended:
            if why not in ("finished", "cancelled"):
                self._append_log(f"Training job ended abnormally: {why}.")
            self._end_job_step(pending, code if code else -1)
        self._job = None
        self._chain_job = None
        if self.training_process is watcher:
            self.training_process = None
        if self.nn_training_process is watcher:
            self.nn_training_process = None
        self._refresh_controls()
        self._refresh_neural_panel()

    def retrain_everything(self, zoo_ids=None) -> None:
        """
        Retrain the classifier AND the neural net with one button: one job,
        two steps, strictly in sequence. The net only starts if the
        classifier succeeded.

        `zoo_ids` adds a third step that re-measures and re-fits those zoo
        models (training/train_zoo.py); it only starts if the net succeeded.
        """
        chain = getattr(self, "_chain_job", None)
        if chain is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Training in progress",
                "The ML + neural net training is running. Cancel it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer == QtWidgets.QMessageBox.Yes:
                if self._job_step == "nn":
                    self._nn_training_cancelled = True
                else:
                    self._training_cancelled = True
                chain.kill()
            return
        if self.training_process is not None or \
                self.nn_training_process is not None:
            QtWidgets.QMessageBox.information(
                self, "Retrain", "A training run is already going. One "
                "trainer at a time.")
            return
        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                "Finish or stop the recording session first.\n\n"
                "Training saturates the CPU and would disturb the live "
                "stream, which could corrupt the recordings in progress.")
            return
        ml_script = os.path.join(cfg.PROJECT_DIR, "training", "train_model.py")
        nn_script = os.path.join(cfg.PROJECT_DIR, "training", "train_nn.py")
        missing = [p for p in (ml_script, nn_script) if not os.path.exists(p)]
        if missing or self.nn_engine is None:
            QtWidgets.QMessageBox.warning(
                self, "Retrain",
                "Cannot run both: "
                + (f"not found: {', '.join(missing)}" if missing
                   else "the neural net is disabled in the configuration."))
            return
        summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        if summary["total_raw"] < 12:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                f"Only {summary['total_raw']} recordings on disk. Collect "
                "more before training -- a handful cannot be validated.")
            return

        info = self.compute_info or {}
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Retrain the classifier AND the neural net")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(f"Retrain both models on all {summary['total_raw']} "
                    f"recordings on disk, one after the other.")
        box.setInformativeText(
            "1. the classifier (train_model.py, CPU) -- this is the model "
            "the app predicts with\n"
            "2. the neural net (train_nn.py, "
            + ("GPU" if info.get("cuda") else "CPU")
            + ") -- the separate second opinion, never part of the answer\n\n"
            "Step 2 only starts if step 1 succeeds.\n\n"
            + (f"3. the model zoo: {len(zoo_ids)} model(s) re-measured and "
               "re-saved (train_zoo.py) -- only after step 2 succeeds\n\n"
               if zoo_ids else "")
            + "Quick: reduced grids, a few minutes in total.\n"
            "Full:  both full sweeps, roughly 30-40 minutes.\n\n"
            "Runs in the background: you can close the app and it keeps "
            "going. Reopen it to see the progress.")
        quick_button = box.addButton("Quick", QtWidgets.QMessageBox.AcceptRole)
        full_button = box.addButton("Full sweep",
                                    QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(quick_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (quick_button, full_button):
            return
        quick = clicked is quick_button

        self.stop_prediction()
        self.stop_neural()
        self.expand_section("prediction")
        self.expand_section("training")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel(
            f"ML + NN  --  {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)") + ", step 1 of 2")
        self.training_output = []
        self._chain_ml_note = ""
        steps = [
            {"name": "ml", "argv": [sys.executable]
             + self._ml_train_args(ml_script, quick)},
            {"name": "nn", "argv": [sys.executable]
             + self._nn_train_args(nn_script, quick)},
        ]
        if zoo_ids:
            steps.append({"name": "zoo", "argv": [sys.executable]
                          + self._zoo_train_args(list(zoo_ids))})
        self._launch_training_job("all", steps)
        self._after_ml_launch(summary, quick)
        self._refresh_controls()

    # -- the model zoo --------------------------------------------------------

    def _zoo_train_args(self, ids) -> List[str]:
        """train_zoo.py's command line: re-measure and re-fit these models."""
        script = os.path.join(cfg.PROJECT_DIR, "training", "train_zoo.py")
        args = ["-u", script, "all", "--ids", *ids,
                "--root", os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER),
                "--augment", ",".join(cfg.TRAIN_AUGMENT) or "none"]
        if cfg.TRAIN_ADAPT:
            args.append("--adapt")
        return args

    def retrain_zoo(self, ids) -> None:
        """Re-measure and re-fit the given zoo models, in the background."""
        ids = [i for i in ids]
        if not ids:
            return
        if self.training_process is not None or \
                self.nn_training_process is not None:
            QtWidgets.QMessageBox.information(
                self, "Retrain", "A training run is already going. One "
                "trainer at a time.")
            return
        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Retrain", "Finish or stop the recording session "
                "first: training saturates the CPU and would disturb the "
                "live stream.")
            return
        self.stop_prediction()
        self.stop_neural()
        self.expand_section("training")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel(f"model zoo  --  {len(ids)} model(s)")
        self.training_output = []
        steps = [{"name": "zoo", "argv": [sys.executable]
                  + self._zoo_train_args(ids)}]
        self._launch_training_job("zoo", steps)
        self.training_phase_text = "model zoo: measuring..."
        self.training_progress.setValue(0)
        self.training_progress.setVisible(True)
        self._refresh_training_status()
        self._refresh_controls()

    def _begin_zoo_step(self) -> None:
        self.training_started_at = getattr(self, "training_started_at",
                                           time.perf_counter())
        self.training_phase_text = "model zoo: measuring and re-fitting..."
        self.training_progress.setValue(0)
        self.training_progress.setVisible(True)
        self._append_log("Model zoo training started (leave-one-person-out "
                         "measurement, then the final fit of each model).")
        self._refresh_training_status()

    def _on_zoo_lines(self, lines: List[str]) -> None:
        """train_zoo.py's output: a progress bar and the lines that matter."""
        for line in lines:
            line = line.rstrip()
            if not line or line.startswith("TRAINEVENT "):
                continue
            if line.startswith("PROGRESS "):
                self._update_training_progress(line)
                continue
            self.training_output.append(line)
            if len(self.training_output) > 4000:
                del self.training_output[:2000]
            if line.startswith(("[eval", "[fit", "[ensembles", "done in")):
                self._append_log(f"zoo: {line.strip()}")
            self.training_phase_text = line.strip()[:90]
            self._refresh_training_status()

    def _on_zoo_finished(self, exit_code: int, chained: bool = False) -> None:
        self.training_progress.setVisible(False)
        if self.training_process is not None and \
                getattr(self.training_process, "kind", "") == "zoo":
            self.training_process = None
        took = fmt_duration(time.perf_counter() - self.training_started_at)
        if exit_code != 0:
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(
                f"model zoo training failed (exit {exit_code})")
            self._append_log(f"Model zoo training failed (exit {exit_code}); "
                             "the models already in the zoo are untouched.")
            if not chained:
                QtWidgets.QMessageBox.warning(
                    self, "Model zoo", f"train_zoo.py exited with code "
                    f"{exit_code}.\n\n"
                    + "\n".join(self.training_output[-15:]))
            return
        self.training_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self.training_status.setText(f"model zoo trained, took {took}")
        self._append_log(f"Model zoo trained in {took}. Stars and scores are "
                         "recomputed from the new measurements.")
        if getattr(cfg, "MODEL_SELECTION_ENABLED", False):
            self._reload_selected_models()

    def _resume_training_job(self) -> None:
        """
        At startup: reattach to a training run that outlived the previous
        window, and report runs that finished while no window was open.
        """
        models_dir = self._models_dir()
        for job_dir in unseen_finished_jobs(models_dir):
            status = read_job_status(job_dir) or {}
            kind = {"ml": "classifier", "nn": "neural net",
                    "zoo": "model zoo",
                    "all": "classifier + neural net"}.get(
                status.get("kind"), status.get("kind") or "training")
            when = datetime.fromtimestamp(
                status.get("finished_unix") or time.time()).strftime(
                "%Y-%m-%d %H:%M")
            self._append_log(
                f"While the app was closed: {kind} training "
                + ("finished" if status.get("state") == "finished"
                   and not status.get("exit_code") else
                   f"ended ({status.get('state')}, exit "
                   f"{status.get('exit_code')})")
                + f" at {when}. The models on disk are the ones it left.")
            mark_job_seen(job_dir)

        job_dir = active_training_job(models_dir)
        if job_dir is None:
            return
        watcher = self._attach_training_job(job_dir)
        self.training_started_at = (time.perf_counter()
                                    - (time.time() - watcher.started_unix()))
        self.expand_section("training")
        self._reset_training_panel(
            f"background job {os.path.basename(job_dir)} (reattached)")
        self.training_output = []
        self.training_phase_text = "reattached -- catching up with the log..."
        self.training_progress.setValue(0)
        self.training_progress.setVisible(True)
        if self.training_process is not None:
            self.retrain_button.setText("CANCEL TRAINING")
        self.training_status.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self._refresh_training_status()
        self._refresh_controls()
        self._append_log(
            "A training run started before the app was closed is still going "
            "-- reattached to it.")

    def _on_training_lines(self, lines: List[str]) -> None:
        """train_model.py's output, a batch of complete lines from the job log."""
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                self._update_training_progress(line)
                continue
            if line.startswith("TRAINEVENT "):
                # The detailed panel feed. A malformed line is ignored
                # rather than allowed to break the output pump.
                try:
                    self._on_train_event(json.loads(line[len("TRAINEVENT "):]))
                except (ValueError, TypeError, KeyError):
                    pass
                continue

            self.training_output.append(line)
            if len(self.training_output) > 4000:
                del self.training_output[:2000]
            # Keep the log readable: show the lines that mean something.
            if any(key in line for key in (
                    "Stage", "Evaluated", "WINNER", "saved:", "per-window",
                    "per-recording", "...", "recordings", "subject")):
                self._append_log(f"train: {line.strip()}")
            self.training_phase_text = line.strip()[:90]
            self._refresh_training_status()

    def _update_training_progress(self, line: str) -> None:
        """
        Turn a `PROGRESS done total phase` line into a bar and a time estimate.

        The estimate comes from the rate achieved so far rather than a fixed
        guess, so it corrects itself as the sweep goes: combinations differ a
        lot in cost and an up-front guess would be wrong by minutes.
        """
        parts = line.split()
        try:
            done, total = int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return
        phase = " ".join(parts[3:]) or "working"

        fraction = done / max(1, total)
        self.training_progress.setVisible(True)
        self.training_progress.setValue(int(np.clip(fraction, 0, 1) * 1000))

        elapsed = time.perf_counter() - self.training_started_at
        eta = None
        # Ignore the first few steps: the rate is meaningless until the
        # process has warmed up and loaded the dataset.
        if done >= 5 and fraction > 0.01:
            eta = max(0.0, elapsed / fraction - elapsed)

        self.training_phase_text = (
            f"{phase}  {done}/{total}"
            + (f"  --  about {fmt_duration(eta)} left" if eta is not None
               else "  --  estimating...")
        )
        self.training_progress.setFormat(
            f"%p%  ({fmt_duration(eta)} left)" if eta is not None else "%p%")
        self._refresh_training_status()

    def _refresh_training_status(self) -> None:
        elapsed = time.perf_counter() - self.training_started_at
        self.training_status.setText(
            f"[CPU]  [{fmt_duration(elapsed)} elapsed]  "
            f"{self.training_phase_text}")

    def _on_training_finished(self, exit_code: int, _status=None,
                              announce: bool = True) -> None:
        """
        The classifier step ended. `announce=False` inside an ML + NN chain:
        the result still lands in the panel and the log, but the modal box
        waits for the end of the chain instead of interrupting it.
        """
        self.training_process = None
        self.retrain_button.setText("RETRAIN ON ALL DATA")
        self.training_progress.setVisible(False)

        # Terminal state on the panel. The completed case is painted by the
        # trainer's own final TRAINEVENT, which carries the real metrics;
        # these two are the paths where no such event ever arrives.
        cancelled = bool(getattr(self, "_training_cancelled", False))
        if cancelled:
            self._set_train_field("status", "cancelled", COL_WARN)
            self._set_train_field("eta", "--")
            self.train_message.setText("Cancelled before finishing. "
                                       "The previous model is untouched.")
        elif exit_code != 0:
            self._set_train_field("status", "failed", COL_BAD)
            self._set_train_field("eta", "--")
            self.train_message.setText(
                f"train_model.py exited with code {exit_code}. "
                "The previous model is untouched.")
        self._training_cancelled = False
        took = fmt_duration(time.perf_counter() - self.training_started_at)
        tail = "\n".join(self.training_output[-25:])

        # Exit code 2 is the trainer's "there is nothing here to learn from"
        # -- not enough gesture classes. That is a fact about the dataset, not
        # a crash, so it gets advice instead of a red failure box.
        if exit_code == cfg.TRAIN_EXIT_NEED_MORE_DATA:
            counts = self._gesture_counts()
            have = [g for g, n in counts.items() if n]
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(
                f"not trained: only {len(have)} gesture class"
                + ("" if len(have) == 1 else "es") + " recorded")
            self._append_log(
                "Training skipped: a classifier needs at least two gestures, "
                f"and only {', '.join(have) or 'none'} is on disk.")
            missing = [g for g in cfg.GESTURES if not counts.get(g)]
            QtWidgets.QMessageBox.information(
                self, "More gestures needed",
                "There is only one gesture on disk, so there is nothing to "
                "tell apart yet.\n\n"
                + "On disk:\n"
                + "\n".join(f"    {g}: {counts[g]}" for g in have)
                + ("\n\nNot recorded yet:\n    " + ", ".join(missing)
                   if missing else "")
                + "\n\nRecord a second gesture and press RETRAIN again. "
                "Nothing was changed; your existing model is untouched.")
            self._refresh_gesture_counts()
            return

        if cancelled:
            # The operator stopped it. That is not a failure, and a red error
            # box claiming one would be wrong -- and would also pop up behind
            # their back after they deliberately pressed cancel.
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(f"training cancelled after {took}")
            self._append_log(
                f"Retraining cancelled after {took}. The previous model is "
                "untouched.")
            return

        if exit_code != 0:
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(f"training failed (exit {exit_code})")
            self._append_log(f"Retraining failed with exit code {exit_code}.")
            QtWidgets.QMessageBox.warning(
                self, "Training failed",
                f"train_model.py exited with code {exit_code}.\n\n{tail}")
            return

        before = self.predictor.stats.get("recording_vote_accuracy")
        ok = self.predictor.load()
        # The trainer writes a fresh panel beside the model. Reload it only if
        # it was already in memory; otherwise leave it for the first MODEL
        # STATS open, exactly as at startup.
        if self.decision_panel.loaded:
            self.decision_panel.warm = False
            self.decision_panel.load()
        self._rebuild_prob_bars()
        self._refresh_prediction_panel()
        #  The dataset fingerprint has just been recomputed, and the net's
        #  checkpoint was NOT part of this sweep -- so if the dataset grew,
        #  this is the moment its panel should start saying OUT OF DATE.
        self._refresh_neural_panel()

        if not ok:
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(
                f"trained, but the model would not load: {self.predictor.status}")
            self._append_log(
                f"Training finished but loading failed: {self.predictor.status}")
            return

        stats = self.predictor.stats
        after = stats.get("recording_vote_accuracy")
        delta = ""
        if before is not None and after is not None:
            change = (after - before) * 100
            delta = f"   ({change:+.1f} points vs the previous model)"

        self.training_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self.training_status.setText(
            (f"{stats.get('model_name', 'model')}: "
             f"{after * 100:.1f} % per gesture, took {took}") if after
            else f"model reloaded in {took}")
        self._append_log(
            f"Retraining done: {stats.get('model_name')} "
            f"{(after or 0) * 100:.1f} % per gesture{delta}")
        self._last_ml_result = (
            f"Classifier: {stats.get('model_name', '?')}, "
            f"{(after or 0) * 100:.1f} % per gesture{delta}")
        if not announce:
            return

        QtWidgets.QMessageBox.information(
            self, "Retraining complete",
            f"New model: {stats.get('model_name', '?')}\n\n"
            f"Per gesture : {(after or 0) * 100:.1f} %{delta}\n"
            f"Per window  : {stats.get('window_accuracy', 0) * 100:.1f} %\n"
            f"Trained on  : {stats.get('n_recordings', '?')} of "
            f"{stats.get('n_recordings_total', '?')} recordings, "
            f"{len(stats.get('subjects', []))} subject(s)\n"
            # Spelled out at the one moment the question is live, so it does
            # not have to be reconstructed from the panel a week later.
            f"Data        : "
            f"{describe_training_data(self.predictor.payload_meta.get('data'))}"
            "\n\nOpen MODEL STATS for the full breakdown.",
        )

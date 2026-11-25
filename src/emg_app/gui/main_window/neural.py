"""
The NEURAL NETWORK panel: the separate second opinion (CLAUDE.md 8L). Never
feeds the classifier's answer.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import json
import math
import os
import sys
import time
from typing import List, TYPE_CHECKING

import numpy as np

from emg_app import config as cfg
from emg_app import settings_store
from emg_app import zoo
from emg_app.class_mask import NO_ANSWER, describe_off, kept_note, off_key
from emg_app.dataset import dataset_summary
from emg_app.gui.dialogs.neural_setup import NeuralSetupDialog
from emg_app.gui.dialogs.neural_stats import NeuralStatsDialog
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import (
    COL_BAD, COL_BORDER, COL_GOOD, COL_TEXT_DIM, COL_WARN, FS_SM)
from emg_app.gui.widgets import make_label, wrap_label
from emg_app.neural import (
    describe_nn_spec, load_nn_spec, load_nn_stats, nn_spec_args, nn_spec_path)
from emg_app.utils import fmt_duration, nominal_rate

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.neural import NeuralPrediction


class NeuralMixin:
    """MainWindow -- The NEURAL NETWORK panel."""
    # ======================================================================
    # NEURAL NETWORK PANEL  (section 8L)
    # ======================================================================

    def _group_neural(self):
        """
        The second opinion, with its own everything.

        Laid out to make the separation unmissable rather than merely true:
        its own group box under LIVE CLASSIFIER, a standing caption saying it
        does not contribute to the answer above, its own START button, its own
        bars, its own device chip and its own statistics dialog. Nothing in
        here reads `self.predictor` and nothing writes to it.
        """
        group = QtWidgets.QGroupBox("NEURAL NETWORK  (separate opinion)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        caption = wrap_label(
            "A different model on the same signal: a small sequence net that "
            "reads how the envelope CHANGES across the window. It is NOT "
            "part of the prediction above -- separate file, separate button, "
            "separate numbers, never averaged in.", FS_SM, False, COL_TEXT_DIM)
        layout.addWidget(caption)

        #  Result. Styled like the classifier's so the two are comparable at
        #  a glance, but smaller: the deployed answer stays the headline.
        self.nn_label = make_label("--", 22, True, COL_TEXT_DIM)
        self.nn_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.nn_label)

        self.nn_conf = make_label("", 12, True, COL_TEXT_DIM)
        self.nn_conf.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.nn_conf)

        #  Its own bars, filled only from this net's own softmax.
        self.nn_prob_box = QtWidgets.QWidget()
        bars = QtWidgets.QGridLayout(self.nn_prob_box)
        bars.setContentsMargins(0, 2, 0, 2)
        bars.setSpacing(2)
        bars.setColumnStretch(1, 1)
        layout.addWidget(self.nn_prob_box)

        #  Where the two models disagree. The interesting line: a single
        #  blended number would hide exactly this.
        self.nn_agreement = wrap_label("", FS_SM, True, COL_TEXT_DIM)
        layout.addWidget(self.nn_agreement)

        #  How each net of a soft vote answered (empty for one net).
        self.nn_vote_label = wrap_label("", FS_SM, False, COL_TEXT_DIM)
        self.nn_vote_label.setTextFormat(QtCore.Qt.RichText)
        self.nn_vote_label.setVisible(False)
        layout.addWidget(self.nn_vote_label)

        buttons = QtWidgets.QHBoxLayout()
        self.nn_start_button = QtWidgets.QPushButton("START NN")
        self.nn_start_button.setObjectName("primary")
        self.nn_start_button.setMinimumHeight(34)
        self.nn_start_button.setToolTip(
            "Start the neural net on the live signal, in its own process.\n\n"
            "It runs alongside the LIVE CLASSIFIER without affecting it: both "
            "can run at once, and whatever this one says, the prediction "
            "above is produced by the scikit-learn model alone.  Ctrl+Shift+N")
        self.nn_start_button.clicked.connect(self.on_nn_clicked)
        buttons.addWidget(self.nn_start_button, 2)

        self.nn_stats_button = QtWidgets.QPushButton("NN STATS")
        self.nn_stats_button.setMinimumHeight(34)
        self.nn_stats_button.setToolTip(
            "This net's own cross-validated numbers, its architecture, every "
            "candidate the sweep tried, and a side-by-side against the "
            "deployed model. Works with the net stopped.")
        self.nn_stats_button.clicked.connect(self.show_neural_stats)
        buttons.addWidget(self.nn_stats_button, 1)
        layout.addLayout(buttons)

        self.nn_train_button = QtWidgets.QPushButton("TRAIN NEURAL NET")
        self.nn_train_button.setMinimumHeight(30)
        self.nn_train_button.setToolTip(
            "Sweep the architectures on EVERY recording on disk and save the "
            "best to its own checkpoint.\n\n"
            "Unlike RETRAIN, this one really does use the GPU when there is "
            "one. It does not touch the deployed model.")
        self.nn_train_button.clicked.connect(self.train_neural_net)
        layout.addWidget(self.nn_train_button)

        self.nn_setup_button = QtWidgets.QPushButton("NEURAL SETUP...")
        self.nn_setup_button.setMinimumHeight(30)
        self.nn_setup_button.setToolTip(
            "Choose which architectures the next neural training compares, "
            "the window lengths, epochs, hidden size, dropout and learning "
            "rate. Used by the FULL SWEEP; QUICK ignores it.")
        self.nn_setup_button.clicked.connect(self.edit_nn_setup)
        layout.addWidget(self.nn_setup_button)
        self.nn_setup_summary = make_label(
            describe_nn_spec(load_nn_spec()[0]), FS_SM, False, COL_TEXT_DIM)
        self.nn_setup_summary.setWordWrap(True)
        layout.addWidget(self.nn_setup_summary)

        #  What the worker is really running on. Not the hardware probe above:
        #  that answers "is a GPU installed", this answers "is the net using
        #  it", and until the worker says so it reads as unknown.
        self.nn_device_label = wrap_label("", FS_SM, True, COL_TEXT_DIM)
        layout.addWidget(self.nn_device_label)

        self.nn_status_label = wrap_label("", FS_SM, False, COL_TEXT_DIM)
        layout.addWidget(self.nn_status_label)

        self.nn_perf_label = wrap_label("", FS_SM, False, COL_TEXT_DIM)
        self.nn_perf_label.setFont(QtGui.QFont("Consolas", 8))
        layout.addWidget(self.nn_perf_label)

        self._nn_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+N"), self)
        self._nn_shortcut.activated.connect(self.toggle_neural)

        self._rebuild_nn_bars()
        self._refresh_neural_panel()
        return group

    def _rebuild_nn_bars(self) -> None:
        """One bar per label THIS net knows -- taken from its own metadata."""
        if not hasattr(self, "nn_prob_box"):
            return
        grid = self.nn_prob_box.layout()
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.nn_prob_bars = {}

        engine = self.nn_engine
        #  The net's own label list, never the predictor's. The two models can
        #  legitimately be trained on different data and disagree about the
        #  label set, and borrowing the other one's list would silently
        #  mislabel every bar.
        labels = list(engine.labels) if (engine and engine.ready) else []
        gone = off_key(labels)
        for row, name in enumerate(labels):
            #  Switched off in the settings: dimmed, and it says so.
            off = name in gone
            caption = make_label(name + ("  (off)" if off else ""), 9, True,
                                 COL_TEXT_DIM if off else cfg.label_color(name))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setEnabled(not off)
            value = make_label("off" if off else "0 %", 9, False, COL_TEXT_DIM)
            value.setFixedWidth(38)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(caption, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(value, row, 2)
            self.nn_prob_bars[name] = (bar, value)
        self.nn_prob_box.setVisible(bool(labels))

    def _refresh_neural_panel(self, force: bool = True) -> None:
        """
        Buttons, badges and the status line for the engine's current state.

        `force=False` from the 100 ms UI tick. The buttons are free to
        recompute, but the status line asks `data_status()`, which reads the
        sidecar off disk -- ten times a second for an answer that changes
        when a sweep finishes. Throttled for the same reason the model
        provenance line is (section 8c).
        """
        engine = self.nn_engine
        if engine is None or not hasattr(self, "nn_start_button"):
            return
        now = time.perf_counter()
        if not force and now - getattr(self, "_nn_status_at", 0.0) < 1.0:
            self._refresh_nn_buttons()
            return
        self._nn_status_at = now

        self._refresh_nn_buttons()
        running = engine.running

        headline, detail, severity = engine.device_badge()
        colour = {"good": COL_GOOD, "warn": COL_WARN}.get(severity,
                                                          COL_TEXT_DIM)
        self.nn_device_label.setText(headline)
        self.nn_device_label.setToolTip(detail)
        self.nn_device_label.setStyleSheet(
            f"color: {colour}; background: transparent;")

        if not engine.trained():
            self.nn_status_label.setText(
                f"No checkpoint at {engine.active_model_path}.\n"
                "Press TRAIN NEURAL NET, or run:  python training/train_nn.py")
            self.nn_status_label.setStyleSheet(
                f"color: {COL_TEXT_DIM}; background: transparent;")
            return

        state_colour = {"ready": COL_GOOD, "error": COL_BAD,
                        "starting": COL_WARN}.get(engine.state, COL_TEXT_DIM)
        text = engine.describe() if engine.ready else engine.status
        entries = zoo.selected_entries("nn")
        if entries and not engine.ready:
            text += ("\nSelected in Settings: "
                     + (entries[0].name if len(entries) == 1 else
                        f"soft vote of {len(entries)} nets: "
                        + " + ".join(e.name for e in entries))
                     + " -- START NN loads it.")
        #  Which gestures the net may answer with: the worker's labels when it
        #  runs, the sidecar's when it does not -- the mask applies either way.
        net_labels = engine.labels or (
            (load_nn_stats(engine.active_model_path) or {}).get("labels") or [])
        mask = describe_off(net_labels)
        if mask:
            text += "\n" + mask

        #  Whether the net is CURRENT is asked of the checkpoint on disk, not
        #  of the running worker, so it is answered whether or not the net is
        #  started -- and stopped is its normal state. Reporting it only while
        #  running meant the one case that matters went unreported: RETRAIN
        #  updates the scikit-learn model and leaves this checkpoint behind,
        #  and nothing said so until somebody happened to press START NN.
        #  Same principle as section 8c: the question has an answer at all
        #  times, or it is reported as unanswerable.
        data = engine.data_status(self._nn_fingerprint())
        if data.get("state") == "stale":
            n_new = data.get("n_new") or 0
            text += "\nOUT OF DATE -- the dataset has changed since this net "
            text += (f"was trained ({n_new} new recording(s)). "
                     if n_new else "was trained. ")
            text += "TRAIN NEURAL NET to catch it up."
            state_colour = COL_WARN
        self.nn_status_label.setText(text)
        self.nn_status_label.setToolTip(data.get("detail", ""))
        self.nn_status_label.setStyleSheet(
            f"color: {state_colour}; background: transparent;")

    def _refresh_nn_buttons(self) -> None:
        """The cheap half of the panel refresh: nothing here touches disk."""
        engine = self.nn_engine
        if engine is None or not hasattr(self, "nn_start_button"):
            return
        running = engine.running
        self.nn_start_button.setText("STOP NN" if running else "START NN")
        self.nn_start_button.setObjectName("danger" if running else "primary")
        self.nn_start_button.setStyleSheet("")
        #  Training and inference would fight over the same GPU, so one at a
        #  time -- the same discipline RETRAIN already applies to the CPU.
        self.nn_start_button.setEnabled(
            engine.trained() and self.nn_training_process is None)
        self.nn_train_button.setEnabled(
            self.nn_training_process is None
            and self.training_process is None
            and not self.sequence.running
            and not running)
        self.nn_train_button.setText(
            "CANCEL NN TRAINING" if self.nn_training_process is not None
            else "TRAIN NEURAL NET")

    def _nn_fingerprint(self):
        """
        The dataset fingerprint, borrowed from the classifier panel's cache.

        Walking the tree again here would double a 30 ms directory scan on a
        GUI timer for an answer that is already in hand (section 8c).
        """
        return getattr(self, "_fingerprint", None)

    # -- control --------------------------------------------------------------

    def toggle_neural(self) -> None:
        engine = self.nn_engine
        if engine is None:
            return
        if engine.running:
            self.stop_neural()
            return
        if not engine.start():
            #  Every refusal already carries its reason; show it rather than
            #  a generic failure, because the common one -- "this wearer has
            #  not calibrated" -- is fixed by pressing a different button.
            self.nn_label.setText("--")
            self._refresh_neural_panel()
            self._append_log(f"Neural net not started: {engine.status}")
            if "calibration" in engine.status.lower():
                QtWidgets.QMessageBox.information(
                    self, "Neural net", engine.status)
            return
        engine.reset()
        self.nn_label.setText("STARTING")
        self.nn_label.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self.nn_conf.setText("loading torch and the checkpoint...")
        self.nn_timer.start(max(30, int(cfg.NN_INTERVAL_MS)))
        self._refresh_neural_panel()
        self._append_log(
            f"Neural net starting ({engine.script}, device "
            f"{engine.device!r}) -- a separate process; the live classifier "
            f"is unaffected.")

    def stop_neural(self) -> None:
        engine = self.nn_engine
        if engine is None:
            return
        was = engine.running
        self.nn_timer.stop()
        engine.stop()
        self.nn_label.setText("--")
        self.nn_label.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: transparent;")
        self.nn_conf.setText("")
        self.nn_agreement.setText("")
        for bar, value in self.nn_prob_bars.values():
            bar.setValue(0)
            value.setText("--")
        self._refresh_neural_panel()
        if was:
            self._append_log("Neural net stopped.")

    def _tick_neural(self) -> None:
        """
        The net's own timer: collect answers, then send one window.

        Collect FIRST. `NN_MAX_IN_FLIGHT` is 1, so sending before reading
        would see the previous request still outstanding and drop every
        second window for no reason.
        """
        engine = self.nn_engine
        if engine is None:
            return
        produced = engine.poll()
        for prediction in produced:
            self._show_nn_prediction(prediction)

        if engine.state in ("error", "stopped"):
            self.nn_timer.stop()
            #  An error can leave the worker alive -- a width mismatch, say,
            #  is the engine refusing a healthy process. Shut it down, or it
            #  keeps a CUDA context for a panel that has stopped listening.
            if engine.state == "error":
                engine.stop_quietly()
            self.nn_label.setText("--" if engine.state == "stopped" else "ERROR")
            self.nn_label.setStyleSheet(
                f"color: {COL_TEXT_DIM if engine.state == 'stopped' else COL_BAD};"
                f" background: transparent;")
            self.nn_conf.setText("")
            self._refresh_neural_panel()
            if engine.state == "error":
                self._append_log(f"Neural net error: {engine.status}")
                for line in engine.worker_output(4):
                    self._append_log(f"  nn: {line}")
            return

        if not engine.ready:
            self._refresh_neural_panel()
            return
        if not self.nn_prob_bars:
            #  The worker has just reported its labels.
            self._rebuild_nn_bars()
            self._refresh_neural_panel()

        rate = self.acq.stats()["rate"] or nominal_rate()
        window_s = engine.window_s or (engine.win_samples / engine.model_fs)
        want = int(window_s * rate * 1.25) + 4
        emg, t = self.acq.ring.snapshot(max(8, want))
        if emg.shape[0] < 4:
            return
        if t.size and (t[-1] - t[0]) > window_s:
            keep = t >= (t[-1] - window_s)
            emg, t = emg[keep], t[keep]
        engine.submit(emg, t)
        self._refresh_nn_perf()

    def _show_nn_prediction(self, prediction: "NeuralPrediction") -> None:
        """
        Draw ONE neural-net answer. Writes only to this panel's own widgets.

        Note what it does not do: it never appends to
        `LivePredictor.history`, never calls `set_prediction_overlay` (the
        graph overlay belongs to the deployed answer, and two labels fighting
        over one corner is exactly the blurring section 8L exists to prevent),
        and never touches `self.prob_bars`.
        """
        if not getattr(prediction, "answered", True):
            #  The mask left nothing to answer with: say so, never guess.
            self.nn_label.setText(NO_ANSWER)
            self.nn_label.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.nn_conf.setText(
                "no answer: the net only sees switched-off gestures")
            self.nn_conf.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.nn_agreement.setText("")
            self.nn_vote_label.setVisible(False)
            for name, (bar, value) in self.nn_prob_bars.items():
                bar.setValue(0)
                value.setText("off" if name in prediction.off else "--")
            self._refresh_nn_perf()
            return
        certain = prediction.certain
        colour = COL_GOOD if certain else COL_WARN
        self.nn_label.setText(prediction.label if certain
                              else f"{prediction.label}?")
        self.nn_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.nn_conf.setText(
            f"{prediction.percent}"
            + ("" if certain else "   (below the confidence threshold)")
            + (f"   ·  {prediction.smoothed_over} window vote"
               if prediction.smoothed_over > 1 else "")
            + kept_note(prediction))
        self.nn_conf.setStyleSheet(f"color: {colour}; background: transparent;")

        for name, (bar, value) in self.nn_prob_bars.items():
            if name in getattr(prediction, "off", ()):
                bar.setValue(0)
                value.setText("off")
                continue
            if name in prediction.labels:
                p = prediction.probabilities[prediction.labels.index(name)]
            else:
                p = float("nan")
            if math.isnan(p):
                bar.setValue(0)
                value.setText("--")
                continue
            bar.setValue(int(np.clip(p, 0, 1) * 1000))
            value.setText(f"{p * 100:.0f} %")
            top = name == prediction.label
            bar.setStyleSheet(
                f"QProgressBar {{ background: #0b0e13; border: 1px solid "
                f"{COL_BORDER}; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: "
                f"{colour if top else COL_TEXT_DIM}; border-radius: 2px; }}")

        from emg_app.vote_view import vote_html
        rows = self.nn_engine.vote_rows(prediction.label)
        self.nn_vote_label.setVisible(bool(rows))
        self.nn_vote_label.setText(
            vote_html(rows, prediction.label, COL_GOOD, COL_WARN, COL_TEXT_DIM)
            if rows else "")

        #  Agreement, only when the classifier is actually running and its
        #  answer is recent. Comparing against a prediction from a minute ago
        #  would invent a disagreement out of two different moments.
        other = self.predictor.last
        fresh = (other is not None
                 and abs(prediction.unix_time - other.unix_time) < 1.5)
        if not fresh:
            self.nn_agreement.setText("")
        elif other.label == prediction.label:
            self.nn_agreement.setText(
                f"agrees with the classifier ({other.label})")
            self.nn_agreement.setStyleSheet(
                f"color: {COL_GOOD}; background: transparent;")
        else:
            self.nn_agreement.setText(
                f"DISAGREES: the classifier says {other.label} "
                f"{other.percent}, the net says {prediction.label} "
                f"{prediction.percent}. The app's answer is the "
                f"classifier's.")
            self.nn_agreement.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
        self._refresh_nn_perf()

    def _refresh_nn_perf(self) -> None:
        """The measured cost, next to the budget it has to fit in."""
        engine = self.nn_engine
        if engine is None or not hasattr(self, "nn_perf_label"):
            return
        summary = engine.stats_summary()
        if math.isnan(summary["latency_ms"]):
            self.nn_perf_label.setText("")
            return
        over = summary["latency_ms"] > cfg.NN_INTERVAL_MS
        self.nn_perf_label.setText(
            f"{summary['latency_ms']:6.1f} ms round trip "
            f"({summary['infer_ms']:.2f} ms forward)  |  budget "
            f"{cfg.NN_INTERVAL_MS} ms  |  {summary['answered']} answered, "
            f"{summary['dropped']} dropped")
        self.nn_perf_label.setStyleSheet(
            f"color: {COL_WARN if over else COL_TEXT_DIM}; "
            f"background: transparent;")

    def show_neural_stats(self) -> None:
        """
        Open the net's own statistics. Not modal, for the same reason MODEL
        STATS is not: you must be able to reach START NN while it is open.
        """
        if self.nn_stats_dialog is not None \
                and self.nn_stats_dialog.isVisible():
            self.nn_stats_dialog.raise_()
            self.nn_stats_dialog.activateWindow()
            return
        dialog = NeuralStatsDialog(self.nn_engine, self,
                                   predictor=self.predictor)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda *_: setattr(self, "nn_stats_dialog",
                                                   None))
        self.nn_stats_dialog = dialog
        dialog.show()

    # -- training -------------------------------------------------------------

    def train_neural_net(self) -> None:
        """
        Sweep the architectures on everything on disk, in a subprocess.

        A subprocess for the same two reasons RETRAIN is one -- it saturates
        the machine, and a crash in training must not take the recorder down
        -- plus a third that is specific to this one: it imports torch and
        attaches a CUDA context, which is exactly what the recorder must not
        do (section 13).
        """
        engine = self.nn_engine
        if engine is None:
            return
        if self.nn_training_process is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Neural training in progress",
                "The neural net is already training. Cancel it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer == QtWidgets.QMessageBox.Yes:
                self._nn_training_cancelled = True
                self.nn_training_process.kill()
            return
        if self.training_process is not None:
            QtWidgets.QMessageBox.information(
                self, "Train neural net",
                "The scikit-learn sweep is still running. One trainer at a "
                "time -- they would fight over the machine.")
            return
        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Train neural net",
                "Finish or stop the recording session first.\n\n"
                "Training saturates the CPU and the GPU and would disturb "
                "the live stream, which could corrupt the recordings in "
                "progress.")
            return

        script = os.path.join(cfg.PROJECT_DIR, "training", "train_nn.py")
        if not os.path.exists(script):
            QtWidgets.QMessageBox.warning(
                self, "Train neural net",
                f"train_nn.py not found at:\n{script}")
            return

        summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        if summary["total_raw"] < 12:
            QtWidgets.QMessageBox.information(
                self, "Train neural net",
                f"Only {summary['total_raw']} recordings on disk. A net "
                "needs more than a handful before it can be validated.")
            return

        info = self.compute_info or {}
        device_line = (
            f"Runs on: GPU ({info.get('gpu')}) -- this trainer really does "
            f"use CUDA, unlike RETRAIN."
            if info.get("cuda") else
            f"Runs on: CPU -- "
            + (str(info.get("reason") or "no usable GPU found")
               if info.get("probed") else "the device probe has not answered "
                                          "yet") + ".")

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Train the neural network")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(f"Train the neural net on all {summary['total_raw']} "
                    f"recordings on disk.")
        box.setInformativeText(
            "This is the SEPARATE model. It writes its own checkpoint and "
            "does not touch the deployed classifier -- after it finishes, "
            "the LIVE CLASSIFIER still predicts with exactly the model it "
            "predicts with now.\n\n"
            f"Quick: the default architecture at one window length, about a "
            f"minute.\n"
            f"Full:  every architecture x window on the same folds, then the "
            f"DC-only control and the per-wearer scores -- roughly 15-25 "
            f"minutes.\n\n"
            f"{device_line}\n\n"
            f"Calibration: {cfg.CALIBRATION_MODE} (the same mode the classifier "
            f"uses, so the two stay comparable).")
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

        #  The worker holds the previous checkpoint open and a CUDA context on
        #  the same card the trainer wants. Stop it first.
        self.stop_neural()
        self.expand_section("neural")
        self.expand_section("training")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel(
            f"train_nn.py  --  {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)"))
        self.training_output = []
        args = self._nn_train_args(script, quick)
        self._launch_training_job("nn", [{"name": "nn",
                                          "argv": [sys.executable] + args}])
        self._after_nn_launch(summary, quick)

    def _nn_train_args(self, script: str, quick: bool) -> List[str]:
        """train_nn.py's command line, exactly as the GUI is configured."""
        engine = self.nn_engine
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        out_dir = os.path.dirname(os.path.abspath(
            engine.full_model_path)) or cfg.PROJECT_DIR
        #  The same paths the APP is configured with, for the same reason
        #  RETRAIN passes them (section 13): otherwise a changed DATA_FOLDER
        #  or NN_MODEL_PATH means the subprocess trains on, and overwrites,
        #  something else.
        args = ["-u", script, "--root", root, "--out-dir", out_dir,
                "--model-path", engine.full_model_path,
                "--calibration", cfg.CALIBRATION_MODE,
                #  "auto" is an app-side convenience; the trainer is given a
                #  real choice, and refuses rather than silently using the
                #  CPU when CUDA was expected.
                "--device", "cuda" if (self.compute_info or {}).get("cuda")
                else "cpu",
                #  The Training tab of the settings: what the net is trained
                #  with, passed explicitly like every other path and flag.
                "--augment", ",".join(cfg.TRAIN_AUGMENT_NN) or "none",
                "--adapt" if cfg.TRAIN_ADAPT_NN else "--no-adapt"]
        #  Settings -> Training -> "Train without these people".
        args += settings_store.person_args(root)
        if quick:
            args.append("--quick")
        else:
            args += nn_spec_args(load_nn_spec()[0])
        return args

    def _after_nn_launch(self, summary: dict, quick: bool) -> None:
        self.training_phase_text = (
            f"neural net: starting on {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)") + "...")
        self.training_progress.setValue(0)
        self.training_progress.setFormat("%p%")
        self.training_progress.setVisible(True)
        self.training_status.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self._refresh_training_status()
        self._refresh_neural_panel()
        self._append_log(
            f"Neural training started on all {summary['total_raw']} "
            f"recordings ({'quick' if quick else 'full sweep'}). The "
            f"deployed classifier is untouched.")

    def _on_nn_training_lines(self, lines: List[str]) -> None:
        """
        Same pump as the sklearn trainer's, same two line formats.

        `train_nn.py` prints `PROGRESS`/`TRAINEVENT` deliberately so that one
        panel and one parser serve both trainers -- the only difference in
        the events is that this one's `device` field can say cuda.
        """
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                self._update_training_progress(line)
                continue
            if line.startswith("TRAINEVENT "):
                try:
                    self._on_train_event(json.loads(line[len("TRAINEVENT "):]))
                except (ValueError, TypeError, KeyError):
                    pass
                continue
            self.training_output.append(line)
            if len(self.training_output) > 4000:
                del self.training_output[:2000]
            if any(key in line for key in (
                    "BEST", "window ", "vote ", "DC only", "subject-",
                    "Saved", "COMPUTE", "recordings", "WARNING", "MIXED")):
                self._append_log(f"nn: {line.strip()}")
            self.training_phase_text = "nn: " + line.strip()[:86]
            self._refresh_training_status()

    def _on_nn_training_finished(self, exit_code: int, _status=None) -> None:
        self.nn_training_process = None
        self.training_progress.setVisible(False)
        cancelled = bool(getattr(self, "_nn_training_cancelled", False))
        self._nn_training_cancelled = False
        took = fmt_duration(time.perf_counter() - self.training_started_at)
        tail = "\n".join(self.training_output[-25:])
        self._refresh_neural_panel()

        if cancelled:
            self._set_train_field("status", "cancelled", COL_WARN)
            self.train_message.setText("Neural training cancelled. The "
                                       "previous checkpoint is untouched.")
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(
                f"neural training cancelled after {took}")
            self._append_log(f"Neural training cancelled after {took}.")
            return

        if exit_code == cfg.TRAIN_EXIT_NEED_MORE_DATA:
            self._set_train_field("status", "not trained", COL_WARN)
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(
                "neural net not trained: fewer than two gesture classes")
            self._append_log("Neural training skipped: a classifier needs at "
                             "least two gestures.")
            return

        if exit_code != 0:
            self._set_train_field("status", "failed", COL_BAD)
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(
                f"neural training failed (exit {exit_code})")
            self._append_log(
                f"Neural training failed with exit code {exit_code}.")
            QtWidgets.QMessageBox.warning(
                self, "Neural training failed",
                f"train_nn.py exited with code {exit_code}.\n\n{tail}")
            return

        stats = (load_nn_stats(self.nn_engine.model_path) or {}).get("stats") \
            or {}
        self.training_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self.training_status.setText(
            f"NN-{stats.get('kind', '?')}: "
            f"{(stats.get('window_accuracy') or 0) * 100:.1f} % per window, "
            f"took {took}")
        self._append_log(
            f"Neural training done: NN-{stats.get('kind')} "
            f"{(stats.get('window_accuracy') or 0) * 100:.1f} % per window "
            f"({stats.get('grouping', '?')}-grouped). The deployed "
            f"classifier is unchanged.")

        control = stats.get("dc_only_control") or {}
        gain = ((stats.get("window_accuracy") or 0)
                - (control.get("window_accuracy") or 0)) * 100
        #  At the end of an ML + NN chain this is the one box for both, so it
        #  carries the classifier's result too.
        chain_note = getattr(self, "_chain_ml_note", "")
        self._chain_ml_note = ""
        QtWidgets.QMessageBox.information(
            self, ("ML + neural training complete" if chain_note
                   else "Neural training complete"),
            (f"{chain_note}\n\n" if chain_note else "")
            + f"New net: NN-{stats.get('kind', '?')} "
            f"({stats.get('n_parameters', '?')} parameters)\n\n"
            f"Per window  : {(stats.get('window_accuracy') or 0) * 100:.1f} %"
            f"  ({stats.get('grouping', '?')}-grouped)\n"
            f"Per gesture : "
            f"{(stats.get('recording_vote_accuracy') or 0) * 100:.1f} %\n"
            f"Baseline    : "
            f"{(stats.get('majority_class_baseline') or 0) * 100:.1f} % "
            f"(majority class)\n"
            + (f"Time buys   : {gain:+.1f} points over the 8 channel means "
               f"alone\n" if control else "")
            + "\nThe deployed LIVE CLASSIFIER is unchanged -- this model is "
              "kept separate and never votes. Press START NN to run it, or "
              "NN STATS for the full breakdown.")
        self._refresh_neural_panel()

    def edit_nn_setup(self) -> None:
        """Open the neural-net setup; affects the next FULL SWEEP only."""
        path = nn_spec_path()
        dialog = NeuralSetupDialog(load_nn_spec(path)[0], path, self)
        dialog.exec()
        if not dialog.saved:
            return
        self.nn_setup_summary.setText(describe_nn_spec(dialog.spec))
        self._append_log(f"Neural setup saved: {describe_nn_spec(dialog.spec)}"
                         f" ({', '.join(dialog.spec['kinds'])}).")

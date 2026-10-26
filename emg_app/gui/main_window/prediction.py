"""
LIVE CLASSIFIER: the deployed model, MODEL STATS, TEST MODEL, calibration
status and the GPU/CPU badge.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import math
import os
import sys
import time

import numpy as np

from emg_app import config as cfg
from emg_app.class_mask import NO_ANSWER, describe_off, kept_note, off_key
from emg_app.gestures import describe_gaps, model_gesture_gaps
from emg_app.calibration import (
    calibration_status, describe_calibration, load_calibration)
from emg_app.compute import (
    COMPUTE_PROBE_SNIPPET, COMPUTE_PROBE_TIMEOUT_S, blank_compute_info,
    compute_badge, parse_compute_probe)
from emg_app.dataset import dataset_fingerprint, describe_training_data
from emg_app.gui.dialogs.calibration import CalibrationDialog
from emg_app.gui.dialogs.model_stats import ModelStatsDialog
from emg_app.gui.dialogs.model_test import ModelTestDialog
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import (
    COL_BAD, COL_BORDER, COL_GOOD, COL_TEXT_DIM, COL_WARN, FS_SM)
from emg_app.gui.widgets import make_label
from emg_app.prediction import latest_model_window


class PredictionMixin:
    """MainWindow -- LIVE CLASSIFIER."""
    def _group_prediction(self):
        group = QtWidgets.QGroupBox("LIVE CLASSIFIER")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        # Mode selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Mode", 10, False, COL_TEXT_DIM))
        self.pred_mode_combo = QtWidgets.QComboBox()
        self.pred_mode_combo.addItem("One shot (stop after each)", "one_shot")
        self.pred_mode_combo.addItem("Continuous", "continuous")
        index = self.pred_mode_combo.findData(cfg.PREDICTION_MODE)
        self.pred_mode_combo.setCurrentIndex(max(0, index))
        self.pred_mode_combo.setToolTip(
            "One shot: one prediction per press, then it waits.\n"
            "Continuous: keeps classifying until you press STOP."
        )
        self.pred_mode_combo.currentIndexChanged.connect(self._on_pred_mode_changed)
        row.addWidget(self.pred_mode_combo, 1)
        layout.addLayout(row)

        # Result
        self.prediction_label = make_label("--", 26, True, COL_TEXT_DIM)
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        self.prediction_conf = make_label("", 13, True, COL_TEXT_DIM)
        self.prediction_conf.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.prediction_conf)

        # Per-class probability bars -- the honest view of how sure it is.
        self.prob_bars = {}
        self.prob_box = QtWidgets.QWidget()
        bars = QtWidgets.QGridLayout(self.prob_box)
        bars.setContentsMargins(0, 2, 0, 2)
        bars.setSpacing(2)
        bars.setColumnStretch(1, 1)
        layout.addWidget(self.prob_box)

        self.prediction_note = make_label("", 10, False, COL_TEXT_DIM)
        self.prediction_note.setWordWrap(True)
        layout.addWidget(self.prediction_note)

        buttons = QtWidgets.QHBoxLayout()
        self.predict_button = QtWidgets.QPushButton("PREDICT")
        self.predict_button.setObjectName("primary")
        self.predict_button.setMinimumHeight(34)
        self.predict_button.clicked.connect(self.toggle_prediction)
        buttons.addWidget(self.predict_button, 2)

        self.stats_button = QtWidgets.QPushButton("MODEL STATS")
        self.stats_button.setMinimumHeight(34)
        self.stats_button.clicked.connect(self.show_model_stats)
        buttons.addWidget(self.stats_button, 1)
        layout.addLayout(buttons)

        self.test_button = QtWidgets.QPushButton("TEST MODEL...")
        self.test_button.setMinimumHeight(30)
        self.test_button.setToolTip(
            "Score the saved model LIVE, on whoever is wearing the band: it "
            "cues each gesture a few times, predicts through the same path "
            "PREDICT uses, and counts hits into its own confusion matrix.\n\n"
            "This is the 'how often is it right for me' number that "
            "cross-validation does not measure. Honest only on a sitting "
            "the model did not train on -- re-don the band first.")
        self.test_button.clicked.connect(self.open_model_test)
        layout.addWidget(self.test_button)

        self.retrain_button = QtWidgets.QPushButton("RETRAIN ON ALL DATA")
        self.retrain_button.setMinimumHeight(30)
        self.retrain_button.setToolTip(
            "Retrain using EVERY recording on disk -- the ones you just made "
            "plus all earlier sessions and people. New data is added to the "
            "old, never trained on by itself.\n\n"
            "Runs in the background: closing the app does not stop it."
        )
        self.retrain_button.clicked.connect(self.retrain_model)
        layout.addWidget(self.retrain_button)

        self.retrain_all_button = QtWidgets.QPushButton(
            "RETRAIN ML + NEURAL NET")
        self.retrain_all_button.setMinimumHeight(30)
        self.retrain_all_button.setToolTip(
            "Retrain the classifier, then the neural net, one after the "
            "other in one background job -- never at the same time, they "
            "would fight over the machine. If the classifier fails, the net "
            "is not started.\n\n"
            "Closing the app does not stop it; reopen to see the progress.")
        self.retrain_all_button.clicked.connect(self.retrain_everything)
        self.retrain_all_button.setVisible(cfg.ENABLE_NEURAL_NET)
        layout.addWidget(self.retrain_all_button)

        self.spec_button = QtWidgets.QPushButton("MODEL SPEC...")
        self.spec_button.setMinimumHeight(30)
        self.spec_button.setToolTip(
            "Choose which classifiers the next retrain sweeps, set their "
            "hyperparameters, and decide whether they also vote as an "
            "ensemble.  Ctrl+Shift+M"
        )
        self.spec_button.clicked.connect(self.edit_model_spec)
        layout.addWidget(self.spec_button)

        self.spec_summary = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.spec_summary.setWordWrap(True)
        layout.addWidget(self.spec_summary)

        #  Per-wearer calibration. Optional, and the button says which of the
        #  two things it would do: measure a first one, or replace the one
        #  that is in force. A calibration never expires on its own.
        if cfg.CALIBRATION_ENABLED:
            self.calibration_button = QtWidgets.QPushButton("CALIBRATE...")
            self.calibration_button.setMinimumHeight(30)
            self.calibration_button.setToolTip(
                "Measure this wearer's rest and strong-contraction levels so "
                "a given number means the same effort on anybody.\n\n"
                "Six seconds, measured once, valid until you press "
                "RECALIBRATE. Nothing recorded is modified, and it is applied "
                "only in the mode you choose -- 'off' leaves the pipeline "
                "exactly as it was.  Ctrl+Shift+C")
            self.calibration_button.clicked.connect(self.open_calibration)
            layout.addWidget(self.calibration_button)

            self.calibration_label = make_label("", FS_SM, True, COL_TEXT_DIM)
            self.calibration_label.setWordWrap(True)
            layout.addWidget(self.calibration_label)

            self._calibration_shortcut = QtGui.QShortcut(
                QtGui.QKeySequence("Ctrl+Shift+C"), self)
            self._calibration_shortcut.activated.connect(self.open_calibration)
            self._refresh_calibration_panel()

        #  Is a model trained, on what, and is it still current? The detail
        #  goes in the tooltip and in MODEL STATS; this is the headline.
        self.model_status_label = make_label("", FS_SM, True, COL_TEXT_DIM)
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        self.model_data_label = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.model_data_label.setWordWrap(True)
        layout.addWidget(self.model_data_label)

        #  Which processor the training actually runs on. Filled in by the
        #  background probe; never guessed.
        self.compute_label = make_label("", FS_SM, True, COL_TEXT_DIM)
        self.compute_label.setWordWrap(True)
        layout.addWidget(self.compute_label)
        self._apply_compute_badge()
        self._refresh_model_status(force=True)

        layout.addWidget(make_label("TRAINING DATA ON DISK", FS_SM, True,
                                    COL_TEXT_DIM))
        self.gesture_counts_label = QtWidgets.QLabel()
        self.gesture_counts_label.setWordWrap(True)
        self.gesture_counts_label.setTextFormat(QtCore.Qt.RichText)
        self.gesture_counts_label.setToolTip(
            "Raw recordings per gesture across the whole dataset -- what the "
            "trainer will actually see. A gesture in grey has none yet.")
        layout.addWidget(self.gesture_counts_label)

        self._spec_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+M"), self)
        self._spec_shortcut.activated.connect(self.edit_model_spec)

        self.training_progress = QtWidgets.QProgressBar()
        self.training_progress.setRange(0, 1000)
        self.training_progress.setTextVisible(True)
        self.training_progress.setFormat("%p%")
        self.training_progress.setFixedHeight(14)
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)

        self.training_status = make_label("", 10, False, COL_TEXT_DIM)
        self.training_status.setWordWrap(True)
        layout.addWidget(self.training_status)

        self._rebuild_prob_bars()
        self._refresh_prediction_panel()
        return group

    def _rebuild_prob_bars(self) -> None:
        """One labelled bar per gesture the loaded model knows about."""
        grid = self.prob_box.layout()
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.prob_bars = {}

        labels = self.predictor.labels if self.predictor.ready else []
        gone = off_key(labels)
        for row, name in enumerate(labels):
            #  The caption wears the gesture's own colour (cfg.label_color), so
            #  a gesture added later is recognisable at a glance; the BAR keeps
            #  the certainty colour set in _show_prediction. A gesture switched
            #  off in the settings is dimmed and says so instead of 0 %.
            off = name in gone
            caption = make_label(name + ("  (off)" if off else ""), 9, True,
                                 COL_TEXT_DIM if off else cfg.label_color(name))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setEnabled(not off)
            value = make_label("off" if off else "0 %", 9, False, COL_TEXT_DIM)
            value.setFixedWidth(38)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(caption, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(value, row, 2)
            self.prob_bars[name] = (bar, value)
        self.prob_box.setVisible(bool(labels))

    # ======================================================================
    # Live prediction
    # ======================================================================

    # -- panel state ----------------------------------------------------------

    def _refresh_prediction_panel(self) -> None:
        ready = self.predictor.ready
        self.predict_button.setEnabled(ready)
        self.pred_mode_combo.setEnabled(ready)
        if hasattr(self, "test_button"):
            self.test_button.setEnabled(ready)
        # The model just changed state, so the provenance line is stale.
        self._refresh_model_status(force=True)

        if ready:
            mask = describe_off(self.predictor.labels)
            self.prediction_note.setText(
                self.predictor.describe() + (f"\n{mask}" if mask else ""))
            return

        self.prediction_label.setText("--")
        self.prediction_label.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: transparent;")
        self.prediction_conf.setText("")
        reasons = {
            "disabled in the configuration":
                "Disabled. Set ENABLE_LIVE_PREDICTION = True in the config.",
            "no model file":
                f"No model at {self.predictor.path}.\n"
                "Collect data, then run:  python training/train_model.py",
            "joblib is not installed":
                "joblib is missing:  python -m pip install joblib",
        }
        self.prediction_note.setText(
            reasons.get(self.predictor.status,
                        f"Unavailable: {self.predictor.status}"))

    def _refresh_classifier_mask(self) -> None:
        """
        The mask changed (settings, or a restart): redraw everything that
        shows what the classifiers may answer with. Both models, and nothing
        is retrained -- the next window is simply scored under the new mask.
        """
        if hasattr(self, "prob_box"):
            self._rebuild_prob_bars()
        if getattr(self, "nn_engine", None) is not None:
            self._rebuild_nn_bars()
            self._refresh_neural_panel()
        self._refresh_prediction_panel()

    def _on_pred_mode_changed(self) -> None:
        self._set_global("PREDICTION_MODE", self.prediction_mode)
        self.stop_prediction()

    @property
    def prediction_mode(self) -> str:
        return self.pred_mode_combo.currentData() or "one_shot"

    # -- control --------------------------------------------------------------

    def toggle_prediction(self) -> None:
        if not self.predictor.ready:
            return
        if self.prediction_active or self.one_shot_deadline:
            self.stop_prediction()
        elif self.prediction_mode == "continuous":
            self.start_continuous_prediction()
        else:
            self.start_one_shot_prediction()

    def start_continuous_prediction(self) -> None:
        self.predictor.reset()
        self.prediction_active = True
        self.one_shot_deadline = 0.0
        self.predict_button.setText("STOP")
        self.predict_button.setObjectName("danger")
        self.predict_button.setStyleSheet("")
        self.pred_timer.start(max(30, int(cfg.PREDICTION_INTERVAL_MS)))
        self._append_log("Continuous prediction started.")

    def start_one_shot_prediction(self) -> None:
        """
        One prediction, then stop.

        The window is taken AFTER a short delay so you have time to get into
        the gesture -- classifying the instant the button is pressed would
        just classify whatever you were doing while reaching for the mouse.
        """
        self.predictor.reset()
        self.prediction_active = False
        now = time.perf_counter()
        # Two phases: a grace period to get into the gesture, then the window
        # that is actually classified. The gesture must be held through BOTH.
        self.one_shot_capture_at = now + cfg.ONE_SHOT_DELAY_S
        self.one_shot_deadline = (self.one_shot_capture_at
                                  + self.predictor.window_seconds())
        self.predict_button.setText("CANCEL")
        self.prediction_label.setText("DO THE GESTURE")
        self.prediction_label.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self.prediction_conf.setText("and hold it...")
        self.plot.set_prediction_overlay(
            "DO THE GESTURE NOW", COL_WARN, "hold it until the result appears")
        self.pred_timer.start(30)
        self._append_log(
            f"One-shot armed: hold the gesture for "
            f"{cfg.ONE_SHOT_DELAY_S + self.predictor.window_seconds():.1f} s.")

    def stop_prediction(self) -> None:
        was = self.prediction_active or bool(self.one_shot_deadline)
        self.prediction_active = False
        self.one_shot_deadline = 0.0
        self.one_shot_capture_at = 0.0
        self.pred_timer.stop()
        self.predict_button.setText("PREDICT")
        self.predict_button.setObjectName("primary")
        self.predict_button.setStyleSheet("")
        self.plot.set_prediction_overlay("")
        if was:
            self._append_log("Prediction stopped.")

    # -- the prediction tick --------------------------------------------------

    def _tick_prediction(self) -> None:
        if not self.predictor.ready:
            self.stop_prediction()
            self._refresh_prediction_panel()
            return

        # One-shot: wait out the grace period and the capture window, then
        # fire exactly once. The two phases are shown separately so it is
        # never ambiguous whether the gesture should be happening yet.
        if self.one_shot_deadline:
            now = time.perf_counter()
            remaining = self.one_shot_deadline - now
            if remaining > 0:
                if now < self.one_shot_capture_at:
                    self.prediction_label.setText("DO THE GESTURE")
                    self.prediction_conf.setText(
                        f"hold it -- recording the signal in "
                        f"{self.one_shot_capture_at - now:.1f} s")
                else:
                    self.prediction_label.setText("HOLD...")
                    self.prediction_conf.setText(
                        f"reading the signal now ({remaining:.1f} s left)")
                    self.plot.set_prediction_overlay(
                        "READING THE SIGNAL", COL_BAD, "keep holding")
                return
            self.one_shot_deadline = 0.0
            self.one_shot_capture_at = 0.0
            prediction = self._run_prediction()
            self.pred_timer.stop()
            self.predict_button.setText("PREDICT")
            if prediction is None:
                self.prediction_label.setText("NO DATA")
                self.prediction_label.setStyleSheet(
                    f"color: {COL_BAD}; background: transparent;")
                self.prediction_conf.setText("not enough samples")
                self.plot.set_prediction_overlay("")
            else:
                self.one_shot_clear_at = time.perf_counter() + cfg.ONE_SHOT_HOLD_S
            return

        if not self.prediction_active:
            return
        self._run_prediction()

    def _run_prediction(self):
        """Pull the newest window out of the ring and classify it."""
        emg, t = latest_model_window(self.acq, self.predictor.window_seconds())
        if emg.shape[0] < 4:
            return None

        prediction = self.predictor.predict(emg, t)
        if prediction is None:
            self._refresh_prediction_panel()
            self.stop_prediction()
            return None
        self._show_prediction(prediction)
        return prediction

    def _show_prediction(self, prediction) -> None:
        #  The mask can leave nothing to answer with (the model put all of its
        #  probability on switched-off gestures). Say that, not a guess.
        if not getattr(prediction, "answered", True):
            self.prediction_label.setText(NO_ANSWER)
            self.prediction_label.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.prediction_conf.setText(
                "no answer: the model only sees switched-off gestures")
            self.prediction_conf.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.plot.set_prediction_overlay(
                NO_ANSWER, COL_WARN, "only switched-off gestures seen")
            #  The bars would otherwise keep the previous window's numbers.
            for name, (bar, value) in self.prob_bars.items():
                bar.setValue(0)
                value.setText("off" if name in prediction.off else "--")
            return
        certain = prediction.certain
        colour = COL_GOOD if certain else COL_WARN
        self.prediction_label.setText(prediction.label if certain
                                      else f"{prediction.label}?")
        self.prediction_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.prediction_conf.setText(
            f"{prediction.percent}"
            + ("" if certain else "   (below the confidence threshold)")
            + (f"   ·  {prediction.smoothed_over} window vote"
               if prediction.smoothed_over > 1 else "")
            + kept_note(prediction))
        self.prediction_conf.setStyleSheet(
            f"color: {colour}; background: transparent;")

        for name, (bar, value) in self.prob_bars.items():
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

        self.plot.set_prediction_overlay(
            f"{prediction.label}  {prediction.percent}", colour,
            "" if certain else "low confidence")

    def open_model_test(self) -> None:
        """TEST MODEL: a live, cued score of the saved model on this wearer."""
        if not self.predictor.ready:
            QtWidgets.QMessageBox.information(
                self, "Test the model",
                f"No usable model: {self.predictor.status}")
            return
        dialog = getattr(self, "model_test_dialog", None)
        if dialog is None:
            dialog = ModelTestDialog(self)
            self.model_test_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def show_model_stats(self) -> None:
        """
        Open MODEL STATS, without blocking the window behind it.

        Deliberately NOT modal. The DECISION TRAIL view follows the live
        classifier, and a modal dialog would mean you could watch it only
        when PREDICT happened to be running already -- you could not reach
        the button to start it. Reusing one instance keeps a second copy from
        polling the panel on its own timer.
        """
        if self.stats_dialog is not None and self.stats_dialog.isVisible():
            self.stats_dialog.raise_()
            self.stats_dialog.activateWindow()
            return
        self.decision_panel.load()
        dialog = ModelStatsDialog(self.predictor, self,
                                  panel=self.decision_panel)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(self._on_stats_closed)
        self.stats_dialog = dialog
        dialog.show()

    def _on_stats_closed(self, _result=None) -> None:
        self.stats_dialog = None
        self._rebuild_prob_bars()
        self._refresh_prediction_panel()

    # -- per-wearer calibration ----------------------------------------------

    def open_calibration(self) -> None:
        """
        Measure or replace the current wearer's calibration.

        A calibration belongs to the person in the SUBJECT panel, so it is
        measured for `self.person_id` and stored in that person's folder. It
        stays in force until this dialog is used again -- there is no expiry
        and nothing re-estimates it per recording.

        Changing the calibration or its mode does NOT change the loaded
        model, which was fitted on whatever scale was in force when it was
        trained. That is recorded inside the payload, so the mismatch is
        reported rather than silently applied.
        """
        if not cfg.CALIBRATION_ENABLED:
            return
        if self.sequence.running or self.free_capture is not None:
            self.status.showMessage(
                "stop the recording first -- calibrating reads the live "
                "signal", 6000)
            return

        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        before = load_calibration(root, self.person_id)
        before_mode = cfg.CALIBRATION_MODE
        dialog = CalibrationDialog(self, self.acq, self.person_id, root,
                                   cfg.CALIBRATION_MODE)
        dialog.exec()

        if dialog.mode != before_mode:
            self._set_global("CALIBRATION_MODE", dialog.mode)
            self._append_log(f"Calibration mode: {before_mode} -> "
                             f"{dialog.mode}.")

        if dialog.saved:
            after = load_calibration(root, self.person_id)
            if after is None:
                self._append_log(
                    f"Calibration cleared for {self.person_id} "
                    f"(kept in {cfg.CALIBRATION_LOG_NAME}).")
            else:
                self._append_log(
                    f"Calibration {'replaced' if before else 'measured'} for "
                    f"{self.person_id}: "
                    f"{describe_calibration(after, cfg.CALIBRATION_MODE)}")

        self._refresh_calibration_panel()
        if dialog.mode != before_mode or dialog.saved:
            #  The payload records which scale it was fitted on, so a change
            #  here can make the loaded model unusable. Reload it so the
            #  panel says that now instead of at the next prediction.
            self.predictor.load()
            self._rebuild_prob_bars()
            self._refresh_prediction_panel()
            self._refresh_model_status(force=True)
            if (self.predictor.calibration_mode != cfg.CALIBRATION_MODE
                    and self.predictor.model is not None):
                self._append_log(
                    f"  the loaded model was trained with calibration "
                    f"{self.predictor.calibration_mode!r} -- retrain to use "
                    f"{cfg.CALIBRATION_MODE!r}.")
            #  The net is fitted on a scale too, and it is running in another
            #  process that already resolved this wearer's calibration. A new
            #  calibration mid-run would leave it predicting on the old
            #  numbers, so it is stopped rather than left wrong; START NN
            #  re-resolves from the new record.
            if self.nn_engine is not None and self.nn_engine.running:
                self._append_log(
                    "  the neural net was running on the previous "
                    "calibration -- stopped. Press START NN to restart it on "
                    "the new one.")
                self.stop_neural()
            self._refresh_neural_panel()

    def _refresh_calibration_panel(self) -> None:
        """One line: is this wearer calibrated, and is it being applied?"""
        if not cfg.CALIBRATION_ENABLED or not hasattr(self, "calibration_label"):
            return
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        info = calibration_status(root, self.person_id, cfg.CALIBRATION_MODE)
        colour = {"active": COL_GOOD, "stored": COL_WARN,
                  "invalid": COL_BAD, "none": COL_TEXT_DIM,
                  "disabled": COL_TEXT_DIM}.get(info["state"], COL_TEXT_DIM)
        headline = {
            "active": "CALIBRATED",
            "stored": "CALIBRATED -- not applied",
            "invalid": "CALIBRATION REFUSED",
            "none": "NOT CALIBRATED",
            "disabled": "CALIBRATION OFF",
        }[info["state"]]
        self.calibration_label.setText(f"{headline}  ({info['summary']})")
        self.calibration_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.calibration_button.setText(
            "RECALIBRATE..." if info["record"] is not None else "CALIBRATE...")
        tip = info["summary"]
        if info["problems"]:
            tip += "\n\n" + "\n".join(info["problems"])
        self.calibration_label.setToolTip(tip)

    # -- model status ---------------------------------------------------------

    def _refresh_model_status(self, force: bool = False) -> None:
        """
        Show whether a model exists, what it was trained on, and whether the
        dataset has moved on since.

        The fingerprint is a directory walk (~30 ms for 119 recordings), so
        it is throttled rather than recomputed on every UI tick. `force` is
        for the moments when the answer definitely changed: a model loading,
        a recording being saved, a retrain finishing.
        """
        if not hasattr(self, "model_status_label"):
            return
        now = time.perf_counter()
        due = now - getattr(self, "_fingerprint_at", 0.0) > \
            getattr(self, "_fingerprint_every", 5.0)
        # Never walk the tree while samples are being captured. The walk is
        # on the GUI thread, and a recording in progress is the one moment
        # the interface must not stutter.
        if due and self.acq.capturing:
            due = False
        if force or due:
            t0 = time.perf_counter()
            try:
                self._fingerprint = dataset_fingerprint(
                    os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
            except OSError:
                self._fingerprint = None
            took = time.perf_counter() - t0
            # Back off on a big dataset rather than hitching the UI every
            # five seconds: 30 ms at 119 recordings, but this grows linearly
            # and the answer changes rarely.
            self._fingerprint_every = 5.0 if took < 0.15 else min(300.0,
                                                                  took * 400)
            self._fingerprint_at = now

        status = self.predictor.data_status(getattr(self, "_fingerprint", None))
        colour = {"good": COL_GOOD, "warn": COL_WARN,
                  "bad": COL_BAD}.get(status["severity"], COL_TEXT_DIM)
        self.model_status_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.model_status_label.setText(f"MODEL   {status['headline']}")
        self.model_status_label.setToolTip(status["detail"])

        #  The second line is the provenance itself -- what data went in --
        #  because that is the question that gets asked weeks later.
        payload = dict(self.predictor.payload_meta) if self.predictor.ready else {}
        data = payload.get("data") or {}
        #  A gesture with recordings that this model was never trained on
        #  (added in GESTURES... and recorded since). Said here, never acted
        #  on: retraining is the operator's decision.
        on_disk = (getattr(self, "_fingerprint", None) or {}).get("labels") or {}
        gaps = (describe_gaps(model_gesture_gaps(self.predictor.labels,
                                                 on_disk), "MODEL")
                if self.predictor.ready else "")
        if self.predictor.ready and data:
            self.model_data_label.setText(
                f"trained on  {describe_training_data(data)}"
                + (f"\n{gaps}" if gaps else ""))
        elif self.predictor.ready:
            self.model_data_label.setText(
                "trained on  (not recorded by this model)")
        else:
            on_disk = getattr(self, "_fingerprint", None) or {}
            self.model_data_label.setText(
                f"on disk  {describe_training_data(on_disk)}"
                if on_disk.get("n_recordings") else "no recordings on disk yet")
        self.model_data_label.setToolTip(status["detail"])

    # -- compute device badge -------------------------------------------------

    def _apply_compute_badge(self) -> None:
        """Paint the GPU/CPU badge from whatever the probe knows so far."""
        if not hasattr(self, "compute_label"):
            return
        head, detail, severity = compute_badge(self.compute_info)
        colour = {"good": COL_GOOD, "warn": COL_WARN}.get(severity,
                                                          COL_TEXT_DIM)
        self.compute_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.compute_label.setText(f"COMPUTE   {head}")
        self.compute_label.setToolTip(detail)

        if getattr(self, "status_compute", None) is not None:
            if not self.compute_info.get("probed"):
                short = ("GPU/CPU UNKNOWN" if self.compute_info.get("failed")
                         else "GPU/CPU ?")
            elif self.compute_info.get("cuda"):
                short = f"GPU  {self.compute_info.get('gpu') or 'CUDA'}"
            else:
                short = "CPU ONLY"
            self.status_compute.setText(short)
            self.status_compute.setStyleSheet(
                f"color: {colour}; padding: 0 10px;")
            self.status_compute.setToolTip(f"{head}\n{detail}")

    def start_compute_probe(self) -> None:
        """
        Ask a subprocess whether CUDA is usable, and badge the answer.

        A subprocess because importing torch here would take seconds and
        attach a GPU context to the process that must not drop EMG samples
        (CLAUDE.md section 13). Asynchronous because that import is slow
        enough -- measured ~7 s on this machine -- to visibly freeze the
        window if it ran inline.
        """
        if self.compute_process is not None:
            return
        self.compute_process = QtCore.QProcess(self)
        self.compute_process.setProgram(sys.executable)
        self.compute_process.setArguments(["-c", COMPUTE_PROBE_SNIPPET])
        self.compute_process.setProcessChannelMode(
            QtCore.QProcess.MergedChannels)
        #  Both signals carry their own process, for the same reason the
        #  timeout below does: a killed probe's `finished` arrives
        #  asynchronously, so it can land after a replacement probe has
        #  already started and would otherwise be read as that one answering
        #  with empty output. Same family of bug as the stale timer.
        probe_ref = self.compute_process
        self.compute_process.finished.connect(
            lambda *_: self._on_compute_probe_finished(probe_ref))
        self.compute_process.errorOccurred.connect(
            lambda *_: self._on_compute_probe_error(probe_ref))
        self.compute_process.start()
        # A probe that never answers would leave the badge saying "checking"
        # for the rest of the session. Say so instead.
        #
        # The timer is bound to THIS process, which is not a detail: a
        # singleShot cannot be cancelled, so a probe that answered in 8 s
        # still has a timer due to fire at 30 s. Without the identity check
        # that stale timer kills whatever probe happens to be running when it
        # lands -- so a second probe started 25 s after the first one
        # finished was killed 5 s in and reported "did not answer", with a
        # working GPU on the machine. Found by tests/test_gui.py once enough
        # checks ran before the badge section to cross the 30 s mark.
        probe = self.compute_process
        QtCore.QTimer.singleShot(
            int(COMPUTE_PROBE_TIMEOUT_S * 1000),
            lambda: self._compute_probe_timeout(probe))
        self._apply_compute_badge()

    def _compute_probe_timeout(self, probe=None) -> None:
        if self.compute_process is None:
            return                      # it answered in time
        if probe is not None and probe is not self.compute_process:
            return                      # a stale timer from an earlier probe
        try:
            self.compute_process.kill()
        except RuntimeError:
            pass
        self._finish_compute_probe(
            blank_compute_info("the device probe did not answer in "
                               f"{COMPUTE_PROBE_TIMEOUT_S:.0f} s",
                               failed=True))

    def _on_compute_probe_finished(self, probe=None) -> None:
        if probe is not None and probe is not self.compute_process:
            return              # a late signal from a probe we replaced
        text = ""
        if self.compute_process is not None:
            try:
                text = bytes(
                    self.compute_process.readAllStandardOutput()).decode(
                        "utf-8", errors="replace")
            except RuntimeError:
                text = ""
        self._finish_compute_probe(parse_compute_probe(text))

    def _on_compute_probe_error(self, probe=None) -> None:
        if probe is not None and probe is not self.compute_process:
            return              # likewise: the error belongs to an old probe
        self._finish_compute_probe(
            blank_compute_info("the device probe could not be started",
                               failed=True))

    def _finish_compute_probe(self, info: dict) -> None:
        if self.compute_process is not None:
            try:
                self.compute_process.deleteLater()
            except RuntimeError:
                # Both finished and errorOccurred can land here, and during
                # interpreter teardown the C++ side may already be gone. The
                # answer still matters; the handle does not.
                pass
            self.compute_process = None
        self.compute_info = info
        self._apply_compute_badge()
        head, _, _ = compute_badge(info)
        self._append_log(f"Compute device: {head}.")

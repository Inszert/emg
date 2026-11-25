"""
The subject / plan panels and the cued recording session: start, pause,
review (ACCEPT / REDO) and capture handling.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import os
import time

from emg_app import config as cfg
from emg_app.acquisition import Capture
from emg_app.gui.dialogs.gesture_setup import GestureSetupDialog
from emg_app.class_mask import validate_off
from emg_app.gestures import save_gesture_setup
from emg_app.neural import load_nn_stats
from emg_app.gui.dialogs.quality import QualityDialog
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import COL_TEXT, COL_TEXT_DIM
from emg_app.gui.widgets import make_label
from emg_app.recording import (
    DatasetWriter, Recording, STAGE_RAW, STAGE_TRIMMED)
from emg_app.sequence import STATE_RECORDING, STATE_REVIEW, build_plan
from emg_app.utils import fmt_duration, iso_time, safe_slug


class SessionMixin:
    """MainWindow -- The subject / plan panels and the cued recording session."""
    def _group_subject(self):
        group = QtWidgets.QGroupBox("SUBJECT / SESSION")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setSpacing(6)

        self.person_edit = QtWidgets.QLineEdit(self.person_id)
        self.person_edit.setToolTip(
            "Use a NEW id for every new subject. Data is kept in separate "
            "folders per person so that leave-one-person-out validation works."
        )
        self.session_edit = QtWidgets.QLineEdit(self.session_id)
        self.session_edit.setToolTip(
            "Start a new session every time the armband is taken off and put "
            "back on -- electrode positions shift."
        )
        self.hand_combo = QtWidgets.QComboBox()
        self.hand_combo.addItems(["right", "left"])
        self.hand_combo.setCurrentText(cfg.HANDEDNESS)
        self.arm_combo = QtWidgets.QComboBox()
        self.arm_combo.addItems(["right", "left"])
        self.arm_combo.setCurrentText(cfg.ARM_WORN_ON)
        self.device_edit = QtWidgets.QLineEdit(cfg.DEVICE_NAME)
        self.notes_edit = QtWidgets.QLineEdit(cfg.SUBJECT_NOTES)
        self.notes_edit.setPlaceholderText("notes stored with every recording")

        form.addRow("Person id", self.person_edit)
        form.addRow("Session id", self.session_edit)
        form.addRow("Handedness", self.hand_combo)
        form.addRow("Armband on", self.arm_combo)
        form.addRow("Device", self.device_edit)
        form.addRow("Notes", self.notes_edit)

        self.folder_label = make_label("", 10, False, COL_TEXT_DIM)
        self.folder_label.setWordWrap(True)
        form.addRow("Folder", self.folder_label)

        self.person_edit.editingFinished.connect(self._on_identity_changed)
        self.session_edit.editingFinished.connect(self._on_identity_changed)
        for widget in (self.hand_combo, self.arm_combo):
            widget.currentTextChanged.connect(self._on_metadata_changed)
        self.device_edit.editingFinished.connect(self._on_metadata_changed)
        self.notes_edit.editingFinished.connect(self._on_metadata_changed)

        return group

    def _group_plan(self):
        group = QtWidgets.QGroupBox("RECORDING PLAN")
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(6)

        grid.addWidget(make_label("Repetitions", 10, False, COL_TEXT_DIM), 0, 0)
        self.reps_spin = QtWidgets.QSpinBox()
        self.reps_spin.setRange(1, 999)
        self.reps_spin.setValue(cfg.REPETITIONS)
        grid.addWidget(self.reps_spin, 0, 1)

        grid.addWidget(make_label("Order", 10, False, COL_TEXT_DIM), 0, 2)
        self.order_combo = QtWidgets.QComboBox()
        self.order_combo.addItems(["interleaved", "blocked", "random"])
        self.order_combo.setCurrentText(cfg.GESTURE_ORDER)
        #  All three described, not just the default. "blocked" is the one
        #  people actually ask for -- five of the same gesture in a row
        #  instead of being bounced between gestures -- and it was
        #  undiscoverable while the tooltip only explained "interleaved".
        self.order_combo.setToolTip(
            "blocked: every repetition of one gesture back to back, then the "
            "next gesture. Easiest for the subject -- you hold one pose and "
            "repeat it.\n\n"
            "interleaved: one round of every gesture at a time. Spreads "
            "fatigue and electrode drift evenly across the label set instead "
            "of baking it into whichever gesture came last.\n\n"
            "random: fully shuffled. Best against order effects.\n\n"
            "Blocked is the comfortable choice; interleaved is the honest "
            "one, because in blocked order each gesture's data all comes "
            "from one stretch of time."
        )
        grid.addWidget(self.order_combo, 0, 3)

        #  How long each repetition is held. In blocked order this is the
        #  number that decides whether five reps in a row is comfortable or
        #  exhausting, so it belongs on screen rather than in the config.
        grid.addWidget(make_label("Hold each rep", 10, False, COL_TEXT_DIM),
                       1, 0)
        self.hold_spin = QtWidgets.QDoubleSpinBox()
        self.hold_spin.setRange(0.5, 30.0)
        self.hold_spin.setSingleStep(0.5)
        self.hold_spin.setDecimals(1)
        self.hold_spin.setSuffix(" s")
        self.hold_spin.setValue(cfg.RECORD_TIME)
        self.hold_spin.setToolTip(
            "Length of the labelled RECORD window for each repetition.\n\n"
            "Shorter is less tiring and drifts less, which matters most in "
            "blocked order; longer gives more windows per repetition. At "
            "2.5 s a repetition yields about 7 windows at the model's 0.3 s "
            "/ 0.5 overlap setting.\n\n"
            "PRE/POST_GESTURE_MS of extra signal is captured around it "
            "either way, so an early or late reaction is still in the raw "
            "file."
        )
        grid.addWidget(self.hold_spin, 1, 1)

        grid.addWidget(make_label("Manual gesture", 10, False, COL_TEXT_DIM),
                       1, 2)
        self.gesture_combo = QtWidgets.QComboBox()
        self.gesture_combo.addItems(cfg.GESTURES)
        self.gesture_combo.setToolTip("Label used by the manual RECORD button")
        grid.addWidget(self.gesture_combo, 1, 3)

        #  Which gestures this session records at all. The optional ones ship
        #  switched off, so this is how they get switched on.
        self.gestures_button = QtWidgets.QPushButton("GESTURES...")
        self.gestures_button.setToolTip(
            "Choose which gestures this session records, switch on the "
            "optional ones (PINCH, THUMB_OVER_INDEX, POINT), or add your "
            "own.  Ctrl+Shift+G")
        self.gestures_button.clicked.connect(self.edit_gestures)
        grid.addWidget(self.gestures_button, 4, 0, 1, 4)

        self._gestures_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+G"), self)
        self._gestures_shortcut.activated.connect(self.edit_gestures)

        self.plan_label = make_label("", 10, False, COL_TEXT_DIM)
        self.plan_label.setWordWrap(True)
        grid.addWidget(self.plan_label, 2, 0, 1, 4)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        grid.addWidget(self.progress_bar, 3, 0, 1, 4)

        self.next_label = make_label("", 11, True, COL_TEXT)
        grid.addWidget(self.next_label, 4, 0, 1, 4)

        self.reps_spin.valueChanged.connect(self._refresh_plan_info)
        self.order_combo.currentTextChanged.connect(self._refresh_plan_info)
        self.hold_spin.valueChanged.connect(self._refresh_plan_info)
        return group

    def _set_global(self, name: str, value) -> None:
        """
        Mirror a GUI field into the module-level config.

        emg_app.config is the single source of truth for what gets written
        into session_config.json and every *_meta.json, so an edit in the GUI
        has to land there too -- otherwise the stored metadata would lie.
        """
        setattr(cfg, name, value)

    def _on_identity_changed(self) -> None:
        if self.sequence.running:
            return
        person = safe_slug(self.person_edit.text(), "person")
        session = safe_slug(self.session_edit.text(), "session")
        if (person, session) == (self.person_id, self.session_id):
            self._refresh_plan_info()
            return

        self.person_id, self.session_id = person, session
        self.person_edit.setText(person)
        self.session_edit.setText(session)
        self._set_global("PERSON_ID", person)
        self._set_global("SESSION_ID", session)
        self.writer = DatasetWriter(
            os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER), person, session
        )
        existing = self.writer.count_existing()
        self._append_log(
            f"Subject set to {person} / {session} "
            f"({existing} recording(s) already on disk)."
        )
        # Markers belong to a person/session pair, so switching subject has to
        # swap the marker log too -- otherwise notes would be drawn over, and
        # cut from, somebody else's session. Takes are per-session as well.
        self._load_session_markers()
        self._refresh_take_history()
        self._refresh_plan_info()
        # A calibration belongs to the PERSON, so switching subject switches
        # which calibration is in force -- and can make the loaded model
        # unusable, if it was trained calibrated and this wearer has none.
        self._refresh_calibration_panel()
        if cfg.CALIBRATION_ENABLED and self.predictor.calibration_mode != "off":
            self.predictor.load()
            self._refresh_prediction_panel()

    def edit_gestures(self) -> None:
        """
        Choose which gestures this session records.

        Refuses while a session is running, for the same reason the subject
        fields do: the plan is already built and half-walked, and changing the
        label set under it would leave the remaining repetitions cueing
        gestures the plan does not contain.
        """
        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Gestures",
                "Finish or stop the session first -- the plan is already "
                "built from the current gesture list.")
            return

        dialog = GestureSetupDialog(self, selected=list(cfg.GESTURES),
                                    model_labels=self._model_label_sets(),
                                    classifier_off=list(cfg.CLASSIFIER_OFF))
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._apply_gesture_setup(dialog.selected(), dialog.custom_gestures(),
                                  dialog.classifier_off())

    def _apply_gesture_setup(self, chosen, custom, off) -> None:
        """
        Make an edited gesture set the live one: the plan, every label combo,
        the classifier mask -- and save it. Shared by GESTURES... and the
        settings dialog, so the two cannot disagree about what applying means.

        Switching a gesture off or on in the CLASSIFIER retrains nothing; the
        masks live in the models' output (emg_app/class_mask.py). Only a
        gesture no model was trained on needs a retrain, and that is said,
        never done.
        """
        known = self._model_label_sets()
        off, problems = validate_off(off, known)
        for line in problems:
            self._append_log(line)
        off_before = list(cfg.CLASSIFIER_OFF)
        if not chosen or (chosen == list(cfg.GESTURES)
                          and custom == dict(cfg.CUSTOM_GESTURES)
                          and sorted(off) == sorted(off_before)):
            return

        before = list(cfg.GESTURES)
        self._set_global("CUSTOM_GESTURES", custom)
        self._set_global("GESTURES", chosen)
        self._set_global("CLASSIFIER_OFF", list(off))
        #  Saved with the data so it survives a restart (app.main() restores
        #  it). A failed write is reported, never fatal: the set still holds
        #  for this run.
        try:
            path = save_gesture_setup(chosen, custom, classifier_off=off)
            self._append_log(f"Gesture set saved to {path}")
        except OSError as exc:
            self._append_log(f"Gesture set NOT saved ({exc}); it holds for "
                             "this run only.")
        added = [g for g in chosen if g not in before]
        dropped = [g for g in before if g not in chosen]

        #  Every place a label is offered has to follow, or the combos would
        #  still be handing out labels that are no longer in the plan.
        for combo in (getattr(self, "gesture_combo", None),
                      getattr(self, "label_gesture_combo", None),
                      getattr(self, "take_gesture_combo", None)):
            if combo is None:
                continue
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(chosen)
            if keep in chosen:
                combo.setCurrentText(keep)
            combo.blockSignals(False)

        self._refresh_plan_info()
        self._refresh_gesture_counts()
        self._refresh_label_counts()
        #  The classifier's bars are built from the MODEL's labels, not from
        #  the recording list, so adding a gesture to the plan leaves them
        #  alone -- a model trained on six classes still predicts six. What
        #  DOES change them is the mask.
        parts = []
        if added:
            parts.append("added " + ", ".join(added))
        if dropped:
            parts.append("no longer cued: " + ", ".join(dropped))
        if parts:
            self._append_log("Gestures: " + "; ".join(parts) + "."
                             + (" Recordings already on disk are untouched."
                                if dropped else ""))
        switched_off = [g for g in off if g not in off_before]
        switched_on = [g for g in off_before if g not in off]
        if switched_off or switched_on:
            bits = []
            if switched_off:
                bits.append("OFF: " + ", ".join(switched_off))
            if switched_on:
                bits.append("back ON: " + ", ".join(switched_on))
            self._append_log(
                "Classifier (ML and neural net) -- " + "; ".join(bits)
                + ". Nothing was retrained; it takes effect on the next "
                "window.")
            self.status.showMessage(
                "Classifier: " + "; ".join(bits) + " (no retrain needed)",
                8000)
        self._refresh_classifier_mask()
        #  NOTHING is retrained here. Say plainly which models do not know
        #  the new gestures, so the retrain is a decision, not a surprise.
        unknown = [g for g in added
                   if known and any(g not in labels
                                    for labels in known.values())]
        if unknown:
            self._append_log(
                "Not in the trained models yet: " + ", ".join(unknown)
                + ". Record them (and calibrate -- the sweep now includes "
                "them), then RETRAIN when you decide. Nothing retrains by "
                "itself.")
        if added:
            self.status.showMessage(
                "Added %s -- record it, then RETRAIN when you choose."
                % ", ".join(added), 10000)
        self._refresh_model_status(force=True)

    def _model_label_sets(self) -> dict:
        """What each trained model knows: {"ML model": [...], "neural net": [...]}."""
        out = {}
        if getattr(self, "predictor", None) is not None and self.predictor.ready:
            out["ML model"] = list(self.predictor.labels)
        engine = getattr(self, "nn_engine", None)
        sidecar = (load_nn_stats(engine.active_model_path) or {}) if engine else {}
        if sidecar.get("labels"):
            out["neural net"] = list(sidecar["labels"])
        return out

    def _on_metadata_changed(self) -> None:
        self._set_global("HANDEDNESS", self.hand_combo.currentText())
        self._set_global("ARM_WORN_ON", self.arm_combo.currentText())
        self._set_global("DEVICE_NAME", self.device_edit.text())
        self._set_global("SUBJECT_NOTES", self.notes_edit.text())

    def _refresh_plan_info(self) -> None:
        self._set_global("REPETITIONS", int(self.reps_spin.value()))
        self._set_global("GESTURE_ORDER", self.order_combo.currentText())
        #  Mirrored like the rest: the duration goes into every recording's
        #  config snapshot, so an edit here has to reach the global or the
        #  stored metadata would claim a hold length that never happened.
        if hasattr(self, "hold_spin"):
            self._set_global("RECORD_TIME", float(self.hold_spin.value()))
        plan = build_plan()
        per_item = (cfg.REST_TIME + cfg.PREPARE_TIME + cfg.COUNTDOWN_TIME + cfg.RECORD_TIME
                    + cfg.POST_GESTURE_MS / 1000.0 + cfg.PAUSE_TIME)
        self.plan_label.setText(
            f"{len(cfg.GESTURES)} gestures x {self.reps_spin.value()} reps = "
            f"{len(plan)} recordings,  about {fmt_duration(len(plan) * per_item)} "
            f"({per_item:.1f} s each).  Gestures: {', '.join(cfg.GESTURES)}"
        )
        self.folder_label.setText(
            os.path.relpath(self.writer.session_dir, cfg.PROJECT_DIR)
            + f"   ({self.writer.count_existing()} on disk)"
        )
        self.progress_bar.setMaximum(max(1, len(plan)))

    # ======================================================================
    # Session control
    # ======================================================================

    def start_session(self) -> None:
        if not self.acq.connected:
            answer = QtWidgets.QMessageBox.question(
                self, "Myo not connected",
                f"The armband is currently {self.acq.state}.\n\n"
                "Start the session anyway? Recordings made while the link is "
                "down will be flagged as incomplete.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        self._on_identity_changed()
        self._on_metadata_changed()
        self._refresh_plan_info()

        self.writer.write_session_config({
            "started_iso": iso_time(time.time()),
            "plan_length": len(build_plan()),
        })

        self.expand_section("plan")
        self.sequence.start(build_plan())
        self._append_log(
            f"Session started: {len(self.sequence.plan)} recordings "
            f"({self.order_combo.currentText()})."
        )

    def toggle_pause(self) -> None:
        if self.sequence.paused:
            self.sequence.resume()
            self._append_log("Session resumed.")
        else:
            self.sequence.pause()
            self._append_log("Session paused.")

    def stop_session(self) -> None:
        if self.record_button.isChecked():
            self.record_button.setChecked(False)
        self.sequence.stop()
        self.acq.stop_capture()
        self._append_log("Session stopped.")

    def skip_current(self) -> None:
        item = self.sequence.current
        self.sequence.skip()
        if item:
            self._append_log(f"Skipped {item[0]} rep {item[1]}.")

    def toggle_manual_record(self, checked: bool) -> None:
        if checked:
            label = self.gesture_combo.currentText()
            repetition = self.writer.next_repetition(label)
            self.sequence.start_manual(label, repetition)
            self.record_button.setText("STOP RECORDING")
            self._append_log(f"Manual recording: {label} rep {repetition}.")
        else:
            self.sequence.stop_manual()
            self.record_button.setText("RECORD (manual)")

    # ======================================================================
    # Capture callbacks (called from the sequence controller, GUI thread)
    # ======================================================================

    def _on_sequence_state(self, state: str, previous: str) -> None:
        if state == STATE_RECORDING:
            self.acq.vibrate(1)
        elif previous == STATE_RECORDING:
            self.acq.vibrate(1)

        if previous == STATE_REVIEW and state != STATE_REVIEW:
            self.plot.exit_review()
            self.review_recording = None
            self.review_info.setText("")

        # A manual recording that has been reviewed leaves the button stuck in
        # its "STOP RECORDING" state otherwise.
        if previous == STATE_RECORDING and state != STATE_RECORDING \
                and self.record_button.isChecked():
            self.record_button.setChecked(False)
            self.record_button.setText("RECORD (manual)")

    # ======================================================================
    # Review step
    # ======================================================================

    def _begin_review(self, rec: Recording) -> None:
        """
        Freeze the graph on the take that just finished and wait for a verdict.

        The live stream keeps running into the ring buffer behind this, so
        nothing is lost while you look.
        """
        self.review_recording = rec
        bounds = self.plot.review(rec, rec.activity)
        self._update_review_info(*bounds)
        # Focus the button so Space/Enter accept, without stealing those keys
        # from the text fields the rest of the time.
        self.accept_button.setFocus(QtCore.Qt.OtherFocusReason)

    def _update_review_info(self, start_s: float = None,
                            end_s: float = None) -> None:
        rec = self.review_recording
        if rec is None:
            self.review_info.setText("")
            return

        start, end = self.plot.trim_indices(rec.n_samples)
        kept = end - start
        fraction = kept / max(1, rec.n_samples) * 100
        issues = ", ".join(sorted({i.code for i in rec.issues})) or "ok"
        self.review_info.setText(
            f"{rec.label} rep {rec.repetition}:  keeping {kept} of "
            f"{rec.n_samples} samples ({fraction:.0f} %), "
            f"{kept / max(rec.fs, 1.0):.2f} s   |   quality: {issues}"
        )

    def accept_review(self) -> None:
        """Keep the take, write the trimmed copy, go to the next repetition."""
        if not self.sequence.reviewing:
            return
        rec = self.review_recording

        if rec is not None and cfg.REVIEW_SAVE_TRIM_ON_ACCEPT:
            start, end = self.plot.trim_indices(rec.n_samples)
            # Writing a "trimmed" file identical to the raw one is just noise.
            if end - start >= 2 and (start > 0 or end < rec.n_samples):
                trimmed = rec.slice(start, end, STAGE_TRIMMED, {
                    "trim_method": "review",
                    "trim_reviewed_by_operator": True,
                })
                try:
                    self.writer.save(trimmed, STAGE_TRIMMED)
                    self._mark_table_trimmed(rec)
                    self._append_log(
                        f"{rec.label} rep {rec.repetition}: accepted, trimmed "
                        f"to {trimmed.n_samples} samples."
                    )
                except OSError as exc:
                    self._append_log(f"could not write the trimmed copy: {exc}")
                    QtWidgets.QMessageBox.warning(
                        self, "Trim",
                        f"Could not write the trimmed copy:\n\n{exc}\n\n"
                        "The raw recording is safe on disk.",
                    )
            else:
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: accepted, kept in full."
                )
        elif rec is not None:
            self._append_log(f"{rec.label} rep {rec.repetition}: accepted.")

        self.sequence.accept_review()

    def redo_review(self) -> None:
        """Reject the take, quarantine its files, record the same one again."""
        if not self.sequence.reviewing:
            return
        rec = self.review_recording

        if rec is not None:
            if cfg.REVIEW_QUARANTINE_ON_REDO:
                moved = self._quarantine(rec)
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: REDO -- "
                    f"{moved} file(s) moved to _deleted."
                )
            else:
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: REDO -- "
                    "rejected take kept on disk."
                )
            self._remove_recording(rec)

        self.sequence.redo_review()
        self._refresh_plan_info()

    def _on_sequence_finished(self) -> None:
        self._append_log(f"Session complete: {len(self.recordings)} recordings.")
        QtWidgets.QMessageBox.information(
            self, "Session complete",
            f"Collected {len(self.recordings)} recordings for "
            f"{self.person_id} / {self.session_id}.\n\n"
            f"{self.writer.session_dir}",
        )

    def _on_capture_start(self, label: str, repetition: int) -> Capture:
        return self.acq.start_capture(label, repetition, meta={
            "emg_mode": cfg.EMG_MODE,
            "handedness": cfg.HANDEDNESS,
            "arm_worn_on": cfg.ARM_WORN_ON,
            "device": cfg.DEVICE_NAME,
            "notes": self.notes_edit.text(),
            "plan_index": self.sequence.index,
            "plan_length": len(self.sequence.plan),
            "gesture_order": cfg.GESTURE_ORDER,
        })

    def _on_capture_discard(self, capture: Capture, label: str,
                            repetition: int) -> None:
        """An incomplete recording, abandoned because the link dropped."""
        self.acq.stop_capture()
        self._append_log(
            f"{label} rep {repetition}: discarded after {len(capture)} "
            "sample(s) -- link lost. The repetition will be retried."
        )

    def _on_capture_stop(self, capture: Capture, label: str,
                         repetition: int) -> None:
        self.acq.stop_capture()

        rec = capture.to_recording(self.person_id, self.session_id, {
            "myo_state_at_stop": self.acq.state,
        })
        if rec.n_samples < 2:
            self._append_log(
                f"{label} rep {repetition}: no samples captured -- nothing saved."
            )
            return

        expected = cfg.RECORD_TIME + (cfg.PRE_GESTURE_MS + cfg.POST_GESTURE_MS) / 1000.0
        rec.run_quality_checks(expected_seconds=expected,
                               had_disconnect=capture.had_disconnect)
        if cfg.AUTO_TRIM:
            rec.detect_activity()

        trimmed = None
        if cfg.AUTO_SAVE:
            try:
                self.writer.save(rec, STAGE_RAW)
                # With the review step on, the trimmed copy is written when the
                # operator accepts (using the band they may have adjusted), so
                # writing one here too would only create a duplicate.
                if cfg.AUTO_TRIM and cfg.AUTO_TRIM_WRITE_FILE and not cfg.REVIEW_EACH_RECORDING:
                    trimmed = rec.auto_trimmed()
                    if trimmed is not None:
                        self.writer.save(trimmed, STAGE_TRIMMED)
            except OSError as exc:
                self._append_log(f"SAVE FAILED for {label} rep {repetition}: {exc}")
                QtWidgets.QMessageBox.critical(
                    self, "Save failed",
                    f"Could not write the recording:\n\n{exc}\n\n"
                    "The samples are still in memory -- fix the problem and "
                    "use TRIM to re-save, or stop the session.",
                )

        self.recordings.append(rec)
        self._add_table_row(rec, trimmed is not None)

        level = rec.worst_issue_level
        self._append_log(
            f"{label} rep {repetition}: {rec.n_samples} samples, "
            f"{rec.duration:.2f} s"
            + (f", quality: {level}" if level else ", quality: ok")
        )
        self._refresh_plan_info()

        if cfg.REVIEW_EACH_RECORDING:
            # The review screen shows the quality findings inline, so a modal
            # on top of it would just be in the way.
            self._begin_review(rec)
        elif level == "error" and cfg.QC_PAUSE_ON in ("error", "warning"):
            self._handle_bad_recording(rec)
        elif level == "warning" and cfg.QC_PAUSE_ON == "warning":
            self._handle_bad_recording(rec)

    def _handle_bad_recording(self, rec: Recording) -> None:
        was_running = self.sequence.running and not self.sequence.paused
        if was_running:
            self.sequence.pause()

        dialog = QualityDialog(rec, self)
        dialog.exec()

        if dialog.repeat_requested:
            # Re-queue the same gesture at the current position.
            self.sequence.plan.insert(self.sequence.index, (rec.label, rec.repetition))
            self._append_log(f"Re-queued {rec.label} rep {rec.repetition}.")

        if was_running:
            self.sequence.resume()

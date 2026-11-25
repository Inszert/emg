"""
LABEL ON GRAPH: select a span of the live graph and save it as a labelled
recording.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import numpy as np

from emg_app import config as cfg
from emg_app.gui.qt import QtGui, QtWidgets
from emg_app.gui.tokens import COL_BAD, COL_GOOD, COL_TEXT_DIM
from emg_app.gui.widgets import make_label
from emg_app.markers import markers_between
from emg_app.recording import Recording, STAGE_RAW


class LabelingMixin:
    """MainWindow -- LABEL ON GRAPH."""
    def _group_label_on_graph(self):
        """
        Free recording: perform a gesture whenever you like, then box it on
        the graph and label it. Nothing enters the dataset until you select
        it, so hesitation and false starts never get recorded.
        """
        group = QtWidgets.QGroupBox("LABEL ON GRAPH  (free recording)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.label_mode_button = QtWidgets.QPushButton("START LABEL MODE")
        self.label_mode_button.setCheckable(True)
        self.label_mode_button.setMinimumHeight(32)
        self.label_mode_button.setToolTip(
            "Stream continuously and label by hand. No countdown: do the "
            "gesture whenever you like, then drag a box around it."
        )
        self.label_mode_button.toggled.connect(self.toggle_label_mode)
        layout.addWidget(self.label_mode_button)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Gesture", 10, False, COL_TEXT_DIM))
        self.label_gesture_combo = QtWidgets.QComboBox()
        self.label_gesture_combo.addItems(cfg.GESTURES)
        self.label_gesture_combo.setToolTip(
            "What the selection will be labelled as. Set this BEFORE you do "
            "the gesture, so you only have to press ADD afterwards."
        )
        row.addWidget(self.label_gesture_combo, 1)
        layout.addLayout(row)

        # The default cut length. One value, used by SELECT LAST n s here and
        # by the cut dialog that carves clips out of a take, mirrored into
        # LABEL_DEFAULT_SELECTION_S so it lands in every *_meta.json too.
        row_len = QtWidgets.QHBoxLayout()
        row_len.addWidget(make_label("Default cut (s)", 10, False, COL_TEXT_DIM))
        self.cut_length_spin = QtWidgets.QDoubleSpinBox()
        self.cut_length_spin.setDecimals(2)
        self.cut_length_spin.setSingleStep(0.25)
        self.cut_length_spin.setRange(cfg.LABEL_MIN_SELECTION_S,
                                      max(cfg.LABEL_MIN_SELECTION_S,
                                          cfg.RING_BUFFER_SECONDS))
        self.cut_length_spin.setValue(float(cfg.LABEL_DEFAULT_SELECTION_S))
        self.cut_length_spin.setToolTip(
            "How long a fresh selection is, in seconds. Applies to every cut "
            "from now on -- here and in the cut-clips dialog -- instead of "
            "the fixed 2 s."
        )
        self.cut_length_spin.valueChanged.connect(self._on_cut_length_changed)
        row_len.addWidget(self.cut_length_spin, 1)
        layout.addLayout(row_len)

        self.select_button = QtWidgets.QPushButton(
            f"SELECT LAST {cfg.LABEL_DEFAULT_SELECTION_S:.2f} s")
        self.select_button.setMinimumHeight(30)
        self.select_button.setToolTip(
            "Drop a selection box over the last few seconds, then drag "
            "its edges to fit the gesture exactly."
        )
        self.select_button.clicked.connect(self.arm_selection)
        layout.addWidget(self.select_button)

        self.selection_info = make_label("", 10, False, COL_TEXT_DIM)
        self.selection_info.setWordWrap(True)
        layout.addWidget(self.selection_info)

        self.add_label_button = QtWidgets.QPushButton("ADD SELECTION")
        self.add_label_button.setObjectName("primary")
        self.add_label_button.setMinimumHeight(34)
        self.add_label_button.setToolTip(
            "Save the boxed signal as a labelled recording.  Ctrl+Space"
        )
        self.add_label_button.clicked.connect(self.add_selection_as_label)
        layout.addWidget(self.add_label_button)

        row2 = QtWidgets.QHBoxLayout()
        self.undo_label_button = QtWidgets.QPushButton("UNDO LAST")
        self.undo_label_button.setObjectName("danger")
        self.undo_label_button.setToolTip(
            "Remove the last labelled mark. Its files move to _deleted, "
            "never erased."
        )
        self.undo_label_button.clicked.connect(self.undo_last_label)
        row2.addWidget(self.undo_label_button)

        self.label_counts = make_label("", 10, False, COL_GOOD)
        self.label_counts.setWordWrap(True)
        row2.addWidget(self.label_counts, 1)
        layout.addLayout(row2)

        self._accept_label_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Space"), self)
        self._accept_label_shortcut.activated.connect(self.add_selection_as_label)

        return group

    # ======================================================================
    # LABEL ON GRAPH
    # ======================================================================

    def toggle_label_mode(self, enabled: bool) -> None:
        if enabled and self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Label on graph",
                "Stop the cued session first -- the two recording styles "
                "would fight over the same labels.")
            self.label_mode_button.setChecked(False)
            return

        if enabled:
            self.expand_section("label")
        self.plot.set_label_mode(enabled)
        self.label_mode_button.setText(
            "STOP LABEL MODE" if enabled else "START LABEL MODE")
        if enabled:
            self._append_log(
                "Label mode on: do a gesture, then SELECT and ADD it.")
        else:
            self.plot.clear_selection()
            self.selection_info.setText("")
        self._refresh_label_counts()

    def arm_selection(self) -> None:
        if not self.label_mode_button.isChecked():
            self.label_mode_button.setChecked(True)
        _, t_now = self.acq.ring.latest()
        if not t_now:
            self.selection_info.setText("no signal yet")
            return
        self.plot.arm_selection(t_now, cfg.LABEL_DEFAULT_SELECTION_S)
        self._on_selection_changed(*self.plot.selection_bounds())

    def _on_selection_changed(self, start_abs: float, end_abs: float) -> None:
        emg, _ = self._slice_ring(start_abs, end_abs)
        duration = max(0.0, end_abs - start_abs)
        self.selection_info.setText(
            f"selection {duration:.2f} s, {emg.shape[0]} samples"
            + ("   -- too short" if duration < cfg.LABEL_MIN_SELECTION_S else ""))
        self.selection_info.setStyleSheet(
            f"color: {COL_BAD if duration < cfg.LABEL_MIN_SELECTION_S else COL_TEXT_DIM};"
            " background: transparent;")

    def _slice_ring(self, start_abs: float, end_abs: float):
        """Pull the samples between two absolute times out of the ring."""
        emg, t = self.acq.ring.snapshot()
        if t.size == 0:
            return np.zeros((0, cfg.N_CHANNELS), np.int32), np.zeros(0, np.float64)
        mask = (t >= start_abs) & (t <= end_abs)
        return emg[mask], t[mask]

    def _write_span_as_recording(self, start_abs: float, end_abs: float,
                                 label: str, capture_style: str,
                                 extra_meta: dict = None, dialog_title: str = "Add"):
        """
        Cut a span out of the ring buffer and write it as a labelled
        recording. Shared by ADD SELECTION and CUT SINCE LAST MARKER so the
        two cannot drift apart -- a recording written by one route has to be
        indistinguishable from one written by the other.

        Returns the Recording on success, None if it was rejected or failed.
        """
        emg, t = self._slice_ring(start_abs, end_abs)

        if (end_abs - start_abs) < cfg.LABEL_MIN_SELECTION_S or emg.shape[0] < 4:
            QtWidgets.QMessageBox.information(
                self, dialog_title,
                f"The span is too short ({end_abs - start_abs:.2f} s, "
                f"{emg.shape[0]} samples).\n\nMake it wider, or lower "
                "LABEL_MIN_SELECTION_S.")
            return None

        meta = {
            "emg_mode": cfg.EMG_MODE,
            "handedness": cfg.HANDEDNESS,
            "arm_worn_on": cfg.ARM_WORN_ON,
            "device": cfg.DEVICE_NAME,
            "notes": self.notes_edit.text(),
            "capture_style": capture_style,
            "selected_by_operator": True,
            "selection_seconds": round(end_abs - start_abs, 4),
        }
        meta.update(extra_meta or {})

        # Any marker dropped inside this span is part of the story of the
        # recording, so it travels with it in the metadata.
        inside = markers_between(self.session_markers, start_abs, end_abs)
        if inside:
            meta["markers"] = [m.to_dict() for m in inside]

        repetition = self.writer.next_repetition(label)
        rec = Recording(
            emg=emg, timestamps=t, label=label,
            person_id=self.person_id, session_id=self.session_id,
            repetition=repetition, stage=STAGE_RAW,
            # The whole span IS the gesture: the operator drew the
            # boundaries, so there is no pre/post roll to exclude.
            label_start_index=0, label_end_index=emg.shape[0],
            meta=meta,
        )
        rec.run_quality_checks(expected_seconds=end_abs - start_abs)
        if cfg.AUTO_TRIM:
            rec.detect_activity()

        try:
            self.writer.save(rec, STAGE_RAW)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, dialog_title, f"Could not write the recording:\n{exc}")
            return None

        self.recordings.append(rec)
        self.expand_section("recordings")
        self._add_table_row(rec, False)
        self.plot.add_mark(start_abs, end_abs, label, rec.uid)
        self._append_log(
            f"Labelled {label} rep {repetition}: {rec.n_samples} samples, "
            f"{rec.duration:.2f} s"
            + (f", {len(inside)} marker(s) inside" if inside else "")
            + (f" (quality: {rec.worst_issue_level})"
               if rec.worst_issue_level else ""))
        self.acq.vibrate(1)
        self._refresh_label_counts()
        self._refresh_gesture_counts()
        self._refresh_plan_info()
        return rec

    def add_selection_as_label(self) -> None:
        """Turn the boxed region into a labelled recording on disk."""
        bounds = self.plot.selection_bounds()
        if bounds is None:
            self.selection_info.setText("nothing selected -- press SELECT first")
            return

        start_abs, end_abs = bounds
        label = self.label_gesture_combo.currentText()
        rec = self._write_span_as_recording(
            start_abs, end_abs, label, "label_on_graph",
            dialog_title="Add selection")
        if rec is None:
            return

        self.plot.clear_selection()
        self.selection_info.setText(
            f"added {label}: {rec.n_samples} samples, {rec.duration:.2f} s")
        self.selection_info.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")

    # ======================================================================
    # Free take  (record the whole stretch now, cut clips out of it later)
    # ======================================================================

    def _on_cut_length_changed(self, value: float) -> None:
        """
        The default cut length, changed in the GUI.

        Mirrored into the module global so it reaches *_meta.json via
        recording_config_snapshot(), and the button is relabelled so what it
        will do is readable without hovering it.
        """
        self._set_global("LABEL_DEFAULT_SELECTION_S", float(value))
        self.select_button.setText(f"SELECT LAST {value:.2f} s")

    def undo_last_label(self) -> None:
        mark = self.plot.remove_last_mark()
        if mark is None:
            return
        rec = next((r for r in reversed(self.recordings)
                    if r.uid == mark.get("uid")), None)
        if rec is None:
            self._refresh_label_counts()
            return
        moved = self._quarantine(rec)
        self._remove_recording(rec)
        self._append_log(
            f"Removed {rec.label} rep {rec.repetition} "
            f"({moved} file(s) moved to _deleted).")
        self._refresh_label_counts()
        self._refresh_plan_info()

"""
Markers and notes on the live graph, and CUT SINCE LAST MARKER.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import time
from datetime import datetime

from emg_app import config as cfg
from emg_app.gui.qt import QtGui, QtWidgets
from emg_app.gui.tokens import COL_BAD, COL_GOOD, COL_TEXT_DIM
from emg_app.gui.widgets import make_label
from emg_app.markers import Marker


class MarkersMixin:
    """MainWindow -- Markers and notes on the live graph, and CUT SINCE LAST MARKER."""
    def _group_markers(self):
        """
        Timestamped notes dropped onto the live stream, and cutting the span
        between the newest one and now into a training recording.
        """
        group = QtWidgets.QGroupBox("MARKERS & NOTES")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.marker_text = QtWidgets.QLineEdit()
        self.marker_text.setPlaceholderText(
            "what just happened? e.g. electrode slipped")
        self.marker_text.setToolTip(
            "Typed once, dropped with ADD NOTE. The text is drawn on the "
            "graph at the moment you pressed the button and written to "
            "markers.jsonl in the session folder."
        )
        self.marker_text.returnPressed.connect(self.add_note_marker)
        layout.addWidget(self.marker_text)

        row = QtWidgets.QHBoxLayout()
        self.add_note_button = QtWidgets.QPushButton("ADD NOTE")
        self.add_note_button.setObjectName("primary")
        self.add_note_button.setMinimumHeight(30)
        self.add_note_button.setToolTip(
            "Drop a note at this instant.  Ctrl+M")
        self.add_note_button.clicked.connect(self.add_note_marker)
        row.addWidget(self.add_note_button)

        self.add_break_button = QtWidgets.QPushButton("BREAK")
        self.add_break_button.setObjectName("warn")
        self.add_break_button.setMinimumHeight(30)
        self.add_break_button.setToolTip(
            "Drop a separator: a dashed line that visibly splits what came "
            "before from what comes next.  Ctrl+B")
        self.add_break_button.clicked.connect(self.add_break_marker)
        row.addWidget(self.add_break_button)
        layout.addLayout(row)

        self.marker_cut_check = QtWidgets.QCheckBox("allow cutting at markers")
        self.marker_cut_check.setChecked(bool(cfg.MARKER_CUT_ENABLED))
        self.marker_cut_check.setToolTip(
            "When on, CUT SINCE LAST MARKER turns the span between the "
            "newest marker and now into a labelled training recording, using "
            "the gesture chosen in LABEL ON GRAPH above."
        )
        self.marker_cut_check.toggled.connect(self._on_marker_cut_toggled)
        layout.addWidget(self.marker_cut_check)

        self.cut_marker_button = QtWidgets.QPushButton("CUT SINCE LAST MARKER")
        self.cut_marker_button.setMinimumHeight(30)
        self.cut_marker_button.setToolTip(
            "Save everything between the newest marker and now as training "
            "data.  Ctrl+Shift+Space")
        self.cut_marker_button.clicked.connect(self.cut_since_last_marker)
        layout.addWidget(self.cut_marker_button)

        self.marker_status = make_label("no markers yet", 10, False, COL_TEXT_DIM)
        self.marker_status.setWordWrap(True)
        layout.addWidget(self.marker_status)

        self.marker_list = QtWidgets.QListWidget()
        self.marker_list.setMaximumHeight(96)
        self.marker_list.setToolTip(
            "Markers logged this session, newest last. The log is "
            "append-only: a marker is never edited or deleted, a correction "
            "is simply another marker."
        )
        layout.addWidget(self.marker_list)

        self._note_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+M"), self)
        self._note_shortcut.activated.connect(self.add_note_marker)
        self._break_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        self._break_shortcut.activated.connect(self.add_break_marker)
        self._cut_marker_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+Space"), self)
        self._cut_marker_shortcut.activated.connect(self.cut_since_last_marker)

        self._sync_marker_controls()
        return group

    # ======================================================================
    # Markers / notes
    # ======================================================================

    def _marker_now(self) -> float:
        """
        The timestamp a marker should carry: the newest sample's time, so the
        marker lines up with the signal rather than with the GUI clock. Falls
        back to wall time when nothing is streaming yet.
        """
        _, t_now = self.acq.ring.latest()
        return float(t_now) if t_now else time.time()

    def _sync_marker_controls(self) -> None:
        if not hasattr(self, "cut_marker_button"):
            return
        allowed = bool(self.marker_cut_check.isChecked())
        have = bool(self.session_markers)
        self.cut_marker_button.setEnabled(allowed and have)

    def _on_marker_cut_toggled(self, checked: bool) -> None:
        self._set_global("MARKER_CUT_ENABLED", bool(checked))
        self._sync_marker_controls()
        self._append_log(
            f"Cutting at markers {'enabled' if checked else 'disabled'}.")

    def add_note_marker(self) -> None:
        self._add_marker("note")

    def add_break_marker(self) -> None:
        self._add_marker("break")

    def _add_marker(self, kind: str) -> None:
        """Log a marker and draw it. Nothing already on disk is touched."""
        text = self.marker_text.text().strip()
        if kind == "note" and not text:
            self.marker_status.setText("type the note first, then ADD NOTE")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.marker_text.setFocus()
            return

        marker = Marker(
            t_unix=self._marker_now(), text=text, kind=kind,
            person_id=self.person_id, session_id=self.session_id,
        )
        try:
            self.writer.append_marker(marker)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Add marker", f"Could not write the marker:\n{exc}")
            return

        self.session_markers.append(marker)
        self.plot.add_marker(marker.t_unix, marker.text, marker.kind, marker.uid)
        self.marker_text.clear()

        shown = marker.short_text(40)
        self.marker_status.setText(
            f"{'break' if marker.is_break else 'note'} at "
            f"{datetime.fromtimestamp(marker.t_unix).strftime('%H:%M:%S')}: {shown}")
        self.marker_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self._refresh_marker_list()
        self._redraw_markers_now()
        self._append_log(
            f"Marker ({marker.kind}) at "
            f"{datetime.fromtimestamp(marker.t_unix).strftime('%H:%M:%S')}: "
            f"{marker.text or '(no text)'}")
        self.acq.vibrate(1)
        self._sync_marker_controls()

    def _redraw_markers_now(self) -> None:
        """
        Draw the markers straight away instead of waiting for the next frame.

        update_data() is what normally redraws them, and it returns early when
        the graph is frozen, in review mode, or -- the case that actually bit
        -- when no samples are arriving because the armband is not connected.
        Dropping a note and seeing nothing happen reads as the note being
        lost, so this paints it on the spot.
        """
        try:
            self.plot._draw_markers(self.plot._t_now or time.time())
        except Exception:                                    # pragma: no cover
            pass            # drawing a caption must never break recording

    def _refresh_marker_list(self) -> None:
        """Newest last, capped -- the file keeps all of them regardless."""
        self.marker_list.clear()
        for marker in self.session_markers[-40:]:
            stamp = datetime.fromtimestamp(marker.t_unix).strftime("%H:%M:%S")
            prefix = "|" if marker.is_break else "-"
            item = QtWidgets.QListWidgetItem(
                f"{prefix} {stamp}  {marker.short_text(44)}")
            item.setForeground(QtGui.QColor(
                cfg.MARKER_COLORS.get(marker.kind, cfg.MARKER_COLORS["note"])))
            item.setToolTip(marker.text or "(no text)")
            self.marker_list.addItem(item)
        self.marker_list.scrollToBottom()

    def _load_session_markers(self) -> None:
        """
        Pick up markers already logged for this person/session, so restarting
        the tool mid-session does not lose the context. Ones older than the
        ring buffer simply sit off the left edge of the graph.
        """
        try:
            self.session_markers = self.writer.load_markers()
        except OSError:                                      # pragma: no cover
            self.session_markers = []
        self.plot.clear_markers()
        for marker in self.session_markers:
            self.plot.add_marker(marker.t_unix, marker.text,
                                 marker.kind, marker.uid)
        if hasattr(self, "marker_list"):
            self._refresh_marker_list()
            self._redraw_markers_now()
            if self.session_markers:
                self.marker_status.setText(
                    f"{len(self.session_markers)} marker(s) already logged "
                    "for this session")
                self.marker_status.setStyleSheet(
                    f"color: {COL_TEXT_DIM}; background: transparent;")
        self._sync_marker_controls()

    def cut_since_last_marker(self) -> None:
        """Save the span between the newest marker and now as training data."""
        if not self.marker_cut_check.isChecked():
            self.marker_status.setText(
                "cutting at markers is switched off -- tick the box above")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            return
        if not self.session_markers:
            self.marker_status.setText("no marker to cut from -- add one first")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            return

        marker = self.session_markers[-1]
        start_abs = marker.t_unix
        end_abs = self._marker_now()
        label = self.label_gesture_combo.currentText()

        rec = self._write_span_as_recording(
            start_abs, end_abs, label, "cut_at_marker",
            extra_meta={
                "cut_from_marker_uid": marker.uid,
                "cut_from_marker_text": marker.text,
                "cut_from_marker_kind": marker.kind,
            },
            dialog_title="Cut since last marker",
        )
        if rec is None:
            return

        self.marker_status.setText(
            f"cut {label}: {rec.duration:.2f} s since "
            f"{marker.short_text(24)}")
        self.marker_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")

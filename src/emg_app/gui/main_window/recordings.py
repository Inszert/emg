"""
The RECORDINGS table and what acts on saved files: trim, open, delete,
quarantine, export, gesture counts.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import json
import os
import time
import traceback

from emg_app import config as cfg
from emg_app.dataset import dataset_summary
from emg_app.gui.dialogs.trim import TrimDialog
from emg_app.gui.qt import QtGui, QtWidgets
from emg_app.gui.tokens import (
    COL_BAD, COL_GOOD, COL_TEXT_DIM, COL_WARN, FS_SM)
from emg_app.gui.workers import DatasetExportWorker
from emg_app.recording import (
    DELETED_DIRNAME, DatasetWriter, Recording, STAGE_RAW, STAGE_TAKE,
    STAGE_TRIMMED, _stage_from_filename, load_recording)
from emg_app.utils import (
    ensure_dir, iso_time, safe_slug, timestamp_slug, unique_path)


class RecordingsMixin:
    """MainWindow -- The RECORDINGS table and what acts on saved files."""
    def _group_recordings(self):
        group = QtWidgets.QGroupBox("RECORDINGS (this run)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Gesture", "Rep", "Dur", "Quality", "Trim"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(190)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda _index: self.trim_selected())
        layout.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        self.trim_button = QtWidgets.QPushButton("TRIM")
        self.trim_button.setToolTip("Inspect and cut the selected recording (double-click)")
        self.trim_button.clicked.connect(self.trim_selected)
        row.addWidget(self.trim_button)

        self.open_file_button = QtWidgets.QPushButton("OPEN FILE(S)...")
        self.open_file_button.setToolTip(
            "Pull up any recording or take already on disk -- including from "
            "an earlier session -- to look at its graph, annotate it, and cut "
            "training data out of it. Several files may be selected: each is "
            "opened separately and stays its own recording."
        )
        self.open_file_button.clicked.connect(self.open_recording_file)
        row.addWidget(self.open_file_button)
        layout.addLayout(row)

        row_b = QtWidgets.QHBoxLayout()
        self.autotrim_button = QtWidgets.QPushButton("AUTO-TRIM ALL")
        self.autotrim_button.setToolTip(
            "Run activity detection over every recording of this run and write "
            "the trimmed copies. Raw files are untouched."
        )
        self.autotrim_button.clicked.connect(self.auto_trim_all)
        row_b.addWidget(self.autotrim_button)
        layout.addLayout(row_b)

        row2 = QtWidgets.QHBoxLayout()
        self.delete_button = QtWidgets.QPushButton("DELETE LAST")
        self.delete_button.setObjectName("danger")
        self.delete_button.setToolTip(
            "Moves the last recording into the _deleted folder. Nothing is "
            "ever erased from disk."
        )
        self.delete_button.clicked.connect(self.delete_last)
        row2.addWidget(self.delete_button)

        self.clear_button = QtWidgets.QPushButton("CLEAR LIST")
        self.clear_button.setToolTip(
            "Empties this list and the live graph buffer. Files on disk stay."
        )
        self.clear_button.clicked.connect(self.clear_list)
        row2.addWidget(self.clear_button)
        layout.addLayout(row2)

        return group

    def _mark_table_trimmed(self, rec: Recording) -> None:
        if rec in self.recordings:
            item = self.table.item(self.recordings.index(rec), 5)
            if item:
                item.setText("yes")

    def _remove_recording(self, rec: Recording) -> None:
        if rec not in self.recordings:
            return
        row = self.recordings.index(rec)
        self.recordings.pop(row)
        self.table.removeRow(row)
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setText(str(i + 1))

    # ======================================================================
    # Recording list actions
    # ======================================================================

    def _add_table_row(self, rec: Recording, trimmed: bool) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        level = rec.worst_issue_level
        quality_text = {None: "ok", "info": "info",
                        "warning": "warn", "error": "ERROR"}[level]
        quality_color = {None: COL_GOOD, "info": COL_TEXT_DIM,
                         "warning": COL_WARN, "error": COL_BAD}[level]

        values = [
            str(row + 1), rec.label, str(rec.repetition),
            f"{rec.duration:.2f}s", quality_text, "yes" if trimmed else "-",
        ]
        for col, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            if col == 4:
                item.setForeground(QtGui.QColor(quality_color))
            if level:
                item.setToolTip("\n".join(str(i) for i in rec.issues))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        self.table.selectRow(row)

    def _selected_recording(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.recordings):
            return row, self.recordings[row]
        return -1, None

    def trim_selected(self) -> None:
        row, rec = self._selected_recording()
        if rec is None:
            QtWidgets.QMessageBox.information(
                self, "Trim", "Select a recording in the list first."
            )
            return
        dialog = TrimDialog(rec, self.writer, self)
        accepted = dialog.exec()
        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {rec.label} rep {rec.repetition}.")
        if accepted and dialog.result_recording is not None:
            item = self.table.item(row, 5)
            if item:
                item.setText("yes")
            self._append_log(
                f"Trimmed {rec.label} rep {rec.repetition} to "
                f"{dialog.result_recording.n_samples} samples."
            )

    def open_recording_file(self) -> None:
        """
        Load one or more recordings from disk and open each in turn.

        This is the route to a take from an earlier session: the RECORDINGS
        table only holds what this run produced, but any file under the
        dataset root can be pulled up, annotated and cut.

        Several files may be selected at once. They are handled one at a time
        and each stays its own recording -- see open_paths_for_review().
        """
        start_dir = self.writer.session_dir
        if not os.path.isdir(start_dir):
            start_dir = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)

        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open recordings or takes (several may be selected)",
            start_dir,
            "Recordings (*.csv *.npz *.parquet);;All files (*)")
        if not paths:
            return
        self.open_paths_for_review(paths)

    def open_paths_for_review(self, paths) -> None:
        """
        Work through a list of files from disk, one dialog each.

        Each file is its own recording and is kept that way: selecting three
        2 s clips gives three training inputs of 2 s, never one of 6 s.
        Joining them would invent a gesture nobody performed and would hide a
        discontinuity from the windowing code, which assumes the samples of a
        recording are contiguous in time.
        """
        paths = [p for p in paths if p]
        takes = [p for p in paths if _stage_from_filename(p) == STAGE_TAKE]
        others = [p for p in paths if p not in takes]

        if len(paths) > 1:
            self._append_log(
                f"Opening {len(paths)} file(s) separately"
                + (f" ({len(takes)} take(s))" if takes else "") + ".")

        # Takes first: they produce new recordings, and doing them as a batch
        # keeps their progress messages together in the log.
        if takes:
            self.open_takes_for_cutting(takes)

        for index, path in enumerate(others, start=1):
            if len(others) > 1:
                self._append_log(
                    f"File {index} of {len(others)}: {os.path.basename(path)}")
            self._review_recording_file(path)

    def _review_recording_file(self, path: str) -> None:
        """One already-labelled recording: graph, notes, optional trim."""
        try:
            rec = load_recording(path)
        except Exception as exc:                 # noqa: BLE001 -- report anything
            QtWidgets.QMessageBox.critical(
                self, "Open recording",
                f"Could not read this file:\n\n{os.path.basename(path)}\n\n{exc}")
            return

        if rec.stage == STAGE_TAKE:              # stage came from the metadata
            self.open_take_for_cutting(path)
            return

        # Annotations and any trimmed copy have to land next to the file the
        # operator actually opened, which may belong to another person or
        # session than the one currently selected in the GUI.
        writer = DatasetWriter(
            self._dataset_root_for(path), rec.person_id, rec.session_id)
        rec.source = path
        if not rec.files:
            rec.files[f"{rec.stage}:{os.path.splitext(path)[1].lstrip('.')}"] = (
                os.path.relpath(path, writer.root))

        self._append_log(f"Opened {os.path.basename(path)} for review.")
        dialog = TrimDialog(rec, writer, self)
        dialog.exec()
        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {os.path.basename(path)}.")
        if dialog.result_recording is not None:
            self._append_log(
                f"Cut {dialog.result_recording.n_samples} samples out of "
                f"{os.path.basename(path)} as training data.")

    @staticmethod
    def _dataset_root_for(path: str) -> str:
        """
        The dataset root a recording sits under:
        <root>/<person>/<session>/<GESTURE>/<file>  ->  <root>

        Falls back to the file's own folder if the layout is not recognised,
        which keeps sidecars next to the data rather than somewhere surprising.
        """
        folder = os.path.dirname(os.path.abspath(path))
        root = os.path.dirname(os.path.dirname(os.path.dirname(folder)))
        return root if os.path.isdir(root) else folder

    def auto_trim_all(self) -> None:
        if not self.recordings:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Auto-trim all",
            f"Run activity detection over {len(self.recordings)} recording(s) "
            "and write a trimmed copy of each?\n\n"
            "Raw files are not modified.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        written = skipped = 0
        for row, rec in enumerate(self.recordings):
            trimmed = rec.auto_trimmed()
            if trimmed is None:
                skipped += 1
                continue
            try:
                self.writer.save(trimmed, STAGE_TRIMMED)
            except OSError as exc:
                self._append_log(f"auto-trim failed for row {row + 1}: {exc}")
                skipped += 1
                continue
            written += 1
            item = self.table.item(row, 5)
            if item:
                item.setText("yes")

        self._append_log(f"Auto-trim: {written} written, {skipped} skipped.")
        QtWidgets.QMessageBox.information(
            self, "Auto-trim complete",
            f"{written} trimmed file(s) written.\n"
            f"{skipped} recording(s) skipped (no clear activity, REST, or an "
            "error).\n\nAll raw files are unchanged.",
        )

    def delete_last(self) -> None:
        if not self.recordings:
            return
        rec = self.recordings[-1]
        answer = QtWidgets.QMessageBox.question(
            self, "Delete last recording",
            f"Move {rec.label} rep {rec.repetition} out of the dataset?\n\n"
            "The files are moved into the _deleted folder, not erased, so the "
            "raw data can always be recovered.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        moved = self._quarantine(rec)
        self.recordings.pop()
        self.table.removeRow(self.table.rowCount() - 1)
        self._append_log(
            f"Moved {rec.label} rep {rec.repetition} to _deleted "
            f"({moved} file(s))."
        )
        self._refresh_plan_info()

    def _quarantine(self, rec: Recording) -> int:
        """
        Move a recording's files into <session>/_deleted/.

        Deliberately not os.remove: a mis-click during a long session should
        cost a folder move, not an hour of a subject's time.
        """
        target = ensure_dir(os.path.join(self.writer.session_dir,
                                         DELETED_DIRNAME))
        moved = 0
        for key, relative in list(rec.files.items()):
            source = os.path.join(self.writer.root, relative)
            if not os.path.exists(source):
                continue
            destination = unique_path(os.path.join(target, os.path.basename(source)))
            try:
                os.replace(source, destination)
                moved += 1
                rec.files[key] = os.path.relpath(destination, self.writer.root)
            except OSError as exc:
                self._append_log(f"could not move {source}: {exc}")

        if moved:
            entry = {
                "uid": rec.uid, "stage": "deleted", "label": rec.label,
                "repetition": rec.repetition, "person_id": rec.person_id,
                "session_id": rec.session_id,
                "deleted_iso": iso_time(time.time()),
                "files": dict(rec.files),
            }
            with open(self.writer.manifest_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return moved

    def clear_list(self) -> None:
        if not self.recordings:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Clear list",
            f"Remove {len(self.recordings)} recording(s) from this list and "
            "reset the live graph?\n\nFiles already written to disk are kept.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.recordings.clear()
        self.table.setRowCount(0)
        self.acq.ring.clear()
        self._append_log("List cleared (files on disk kept).")

    # ======================================================================
    # Graph / dataset export
    # ======================================================================

    def save_graph(self) -> None:
        item = self.sequence.current
        gesture = item[0] if item else self.gesture_combo.currentText()
        default = os.path.join(
            self.writer.graphs_dir,
            f"{self.person_id}_{self.session_id}_{safe_slug(gesture)}_"
            f"{timestamp_slug()}.png",
        )
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save graph", default,
            "PNG image (*.png);;SVG vector (*.svg)",
        )
        if not path:
            return
        if "svg" in selected.lower() and not path.lower().endswith(".svg"):
            path += ".svg"
        elif not os.path.splitext(path)[1]:
            path += ".png"

        self._update_plot_title()
        try:
            written = self.plot.export_image(path)
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Save graph",
                                           f"Export failed:\n\n{exc}")
            return
        self._append_log(f"Graph saved: {written}")
        self.status.showMessage(f"Graph saved to {written}", 8000)

    def export_dataset(self) -> None:
        if self.export_worker is not None:
            return
        out_dir = os.path.join(cfg.PROJECT_DIR, "exports")
        self.export_worker = DatasetExportWorker(
            os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER), out_dir, "prefer_trimmed", self
        )
        self.export_worker.progress.connect(
            lambda text: self.status.showMessage(text, 0)
        )
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_button.setEnabled(False)
        self._append_log("Dataset export started...")
        self.export_worker.start()

    def _on_export_done(self, message: str) -> None:
        self._append_log("Export complete.")
        QtWidgets.QMessageBox.information(self, "Export complete", message)

    def _on_export_failed(self, message: str) -> None:
        self._append_log(f"Export failed: {message}")
        QtWidgets.QMessageBox.warning(self, "Export failed", message)

    def _on_export_finished(self) -> None:
        self.export_worker = None
        self.status.clearMessage()

    def _gesture_counts(self) -> dict:
        """
        How many raw recordings exist per gesture, across the WHOLE dataset.

        Counted from disk rather than from this run, because that is what the
        trainer will see: the question being answered is "do I have enough to
        train", not "what did I just do".
        """
        counts = {gesture: 0 for gesture in cfg.GESTURES}
        try:
            summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        except OSError:                                      # pragma: no cover
            return counts
        for person in summary.get("persons", {}).values():
            for session in person.values():
                for gesture, number in session.items():
                    counts[gesture] = counts.get(gesture, 0) + int(number)
        return counts

    def _refresh_gesture_counts(self) -> None:
        """The 'how much data do I have' readout in the classifier panel."""
        if not hasattr(self, "gesture_counts_label"):
            return
        counts = self._gesture_counts()
        total = sum(counts.values())
        classes = [g for g, n in counts.items() if n]

        rows = []
        for gesture in cfg.GESTURES:
            number = counts.get(gesture, 0)
            colour = cfg.label_color(gesture) if number else COL_TEXT_DIM
            rows.append(
                f'<span style="color:{colour}">{gesture}&nbsp;<b>{number}</b>'
                f'</span>')
        warning = ""
        if len(classes) < 2:
            warning = (f' &mdash; <span style="color:{COL_WARN}">a model '
                       f'needs at least 2</span>')
        headline = (f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
                    f'{total} recording(s) on disk across {len(classes)} '
                    f'gesture(s){warning}</div>')
        self.gesture_counts_label.setText(
            headline + '<div style="padding-top:3px">'
            + " &nbsp; ".join(rows) + '</div>')

    def _refresh_label_counts(self) -> None:
        """How many of each gesture are on disk for this subject/session."""
        counts = {}
        for entry in self.writer.read_manifest():
            if entry.get("stage") != STAGE_RAW:
                continue
            counts[entry.get("label", "?")] = \
                counts.get(entry.get("label", "?"), 0) + 1
        if not counts:
            self.label_counts.setText("nothing recorded in this session yet")
            return
        total = sum(counts.values())
        self.label_counts.setText(
            f"{total} on disk:  "
            + "  ".join(f"{name} {counts[name]}" for name in sorted(counts)))

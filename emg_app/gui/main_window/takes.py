"""
Free takes: record one long take, cut clips from it later.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

import os
import time
from datetime import datetime

from emg_app import config as cfg
from emg_app.dataset import list_all_takes
from emg_app.gui.dialogs.trim import TrimDialog
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import COL_BAD, COL_GOOD, COL_TEXT_DIM, COL_WARN, FS_SM
from emg_app.gui.widgets import make_label
from emg_app.markers import markers_between
from emg_app.recording import (
    DELETED_DIRNAME, DatasetWriter, STAGE_TAKE, load_recording,
    take_uid_from_stem)
from emg_app.utils import fmt_duration


class TakesMixin:
    """MainWindow -- Free takes."""
    def _group_free_take(self):
        """
        Record the whole session in one piece now, cut it up later.

        The counterpart to LABEL ON GRAPH: there you must label within
        RING_BUFFER_SECONDS because nothing is on disk until you select it,
        here the whole stretch goes to disk unlabelled and the cutting can
        happen tomorrow, on a machine with no armband.
        """
        group = QtWidgets.QGroupBox("FREE TAKE  (record now, cut later)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Intended gesture", 10, False, COL_TEXT_DIM))
        self.take_gesture_combo = QtWidgets.QComboBox()
        self.take_gesture_combo.addItems(cfg.GESTURES)
        self.take_gesture_combo.setToolTip(
            "What you sat down to collect. It names the take and is its "
            "default label when cutting, but it does NOT label the data: "
            "every clip is labelled when you cut it out."
        )
        row.addWidget(self.take_gesture_combo, 1)
        layout.addLayout(row)

        self.free_take_button = QtWidgets.QPushButton("START FREE TAKE")
        self.free_take_button.setObjectName("primary")
        self.free_take_button.setCheckable(True)
        self.free_take_button.setMinimumHeight(38)
        self.free_take_button.setToolTip(
            f"Record continuously for up to {cfg.FREE_TAKE_MAX_MINUTES:.0f} "
            "minutes, then write the whole stretch to disk in one piece.  "
            "Ctrl+Shift+R"
        )
        self.free_take_button.toggled.connect(self.toggle_free_take)
        layout.addWidget(self.free_take_button)

        self.free_take_status = make_label(
            "idle -- nothing being recorded", 11, True, COL_TEXT_DIM)
        self.free_take_status.setWordWrap(True)
        layout.addWidget(self.free_take_status)

        self.free_take_detail = make_label("", 10, False, COL_TEXT_DIM)
        self.free_take_detail.setWordWrap(True)
        layout.addWidget(self.free_take_detail)

        #  The header names the subject and session, because the list is
        #  SCOPED to them and that scoping was previously invisible: a take
        #  recorded under one SESSION_ID simply is not here when the app is
        #  reopened under another, which looks exactly like a take that never
        #  got saved.
        self.take_list_header = make_label("TAKES ON DISK", 10, True,
                                           COL_TEXT_DIM)
        layout.addWidget(self.take_list_header)
        #  Filters. The list shows every take of every subject and session;
        #  these narrow it. Both default to "all".
        filter_row = QtWidgets.QHBoxLayout()
        self.take_person_filter = QtWidgets.QComboBox()
        self.take_person_filter.setToolTip("Show only this subject's takes.")
        self.take_date_filter = QtWidgets.QComboBox()
        self.take_date_filter.setToolTip("Show only takes recorded on this day.")
        filter_row.addWidget(self.take_person_filter, 1)
        filter_row.addWidget(self.take_date_filter, 1)
        layout.addLayout(filter_row)
        self.take_person_filter.currentIndexChanged.connect(
            lambda _i: self._apply_take_filter())
        self.take_date_filter.currentIndexChanged.connect(
            lambda _i: self._apply_take_filter())

        self.take_list = QtWidgets.QListWidget()
        self.take_list.setMinimumHeight(110)
        self.take_list.setMaximumHeight(190)
        # Several takes can be queued up: Ctrl+click or Shift+click. They are
        # opened one after another, and each stays its own recording -- see
        # open_takes_for_cutting().
        self.take_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.take_list.setToolTip(
            "Every take saved, for every subject and session -- narrow it with "
            "the filters above. Double-click one "
            "to open it, or Ctrl/Shift+click several and open them in a row. "
            "Each take -- and every clip cut from it -- stays a separate "
            "recording."
        )
        self.take_list.itemDoubleClicked.connect(
            lambda _item: self.open_selected_take())
        layout.addWidget(self.take_list)

        #  Where the rest of this subject's takes are, when the current
        #  session is not where they went. OPEN FILE(S)... reaches them
        #  without changing the session, which is the non-destructive answer.
        self.take_elsewhere = make_label("", FS_SM, False, COL_WARN)
        self.take_elsewhere.setWordWrap(True)
        layout.addWidget(self.take_elsewhere)

        row2 = QtWidgets.QHBoxLayout()
        self.open_take_button = QtWidgets.QPushButton("OPEN TAKE(S) & CUT CLIPS")
        self.open_take_button.setMinimumHeight(30)
        self.open_take_button.setToolTip(
            "Open the selected take(s): select a span, pick a gesture, "
            "EXTRACT CLIP, repeat. Ctrl/Shift+click several to work through "
            "them in a row. Every clip is its own recording -- clips are "
            "never joined together. The takes themselves are never modified."
        )
        self.open_take_button.clicked.connect(self.open_selected_take)
        row2.addWidget(self.open_take_button, 1)

        refresh = QtWidgets.QPushButton("REFRESH")
        refresh.setToolTip("Re-read the _takes folder.")
        refresh.clicked.connect(self._refresh_take_history)
        row2.addWidget(refresh)
        layout.addLayout(row2)

        self.delete_take_button = QtWidgets.QPushButton("DELETE TAKE(S)")
        self.delete_take_button.setMinimumHeight(28)
        # Starts disabled rather than waiting for the first _refresh_controls
        # tick: there is nothing selected yet, and a delete button that is
        # live for 100 ms before it should be is still live.
        self.delete_take_button.setEnabled(False)
        self.delete_take_button.setToolTip(
            "Move the selected take(s) out of _takes and into _deleted. "
            "Nothing is erased, so a take can always be recovered. Clips "
            "already cut from a take are separate recordings and are left "
            "exactly as they are."
        )
        self.delete_take_button.clicked.connect(self.delete_selected_takes)
        layout.addWidget(self.delete_take_button)

        self._free_take_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+R"), self)
        self._free_take_shortcut.activated.connect(
            self.free_take_button.toggle)

        return group

    def toggle_free_take(self, checked: bool) -> None:
        if checked:
            self.start_free_take()
        else:
            self.stop_free_take()

    def start_free_take(self) -> None:
        """Open a long capture. Nothing is written until STOP & SAVE."""
        if self.sequence.running:
            # The acquisition thread has one capture slot, so a take and the
            # cued sequence cannot both own it.
            self._set_free_take_status(
                "cannot start: a recording session is running", COL_BAD)
            self.free_take_button.setChecked(False)
            return

        label = self.take_gesture_combo.currentText()
        self.free_take_label = label
        self.free_take_number = self.writer.next_take_number()
        self.free_capture = self.acq.start_capture(
            label, 0,
            meta={
                "emg_mode": cfg.EMG_MODE,
                "handedness": cfg.HANDEDNESS,
                "arm_worn_on": cfg.ARM_WORN_ON,
                "device": cfg.DEVICE_NAME,
                "notes": self.notes_edit.text(),
                "capture_style": "free_take",
                "intended_gesture": label,
                "take_number": self.free_take_number,
                "free_take_max_minutes": cfg.FREE_TAKE_MAX_MINUTES,
            },
            max_seconds=cfg.FREE_TAKE_MAX_MINUTES * 60.0,
        )
        self.free_capture.mark_label_start()

        self.expand_section("free_take")
        self.free_take_button.setText("STOP & SAVE TAKE")
        self._set_free_take_status(
            f"RECORDING take {self.free_take_number:03d} -- {label}", COL_BAD)
        self._append_log(
            f"Free take {self.free_take_number:03d} started "
            f"({label}, up to {cfg.FREE_TAKE_MAX_MINUTES:.0f} min).")
        self.acq.vibrate(1)
        self._refresh_controls()

    def stop_free_take(self) -> None:
        """Close the capture and write the whole stretch into _takes/."""
        capture = self.free_capture
        self.free_capture = None
        self.free_take_button.setText("START FREE TAKE")

        if capture is None:
            self._set_free_take_status("idle -- nothing being recorded",
                                       COL_TEXT_DIM)
            return

        self.acq.stop_capture()
        self.acq.vibrate(2)

        rec = capture.to_recording(self.person_id, self.session_id)
        if rec.n_samples < 2 or rec.duration < cfg.FREE_TAKE_MIN_SECONDS:
            # Not worth a file. Nothing was written, so nothing to clean up.
            self._set_free_take_status(
                f"discarded: only {rec.duration:.2f} s "
                f"({rec.n_samples} samples), the minimum is "
                f"{cfg.FREE_TAKE_MIN_SECONDS:.0f} s",
                COL_WARN)
            self._append_log(
                f"Free take discarded ({rec.duration:.2f} s, too short).")
            self._refresh_controls()
            return

        # A take is not a gesture instance, so it gets no quality verdict on
        # whether it "contains activity": it is supposed to contain long
        # stretches of nothing. Only the honest signal checks run.
        rec.run_quality_checks(had_disconnect=capture.had_disconnect)

        # Markers dropped during the take travel with it, which is what makes
        # them useful later: they are the landmarks you cut against.
        inside = markers_between(self.session_markers,
                                 float(rec.timestamps[0]),
                                 float(rec.timestamps[-1]))
        if inside:
            rec.meta["markers"] = [m.to_dict() for m in inside]
        rec.meta["capture_overflowed"] = capture.overflowed

        try:
            written = self.writer.save_take(rec, self.free_take_number)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Free take", f"Could not write the take:\n{exc}")
            self._set_free_take_status(f"SAVE FAILED: {exc}", COL_BAD)
            return

        name = os.path.basename(written.get("csv") or written.get("npz") or "")
        self._set_free_take_status(
            f"saved take {self.free_take_number:03d}: "
            f"{fmt_duration(rec.duration)}, {rec.n_samples} samples", COL_GOOD)
        self.free_take_detail.setText(
            f"{name}\nOpen it below to cut clips out of it."
            + (f"   {len(inside)} marker(s) inside." if inside else "")
            + ("   NOTE: the length cap was reached, so the end may be "
               "missing." if capture.overflowed else ""))
        self._append_log(
            f"Free take {self.free_take_number:03d} saved: "
            f"{rec.n_samples} samples, {fmt_duration(rec.duration)}"
            + (f", {len(inside)} marker(s)" if inside else "")
            + (f" (quality: {rec.worst_issue_level})"
               if rec.worst_issue_level else ""))

        self._refresh_take_history()
        self._refresh_controls()

    def _tick_free_take(self) -> None:
        """
        Keep the operator informed, and stop at the cap.

        Called from the 100 ms UI timer, which is also what bounds how long a
        take can overrun its cap: the acquisition Capture stops accepting
        samples by itself, so overrunning costs nothing but a stale readout.
        """
        capture = self.free_capture
        if capture is None:
            return

        elapsed = time.time() - capture.started_unix
        cap = max(1.0, cfg.FREE_TAKE_MAX_MINUTES * 60.0)
        samples = len(capture)
        fraction = elapsed / cap

        self._set_free_take_status(
            f"RECORDING take {self.free_take_number:03d} -- "
            f"{self.free_take_label}   {fmt_duration(elapsed)}",
            COL_BAD)
        self.free_take_detail.setText(
            f"{samples:,} samples   {fraction * 100:.0f} % of the "
            f"{cfg.FREE_TAKE_MAX_MINUTES:.0f} min limit   --   "
            "drop BREAK markers as you go, they become the cut points"
        )
        self.free_take_detail.setStyleSheet(
            f"color: {COL_WARN if fraction > cfg.FREE_TAKE_WARN_FRACTION else COL_TEXT_DIM};"
            " background: transparent;")

        if elapsed >= cap or capture.overflowed:
            self._append_log(
                f"Free take {self.free_take_number:03d} hit the "
                f"{cfg.FREE_TAKE_MAX_MINUTES:.0f} min cap -- saving it.")
            # Drives stop_free_take() through the toggled signal, so the
            # button and the state can never disagree.
            self.free_take_button.setChecked(False)

    def _set_free_take_status(self, text: str, color: str) -> None:
        self.free_take_status.setText(text)
        self.free_take_status.setStyleSheet(
            f"color: {color}; background: transparent;")

    def _refresh_take_history(self) -> None:
        """Re-read every _takes/ folder under the dataset root."""
        if not hasattr(self, "take_list"):
            return
        try:
            self._all_takes = list_all_takes(
                os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        except OSError as exc:                               # pragma: no cover
            self._append_log(f"could not read takes: {exc}")
            self._all_takes = []

        def day(info):
            return datetime.fromtimestamp(info["created_unix"]).strftime(
                "%Y-%m-%d")

        persons = sorted({t["person_id"] for t in self._all_takes})
        days = sorted({day(t) for t in self._all_takes}, reverse=True)
        for combo, first, values in (
                (self.take_person_filter, "All persons", persons),
                (self.take_date_filter, "All dates", days)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([first] + values)
            combo.setCurrentIndex(max(0, combo.findText(keep)))
            combo.blockSignals(False)
        self._apply_take_filter()

    def _apply_take_filter(self) -> None:
        """Fill the list from the cached takes, narrowed by the two filters."""
        if not hasattr(self, "take_list"):
            return
        person = self.take_person_filter.currentText()
        date = self.take_date_filter.currentText()
        everything = getattr(self, "_all_takes", [])
        takes = [
            t for t in everything
            if (person in ("", "All persons") or t["person_id"] == person)
            and (date in ("", "All dates") or datetime.fromtimestamp(
                t["created_unix"]).strftime("%Y-%m-%d") == date)
        ]
        self.take_list_header.setText(
            f"TAKES ON DISK  --  {len(takes)} of {len(everything)}")
        self._show_takes_elsewhere()
        self.take_list.clear()

        if not takes:
            item = QtWidgets.QListWidgetItem(
                "no takes match the filters" if everything else "no takes yet")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.take_list.addItem(item)
            return

        for info in takes:
            stamp = datetime.fromtimestamp(info["created_unix"]).strftime(
                "%Y-%m-%d %H:%M")
            item = QtWidgets.QListWidgetItem(
                f"{info['person_id']} / {info['session_id']}   "
                f"{info['label']}   {stamp}   "
                f"{fmt_duration(info['duration_s'])}   "
                f"{info['n_samples']:,} samples")
            item.setData(QtCore.Qt.UserRole, info["path"])
            # The stem, not the path: DELETE TAKE(S) moves every format the
            # take was written in, and the list only carries one of them.
            item.setData(QtCore.Qt.UserRole + 1, info["stem"])
            item.setData(QtCore.Qt.UserRole + 2, info["person_id"])
            item.setData(QtCore.Qt.UserRole + 3, info["session_id"])
            item.setToolTip(info["path"])
            self.take_list.addItem(item)
        self.take_list.setCurrentRow(0)

    def _show_takes_elsewhere(self) -> None:
        """
        Name this subject's other sessions that hold takes.

        The message is deliberately about where they are and how to reach
        them, not an offer to switch session: changing SESSION_ID is what
        decides where the NEXT recording is written, so doing it silently to
        make a list look right would be the wrong trade.
        """
        # The list now covers every subject and session, so there is no
        # "elsewhere" to point at; the label stays (empty) for callers.
        if hasattr(self, "take_elsewhere"):
            self.take_elsewhere.setText("")

    def open_selected_take(self) -> None:
        """Open every selected take in turn and cut clips out of each."""
        if not hasattr(self, "take_list"):
            return
        items = self.take_list.selectedItems()
        if not items:
            item = self.take_list.currentItem()
            items = [item] if item is not None else []

        paths = [i.data(QtCore.Qt.UserRole) for i in items]
        paths = [p for p in paths if p]
        if not paths:
            self._set_free_take_status(
                "select a take in the list first", COL_WARN)
            return
        self.open_takes_for_cutting(paths)

    def _writer_for(self, person: str, session: str):
        """A writer for the take's own subject/session, not the current one."""
        if (person == self.writer.person_id
                and session == self.writer.session_id):
            return self.writer
        return DatasetWriter(self.writer.root, person, session)

    def delete_selected_takes(self) -> None:
        """
        Move the selected take(s) into _deleted/ after confirming.

        The confirmation names how many clips were cut from each take,
        because that is the thing someone would want to know before throwing
        a take away: the clips themselves are untouched (they are ordinary
        recordings with their own uids), but their `parent_take_file` will
        point into _deleted/ afterwards and the span cannot be re-cut.
        """
        if not hasattr(self, "take_list"):
            return
        if self.free_capture is not None:
            self._set_free_take_status(
                "stop the running take before deleting takes", COL_WARN)
            return

        items = self.take_list.selectedItems()
        if not items:
            item = self.take_list.currentItem()
            items = [item] if item is not None else []
        targets = [(i.data(QtCore.Qt.UserRole + 1),
                    i.data(QtCore.Qt.UserRole + 2),
                    i.data(QtCore.Qt.UserRole + 3)) for i in items]
        targets = [t for t in targets if t[0] and t[1] and t[2]]
        stems = [t[0] for t in targets]
        if not stems:
            self._set_free_take_status(
                "select a take in the list first", COL_WARN)
            return

        lines, total_clips = [], 0
        for stem, person, session in targets:
            uid = take_uid_from_stem(stem)
            n_clips = (len(self._writer_for(person, session)
                           .clips_from_take(uid)) if uid else 0)
            total_clips += n_clips
            lines.append(f"    {stem}"
                         + (f"   ({n_clips} clip(s) cut from it)"
                            if n_clips else ""))

        detail = "\n".join(lines)
        note = ""
        if total_clips:
            note = (f"\n\n{total_clips} clip(s) were cut from these takes. "
                    "Those clips are separate recordings and are NOT "
                    "deleted -- they stay in the dataset. You will not be "
                    "able to cut any more clips from these takes, though.")

        answer = QtWidgets.QMessageBox.question(
            self, "Delete take(s)",
            f"Move {len(stems)} take(s) out of _takes?\n\n{detail}\n\n"
            "The files move into the _deleted folder, not erased, so the "
            "signal can always be recovered." + note,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        deleted, moved = 0, 0
        for stem, person, session in targets:
            result = self._writer_for(person, session).delete_take(stem)
            if result["moved"]:
                deleted += 1
                moved += len(result["moved"])
                self._append_log(
                    f"Moved take {result['label']} {stem} to "
                    f"{DELETED_DIRNAME} ({len(result['moved'])} file(s)).")
            else:
                self._append_log(f"could not delete take {stem}: no files "
                                 "matched -- was it already moved?")

        self._refresh_take_history()
        if deleted:
            self._set_free_take_status(
                f"{deleted} take(s) moved to {DELETED_DIRNAME} "
                f"({moved} file(s))", COL_TEXT_DIM)
        else:
            self._set_free_take_status(
                "nothing was moved -- see the log", COL_WARN)

    def open_takes_for_cutting(self, paths) -> None:
        """
        Work through several takes, one dialog each.

        Each take is opened separately and every clip cut from it is written
        as its own recording. Nothing is ever joined up: three 2 s clips are
        three training inputs of 2 s, never one of 6 s. Concatenating them
        would be wrong twice over -- it would invent a gesture nobody
        performed, and it would hide from the windowing code that the samples
        either side of a join are seconds or minutes apart.
        """
        paths = [p for p in paths if p]
        if not paths:
            return

        total = 0
        for index, path in enumerate(paths, start=1):
            if len(paths) > 1:
                self._append_log(
                    f"Take {index} of {len(paths)}: {os.path.basename(path)}")
            total += len(self.open_take_for_cutting(path) or [])

        if len(paths) > 1:
            self._set_free_take_status(
                f"extracted {total} clip(s) from {len(paths)} takes, "
                f"each a separate recording", COL_GOOD)
            self._append_log(
                f"Finished {len(paths)} takes: {total} separate "
                f"recording(s) written.")

    def open_take_for_cutting(self, path: str):
        """
        Load a take from disk and open the cut dialog on it.

        Shared by the history list and by OPEN FILE..., so a take opened
        either way behaves identically. Returns the clips that were written,
        so a caller working through several takes can total them up.
        """
        try:
            rec = load_recording(path)
        except Exception as exc:                 # noqa: BLE001 -- report anything
            QtWidgets.QMessageBox.critical(
                self, "Open take", f"Could not read this take:\n\n{exc}")
            return []

        # Clips land next to the take's own person/session, which may not be
        # the subject currently selected in the GUI.
        writer = DatasetWriter(
            self._dataset_root_for_take(path), rec.person_id, rec.session_id)
        rec.source = path
        if not rec.files:
            rec.files[f"{STAGE_TAKE}:{os.path.splitext(path)[1].lstrip('.')}"] = (
                os.path.relpath(path, writer.root))

        self._append_log(
            f"Opened take {os.path.basename(path)} "
            f"({rec.n_samples} samples, {fmt_duration(rec.duration)}).")
        dialog = TrimDialog(rec, writer, self, extract_mode=True)
        dialog.exec()

        clips = list(getattr(dialog, "extracted", []))
        for clip in clips:
            self.recordings.append(clip)
            self._add_table_row(clip, False)
        if clips:
            self._append_log(
                f"Extracted {len(clips)} clip(s) from "
                f"{os.path.basename(path)}: "
                + ", ".join(f"{c.label} rep {c.repetition}" for c in clips))
            self._set_free_take_status(
                f"extracted {len(clips)} clip(s) from this take", COL_GOOD)
        else:
            self._set_free_take_status("no clips extracted", COL_TEXT_DIM)

        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {os.path.basename(path)}.")
        self._refresh_label_counts()
        self._refresh_gesture_counts()
        self._refresh_plan_info()
        self._refresh_controls()
        return clips

    @staticmethod
    def _dataset_root_for_take(path: str) -> str:
        """
        The dataset root a TAKE sits under:
        <root>/<person>/<session>/_takes/<file>  ->  <root>

        Same depth as a gesture folder, but spelled out separately because a
        take's parent folder is _takes rather than a gesture name, and getting
        this wrong would scatter clips outside the dataset.
        """
        folder = os.path.dirname(os.path.abspath(path))
        root = os.path.dirname(os.path.dirname(os.path.dirname(folder)))
        return root if os.path.isdir(root) else folder

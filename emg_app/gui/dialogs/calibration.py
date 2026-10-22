"""
CALIBRATE.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import COL_BAD, COL_GOOD, COL_TEXT_DIM, COL_WARN
from emg_app.utils import nominal_rate, safe_slug
from emg_app.calibration import (
    calibration_capture_quality, calibration_from_captures,
    calibration_from_gesture_captures, calibration_from_recordings,
    calibration_levels, calibration_rest_quality, clear_calibration,
    describe_calibration, load_calibration, save_calibration,
    validate_calibration)
from emg_app.dataset import load_dataset
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.widgets import (
    fit_table_height, fit_to_screen, make_label, scroll_dialog_body)


class CalibrationDialog(QtWidgets.QDialog):
    """
    Measure one wearer onto the common scale, or derive it from the data they
    already recorded.

    Three routes, and the first is the one to use:

      * **GUIDED GESTURE SWEEP** -- cue REST, then each gesture in turn, and
        take each channel's reference as the loudest that channel ever got
        across the set. A single clench only drives the muscles that clench
        uses: measured on the synthetic stand-in for this dataset, a
        one-gesture reference under-scales seven of eight channels by
        1.9x-3.2x, and the two quietest by 2.2x and 2.9x. Each capture is
        judged as it lands (`calibration_capture_quality`), so a gesture that
        did not register is marked and can be redone there and then instead
        of being stored as a shrug.
      * **quick rest + clench** -- two captures, kept because it is six
        seconds and better than nothing.
      * **DERIVE FROM RECORDED DATA** -- no armband needed; percentiles of
        everything that wearer has already recorded, labels ignored.

    Nothing recorded is touched by any of them: a calibration is a pair of
    level vectors, and every recording on disk stays byte-identical whether
    one exists or not.
    """

    #  Phases. Reading the ring at the END of a countdown rather than
    #  accumulating during it keeps this off the sample path entirely: the
    #  acquisition thread is already storing the samples, so a GUI timer that
    #  misses a tick costs nothing.
    IDLE, PREPARE, COUNTING = "idle", "prepare", "counting"

    def __init__(self, parent, acquisition, person_id: str, root: str = None,
                 mode: str = None):
        super().__init__(parent)
        self.setWindowTitle("Per-wearer calibration")
        self.acq = acquisition
        self.person_id = safe_slug(person_id, "person")
        self.root = root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        self.mode = (cfg.CALIBRATION_MODE if mode is None else mode) or "off"

        self.rest_block = None
        self.effort_block = None
        self.gesture_blocks: Dict[str, np.ndarray] = {}
        self.record = load_calibration(self.root, self.person_id)
        self.saved = False
        self.phase = self.IDLE
        self._target = ""
        self._deadline = 0.0
        self._queue: List[str] = []

        self._build()
        self._refresh()

        #  Runs the whole time the dialog is open, not just during a capture:
        #  it also keeps the "is the stream alive" line honest, which is the
        #  thing that was impossible to tell before.
        self.tick = QtCore.QTimer(self)
        self.tick.setInterval(100)
        self.tick.timeout.connect(self._on_tick)
        self.tick.start()

    # -- which gestures -------------------------------------------------------

    def sweep_gestures(self) -> List[str]:
        """The active gestures to cue, REST excluded -- it sets the floor."""
        names = cfg.CALIBRATION_GESTURES or cfg.GESTURES
        return [g for g in names if g != "REST"]

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(make_label(f"Calibrating: {self.person_id}", 14, True))
        intro = make_label(
            "The armband reads a different number on every wearer for the "
            "same effort. A calibration measures two levels per channel -- "
            "relaxed, and under load -- so that a given number means the same "
            "amount of effort on anybody.\n\n"
            "Measured ONCE, valid until you calibrate again. No recording is "
            "modified.",
            10, False, COL_TEXT_DIM)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.state_label = make_label("", 11, True)
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        #  Live stream state. Without this the only way to find out why a
        #  capture would not start was to press the button and read an error.
        self.stream_label = make_label("", 10, True, COL_TEXT_DIM)
        self.stream_label.setWordWrap(True)
        layout.addWidget(self.stream_label)

        # ---- the guided sweep ------------------------------------------
        box = QtWidgets.QGroupBox("GUIDED GESTURE SWEEP  (recommended)")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(6)
        hint = make_label(
            "Cues REST and then each gesture. REST sets each channel's floor "
            "and is checked too: a hand that was not fully relaxed is "
            "rejected and can be redone. Each channel's reference is the "
            "loudest it gets across the whole set, which is why this beats "
            "one generic squeeze: a squeeze leaves most channels quiet and "
            "would set their scale from nearly nothing.",
            9, False, COL_TEXT_DIM)
        hint.setWordWrap(True)
        outer.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        self.sweep_button = QtWidgets.QPushButton("START SWEEP")
        self.sweep_button.setObjectName("primary")
        self.sweep_button.setMinimumHeight(32)
        self.sweep_button.clicked.connect(self.start_sweep)
        row.addWidget(self.sweep_button, 2)
        self.redo_button = QtWidgets.QPushButton("REDO SELECTED")
        self.redo_button.setMinimumHeight(32)
        self.redo_button.setToolTip(
            "Capture just the selected row again. A gesture that did not "
            "register is excluded from the reference until it does, so "
            "repeating it is the whole point rather than a fallback.")
        self.redo_button.clicked.connect(self.redo_selected)
        row.addWidget(self.redo_button, 1)
        self.abort_button = QtWidgets.QPushButton("STOP")
        self.abort_button.setMinimumHeight(32)
        self.abort_button.clicked.connect(self.abort_sweep)
        row.addWidget(self.abort_button, 1)
        outer.addLayout(row)

        self.sweep_table = QtWidgets.QTableWidget(0, 4)
        self.sweep_table.setHorizontalHeaderLabels(
            ["gesture", "samples", "peak", "verdict"])
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sweep_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.sweep_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        #  The verdict is the column that carries a sentence; the other three
        #  are short, so they take what they need and it takes the rest.
        head = self.sweep_table.horizontalHeader()
        for col in range(3):
            head.setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeToContents)
        head.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.sweep_table.itemSelectionChanged.connect(self._refresh)
        outer.addWidget(self.sweep_table)
        layout.addWidget(box)
        box.setVisible(cfg.CALIBRATION_GUIDED_SWEEP)

        self.countdown_label = make_label("", 22, True, COL_WARN)
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.countdown_label)

        #  What the armband is reading right now, so a wearer can see the
        #  gesture register before the countdown ends.
        self.live_label = make_label("", 9, False, COL_TEXT_DIM)
        self.live_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.live_label)

        # ---- the quick path --------------------------------------------
        quick = QtWidgets.QGroupBox(
            "QUICK  (rest + one strong squeeze)" if cfg.CALIBRATION_GUIDED_SWEEP
            else "REST + SQUEEZE")
        grid = QtWidgets.QGridLayout(quick)
        grid.setSpacing(6)
        #  One button for the whole thing: relax, then a "get ready" beat,
        #  then squeeze. The two below redo either half on its own.
        self.both_button = QtWidgets.QPushButton(
            f"CALIBRATE  (relax {cfg.CALIBRATION_REST_SECONDS:.0f} s, then "
            f"squeeze {cfg.CALIBRATION_EFFORT_SECONDS:.0f} s)")
        self.both_button.setObjectName("primary")
        self.both_button.setMinimumHeight(34)
        self.both_button.clicked.connect(self.start_rest_squeeze)
        grid.addWidget(self.both_button, 0, 0, 1, 2)

        self.rest_button = QtWidgets.QPushButton(
            f"RELAX ({cfg.CALIBRATION_REST_SECONDS:.0f} s)")
        self.rest_button.setMinimumHeight(30)
        self.rest_button.clicked.connect(lambda: self._start("rest"))
        grid.addWidget(self.rest_button, 1, 0)
        self.rest_result = make_label("not measured", 10, False, COL_TEXT_DIM)
        self.rest_result.setWordWrap(True)
        grid.addWidget(self.rest_result, 1, 1)

        self.effort_button = QtWidgets.QPushButton(
            f"SQUEEZE HARD ({cfg.CALIBRATION_EFFORT_SECONDS:.0f} s)")
        self.effort_button.setMinimumHeight(30)
        self.effort_button.clicked.connect(lambda: self._start("effort"))
        grid.addWidget(self.effort_button, 2, 0)
        self.effort_result = make_label("not measured", 10, False, COL_TEXT_DIM)
        self.effort_result.setWordWrap(True)
        grid.addWidget(self.effort_result, 2, 1)
        grid.setColumnStretch(1, 1)
        layout.addWidget(quick)

        self.derive_button = QtWidgets.QPushButton(
            "DERIVE FROM RECORDED DATA (no armband needed)")
        self.derive_button.setMinimumHeight(28)
        self.derive_button.setToolTip(
            "Take the same two level vectors from percentiles of every "
            "recording this wearer already has, with the labels IGNORED.\n\n"
            "This is how a dataset recorded before calibration existed gets "
            "calibrated. It uses the wearer's own unlabelled signal, which is "
            "the same assumption a live calibration makes -- but a score "
            "measured that way is not zero-shot, and the trainer says so.")
        self.derive_button.clicked.connect(self._derive)
        layout.addWidget(self.derive_button)

        # ---- per-channel readout ---------------------------------------
        self.table = QtWidgets.QTableWidget(cfg.N_CHANNELS, 3)
        self.table.setHorizontalHeaderLabels(["channel", "rest", "reference"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        fit_table_height(self.table, cfg.N_CHANNELS)
        layout.addWidget(self.table)

        # ---- how it is applied -----------------------------------------
        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Apply as", 10, False, COL_TEXT_DIM))
        self.mode_combo = QtWidgets.QComboBox()
        for name, caption in (
                ("off", "off -- stored but not applied"),
                ("gain", "gain -- one divisor, spatial profile untouched"),
                ("per_channel", "per channel -- each channel own divisor"),
                ("affine", "affine -- per channel, rest subtracted")):
            self.mode_combo.addItem(caption, name)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(self.mode)))
        self.mode_combo.setToolTip(
            "Which of the measured numbers actually get applied.\n\n"
            "off          nothing changes; the model sees device counts.\n"
            "gain         one divisor for all eight channels. Conservative: "
            "every ratio between channels, which is what distinguishes the "
            "gestures, survives exactly.\n"
            "per channel  also equalises where the pods sat.\n"
            "affine       per channel plus the wearer's rest level removed. "
            "Measured best on this dataset: +5.3 points on a wearer the model "
            "never saw.\n\n"
            "Changing it means the saved model no longer matches -- retrain.")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self.mode_combo, 1)
        layout.addLayout(row)

        self.warning_label = make_label("", 10, False, COL_WARN)
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        # ---- buttons ----------------------------------------------------
        buttons = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("SAVE CALIBRATION")
        self.save_button.setObjectName("primary")
        self.save_button.setMinimumHeight(32)
        self.save_button.clicked.connect(self._save)
        buttons.addWidget(self.save_button, 2)

        self.clear_button = QtWidgets.QPushButton("CLEAR")
        self.clear_button.setMinimumHeight(32)
        self.clear_button.setToolTip(
            "Drop this wearer's calibration. The full record is written to "
            "calibration_log.jsonl before the file goes, so the measurement "
            "is not lost.")
        self.clear_button.clicked.connect(self._clear)
        buttons.addWidget(self.clear_button, 1)

        close = QtWidgets.QPushButton("CLOSE")
        close.setMinimumHeight(32)
        close.clicked.connect(self.reject)
        buttons.addWidget(close, 1)
        layout.addLayout(buttons)

        #  The body is ~950 px tall, which is more than a 1080p screen has at
        #  125 % scaling. It scrolls; SAVE / CLEAR / CLOSE stay pinned.
        scroll_dialog_body(self, keep_last=1)
        fit_to_screen(self, 680, 900)

    # -- stream state ---------------------------------------------------------

    def stream_state(self) -> tuple:
        """
        (ok, message). Judged on the age of the newest SAMPLE, not on the link
        state: what a capture needs is samples, and an armband that has gone
        to sleep is still "connected" while supplying none.
        """
        if self.acq is None:
            return False, "no acquisition -- the armband is not set up"
        try:
            stats = self.acq.stats()
        except Exception as exc:                               # pragma: no cover
            return False, f"cannot read the stream: {exc}"
        age = stats.get("last_sample_age")
        rate = stats.get("rate") or 0.0
        if age is None:
            state = getattr(self.acq, "state", "") or "not connected"
            return False, f"no samples yet ({state})"
        if age > cfg.CALIBRATION_STALE_SECONDS:
            return False, (f"no sample for {age:.1f} s -- the stream has "
                           f"stopped")
        return True, f"streaming, {rate:.0f} Hz"

    def _streaming(self) -> bool:
        return self.stream_state()[0]

    # -- live capture ---------------------------------------------------------

    def _start(self, which: str, prepare: bool = False) -> None:
        """`which` is "rest", "effort", or a gesture label."""
        if self.phase != self.IDLE:
            return
        ok, message = self.stream_state()
        if not ok:
            self._queue = []
            self.warning_label.setText(
                f"Cannot capture: {message}. Connect the armband, or use "
                f"DERIVE FROM RECORDED DATA.")
            self._refresh()
            return
        self.warning_label.setText("")
        self._target = which
        if prepare:
            self.phase = self.PREPARE
            self._deadline = time.perf_counter() + cfg.CALIBRATION_PREPARE_SECONDS
        else:
            self.phase = self.COUNTING
            self._deadline = time.perf_counter() + self._seconds_for(which)
        self._refresh()

    def _seconds_for(self, which: str) -> float:
        if which == "rest":
            return float(cfg.CALIBRATION_REST_SECONDS)
        if which == "effort":
            return float(cfg.CALIBRATION_EFFORT_SECONDS)
        return float(cfg.CALIBRATION_GESTURE_SECONDS)

    def _cue_for(self, which: str) -> str:
        if which == "rest":
            return "RELAX"
        if which == "effort":
            return "SQUEEZE HARD"
        #  A gesture: its name plus the instruction -- which is how a gesture
        #  added in GESTURES... is explained the first time it is calibrated.
        hint = cfg.gesture_cue(which)
        return f"{which}  ({hint})" if hint else which

    def _on_tick(self) -> None:
        if self.phase == self.IDLE:
            self._refresh_stream_line()
            return

        remaining = self._deadline - time.perf_counter()
        cue = self._cue_for(self._target)

        if self.phase == self.PREPARE:
            if remaining > 0:
                self.countdown_label.setText(f"GET READY:  {cue}")
                self.live_label.setText("")
                return
            self.phase = self.COUNTING
            self._deadline = time.perf_counter() + self._seconds_for(self._target)
            return

        if remaining > 0:
            self.countdown_label.setText(f"{cue}   {remaining:.1f} s")
            self.live_label.setText(self._live_text())
            #  A capture that loses the stream half way through is not a
            #  capture. Abandon it and say so rather than storing a stub.
            if not self._streaming():
                self.phase = self.IDLE
                self._target = ""
                self._queue = []
                self.countdown_label.setText("")
                self.warning_label.setText(
                    "The stream stopped during the capture -- nothing was "
                    "stored. Reconnect and try again.")
                self._refresh()
            return

        self.countdown_label.setText("")
        self.live_label.setText("")
        self._capture(self._target)
        self.phase = self.IDLE
        self._target = ""
        self._advance_sweep()
        self._refresh()

    def _live_text(self) -> str:
        """The newest sample's per-channel levels, as a one-line meter."""
        if self.acq is None:
            return ""
        emg, _t = self.acq.ring.snapshot(12)
        if emg.shape[0] == 0:
            return ""
        levels = np.asarray(emg, dtype=np.float64).mean(axis=0)
        return "  ".join(f"{cfg.CHANNEL_NAMES[i]} {levels[i]:4.0f}"
                         for i in range(min(cfg.N_CHANNELS, levels.size)))

    def _capture(self, which: str) -> None:
        """Take the span that just elapsed out of the ring buffer."""
        seconds = self._seconds_for(which)
        try:
            rate = self.acq.stats().get("rate") or nominal_rate()
        except Exception:                                      # pragma: no cover
            rate = nominal_rate()
        want = int(seconds * rate * 1.3) + 8
        emg, t = self.acq.ring.snapshot(max(8, want))
        if t.size:
            keep = t >= (t[-1] - seconds)
            emg = emg[keep]
        if which == "rest":
            self.rest_block = emg
        elif which == "effort":
            self.effort_block = emg
        else:
            self.gesture_blocks[which] = emg

    # -- the sweep ------------------------------------------------------------

    def start_sweep(self) -> None:
        """REST first -- it sets the floor every gesture is judged against."""
        if self.phase != self.IDLE:
            return
        self.gesture_blocks = {}
        self.effort_block = None
        self._queue = ["rest"] + self.sweep_gestures()
        self._advance_sweep()

    def start_rest_squeeze(self) -> None:
        """REST, then SQUEEZE, from one press -- the whole calibration."""
        if self.phase != self.IDLE:
            return
        self.gesture_blocks = {}
        self.effort_block = None
        self._queue = ["rest", "effort"]
        self._advance_sweep()

    def abort_sweep(self) -> None:
        """Stop cueing. Whatever was captured so far is kept."""
        self._queue = []
        self.phase = self.IDLE
        self._target = ""
        self.countdown_label.setText("")
        self.live_label.setText("")
        self._refresh()

    def redo_selected(self) -> None:
        label = self.selected_gesture()
        if not label or self.phase != self.IDLE:
            return
        self._queue = []
        #  The REST row is the "rest" capture -- the same one RELAX takes.
        self._start("rest" if label == "REST" else label, prepare=True)

    def rest_quality(self) -> dict:
        return calibration_rest_quality(self.rest_block)

    def _usable_rest(self):
        """The rest capture, or None when it was rejected -- never a tense one."""
        if self.rest_block is None or not self.rest_quality()["ok"]:
            return None
        return self.rest_block

    def selected_gesture(self) -> str:
        row = self.sweep_table.currentRow()
        item = self.sweep_table.item(row, 0) if row >= 0 else None
        return item.text() if item is not None else ""

    def _advance_sweep(self) -> None:
        if not self._queue:
            return
        which = self._queue.pop(0)
        #  No "GET READY" for rest -- the wearer is already relaxed.
        self._start(which, prepare=(which != "rest"))

    # -- derive ---------------------------------------------------------------

    def _derive(self) -> None:
        """Build the record from recordings instead of from a live capture."""
        self.warning_label.setText("")
        try:
            recs = load_dataset(self.root, persons=[self.person_id],
                                stage="prefer_trimmed", prefer="npz")
        except Exception as exc:                               # pragma: no cover
            self.warning_label.setText(f"could not read the dataset: {exc}")
            return
        if not recs:
            self.warning_label.setText(
                f"{self.person_id} has no recordings under {self.root} to "
                f"derive a calibration from.")
            return
        record = calibration_from_recordings(recs, self.person_id)
        if record is None:
            self.warning_label.setText("no usable samples in those recordings.")
            return
        self.record = record
        self.rest_block = self.effort_block = None
        self.gesture_blocks = {}
        self._refresh()
        ok, problems = validate_calibration(record)
        self.warning_label.setText(
            f"Derived from {len(recs)} recordings, labels not used. "
            + ("Press SAVE CALIBRATION to keep it."
               if ok else "REFUSED: " + problems[0]))

    # -- state ----------------------------------------------------------------

    def pending_record(self) -> Optional[dict]:
        """
        The record SAVE would write, in order of preference: the gesture
        sweep, then the quick two-capture pair, then whatever is already
        loaded (stored or derived).
        """
        try:
            rate = (self.acq.stats().get("rate") if self.acq else None)
        except Exception:                                      # pragma: no cover
            rate = None
        fs = rate or nominal_rate()

        #  A rejected rest builds nothing: it is the floor `affine` subtracts.
        rest = self._usable_rest()
        if rest is not None and self.gesture_blocks:
            return calibration_from_gesture_captures(
                rest, self.gesture_blocks,
                person_id=self.person_id, session_id=cfg.SESSION_ID, fs=fs)
        if rest is not None and self.effort_block is not None:
            return calibration_from_captures(
                rest, self.effort_block,
                person_id=self.person_id, session_id=cfg.SESSION_ID,
                fs=fs, source="live")
        return self.record

    def _level_text(self, block, percentile: float) -> str:
        if block is None:
            return "not measured"
        n = int(np.asarray(block).shape[0])
        if n < cfg.CALIBRATION_MIN_SAMPLES:
            return f"only {n} samples -- need {cfg.CALIBRATION_MIN_SAMPLES}"
        levels = np.percentile(np.asarray(block, dtype=np.float64),
                               percentile, axis=0)
        return f"{n} samples, mean level {np.mean(levels):.0f}"

    def _refresh_stream_line(self) -> None:
        ok, message = self.stream_state()
        self.stream_label.setText(("STREAM: " if ok else "NO STREAM: ")
                                  + message)
        self.stream_label.setStyleSheet(
            f"color: {COL_GOOD if ok else COL_WARN}; background: transparent;")
        busy = self.phase != self.IDLE
        for widget in (self.sweep_button, self.rest_button,
                       self.effort_button, self.both_button):
            widget.setEnabled(ok and not busy)
        self.redo_button.setEnabled(ok and not busy
                                    and bool(self.selected_gesture()))
        self.abort_button.setEnabled(busy or bool(self._queue))

    def _refresh(self) -> None:
        busy = self.phase != self.IDLE
        self._refresh_stream_line()
        for widget in (self.derive_button, self.clear_button,
                       self.mode_combo):
            widget.setEnabled(not busy)

        rest_q = self.rest_quality()
        rest_text = self._level_text(self.rest_block,
                                     cfg.CALIBRATION_REST_PERCENTILE)
        if self.rest_block is not None and not rest_q["ok"]:
            rest_text += f" -- REJECTED: {rest_q['reason']}"
        self.rest_result.setText(rest_text)
        self.effort_result.setText(
            self._level_text(self.effort_block, cfg.CALIBRATION_EFFORT_PERCENTILE))
        self._fill_sweep_table()

        record = self.pending_record()
        self.clear_button.setEnabled(
            not busy
            and load_calibration(self.root, self.person_id) is not None)

        #  Captures waiting on a rejected rest: say so, and do not offer to
        #  save whatever older record pending_record() fell back to.
        if (self.rest_block is not None and not rest_q["ok"]
                and (self.gesture_blocks or self.effort_block is not None)):
            self.state_label.setText(
                "REST was not relaxed -- nothing can be built until it is "
                "redone.")
            self.state_label.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.warning_label.setText(
                f"REST rejected: {rest_q['reason']}. Select the REST row and "
                f"press REDO SELECTED (or press RELAX).")
            self.save_button.setEnabled(False)
            self._fill_table(None, None)
            return

        if record is None:
            self.state_label.setText(f"{self.person_id} has no calibration.")
            self.state_label.setStyleSheet(
                f"color: {COL_TEXT_DIM}; background: transparent;")
            self.save_button.setEnabled(False)
            self._fill_table(None, None)
            return

        ok, problems = validate_calibration(record)
        self.save_button.setEnabled(bool(ok) and not busy)
        self.state_label.setText(describe_calibration(record, self.mode))
        self.state_label.setStyleSheet(
            f"color: {COL_GOOD if ok else COL_BAD}; background: transparent;")
        if problems:
            self.warning_label.setText("REFUSED: " + "  ".join(problems))
        levels = calibration_levels(record)
        self._fill_table(*(levels if levels else (None, None)))

    def _fill_sweep_table(self) -> None:
        """
        REST first, then one row per gesture, with how each capture landed.
        Gestures are judged against the rest only once the rest is accepted.
        """
        wanted = self.sweep_gestures()
        extra = [g for g in self.gesture_blocks if g not in wanted]
        rows = ["REST"] + wanted + sorted(extra)
        rest_q = self.rest_quality()
        rest_levels = rest_q["levels"] if rest_q["ok"] else None

        keep = self.selected_gesture()
        self.sweep_table.setRowCount(len(rows))
        fit_table_height(self.sweep_table, len(rows))
        for row, label in enumerate(rows):
            block = self.gesture_blocks.get(label)
            if label == "REST":
                if self.rest_block is None:
                    cells = [label, "--", "--", "not captured"]
                    colour = COL_TEXT_DIM
                else:
                    cells = [label, str(rest_q["n_samples"]),
                             (f"CH{rest_q['loud_channel']} "
                              f"{rest_q['level']:.0f}"
                              if rest_q["loud_channel"] else "--"),
                             "accepted" if rest_q["ok"] else rest_q["reason"]]
                    colour = COL_GOOD if rest_q["ok"] else COL_BAD
            elif block is None:
                cells = [label, "--", "--", "not captured"]
                colour = COL_TEXT_DIM
            else:
                q = calibration_capture_quality(block, rest_levels)
                cells = [label, str(q["n_samples"]),
                         (f"CH{q['peak_channel']} "
                          f"{q['contrast']:.1f}x" if q["peak_channel"] else "--"),
                         "accepted" if q["ok"] else q["reason"]]
                colour = COL_GOOD if q["ok"] else COL_BAD
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)       # a long verdict is elided
                if col:
                    item.setForeground(QtGui.QColor(colour))
                self.sweep_table.setItem(row, col, item)
            if label == keep:
                self.sweep_table.selectRow(row)

    def _fill_table(self, rest, reference) -> None:
        for row in range(cfg.N_CHANNELS):
            self.table.setItem(row, 0,
                               QtWidgets.QTableWidgetItem(cfg.CHANNEL_NAMES[row]))
            for col, values in ((1, rest), (2, reference)):
                text = "--" if values is None else f"{values[row]:.0f}"
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignRight
                                      | QtCore.Qt.AlignVCenter)
                self.table.setItem(row, col, item)

    def _on_mode_changed(self) -> None:
        self.mode = self.mode_combo.currentData() or "off"
        self._refresh()

    # -- actions --------------------------------------------------------------

    def _save(self) -> None:
        record = self.pending_record()
        if record is None:
            return
        try:
            path = save_calibration(record, self.root, self.person_id)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Calibration",
                                          f"Not saved:\n\n{exc}")
            return
        self.record = record
        self.rest_block = self.effort_block = None
        self.gesture_blocks = {}
        self.saved = True
        self.result_path = path
        self.accept()

    def _clear(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Clear calibration",
            f"Drop {self.person_id}'s calibration?\n\n"
            "The full record goes into calibration_log.jsonl first, so the "
            "measurement is not lost and can be put back by hand.\n\n"
            "A model that was trained WITH calibration will refuse to "
            "predict until this wearer has one again.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        clear_calibration(self.root, self.person_id, "cleared from the GUI")
        self.record = None
        self.rest_block = self.effort_block = None
        self.gesture_blocks = {}
        self.saved = True
        self.accept()

    def closeEvent(self, event) -> None:                       # pragma: no cover
        self.tick.stop()
        super().closeEvent(event)

"""
The trim / cut dialog.
"""

import os
from datetime import datetime

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_GOOD, COL_TEXT_DIM, FS_MD, H_ACTION, H_CONTROL,
    SP_SM)
from emg_app.utils import emg_range
from emg_app.markers import Marker, load_recording_notes, markers_between
from emg_app.recording import (
    DatasetWriter, Recording, STAGE_RAW, STAGE_TAKE, STAGE_TRIMMED)
from emg_app.gui.qt import QtCore, QtGui, QtWidgets, pg
from emg_app.gui.widgets import fit_to_screen, make_label


class TrimDialog(QtWidgets.QDialog):
    """
    Inspect a recording and cut the useful part out of it.

        |  useless  |  gesture  |  useless  |
                    ^           ^
                  start        end

    The original Recording object is never modified. Accepting produces a NEW
    Recording via rec.slice(), which is saved as a separate *_trimmed.* file
    next to the raw one.

    With `extract_mode=True` it does a second, different job: carving MANY
    labelled clips out of one long free take. Each EXTRACT CLIP writes an
    ordinary stage-"raw" recording into its gesture folder and leaves the
    dialog open, so a five-minute take becomes twenty training recordings
    without reopening anything. The take itself is never modified either.
    """

    def __init__(self, rec: Recording, writer: DatasetWriter, parent=None,
                 extract_mode: bool = False):
        super().__init__(parent)
        self.rec = rec
        self.writer = writer
        self.result_recording = None
        self.notes = []                  # notes read from the sidecar
        self.inside_markers = []         # session markers inside this take
        self.notes_added = 0             # how many the operator added here
        self._note_lines = []            # [(plot, item)] drawn for markers

        # -- extract mode state
        self.extract_mode = bool(extract_mode)
        self.extracted = []              # clips written out of this take
        self._clip_bands = []            # [(plot, item)] drawn for clips

        if self.extract_mode:
            self.setWindowTitle(
                f"Cut clips out of -- {rec.label} take "
                f"({rec.duration:.0f} s)")
        else:
            self.setWindowTitle(f"Trim -- {rec.label} rep {rec.repetition}")
        # The eight channel rows plus the envelope need height; on a laptop
        # this comes out smaller and the plots simply get shorter rather than
        # the dialog running off the bottom of the screen.
        fit_to_screen(self, 1180, 820, min_width=700, min_height=480)

        self.fs = rec.fs
        self.t = rec.t_rel.astype(np.float64)
        self.signal = rec.emg.astype(np.float32)
        self.env = rec.envelope(cfg.TRIM_ENVELOPE_MS)
        self.env_mean = self.env.mean(axis=1)
        self.activity = rec.activity or rec.detect_activity()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(make_label(
            f"{rec.person_id} / {rec.session_id} / {rec.label}  "
            + (f"take" if self.extract_mode else f"rep {rec.repetition}")
            + f"  --  {rec.n_samples} samples, "
              f"{rec.duration:.2f} s @ {self.fs:.0f} Hz", 12, True
        ))

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(0)
        root.addWidget(self.glw, 1)

        lo, _ = emg_range(rec.meta.get("emg_mode", cfg.EMG_MODE))
        signed = lo < 0

        self.regions = []
        self.plots = []
        for i in range(cfg.N_CHANNELS):
            plot = self.glw.addPlot(row=i, col=0)
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setLabel("left", cfg.CHANNEL_NAMES[i])
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("left").setWidth(50)
            plot.getAxis("left").setTextPen(pg.mkPen(cfg.CHANNEL_COLORS[i]))
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("bottom").setHeight(2)
            if i:
                plot.setXLink(self.plots[0])
            curve = plot.plot(pen=pg.mkPen(cfg.CHANNEL_COLORS[i], width=1))
            curve.setDownsampling(auto=True, method="peak")
            curve.setData(self.t, self.signal[:, i])
            span = float(max(abs(self.signal[:, i]).max(), 1.0)) * 1.1
            plot.setYRange(-span if signed else 0, span, padding=0)

            region = pg.LinearRegionItem(brush=pg.mkBrush(56, 189, 248, 42),
                                         pen=pg.mkPen(56, 189, 248, 160))
            region.setZValue(-10)
            plot.addItem(region)
            self.regions.append(region)
            self.plots.append(plot)

        # Envelope row with the detection thresholds drawn in.
        self.env_plot = self.glw.addPlot(row=cfg.N_CHANNELS, col=0)
        self.env_plot.showGrid(x=True, y=True, alpha=0.12)
        self.env_plot.setLabel("left", "RMS")
        self.env_plot.setLabel("bottom", "time", units="s")
        self.env_plot.getAxis("left").setWidth(50)
        self.env_plot.setXLink(self.plots[0])
        self.env_plot.setMaximumHeight(130)
        self.env_plot.plot(self.t, self.env_mean,
                           pen=pg.mkPen("#e6e9ef", width=1))
        for value, color, name in (
            (self.activity.threshold_high, "#ef4444", "high"),
            (self.activity.threshold_low, "#f59e0b", "low"),
            (self.activity.baseline, "#64748b", "baseline"),
        ):
            line = pg.InfiniteLine(pos=value, angle=0,
                                   pen=pg.mkPen(color, style=QtCore.Qt.DashLine))
            line.setToolTip(f"{name} = {value:.1f}")
            self.env_plot.addItem(line)
        self.env_region = pg.LinearRegionItem(brush=pg.mkBrush(56, 189, 248, 42),
                                              pen=pg.mkPen(56, 189, 248, 160))
        self.env_plot.addItem(self.env_region)
        self.regions.append(self.env_region)

        # Label window (what the operator originally cued) as a reference.
        if rec.label_end_index > rec.label_start_index:
            for plot in self.plots[:1]:
                cue = pg.LinearRegionItem(
                    values=(float(self.t[rec.label_start_index]),
                            float(self.t[min(rec.label_end_index,
                                             rec.n_samples - 1)])),
                    movable=False,
                    brush=pg.mkBrush(34, 197, 94, 26),
                    pen=pg.mkPen(34, 197, 94, 90),
                )
                cue.setZValue(-30)
                plot.addItem(cue)

        # ---- controls --------------------------------------------------------
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(make_label("Start (s)", 11, False, COL_TEXT_DIM))
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.01)
        self.start_spin.setRange(0.0, float(self.t[-1]) if self.t.size else 0.0)
        controls.addWidget(self.start_spin)

        controls.addWidget(make_label("End (s)", 11, False, COL_TEXT_DIM))
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.01)
        self.end_spin.setRange(0.0, float(self.t[-1]) if self.t.size else 0.0)
        controls.addWidget(self.end_spin)

        self.info = make_label("", 11, False, COL_TEXT_DIM)
        controls.addWidget(self.info, 1)

        auto = QtWidgets.QPushButton("Auto-detect")
        auto.setToolTip("Run the configured EMG activity detector")
        auto.clicked.connect(self.apply_auto)
        controls.addWidget(auto)

        full = QtWidgets.QPushButton("Full range")
        full.clicked.connect(self.apply_full)
        controls.addWidget(full)

        preview = QtWidgets.QPushButton("Preview selection")
        preview.clicked.connect(self.preview_selection)
        controls.addWidget(preview)

        root.addLayout(controls)

        # ---- extract clips ---------------------------------------------------
        if self.extract_mode:
            root.addWidget(self._build_extract_panel())

        # ---- notes -----------------------------------------------------------
        # Annotating a take that is already on disk. The signal file is
        # immutable, so a note goes to a *_notes.json sidecar beside it.
        if cfg.MARKERS_ENABLED:
            root.addWidget(self._build_notes_panel())

        if self.extract_mode:
            note = make_label(
                "The take is never modified. Each EXTRACT CLIP writes a new "
                "recording into its gesture folder -- that is the training "
                "data. Cut as many clips as you like, then press DONE.",
                11, False, COL_TEXT_DIM
            )
        else:
            note = make_label(
                "The raw recording is never modified. Saving writes a separate "
                "*_trimmed.csv / *_trimmed.npz next to it. Notes go to a separate "
                "*_notes.json and never touch the signal either.", 11, False, COL_TEXT_DIM
            )
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        if self.extract_mode:
            # No "save trimmed copy" here: a take is not something you trim
            # down to one useful part, it is something you harvest many parts
            # from, and each of those is already written by EXTRACT CLIP.
            done = QtWidgets.QPushButton("DONE")
            done.setObjectName("primary")
            done.setDefault(True)
            done.clicked.connect(self.accept)
            buttons.addWidget(done)
        else:
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            save = QtWidgets.QPushButton("Save trimmed copy")
            save.setObjectName("primary")
            save.setDefault(True)
            save.clicked.connect(self.save_trimmed)
            buttons.addWidget(cancel)
            buttons.addWidget(save)
        root.addLayout(buttons)

        # ---- wiring ----------------------------------------------------------
        self._syncing = False
        for region in self.regions:
            region.sigRegionChanged.connect(self._on_region_changed)
        self.start_spin.valueChanged.connect(self._on_spin_changed)
        self.end_spin.valueChanged.connect(self._on_spin_changed)

        if self.extract_mode:
            # Start at the beginning with one clip-length window: carving a
            # long take is a sequential job, and auto-detect would jump to
            # whichever burst happens to be strongest.
            self.select_first_clip()
        elif self.activity.found:
            self.apply_auto()
        else:
            self.apply_full()

    # -- extract mode ---------------------------------------------------------

    def _build_extract_panel(self):
        """Gesture, clip length, stepping, and the list of clips written."""
        group = QtWidgets.QGroupBox("EXTRACT TRAINING CLIPS")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(SP_SM)
        group.setMaximumHeight(300)

        # Where you are in the take, in words. Carving a five-minute take is a
        # sequential job and the one thing the old panel never said was "how
        # far along am I".
        self.position_label = make_label("", FS_MD, True, COL_ACCENT)
        self.position_label.setWordWrap(True)
        layout.addWidget(self.position_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Gesture", 11, False, COL_TEXT_DIM))
        self.extract_gesture = QtWidgets.QComboBox()
        self.extract_gesture.addItems(cfg.GESTURES)
        if self.rec.label in cfg.GESTURES:
            self.extract_gesture.setCurrentText(self.rec.label)
        self.extract_gesture.setToolTip(
            "What the next clip will be labelled as. Change it as often as "
            "you like -- one take can yield clips of several gestures.")
        row.addWidget(self.extract_gesture)

        row.addWidget(make_label("Clip length (s)", 11, False, COL_TEXT_DIM))
        self.clip_length_spin = QtWidgets.QDoubleSpinBox()
        self.clip_length_spin.setDecimals(2)
        self.clip_length_spin.setSingleStep(0.25)
        self.clip_length_spin.setRange(cfg.LABEL_MIN_SELECTION_S,
                                       max(cfg.LABEL_MIN_SELECTION_S,
                                           float(self.t[-1]) if self.t.size else 1.0))
        self.clip_length_spin.setValue(
            min(float(cfg.LABEL_DEFAULT_SELECTION_S),
                self.clip_length_spin.maximum()))
        self.clip_length_spin.setToolTip(
            "The default cut length, in seconds. Mirrored into "
            "LABEL_DEFAULT_SELECTION_S, so it is also what SELECT LAST n s "
            "uses on the live graph and what gets written into every "
            "*_meta.json.")
        self.clip_length_spin.valueChanged.connect(self._on_clip_length_changed)
        row.addWidget(self.clip_length_spin)

        fit = QtWidgets.QPushButton("FIT LENGTH")
        fit.setToolTip("Resize the selection to exactly the clip length, "
                       "keeping its left edge.")
        fit.clicked.connect(self.fit_selection_to_length)
        row.addWidget(fit)
        row.addStretch(1)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        first = QtWidgets.QPushButton("|<  FIRST")
        first.setMinimumHeight(H_CONTROL)
        first.clicked.connect(self.select_first_clip)
        row2.addWidget(first)

        prev_btn = QtWidgets.QPushButton("<<  PREV")
        prev_btn.setMinimumHeight(H_CONTROL)
        prev_btn.setToolTip(
            "Step the selection one clip length backwards.  Ctrl+Left")
        prev_btn.clicked.connect(lambda: self.step_selection(-1))
        row2.addWidget(prev_btn)

        next_btn = QtWidgets.QPushButton("NEXT  >>")
        next_btn.setMinimumHeight(H_CONTROL)
        next_btn.setToolTip(
            "Step the selection one clip length forwards. This is how you "
            "walk through a long take.  Ctrl+Right")
        next_btn.clicked.connect(lambda: self.step_selection(+1))
        row2.addWidget(next_btn)

        row2.addStretch(1)
        layout.addLayout(row2)

        # The one action that writes data gets its own row and twice the
        # height: in a dialog of eight buttons, the one you press twenty times
        # should not look like the one you press once.
        self.extract_button = QtWidgets.QPushButton(
            "EXTRACT CLIP      Ctrl+Space")
        self.extract_button.setObjectName("primary")
        self.extract_button.setMinimumHeight(H_ACTION)
        self.extract_button.setToolTip(
            "Write the selected span as a labelled training recording in its "
            "gesture folder, then step forward to the next span.")
        self.extract_button.clicked.connect(self.extract_clip)
        layout.addWidget(self.extract_button)

        self.clip_list = QtWidgets.QListWidget()
        self.clip_list.setMaximumHeight(110)
        self.clip_list.setToolTip(
            "Clips written out of this take, in the order you cut them. Each "
            "is its own recording -- they are never joined together.")
        layout.addWidget(self.clip_list)

        self.extract_status = make_label(
            "nothing extracted yet", 10, False, COL_TEXT_DIM)
        self.extract_status.setWordWrap(True)
        layout.addWidget(self.extract_status)

        self._extract_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Space"), self)
        self._extract_shortcut.activated.connect(self.extract_clip)
        self._next_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Right"), self)
        self._next_shortcut.activated.connect(lambda: self.step_selection(+1))
        self._prev_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Left"), self)
        self._prev_shortcut.activated.connect(lambda: self.step_selection(-1))

        self._refresh_clip_list()
        return group

    def _on_clip_length_changed(self, value: float) -> None:
        """The spin box is the authority for the default cut length."""
        cfg.LABEL_DEFAULT_SELECTION_S = float(value)

    @property
    def clip_length(self) -> float:
        spin = getattr(self, "clip_length_spin", None)
        return float(spin.value()) if spin is not None else float(cfg.LABEL_DEFAULT_SELECTION_S)

    def select_first_clip(self) -> None:
        if not self.t.size:
            return
        t0 = float(self.t[0])
        self._set_selection(t0, min(float(self.t[-1]), t0 + self.clip_length))

    def fit_selection_to_length(self) -> None:
        if not self.t.size:
            return
        t0 = float(self.start_spin.value())
        self._set_selection(t0, min(float(self.t[-1]), t0 + self.clip_length))

    def step_selection(self, direction: int) -> None:
        """
        Slide the selection one clip length along, keeping its width.

        Walking a long take is the common case, so this is bound to
        Ctrl+Left / Ctrl+Right as well as the buttons.
        """
        if not self.t.size:
            return
        width = max(self.clip_length, float(self.end_spin.value()
                                            - self.start_spin.value()))
        shift = float(direction) * width
        t0 = float(self.start_spin.value()) + shift
        t0 = float(np.clip(t0, self.t[0], max(self.t[0], self.t[-1] - width)))
        self._set_selection(t0, min(float(self.t[-1]), t0 + width))
        self._keep_selection_in_view()

    def _keep_selection_in_view(self) -> None:
        """
        Pan, never zoom: shift the view only if the selection has left it, and
        keep the current zoom level. Zooming here made the graph jump in on
        the first cut and stay there.
        """
        t0, t1 = self.start_spin.value(), self.end_spin.value()
        (x0, x1), _ = self.plots[0].viewRange()
        if t0 >= x0 and t1 <= x1:
            return
        span = x1 - x0
        new0 = t0 - 0.05 * span if t1 - t0 <= span else t0
        self.plots[0].setXRange(new0, new0 + span, padding=0)

    def _refresh_clip_list(self) -> None:
        self.clip_list.clear()
        if not self.extracted:
            item = QtWidgets.QListWidgetItem(
                "no clips extracted from this take yet")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.clip_list.addItem(item)
            return
        for clip in self.extracted:
            start = clip.meta.get("take_start_s", 0.0)
            end = clip.meta.get("take_end_s", 0.0)
            item = QtWidgets.QListWidgetItem(
                f"{clip.label}  rep {clip.repetition:03d}   "
                f"{start:.2f}-{end:.2f} s   {clip.n_samples} samples"
                + (f"   [{clip.worst_issue_level}]"
                   if clip.worst_issue_level else ""))
            item.setForeground(QtGui.QColor(
                cfg.label_color(clip.label)))
            self.clip_list.addItem(item)
        self.clip_list.scrollToBottom()

    def _draw_clip_bands(self) -> None:
        """
        Show which parts of the take have already been harvested, so you can
        see at a glance what is left.
        """
        for plot, item in self._clip_bands:
            plot.removeItem(item)
        self._clip_bands = []

        for clip in self.extracted:
            start = float(clip.meta.get("take_start_s", 0.0))
            end = float(clip.meta.get("take_end_s", 0.0))
            colour = QtGui.QColor(cfg.label_color(clip.label))
            brush = pg.mkBrush(colour.red(), colour.green(), colour.blue(), 40)
            pen = pg.mkPen(colour.name(), width=1)
            for plot in (self.plots[0], self.env_plot):
                band = pg.LinearRegionItem(values=(start, end), movable=False,
                                           brush=brush, pen=pen)
                band.setZValue(-25)
                plot.addItem(band)
                self._clip_bands.append((plot, band))

            caption = pg.TextItem(clip.label, color=colour.name(), anchor=(0, 1))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setPos(start, 0)
            self.plots[0].addItem(caption)
            self._clip_bands.append((self.plots[0], caption))

    def extract_clip(self):
        """
        Write the selected span as a labelled training recording.

        The clip gets a NEW uid rather than inheriting the take's: uid is what
        pairs a raw file with its trimmed copies, and twenty clips sharing the
        take's uid would look like twenty versions of one recording. The link
        back to the take is kept in the metadata instead.
        """
        if not self.extract_mode:
            return None

        start, end = self.selection_indices()
        n = end - start
        if not self.t.size or n < 4:
            self._set_extract_status(
                f"selection is too short ({n} samples)", COL_BAD)
            return None

        t_start = float(self.t[start])
        t_end = float(self.t[min(end - 1, self.t.size - 1)])
        duration = t_end - t_start
        if duration < cfg.LABEL_MIN_SELECTION_S:
            self._set_extract_status(
                f"selection is only {duration:.2f} s -- the minimum is "
                f"{cfg.LABEL_MIN_SELECTION_S:.2f} s", COL_BAD)
            return None

        label = self.extract_gesture.currentText()
        parent = self.rec

        # Subject metadata comes from the TAKE, not from today's config: the
        # handedness and electrode placement that matter are the ones in force
        # when the signal was recorded, which may have been weeks ago.
        parent_meta = dict(parent.meta or {})
        meta = {
            "emg_mode": parent_meta.get("emg_mode", cfg.EMG_MODE),
            "handedness": parent_meta.get("handedness", cfg.HANDEDNESS),
            "arm_worn_on": parent_meta.get("arm_worn_on", cfg.ARM_WORN_ON),
            "device": parent_meta.get("device", cfg.DEVICE_NAME),
            "notes": parent_meta.get("notes", ""),
            "capture_style": "cut_from_take",
            "selected_by_operator": True,
            "selection_seconds": round(duration, 4),
            "parent_take_uid": parent.uid,
            "parent_take_label": parent.label,
            "parent_take_stage": parent.stage,
            "parent_take_file": (parent.files.get(f"{STAGE_TAKE}:npz")
                                 or parent.files.get(f"{STAGE_TAKE}:csv")
                                 or parent.source),
            "parent_take_n_samples": parent.n_samples,
            "take_start_index": int(start),
            "take_end_index": int(end),
            "take_start_s": round(t_start, 4),
            "take_end_s": round(t_end, 4),
        }

        # Markers inside the clip travel with it, exactly as they do for a
        # span cut from the live ring buffer.
        abs_start = float(parent.timestamps[start])
        abs_end = float(parent.timestamps[min(end - 1, parent.n_samples - 1)])
        inside = markers_between(self.inside_markers, abs_start, abs_end)
        if inside:
            meta["markers"] = [m.to_dict() for m in inside]

        repetition = self.writer.next_repetition(label)
        clip = Recording(
            emg=parent.emg[start:end].copy(),
            timestamps=parent.timestamps[start:end].copy(),
            label=label,
            person_id=parent.person_id,
            session_id=parent.session_id,
            repetition=repetition,
            stage=STAGE_RAW,
            # The operator drew the boundaries, so the whole clip is the
            # gesture: there is no pre/post roll to exclude.
            label_start_index=0,
            label_end_index=int(n),
            meta=meta,
        )
        clip.run_quality_checks(expected_seconds=duration)
        if cfg.AUTO_TRIM:
            clip.detect_activity()

        try:
            written = self.writer.save(clip, STAGE_RAW)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Extract clip",
                f"Could not write the clip:\n{exc}")
            return None

        self.extracted.append(clip)
        self._refresh_clip_list()
        self._refresh_position()
        self._draw_clip_bands()
        self._set_extract_status(
            f"wrote {os.path.basename(written.get('csv') or written.get('npz'))}"
            f"  --  {len(self.extracted)} clip(s) from this take",
            COL_GOOD)

        # Step forward so the next press of NEXT/EXTRACT continues where this
        # clip ended rather than re-cutting the same span.
        self.step_selection(+1)
        return clip

    def _set_extract_status(self, text: str, color: str) -> None:
        self.extract_status.setText(text)
        self.extract_status.setStyleSheet(
            f"color: {color}; background: transparent;")

    # -- notes ----------------------------------------------------------------

    def _build_notes_panel(self):
        """Existing notes for this take, plus a field to add another."""
        group = QtWidgets.QGroupBox("NOTES ON THIS RECORDING")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)
        group.setMaximumHeight(190)

        self.notes_list = QtWidgets.QListWidget()
        self.notes_list.setMaximumHeight(86)
        layout.addWidget(self.notes_list)

        row = QtWidgets.QHBoxLayout()
        self.note_input = QtWidgets.QLineEdit()
        self.note_input.setPlaceholderText(
            "a note about this recording, e.g. subject moved the wrist")
        self.note_input.returnPressed.connect(self.add_note)
        row.addWidget(self.note_input, 1)

        add = QtWidgets.QPushButton("ADD NOTE")
        add.setObjectName("primary")
        add.setToolTip(
            "Saved immediately to the *_notes.json sidecar. You do not have "
            "to press Save for a note to persist."
        )
        add.clicked.connect(self.add_note)
        row.addWidget(add)
        layout.addLayout(row)

        self.notes_status = make_label("", 10, False, COL_TEXT_DIM)
        self.notes_status.setWordWrap(True)
        layout.addWidget(self.notes_status)

        self._reload_notes()
        return group

    def _reload_notes(self) -> None:
        """Re-read the sidecar and redraw both the list and the graph lines."""
        path = self.rec.primary_path(self.writer.root)
        self.notes = load_recording_notes(path) if path else []

        # Session markers that fall inside this take are part of its story
        # too, so they are shown here even though they live in the session
        # log rather than in this recording's sidecar.
        try:
            session = self.writer.load_markers()
        except OSError:                                      # pragma: no cover
            session = []
        t0 = float(self.rec.timestamps[0]) if self.rec.n_samples else 0.0
        t1 = float(self.rec.timestamps[-1]) if self.rec.n_samples else 0.0

        # Prefer the markers recorded in the metadata at cut time. They were
        # computed from the span's real boundaries; recomputing from the first
        # and last SAMPLE drops a marker that sits exactly on the boundary,
        # which is precisely the case for CUT SINCE LAST MARKER -- the marker
        # it cut from would vanish from its own recording.
        stored = self.rec.meta.get("markers")
        recomputed = markers_between(session, t0, t1)
        if isinstance(stored, list) and stored:
            by_uid = {}
            for entry in stored:
                if isinstance(entry, dict):
                    try:
                        marker = Marker.from_dict(entry)
                    except (TypeError, ValueError):          # pragma: no cover
                        continue
                    by_uid[marker.uid] = marker
            for marker in recomputed:
                by_uid.setdefault(marker.uid, marker)
            self.inside_markers = sorted(by_uid.values(),
                                         key=lambda m: m.t_unix)
        else:
            self.inside_markers = recomputed

        self.notes_list.clear()
        if not self.notes and not self.inside_markers:
            item = QtWidgets.QListWidgetItem("no notes on this recording yet")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.notes_list.addItem(item)
        for marker in self.inside_markers:
            stamp = f"{marker.t_unix - t0:+.2f} s"
            item = QtWidgets.QListWidgetItem(
                f"| {stamp}  (session marker)  {marker.short_text(44)}")
            item.setForeground(QtGui.QColor(
                cfg.MARKER_COLORS.get(marker.kind, cfg.MARKER_COLORS["note"])))
            item.setToolTip(marker.text or "(no text)")
            self.notes_list.addItem(item)
        for marker in self.notes:
            stamp = datetime.fromtimestamp(marker.created_unix).strftime("%Y-%m-%d %H:%M")
            item = QtWidgets.QListWidgetItem(f"- {stamp}  {marker.short_text(52)}")
            item.setToolTip(marker.text or "(no text)")
            self.notes_list.addItem(item)
        self.notes_list.scrollToBottom()
        self._draw_note_lines(t0)

    def _draw_note_lines(self, t0: float) -> None:
        """
        Vertical separators where a marker was dropped inside this take.

        Created once per reload rather than pooled: this dialog is modal and
        static, so there is no frame budget to protect here.
        """
        # Each item is remembered together with the plot that owns it, so
        # removal is exact rather than a guess about which scene it is in.
        for plot, item in getattr(self, "_note_lines", []):
            plot.removeItem(item)
        self._note_lines = []

        for marker in self.inside_markers:
            x = float(marker.t_unix - t0)
            colour = cfg.MARKER_COLORS.get(marker.kind, cfg.MARKER_COLORS["note"])
            for plot in self.plots:
                line = pg.InfiniteLine(
                    pos=x, angle=90, movable=False,
                    pen=pg.mkPen(colour, width=2 if marker.is_break else 1,
                                 style=QtCore.Qt.DashLine if marker.is_break
                                 else QtCore.Qt.SolidLine),
                )
                line.setZValue(-8)
                line.setToolTip(marker.text or marker.kind)
                plot.addItem(line)
                self._note_lines.append((plot, line))

            caption = pg.TextItem(marker.short_text(24), color=colour,
                                  anchor=(0, 0))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setPos(x, 0)
            self.plots[0].addItem(caption)
            self._note_lines.append((self.plots[0], caption))

    def add_note(self) -> None:
        text = self.note_input.text().strip()
        if not text:
            self.notes_status.setText("type the note first")
            self.notes_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.note_input.setFocus()
            return
        try:
            path, _marker = self.writer.add_note_to_recording(self.rec, text)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Add note", f"Could not write the note:\n{exc}")
            return

        self.notes_added = getattr(self, "notes_added", 0) + 1
        self.note_input.clear()
        self.notes_status.setText(f"saved to {os.path.basename(path)}")
        self.notes_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self._reload_notes()

    # -- selection ------------------------------------------------------------

    def _index_of(self, seconds: float) -> int:
        if self.t.size == 0:
            return 0
        return int(np.clip(np.searchsorted(self.t, seconds), 0, self.t.size - 1))

    def _set_selection(self, t0: float, t1: float) -> None:
        if self.t.size == 0:
            return
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        t0 = float(np.clip(t0, self.t[0], self.t[-1]))
        t1 = float(np.clip(t1, self.t[0], self.t[-1]))
        if t1 - t0 < 1.0 / max(self.fs, 1.0):
            t1 = min(self.t[-1], t0 + 1.0 / max(self.fs, 1.0))

        self._syncing = True
        try:
            for region in self.regions:
                region.setRegion((t0, t1))
            self.start_spin.setValue(t0)
            self.end_spin.setValue(t1)
        finally:
            self._syncing = False
        self._update_info()

    def selection_indices(self):
        start = self._index_of(self.start_spin.value())
        end = min(self.rec.n_samples, self._index_of(self.end_spin.value()) + 1)
        return start, max(start + 1, end)

    def _on_region_changed(self, region) -> None:
        if self._syncing:
            return
        t0, t1 = region.getRegion()
        self._set_selection(t0, t1)

    def _on_spin_changed(self, _value) -> None:
        if self._syncing:
            return
        self._set_selection(self.start_spin.value(), self.end_spin.value())

    def _refresh_position(self) -> None:
        """Where the selection sits in the take, and how much is left."""
        if not getattr(self, "extract_mode", False):
            return
        label = getattr(self, "position_label", None)
        if label is None or not self.t.size:
            return
        t0 = float(self.start_spin.value())
        t1 = float(self.end_spin.value())
        total = float(self.t[-1])
        done = len(getattr(self, "extracted", []))
        label.setText(
            f"at {t0:.1f}-{t1:.1f} s of {total:.1f} s "
            f"({t0 / max(total, 0.001) * 100:.0f} % through the take)"
            + (f"   --   {done} clip(s) cut so far" if done else
               "   --   nothing cut yet"))

    def _update_info(self) -> None:
        self._refresh_position()
        start, end = self.selection_indices()
        kept = end - start
        self.info.setText(
            f"keeping {kept} / {self.rec.n_samples} samples "
            f"({kept / max(1, self.rec.n_samples) * 100:.0f} %), "
            f"{kept / max(self.fs, 1.0):.2f} s  --  "
            f"removing {self.rec.n_samples - kept} samples"
        )

    # -- actions --------------------------------------------------------------

    def apply_auto(self) -> None:
        result = self.rec.detect_activity()
        self.activity = result
        if not result.found:
            QtWidgets.QMessageBox.information(
                self, "Auto-detect",
                "No clear EMG activity was found.\n\n"
                f"Reason: {result.reason}\n\n"
                "Adjust TRIM_THRESHOLD_HIGH / TRIM_ABS_FLOOR in the config, "
                "or select the region by hand.",
            )
            return
        self._method = "auto"
        self._set_selection(float(self.t[result.start]),
                            float(self.t[min(result.end, self.t.size - 1)]))

    def apply_full(self) -> None:
        self._method = "manual"
        if self.t.size:
            self._set_selection(float(self.t[0]), float(self.t[-1]))

    def preview_selection(self) -> None:
        t0, t1 = self.start_spin.value(), self.end_spin.value()
        pad = max(0.05, (t1 - t0) * 0.15)
        self.plots[0].setXRange(t0 - pad, t1 + pad, padding=0)

    def save_trimmed(self) -> None:
        start, end = self.selection_indices()
        if end - start < 2:
            QtWidgets.QMessageBox.warning(self, "Trim",
                                          "The selection is too short.")
            return

        trimmed = self.rec.slice(start, end, STAGE_TRIMMED, {
            "trim_method": getattr(self, "_method", "manual"),
            "trim_dialog": True,
        })
        try:
            written = self.writer.save(trimmed, STAGE_TRIMMED)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Trim",
                                           f"Could not write the trimmed file:\n{exc}")
            return

        self.result_recording = trimmed
        QtWidgets.QMessageBox.information(
            self, "Trim saved",
            "Trimmed copy written:\n\n"
            + "\n".join(os.path.basename(p) for p in written.values())
            + "\n\nThe raw recording is unchanged.",
        )
        self.accept()

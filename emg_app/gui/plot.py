"""
EmgPlotWidget: the live / review graph.
"""

import os
from typing import Tuple

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import COL_ACCENT, COL_TEXT_DIM
from emg_app.utils import emg_range, ensure_dir, unique_path
from emg_app.signal_processing import envelope
from emg_app.quality import measured_rate
from emg_app.markers import MARKER_KINDS
from emg_app.gui.qt import QtCore, QtGui, QtSvg, QtWidgets, pg
from emg_app.gui.widgets import _marker_caption


# ------------------------------------------------------------------------------
# The real-time plot
# ------------------------------------------------------------------------------


class EmgPlotWidget(QtWidgets.QWidget):
    """
    Eight stacked channel bands with a shared, linked time axis.

    Performance notes (this is the part that has to keep up with the armband):
      * one PlotDataItem per channel, updated with setData -- no re-creation
      * peak-mode auto downsampling, so a 30 s window still shows every spike
        instead of smoothing it away
      * clipToView, so off-screen samples cost nothing
      * x axis is seconds relative to now, which avoids large float64 values
        in the renderer
    """

    # Emitted while the operator drags the review trim band (start_s, end_s).
    trim_changed = QtCore.Signal(float, float)
    # Emitted while dragging the LABEL ON GRAPH selection (absolute unix times).
    selection_changed = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(4, 2, 8, 2)
        self.glw.ci.setSpacing(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw)

        lo, hi = emg_range()
        self._signed = lo < 0
        self._y_limit = [max(1.0, (hi - lo) * 0.25)] * cfg.N_CHANNELS

        self.window_s = float(cfg.PLOT_WINDOW_S)
        self.autoscale = bool(cfg.PLOT_AUTOSCALE)
        self.manual_scale = float(max(1.0, (hi - lo) * 0.5))
        self.show_envelope = bool(cfg.SHOW_ENVELOPE)
        self.frozen = False

        self._frame = 0
        self._last_stats = (np.zeros(cfg.N_CHANNELS), np.zeros(cfg.N_CHANNELS),
                            np.zeros(cfg.N_CHANNELS), np.zeros(cfg.N_CHANNELS))

        # Title row: this doubles as the caption of an exported image.
        self.title_item = self.glw.addLabel(
            cfg.APP_NAME, row=0, col=0, size="10pt", color=COL_TEXT_DIM
        )

        # Review mode: the plot shows one finished recording instead of the
        # live stream, with a draggable band marking what will be kept.
        self.review_mode = False
        self._review_t = np.zeros(0, dtype=np.float32)
        self._syncing_trim = False

        # True once the operator has zoomed/panned by hand: the autoscale
        # then stops overwriting the axes every frame.
        self._user_zoomed = False

        # LABEL ON GRAPH state. Selections and marks live in absolute unix
        # time and are re-projected every frame (see _project_selection).
        self.label_mode = False
        self.selection_abs = None
        self.marks = []
        self._syncing_select = False
        self._t_now = None

        self.plots, self.curves, self.env_curves = [], [], []
        self.value_items, self.regions, self.trim_regions = [], [], []
        self.select_regions = []
        self.mark_pool = [[] for _ in range(cfg.LABEL_MAX_VISIBLE_MARKS)]
        self.mark_labels = []
        # Markers: a timestamped note or break, drawn as a vertical separator
        # across every channel plus one caption on the top plot. Same pooling
        # discipline as the labelled bands -- no scene items are created or
        # destroyed while streaming.
        self.markers = []
        self.marker_pool = [[] for _ in range(cfg.MARKER_MAX_VISIBLE)]
        self.marker_labels = []
        self._marker_slot_kind = [None] * cfg.MARKER_MAX_VISIBLE
        self._mark_slot_style = [None] * cfg.LABEL_MAX_VISIBLE_MARKS

        for i in range(cfg.N_CHANNELS):
            plot = self.glw.addPlot(row=i + 1, col=0)
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setMouseEnabled(x=True, y=True)
            # The x axis is "seconds relative to now", so 0 is the present and
            # there is nothing to the right of it. Without a limit pyqtgraph
            # happily pans into the future and shows a blank canvas; and
            # nothing exists further back than the ring buffer either.
            self._apply_live_limits(plot)
            # Any mouse zoom/pan hands the axes to the operator until RESET.
            plot.sigRangeChangedManually.connect(self._on_manual_zoom)
            # Markers are anchored in absolute time, so they have to be
            # re-projected whenever the view moves -- not only when a new
            # frame of samples arrives.
            plot.sigRangeChanged.connect(self._on_range_changed)
            plot.setMenuEnabled(True)
            plot.hideButtons()
            plot.setLabel("left", cfg.CHANNEL_NAMES[i])
            axis = plot.getAxis("left")
            #  No "(x0.001)" suffix: the counts are unitless, and at 150 %
            #  scaling a lane is ~40 px tall -- "CH1 (x0.001)" written
            #  vertically runs into the next lane's label, "CH1" does not.
            axis.enableAutoSIPrefix(False)
            axis.setWidth(52)
            axis.setStyle(tickFont=QtGui.QFont("Consolas", 7))
            axis.setTextPen(pg.mkPen(cfg.CHANNEL_COLORS[i]))

            bottom = plot.getAxis("bottom")
            if i < cfg.N_CHANNELS - 1:
                bottom.setStyle(showValues=False)
                bottom.setHeight(2)
            else:
                plot.setLabel("bottom", "time", units="s")
                bottom.setStyle(tickFont=QtGui.QFont("Consolas", 7))

            if i > 0:
                plot.setXLink(self.plots[0])

            curve = plot.plot(pen=pg.mkPen(cfg.CHANNEL_COLORS[i], width=cfg.PLOT_LINE_WIDTH))
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

            env = plot.plot(pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DashLine))
            env.setDownsampling(auto=True, method="peak")
            env.setClipToView(True)
            env.setVisible(self.show_envelope)

            # Recording span highlight.
            region = pg.LinearRegionItem(
                values=(0, 0), movable=False,
                brush=pg.mkBrush(239, 68, 68, 38),
                pen=pg.mkPen(239, 68, 68, 120),
            )
            region.setZValue(-20)
            region.setVisible(False)
            plot.addItem(region)

            # Review trim band: what will be kept. Draggable on every channel,
            # all eight kept in sync.
            trim = pg.LinearRegionItem(
                values=(0, 0), movable=True,
                brush=pg.mkBrush(34, 197, 94, 48),
                pen=pg.mkPen(34, 197, 94, 210),
            )
            trim.setZValue(-15)
            trim.setVisible(False)
            trim.sigRegionChanged.connect(self._on_trim_region_changed)
            plot.addItem(trim)

            # Labelled bands from LABEL ON GRAPH mode. Pooled: creating and
            # destroying scene items every frame is what makes a live plot
            # stutter.
            for slot in range(cfg.LABEL_MAX_VISIBLE_MARKS):
                band = pg.LinearRegionItem(values=(0, 0), movable=False)
                band.setZValue(-25)
                band.setVisible(False)
                plot.addItem(band)
                self.mark_pool[slot].append(band)

            # Marker separators: one vertical line per channel per slot.
            for slot in range(cfg.MARKER_MAX_VISIBLE):
                line = pg.InfiniteLine(pos=0, angle=90, movable=False)
                line.setZValue(-15)
                line.setVisible(False)
                plot.addItem(line)
                self.marker_pool[slot].append(line)

            # The selection box, draggable on every channel and kept in sync.
            select = pg.LinearRegionItem(
                values=(0, 0), movable=True,
                brush=pg.mkBrush(250, 250, 250, 40),
                pen=pg.mkPen("#ffffff", width=2),
            )
            select.setZValue(-5)
            select.setVisible(False)
            select.sigRegionChanged.connect(self._on_select_region_changed)
            plot.addItem(select)
            self.select_regions.append(select)

            value = pg.TextItem("", color=cfg.CHANNEL_COLORS[i], anchor=(1, 0))
            value.setFont(QtGui.QFont("Consolas", 8))
            plot.addItem(value)

            self.plots.append(plot)
            self.curves.append(curve)
            self.env_curves.append(env)
            self.regions.append(region)
            self.trim_regions.append(trim)
            self.value_items.append(value)

        # Mark captions sit on the top plot only, so they do not repeat eight
        # times down the screen.
        for _ in range(cfg.LABEL_MAX_VISIBLE_MARKS):
            text = pg.TextItem("", anchor=(0, 1))
            text.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            text.setVisible(False)
            self.plots[0].addItem(text)
            self.mark_labels.append(text)

        # Marker captions, anchored the opposite way to the gesture captions
        # so the two do not sit on top of each other at the same instant.
        for _ in range(cfg.MARKER_MAX_VISIBLE):
            caption = pg.TextItem("", anchor=(0, 0))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setVisible(False)
            self.plots[0].addItem(caption)
            self.marker_labels.append(caption)

        self.plots[0].setXRange(-self.window_s, 0, padding=0)

        # Big centred overlay for the countdown. A QLabel on top of the canvas
        # is far cheaper than drawing text into the scene every frame.
        self.overlay = QtWidgets.QLabel("", self.glw)
        self.overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay.setStyleSheet(
            "color: rgba(255,255,255,210); background: transparent;"
            "font-size: 150px; font-weight: 800;"
        )
        self.overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.overlay.hide()

        # Prediction read-out, top-LEFT so it never collides with the
        # countdown badge in the top-right.
        self.prediction_overlay = QtWidgets.QLabel("", self.glw)
        self.prediction_overlay.setAlignment(QtCore.Qt.AlignLeft
                                             | QtCore.Qt.AlignTop)
        self.prediction_overlay.setAttribute(
            QtCore.Qt.WA_TransparentForMouseEvents)
        self.prediction_overlay.hide()

    # -- geometry -------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.glw.rect())
        self._place_prediction_overlay()

    def _place_prediction_overlay(self) -> None:
        rect = self.glw.rect()
        self.prediction_overlay.setGeometry(
            QtCore.QRect(rect.left() + 62, rect.top() + 4,
                         max(220, rect.width() // 2), 76))

    def set_prediction_overlay(self, text: str, color: str = None,
                               sub: str = "") -> None:
        """
        Draw the live prediction over the traces.

        Left-aligned in the top-left corner and deliberately compact: the
        point of watching the graph during a prediction is to see the signal
        that produced it.
        """
        if not text or not cfg.PREDICTION_OVERLAY:
            self.prediction_overlay.hide()
            return
        colour = color or COL_ACCENT
        body = (f"<div style='color:{colour}; font-size:30px;"
                f" font-weight:800; line-height:1.0'>{text}</div>")
        if sub:
            body += (f"<div style='color:rgba(230,233,239,170);"
                     f" font-size:11px'>{sub}</div>")
        self.prediction_overlay.setText(body)
        self.prediction_overlay.setStyleSheet("background: transparent;")
        self._place_prediction_overlay()
        self.prediction_overlay.show()
        self.prediction_overlay.raise_()

    # -- controls -------------------------------------------------------------

    def set_window_seconds(self, seconds: float) -> None:
        self.window_s = float(max(0.2, seconds))
        # Choosing a window is an explicit instruction, so it overrides a
        # previous manual zoom rather than being ignored by it.
        self._user_zoomed = False
        self.plots[0].setXRange(-self.window_s, 0, padding=0)

    def set_autoscale(self, enabled: bool) -> None:
        self.autoscale = bool(enabled)

    def set_manual_scale(self, value: float) -> None:
        self.manual_scale = float(max(1.0, value))

    def set_envelope_visible(self, enabled: bool) -> None:
        self.show_envelope = bool(enabled)
        for env in self.env_curves:
            env.setVisible(enabled)
            if not enabled:
                env.setData([], [])

    def set_channel_visible(self, index: int, visible: bool) -> None:
        self.plots[index].setVisible(bool(visible))

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)

    @staticmethod
    def _apply_live_limits(plot) -> None:
        """
        Confine panning to the live axis: nothing exists after "now", and
        nothing before the start of the ring buffer.
        """
        plot.setLimits(xMax=cfg.PLOT_FUTURE_MARGIN_S,
                       xMin=-float(cfg.RING_BUFFER_SECONDS) * 1.05)

    def _clear_limits(self) -> None:
        """Lift the live limits -- a reviewed take has its own time axis."""
        for plot in self.plots:
            plot.setLimits(xMin=None, xMax=None)

    def _restore_limits(self) -> None:
        for plot in self.plots:
            self._apply_live_limits(plot)

    def _on_manual_zoom(self, *_args) -> None:
        self._user_zoomed = True

    @property
    def user_zoomed(self) -> bool:
        """True while the operator's own zoom is in charge of the axes."""
        return self._user_zoomed

    def reset_view(self) -> None:
        self._user_zoomed = False
        for plot in self.plots:
            plot.enableAutoRange(enable=False)
        self.plots[0].setXRange(-self.window_s, 0, padding=0)
        lo, hi = emg_range()
        self._y_limit = [max(1.0, (hi - lo) * 0.25)] * cfg.N_CHANNELS

    def set_overlay(self, text: str, color: str = None) -> None:
        """
        Draw the cue on top of the graph.

        Placement follows COUNTDOWN_OVERLAY. The default "corner" exists
        because a 150 px digit in the middle of the canvas hides exactly the
        traces you are trying to watch while you perform the gesture -- the
        big version of the cue already lives in the banner above the graph.
        """
        if not text or cfg.COUNTDOWN_OVERLAY == "off":
            self.overlay.hide()
            return

        rect = self.glw.rect()
        if cfg.COUNTDOWN_OVERLAY == "center":
            size = 150 if len(text) <= 2 else max(34, int(240 / len(text)))
            align = QtCore.Qt.AlignCenter
            geometry = rect
        else:                                   # "corner"
            size = 56 if len(text) <= 2 else max(20, int(150 / len(text)))
            align = QtCore.Qt.AlignRight | QtCore.Qt.AlignTop
            width = min(rect.width(), max(140, size * max(2, len(text))))
            geometry = QtCore.QRect(
                rect.right() - width - 60, rect.top() + 6,
                width, int(size * 1.35),
            )

        self.overlay.setAlignment(align)
        self.overlay.setStyleSheet(
            f"color: {color or 'rgba(255,255,255,200)'}; background: transparent;"
            f"font-size: {size}px; font-weight: 800;"
        )
        self.overlay.setText(text)
        self.overlay.setGeometry(geometry)
        self.overlay.show()
        self.overlay.raise_()

    # -- label on graph -------------------------------------------------------
    #
    # Everything here is anchored in ABSOLUTE unix time and re-projected onto
    # the x axis every frame. The axis is "seconds relative to now", so a
    # selection stored in axis coordinates would slide along the signal as
    # time passed and end up labelling the wrong samples.

    def set_label_mode(self, enabled: bool) -> None:
        self.label_mode = bool(enabled)
        if not enabled:
            self.selection_abs = None
        for region in self.select_regions:
            region.setVisible(bool(enabled) and self.selection_abs is not None)

    def arm_selection(self, t_now: float, width_s: float = None) -> None:
        """Drop a fresh selection box at the right-hand edge of the graph."""
        width = float(width_s or cfg.LABEL_DEFAULT_SELECTION_S)
        self.selection_abs = (t_now - width, t_now)
        for region in self.select_regions:
            region.setVisible(True)
        self._project_selection(t_now)

    def clear_selection(self) -> None:
        self.selection_abs = None
        for region in self.select_regions:
            region.setVisible(False)

    def selection_bounds(self):
        """The selection as absolute (start, end) unix times, or None."""
        return self.selection_abs

    def add_mark(self, start_abs: float, end_abs: float, label: str,
                 uid: str = "") -> None:
        self.marks.append({
            "start": float(start_abs), "end": float(end_abs),
            "label": str(label), "uid": uid,
        })
        if len(self.marks) > 400:
            del self.marks[:200]

    def remove_last_mark(self):
        return self.marks.pop() if self.marks else None

    def clear_marks(self) -> None:
        self.marks.clear()
        self._mark_slot_style = [None] * cfg.LABEL_MAX_VISIBLE_MARKS
        for pool in self.mark_pool:
            for region in pool:
                region.setVisible(False)
        for text in self.mark_labels:
            text.setVisible(False)

    def _project_selection(self, t_now: float) -> None:
        if self.selection_abs is None:
            return
        a, b = self.selection_abs
        self._syncing_select = True
        try:
            for region in self.select_regions:
                region.setRegion((a - t_now, b - t_now))
        finally:
            self._syncing_select = False

    def _on_select_region_changed(self, region) -> None:
        """A drag happened: convert straight back to absolute time."""
        if self._syncing_select or self._t_now is None:
            return
        low, high = region.getRegion()
        a, b = sorted((float(low), float(high)))
        self.selection_abs = (a + self._t_now, b + self._t_now)
        self._project_selection(self._t_now)
        self.selection_changed.emit(*self.selection_abs)

    def _draw_marks(self, t_now: float) -> None:        # noqa: D401
        """
        Re-project the labelled bands, newest first, hiding the ones that have
        scrolled off the left edge.
        """
        window = self._visible_span()
        shown = 0
        for mark in reversed(self.marks):
            if shown >= cfg.LABEL_MAX_VISIBLE_MARKS:
                break
            start = mark["start"] - t_now
            end = mark["end"] - t_now
            if end < -window * 1.02:            # fully off-screen to the left
                continue
            colour = cfg.LABEL_COLORS.get(
                mark["label"], cfg.CHANNEL_COLORS[hash(mark["label"]) % cfg.N_CHANNELS])

            # Restyle only when this slot changes gesture. Re-applying a brush
            # and pen to 8 items every frame is pure waste, and
            # LinearRegionItem has no setPen -- the pen lives on its boundary
            # lines.
            if self._mark_slot_style[shown] != mark["label"]:
                self._mark_slot_style[shown] = mark["label"]
                rgb = QtGui.QColor(colour)
                brush = pg.mkBrush(rgb.red(), rgb.green(), rgb.blue(), 60)
                pen = pg.mkPen(colour, width=1)
                for region in self.mark_pool[shown]:
                    region.setBrush(brush)
                    for line in region.lines:
                        line.setPen(pen)
                    region.update()

            for i, region in enumerate(self.mark_pool[shown]):
                region.setRegion((start, end))
                region.setVisible(self.plots[i].isVisible())
            text = self.mark_labels[shown]
            text.setText(mark["label"])
            text.setColor(colour)
            text.setPos(start, 0)
            text.setVisible(True)
            shown += 1

        for index in range(shown, cfg.LABEL_MAX_VISIBLE_MARKS):
            for region in self.mark_pool[index]:
                region.setVisible(False)
            self.mark_labels[index].setVisible(False)

    # -- markers --------------------------------------------------------------

    def add_marker(self, t_abs: float, text: str, kind: str = "note",
                   uid: str = "") -> None:
        """Remember a marker for drawing. Does not write anything to disk."""
        self.markers.append({
            "t": float(t_abs), "text": str(text),
            "kind": kind if kind in MARKER_KINDS else "note", "uid": uid,
        })
        if len(self.markers) > 400:
            del self.markers[:200]

    def remove_last_marker(self):
        return self.markers.pop() if self.markers else None

    def clear_markers(self) -> None:
        self.markers.clear()
        self._marker_slot_kind = [None] * cfg.MARKER_MAX_VISIBLE
        for pool in self.marker_pool:
            for line in pool:
                line.setVisible(False)
        for caption in self.marker_labels:
            caption.setVisible(False)

    def _visible_span(self) -> float:
        """
        How far back the view currently reaches, in seconds.

        This is NOT self.window_s once the operator has zoomed: using the
        nominal window meant anything older than it was culled, so markers and
        bands vanished the moment you zoomed out to look for them.
        """
        span = self.window_s
        try:
            (x_lo, _x_hi), _ = self.plots[0].viewRange()
            span = max(span, abs(float(x_lo)))
        except Exception:                                    # pragma: no cover
            pass
        return span

    def _on_range_changed(self, *_args) -> None:
        """Keep absolute-time items glued to the signal while the view moves."""
        if self._t_now:
            self._draw_markers(float(self._t_now))
            self._draw_marks(float(self._t_now))

    def _draw_markers(self, t_now: float) -> None:
        """
        Re-project the marker separators, newest first, hiding the ones that
        have scrolled off the left edge.
        """
        window = self._visible_span()
        shown = 0
        for marker in reversed(self.markers):
            if shown >= cfg.MARKER_MAX_VISIBLE:
                break
            x = marker["t"] - t_now
            if x < -window * 1.02:              # scrolled off to the left
                continue

            kind = marker["kind"]
            colour = cfg.MARKER_COLORS.get(kind, cfg.MARKER_COLORS["note"])

            # Restyle a slot only when its kind changes, for the same reason
            # the labelled bands do: re-penning 8 items every frame is waste.
            if self._marker_slot_kind[shown] != kind:
                self._marker_slot_kind[shown] = kind
                pen = pg.mkPen(
                    colour,
                    width=2 if kind == "break" else 1,
                    style=QtCore.Qt.DashLine if kind == "break"
                    else QtCore.Qt.SolidLine,
                )
                for line in self.marker_pool[shown]:
                    line.setPen(pen)

            for i, line in enumerate(self.marker_pool[shown]):
                line.setPos(x)
                line.setVisible(self.plots[i].isVisible())

            caption = self.marker_labels[shown]
            prefix = "| " if kind == "break" else ""
            caption.setText(prefix + _marker_caption(marker))
            caption.setColor(colour)
            caption.setPos(x, 0)
            caption.setVisible(True)
            shown += 1

        for index in range(shown, cfg.MARKER_MAX_VISIBLE):
            for line in self.marker_pool[index]:
                line.setVisible(False)
            self.marker_labels[index].setVisible(False)

    # -- review mode ----------------------------------------------------------

    def review(self, rec, activity=None) -> Tuple[float, float]:
        """
        Freeze the graph on a finished recording and draw the proposed cut.

        Returns the (start_s, end_s) the band was initialised to. The live
        stream keeps filling the ring buffer underneath; nothing is lost.
        """
        self.review_mode = True
        self.overlay.hide()
        # A take is shown on its own 0..duration axis, which the live limits
        # would forbid.
        self._clear_limits()

        t = rec.t_rel.astype(np.float32)
        signal = rec.emg.astype(np.float32)
        self._review_t = t

        if t.size == 0:
            return (0.0, 0.0)

        env = envelope(signal, max(1.0, rec.fs)) if self.show_envelope else None

        for i in range(cfg.N_CHANNELS):
            self.curves[i].setData(t, signal[:, i])
            if env is not None:
                self.env_curves[i].setData(t, env[:, i])
            self.regions[i].setVisible(False)

            span = max(float(np.abs(signal[:, i]).max()), 8.0) * cfg.PLOT_Y_PADDING
            self.plots[i].setYRange(-span if self._signed else 0, span, padding=0)

        self.plots[0].setXRange(float(t[0]), float(t[-1]), padding=0.02)

        if activity is not None and getattr(activity, "found", False):
            start = int(np.clip(activity.start, 0, t.size - 1))
            end = int(np.clip(activity.end - 1, start, t.size - 1))
            bounds = (float(t[start]), float(t[end]))
        else:
            bounds = (float(t[0]), float(t[-1]))

        self.set_trim_bounds(*bounds)
        for trim in self.trim_regions:
            trim.setVisible(bool(cfg.REVIEW_SHOW_TRIM))
        return bounds

    def exit_review(self) -> None:
        """Back to the live stream."""
        self.review_mode = False
        self._review_t = np.zeros(0, dtype=np.float32)
        for trim in self.trim_regions:
            trim.setVisible(False)
        # Back on the live axis, so the live pan limits apply again. Restore
        # them BEFORE setting the range, or the range would be clamped against
        # limits that are about to change.
        self._restore_limits()
        self.plots[0].setXRange(-self.window_s, 0, padding=0)

    def set_trim_bounds(self, start_s: float, end_s: float) -> None:
        self._syncing_trim = True
        try:
            for trim in self.trim_regions:
                trim.setRegion((float(start_s), float(end_s)))
        finally:
            self._syncing_trim = False

    def trim_bounds(self) -> Tuple[float, float]:
        low, high = self.trim_regions[0].getRegion()
        return (float(min(low, high)), float(max(low, high)))

    def trim_indices(self, n_samples: int) -> Tuple[int, int]:
        """Current trim band as [start, end) sample indices."""
        t = self._review_t
        if t.size == 0:
            return (0, n_samples)
        low, high = self.trim_bounds()
        start = int(np.clip(np.searchsorted(t, low), 0, max(0, t.size - 1)))
        end = int(np.clip(np.searchsorted(t, high) + 1, start + 1, t.size))
        return (start, min(end, n_samples))

    def _on_trim_region_changed(self, region) -> None:
        if self._syncing_trim:
            return
        low, high = region.getRegion()
        self.set_trim_bounds(low, high)
        self.trim_changed.emit(float(min(low, high)), float(max(low, high)))

    def set_title(self, text: str) -> None:
        self.title_item.setText(text, size="10pt", color=COL_TEXT_DIM)

    def set_recording_span(self, start_rel: float, end_rel: float,
                           visible: bool = True) -> None:
        for region in self.regions:
            region.setVisible(visible)
            if visible:
                region.setRegion((start_rel, end_rel))

    # -- the frame ------------------------------------------------------------

    def update_data(self, emg: np.ndarray, t: np.ndarray) -> None:
        """Push one frame of samples. Called from the GUI timer only."""
        if self.frozen or self.review_mode or emg.shape[0] == 0:
            return

        self._frame += 1
        now = t[-1]
        self._t_now = float(now)
        x = (t - now).astype(np.float32)

        # How far back the curves must reach. Using self.window_s alone meant
        # that zooming OUT past the window showed empty space -- the data was
        # in the ring buffer, it was just never handed to the curve. Follow
        # the visible range instead, so zooming out reveals real signal.
        span = self.window_s
        try:
            (x_lo, _x_hi), _ = self.plots[0].viewRange()
            span = max(span, abs(float(x_lo)))
        except Exception:                                    # pragma: no cover
            pass
        keep = x >= -span * 1.05
        if not keep.all():
            x = x[keep]
            emg = emg[keep]
        if x.size == 0:
            return

        signal = emg.astype(np.float32)
        env = (envelope(signal, max(1.0, measured_rate(t)))
               if self.show_envelope else None)

        mins = signal.min(axis=0)
        maxs = signal.max(axis=0)
        last = signal[-1]
        rms = np.sqrt(np.mean((signal - signal.mean(axis=0)) ** 2, axis=0))
        self._last_stats = (last, mins, maxs, rms)

        label_every = max(1, cfg.GUI_FPS // 8)

        for i in range(cfg.N_CHANNELS):
            plot = self.plots[i]
            if not plot.isVisible():
                continue

            self.curves[i].setData(x, signal[:, i])
            if env is not None:
                self.env_curves[i].setData(x, env[:, i])

            # Vertical range.
            if self.autoscale:
                span = max(abs(float(maxs[i])), abs(float(mins[i])))
                target = max(span * cfg.PLOT_Y_PADDING, 8.0)
                current = self._y_limit[i]
                # Grow instantly (never clip a peak), shrink slowly (no jitter).
                self._y_limit[i] = target if target > current else \
                    current * 0.93 + target * 0.07
                limit = self._y_limit[i]
            else:
                limit = self.manual_scale

            # Once the operator has zoomed by hand, leave the axes alone --
            # otherwise every frame snaps the view back and zooming is
            # impossible. RESET VIEW gives control back to the autoscale.
            if not self._user_zoomed:
                if self._signed:
                    plot.setYRange(-limit, limit, padding=0)
                else:
                    plot.setYRange(0, limit, padding=0)

            if i == 0 and self.mark_labels:
                # Keep the mark captions pinned near the top of the band.
                for text in self.mark_labels:
                    if text.isVisible():
                        text.setPos(text.pos().x(), limit * 0.95)

            if self._frame % label_every == 0:
                top = limit * 0.92 if not self._signed else limit * 0.8
                self.value_items[i].setPos(0, top)
                self.value_items[i].setText(
                    f"{int(last[i]):>5d}   min {int(mins[i]):>5d}  "
                    f"max {int(maxs[i]):>5d}  rms {rms[i]:6.1f}"
                )

        # Re-project the absolute-time overlays onto the moving axis.
        if self.marks:
            self._draw_marks(float(now))
            self._draw_markers(float(now))
        if self.selection_abs is not None:
            self._project_selection(float(now))

    def channel_stats(self):
        """(last, min, max, rms) arrays from the most recent frame."""
        return self._last_stats

    # -- export ---------------------------------------------------------------

    def export_image(self, path: str, width: int = 2400,
                     description: str = "") -> str:
        """
        Save the current graph as PNG (raster) or SVG (vector).

        SVG goes through Qt's own QSvgGenerator rather than PyQtGraph's
        SVGExporter: the latter crashes on these plots in pyqtgraph 0.14
        (ValueError in correctCoordinates while re-parsing path data).
        Rendering the scene with QPainter gives a true vector file and also
        lets the caption and metadata be embedded in the SVG header.
        """
        path = unique_path(path)
        ensure_dir(os.path.dirname(path))
        scene = self.glw.scene()
        rect = scene.sceneRect()

        if path.lower().endswith(".svg"):
            generator = QtSvg.QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(QtCore.QSize(max(1, int(rect.width())),
                                           max(1, int(rect.height()))))
            generator.setViewBox(QtCore.QRectF(0, 0, rect.width(), rect.height()))
            generator.setTitle(str(self.title_item.text))
            generator.setDescription(description or str(self.title_item.text))
            painter = QtGui.QPainter(generator)
            try:
                scene.render(
                    painter,
                    QtCore.QRectF(0, 0, rect.width(), rect.height()),
                    rect,
                )
            finally:
                painter.end()
            return path

        exporter = pg.exporters.ImageExporter(scene)
        try:
            exporter.parameters()["width"] = int(width)
        except Exception:
            pass
        exporter.export(path)
        return path

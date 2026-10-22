"""
MODEL STATS views: DECISION TRAIL and FEATURE SPACE.
"""

import time
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BG, COL_BORDER, COL_GOOD, COL_PANEL, COL_PANEL_2,
    COL_TEXT, COL_TEXT_DIM, COL_WARN, RAD_SM, SP_SM)
from emg_app.decision_panel import TrailStep
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.widgets import make_label
from emg_app.gui.theme import class_color

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.decision_panel import DecisionPanel
    from emg_app.prediction import LivePredictor


class DecisionTrailWidget(QtWidgets.QWidget):
    """
    Every candidate's decision on one window, and how they fuse into one.

    Draws four columns -- the input, the candidates, the wires, the verdict --
    and nothing it was not given. With no step set it paints the reason why
    instead of an empty frame or a plausible-looking placeholder.

    It is a painter, not a source: `set_step()` is the only way anything gets
    on screen, and the caller is the one that decides whether the numbers are
    live or recorded. That keeps the "no fake predictions" rule checkable in
    one place.
    """

    ROW_H = 34
    PAD = 12
    IN_W = 148
    WIRE_W = 64
    OUT_W = 196

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step: Optional[TrailStep] = None
        self.channels: Optional[np.ndarray] = None
        self.channel_peak = 1.0
        self.caption = ""
        self.empty_reason = "no decision panel loaded"
        self.cv: dict = {}
        self.setMinimumHeight(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.MinimumExpanding)

    def set_empty(self, reason: str) -> None:
        self.step = None
        self.empty_reason = reason
        self.update()

    def set_step(self, step: Optional[TrailStep], channels=None,
                 caption: str = "", cv: dict = None) -> None:
        self.step = step
        self.channels = None if channels is None else np.asarray(channels,
                                                                 dtype=float)
        if self.channels is not None and self.channels.size:
            self.channel_peak = max(1.0, float(np.max(self.channels)))
        self.caption = caption
        if cv is not None:
            self.cv = cv
        self.update()

    def sizeHint(self):
        rows = len(self.step.names) if self.step else 7
        return QtCore.QSize(720, 86 + rows * self.ROW_H + 60)

    # -- painting -------------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(COL_BG))

        if self.step is None or not self.step.names:
            self._paint_empty(painter)
            painter.end()
            return

        width, height = self.width(), self.height()
        #  Shed columns rather than squeezing them when the dialog is narrow:
        #  a 40 px wide probability bar communicates nothing.
        show_input = width >= 640 and self.channels is not None
        show_wires = width >= 520
        left = self.PAD + (self.IN_W + self.PAD if show_input else 0)
        right_w = min(self.OUT_W, max(150, int(width * 0.3)))
        wire_w = self.WIRE_W if show_wires else 0
        models_w = width - left - wire_w - right_w - self.PAD * 2
        if models_w < 120:
            models_w = max(100, width - left - right_w - self.PAD * 2)
            wire_w, show_wires = 0, False

        top = self.PAD + 22
        rows = len(self.step.names)

        self._paint_header(painter, width)
        if show_input:
            self._paint_input(painter, self.PAD, top, self.IN_W,
                              rows * self.ROW_H)
        centres = self._paint_models(painter, left, top, models_w)
        out_x = left + models_w + wire_w + self.PAD * 2
        fuse_top = top + max(0, (rows * self.ROW_H) // 2 - 58)
        if show_wires:
            self._paint_wires(painter, left + models_w + self.PAD, centres,
                              wire_w, out_x, fuse_top + 46)
        self._paint_fusion(painter, out_x, fuse_top, right_w)

        if self.caption:
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font = painter.font()
            font.setPointSize(8)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(
                QtCore.QRect(self.PAD, top + rows * self.ROW_H + 6,
                             max(60, width - self.PAD * 2), 34),
                QtCore.Qt.TextWordWrap | QtCore.Qt.AlignTop, self.caption)
        painter.end()

    def _paint_empty(self, painter):
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(self.PAD, self.PAD,
                                              -self.PAD, -self.PAD),
                         QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                         self.empty_reason)

    def _paint_header(self, painter, width):
        step = self.step
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        tag = "LIVE WINDOW" if step.source == "live" else "RECORDED WINDOW"
        colour = COL_GOOD if step.source == "live" else COL_ACCENT
        painter.setPen(QtGui.QColor(colour))
        painter.drawText(self.PAD, self.PAD + 10, tag)
        if step.note:
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font.setBold(False)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            room = max(40, width - self.PAD * 2 - metrics.horizontalAdvance(tag) - 14)
            painter.drawText(
                self.PAD + metrics.horizontalAdvance(tag) + 14, self.PAD + 10,
                metrics.elidedText(step.note, QtCore.Qt.ElideRight, room))

    def _paint_input(self, painter, x, y, w, h):
        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BORDER), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), RAD_SM, RAD_SM)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(int(x + 8), int(y + 15), "INPUT  CHANNEL LEVEL")

        values = self.channels
        n = int(values.size)
        if not n:
            return
        bar_h = max(5, int((h - 32) / n) - 3)
        for i in range(n):
            row_y = y + 24 + i * (bar_h + 3)
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font.setBold(False)
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(int(x + 8), int(row_y + bar_h - 1),
                             cfg.CHANNEL_NAMES[i] if i < len(cfg.CHANNEL_NAMES)
                             else f"CH{i + 1}")
            track = QtCore.QRectF(x + 34, row_y, w - 76, bar_h)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_PANEL_2))
            painter.drawRect(track)
            frac = max(0.0, min(1.0, float(values[i]) / self.channel_peak))
            if frac > 0:
                painter.setBrush(QtGui.QColor(cfg.CHANNEL_COLORS[
                    i % len(cfg.CHANNEL_COLORS)]))
                painter.drawRect(QtCore.QRectF(track.x(), track.y(),
                                               track.width() * frac, bar_h))
            painter.setPen(QtGui.QColor(COL_TEXT))
            painter.drawText(
                QtCore.QRectF(x + w - 40, row_y - 1, 34, bar_h + 2),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{float(values[i]):.0f}")

    def _paint_models(self, painter, x, y, w):
        step = self.step
        centres = []
        target = step.decision(step.fused_from_mean)
        font = painter.font()

        for i, name in enumerate(step.names):
            probabilities = step.per_model[i]
            pick = step.decision(probabilities)
            confidence = float(probabilities[pick])
            row_y = y + i * self.ROW_H
            centres.append((row_y + self.ROW_H / 2 - 4, pick, confidence))

            #  A row that disagrees with the vote is the only thing worth
            #  highlighting here, so it is the only thing highlighted.
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_PANEL if pick == target
                                          else COL_PANEL_2))
            painter.drawRoundedRect(
                QtCore.QRectF(x, row_y, w, self.ROW_H - 4), 3, 3)
            painter.setBrush(QtGui.QColor(class_color(pick)))
            painter.drawRect(QtCore.QRectF(x, row_y, 3, self.ROW_H - 4))

            font.setPointSize(9)
            font.setBold(name == step.deployed_name)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(COL_TEXT))
            metrics = painter.fontMetrics()
            name_w = max(60, int(w * 0.34))
            shown = name + ("  *" if name == step.deployed_name else "")
            painter.drawText(
                QtCore.QRectF(x + 9, row_y, name_w, self.ROW_H - 4),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                metrics.elidedText(shown, QtCore.Qt.ElideRight, name_w))

            #  The probability split: one segment per class, to scale.
            bar_x = x + 9 + name_w + 6
            bar_w = max(40.0, w - (name_w + 150))
            bar = QtCore.QRectF(bar_x, row_y + self.ROW_H / 2 - 5, bar_w, 8)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_BG))
            painter.drawRect(bar)
            offset = 0.0
            for k, p in enumerate(probabilities):
                seg = max(0.0, float(p)) * bar_w
                if seg <= 0.2:
                    offset += seg
                    continue
                painter.setBrush(QtGui.QColor(class_color(k)))
                painter.drawRect(QtCore.QRectF(bar.x() + offset, bar.y(),
                                               seg, bar.height()))
                offset += seg

            font.setPointSize(8)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(class_color(pick)))
            painter.drawText(
                QtCore.QRectF(bar_x + bar_w + 6, row_y, 78, self.ROW_H - 4),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                f"{step.labels[pick] if pick < len(step.labels) else pick}")
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            painter.drawText(
                QtCore.QRectF(x + w - 62, row_y, 56, self.ROW_H - 4),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{confidence:.3f}")
        return centres

    def _paint_wires(self, painter, x, centres, w, out_x, out_y):
        """
        One curve per candidate into the vote.

        Width and opacity carry that candidate's confidence, and the colour
        carries its decision -- so a split vote is visible as two colours
        arriving at the same point, which is the thing worth seeing.
        """
        end_x = out_x - 2
        for y, pick, confidence in centres:
            path = QtGui.QPainterPath(QtCore.QPointF(x, y))
            mid = x + (end_x - x) * 0.55
            path.cubicTo(QtCore.QPointF(mid, y),
                         QtCore.QPointF(mid, out_y),
                         QtCore.QPointF(end_x, out_y))
            colour = QtGui.QColor(class_color(pick))
            colour.setAlpha(int(70 + 160 * max(0.0, min(1.0, confidence))))
            painter.setPen(QtGui.QPen(colour, 0.9 + 2.0 * confidence))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(COL_TEXT))
        painter.drawEllipse(QtCore.QPointF(end_x, out_y), 3.0, 3.0)

    def _paint_fusion(self, painter, x, y, w):
        step = self.step
        fused = step.fused if step.fused is not None else step.fused_from_mean
        pick = step.decision(fused)
        confidence = float(fused[pick])
        height = 118

        painter.setPen(QtGui.QPen(QtGui.QColor(COL_TEXT), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, height),
                                RAD_SM, RAD_SM)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(int(x + 10), int(y + 15), "SOFT VOTE")

        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        mean_pick = step.decision(step.fused_from_mean)
        painter.drawText(
            QtCore.QRectF(x + 10, y + 19, w - 20, 26),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            f"mean of {len(step.names)} = "
            f"{step.fused_from_mean[mean_pick]:.4f}"
            + ("" if step.fused is None else
               f"   ensemble {fused[pick]:.4f}"))

        font.setPointSize(17)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(class_color(pick)))
        painter.drawText(
            QtCore.QRectF(x + 10, y + 42, w - 20, 30),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            step.labels[pick] if pick < len(step.labels) else str(pick))

        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(
            QtCore.QRectF(x + 10, y + 70, w - 20, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"{confidence:.3f}   {step.agreement} of {len(step.names)} agree")

        #  What actually runs on the robot, stated every time: a diagram of
        #  seven models fusing into one would otherwise imply the ensemble
        #  ships, and it does not.
        deployed_text = f"DEPLOYED  {step.deployed_name}"
        if step.deployed is not None:
            d_pick = step.decision(step.deployed)
            deployed_text += (f"  ->  "
                              f"{step.labels[d_pick] if d_pick < len(step.labels) else d_pick}"
                              f"  {step.deployed[d_pick]:.3f}")
        painter.setPen(QtGui.QColor(COL_WARN))
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(x + 10, y + 88, w - 20, 26),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            deployed_text)


class PanelWarmUp(QtCore.QRunnable):
    """
    Runs `DecisionPanel.warm_up()` on a worker thread.

    One call per app run, and it only touches fitted estimators read-only,
    so there is nothing to synchronise. The view polls `panel.warming` on
    the timer it already has rather than waiting on a signal -- one less
    cross-thread connection for a job whose only output is a flag.
    """

    def __init__(self, panel: "DecisionPanel"):
        super().__init__()
        self.panel = panel
        self.setAutoDelete(True)

    def run(self):                                         # pragma: no cover
        try:
            self.panel.warm_up()
        except Exception:
            self.panel.warming = False


class FeatureSpaceWidget(QtWidgets.QWidget):
    """
    Every training window as one point, in two dimensions, plus the live one.

    The projection is an unsupervised PCA computed by the trainer (see
    `feature_space_projection`). It is NOT an LDA, and that matters: an LDA
    picks its axes to separate the labels, so its picture looks clean even
    when the separation comes from electrode drift rather than from the
    gesture. What is drawn here is the structure that is really in the
    features, which on this dataset means the class-take confound is visible
    rather than hidden.

    Three colourings, because each answers a different question:

      * BY GESTURE  -- do the labels form their own clusters?
      * BY TAKE     -- or do the clusters actually follow the recording
                       session? If this looks tidier than BY GESTURE, the
                       model is reading the block (docs/EDA.md section 2).
      * BY LEVEL    -- where does amplitude sit? This is the picture of
                       docs/EDA.md R4: the gap between resting and the
                       effortful holds that a relaxed gesture falls into.

    A painter, not a source. With no projection it paints the reason.
    """

    MARGIN_L, MARGIN_R = 46, 12
    MARGIN_T, MARGIN_B = 10, 34

    def __init__(self, panel: "DecisionPanel", parent=None):
        super().__init__(parent)
        self.panel = panel
        self.mode = "label"
        self.live_point: Optional[Tuple[float, float]] = None
        self.live_label: Optional[str] = None
        self.trail: List[Tuple[float, float]] = []
        self.setMinimumHeight(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._bounds = None

    # -- inputs -----------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_live(self, point, label: str = None) -> None:
        """
        Place the live window. `None` clears it rather than leaving the last
        one on screen pretending to be current.
        """
        self.live_point = point
        self.live_label = label
        if point is not None:
            #  A short tail, so a gesture decaying out of its cluster is
            #  visible as a direction and not just a jumping dot.
            self.trail.append(point)
            del self.trail[:-40]
        self.update()

    def clear_trail(self) -> None:
        self.trail = []
        self.update()

    # -- geometry ---------------------------------------------------------
    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self.MARGIN_L, self.MARGIN_T,
            max(10, self.width() - self.MARGIN_L - self.MARGIN_R),
            max(10, self.height() - self.MARGIN_T - self.MARGIN_B))

    def _data_bounds(self):
        """Axis limits from the training cloud, padded, cached per load."""
        if self._bounds is not None:
            return self._bounds
        coords = np.asarray(self.panel.space["coords"], dtype=np.float64)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        #  Padded so the live point can sit outside the training cloud and
        #  still be drawn -- that it CAN is the interesting case.
        lo = lo - span * 0.12
        hi = hi + span * 0.12
        self._bounds = (lo, hi)
        return self._bounds

    def _to_px(self, rect, lo, hi, x, y):
        fx = (x - lo[0]) / max(1e-9, hi[0] - lo[0])
        fy = (y - lo[1]) / max(1e-9, hi[1] - lo[1])
        return (rect.left() + fx * rect.width(),
                rect.bottom() - fy * rect.height())

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor(COL_BG))

        if not self.panel.has_space:
            self._paint_missing(painter)
            painter.end()
            return

        space = self.panel.space
        rect = self._plot_rect()
        lo, hi = self._data_bounds()

        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BORDER), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRect(rect)

        coords = np.asarray(space["coords"], dtype=np.float64)
        truth = np.asarray(space["truth"], dtype=int)
        labels = list(self.panel.labels)

        #  Grouped into one polygon per colour and drawn with drawPoints:
        #  one call per colour instead of one per point, which is what keeps
        #  several thousand points off the GUI thread's budget.
        groups = self._colour_groups(space, truth, labels)
        for colour, idx in groups:
            if len(idx) == 0:
                continue
            poly = QtGui.QPolygonF()
            for x, y in coords[idx]:
                px, py = self._to_px(rect, lo, hi, x, y)
                poly.append(QtCore.QPointF(px, py))
            pen = QtGui.QPen(QtGui.QColor(colour), 3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(poly)

        self._paint_live(painter, rect, lo, hi)
        self._paint_axes(painter, rect, space)
        self._paint_legend(painter, rect, groups, space, labels)
        painter.end()

    def _colour_groups(self, space, truth, labels):
        """(colour, row indices) per colour, for the current mode."""
        if self.mode == "take" and space.get("take") is not None:
            take = np.asarray(space["take"], dtype=int)
            out = []
            for t in range(int(take.max()) + 1 if take.size else 0):
                out.append((class_color(t), np.flatnonzero(take == t)))
            return out
        if self.mode == "level" and space.get("level") is not None:
            level = np.asarray(space["level"], dtype=float)
            #  Fixed bands rather than quantiles, so the colours mean an
            #  amplitude and not a rank -- the point is to read the gap.
            edges = [0, 45, 90, 140, 200, 280, 1e9]
            ramp = ["#1e3a5f", "#2563eb", "#22c55e", "#f59e0b",
                    "#f97316", "#ef4444"]
            out = []
            for i in range(len(edges) - 1):
                out.append((ramp[i], np.flatnonzero(
                    (level >= edges[i]) & (level < edges[i + 1]))))
            return out
        return [(cfg.LABEL_COLORS.get(labels[k], class_color(k)),
                 np.flatnonzero(truth == k)) for k in range(len(labels))]

    def _paint_live(self, painter, rect, lo, hi) -> None:
        if not self.trail:
            return
        #  The tail, oldest faintest.
        for i, (x, y) in enumerate(self.trail[:-1]):
            px, py = self._to_px(rect, lo, hi, x, y)
            alpha = int(30 + 120 * (i / max(1, len(self.trail) - 1)))
            colour = QtGui.QColor(COL_TEXT)
            colour.setAlpha(alpha)
            painter.setPen(QtGui.QPen(colour, 2))
            painter.drawPoint(QtCore.QPointF(px, py))
        if self.live_point is None:
            return
        px, py = self._to_px(rect, lo, hi, *self.live_point)
        colour = QtGui.QColor(cfg.LABEL_COLORS.get(self.live_label or "", COL_TEXT))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BG), 4))
        painter.drawEllipse(QtCore.QPointF(px, py), 7, 7)
        painter.setPen(QtGui.QPen(colour, 2))
        painter.drawEllipse(QtCore.QPointF(px, py), 7, 7)
        painter.drawLine(QtCore.QPointF(px - 11, py), QtCore.QPointF(px - 4, py))
        painter.drawLine(QtCore.QPointF(px + 4, py), QtCore.QPointF(px + 11, py))
        painter.drawLine(QtCore.QPointF(px, py - 11), QtCore.QPointF(px, py - 4))
        painter.drawLine(QtCore.QPointF(px, py + 4), QtCore.QPointF(px, py + 11))

    def _paint_axes(self, painter, rect, space) -> None:
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        explained = space.get("explained") or [0.0, 0.0]
        painter.drawText(
            QtCore.QRectF(rect.left(), rect.bottom() + 4, rect.width(), 14),
            QtCore.Qt.AlignHCenter,
            f"PC1 -- {explained[0] * 100:.1f} % of the variance")
        painter.save()
        painter.translate(12, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QtCore.QRectF(-rect.height() / 2, -10,
                                       rect.height(), 14),
                         QtCore.Qt.AlignHCenter,
                         f"PC2 -- {explained[1] * 100:.1f} %")
        painter.restore()

    def _paint_legend(self, painter, rect, groups, space, labels) -> None:
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        if self.mode == "take":
            names = [f"take {i + 1}" for i in range(len(groups))]
            names = names[:10]
        elif self.mode == "level":
            names = ["<45", "45-90", "90-140", "140-200", "200-280", "280+"]
        else:
            names = labels
        x = rect.left() + 8
        y = rect.top() + 8
        for i, name in enumerate(names):
            if i >= len(groups):
                break
            colour, idx = groups[i]
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(colour))
            painter.drawRect(QtCore.QRectF(x, y + 3, 7, 7))
            painter.setPen(QtGui.QColor(COL_TEXT))
            text = f"{name}  {len(idx)}"
            painter.drawText(QtCore.QRectF(x + 11, y, 150, 12),
                             QtCore.Qt.AlignLeft, text)
            y += 13
            if y > rect.bottom() - 16:
                break

    def _paint_missing(self, painter) -> None:
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        if not self.panel.ready:
            text = ("No decision panel loaded, so there is no feature space "
                    "to draw.\nRetrain, or run:  python training/explain_votes.py")
        else:
            text = ("This panel has no feature-space projection.\nIt was "
                    "built before the view existed -- rerun "
                    "explain_votes.py to add one.")
        painter.drawText(self.rect().adjusted(14, 14, -14, -14),
                         QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap, text)


class FeatureSpaceView(QtWidgets.QWidget):
    """The FEATURE SPACE section: the colouring controls plus the scatter."""

    def __init__(self, panel: "DecisionPanel", predictor: "LivePredictor",
                 parent=None):
        super().__init__(parent)
        self.panel = panel
        self.predictor = predictor
        self._seen = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SP_SM)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(SP_SM)
        bar.addWidget(make_label("COLOUR BY", 9, False, COL_TEXT_DIM))
        self.mode_select = QtWidgets.QComboBox()
        space = panel.space or {}
        self.mode_select.addItem("gesture", "label")
        if space.get("take") is not None:
            self.mode_select.addItem("take it was cut from", "take")
        if space.get("level") is not None:
            self.mode_select.addItem("amplitude level", "level")
        self.mode_select.setToolTip(
            "gesture: do the labels cluster?\n"
            "take: or do the clusters follow the recording session instead? "
            "If this looks tidier than 'gesture', the model is reading the "
            "block, not the gesture.\n"
            "amplitude level: where the effort sits, and the gap a relaxed "
            "hold falls into.")
        self.mode_select.currentIndexChanged.connect(self._mode_changed)
        bar.addWidget(self.mode_select)

        self.clear_button = QtWidgets.QPushButton("CLEAR TRAIL")
        self.clear_button.setToolTip("Forget the live window's recent path.")
        self.clear_button.clicked.connect(lambda: self.plot.clear_trail())
        bar.addWidget(self.clear_button)
        bar.addStretch(1)
        self.readout = make_label("", 9, False, COL_TEXT_DIM)
        bar.addWidget(self.readout)
        layout.addLayout(bar)

        self.plot = FeatureSpaceWidget(panel, self)
        layout.addWidget(self.plot, 1)

        self.caption = make_label("", 9, False, COL_TEXT_DIM)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self._update_caption()
        self.controls_enabled(panel.has_space)

    def controls_enabled(self, on: bool) -> None:
        self.mode_select.setEnabled(on)
        self.clear_button.setEnabled(on)

    def _mode_changed(self) -> None:
        self.plot.set_mode(self.mode_select.currentData() or "label")
        self._update_caption()

    def _update_caption(self) -> None:
        space = self.panel.space or {}
        if not self.panel.has_space:
            self.caption.setText("")
            return
        head = (f"{space.get('n_points', 0)} of "
                f"{space.get('n_windows', 0)} training windows, projected "
                f"from {space.get('n_features', 0)} features by an "
                f"unsupervised PCA -- the axes were chosen without seeing "
                f"the labels, so any separation here is really in the data. ")
        mode = self.mode_select.currentData() or "label"
        if mode == "take":
            tail = ("Clusters that follow the take rather than the gesture "
                    "are the confound in docs/EDA.md section 2: the model "
                    "can score well by recognising which recording block it "
                    "is looking at.")
        elif mode == "level":
            tail = ("The amplitude bands. A gesture recorded only at full "
                    "effort leaves a gap above rest that a relaxed hold "
                    "falls into, which is docs/EDA.md R4.")
        else:
            tail = ("The ring is the window the classifier just used -- the "
                    "same feature vector, placed by the same transform. "
                    "Watch it leave its cluster when a hold relaxes.")
        self.caption.setText(head + tail)

    def refresh_live(self) -> None:
        """Place the newest predicted window. Called on the dialog's timer."""
        if not self.panel.has_space:
            return
        predictor = self.predictor
        features = predictor.last_features
        if features is None:
            self.readout.setText("no live window yet")
            return
        #  Keyed on the prediction's own timestamp, exactly as the trail
        #  does: `id(features)` would be recycled by the allocator and could
        #  make a new window look like the previous one.
        last = predictor.last
        stamp = getattr(last, "unix_time", 0.0) if last else 0.0
        if stamp and stamp == self._seen:
            return
        self._seen = stamp
        point = self.panel.project(features)
        if point is None:
            self.readout.setText("live vector does not fit this projection")
            return
        label = getattr(last, "raw_label", None) if last else None
        self.plot.set_live(point, label)
        self.readout.setText(
            f"live: PC1 {point[0]:+.2f}  PC2 {point[1]:+.2f}"
            + (f"  -> {label}" if label else ""))


class DecisionTrailView(QtWidgets.QWidget):
    """
    The DECISION TRAIL section of MODEL STATS: controls plus the drawing.

    Two sources, and which one is on screen is never ambiguous:

      * LIVE follows the classifier. While PREDICT is running every new
        window re-runs all the candidates on the very feature vector the
        deployed model just used. With nothing streaming it holds the last
        window and says so.
      * RECORDED scrubs the out-of-fold trail stored at training time, which
        is where the honest numbers are.

    With no panel file the whole section reduces to one sentence explaining
    how to get one. Nothing here can show a number the panel did not supply.
    """

    def __init__(self, panel: "DecisionPanel", predictor: "LivePredictor",
                 parent=None):
        super().__init__(parent)
        self.panel = panel
        self.predictor = predictor
        self.row = 0
        self.live = True
        self._rows: List[int] = []
        self._splits: List[int] = []
        self._live_seen = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SP_SM)

        self.controls = QtWidgets.QWidget()
        bar = QtWidgets.QHBoxLayout(self.controls)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(SP_SM)

        self.live_button = QtWidgets.QPushButton("LIVE")
        self.live_button.setCheckable(True)
        self.live_button.setChecked(True)
        self.live_button.setToolTip(
            "Follow the classifier: every predicted window is re-run through "
            "all the candidates.")
        self.live_button.clicked.connect(lambda: self._set_live(True))
        self.recorded_button = QtWidgets.QPushButton("RECORDED")
        self.recorded_button.setCheckable(True)
        self.recorded_button.setToolTip(
            "Scrub the out-of-fold decisions captured during training.")
        self.recorded_button.clicked.connect(lambda: self._set_live(False))
        bar.addWidget(self.live_button)
        bar.addWidget(self.recorded_button)

        self.rec_select = QtWidgets.QComboBox()
        self.rec_select.setMinimumWidth(190)
        self.rec_select.currentIndexChanged.connect(self._pick_recording)
        bar.addWidget(self.rec_select, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._pick_window)
        bar.addWidget(self.slider, 2)

        self.position = make_label("--", 9, False, COL_TEXT_DIM)
        bar.addWidget(self.position)

        self.split_button = QtWidgets.QPushButton("NEXT DISAGREEMENT")
        self.split_button.setToolTip(
            "Jump to the next recorded window where the candidates did not "
            "all pick the same gesture.")
        self.split_button.clicked.connect(self._next_split)
        bar.addWidget(self.split_button)
        layout.addWidget(self.controls)

        self.trail = DecisionTrailWidget()
        layout.addWidget(self.trail, 1)

        self.footer = make_label("", 9, False, COL_TEXT_DIM)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

        self._build()

    # -- setup ----------------------------------------------------------------

    def _build(self) -> None:
        panel = self.panel
        if not panel.ready:
            self.controls.setVisible(False)
            self.trail.set_empty(
                "No decision panel.\n\n" + panel.status +
                "\n\nRetrain (RETRAIN, or python training/train_model.py) and this "
                "view will show every candidate's vote on each window.")
            self.footer.setText("")
            return

        stale = panel.stale_reason(self.predictor)
        if stale:
            self.controls.setVisible(False)
            self.trail.set_empty(
                "The decision panel does not match the loaded model.\n\n"
                + stale)
            self.footer.setText("")
            return

        #  Pay the candidates' first-call cost on a worker thread now, so the
        #  first live window does not stall the GUI for ~2.5 s.
        if not panel.warm and not panel.warming:
            QtCore.QThreadPool.globalInstance().start(PanelWarmUp(panel))

        self._splits = panel.split_windows()
        self.rec_select.blockSignals(True)
        self.rec_select.clear()
        for index, entry in enumerate(panel.recordings):
            wrong = [n for n in panel.names
                     if not entry["models"].get(n, {}).get("correct", True)]
            text = (f"{entry['label']}  {entry['uid']}  "
                    f"{entry['n_windows']}w  fold {entry['fold'] + 1}")
            if wrong:
                text += f"  [{len(wrong)} wrong]"
            self.rec_select.addItem(text, index)
        self.rec_select.blockSignals(False)
        self.split_button.setEnabled(bool(self._splits))

        counts = ", ".join(
            f"{name} {sum(1 for r in panel.recordings if r['label'] == name)}"
            for name in panel.labels)
        verified = panel.fusion.get("verified")
        difference = panel.fusion.get("max_abs_diff_vs_mean")
        self.footer.setText(
            f"Panel built {panel.created_iso[:16].replace('T', ' ')} from "
            f"{panel.n_windows} windows of {panel.n_recordings} recordings "
            f"({counts}).  Fusion: {panel.fusion.get('rule', '?')}"
            + (f"; checked against the stored ensemble, max difference "
               f"{difference:.1e}." if isinstance(difference, float) else ".")
            + ("" if verified else
               "  The check did not pass -- read the ensemble's own output, "
               "not the mean.")
            + f"  Cross-validation: {panel.protocol.get('cv', '?')}, "
              f"{panel.protocol.get('n_splits', '?')} folds."
        )
        self._set_live(True)

    def _set_live(self, live: bool) -> None:
        self.live = bool(live)
        self.live_button.setChecked(self.live)
        self.recorded_button.setChecked(not self.live)
        for widget in (self.rec_select, self.slider, self.split_button):
            widget.setEnabled(not self.live)
        self.split_button.setEnabled(not self.live and bool(self._splits))
        if self.live:
            self.refresh_live(force=True)
        else:
            if self.rec_select.count() and not self._rows:
                self._pick_recording(self.rec_select.currentIndex())
            self._show_recorded()

    # -- recorded -------------------------------------------------------------

    def _pick_recording(self, index: int) -> None:
        if index < 0 or index >= len(self.panel.recordings):
            return
        rows = self.panel.windows_of(index)
        self._rows = rows
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(rows) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.row = rows[0] if rows else 0
        if not self.live:
            self._show_recorded()

    def _pick_window(self, value: int) -> None:
        rows = self._rows
        if 0 <= value < len(rows):
            self.row = rows[value]
            if not self.live:
                self._show_recorded()

    def _next_split(self) -> None:
        if not self._splits:
            return
        later = [r for r in self._splits if r > self.row]
        target = later[0] if later else self._splits[0]
        index = self.panel.recording_index_of(target)
        if index >= 0:
            if index != self.rec_select.currentIndex():
                self.rec_select.setCurrentIndex(index)
            if target in self._rows:
                self.slider.setValue(self._rows.index(target))
        self.row = target
        self._show_recorded()

    def _show_recorded(self) -> None:
        step = self.panel.recorded_step(self.row)
        if step is None:
            self.trail.set_empty(
                "This panel carries no per-window trail; only the live view "
                "and the scores are available.")
            return
        rows = self._rows
        position = rows.index(self.row) + 1 if self.row in rows else 1
        self.position.setText(f"{position} / {max(1, len(rows))}")
        truth = self.panel.recorded_truth(self.row)
        fused = step.fused if step.fused is not None else step.fused_from_mean
        correct = step.decision(fused) == truth
        self.trail.set_step(
            step, channels=self.panel.recorded_channels(self.row),
            caption=("The vote was " + ("right" if correct else "WRONG")
                     + " on this window.  These are out-of-fold numbers: the "
                     "recording was held out of the fold that produced them, "
                     "so they are the honest ones."),
            cv=self.panel.cv)

    # -- live -----------------------------------------------------------------

    def refresh_live(self, force: bool = False) -> None:
        """Re-run the candidates on the newest predicted window."""
        if not self.live or not self.panel.ready:
            return
        if self.panel.stale_reason(self.predictor):
            return
        if self.panel.warming:
            self.trail.set_empty(
                "Preparing the candidates...\n\nEach one's first prediction "
                "builds its kernel; this happens once. Live windows appear as "
                "soon as it finishes.")
            self.position.setText("...")
            return

        features = self.predictor.last_features
        if features is None:
            self.trail.set_empty(
                "Waiting for a live window.\n\nPress PREDICT in the LIVE "
                "CLASSIFIER panel, or switch to RECORDED to scrub the "
                "windows captured during training.")
            self.position.setText("--")
            return

        last = self.predictor.last
        stamp = getattr(last, "unix_time", 0.0) if last else 0.0
        if not force and stamp == self._live_seen:
            return
        self._live_seen = stamp

        step = self.panel.step_for_features(features, self.predictor)
        if step is None:
            self.trail.set_empty("The decision panel could not run on this "
                                 "window: " + self.panel.status)
            return
        window = self.predictor.last_window
        channels = None
        if window is not None and np.asarray(window).ndim == 2:
            channels = np.mean(np.asarray(window, dtype=float), axis=0)
        age = time.time() - stamp if stamp else None
        self.position.setText("live" if (age is not None and age < 2.0)
                              else "held")
        self.trail.set_step(
            step, channels=channels,
            caption=("Every candidate ran on the same feature vector the "
                     "deployed model just used. They were fitted on the whole "
                     "dataset, so agreement here shows how the vote forms -- "
                     "it is not an accuracy estimate. For that, use RECORDED "
                     "or the scores below."),
            cv=self.panel.cv)

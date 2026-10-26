"""
Small shared widgets and layout helpers.
"""

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_BAD, COL_BORDER, COL_GOOD, COL_PANEL, COL_TEXT_DIM, COL_WARN, SP_SM)
from emg_app.sequence import STATE_COLORS, STATE_IDLE
from emg_app.gui.qt import QtCore, QtGui, QtWidgets


class WheelGuard(QtCore.QObject):
    """
    Stops the mouse wheel from changing a control it is merely passing over.

    Qt's default is that a wheel event over a combo box, spin box or slider
    changes its value. In a tall scrolling side panel that means scrolling
    past a control silently edits it -- the gesture, the window length, a
    hyperparameter -- and the operator never sees it happen. Here that would
    be written into recording metadata, so it is a data-integrity bug, not a
    nuisance.

    The rule: a wheel event is only accepted when the control has keyboard
    focus, i.e. the operator deliberately clicked into it first. Otherwise the
    event is passed up to the scroll area, which is what they meant.
    """

    GUARDED = ()          # filled in below, once QtWidgets is imported

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel and isinstance(obj, self.GUARDED):
            if not obj.hasFocus():
                event.ignore()
                return True
        return False


def install_wheel_guard(app) -> "WheelGuard":
    """Apply the wheel guard to every control in the application."""
    WheelGuard.GUARDED = (
        QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox,
        QtWidgets.QSlider,
    )
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    # Wheel-over-without-focus is now inert, so these need a click to change:
    # make that click actually give them focus.
    app._wheel_guard = guard
    return guard


def gear_icon(color: str = "#9aa3b2", size: int = 64) -> "QtGui.QIcon":
    """
    A settings cog, painted rather than typed.

    A "⚙" character depends on which font Qt finds it in: it renders on a
    normal Windows desktop and as an empty box on the offscreen platform the
    UI guide's screenshots are made on. Drawing it costs twenty lines and
    looks the same everywhere. Eight teeth on a ring, with a hole -- the
    even-odd fill rule makes the hole from the inner circle.
    """
    import math

    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    c = size / 2.0
    r_out, r_in, r_hole = size * 0.47, size * 0.36, size * 0.16
    teeth = 8
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.OddEvenFill)
    points = []
    for k in range(teeth):
        a = 2 * math.pi * k / teeth
        # half-widths in radians: the tooth top is narrower than its root
        for r, da in ((r_in, -0.30), (r_out, -0.17), (r_out, 0.17),
                      (r_in, 0.30)):
            points.append(QtCore.QPointF(c + r * math.cos(a + da),
                                         c + r * math.sin(a + da)))
    path.addPolygon(QtGui.QPolygonF(points))
    path.closeSubpath()
    path.addEllipse(QtCore.QPointF(c, c), r_hole, r_hole)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawPath(path)
    painter.end()
    return QtGui.QIcon(pix)


def _marker_caption(marker: dict) -> str:
    """Elided caption for a marker dict, mirroring Marker.short_text()."""
    text = " ".join(str(marker.get("text", "")).split())
    if not text:
        return "BREAK" if marker.get("kind") == "break" else "note"
    limit = cfg.MARKER_LABEL_MAX_CHARS
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "\u2026"


def make_label(text: str, size: int = 12, bold: bool = False,
               color: str = None, align=None):
    label = QtWidgets.QLabel(text)
    font = label.font()
    font.setPointSize(size)
    font.setBold(bold)
    label.setFont(font)
    if color:
        label.setStyleSheet(f"color: {color}; background: transparent;")
    else:
        label.setStyleSheet("background: transparent;")
    if align is not None:
        label.setAlignment(align)
    return label


# ------------------------------------------------------------------------------
# Acquisition -> GUI bridge
# ------------------------------------------------------------------------------


class AcquisitionBridge(QtCore.QObject):
    """
    Turns worker-thread callbacks into Qt signals.

    Qt queues cross-thread signal emissions, so this is the one safe way to let
    the acquisition thread talk to widgets. It carries no data payload per
    sample -- the GUI reads samples from the ring buffer on its own timer,
    which keeps the event loop free no matter how fast the armband streams.
    """
    state_changed = QtCore.Signal(str, str)
    log_line = QtCore.Signal(str)


# ------------------------------------------------------------------------------
# Status banner
# ------------------------------------------------------------------------------


def available_screen_size(widget=None):
    """
    The usable area of the screen a widget will appear on, in the same logical
    pixels Qt lays out in.

    Falls back to the primary screen, and then to a conservative 1280x720,
    because a headless/offscreen run has no screen at all and must not crash
    here.
    """
    screen = None
    if widget is not None:
        handle = widget.window().windowHandle() if widget.window() else None
        screen = handle.screen() if handle is not None else None
        if screen is None and widget.parent() is not None:
            screen = widget.parent().screen()
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:                                       # pragma: no cover
        return 1280, 720
    rect = screen.availableGeometry()
    return max(640, rect.width()), max(480, rect.height())


def fit_to_screen(widget, want_width: int, want_height: int,
                  min_width: int = None, min_height: int = None,
                  centre: bool = True):
    """
    Size a window to `want_*`, but never beyond the screen it opens on.

    Also clamps the MINIMUM size, which is the part that actually matters: a
    minimum larger than the screen cannot be resized away by the operator, so
    the window would stay off the edge of the display no matter what they did.

    Returns (width, height) actually applied.
    """
    avail_w, avail_h = available_screen_size(widget)
    cap_w = int(avail_w * cfg.WINDOW_FIT_FRACTION)
    cap_h = int(avail_h * cfg.WINDOW_FIT_FRACTION)

    width = max(320, min(int(want_width), cap_w))
    height = max(240, min(int(want_height), cap_h))

    if min_width is not None or min_height is not None:
        widget.setMinimumSize(
            max(320, min(int(min_width or 0), cap_w)),
            max(240, min(int(min_height or 0), cap_h)),
        )

    widget.resize(width, height)

    if centre:
        # Qt centres a dialog on its parent by itself, but a window that was
        # just clamped can still be positioned partly off-screen, so place it
        # explicitly inside the available area.
        screen = (widget.window().windowHandle().screen()
                  if widget.window() and widget.window().windowHandle()
                  else QtGui.QGuiApplication.primaryScreen())
        if screen is not None:
            rect = screen.availableGeometry()
            widget.move(rect.left() + max(0, (rect.width() - width) // 2),
                        rect.top() + max(0, (rect.height() - height) // 2))

    return width, height


def scroll_dialog_body(dialog, keep_last: int = 1) -> None:
    """
    Put everything in a dialog's layout into a scroll area, except the last
    `keep_last` items (the button row), which stay pinned underneath it.

    Windows display scaling is why this exists. At 125 % a 1920x1080 screen
    is 1536x826 logical pixels, at 150 % it is 1280x688, and Qt lays out in
    logical pixels -- so a dialog whose content needs 950 px is simply taller
    than the screen, and fit_to_screen cannot shrink it below its layout's
    minimum. Scrolling is the only way the bottom stays reachable. When it
    fits, no scrollbar appears and nothing looks different.

    Call it once the layout is complete, then fit_to_screen.
    """
    layout = dialog.layout()
    if layout is None:
        return
    tail = []
    for _ in range(min(keep_last, layout.count())):
        tail.insert(0, layout.takeAt(layout.count() - 1))
    margins = layout.contentsMargins()

    body = QtWidgets.QWidget()
    body.setLayout(layout)                    # Qt takes it off the dialog
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setWidget(body)

    outer = QtWidgets.QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, margins.bottom())
    outer.setSpacing(layout.spacing())
    outer.addWidget(scroll, 1)
    for item in tail:
        child = item.layout() or item.widget()
        if isinstance(child, QtWidgets.QLayout):
            child.setParent(None)
            child.setContentsMargins(margins.left(), 0, margins.right(), 0)
            outer.addLayout(child)
        elif child is not None:
            outer.addWidget(child)
    dialog._scroll_body = scroll


def fit_table_height(table, rows: int) -> None:
    """
    Make a table exactly tall enough for `rows`, measured from the table's own
    row and header heights rather than an assumed 24 px: at 125-150 % scaling
    the font makes rows taller, and a guessed height cuts the table short so
    it scrolls inside itself.
    """
    row_h = table.verticalHeader().defaultSectionSize()
    head_h = table.horizontalHeader().sizeHint().height()
    table.setFixedHeight(row_h * max(1, rows) + head_h
                         + 2 * table.frameWidth() + 2)


class CollapsibleSection(QtWidgets.QWidget):
    """
    A panel with a header you can click to fold it away.

    The side panel stacks nine groups; on anything smaller than a big monitor
    they do not all fit, and scrolling past the one you want is the single
    most common irritation in this UI. Folding the ones you are not using is
    the cheapest fix that does not move anything around.

    The content widget is reparented, not rebuilt, so every `self.*` reference
    inside it keeps working and no behaviour changes.
    """

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, content: QtWidgets.QWidget,
                 expanded: bool = True, parent=None):
        super().__init__(parent)
        self._content = content

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QtWidgets.QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded
                                 else QtCore.Qt.RightArrow)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        self.header.setObjectName("sectionHeader")
        self.header.setMinimumHeight(26)
        self.header.toggled.connect(self._on_toggled)
        layout.addWidget(self.header)

        # The content sits in a named frame so the stylesheet can draw the
        # card that joins it to its header.
        body = QtWidgets.QWidget()
        body.setObjectName("sectionBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        body_layout.setSpacing(0)
        body_layout.addWidget(content)
        layout.addWidget(body)
        self._body = body

        body.setVisible(bool(expanded))

    def _on_toggled(self, expanded: bool) -> None:
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded
                                 else QtCore.Qt.RightArrow)
        self._body.setVisible(expanded)
        self.toggled.emit(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(bool(expanded))

    @property
    def expanded(self) -> bool:
        return self.header.isChecked()


def collapsible(group: QtWidgets.QGroupBox, expanded: bool = True):
    """
    Wrap an existing QGroupBox in a CollapsibleSection.

    The box's own title moves to the header, so there is one title rather than
    two, and the frame is dropped because the header already delimits the
    section.
    """
    title = group.title()
    group.setTitle("")
    group.setFlat(True)
    return CollapsibleSection(title, group, expanded)


class StatusBanner(QtWidgets.QFrame):
    """The big, unmissable "what should I do right now" panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        #  104 px is a sixth of the window on a 1080p screen at 150 % scaling
        #  (688 logical px tall), taken straight out of the graph. On a short
        #  screen the banner keeps its colour and text and loses the padding.
        _w, avail_h = available_screen_size()
        self.setMinimumHeight(104 if avail_h >= 800 else 64)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self._color = STATE_COLORS[STATE_IDLE]

        self.big = make_label("READY", 34, True, "#ffffff",
                              QtCore.Qt.AlignCenter)
        self.small = make_label("Press START SESSION", 12, False,
                                "rgba(255,255,255,190)", QtCore.Qt.AlignCenter)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)
        layout.addWidget(self.big)
        layout.addWidget(self.small)
        layout.addWidget(self.bar)

        self._apply_color()

    def _apply_color(self) -> None:
        self.setStyleSheet(
            f"StatusBanner {{ background: {self._color};"
            f" border-radius: 10px; border: 1px solid rgba(255,255,255,40); }}"
            f"QProgressBar {{ background: rgba(0,0,0,70); border: none;"
            f" border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: rgba(255,255,255,220);"
            f" border-radius: 4px; }}"
        )

    def update_state(self, state: str, big: str, small: str,
                     fraction: float = None) -> None:
        color = STATE_COLORS.get(state, STATE_COLORS[STATE_IDLE])
        if color != self._color:
            self._color = color
            self._apply_color()
        if self.big.text() != big:
            self.big.setText(big)
        if self.small.text() != small:
            self.small.setText(small)
        self.bar.setVisible(fraction is not None)
        if fraction is not None:
            self.bar.setValue(int(np.clip(fraction, 0.0, 1.0) * 1000))


def wrap_label(text: str, size: int = 12, bold: bool = False,
               colour: str = None):
    """
    `make_label()` plus word wrap, as one expression.

    It exists because `layout.addWidget(make_label(...))` returns None, so a
    `.setWordWrap(True)` chained onto it raises -- a mistake that is easy to
    write, invisible on the page, and only fires when the dialog is opened.
    """
    label = make_label(text, size, bold, colour)
    label.setWordWrap(True)
    return label


def _stat_tile(caption: str, value, note: str, good: float = 0.85,
               warn: float = 0.6, neutral: bool = False):
    """
    One headline number in a rounded box. `value` is a fraction or None.

    Shared by both statistics dialogs so the neural net's numbers are
    presented in exactly the same shape as the deployed model's -- a
    side-by-side comparison in which one side is styled more generously is
    not a comparison.

    `neutral` draws the number without a verdict colour, and the DC-only
    control uses it. For every other accuracy here, higher is better; for
    that one it is not -- a HIGH DC-only score means the wearers differ in
    gain rather than that gestures were learnt (CLAUDE.md 8k). Painting it
    green would state the opposite of what it means.
    """
    box = QtWidgets.QFrame()
    box.setObjectName("card")
    box.setStyleSheet(f"QFrame#card {{ background: {COL_PANEL}; border: 1px solid "
                      f"{COL_BORDER}; border-radius: 8px; }}")
    column = QtWidgets.QVBoxLayout(box)
    column.setSpacing(1)
    column.addWidget(make_label(caption, 9, False, COL_TEXT_DIM))
    text = "--" if value is None else f"{value * 100:.1f} %"
    colour = COL_TEXT_DIM
    if value is not None and not neutral:
        colour = (COL_GOOD if value >= good
                  else COL_WARN if value >= warn else COL_BAD)
    column.addWidget(make_label(text, 22, True, colour))
    note_label = make_label(note, 9, False, COL_TEXT_DIM)
    note_label.setWordWrap(True)
    column.addWidget(note_label)
    return box

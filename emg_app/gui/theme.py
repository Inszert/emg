"""
The stylesheet and class colours.
"""


from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BG, COL_BORDER, COL_PANEL, COL_PANEL_2, COL_TEXT,
    COL_TEXT_DIM, FS_MD, RAD_SM, SP_SM, SP_XS)
import os

from emg_app.gui.qt import pg
from emg_app.gui.widgets import install_wheel_guard


def _asset(name: str) -> str:
    """url() of a glyph in assets/icons, or "none" if it is not there, so a
    missing file costs a tick mark, never a stylesheet parse error."""
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "assets", "icons", name))
    if not os.path.isfile(path):
        return "none"
    return 'url("%s")' % path.replace("\\", "/")


IMG_CHECK, IMG_DOWN, IMG_UP = (_asset("ui_check.png"), _asset("ui_down.png"),
                               _asset("ui_up.png"))
COL_DISABLED_TEXT = "#6b7384"      # 3.6:1 on the disabled fill (was 2.9:1)
COL_CONTROL_BORDER = "#4a5263"     # 2.6:1 on the field fill: a visible box

STYLESHEET = f"""
QWidget {{
    background: {COL_BG};
    color: {COL_TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    background: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {COL_TEXT_DIM};
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton {{
    background: {COL_PANEL_2};
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 16px;
    font-weight: 600;
}}
QPushButton:hover   {{ background: #272c37; border-color: #4a5263; }}
QPushButton:pressed {{ background: #121620; }}
QPushButton:focus   {{ border-color: #3b7a99; }}
QPushButton:disabled {{ color: {COL_DISABLED_TEXT}; background: #14171e; border-color: #22262f; }}
QPushButton#primary   {{ background: #0b5f86; border-color: {COL_ACCENT}; color: #eaf7ff; }}
QPushButton#primary:hover {{ background: #0d74a3; }}
QPushButton#primary:pressed {{ background: #09506f; }}
QPushButton#primary:focus {{ border-color: #bae6fd; }}
QPushButton#primary:disabled {{ color: {COL_DISABLED_TEXT}; background: #14171e; border-color: #22262f; }}
QPushButton#danger    {{ background: #7f1d1d; border-color: #b91c1c; color: #fee2e2; }}
QPushButton#danger:hover  {{ background: #991b1b; }}
QPushButton#warn      {{ background: #78350f; border-color: #b45309; color: #fef3c7; }}
QPushButton#warn:hover    {{ background: #92400e; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    padding: 6px 8px;
    min-height: 16px;
    selection-background-color: {COL_ACCENT};
    selection-color: #04121a;
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {COL_CONTROL_BORDER};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COL_ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled {{ color: {COL_DISABLED_TEXT}; background: #0d1016; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: {IMG_DOWN}; width: 10px; height: 10px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; border: none; border-left: 1px solid {COL_BORDER};
    border-top-right-radius: 5px; background: transparent;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; border: none; border-left: 1px solid {COL_BORDER};
    border-bottom-right-radius: 5px; background: transparent;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {COL_PANEL_2};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: {IMG_UP}; width: 8px; height: 8px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: {IMG_DOWN}; width: 8px; height: 8px;
}}
QComboBox QAbstractItemView {{
    background: {COL_PANEL_2};
    border: 1px solid {COL_BORDER};
    selection-background-color: #0b5f86;
}}
/* Text widgets sit on whatever panel they are in. Without this each label
   and checkbox paints the page colour as a visible stripe on a panel. */
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox:disabled, QLabel:disabled {{ color: {COL_DISABLED_TEXT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {COL_CONTROL_BORDER}; background: #0b0e13;
}}
QCheckBox::indicator:hover {{ border-color: {COL_ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {COL_ACCENT}; border-color: {COL_ACCENT}; image: {IMG_CHECK};
}}
QCheckBox::indicator:disabled {{
    border-color: #2a2f3a; background: #0d1016;
}}
QCheckBox::indicator:checked:disabled {{
    background: #1d5670; border-color: #1d5670;
}}
QCheckBox:focus {{ color: #ffffff; }}
QTabWidget::pane {{ border: 1px solid {COL_BORDER}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: {COL_PANEL}; color: {COL_TEXT_DIM};
    border: 1px solid {COL_BORDER}; border-bottom: none;
    padding: 6px 14px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:hover {{ color: {COL_TEXT}; }}
QTabBar::tab:selected {{
    color: {COL_TEXT}; background: {COL_PANEL_2};
    border-top: 2px solid {COL_ACCENT};
}}
QMenu {{
    background: {COL_PANEL_2}; border: 1px solid {COL_BORDER}; padding: 4px;
}}
QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: #0b5f86; }}
QMenu::item:disabled {{ color: {COL_DISABLED_TEXT}; }}
QMenu::separator {{ height: 1px; background: {COL_BORDER}; margin: 4px 8px; }}
QTableWidget {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    gridline-color: #20242d;
}}
QHeaderView::section {{
    background: {COL_PANEL_2};
    border: none;
    border-bottom: 1px solid {COL_BORDER};
    padding: 6px 8px;
    font-weight: 600;
    color: {COL_TEXT_DIM};
}}
QTableWidget::item:selected {{ background: #0b5f86; }}
QProgressBar {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {COL_ACCENT}; border-radius: 4px; }}
QStatusBar {{ background: {COL_PANEL}; border-top: 1px solid {COL_BORDER}; }}
QStatusBar::item {{ border: none; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {COL_BORDER}; }}
QToolTip {{
    background: {COL_PANEL_2}; color: {COL_TEXT};
    border: 1px solid {COL_BORDER}; padding: 6px 8px;
}}
"""


# Fusion's default selection is a saturated magenta that appears nowhere else
# in this app. Selections are frequent here -- the recordings table, the model
# roster, the take history -- so they get the accent colour instead.
SELECTION_QSS = f"""
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {COL_ACCENT}33;
    color: {COL_TEXT};
    border-left: 3px solid {COL_ACCENT};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background: {COL_PANEL_2};
}}
QTableWidget::item:selected {{
    background: {COL_ACCENT}33;
    color: {COL_TEXT};
}}
QListWidget, QTreeWidget {{
    outline: none;
}}
QToolButton#sectionHeader {{
    background: {COL_PANEL_2};
    color: {COL_TEXT};
    border: 1px solid {COL_BORDER};
    border-radius: {RAD_SM}px;
    padding: {SP_XS}px {SP_SM}px;
    font-size: {FS_MD}px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-align: left;
    margin-top: {SP_SM}px;
}}
QToolButton#sectionHeader:hover {{
    background: {COL_BORDER};
    border-color: {COL_ACCENT};
}}
QToolButton#sectionHeader:checked {{
    border-left: 3px solid {COL_ACCENT};
    background: {COL_BORDER};
}}

/* An open section is a framed card joined to its header, so it is obvious
   where it starts and stops. Before this the content just floated. */
QWidget#sectionBody {{
    background: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-top: none;
    border-radius: 0 0 {RAD_SM}px {RAD_SM}px;
}}

/* Thin, flat scrollbars. The Fusion default is a chunky 3-D affair with
   arrow buttons at both ends, and there are a dozen of them on screen. */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {COL_BORDER};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {COL_TEXT_DIM};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0; border: none; background: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
"""


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET + SELECTION_QSS)
    install_wheel_guard(app)
    pg.setConfigOptions(
        antialias=cfg.PLOT_ANTIALIAS,
        useOpenGL=cfg.PLOT_USE_OPENGL,
        background=COL_BG,
        foreground=COL_TEXT_DIM,
        imageAxisOrder="row-major",
    )


#  Per-class colours for the trail. Deliberately not CHANNEL_COLORS: those
#  mean "electrode 3", and reusing them for "the class FIST" would make two
#  unrelated things look related on the same screen.
CLASS_COLORS = ["#fb923c", "#38bdf8", "#a78bfa", "#34d399",
                "#f472b6", "#facc15", "#60a5fa", "#fb7185"]


def class_color(index: int) -> str:
    return CLASS_COLORS[index % len(CLASS_COLORS)]

"""
The application icon. Never raises: a missing or unreadable file gives an empty
QIcon, which Qt treats as "no icon".
"""

import os
import sys

from emg_app.gui.qt import QtGui

ICON_SIZES = (16, 32, 48, 64, 128, 256)
APP_USER_MODEL_ID = "EMG.MyoDatasetTool.1"

# <project>/assets/icons, resolved from this file so the cwd does not matter.
ICON_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "icons"))


def app_icon(icon_dir: str = None) -> "QtGui.QIcon":
    """A QIcon built from icon_<size>.png (falling back to app.ico)."""
    icon = QtGui.QIcon()
    try:
        folder = icon_dir or ICON_DIR
        for size in ICON_SIZES:
            path = os.path.join(folder, f"icon_{size}.png")
            if os.path.isfile(path):
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    icon.addPixmap(pix)
        if icon.isNull():
            ico = os.path.join(folder, "app.ico")
            if os.path.isfile(ico):
                icon = QtGui.QIcon(ico)
    except Exception:                                    # pragma: no cover
        return QtGui.QIcon()
    return icon


def set_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Windows: own taskbar identity, so the icon is ours and not python's."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception:
        return False


def apply_app_icon(app, window=None, icon_dir: str = None) -> "QtGui.QIcon":
    """Set the icon on the application (so every dialog inherits it) and,
    optionally, on one window. Returns the icon; never raises. With
    `APP_ICON_ENABLED` off (settings, General) nothing is set and the toolkit's
    default icon stays."""
    try:
        from emg_app import config as cfg
        if not getattr(cfg, "APP_ICON_ENABLED", True):
            return QtGui.QIcon()
        set_app_user_model_id()
        icon = app_icon(icon_dir)
        if not icon.isNull():
            app.setWindowIcon(icon)
            if window is not None:
                window.setWindowIcon(icon)
        return icon
    except Exception:                                    # pragma: no cover
        return QtGui.QIcon()

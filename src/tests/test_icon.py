"""
The application icon: loads, never raises, falls back to an empty icon.
Offscreen, no hardware, no torch.   python tests/test_icon.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6 import QtGui, QtWidgets                              # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from emg_app.gui import icon as ic                                # noqa: E402

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


icon = ic.app_icon()
check("icon is not null", not icon.isNull())
sizes = sorted(s.width() for s in icon.availableSizes())
check("all six sizes available", sizes == [16, 32, 48, 64, 128, 256])
for n in (16, 32, 64, 256):
    pm = icon.pixmap(n, n)
    check(f"{n}px pixmap non-empty", not pm.isNull() and pm.width() > 0)

ico = QtGui.QIcon(os.path.join(ic.ICON_DIR, "app.ico"))
check("app.ico has the six sizes",
      sorted(s.width() for s in ico.availableSizes())
      == [16, 32, 48, 64, 128, 256])

with tempfile.TemporaryDirectory() as empty:
    missing = os.path.join(empty, "nope")
    for label, folder in (("missing dir", missing), ("empty dir", empty)):
        try:
            got = ic.app_icon(folder)
            check(f"{label}: no exception, null icon", got.isNull())
        except Exception as exc:                                 # noqa: BLE001
            check(f"{label}: raised {exc!r}", False)
    with open(os.path.join(empty, "icon_32.png"), "wb") as fh:
        fh.write(b"not a png")
    check("corrupt png: null icon, no exception", ic.app_icon(empty).isNull())
    w = QtWidgets.QWidget()
    check("apply with null icon does not raise",
          ic.apply_app_icon(app, w, empty).isNull())

w = QtWidgets.QWidget()
got = ic.apply_app_icon(app, w)
check("apply sets the app icon", not app.windowIcon().isNull())
check("apply sets the window icon", not w.windowIcon().isNull())
check("AppUserModelID call never raises",
      ic.set_app_user_model_id() in (True, False))

#  The switch (settings -> General -> Appearance): off = the toolkit's default.
from emg_app import config as _cfg                                 # noqa: E402
app.setWindowIcon(QtGui.QIcon())
w_off = QtWidgets.QWidget()
_cfg.APP_ICON_ENABLED = False
try:
    off = ic.apply_app_icon(app, w_off)
    check("icon switched OFF: nothing is set (default toolkit icon)",
          off.isNull() and app.windowIcon().isNull()
          and w_off.windowIcon().isNull())
finally:
    _cfg.APP_ICON_ENABLED = True
ic.apply_app_icon(app, w_off)
check("...and ON again sets it", not app.windowIcon().isNull()
      and not w_off.windowIcon().isNull())

import emg_app.app as appmod                                      # noqa: E402,F401
check("emg_app.app still imports", hasattr(appmod, "main"))

print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)

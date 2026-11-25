"""
Qt imports, guarded so the data layer can be used without a display.

Every GUI module imports Qt from here. The acquisition thread never touches
a widget: it emits through a QObject bridge and the GUI pulls from the ring
buffer on its own timer.
"""

try:
    from PySide6 import QtCore, QtGui, QtSvg, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.exporters                       # noqa: F401  (registers exporters)
    HAVE_QT = True
    QT_IMPORT_ERROR = None
except ImportError as _qt_import_error:               # pragma: no cover
    HAVE_QT = False
    QT_IMPORT_ERROR = _qt_import_error
    QtCore = QtGui = QtSvg = QtWidgets = pg = None

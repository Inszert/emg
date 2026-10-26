"""
Entry point: console setup, --summary and main().
"""

import os
import signal
import sys
import time

from emg_app import config as cfg
from emg_app.gestures import restore_gesture_setup
from emg_app.dataset import dataset_summary
from emg_app.gui.qt import HAVE_QT, QT_IMPORT_ERROR, QtCore, QtWidgets
from emg_app.gui.theme import apply_theme
from emg_app.gui.icon import apply_app_icon
from emg_app.gui.main_window import MainWindow


# ==============================================================================
# SECTION 8 -- ENTRY POINT
# ==============================================================================


def force_utf8_console() -> None:
    """
    Make console output safe for non-ASCII paths.

    This project lives under "OneDrive\\Pocitac\\EMG" -- with a c-caron. The
    Windows console defaults to cp1252, so printing that path raises
    UnicodeEncodeError and takes the whole program down with it (observed on
    `--summary`, and it would also hit any warning that quotes a file path).
    errors="replace" makes a stray character cosmetic instead of fatal.

    Only called from main(), so importing this module never reconfigures
    somebody else's streams.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def print_cli_summary() -> None:
    """`python emg_dataset_tool.py --summary` -- dataset stats without a GUI."""
    summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
    print(f"\nDataset root: {summary['root']}")
    print(f"Raw recordings: {summary['total_raw']}   "
          f"Trimmed: {summary['total_trimmed']}\n")
    if not summary["persons"]:
        print("No recordings yet.\n")
        return
    for person in sorted(summary["persons"]):
        print(f"  {person}")
        for session in sorted(summary["persons"][person]):
            counts = summary["persons"][person][session]
            total = sum(counts.values())
            print(f"    {session}  ({total})")
            for label in sorted(counts):
                print(f"      {label:<14} {counts[label]}")
    print()


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    force_utf8_console()

    if "--summary" in argv:
        print_cli_summary()
        return 0

    if not HAVE_QT:
        print("PySide6 and pyqtgraph are required for the GUI.")
        print(f"Import error: {QT_IMPORT_ERROR}")
        print("\nInstall them with:\n    python -m pip install -r requirements.txt")
        return 1

    t_start = time.perf_counter()

    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.AA_DontCreateNativeWidgetSiblings, True
    )
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(cfg.APP_NAME)
    app.setApplicationVersion(cfg.APP_VERSION)
    apply_theme(app)
    apply_app_icon(app)              # every window and dialog inherits it

    #  The gesture set edited in GESTURES... (and any gestures added there),
    #  saved with the data. Restored here and not in MainWindow, so a test
    #  that builds a window with its own GESTURES is never overridden.
    restored = restore_gesture_setup()
    if restored:
        print(f"[startup] gestures: {', '.join(restored)}", flush=True)
    if cfg.CLASSIFIER_OFF:
        #  Said at startup because a gesture the model "does not answer with"
        #  would otherwise look like a model that forgot it.
        print("[startup] switched OFF in the classifiers (settings icon to "
              f"change; nothing retrained): {', '.join(cfg.CLASSIFIER_OFF)}",
              flush=True)

    print(f"[startup] building the interface ...", flush=True)
    window = MainWindow()
    if cfg.WINDOW_MAXIMIZE_IF_SMALL and window.fitted_to_screen:
        # The preferred size did not fit, so the screen is the constraint:
        # maximised is then strictly better than a window the operator would
        # immediately maximise by hand.
        window.showMaximized()
    else:
        window.show()
    print(
        f"[startup] window up after {time.perf_counter() - t_start:.1f} s "
        f"(press Ctrl+C in this console, or close the window, to quit)",
        flush=True,
    )

    # `--train` / `--train-quick`: start a retrain as soon as the window is
    # up, so the sweep can be launched and watched without clicking. It takes
    # exactly the path the RETRAIN button takes -- same subprocess, same
    # progress panel, same completion dialog -- with the confirmation dialog
    # answered for it, because the flag IS the confirmation.
    if "--train" in argv or "--train-quick" in argv:
        quick = "--train-quick" in argv
        QtCore.QTimer.singleShot(
            1200, lambda: window.retrain_model(
                confirm=False, quick=quick))

    # Ctrl+C in the console must end the app cleanly. Qt's event loop runs in
    # C++, so Python never gets to run its SIGINT handler while the loop is
    # blocked: the KeyboardInterrupt instead surfaces at whatever arbitrary
    # point the interpreter next regains control, which in practice is inside
    # a pyqtgraph paintEvent -- and pyqtgraph catches it, prints a traceback,
    # repaints, gets interrupted again, and floods the console. The no-op
    # timer gives the interpreter a slice every 200 ms so the handler below
    # actually runs and shuts the window down the same way the X button does.
    _sigint_timer = QtCore.QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    def _on_sigint(_signum, _frame):
        print("", flush=True)
        print("[startup] Ctrl+C -- shutting down ...", flush=True)
        window.close()
        app.quit()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):                        # pragma: no cover
        pass                                             # not on the main thread

    return app.exec()

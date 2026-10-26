"""
================================================================================
 MYO EMG DATASET TOOL
================================================================================

 Professional EMG recording / visualisation / labeling / dataset-generation tool
 for the Thalmic Myo armband (BLED112 dongle, COM port, myo_raw.py wrapper).

 Run:        python emg_dataset_tool.py
 Install:    python -m pip install -r requirements.txt

 The application lives in the `emg_app` package (see emg_app/README.md for
 the module map). This file is the ENTRY POINT and the stable import surface:

     import emg_dataset_tool as T
     T.load_dataset(...); T.MainWindow; T.CALIBRATION_MODE

 keeps working exactly as it did when the tool was one file. Every name of
 every module is re-exported here, and the USER CONFIGURATION names
 (emg_app/config.py) are forwarded LIVE in both directions:

     T.CALIBRATION_MODE = "off"     # sets emg_app.config.CALIBRATION_MODE,
                                    # which is what every module reads

 See CLAUDE.md and docs/ for architecture, dataset format and ML pipeline.
================================================================================
"""

import sys

# Printed before numpy/Qt/pyqtgraph are pulled in. On a cold file cache those
# imports take a few seconds with no window and no output, which reads as a
# hang; this is the first thing on screen. Only when run as a script, so an
# `import emg_dataset_tool` from a trainer or a notebook stays silent.
if __name__ == "__main__":
    print("Myo EMG Dataset Tool -- loading libraries ...", flush=True)

import importlib
import types

from emg_app import config as _cfg

# In dependency order. The GUI modules are only imported when Qt is present,
# so the data layer stays usable from a training script without PySide6.
_DATA_MODULES = (
    "gui.tokens", "utils", "signal_processing", "calibration", "features",
    "quality", "markers", "recording", "model_spec", "compute", "dataset",
    "acquisition", "neural", "jobs", "sequence", "model_test", "class_mask",
    "decision_panel", "prediction", "zoo", "adaptation", "settings_store",
)
_GUI_MODULES = (
    "gui.qt", "gui.widgets", "gui.theme", "gui.plot", "gui.decision_trail",
    "gui.workers", "gui.dialogs.quality", "gui.dialogs.model_stats",
    "gui.dialogs.neural_stats", "gui.dialogs.gesture_setup",
    "gui.dialogs.settings",
    "gui.dialogs.calibration", "gui.dialogs.neural_setup",
    "gui.dialogs.model_spec", "gui.dialogs.trim", "gui.dialogs.model_test",
    "gui.main_window", "app",
)

_CONFIG_NAMES = frozenset(k for k in vars(_cfg) if not k.startswith("__"))


def _export(module_name: str) -> types.ModuleType:
    module = importlib.import_module(f"emg_app.{module_name}")
    for key, value in vars(module).items():
        if key.startswith("__") or key == "cfg" or key in _CONFIG_NAMES:
            continue
        globals().setdefault(key, value)
    return module


for _name in _DATA_MODULES:
    _export(_name)
if _export("gui.qt").HAVE_QT:
    for _name in _GUI_MODULES[1:]:
        _export(_name)
else:                                                  # pragma: no cover
    _export("app")


def __getattr__(name):
    """USER CONFIGURATION names are read live from emg_app.config."""
    if name in _CONFIG_NAMES:
        return getattr(_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _CONFIG_NAMES)


class _ConfigForwardingModule(types.ModuleType):
    """Assigning a USER CONFIGURATION name here assigns it in emg_app.config."""

    def __setattr__(self, name, value):
        if name in _CONFIG_NAMES:
            setattr(_cfg, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ConfigForwardingModule


if __name__ == "__main__":
    sys.exit(main())  # noqa: F821  (re-exported from emg_app.app)

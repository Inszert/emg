"""
Persistent application settings (Qt-free).

Everything in `SETTINGS` lives in `cfg` while the app runs and is saved to
`<DATA_FOLDER>/app_settings.json`. It sits next to `gestures.json` for the same
reason: it describes how this dataset's models are used, so it travels with the
data. An unreadable or partly invalid file is ignored key by key -- a bad value
never empties the settings, and an unknown key never crashes startup.

Restored by `app.main()` (not by `MainWindow`, so a test's own values are not
overridden), exactly like the gesture set.

Defaults (in `emg_app/config.py`) reproduce the behaviour from before these
settings existed: no model selection, no augmentation, no adaptation. The two
prompts default to ON because they were asked for.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from emg_app import config as cfg

#  name -> (type, validator-or-None). Types: bool, list (of str).
SETTINGS = {
    "MODEL_SELECTION_ENABLED": bool,
    "SELECTED_ML": list,
    "SELECTED_NN": list,
    "TRAIN_AUGMENT_NN": list,
    "TRAIN_AUGMENT_ML": list,
    "TRAIN_ADAPT_NN": bool,
    "TRAIN_ADAPT_ML": bool,
    "ADAPT_LIVE": bool,
    "ASK_CALIBRATION_BEFORE_PREDICT": bool,
    "ASK_RETRAIN_ON_NEW_DATA": bool,
    "APP_ICON_ENABLED": bool,
}


def settings_path(root: str = None) -> str:
    root = root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
    return os.path.join(root, cfg.APP_SETTINGS_FILE)


def _clean(name: str, value):
    """`value` coerced to the setting's type, or raise ValueError."""
    kind = SETTINGS[name]
    if kind is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be true or false")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} holds a non-text entry: {item!r}")
        if item not in out:
            out.append(item)
    if name.startswith("TRAIN_AUGMENT"):
        #  A typo must not silently train without augmentation.
        known = ("gain", "noise", "timewarp", "rotate", "chandrop")
        bad = [x for x in out if x not in known]
        if bad:
            raise ValueError(f"unknown augmentation {bad}; known: "
                             f"{', '.join(known)}")
    return out


def current() -> Dict[str, object]:
    """The live values, as they would be saved."""
    return {name: (list(getattr(cfg, name)) if SETTINGS[name] is list
                   else bool(getattr(cfg, name)))
            for name in SETTINGS}


#  What the application ships with: the values config.py holds when this module
#  is imported, i.e. before any saved file has been restored.
DEFAULTS = current()


def load_settings(root: str = None) -> Optional[dict]:
    """The saved values that are valid; `None` when there is no readable file."""
    path = settings_path(root)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    out = {}
    for name in SETTINGS:
        if name in raw:
            try:
                out[name] = _clean(name, raw[name])
            except ValueError:
                continue
    return out


def save_settings(values: dict = None, root: str = None) -> str:
    """
    Write `values` (default: the live ones) atomically. Unknown keys are
    dropped and bad values raise, so a typo cannot reach the file.
    """
    values = current() if values is None else values
    clean = {name: _clean(name, values[name])
             for name in SETTINGS if name in values}
    path = settings_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    os.replace(tmp, path)
    return path


def apply_settings(values: Optional[dict]) -> int:
    """Put the values into `cfg`. Returns how many were applied."""
    n = 0
    for name, value in (values or {}).items():
        if name in SETTINGS:
            setattr(cfg, name, value)
            n += 1
    return n


def restore_settings(root: str = None) -> Optional[dict]:
    """Load the saved file and apply it; `None` if there was nothing."""
    values = load_settings(root)
    if values is None:
        return None
    apply_settings(values)
    return values

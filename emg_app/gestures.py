"""
The gesture set, as the operator edits it: saved, restored, and compared
against what the models know.

GESTURES... adds a gesture in the GUI. This module makes that last: the
in-use list and every added gesture's cue go to `<DATA_FOLDER>/gestures.json`
and `app.main()` restores them on start. The file lives with the DATA, not
with the models, because it describes what is being recorded.

Adding a gesture NEVER retrains anything. A model knows exactly the labels it
was trained on; `model_gesture_gaps()` says which recorded gestures a model
does not know yet, and the operator decides when to retrain.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from emg_app import config as cfg
from emg_app.utils import safe_slug

GESTURE_SETUP_VERSION = 1


def gesture_name(raw: str) -> str:
    """A typed name -> the label used everywhere (and as a folder name)."""
    return safe_slug(str(raw).strip(), "GESTURE").upper()


def gesture_setup_path(root: str = None) -> str:
    root = root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
    return os.path.join(root, cfg.GESTURE_SETUP_FILE)


def load_gesture_setup(root: str = None) -> Optional[dict]:
    """
    The saved setup, cleaned: {"in_use": [...], "custom": {name: cue}}.

    None when there is no file or it is unreadable -- the caller then keeps
    the built-in GESTURES, so a damaged file can never empty the label set.
    """
    try:
        with open(gesture_setup_path(root), encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    custom = {}
    for name, cue in (raw.get("custom") or {}).items():
        name = gesture_name(name)
        if name:
            custom[name] = str(cue or "").strip()
    in_use = []
    for name in raw.get("in_use") or []:
        name = gesture_name(name)
        if name and name not in in_use:
            in_use.append(name)
    if len(in_use) < 2:
        return None
    return {"in_use": in_use, "custom": custom}


def save_gesture_setup(in_use: List[str], custom: Dict[str, str],
                       root: str = None) -> str:
    """Write the setup (atomically: a crash cannot leave half a file)."""
    path = gesture_setup_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {"version": GESTURE_SETUP_VERSION,
            "in_use": [gesture_name(g) for g in in_use],
            "custom": {gesture_name(k): str(v or "").strip()
                       for k, v in (custom or {}).items()}}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2)
    os.replace(tmp, path)
    return path


def apply_gesture_setup(setup: Optional[dict]) -> bool:
    """Make a loaded setup the live one. False (and no change) for None."""
    if not setup:
        return False
    cfg.CUSTOM_GESTURES = dict(setup.get("custom") or {})
    cfg.GESTURES = list(setup["in_use"])
    return True


def restore_gesture_setup(root: str = None) -> Optional[List[str]]:
    """Load and apply in one step; returns the gestures now in use, or None."""
    setup = load_gesture_setup(root)
    return list(cfg.GESTURES) if apply_gesture_setup(setup) else None


def model_gesture_gaps(model_labels, recordings_by_label: Dict[str, int],
                       people_by_label: Dict[str, int] = None) -> List[dict]:
    """
    Gestures that HAVE recordings but that a model does not know.

    One entry per gesture: {"label", "recordings", "people"}. Empty when the
    model covers everything recorded -- including when a new gesture is in
    the plan but not recorded yet, because there is nothing to retrain on.
    """
    known = set(model_labels or [])
    out = []
    for label, n in sorted(recordings_by_label.items()):
        if n and label not in known:
            out.append({"label": label, "recordings": int(n),
                        "people": (None if people_by_label is None else
                                   int(people_by_label.get(label, 0)))})
    return out


def recorded_gestures(summary: dict):
    """(recordings per label, people per label) from dataset_summary()."""
    recs = defaultdict(int)
    people = defaultdict(set)
    for person, sessions in (summary.get("persons") or {}).items():
        for counts in sessions.values():
            for label, n in counts.items():
                recs[label] += n
                if n:
                    people[label].add(person)
    return dict(recs), {k: len(v) for k, v in people.items()}


def describe_gaps(gaps: List[dict], model: str) -> str:
    """One line for the panel, or "" when there is nothing to say."""
    if not gaps:
        return ""
    bits = []
    for g in gaps:
        bit = f"{g['label']} ({g['recordings']} rec"
        if g["people"] is not None and g["people"] < 2:
            bit += f", {g['people']} person"
        bits.append(bit + ")")
    return (f"NOT IN THE {model} YET: " + ", ".join(bits)
            + " -- RETRAIN when you choose")

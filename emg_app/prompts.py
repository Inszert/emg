"""
The decisions behind the two questions the application asks -- Qt-free, so they
can be tested without a window.

1. BEFORE PREDICT / START NN: "calibrate your hand first?"
   `calibration_offer()` decides whether a saved calibration may be offered for
   reuse. It is offered only when ALL of these hold:

     * the person has a stored calibration;
     * it was made BEFORE this run of the application started -- which is what
       "the application was restarted" means: a calibration made in this run is
       already in use and there is nothing to reuse;
     * it is not older than `LAST_CALIBRATION_MAX_AGE_H`;

   and the offer always carries the one thing the application cannot know: that
   the band must have stayed on the arm since. The only objective check it can
   make is on the live signal -- a hand at rest should read about what it read
   at calibration time -- and that check is reported, never used to refuse:
   a tense hand fails it for a harmless reason.

2. NEW GESTURES / NEW DATA: "retrain all models or only the selected?"
   `retrain_reasons()` says why a model is out of date; `retrain_targets()`
   says what each answer would retrain.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

#  A calibration older than this is not offered for reuse: a band that stayed
#  on for a working day has drifted (sweat, pressure), and calibrating costs
#  seconds.
LAST_CALIBRATION_MAX_AGE_H = 8.0
#  The live rest check: per channel the median of the last second may differ
#  from the calibration's rest level by at most this much (counts) -- or by
#  this share of it, whichever is larger.
REST_TOLERANCE_COUNTS = 30.0
REST_TOLERANCE_SHARE = 1.0


def _parse_iso(text: str) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def rest_check(live_block, record: dict) -> Optional[dict]:
    """
    Compare the last second of live signal with the calibration's REST levels.
    `None` when there is no usable live block. Otherwise
    `{"ok": bool, "worst_channel": int, "live": [...], "rest": [...]}`.
    """
    if live_block is None:
        return None
    block = np.asarray(live_block, dtype=float)
    rest = np.asarray((record or {}).get("rest") or [], dtype=float)
    if block.ndim != 2 or block.shape[0] < 5 or rest.size != block.shape[1]:
        return None
    live = np.median(block, axis=0)
    allowed = np.maximum(REST_TOLERANCE_COUNTS, REST_TOLERANCE_SHARE * rest)
    excess = np.abs(live - rest) - allowed
    return {"ok": bool(np.all(excess <= 0)),
            "worst_channel": int(np.argmax(excess)),
            "live": [float(x) for x in live],
            "rest": [float(x) for x in rest]}


def calibration_offer(record: Optional[dict], app_started_ts: float,
                      now: float = None, live_block=None,
                      max_age_h: float = LAST_CALIBRATION_MAX_AGE_H) -> dict:
    """
    May the stored calibration be offered for reuse?

    Returns `{"available": bool, "reason": str, "age_h": float|None,
    "created": str, "rest_check": dict|None}`. `reason` says why it is NOT
    available, or (when it is) what the person has to be sure of.
    """
    now = time.time() if now is None else now
    if not record:
        return {"available": False, "reason": "this person has no stored "
                "calibration", "age_h": None, "created": "",
                "rest_check": None}
    created = record.get("created_unix")
    created = float(created) if isinstance(created, (int, float))         else _parse_iso(record.get("created_iso"))
    if created is None:
        return {"available": False,
                "reason": "the stored calibration has no creation time",
                "age_h": None, "created": "", "rest_check": None}
    age_h = max(0.0, (now - created) / 3600.0)
    base = {"age_h": age_h, "created": str(record.get("created_iso") or "")[:16]
            .replace("T", " "), "rest_check": rest_check(live_block, record)}
    if created >= app_started_ts:
        return {**base, "available": False,
                "reason": "it was made during this run of the application, "
                          "so it is already the one in use"}
    if age_h > max_age_h:
        return {**base, "available": False,
                "reason": f"it is {age_h:.1f} h old (more than "
                          f"{max_age_h:g} h): the band has very likely "
                          f"drifted"}
    return {**base, "available": True,
            "reason": "reuse it only if the band has stayed on your arm "
                      "since then; if you took it off or moved it, "
                      "calibrate again"}


# ==============================================================================
# THE RETRAIN QUESTION
# ==============================================================================


def retrain_reasons(model_states: Dict[str, dict], gaps: List[dict] = None
                    ) -> List[str]:
    """
    Why models are out of date, as sentences. `model_states` maps a model's
    name to what `model_data_status()` returned for it; `gaps` is
    `model_gesture_gaps()` -- gestures recorded but unknown to a model.
    """
    out = []
    for name, state in (model_states or {}).items():
        if not state:
            continue
        if state.get("state") == "stale":
            out.append(f"{name}: {state.get('detail') or 'the dataset has changed since it was trained'}")
    for gap in gaps or []:
        label = gap.get("label")
        if label:
            out.append(f"{gap.get('model', 'the model')} does not know "
                       f"{label} yet ({gap.get('recordings', '?')} "
                       f"recording(s) of it exist)")
    return out


def retrain_targets(choice: str, selection_enabled: bool,
                    selected_ml: List[str], selected_nn: List[str],
                    all_ids: List[str]) -> dict:
    """
    What each answer to the retrain question trains.

    "all"       the deployed ML model and net, and every zoo model
    "selected"  with model selection ON: the selected zoo models (and nothing
                else); with it OFF: the deployed ML model and net, because
                those ARE the selected ones
    anything else trains nothing
    """
    if choice == "all":
        return {"deployed": True, "zoo_ids": list(all_ids)}
    if choice == "selected":
        if selection_enabled and (selected_ml or selected_nn):
            return {"deployed": False,
                    "zoo_ids": list(selected_ml) + list(selected_nn)}
        return {"deployed": True, "zoo_ids": []}
    return {"deployed": False, "zoo_ids": []}

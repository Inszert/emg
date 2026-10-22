"""
Per-wearer calibration (CLAUDE.md 8k): the maths, the quality gates, storage
under datasets/<person>/, and the tables the trainers apply.
"""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from emg_app import config as cfg
from emg_app.utils import ensure_dir, iso_time, safe_slug


# ------------------------------------------------------------------------------
# Per-wearer calibration
# ------------------------------------------------------------------------------
# A calibration is two levels per channel -- REST and REFERENCE (strong
# contraction) -- measured once for one wearer. Everything below turns those
# into the affine map that is applied to a window on its way to the features,
# and nothing here reads the label of anything: a calibration is measured from
# unlabelled signal, which is what makes it something a new wearer can do in
# six seconds before using the model.
#
# The record is a plain dict so it can be JSON on disk and compared in tests
# without numpy. `resolve_calibration()` turns it into the two arrays the
# per-window transform needs, ONCE, so the cost in the live path is one
# subtract and one divide per window.

CALIBRATION_MODES = ("off", "gain", "per_channel", "affine")
CALIBRATION_VERSION = 1


def calibration_levels(record: dict):
    """
    Stored record -> (rest[8], reference[8]) as float arrays.

    Returns None when the record does not carry a usable pair, so every
    caller fails the same way instead of each inventing a fallback.
    """
    if not isinstance(record, dict):
        return None
    try:
        rest = np.asarray(record["rest"], dtype=np.float64).ravel()
        ref = np.asarray(record["reference"], dtype=np.float64).ravel()
    except (KeyError, TypeError, ValueError):
        return None
    #  The record's own width, matched against itself rather than against
    #  the active profile. A calibration measured on an 8-channel band stays
    #  readable after EMG_DEVICE moves on -- it simply will not APPLY to a
    #  wider window, which `apply_calibration` catches by comparing shapes.
    if rest.size == 0 or rest.size != ref.size:
        return None
    if not (np.all(np.isfinite(rest)) and np.all(np.isfinite(ref))):
        return None
    return rest, ref


def validate_calibration(record: dict):
    """
    Is this record safe to divide a dataset by?  -> (ok, problems)

    Never raises, and never "fixes" anything. A model spec gets coerced
    (validate_model_spec) because a clamped hyperparameter still trains
    something sensible; a gain does not work that way. A calibration whose
    reference level is noise would scale a wearer's whole signal by a factor
    of twenty, so the only honest options are accept or refuse.
    """
    problems: List[str] = []
    levels = calibration_levels(record)
    if levels is None:
        return False, ["no usable rest/reference pair -- the two level "
                       "vectors must be the same length and all finite"]
    rest, ref = levels

    if np.any(rest < 0) or np.any(ref < 0):
        problems.append("negative levels (the filtered stream is rectified, "
                        "so every level should be >= 0)")

    ref_scalar = float(np.mean(ref))
    rest_scalar = float(np.mean(rest))
    if ref_scalar < cfg.CALIBRATION_MIN_REFERENCE:
        problems.append(
            f"reference level {ref_scalar:.0f} is below "
            f"CALIBRATION_MIN_REFERENCE ({cfg.CALIBRATION_MIN_REFERENCE:.0f}) -- "
            f"the effort capture looks like rest, not like a contraction")
    contrast = ref_scalar / max(1e-9, rest_scalar)
    if contrast < cfg.CALIBRATION_MIN_CONTRAST:
        problems.append(
            f"effort is only {contrast:.2f}x rest, below "
            f"CALIBRATION_MIN_CONTRAST ({cfg.CALIBRATION_MIN_CONTRAST:.2f}) -- "
            f"the two captures did not measure two different things")
    if np.any(ref <= rest):
        names = cfg.channel_names_for(rest.size)
        quiet = [names[i] for i in range(rest.size) if ref[i] <= rest[i]]
        problems.append("reference is not above rest on " + ", ".join(quiet))

    for key, floor in (("n_rest_samples", cfg.CALIBRATION_MIN_SAMPLES),
                       ("n_effort_samples", cfg.CALIBRATION_MIN_SAMPLES)):
        n = record.get(key)
        if n is not None and int(n) < floor:
            problems.append(f"{key} = {int(n)}, fewer than "
                            f"CALIBRATION_MIN_SAMPLES ({floor})")

    return (not problems), problems


def resolve_calibration(record: dict, mode: str = None,
                        target: float = None) -> Optional[dict]:
    """
    Stored record + mode -> the arrays `apply_calibration()` needs.

    Returns None when nothing should be applied: mode "off", no record, or a
    record that does not validate. None is the "behave exactly as before"
    answer, and every caller treats it that way, so an unusable calibration
    degrades to the uncalibrated pipeline instead of to a wrong one.
    """
    mode = (cfg.CALIBRATION_MODE if mode is None else mode) or "off"
    if mode == "off" or mode not in CALIBRATION_MODES:
        return None
    ok, _problems = validate_calibration(record)
    if not ok:
        return None
    rest, ref = calibration_levels(record)
    target = float(cfg.CALIBRATION_TARGET_LEVEL if target is None else target)

    #  Sized from the RECORD, so the resolved map is as wide as the
    #  calibration that produced it rather than as wide as the config.
    width = int(rest.size)
    if mode == "gain":
        #  ONE divisor for every channel, so every ratio between channels --
        #  the spatial profile the gestures actually differ in -- comes out of
        #  this unchanged.
        offset = np.zeros(width, dtype=np.float64)
        scale = np.full(width, float(np.mean(ref)), dtype=np.float64)
    elif mode == "per_channel":
        offset = np.zeros(width, dtype=np.float64)
        scale = np.asarray(ref, dtype=np.float64).copy()
    else:                                                      # "affine"
        offset = np.asarray(rest, dtype=np.float64).copy()
        scale = np.asarray(ref, dtype=np.float64) - offset

    #  One device count is the quantisation step of the stream, so it is the
    #  natural floor: it caps the gain this can ever apply instead of letting
    #  a hand-edited file ask for a factor of a million.
    scale = np.maximum(scale, 1.0)
    return {
        "mode": mode,
        "offset": offset,
        "scale": scale,
        "target": target,
        "person_id": record.get("person_id", ""),
        "created_iso": record.get("created_iso", ""),
    }


def apply_calibration(block: np.ndarray, cal: dict) -> np.ndarray:
    """
    One window (n, 8) -> the same window on the calibrated scale.

        y = target * (x - offset) / scale

    `cal` is the output of `resolve_calibration()`; None means "return the
    input untouched", which is what makes calibration optional everywhere
    without a flag at every call site.

    The result is floored at 0. Only the "affine" mode can go negative, and
    only below the wearer's own rest level, where the rectified envelope is
    noise -- but `share` (the spatial profile feature) divides by the channel
    total, and a negative channel would make that meaningless rather than
    merely noisy.
    """
    if not cal:
        return block
    x = np.asarray(block, dtype=np.float64)
    #  The window must match the CALIBRATION's width, not the config's. A
    #  mismatch returns the block untouched, which is the same "behave as the
    #  uncalibrated pipeline" answer `resolve_calibration` gives for an
    #  unusable record -- degrading to no calibration rather than to a wrong
    #  one. Broadcasting an 8-wide map onto a 16-wide window would otherwise
    #  either raise here or, worse, succeed on a multiple.
    if x.ndim != 2 or x.shape[1] != np.size(cal["offset"]):
        return block
    out = (x - cal["offset"]) * (float(cal["target"]) / cal["scale"])
    return np.maximum(out, 0.0)


def calibration_record(rest_levels, effort_levels, person_id: str = "",
                       session_id: str = "", fs: float = None,
                       n_rest: int = 0, n_effort: int = 0,
                       source: str = "live", notes: str = "",
                       extra: dict = None) -> dict:
    """Build the record that goes on disk. JSON-safe, no numpy left in it."""
    rest = np.asarray(rest_levels, dtype=np.float64).ravel()
    ref = np.asarray(effort_levels, dtype=np.float64).ravel()
    now = time.time()
    rest_scalar = float(np.mean(rest)) if rest.size else 0.0
    ref_scalar = float(np.mean(ref)) if ref.size else 0.0
    record = {
        "version": CALIBRATION_VERSION,
        "person_id": person_id,
        "session_id": session_id,
        "created_unix": now,
        "created_iso": iso_time(now),
        "emg_mode": cfg.EMG_MODE,
        "fs": float(fs) if fs else None,
        "channel_names": list(cfg.CHANNEL_NAMES),
        "rest": [float(v) for v in rest],
        "reference": [float(v) for v in ref],
        "rest_level": rest_scalar,
        "reference_level": ref_scalar,
        "contrast": ref_scalar / max(1e-9, rest_scalar),
        "target_level": float(cfg.CALIBRATION_TARGET_LEVEL),
        "n_rest_samples": int(n_rest),
        "n_effort_samples": int(n_effort),
        "rest_percentile": float(cfg.CALIBRATION_REST_PERCENTILE),
        "effort_percentile": float(cfg.CALIBRATION_EFFORT_PERCENTILE),
        "source": source,
        "notes": notes,
        "device": cfg.DEVICE_NAME,
        "arm_worn_on": cfg.ARM_WORN_ON,
        "handedness": cfg.HANDEDNESS,
    }
    record.update(extra or {})
    return record


def calibration_from_captures(rest_block: np.ndarray, effort_block: np.ndarray,
                              person_id: str = "", session_id: str = "",
                              fs: float = None, notes: str = "",
                              source: str = "live") -> dict:
    """
    Two short captures -> a calibration record.

    `rest_block` is the wearer relaxed, `effort_block` the wearer holding a
    strong contraction. Neither is labelled and neither is kept: only the two
    level vectors go on disk.
    """
    rest = np.asarray(rest_block, dtype=np.float64)
    effort = np.asarray(effort_block, dtype=np.float64)
    rest_levels = (np.percentile(rest, cfg.CALIBRATION_REST_PERCENTILE, axis=0)
                   if rest.size else np.zeros(cfg.N_CHANNELS))
    effort_levels = (np.percentile(effort, cfg.CALIBRATION_EFFORT_PERCENTILE, axis=0)
                     if effort.size else np.zeros(cfg.N_CHANNELS))
    return calibration_record(
        rest_levels, effort_levels, person_id=person_id,
        session_id=session_id, fs=fs,
        n_rest=int(rest.shape[0]) if rest.ndim == 2 else 0,
        n_effort=int(effort.shape[0]) if effort.ndim == 2 else 0,
        source=source, notes=notes)


def calibration_from_recordings(recordings, person_id: str = "",
                                rest_percentile: float = None,
                                effort_percentile: float = None) -> Optional[dict]:
    """
    A wearer's existing recordings -> a calibration record, WITHOUT labels.

    This is how a dataset recorded before calibration existed gets calibrated
    (CLAUDE.md section 8k). The levels come from percentiles of all of that
    wearer's samples pooled together: the low percentile stands in for the
    rest capture and the high one for the effort capture, which is the same
    pair of numbers a live calibration measures, taken from signal that
    happens to already be on disk.

    **It deliberately does not look at `rec.label`.** Computing the rest level
    from the recordings labelled REST would be using the held-out wearer's
    LABELS to build the transform their own test fold is scored through, which
    is leakage. Pooling everything uses only the wearer's unlabelled signal,
    which is exactly the assumption a per-user calibration makes anyway: the
    new wearer gives you some of their signal before you classify it.

    It is still not a zero-shot protocol, and the trainer says so in words
    whenever a calibrated model is scored subject-grouped.
    """
    recs = [r for r in (recordings or [])
            if getattr(r, "emg", None) is not None and r.emg.size]
    if person_id:
        recs = [r for r in recs if r.person_id == person_id]
    if not recs:
        return None
    lo = (cfg.CALIBRATION_DERIVE_REST_PERCENTILE if rest_percentile is None
          else float(rest_percentile))
    hi = (cfg.CALIBRATION_DERIVE_EFFORT_PERCENTILE if effort_percentile is None
          else float(effort_percentile))

    pooled = np.vstack([np.asarray(r.emg, dtype=np.float64) for r in recs])
    rates = [r.fs for r in recs if getattr(r, "fs", 0)]
    record = calibration_record(
        np.percentile(pooled, lo, axis=0),
        np.percentile(pooled, hi, axis=0),
        person_id=person_id or recs[0].person_id,
        session_id=sorted({r.session_id for r in recs})[0],
        fs=float(np.median(rates)) if rates else None,
        n_rest=int(pooled.shape[0]), n_effort=int(pooled.shape[0]),
        source="derived_from_recordings",
        notes=(f"percentiles {lo:g}/{hi:g} over {len(recs)} recordings "
               f"({pooled.shape[0]} samples), labels not used"),
        extra={"derived_from": {"n_recordings": len(recs),
                                "n_samples": int(pooled.shape[0]),
                                "rest_percentile": lo,
                                "effort_percentile": hi,
                                "sessions": sorted({r.session_id for r in recs}),
                                "uses_labels": False}})
    return record


def calibration_capture_quality(block: np.ndarray, rest_levels=None) -> dict:
    """
    Judge ONE capture while the wearer is still standing there.

    This is what lets the dialog say "that one did not register, do it again"
    instead of storing a shrug and discovering it later. Returns the numbers,
    never a verdict about the whole calibration -- that is
    `validate_calibration()`'s job on the finished record.

        ok            enough samples, and loud enough against rest
        n_samples     how much signal arrived
        contrast      peak channel level / mean rest level
        peak_channel  which pod it landed on, 1-based
        levels        the per-channel levels at EFFORT percentile
        reason        why not, when ok is False
    """
    x = np.asarray(block, dtype=np.float64) if block is not None else None
    if x is None or x.ndim != 2 or x.shape[0] == 0:
        return {"ok": False, "n_samples": 0, "contrast": 0.0,
                "peak_channel": 0, "levels": None,
                "reason": "nothing was captured"}
    n = int(x.shape[0])
    levels = np.percentile(x, cfg.CALIBRATION_EFFORT_PERCENTILE, axis=0)
    rest = (np.asarray(rest_levels, dtype=np.float64).ravel()
            if rest_levels is not None else None)
    #  Against the wearer's own rest when it is known, and against the
    #  hardware's noise floor when it is not, so the first capture of a
    #  session can still be judged.
    floor = (float(np.mean(rest)) if rest is not None and rest.size
             else float(cfg.CALIBRATION_MIN_REFERENCE) / 2.0)
    peak = int(np.argmax(levels))
    contrast = float(levels[peak]) / max(1e-9, floor)
    info = {"ok": True, "n_samples": n, "contrast": contrast,
            "peak_channel": peak + 1, "levels": levels, "reason": ""}
    if n < cfg.CALIBRATION_MIN_SAMPLES:
        info["ok"] = False
        info["reason"] = (f"only {n} samples, need "
                          f"{cfg.CALIBRATION_MIN_SAMPLES}")
    elif contrast < cfg.CALIBRATION_MIN_GESTURE_CONTRAST:
        info["ok"] = False
        info["reason"] = (f"peak only {contrast:.1f}x rest, need "
                          f"{cfg.CALIBRATION_MIN_GESTURE_CONTRAST:.1f}x -- "
                          f"hold it harder")
    elif float(levels[peak]) < cfg.CALIBRATION_MIN_GESTURE_LEVEL:
        #  The ratio can be flattered by noise on a channel sitting at rest;
        #  the absolute level cannot.
        info["ok"] = False
        info["reason"] = (f"peak only {levels[peak]:.0f} counts, need "
                          f"{cfg.CALIBRATION_MIN_GESTURE_LEVEL:.0f} -- "
                          f"hold it harder")
    return info


def calibration_rest_quality(block: np.ndarray) -> dict:
    """
    Judge the REST capture the way `calibration_capture_quality` judges a
    gesture: as it lands, so a rest that was not relaxed can be redone.

        ok            enough samples, and quiet on every channel
        n_samples     how much signal arrived
        levels        the per-channel levels at REST percentile
        loud_channel  the loudest channel, 1-based
        level         that channel's level
        reason        why not, when ok is False
    """
    x = np.asarray(block, dtype=np.float64) if block is not None else None
    if x is None or x.ndim != 2 or x.shape[0] == 0:
        return {"ok": False, "n_samples": 0, "levels": None,
                "loud_channel": 0, "level": 0.0,
                "reason": "nothing was captured"}
    n = int(x.shape[0])
    levels = np.percentile(x, cfg.CALIBRATION_REST_PERCENTILE, axis=0)
    loud = int(np.argmax(levels))
    info = {"ok": True, "n_samples": n, "levels": levels,
            "loud_channel": loud + 1, "level": float(levels[loud]),
            "reason": ""}
    if n < cfg.CALIBRATION_MIN_SAMPLES:
        info["ok"] = False
        info["reason"] = (f"only {n} samples, need "
                          f"{cfg.CALIBRATION_MIN_SAMPLES}")
    elif float(levels[loud]) > cfg.CALIBRATION_MAX_REST_LEVEL:
        #  The median, so one twitch does not fail it -- this is a hand that
        #  stayed tense for most of the capture.
        info["ok"] = False
        info["reason"] = (f"CH{loud + 1} at {levels[loud]:.0f} counts, "
                          f"relaxed is under "
                          f"{cfg.CALIBRATION_MAX_REST_LEVEL:.0f} -- let the "
                          f"hand go limp")
    return info


def calibration_from_gesture_captures(rest_block: np.ndarray,
                                      gesture_blocks: dict,
                                      person_id: str = "",
                                      session_id: str = "",
                                      fs: float = None, notes: str = "",
                                      source: str = "live_gesture_sweep") -> dict:
    """
    REST plus one capture per GESTURE -> a calibration record.

    **Why this beats one generic clench, measured on this dataset.** The
    reference is meant to be each channel's full scale, and one gesture only
    drives the muscles that gesture uses. `Dano`'s derived reference is
    [553, 657, 154, 219, 919, 992, 364, 404] -- CH3 and CH4 reach 154 and 219
    where CH6 reaches 992. Calibrating from a single fist would set CH3's
    scale from whatever a fist happens to do to CH3, which is nearly nothing,
    and then divide that channel by noise. Sweeping the gesture set gives
    every channel a chance to be the loud one.

        reference[c] = max over gestures of percentile(capture_g[:, c])

    The MAX rather than a pooled percentile, because "how loud does this
    channel ever get" should not be diluted by the four gestures that leave
    it quiet, and because it then does not matter how long each capture ran.

    REST is the one capture that is not part of the max: it sets the floor.
    """
    rest = np.asarray(rest_block, dtype=np.float64) if rest_block is not None \
        else np.zeros((0, cfg.N_CHANNELS))
    rest_levels = (np.percentile(rest, cfg.CALIBRATION_REST_PERCENTILE, axis=0)
                   if rest.size else np.zeros(cfg.N_CHANNELS))

    per_gesture = {}
    stacked = []
    n_effort = 0
    for label, block in (gesture_blocks or {}).items():
        quality = calibration_capture_quality(block, rest_levels)
        if quality["levels"] is None:
            continue
        per_gesture[label] = {
            "n_samples": quality["n_samples"],
            "contrast": round(quality["contrast"], 3),
            "peak_channel": quality["peak_channel"],
            "accepted": bool(quality["ok"]),
            "levels": [float(v) for v in quality["levels"]],
        }
        if not quality["ok"]:
            #  A capture that did not register must not pull a channel's
            #  scale DOWN, and it cannot push one up either -- it is simply
            #  not evidence. Recorded, so the dialog can show what happened.
            continue
        stacked.append(quality["levels"])
        n_effort += quality["n_samples"]

    reference = (np.max(np.vstack(stacked), axis=0) if stacked
                 else np.zeros(cfg.N_CHANNELS))
    accepted = sorted(k for k, v in per_gesture.items() if v["accepted"])
    return calibration_record(
        rest_levels, reference, person_id=person_id, session_id=session_id,
        fs=fs, n_rest=int(rest.shape[0]), n_effort=int(n_effort),
        source=source,
        notes=notes or (f"gesture sweep over {len(accepted)} gesture(s): "
                        f"{', '.join(accepted) or 'none accepted'}"),
        extra={"gesture_sweep": {
            "gestures": accepted,
            "rejected": sorted(k for k, v in per_gesture.items()
                               if not v["accepted"]),
            "per_gesture": per_gesture,
            "reference_rule": (f"per channel, max over accepted gestures of "
                               f"the p{cfg.CALIBRATION_EFFORT_PERCENTILE:g} level"),
        }})


def describe_calibration(record: dict, mode: str = None) -> str:
    """One line for a status label or a log. Says 'none' rather than guessing."""
    mode = (cfg.CALIBRATION_MODE if mode is None else mode) or "off"
    if not record:
        return f"no calibration (mode {mode})"
    ok, problems = validate_calibration(record)
    when = (record.get("created_iso") or "")[:16].replace("T", " ")
    head = (f"{record.get('person_id', '?')}  {when}  "
            f"rest {record.get('rest_level', 0):.0f} -> ref "
            f"{record.get('reference_level', 0):.0f} "
            f"({record.get('contrast', 0):.1f}x)")
    if not ok:
        return f"{head}  REFUSED: {problems[0]}"
    if mode == "off":
        return f"{head}  stored, NOT APPLIED (mode off)"
    return f"{head}  applied as {mode}"


# ------------------------------------------------------------------------------
# Reading / ML-ready loading
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Calibration storage -- one record per WEARER, in the wearer's own folder
# ------------------------------------------------------------------------------
#       <root>/<person_id>/calibration.json          the current one
#       <root>/<person_id>/calibration_log.jsonl     append-only history
#
# Per wearer and not next to the model, because a calibration describes a
# person and their armband fit, not a classifier. That placement is what lets
# the trainer calibrate each training subject with that subject's own numbers
# while the GUI calibrates whoever is wearing the band right now, from the
# same files, with no second copy to keep in sync.
#
# The record is the current state; the log is the history, append-only like
# manifest.jsonl and markers.jsonl. A calibration is never edited -- a
# correction is a new measurement, and the previous one is already in the log.


def calibration_root(root: str = None) -> str:
    """
    Resolve a dataset root the way `LivePredictor.full_path` resolves the
    model path: a relative root is relative to the PROJECT, not to whatever
    the current working directory happens to be. Without this a calibration
    would be found when the app is launched from its own folder and silently
    missed when it is launched from anywhere else.
    """
    root = root or cfg.DATA_FOLDER
    return (os.path.abspath(root) if os.path.isabs(root)
            else os.path.abspath(os.path.join(cfg.PROJECT_DIR, root)))


def calibration_path(root: str = None, person_id: str = None) -> str:
    person = safe_slug(person_id or cfg.PERSON_ID, "person")
    return os.path.join(calibration_root(root), person, cfg.CALIBRATION_NAME)


def calibration_log_path(root: str = None, person_id: str = None) -> str:
    person = safe_slug(person_id or cfg.PERSON_ID, "person")
    return os.path.join(calibration_root(root), person, cfg.CALIBRATION_LOG_NAME)


def append_calibration_log(path: str, entry: dict) -> str:
    """One JSON object per line, appended. Nothing in the file is ever rewritten."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return path


def load_calibration(root: str = None, person_id: str = None) -> Optional[dict]:
    """
    The wearer's current calibration, or None.

    A missing file is the normal case, not an error: calibration is optional
    and most wearers will not have one. A corrupt file is also None -- the
    uncalibrated pipeline is the safe answer, and the GUI reports the reason
    separately through `calibration_status()`.
    """
    path = calibration_path(root, person_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def save_calibration(record: dict, root: str = None,
                     person_id: str = None) -> str:
    """
    Write the wearer's calibration and append it to their log.

    Refuses a record that does not validate, so a calibration that would
    divide a wearer's signal by noise cannot reach the disk, let alone the
    trainer. Raises ValueError with the reasons -- the caller has a human in
    front of it and should show them.
    """
    person = safe_slug(person_id or record.get("person_id") or cfg.PERSON_ID,
                       "person")
    ok, problems = validate_calibration(record)
    if not ok:
        raise ValueError("; ".join(problems))

    stored = dict(record)
    stored["person_id"] = person
    path = calibration_path(root, person)
    ensure_dir(os.path.dirname(path))
    previous = load_calibration(root, person)

    #  Write-then-replace: a kill mid-dump cannot leave a half-written
    #  calibration for the next run to divide a dataset by.
    staging = path + ".partial"
    with open(staging, "w", encoding="utf-8") as fh:
        json.dump(stored, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(staging, path)

    append_calibration_log(calibration_log_path(root, person), {
        "action": "calibrated",
        "written_iso": iso_time(time.time()),
        "replaced": bool(previous),
        "replaced_created_iso": (previous or {}).get("created_iso"),
        "record": stored,
    })
    return path


def clear_calibration(root: str = None, person_id: str = None,
                      reason: str = "") -> Optional[dict]:
    """
    Drop the wearer's current calibration. Returns the record that was active.

    The full record goes into the append-only log BEFORE the file is removed,
    so clearing loses nothing: the measurement is still on disk and can be
    put back by hand. That is why this is the one place in the project that
    removes a file it wrote -- a calibration is a measured constant, not
    recorded signal, and safety rule 4 is about signal.
    """
    person = safe_slug(person_id or cfg.PERSON_ID, "person")
    record = load_calibration(root, person)
    if record is None:
        return None
    append_calibration_log(calibration_log_path(root, person), {
        "action": "cleared",
        "written_iso": iso_time(time.time()),
        "reason": reason,
        "record": record,
    })
    try:
        os.remove(calibration_path(root, person))
    except OSError:
        pass
    return record


def load_calibration_log(root: str = None, person_id: str = None) -> List[dict]:
    """Every calibration this wearer has ever had, oldest first."""
    path = calibration_log_path(root, person_id)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
    return out


def load_calibrations(root: str = None, persons=None) -> Dict[str, dict]:
    """
    Every wearer's calibration under `root`, keyed by person id.

    Walks the same top-level folders `discover_recordings()` does and prunes
    the same underscore ones, so an archived dataset cannot contribute a
    calibration to a live run.
    """
    root = calibration_root(root)
    wanted = None if persons is None else {safe_slug(p, "person")
                                           for p in persons}
    out: Dict[str, dict] = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if name.startswith("_") or not os.path.isdir(os.path.join(root, name)):
            continue
        if wanted is not None and name not in wanted:
            continue
        record = load_calibration(root, name)
        if record is not None:
            out[name] = record
    return out


def calibration_table(root: str = None, mode: str = None,
                      persons=None) -> Dict[str, dict]:
    """
    person id -> RESOLVED calibration, for every wearer who has a usable one.

    This is what the trainer wants: resolve once per wearer, then hand the
    right entry to `model_feature_vector()` for each of that wearer's windows.
    A wearer with no calibration, or an unusable one, is simply absent, and
    the caller treats absence as "uncalibrated" -- which is the one safe
    default, because applying someone else's gain would be worse than
    applying none.
    """
    mode = (cfg.CALIBRATION_MODE if mode is None else mode) or "off"
    if mode == "off":
        return {}
    out: Dict[str, dict] = {}
    for person, record in load_calibrations(root, persons).items():
        resolved = resolve_calibration(record, mode)
        if resolved is not None:
            out[person] = resolved
    return out


def fill_missing_calibrations(table: Dict[str, dict], recordings, people,
                              mode: str, records: Dict[str, dict] = None):
    """
    Give every TRAINING wearer without a usable calibration a base one, so the
    feature space is a single scale instead of a mixed one.

    In order of preference, per missing wearer:
      1. derived from their own recorded signal, labels unused
         (`calibration_from_recordings`);
      2. the mean of the other wearers' calibrations (the shared base).
    Nothing is written to disk: the stored `calibration.json` stays the
    wearer's own measurement, and a live wearer still has to calibrate.

    Returns (table, notes) with `notes[person]` saying which route was used.
    """
    table = dict(table)
    notes: Dict[str, str] = {}
    if (mode or "off") == "off":
        return table, notes
    records = records or {}
    for person in people:
        if person in table:
            continue
        derived = calibration_from_recordings(recordings, person)
        resolved = resolve_calibration(derived, mode) if derived else None
        if resolved is not None:
            table[person] = resolved
            notes[person] = "derived from their own recordings (labels unused)"
    missing = [p for p in people if p not in table]
    if missing:
        base = [r for r in records.values() if r and validate_calibration(r)[0]]
        if base:
            rest = np.mean([np.asarray(r["rest"], float) for r in base], axis=0)
            ref = np.mean([np.asarray(r["reference"], float) for r in base],
                          axis=0)
            shared = resolve_calibration(
                calibration_record(rest, ref, person_id="base",
                                   source="mean_of_other_wearers"), mode)
            if shared is not None:
                for person in missing:
                    table[person] = shared
                    notes[person] = (f"base: mean of {len(base)} other "
                                     f"wearer(s)' calibrations")
    return table, notes


def training_calibrations(root: str, mode: str, recordings, people,
                          records: Dict[str, dict] = None):
    """
    The calibrations a TRAINER applies: (table, notes, records_used).

    With TRAINING_CALIBRATION_SOURCE = "derived" every training wearer is
    calibrated from their OWN RECORDINGS (labels unused), whatever is in
    their calibration.json. Two reasons, both measured on 2026-10-22:
    a live REST+SQUEEZE record measures maximum effort while the recordings
    are relaxed holds (Dano: 1.2x-3.1x per channel apart), so mixing the two
    methods puts wearers on different scales; and a live record describes
    how the band sits TODAY, not how it sat when the data was recorded. The
    saved live calibration is still what PREDICTING uses -- it describes the
    band as it is being worn now. "saved" restores the old behaviour.
    """
    records = dict(records if records is not None else load_calibrations(root))
    if (mode or "off") == "off":
        return {}, {}, records
    table: Dict[str, dict] = {}
    notes: Dict[str, str] = {}
    used: Dict[str, dict] = {}
    if cfg.TRAINING_CALIBRATION_SOURCE == "derived":
        for person in people:
            derived = calibration_from_recordings(recordings, person)
            resolved = resolve_calibration(derived, mode) if derived else None
            if resolved is not None:
                table[person] = resolved
                used[person] = derived
                notes[person] = ("derived from their recordings (labels "
                                 "unused) -- training always uses this")
    for person in people:
        if person not in table and person in records:
            resolved = resolve_calibration(records[person], mode)
            if resolved is not None:
                table[person] = resolved
                used[person] = records[person]
    table, filled = fill_missing_calibrations(table, recordings, people, mode,
                                              records)
    notes.update(filled)
    return table, notes, used


def calibration_status(root: str = None, person_id: str = None,
                       mode: str = None) -> dict:
    """
    Everything a status line needs, with the reason spelled out.

    Five states, and exactly one is true at a time:
        "disabled"   CALIBRATION_ENABLED is False
        "none"       no record for this wearer
        "invalid"    a record exists but would not be safe to apply
        "stored"     a valid record exists, but the mode is "off"
        "active"     a valid record exists and is being applied
    """
    mode = (cfg.CALIBRATION_MODE if mode is None else mode) or "off"
    person = safe_slug(person_id or cfg.PERSON_ID, "person")
    record = load_calibration(root, person)
    info = {
        "person_id": person,
        "mode": mode,
        "record": record,
        "path": calibration_path(root, person),
        "problems": [],
        "summary": "",
    }
    if not cfg.CALIBRATION_ENABLED:
        info["state"] = "disabled"
        info["summary"] = "calibration is disabled in the configuration"
        return info
    if record is None:
        info["state"] = "none"
        info["summary"] = f"{person} has no calibration"
        return info
    ok, problems = validate_calibration(record)
    info["problems"] = problems
    if not ok:
        info["state"] = "invalid"
        info["summary"] = f"{person}: calibration refused -- {problems[0]}"
        return info
    info["state"] = "stored" if mode == "off" else "active"
    info["summary"] = describe_calibration(record, mode)
    return info

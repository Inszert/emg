"""
Switching a gesture off in the live classifiers, without retraining.

A model knows exactly the labels it was trained on, so removing a gesture
"properly" means a retrain -- minutes for the ML vote, ~10 for the net -- and
adding it back means another. For a gesture the model already knows that is
unnecessary: the model's answer for the remaining classes can be read off its
own output. Zero the switched-off classes' probabilities and renormalise.

    probabilities  [FIST 0.60, OPEN 0.25, REST 0.05, THUMB 0.10]
    FIST off  ->   [FIST 0.00, OPEN 0.63, REST 0.13, THUMB 0.25]

That is the model's answer to "which of the REMAINING gestures is this?",
the question the operator is asking. It is not the answer a model *trained
without* FIST would give -- that model would also have learned from the other
classes' data differently -- and the app says so wherever the mask is shown.
What it buys is that the switch is instant, reversible, and applies to the ML
model and the neural net alike, because it touches only their output.

What a mask can NOT do is add a class. A gesture the model was never trained
on (recorded, or just defined) has no output column to read; it needs a
retrain, and `classifier_status()` says so.

When every remaining class has probability zero -- the model put all of it on
a switched-off gesture, which a forest does readily -- there is no answer to
renormalise. That is reported as "no answer" (uniform probabilities, `label`
NO_ANSWER), never as a guess dressed as a result.

Qt-free and torch-free: the mask is applied to numbers.
"""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from emg_app import config as cfg

#  What a prediction's label says when the mask left nothing to choose from.
NO_ANSWER = "?"
#  Below this much probability left on the active classes there is no answer.
MIN_MASS = 1e-9
#  A classifier with fewer active classes than this is not a classifier.
MIN_ACTIVE = 2
#  Below this share of the model's own probability left on the gestures that
#  are on, the panels say so: the renormalised answer is real arithmetic but
#  the model was not really looking at any of the gestures that are on.
LOW_KEPT = 0.5


def off_set(off: Iterable[str] = None) -> frozenset:
    """The switched-off gestures: `off`, or the live setting."""
    return frozenset(cfg.CLASSIFIER_OFF if off is None else off)


def off_key(labels: Sequence[str], off: Iterable[str] = None) -> tuple:
    """
    The mask as it applies to THESE labels, comparable between calls.

    Smoothing averages the last few windows' probabilities, so a window
    scored under one mask must never be averaged with one scored under
    another. Callers compare this key and clear their history when it moves.
    """
    gone = off_set(off)
    return tuple(name for name in labels if name in gone)


def active_labels(labels: Sequence[str], off: Iterable[str] = None) -> List[str]:
    """`labels` without the switched-off ones, in the model's order."""
    gone = off_set(off)
    return [name for name in labels if name not in gone]


def apply_class_mask(probabilities, labels: Sequence[str],
                     off: Iterable[str] = None) -> Tuple[np.ndarray, bool]:
    """
    Mask a probability vector. Returns `(probabilities, answered)`.

    The result has the same width and order as `labels`, switched-off classes
    at exactly 0.0 and the rest summing to 1. With nothing switched off it is
    returned UNCHANGED (not renormalised), so the unmasked path is the old
    path bit for bit. `answered` is False when the active classes held no
    probability at all, in which case the vector is uniform over them.
    """
    p = np.asarray(probabilities, dtype=float)
    gone = off_set(off)
    if not gone or p.ndim != 1 or p.shape[0] != len(labels):
        return p, True
    keep = np.array([name not in gone for name in labels], dtype=bool)
    if keep.all():
        return p, True
    if not keep.any():
        return np.zeros_like(p), False
    masked = np.where(keep, np.nan_to_num(p, nan=0.0), 0.0)
    mass = float(masked.sum())
    if mass < MIN_MASS:
        uniform = np.where(keep, 1.0 / int(keep.sum()), 0.0)
        return uniform, False
    return masked / mass, True


def kept_mass(probabilities, labels: Sequence[str],
              off: Iterable[str] = None) -> float:
    """
    The share of the model's OWN probability that sat on gestures still on.

    1.0 with nothing off. After masking, the answer is renormalised to sum to
    1, so a window the model was 99.9 % sure was a switched-off gesture still
    reads as "100 % OPEN" -- true to the arithmetic, misleading to a person.
    This is the number that says how much of the model's belief survived.
    """
    p = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.0)
    gone = off_set(off)
    if not gone or p.ndim != 1 or p.shape[0] != len(labels):
        return 1.0
    total = float(p.sum())
    if total <= 0:
        return 0.0
    keep = np.array([name not in gone for name in labels], dtype=bool)
    return float(p[keep].sum() / total)


def validate_off(off: Iterable[str],
                 model_labels: Dict[str, Sequence[str]] = None
                 ) -> Tuple[List[str], List[str]]:
    """
    `(cleaned, problems)` for a requested switched-off set.

    Never raises. A request that would leave a model with fewer than
    MIN_ACTIVE classes is trimmed back: the gestures are dropped from the
    off-list, last first, until every model keeps enough, and each trim is
    reported. `model_labels` is {"ML model": [...], "neural net": [...]}; with
    none given only duplicates and blanks are cleaned.
    """
    cleaned: List[str] = []
    for name in off or []:
        name = str(name).strip()
        if name and name not in cleaned:
            cleaned.append(name)
    problems: List[str] = []
    for model, labels in (model_labels or {}).items():
        labels = list(labels)
        while len(active_labels(labels, cleaned)) < MIN_ACTIVE:
            touching = [n for n in cleaned if n in labels]
            if not touching:
                break
            back = touching[-1]
            cleaned.remove(back)
            problems.append(
                f"{back} stays ON: the {model} would be left with fewer than "
                f"{MIN_ACTIVE} gestures to tell apart.")
    return cleaned, problems


def classifier_status(labels: Sequence[str], recorded: Dict[str, int] = None,
                      off: Iterable[str] = None) -> Dict[str, dict]:
    """
    Per gesture: what the operator needs to know about it for ONE model.

    {name: {"known": bool, "on": bool, "needs_retrain": bool, "note": str}}

    * known         -- the model has an output column for it
    * on            -- known and not switched off: it takes part in the answer
    * needs_retrain -- not known, but recordings of it exist
    """
    known = set(labels)
    gone = off_set(off)
    recorded = recorded or {}
    out = {}
    for name in list(labels) + [g for g in recorded if g not in known]:
        is_known = name in known
        n = int(recorded.get(name, 0))
        if is_known:
            on = name not in gone
            note = ("on" if on else
                    "switched OFF -- no retrain needed to switch it back on")
        else:
            on = False
            note = ("not in this model: RETRAIN to include it" if n
                    else "not in this model, and nothing recorded yet")
        out[name] = {"known": is_known, "on": on,
                     "needs_retrain": (not is_known) and n > 0, "note": note}
    return out


def describe_off(labels: Sequence[str], off: Iterable[str] = None) -> str:
    """One line for a panel, or "" when nothing the model knows is off."""
    gone = off_key(labels, off)
    if not gone:
        return ""
    return ("switched OFF: " + ", ".join(gone)
            + "  (masked out of the answer, nothing retrained)")


def kept_note(prediction) -> str:
    """
    A caution for a panel when a switched-off gesture swallowed most of the
    model's belief, or "".

    Masking renormalises, so a window the model took for a gesture that is now
    off can still come out as "100 % OPEN". That is the arithmetic, not the
    model's confidence, and this says so. Works on a `Prediction` or a
    `NeuralPrediction` (it only reads `off` and `kept`).
    """
    kept = getattr(prediction, "kept", 1.0)
    if getattr(prediction, "off", None) and kept < LOW_KEPT:
        return (f"   ·  only {kept * 100:.0f} % of the model's own probability "
                "was on gestures that are on")
    return ""

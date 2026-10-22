"""
TEST MODEL scoring: confusion_metrics, ModelTestSession and the per-person
results log (no Qt).
"""

import json
import math
import os
import random
from collections import Counter
from typing import List

import numpy as np

from emg_app.utils import safe_slug


# ------------------------------------------------------------------------------
# TEST MODEL -- a live, cued score of the SAVED model on whoever is wearing it
# ------------------------------------------------------------------------------
# Cross-validation answers "how well would a model trained without these
# recordings do on them". What someone using the app wants to know is
# different: "the model on disk, on me, today -- how often is it right?".
# Nothing measured that. This does: cue each gesture, let the deployed model
# predict through the very path LIVE CLASSIFIER uses, and count.
#
# Three numbers, because they answer three questions:
#   per window   every raw window the model classified (comparable with the
#                trainer's per-window score)
#   smoothed     what the panel actually SHOWS -- the soft vote over recent
#                windows (closest to what you experience)
#   per trial    one answer per cued hold, by majority of its windows
#                (comparable with the trainer's per-recording vote)
#
# Honest only if the sitting is NOT one the model trained on: test after
# taking the band off and putting it back on, or on another day. The dialog
# says so, and every result is appended to <person>/model_tests.jsonl with
# the model it was measured on, so the history survives retrains.

MODEL_TEST_ROUNDS = 3          # every gesture this many times, interleaved
MODEL_TEST_PREPARE_S = 1.5     # "GET READY: FIST"
MODEL_TEST_HOLD_S = 3.0        # hold the gesture this long
MODEL_TEST_SETTLE_S = 1.0      # ...and do not score the first second of it:
                               # reaction time plus the onset would count
                               # every slow start as a model error
MODEL_TEST_LOG_NAME = "model_tests.jsonl"


def confusion_metrics(matrix, labels) -> dict:
    """
    Accuracy, per-class recall/precision/F1 and macro F1, all from ONE
    matrix (rows = truth, columns = predicted), so the numbers shown next to
    a confusion matrix can always be recomputed from it. Macro F1 averages
    only the classes that occur (support > 0).
    """
    m = np.asarray(matrix, dtype=float)
    total = float(m.sum())
    per = {}
    f1s = []
    for i, name in enumerate(labels):
        support = float(m[i].sum())
        column = float(m[:, i].sum())
        recall = m[i, i] / support if support else float("nan")
        precision = m[i, i] / column if column else float("nan")
        f1 = (2 * recall * precision / (recall + precision)
              if support and column and (recall + precision) > 0
              else (0.0 if support else float("nan")))
        per[name] = {"recall": recall, "precision": precision, "f1": f1,
                     "support": int(support)}
        if support:
            f1s.append(f1)
    return {"accuracy": float(np.trace(m) / total) if total else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "per_class": per, "n": int(total)}


class ModelTestSession:
    """
    The cued test, as a wall-clock state machine with no Qt (same pattern as
    SequenceController). The dialog calls `tick(now)` on a timer, asks
    `scoring(now)` before each prediction, and hands predictions to `add()`.
    """

    IDLE, PREPARE, HOLD, DONE = "idle", "prepare", "hold", "done"

    def __init__(self, gestures, labels, rounds: int = None, seed=None,
                 prepare_s: float = None, hold_s: float = None,
                 settle_s: float = None):
        self.labels = list(labels)
        wanted = [g for g in (gestures or self.labels) if g in self.labels]
        self.gestures = wanted or list(self.labels)
        self.rounds = int(rounds or MODEL_TEST_ROUNDS)
        self.prepare_s = float(MODEL_TEST_PREPARE_S if prepare_s is None
                               else prepare_s)
        self.hold_s = float(MODEL_TEST_HOLD_S if hold_s is None else hold_s)
        self.settle_s = float(MODEL_TEST_SETTLE_S if settle_s is None
                              else settle_s)
        rng = random.Random(seed)
        plan = []
        for _ in range(self.rounds):
            block = list(self.gestures)
            rng.shuffle(block)
            #  Never the same gesture twice in a row across a round boundary:
            #  a hold that simply continues is not a new trial.
            if plan and len(block) > 1 and block[0] == plan[-1]:
                block[0], block[1] = block[1], block[0]
            plan.extend(block)
        self.plan = plan
        self.index = -1
        self.state = self.IDLE
        self.deadline = 0.0
        self.hold_started = 0.0
        #  (trial index, truth, raw label, smoothed label)
        self.records: List[tuple] = []

    @property
    def current(self) -> str:
        return self.plan[self.index] if 0 <= self.index < len(self.plan) else ""

    @property
    def finished(self) -> bool:
        return self.state == self.DONE

    def start(self, now: float) -> None:
        self.records = []
        self.index = 0
        self.state = self.PREPARE
        self.deadline = now + self.prepare_s

    def stop(self) -> None:
        """End early. What was scored so far still counts."""
        self.state = self.DONE

    def tick(self, now: float) -> bool:
        """Advance. Returns True when a NEW trial's hold has just begun -- the
        caller resets the predictor's smoothing there, so one trial's
        history never votes in the next."""
        if self.state == self.PREPARE and now >= self.deadline:
            self.state = self.HOLD
            self.hold_started = now
            self.deadline = now + self.hold_s
            return True
        if self.state == self.HOLD and now >= self.deadline:
            self.index += 1
            if self.index >= len(self.plan):
                self.state = self.DONE
            else:
                self.state = self.PREPARE
                self.deadline = now + self.prepare_s
        return False

    def scoring(self, now: float) -> bool:
        return (self.state == self.HOLD
                and now >= self.hold_started + self.settle_s)

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

    def add(self, raw_label: str, label: str) -> None:
        self.records.append((self.index, self.current, raw_label, label))

    def _matrix(self, pairs) -> np.ndarray:
        k = len(self.labels)
        m = np.zeros((k, k), dtype=int)
        where = {name: i for i, name in enumerate(self.labels)}
        for truth, guess in pairs:
            if truth in where and guess in where:
                m[where[truth], where[guess]] += 1
        return m

    def summary(self) -> dict:
        raw = self._matrix((r[1], r[2]) for r in self.records)
        smooth = self._matrix((r[1], r[3]) for r in self.records)
        votes, truth_of = {}, {}
        for trial, truth, raw_label, _ in self.records:
            votes.setdefault(trial, []).append(raw_label)
            truth_of[trial] = truth
        trial_pairs = [(truth_of[t], Counter(v).most_common(1)[0][0])
                       for t, v in sorted(votes.items())]
        trials = self._matrix(trial_pairs)
        planned = min(len(self.plan), max(0, self.index + 1))
        return {
            "labels": list(self.labels),
            "gestures": list(self.gestures),
            "rounds": self.rounds,
            "trials_planned": len(self.plan),
            "trials_scored": len(votes),
            "trials_reached": planned,
            "windows": int(raw.sum()),
            "per_window": confusion_metrics(raw, self.labels),
            "smoothed": confusion_metrics(smooth, self.labels),
            "per_trial": confusion_metrics(trials, self.labels),
            "confusion_window": raw.tolist(),
            "confusion_smoothed": smooth.tolist(),
            "confusion_trial": trials.tolist(),
            "timing": {"prepare_s": self.prepare_s, "hold_s": self.hold_s,
                       "settle_s": self.settle_s},
        }


def model_test_log_path(root: str, person_id: str) -> str:
    return os.path.join(root, safe_slug(person_id, "person"),
                        MODEL_TEST_LOG_NAME)


def save_model_test(root: str, person_id: str, result: dict) -> str:
    """Append one test to the wearer's append-only history. Never rewrites."""
    path = model_test_log_path(root, person_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe(result)) + "\n")
    return path


def load_model_tests(root: str, person_id: str) -> List[dict]:
    out = []
    try:
        with open(model_test_log_path(root, person_id), encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _json_safe(value):
    """NaN is not JSON; a class with no support has an undefined recall."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value

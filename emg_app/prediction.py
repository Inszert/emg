"""
LivePredictor: the deployed classifier on the live stream.
"""

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from emg_app import config as cfg
from emg_app import zoo
from emg_app.utils import nominal_rate
from emg_app.class_mask import (
    NO_ANSWER, active_labels, apply_class_mask, kept_mass, off_key)
from emg_app.calibration import (
    load_calibration, resolve_calibration, validate_calibration)
from emg_app.features import (
    model_feature_names, model_feature_vector, resample_uniform)
from emg_app.dataset import model_data_status
from emg_app.model_spec import serialise_estimator


@dataclass
class Prediction:
    """One classifier output. Every field comes from the model, nothing is invented."""
    label: str                      # smoothed label (what to display)
    confidence: float               # probability of `label`, or NaN
    raw_label: str                  # this window's own argmax, unsmoothed
    raw_confidence: float
    probabilities: List[float]      # smoothed, per class, aligned with `labels`
    raw_probabilities: List[float]  # this window alone, unsmoothed
    labels: List[str]
    n_samples: int                  # samples the window was built from
    smoothed_over: int              # how many windows the vote covered
    unix_time: float
    #  Gestures switched off in the classifier for this answer: masked out of
    #  `probabilities` (exactly 0.0) and of the vote, never retrained away.
    off: List[str] = field(default_factory=list)
    #  False when the mask left nothing to choose from (the model put all of
    #  its probability on switched-off gestures); `label` is then NO_ANSWER.
    answered: bool = True
    #  How much of the model's own probability for THIS window sat on the
    #  gestures that are on (1.0 with nothing off). Low means the model was
    #  looking at a switched-off gesture and the renormalised answer is thin.
    kept: float = 1.0

    @property
    def certain(self) -> bool:
        if math.isnan(self.confidence) or cfg.PREDICTION_MIN_CONFIDENCE <= 0:
            return True
        return self.confidence >= cfg.PREDICTION_MIN_CONFIDENCE

    @property
    def percent(self) -> str:
        if math.isnan(self.confidence):
            return "--"
        return f"{self.confidence * 100:.0f} %"


class MeanProba:
    """
    A soft vote over already-fitted estimators: the mean of their
    `predict_proba`. Built in memory by `LivePredictor` when several zoo models
    are selected; never saved. The members must share the same input features
    (`check_vote` / `_vote_problem` make sure of that before one is built).
    """

    def __init__(self, members):
        self.members = list(members)
        self.n_jobs = 1

    def predict_proba(self, X):
        return np.mean([np.asarray(m.predict_proba(X), dtype=float)
                        for m in self.members], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


#  What two payloads must agree on to be averaged (see zoo.COMPAT_KEYS).
_VOTE_FIELDS = ("labels", "fs", "window_s", "window_samples", "representation",
                "feature_set", "freq_features")


def _vote_problem(payloads, names):
    """`None` if the payloads can be voted, else the reason they cannot."""
    ref = payloads[0]
    for name, other in zip(names[1:], payloads[1:]):
        for key in _VOTE_FIELDS:
            if ref.get(key) != other.get(key):
                return (f"{names[0]} and {name} disagree on {key} "
                        f"({ref.get(key)!r} against {other.get(key)!r}), so "
                        f"their probabilities cannot be averaged")
        a = (ref.get("calibration") or {}).get("mode", "off")
        b = (other.get("calibration") or {}).get("mode", "off")
        if a != b:
            return (f"{names[0]} and {name} were trained on different "
                    f"calibration modes ({a!r} against {b!r})")
        a = (ref.get("adaptation") or {}).get("mode", "off")
        b = (other.get("adaptation") or {}).get("mode", "off")
        if a != b:
            return (f"{names[0]} and {name} differ in adaptation to the "
                    f"person ({a!r} against {b!r}), so they are fed "
                    f"differently standardised features")
    return None


class LivePredictor:
    """
    Real-time gesture prediction.

    Inert until a trained model exists: with no model file nothing is shown
    and the buttons stay disabled. It never invents a label or a percentage.

    The feature path goes through model_feature_vector(), the same function
    train_model.py uses, so the live input cannot drift away from what the
    model was fitted on. The live window is resampled onto the model's own
    grid first (resample_uniform), which is what makes the prediction
    independent of how the BLE link happens to be behaving.

    Expected joblib payload (written by train_model.py):

        {"model": <estimator>, "labels": [...], "fs": 60.0,
         "window_s": 0.2, "window_samples": 12, "representation": "raw",
         "feature_set": "level", "freq_features": True, "stats": {...}}
    """

    def __init__(self, path: str = None):
        self.path = path or cfg.MODEL_PATH
        #  An explicit path is a request for exactly that file, so it bypasses
        #  the zoo selection (tests and tools rely on it).
        self._explicit = path is not None
        #  Names of the models behind `model` when it is a soft vote, else [].
        self.members: List[str] = []
        self.active_paths: List[str] = []
        #  Adaptation to the person (emg_app/adaptation.py): only a model
        #  TRAINED on per-person standardised features carries
        #  {"mode": "person_features"}; the statistics of whoever wears the
        #  band come from `set_adaptation`.
        self.adaptation: dict = {"mode": "off"}
        self.adapt_stats = None
        self.adapt_applied = False
        self.model = None
        self.labels: List[str] = []
        self.window_s = cfg.WINDOW_MS / 1000.0
        self.window_samples = 12
        self.fs = nominal_rate()
        self.representation = "raw"
        self.feature_set = "level"
        self.freq_features = True
        # The per-wearer calibration this model was FITTED with, and the
        # resolved calibration of whoever is wearing the band now. If the
        # model needs one and the wearer has none, `load()` refuses: feeding
        # device counts to a model fitted on a calibrated scale produces
        # confident wrong answers, not an error.
        self.calibration_mode = "off"
        self.calibration: Optional[dict] = None
        self.calibration_note = ""
        self.stats: dict = {}
        self.payload_meta: dict = {}
        self.status = "disabled"
        self.history = deque(maxlen=max(1, cfg.PREDICTION_SMOOTHING))
        self.prob_history = deque(maxlen=max(1, cfg.PREDICTION_SMOOTHING))
        #  The class mask the history was built under (class_mask.off_key).
        #  When it moves, the history is dropped: a window scored with FIST on
        #  must not be averaged with one scored with FIST off.
        self._off_key: tuple = ()
        self.last: Optional[Prediction] = None
        # The exact row the last prediction was made from, kept so the
        # decision panel can run the other candidates on the SAME input
        # instead of building a second feature path that could drift.
        self.last_features: Optional[np.ndarray] = None
        self.last_window: Optional[np.ndarray] = None

    # -- loading --------------------------------------------------------------

    @property
    def full_path(self) -> str:
        return (self.path if os.path.isabs(self.path)
                else os.path.join(cfg.PROJECT_DIR, self.path))

    def load(self) -> bool:
        self.model = None
        self.members = []
        self.active_paths = []
        if not cfg.ENABLE_LIVE_PREDICTION:
            self.status = "disabled in the configuration"
            return False
        #  With model selection OFF (the default) this list is empty and the
        #  original single file is loaded exactly as before.
        paths = [] if self._explicit else zoo.selected_paths("ml")
        if len(paths) >= 2:
            return self._load_vote(paths)
        return self._load_single(paths[0] if paths else self.full_path)

    def _read_payload(self, path: str):
        """`(payload, None)` or `(None, reason)`."""
        if not os.path.exists(path):
            return None, "no model file"
        try:
            import joblib
        except ImportError:
            return None, "joblib is not installed"
        try:
            payload = joblib.load(path)
        except Exception as exc:
            return None, f"could not load the model: {exc}"
        if not isinstance(payload, dict) or "model" not in payload:
            return None, "unsupported model file (expected a dict payload)"
        return payload, None

    def _adopt(self, payload: dict, path: str) -> bool:
        """Take the model's description from `payload`; check its calibration."""
        self.labels = list(payload.get("labels", cfg.GESTURES))
        self.fs = float(payload.get("fs", nominal_rate()))
        self.window_s = float(payload.get(
            "window_s", payload.get("window_ms", cfg.WINDOW_MS) / 1000.0))
        self.window_samples = int(payload.get(
            "window_samples", max(4, round(self.window_s * self.fs))))
        self.representation = payload.get("representation", "raw")
        self.feature_set = payload.get("feature_set", "level")
        self.freq_features = bool(payload.get("freq_features", True))
        self.stats = dict(payload.get("stats", {}))
        self.adaptation = dict(payload.get("adaptation") or {"mode": "off"})
        self.adapt_stats = None
        self.adapt_applied = False
        self.payload_meta = {k: v for k, v in payload.items()
                             if k not in ("model", "stats")}
        self.loaded_mtime = os.path.getmtime(path) \
            if os.path.exists(path) else 0.0
        return self._load_calibration(payload.get("calibration"))

    def _finish(self) -> bool:
        """Fail loudly here rather than producing nonsense at 7 Hz."""
        n_expected = len(model_feature_names(self.feature_set,
                                             self.freq_features))
        self.status = "ready"
        self.reset()
        try:
            dummy = np.zeros((1, n_expected), dtype=np.float32)
            self.model.predict(dummy)
        except Exception as exc:
            self.model = None
            self.status = (f"model expects different features than "
                           f"{self.feature_set!r} produces ({n_expected}): {exc}")
            return False
        return True

    def _load_single(self, path: str) -> bool:
        payload, problem = self._read_payload(path)
        if problem:
            self.status = problem
            return False
        self.model = payload["model"]
        #  One live row at a time: a joblib pool per call tripled the
        #  forests' cost (serialise_estimator). Same answers.
        serialise_estimator(self.model)
        self.active_paths = [path]
        if not self._adopt(payload, path):
            self.model = None
            return False
        return self._finish()

    def _load_vote(self, paths) -> bool:
        """Several zoo models, answered as the mean of their probabilities."""
        payloads, names = [], []
        for path in paths:
            payload, problem = self._read_payload(path)
            if problem:
                self.status = f"{os.path.basename(os.path.dirname(path))}: " \
                              f"{problem}"
                return False
            payloads.append(payload)
            names.append((payload.get("stats") or {}).get("model_name")
                         or os.path.basename(os.path.dirname(path)))
        problem = _vote_problem(payloads, names)
        if problem:
            self.status = f"cannot vote: {problem}"
            return False
        members = [pl["model"] for pl in payloads]
        for member in members:
            serialise_estimator(member)
        self.model = MeanProba(members)
        self.members = list(names)
        self.active_paths = list(paths)
        if not self._adopt(payloads[0], paths[0]):
            self.model = None
            return False
        #  The first member's numbers describe THAT model, not the vote, so
        #  they must not stay on display: keep only what is true of the vote.
        self.stats = {"model_name": "soft vote: " + " + ".join(names),
                      "ensemble_of": list(names),
                      "validation_protocol": "see the members' own statistics"}
        self.payload_meta = dict(self.payload_meta)
        self.payload_meta["ensemble_of"] = list(names)
        return self._finish()

    def _load_calibration(self, block: dict) -> bool:
        """
        Resolve the calibration this model needs for the CURRENT wearer.

        Returns False and sets `status` when the model was fitted on
        calibrated features and the wearer has no usable calibration. That is
        a refusal on purpose: the model would still answer, at ~7 Hz, with
        every amplitude feature off by that wearer's gain, and a confidently
        wrong label is worse than a panel that says what is missing. Same
        rule as the decision panel -- a mismatch is refused, not fed.

        A model trained with "off" never looks at a calibration, so a wearer
        who has one is unaffected by having it.
        """
        self.calibration = None
        self.calibration_note = ""
        block = block if isinstance(block, dict) else {}
        mode = block.get("mode", "off") or "off"
        self.calibration_mode = mode
        if mode == "off":
            return True

        record = load_calibration(cfg.DATA_FOLDER, cfg.PERSON_ID)
        if record is None:
            self.status = (
                f"this model was trained on {mode!r} per-wearer calibration, "
                f"and {cfg.PERSON_ID} has none. Press CALIBRATE, or retrain with "
                f"calibration off.")
            return False
        resolved = resolve_calibration(record, mode)
        if resolved is None:
            _ok, problems = validate_calibration(record)
            self.status = (
                f"{cfg.PERSON_ID}'s calibration cannot be applied in {mode!r} "
                f"mode: {problems[0] if problems else 'invalid record'}")
            return False
        self.calibration = resolved
        trained_on = list(block.get("subjects") or [])
        self.calibration_note = (
            f"{mode} calibration, {cfg.PERSON_ID} measured "
            f"{(record.get('created_iso') or '')[:16].replace('T', ' ')}"
            + (f" | model trained on: {', '.join(trained_on)}"
               if trained_on else ""))
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    # -- adaptation to the person ---------------------------------------------

    @property
    def adapts(self) -> bool:
        """Was this model trained on per-person standardised features?"""
        return self.adaptation.get("mode") == "person_features"

    def set_adaptation(self, mean, std) -> None:
        """The wearer's per-feature statistics (`adaptation.ml_person_stats`)."""
        self.adapt_stats = (np.asarray(mean, dtype=np.float32),
                            np.asarray(std, dtype=np.float32))

    def clear_adaptation(self) -> None:
        self.adapt_stats = None

    def _adapt_features(self, features: np.ndarray) -> np.ndarray:
        """
        Standardise the feature row the way the model was trained -- with the
        wearer's statistics when adaptation is on and they are known, with the
        stored global ones otherwise (the original pipeline for this model).
        Models not trained with adaptation are returned untouched.
        """
        self.adapt_applied = False
        if not self.adapts:
            return features
        if cfg.ADAPT_LIVE and self.adapt_stats is not None:
            mean, std = self.adapt_stats
            self.adapt_applied = True
        else:
            mean = self.adaptation.get("global_mean")
            std = self.adaptation.get("global_std")
            if mean is None or std is None:
                return features
        return ((features - np.asarray(mean, dtype=np.float32))
                / np.asarray(std, dtype=np.float32)).astype(np.float32)

    def reset(self) -> None:
        self.history.clear()
        self.prob_history.clear()
        self.last = None
        self.last_features = None
        self.last_window = None

    def describe(self) -> str:
        if not self.ready:
            return self.status
        name = self.stats.get("model_name", "model")
        acc = self.stats.get("recording_vote_accuracy")
        suffix = f", {acc * 100:.0f} % per-gesture CV" if acc else ""
        cal = (f"  |  {self.calibration_mode} calibration"
               if self.calibration_mode != "off" else "")
        if self.adapts:
            cal += ("  |  adapted to the person" if self.adapt_applied
                    else "  |  adaptation: global statistics")
        return (f"{name}  |  {self.window_s * 1000:.0f} ms @ {self.fs:.0f} Hz  |  "
                f"{self.feature_set} features{cal}{suffix}")

    def data_status(self, fingerprint: dict = None) -> dict:
        """
        Is a model trained, on what, and is that still what is on disk?

        Delegates to `model_data_status()` so the trainers, the GUI and the
        tests all answer the question the same way.
        """
        if not self.ready:
            return model_data_status(None)
        payload = dict(self.payload_meta)
        payload["stats"] = self.stats
        return model_data_status(payload, fingerprint)

    # -- inference ------------------------------------------------------------

    def window_seconds(self) -> float:
        return float(self.window_s)

    def predict(self, emg: np.ndarray, t: np.ndarray) -> Optional[Prediction]:
        """
        Classify one live window.

        `emg`/`t` are raw ring-buffer samples covering roughly window_s
        seconds; they are resampled onto the model's grid here. Returns None
        when there is not enough data to say anything.
        """
        if not self.ready:
            return None
        emg = np.asarray(emg)
        t = np.asarray(t, dtype=np.float64)
        if emg.shape[0] < 4 or t.size != emg.shape[0]:
            return None

        block = resample_uniform(emg, t, self.window_samples)
        features = model_feature_vector(
            block, self.representation, self.feature_set,
            self.freq_features, self.fs, self.calibration,
        )[None, :]
        features = self._adapt_features(features)
        self.last_features = features
        self.last_window = block

        key = off_key(self.labels)
        if key != self._off_key:
            self._off_key = key
            self.history.clear()
            self.prob_history.clear()

        answered = True
        kept = 1.0
        try:
            has_proba = hasattr(self.model, "predict_proba")
            if has_proba:
                probabilities = np.asarray(
                    self.model.predict_proba(features), dtype=float)[0]
                kept = kept_mass(probabilities, self.labels)
                #  Gestures switched off in the settings are masked out of the
                #  answer here -- on the OUTPUT, so no retrain is involved and
                #  switching one back on takes effect on the next window.
                probabilities, answered = apply_class_mask(
                    probabilities, self.labels)
                index = int(np.argmax(probabilities))
                confidence = (float(probabilities[index]) if answered
                              else float("nan"))
            else:
                index = int(np.asarray(self.model.predict(features)).ravel()[0])
                probabilities = np.full(len(self.labels), float("nan"))
                confidence = float("nan")
        except Exception as exc:
            self.status = f"prediction failed: {exc}"
            self.model = None
            return None

        raw_label = self._name(index) if answered else NO_ANSWER
        if answered:
            self.history.append(index)

        # Smoothing is SOFT voting: average the probability vectors over the
        # recent windows and take the argmax of the average.
        #
        # Hard majority voting on labels was tried first and is wrong to show:
        # it can report a label whose current probability is 0.16 while the
        # latest window puts 0.84 on something else, so the number next to the
        # label contradicts the label. Averaging the distributions keeps the
        # displayed confidence consistent with the displayed answer, and is
        # also the better estimator.
        if not has_proba:
            smoothed_probs = probabilities
            smoothed = index
            smooth_conf = confidence
        else:
            #  A window the mask left unanswerable adds nothing to the vote;
            #  earlier answered windows still speak, so one stray window does
            #  not blank the display.
            if answered:
                self.prob_history.append(probabilities)
            if self.prob_history:
                smoothed_probs = np.mean(np.vstack(self.prob_history), axis=0)
                smoothed = int(np.argmax(smoothed_probs))
                smooth_conf = float(smoothed_probs[smoothed])
            else:
                smoothed_probs = probabilities
                smoothed = -1
                smooth_conf = float("nan")

        prediction = Prediction(
            label=NO_ANSWER if smoothed < 0 else self._name(smoothed),
            confidence=smooth_conf,
            raw_label=raw_label,
            raw_confidence=confidence,
            probabilities=[float(p) for p in smoothed_probs],
            raw_probabilities=[float(p) for p in probabilities],
            labels=list(self.labels),
            n_samples=int(emg.shape[0]),
            smoothed_over=len(self.prob_history) or 1,
            unix_time=time.time(),
            off=list(key),
            answered=smoothed >= 0,
            kept=kept,
        )
        self.last = prediction
        return prediction

    def _name(self, index: int) -> str:
        return (self.labels[index] if 0 <= index < len(self.labels)
                else str(index))

    @property
    def active_labels(self) -> List[str]:
        """The gestures this model currently answers with (mask applied)."""
        return active_labels(self.labels)


def latest_model_window(acq, seconds: float):
    """
    The newest `seconds` of signal from the ring buffer -- the window the
    live classifier predicts on. Shared by PREDICT and TEST MODEL, so the
    test scores exactly the input the panel's answer is built from.
    """
    try:
        rate = acq.stats()["rate"] or nominal_rate()
    except Exception:                                      # pragma: no cover
        rate = nominal_rate()
    want = int(seconds * rate * 1.25) + 4
    emg, t = acq.ring.snapshot(max(8, want))
    # Keep only the samples inside the model's own window length.
    if t.size and (t[-1] - t[0]) > seconds:
        keep = t >= (t[-1] - seconds)
        emg, t = emg[keep], t[keep]
    return emg, t

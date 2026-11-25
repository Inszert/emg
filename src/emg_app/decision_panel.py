"""
The DECISION TRAIL panel model (CLAUDE.md 8e): loads gesture_panel.joblib
and runs its members.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np

from emg_app import config as cfg
from emg_app.class_mask import apply_class_mask, describe_off, off_key
from emg_app.model_spec import panel_path, serialise_estimator

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.prediction import LivePredictor


@dataclass
class TrailStep:
    """
    What every candidate said about ONE feature vector, and how it fused.

    `per_model` is aligned with `names`; each entry is that candidate's full
    probability vector over `labels`. `fused` is the soft vote actually
    produced by the stored ensemble, and `fused_from_mean` is the same thing
    recomputed as the plain mean of the members -- the view shows both so the
    fusion rule is demonstrated rather than asserted.
    """
    names: List[str]
    per_model: List[List[float]]
    labels: List[str]
    fused: Optional[List[float]]          # the ensemble's own output
    fused_from_mean: List[float]          # mean of the members, recomputed
    deployed_name: str                    # the model that actually predicts
    deployed: Optional[List[float]]       # its probabilities, or None
    source: str                           # "live" or "recorded"
    note: str = ""

    def decision(self, probabilities) -> int:
        return int(np.argmax(np.asarray(probabilities, dtype=float)))

    def label_of(self, probabilities) -> str:
        index = self.decision(probabilities)
        return (self.labels[index] if 0 <= index < len(self.labels)
                else str(index))

    @property
    def agreement(self) -> int:
        """How many candidates picked the same class as the soft vote."""
        if not self.per_model:
            return 0
        target = self.decision(self.fused_from_mean)
        return sum(1 for p in self.per_model if self.decision(p) == target)


class DecisionPanel:
    """
    The side file that lets MODEL STATS show every candidate's decision.

    Loaded lazily and entirely optional. With no panel file the view says so
    and shows nothing else -- it never falls back to a guess, and it never
    stands in for the deployed model, which keeps predicting from
    `gesture_model.joblib` exactly as before.

    Two sources of truth, kept apart on purpose:

      * `step_for_features()` runs the stored members on a LIVE feature
        vector. Those members were fitted on the whole dataset, so what they
        agree about live is a demonstration, not a score. Every caller gets
        that in `note`.
      * `recorded_step()` reads the out-of-fold trail captured during
        training. Those ARE honest numbers, and they come with the fold the
        window was held out in.

    The live path takes the feature vector LivePredictor already computed, so
    there is exactly one feature function in play. Recomputing features here
    would create a second path that could silently drift -- the same reason
    `model_feature_vector()` is shared with the trainer.
    """

    def __init__(self, path: str = None, model_path: str = None):
        self.path = path or panel_path(model_path)
        self.loaded = False
        self.status = "not loaded"
        self.names: List[str] = []
        self.vote_name: Optional[str] = None
        self.labels: List[str] = []
        self.members: dict = {}
        self.cv: dict = {}
        self.setup: dict = {}
        self.fusion: dict = {}
        self.protocol: dict = {}
        self.recordings: List[dict] = []
        self.space: dict = {}
        self.created_iso = ""
        self.winner = ""
        self.n_windows = 0
        self.n_recordings = 0
        # First-call warm-up state. See warm_up().
        self.warming = False
        self.warm = False
        self.warm_ms = 0.0
        self._trail: dict = {}
        self._mtime = 0.0

    # -- loading --------------------------------------------------------------

    def load(self) -> bool:
        """Read the panel. False (with a reason in `status`) is normal."""
        self.loaded = False
        if not cfg.ENABLE_DECISION_PANEL:
            self.status = "disabled in the configuration"
            return False
        if not os.path.exists(self.path):
            self.status = ("no decision panel yet -- retrain to build one "
                           "(train_model.py writes it next to the model)")
            return False
        try:
            import joblib
        except ImportError:
            self.status = "joblib is not installed"
            return False
        try:
            payload = joblib.load(self.path)
        except Exception as exc:
            self.status = f"could not load the decision panel: {exc}"
            return False
        if not isinstance(payload, dict) or payload.get("kind") != "decision_panel":
            self.status = "that file is not a decision panel"
            return False

        self.names = list(payload.get("member_order") or [])
        self.vote_name = payload.get("vote_name")
        self.labels = list(payload.get("labels") or [])
        self.members = dict(payload.get("members") or {})
        self.cv = dict(payload.get("cv") or {})
        self.setup = dict(payload.get("setup") or {})
        self.fusion = dict(payload.get("fusion") or {})
        self.protocol = dict(payload.get("protocol") or {})
        self.recordings = list(payload.get("recordings") or [])
        self.space = dict(payload.get("space") or {})
        self.created_iso = payload.get("created_iso", "")
        self.winner = payload.get("winner", "")
        self.n_windows = int(payload.get("n_windows", 0) or 0)
        self.n_recordings = int(payload.get("n_recordings",
                                            len(self.recordings)) or 0)
        self._trail = dict(payload.get("trail") or {})
        self._mtime = os.path.getmtime(self.path)

        if not self.names or not self.members:
            self.status = "the decision panel is empty"
            return False
        self._serialise_members()
        self.loaded = True
        self.status = "ready"
        return True

    def _serialise_members(self) -> None:
        """
        Force the members to predict on one thread.

        Measured on this machine: running the seven candidates over a single
        live window took 2.7 s with the forests left at `n_jobs=-1`, and
        0.03 s with them at 1. The work is one row through 400 trees -- a few
        hundred microseconds -- so the entire 2.7 s was joblib standing up a
        worker pool, twice, for every window. At 7 predictions a second the
        view could not keep up and the GUI thread was the one paying.

        `n_jobs` affects only how a fitted forest is evaluated, never what it
        answers, so this changes no number on screen. It is applied on load
        rather than at training time because the saved estimator is also the
        one the sweep used, where many rows at once do want the cores.
        """
        for model in self.members.values():
            serialise_estimator(model)

    @property
    def ready(self) -> bool:
        return self.loaded and bool(self.members)

    def warm_up(self) -> float:
        """
        Pay every candidate's first-call cost once, up front.

        Measured here: the second and later windows cost about 45 ms for all
        seven, but the FIRST costs ~2.5 s, essentially all of it inside
        HistGradientBoosting's initial `predict_proba`, which builds its
        prediction kernel on demand (19 ms RandomForest, 21 ms ExtraTrees,
        under 1 ms for the rest -- and 2416 ms for HistGradBoost's first
        call against 2.5 ms for its second).

        Left unpaid, that lands on whichever live window arrives first and
        freezes the GUI thread for two and a half seconds. The dialog
        therefore runs this on a worker thread before showing anything live.

        A row of zeros is used deliberately: the answer is thrown away, so
        it only has to be the right SHAPE. Nothing it returns is ever shown.
        """
        if not self.ready or self.warm:
            return 0.0
        self.warming = True
        started = time.perf_counter()
        row = np.zeros((1, int(self.setup.get("n_features", 0) or 0)),
                       dtype=np.float32)
        try:
            if row.shape[1]:
                for model in self.members.values():
                    try:
                        model.predict_proba(row)
                    except Exception:                      # pragma: no cover
                        pass
        finally:
            self.warm_ms = (time.perf_counter() - started) * 1000.0
            self.warm = True
            self.warming = False
        return self.warm_ms

    @property
    def has_space(self) -> bool:
        """
        Is there a usable 2-D projection of the feature space?

        Deliberately `loaded` and not `ready`: `ready` also requires the
        fitted members, because the DECISION TRAIL has to run them. Drawing
        the scatter runs no model at all -- it needs the stored transform and
        nothing else -- so a panel that carries a projection can show it even
        if its members are unusable.
        """
        space = self.space or {}
        return bool(self.loaded and space.get("kind") == "pca"
                    and space.get("components") is not None
                    and space.get("coords") is not None)

    def project(self, features) -> Optional[Tuple[float, float]]:
        """
        One live feature vector -> its (x, y) in the stored PCA space.

        The transform is the one fitted at training time, applied here and
        nowhere else, so a live point and a training point are placed by
        exactly the same arithmetic. Refitting a projection on live data
        would move the axes under the scatter and make the two
        incomparable -- the same reason the trail reuses
        `LivePredictor.last_features` instead of recomputing features.

        None when there is no projection, or when the vector is the wrong
        width for it. A silently reshaped vector would land somewhere
        plausible and mean nothing.
        """
        if not self.has_space or features is None:
            return None
        space = self.space
        vec = np.asarray(features, dtype=np.float64).ravel()
        mean = np.asarray(space["mean"], dtype=np.float64)
        if vec.shape[0] != mean.shape[0]:
            return None
        scale = np.asarray(space["scale"], dtype=np.float64)
        comps = np.asarray(space["components"], dtype=np.float64)
        out = ((vec - mean) / scale) @ comps.T
        if not np.all(np.isfinite(out)):
            return None
        return float(out[0]), float(out[1])

    def matches(self, predictor: "LivePredictor") -> bool:
        """
        Is this panel about the model that is loaded?

        A panel built for a different window length or feature set would be
        fed vectors it was never fitted on. Rather than predict nonsense, the
        view refuses and says the panel is stale.
        """
        if not self.ready or not predictor.ready:
            return False
        setup = self.setup
        return (list(self.labels) == list(predictor.labels)
                and setup.get("feature_set") == predictor.feature_set
                and setup.get("representation") == predictor.representation
                and bool(setup.get("freq_features")) == bool(predictor.freq_features)
                and (setup.get("calibration_mode", "off")
                     == predictor.calibration_mode)
                and int(setup.get("window_samples", -1)) == int(predictor.window_samples))

    def stale_reason(self, predictor: "LivePredictor") -> str:
        """Why `matches()` said no, in words the dialog can show."""
        if not self.ready:
            return self.status
        if not predictor.ready:
            return "no model is loaded"
        setup = self.setup
        for field, mine, theirs in (
                ("labels", list(self.labels), list(predictor.labels)),
                ("feature set", setup.get("feature_set"), predictor.feature_set),
                ("representation", setup.get("representation"),
                 predictor.representation),
                ("frequency features", bool(setup.get("freq_features")),
                 bool(predictor.freq_features)),
                ("calibration mode", setup.get("calibration_mode", "off"),
                 predictor.calibration_mode),
                ("window samples", int(setup.get("window_samples", -1)),
                 int(predictor.window_samples))):
            if mine != theirs:
                return (f"the panel was built for {field} {mine!r}, the loaded "
                        f"model uses {theirs!r} -- retrain to rebuild it")
        return ""

    # -- live -----------------------------------------------------------------

    def step_for_features(self, features: np.ndarray,
                          predictor: "LivePredictor" = None) -> Optional[TrailStep]:
        """
        Run every candidate on one feature vector that was already computed.

        `features` is the (1, n) row LivePredictor built for this window, so
        both see bit-identical input. Returns None when the panel cannot
        answer; it never substitutes an estimate.
        """
        if not self.ready:
            return None
        row = np.asarray(features, dtype=np.float32)
        if row.ndim == 1:
            row = row[None, :]
        expected = int(self.setup.get("n_features", row.shape[1]))
        if row.shape[1] != expected:
            self.status = (f"this panel wants {expected} features, got "
                           f"{row.shape[1]}")
            return None

        per_model, names = [], []
        for name in self.names:
            model = self.members.get(name)
            if model is None:
                continue
            try:
                probabilities = np.asarray(
                    model.predict_proba(row), dtype=float)[0]
            except Exception as exc:
                self.status = f"{name} failed on a live window: {exc}"
                return None
            names.append(name)
            per_model.append([float(p) for p in probabilities])
        if not per_model:
            return None

        mean = np.mean(np.asarray(per_model, dtype=float), axis=0)
        #  The stored ensemble is NOT called here. Its soft vote is the mean
        #  of these same members, which build_decision_panel() checks over
        #  the whole dataset and records in `fusion` -- so calling it would
        #  re-run all seven members a second time for a number already in
        #  hand. That doubling is what made the live view miss windows.
        #  `fusion.verified` is the evidence; when it is False the view shows
        #  the ensemble's recorded output instead of trusting the mean.
        fused = None

        deployed_name = self.winner or (predictor.stats.get("model_name", "")
                                        if predictor else "")
        deployed = None
        if deployed_name in names:
            deployed = per_model[names.index(deployed_name)]

        #  Gestures switched off in the settings are masked out of the live
        #  classifier's answer, so they are masked out of what this view
        #  shows too -- otherwise the trail's final answer could name a
        #  gesture the LIVE CLASSIFIER panel above it has been told to skip.
        #  Each opinion is masked on its own, and the fusion is the mask of
        #  the members' plain mean, which is what the deployed vote does.
        note = ("these candidates were fitted on the whole dataset, so "
                "their agreement here demonstrates the vote -- it is not "
                "an accuracy estimate")
        if off_key(self.labels):
            per_model = [[float(v) for v in apply_class_mask(row, self.labels)[0]]
                         for row in per_model]
            mean = apply_class_mask(mean, self.labels)[0]
            if deployed is not None:
                deployed = [float(v) for v in
                            apply_class_mask(deployed, self.labels)[0]]
            note += "; " + describe_off(self.labels)

        return TrailStep(
            names=names, per_model=per_model, labels=list(self.labels),
            fused=fused, fused_from_mean=[float(v) for v in mean],
            deployed_name=deployed_name or "?", deployed=deployed,
            source="live",
            note=note,
        )

    # -- recorded -------------------------------------------------------------

    @property
    def has_trail(self) -> bool:
        return bool(self._trail.get("proba") is not None
                    and self._trail.get("truth") is not None)

    def trail_length(self) -> int:
        if not self.has_trail:
            return 0
        return int(np.asarray(self._trail["truth"]).shape[0])

    def windows_of(self, recording_index: int) -> List[int]:
        """Row numbers of one recording's windows, in order."""
        if not self.has_trail or self._trail.get("rec") is None:
            return []
        rec = np.asarray(self._trail["rec"])
        return [int(i) for i in np.flatnonzero(rec == int(recording_index))]

    def recorded_step(self, row: int) -> Optional[TrailStep]:
        """One window of the out-of-fold trail captured during training."""
        if not self.has_trail:
            return None
        total = self.trail_length()
        if not 0 <= row < total:
            return None
        proba = self._trail["proba"]
        names, per_model = [], []
        for name in self.names:
            block = proba.get(name)
            if block is None:
                continue
            names.append(name)
            per_model.append([float(p) for p in np.asarray(block)[row]])
        if not per_model:
            return None

        fused = None
        if self.vote_name and proba.get(self.vote_name) is not None:
            fused = [float(p) for p in np.asarray(proba[self.vote_name])[row]]
        mean = np.mean(np.asarray(per_model, dtype=float), axis=0)
        truth_index = int(np.asarray(self._trail["truth"])[row])
        fold = int(np.asarray(self._trail["fold"])[row]) if \
            self._trail.get("fold") is not None else -1
        deployed = (per_model[names.index(self.winner)]
                    if self.winner in names else None)

        return TrailStep(
            names=names, per_model=per_model, labels=list(self.labels),
            fused=fused, fused_from_mean=[float(v) for v in mean],
            deployed_name=self.winner or "?", deployed=deployed,
            source="recorded",
            note=(f"recorded label {self.labels[truth_index]}"
                  if 0 <= truth_index < len(self.labels) else "")
                 + (f", held out in fold {fold + 1}" if fold >= 0 else ""),
        )

    def recording_index_of(self, row: int) -> int:
        """Which recording a trail row belongs to, or -1."""
        rec = self._trail.get("rec")
        if rec is None:
            return -1
        rec = np.asarray(rec)
        return int(rec[row]) if 0 <= row < rec.shape[0] else -1

    def recorded_truth(self, row: int) -> int:
        if not self.has_trail:
            return -1
        truth = np.asarray(self._trail["truth"])
        return int(truth[row]) if 0 <= row < truth.shape[0] else -1

    def recorded_channels(self, row: int) -> Optional[np.ndarray]:
        """The per-channel level of a recorded window, when it was stored."""
        chan = self._trail.get("chan")
        if chan is None:
            return None
        chan = np.asarray(chan)
        if not 0 <= row < chan.shape[0]:
            return None
        return chan[row]

    def split_windows(self) -> List[int]:
        """
        Rows where the candidates did NOT all agree.

        The only interesting windows on a dataset this separable, and what
        the view's "next disagreement" button steps through.
        """
        if not self.has_trail:
            return []
        proba = self._trail["proba"]
        blocks = [np.asarray(proba[n]) for n in self.names if n in proba]
        if len(blocks) < 2:
            return []
        picks = np.stack([np.argmax(b, axis=1) for b in blocks], axis=1)
        varies = (picks != picks[:, :1]).any(axis=1)
        return [int(i) for i in np.flatnonzero(varies)]

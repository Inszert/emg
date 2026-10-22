"""
The neural-net engine: spec, sidecar and the line protocol to nn_serve.py.
No Qt, NO torch.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from emg_app import config as cfg
from emg_app.calibration import (
    load_calibration, resolve_calibration, validate_calibration)
from emg_app.features import nn_input_block, resample_uniform
from emg_app.dataset import model_data_status


# ==============================================================================
# SECTION 5b -- NEURAL NETWORK ENGINE  (subprocess; no Qt, no torch)
#
# The second opinion, kept at arm's length from the first one in three
# separate ways:
#
#   1. DIFFERENT PROCESS.  The recorder never imports torch (section 13): the
#      import costs seconds and attaches a GPU context to the process whose
#      one job is not to drop EMG samples. `nn_serve.py` holds the net, and
#      this class talks to it over stdin/stdout with one JSON object per
#      line. `tests/test_no_torch.py` still passes with the net enabled.
#   2. DIFFERENT TYPE.  The engine yields `NeuralPrediction`, never
#      `Prediction`. Nothing can pass one where the other is expected, so the
#      net's output cannot reach `LivePredictor`, the smoothing history or
#      the DECISION TRAIL's soft vote by accident.
#   3. DIFFERENT EVERYTHING ELSE.  Its own checkpoint, its own smoothing
#      window, its own timer in the GUI, its own START button, its own
#      statistics dialog.
#
# GUI PULLS, THE READER THREAD PUSHES -- the same division as
# MyoAcquisition (section 13). A background thread drains the subprocess's
# stdout into a deque; the GUI calls `poll()` on its own timer. Per-answer Qt
# signals at 7 Hz would be harmless, but the symmetry means this class stays
# Qt-free and therefore testable headless, with a fake worker script and no
# torch at all.
# ==============================================================================

#  Line prefixes. The worker's stdout also carries whatever torch or a driver
#  decides to print, so the protocol is prefix-tagged and everything else is
#  kept as diagnostic text rather than being parsed or dropped silently.
NN_META_PREFIX = "NNMETA "
NN_OUT_PREFIX = "NNOUT "
NN_ERR_PREFIX = "NNERR "
NN_BYE_PREFIX = "NNBYE"


# ---- neural-net setup (what TRAIN NEURAL NET sweeps) -------------------------
#  Mirrors nn_arch.KINDS. Not imported from there: nn_arch imports torch and
#  the recorder must not (section 13).
NN_KINDS = {
    "gru":     "plain GRU -- RECOMMENDED at 0.7 s (nn_ml_comb_test "
               "2026-10-22: 72.4 % window / 78.9 % vote, person held out)",
    "cnn_gru": "conv front end + GRU",
    "tcn":     "dilated depthwise convs + attention pooling (smallest)",
    "lstm":    "plain LSTM (historical baseline)",
    "cnn":     "plain 1D CNN (historical baseline)",
}
# The default IS the recommendation: GRU at 0.7 s, 150 epochs. Chosen by
# nn_ml_comb_test 20261022-003918 over gru/cnn_gru/tcn x 0.5-1.5 s windows.
NN_SPEC_DEFAULTS = {
    "kinds": ["gru"],
    "windows": [0.70],
    "overlap": 0.75,
    "epochs": 150,
    "hidden": 32,
    "dropout": 0.25,
    "learning_rate": 3e-3,
    "weight_decay": 1e-3,
    "batch_size": 64,
    "label_smoothing": 0.05,
}


def nn_spec_path(model_path: str = None) -> str:
    """`nn_spec.json` beside the checkpoint."""
    path = model_path or cfg.NN_MODEL_PATH
    if not os.path.isabs(path):
        path = os.path.join(cfg.PROJECT_DIR, path)
    return os.path.join(os.path.dirname(path), "nn_spec.json")


def validate_nn_spec(spec: dict):
    """
    Clamp and complete a net setup. Never raises; returns (spec, problems).
    Same stance as validate_model_spec: a hand-edited file cannot crash a
    sweep that is twenty minutes in.
    """
    problems: List[str] = []
    src = dict(spec) if isinstance(spec, dict) else {}
    out = dict(NN_SPEC_DEFAULTS)

    kinds = [k for k in (src.get("kinds") or []) if k in NN_KINDS]
    if src.get("kinds") is not None and len(kinds) != len(src["kinds"]):
        problems.append("unknown architecture(s) dropped")
    out["kinds"] = list(dict.fromkeys(kinds)) or list(NN_SPEC_DEFAULTS["kinds"])
    if src.get("kinds") is not None and not kinds:
        problems.append("no architecture selected -- all switched back on")

    wins = []
    for w in (src.get("windows") or []):
        try:
            w = round(float(w), 3)
        except (TypeError, ValueError):
            continue
        if 0.1 <= w <= 2.0:
            wins.append(w)
    out["windows"] = sorted(set(wins)) or list(NN_SPEC_DEFAULTS["windows"])
    if src.get("windows") is not None and not wins:
        problems.append("no usable window length (0.1-2.0 s) -- defaults used")

    ranges = {"overlap": (0.0, 0.9, float), "epochs": (5, 500, int),
              "hidden": (8, 256, int), "dropout": (0.0, 0.8, float),
              "learning_rate": (1e-5, 1e-1, float),
              "weight_decay": (0.0, 1e-1, float),
              "batch_size": (8, 512, int),
              "label_smoothing": (0.0, 0.5, float)}
    for key, (lo, hi, cast) in ranges.items():
        if key not in src:
            continue
        try:
            value = cast(src[key])
        except (TypeError, ValueError):
            problems.append(f"{key}: not a number -- default kept")
            continue
        clamped = min(hi, max(lo, value))
        if clamped != value:
            problems.append(f"{key}: {value} clamped to {clamped}")
        out[key] = clamped
    return out, problems


def load_nn_spec(path: str = None):
    path = path or nn_spec_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return validate_nn_spec(json.load(handle))
    except (OSError, ValueError):
        return validate_nn_spec({})


def save_nn_spec(spec: dict, path: str = None) -> str:
    path = path or nn_spec_path()
    clean, _ = validate_nn_spec(spec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2)
    return path


def nn_spec_args(spec: dict) -> List[str]:
    """The train_nn.py flags that express a (validated) setup."""
    s, _ = validate_nn_spec(spec)
    args: List[str] = []
    for k in s["kinds"]:
        args += ["--kind", k]
    for w in s["windows"]:
        args += ["--window", f"{w:g}"]
    args += ["--overlap", f"{s['overlap']:g}", "--epochs", str(s["epochs"]),
             "--hidden", str(s["hidden"]), "--dropout", f"{s['dropout']:g}",
             "--lr", f"{s['learning_rate']:g}",
             "--weight-decay", f"{s['weight_decay']:g}",
             "--batch-size", str(s["batch_size"]),
             "--label-smoothing", f"{s['label_smoothing']:g}"]
    return args


def describe_nn_spec(spec: dict) -> str:
    s, _ = validate_nn_spec(spec)
    return (f"{len(s['kinds'])} architecture(s) x {len(s['windows'])} "
            f"window(s), {s['epochs']} epochs, hidden {s['hidden']}, "
            f"dropout {s['dropout']:g}")


def nn_stats_path(model_path: str = None) -> str:
    """
    The plain-JSON sidecar beside a checkpoint: everything but the weights.

    A `.pt` is a torch pickle, so the application -- which must not import
    torch -- cannot read its own net's metrics out of it. `train_nn.py`
    therefore writes this next to the checkpoint, and NN STATS reads it. That
    is what lets the dialog work with the worker stopped, with no GPU, and on
    a machine where torch was never installed.
    """
    path = model_path or cfg.NN_MODEL_PATH
    return os.path.splitext(path)[0] + ".json"


def load_nn_stats(model_path: str = None) -> Optional[dict]:
    """The sidecar, or None. Never raises: a missing net is a normal state."""
    path = nn_stats_path(model_path)
    if not os.path.isabs(path):
        path = os.path.join(cfg.PROJECT_DIR, path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            found = json.load(fh)
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def nn_encode_request(request_id: int, window: np.ndarray) -> str:
    """One request as a single JSON line. `window` is (win, 8) float."""
    block = np.asarray(window, dtype=np.float32)
    return json.dumps({"id": int(request_id),
                       "window": [[round(float(v), 4) for v in row]
                                  for row in block]}) + "\n"


def nn_parse_line(line: str):
    """
    One line of the worker's stdout -> `(kind, payload)`.

    `kind` is "meta", "out", "err", "bye" or "text"; "text" carries anything
    the worker or its dependencies printed that is not protocol, which is
    kept rather than discarded because it is the only clue when torch itself
    is what failed. Never raises -- a malformed protocol line comes back as
    "err" with the raw text, because a parser that throws inside a reader
    thread kills the reader.
    """
    text = (line or "").rstrip("\r\n")
    for prefix, kind in ((NN_META_PREFIX, "meta"), (NN_OUT_PREFIX, "out"),
                         (NN_ERR_PREFIX, "err")):
        if text.startswith(prefix):
            try:
                found = json.loads(text[len(prefix):])
            except (ValueError, TypeError):
                return "err", {"error": f"unparseable {kind} line: "
                                        f"{text[:160]}"}
            if not isinstance(found, dict):
                return "err", {"error": f"unexpected {kind} payload"}
            return kind, found
    if text.strip() == NN_BYE_PREFIX:
        return "bye", {}
    return "text", {"text": text}


@dataclass
class NeuralPrediction:
    """
    One neural-net output.

    A SEPARATE type from `Prediction` on purpose. The two carry nearly the
    same fields, and that is exactly why they must not be the same class:
    with one type a net answer could be appended to `LivePredictor.history`,
    handed to the decision panel or drawn in the deployed classifier's
    probability bars by a single misplaced line, and nothing would complain.
    The type system is the cheapest place to enforce section 8L's separation
    rule.

    Nothing here is invented: every probability came out of a real
    `softmax(model(window))` in the worker process.
    """
    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    probabilities: List[float]        # smoothed, per class
    raw_probabilities: List[float]    # this window alone
    labels: List[str]
    n_samples: int                    # samples the window was built from
    smoothed_over: int
    latency_ms: float                 # worker round trip, measured
    infer_ms: float                   # the forward pass itself, measured
    device: str
    unix_time: float

    @property
    def percent(self) -> str:
        if math.isnan(self.confidence):
            return "--"
        return f"{self.confidence * 100:.0f} %"

    @property
    def certain(self) -> bool:
        if math.isnan(self.confidence) or cfg.PREDICTION_MIN_CONFIDENCE <= 0:
            return True
        return self.confidence >= cfg.PREDICTION_MIN_CONFIDENCE


class NeuralEngine:
    """
    Drives `nn_serve.py` and turns windows into `NeuralPrediction`s.

    States, and exactly one is true at a time:

        "stopped"   nothing running; `status` says why if it was a failure
        "starting"  the subprocess is up, loading torch and the checkpoint
        "ready"     the worker has reported its metadata and is answering
        "error"     it failed; `status` carries the reason and the worker's
                    own last words

    Never guesses. Until the worker reports `NNMETA`, the device is unknown
    and is shown as unknown -- the same rule as the GPU/CPU badge (section
    8c): the honest answer to an unanswerable question is that it cannot be
    answered yet.
    """

    def __init__(self, model_path: str = None, script: str = None,
                 device: str = None, smoothing: int = None,
                 python_executable: str = None):
        self.model_path = model_path or cfg.NN_MODEL_PATH
        self.script = script or cfg.NN_WORKER_SCRIPT
        self.device = device or cfg.NN_DEVICE
        self.python = python_executable or sys.executable
        self.state = "stopped"
        self.status = "not started"
        self.meta: dict = {}
        self.labels: List[str] = []
        self.win_samples = 0
        self.model_fs = float(cfg.NN_MODEL_FS_FALLBACK)
        self.window_s = 0.0
        #  The wearer's calibration, resolved for the mode the CHECKPOINT was
        #  fitted with. A net trained on calibrated features and then fed
        #  device counts does not break -- it answers confidently on the wrong
        #  scale, which is worse. Same refusal as LivePredictor.load().
        self.calibration_mode = "off"
        self.calibration: Optional[dict] = None
        self.calibration_note = ""
        self.last: Optional[NeuralPrediction] = None
        self.last_window: Optional[np.ndarray] = None

        self._process = None
        self._reader = None
        self._lock = threading.Lock()
        self._inbox = deque(maxlen=64)          # parsed lines from the worker
        self._ready = deque(maxlen=16)          # finished NeuralPredictions
        self._noise = deque(maxlen=40)          # non-protocol worker output
        self._pending: Dict[int, float] = {}    # request id -> send time
        self._next_id = 1
        self._started_at = 0.0
        self._history = deque(maxlen=max(1, smoothing or cfg.NN_SMOOTHING))
        self.counters = {"sent": 0, "answered": 0, "dropped": 0,
                         "errors": 0, "stale": 0}
        self.latency_ms = float("nan")
        self.infer_ms = float("nan")

    # -- paths ---------------------------------------------------------------

    @property
    def full_model_path(self) -> str:
        return (self.model_path if os.path.isabs(self.model_path)
                else os.path.join(cfg.PROJECT_DIR, self.model_path))

    @property
    def full_script_path(self) -> str:
        return (self.script if os.path.isabs(self.script)
                else os.path.join(cfg.PROJECT_DIR, self.script))

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def running(self) -> bool:
        return self.state in ("starting", "ready")

    def trained(self) -> bool:
        """Is there a checkpoint at all? Answerable without the worker."""
        return os.path.exists(self.full_model_path)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """
        Spawn the worker. Returns False and explains itself on any refusal.

        Deliberately cheap to call: every precondition is checked here rather
        than left to fail inside the subprocess, because a message in the
        panel is more use than an exit code.
        """
        if self.running:
            return True
        if not cfg.ENABLE_NEURAL_NET:
            self.state, self.status = "stopped", \
                "disabled in the configuration (ENABLE_NEURAL_NET = False)"
            return False
        if not self.trained():
            self.state = "stopped"
            self.status = (f"no checkpoint at {self.model_path} -- train one "
                           f"with:  python training/train_nn.py")
            return False
        if not os.path.exists(self.full_script_path):
            self.state = "stopped"
            self.status = f"worker script missing: {self.script}"
            return False
        if not self._load_calibration():
            self.state = "stopped"
            return False

        args = [self.python, "-u", self.full_script_path,
                "--model", self.full_model_path, "--device", self.device]
        try:
            #  stderr merged in: the interesting failures (a CPU-only torch
            #  build, a missing DLL, a CUDA driver mismatch) arrive there, and
            #  the protocol is prefix-tagged so mixing the two is safe.
            self._process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, cwd=cfg.PROJECT_DIR, text=True,
                encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            self._process = None
            self.state, self.status = "error", f"could not start the worker: {exc}"
            return False

        self._inbox.clear()
        self._ready.clear()
        self._noise.clear()
        self._pending.clear()
        self._history.clear()
        self.counters = {k: 0 for k in self.counters}
        self.latency_ms = self.infer_ms = float("nan")
        self._started_at = time.perf_counter()
        self.state = "starting"
        self.status = "starting the worker (loading torch)..."
        self._reader = threading.Thread(target=self._read_loop,
                                        name="nn-reader", daemon=True)
        self._reader.start()
        return True

    def stop(self) -> None:
        """Ask the worker to quit, then make sure it did."""
        process, self._process = self._process, None
        self._reader = None
        if process is not None:
            try:
                if process.stdin and not process.stdin.closed:
                    process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    process.stdin.flush()
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=2.0)
            except Exception:
                try:
                    process.kill()
                except OSError:
                    pass
        if self.state != "error":
            self.state, self.status = "stopped", "stopped"
        self._pending.clear()
        self._history.clear()
        self.last = None

    def _load_calibration(self, mode: str = None) -> bool:
        """
        Resolve the current wearer's calibration for the net's own mode.

        With no `mode` the mode comes from the SIDECAR, so the refusal can
        happen before a subprocess is spawned and before torch is imported --
        there is no point attaching a GPU context to discover the wearer has
        not calibrated.

        `mode` is passed explicitly when the WORKER has reported what the
        checkpoint really says. It must override, not merely be compared:
        re-reading the sidecar here would put the sidecar's answer straight
        back and the disagreement would resolve the wrong way round. The
        weights are the fact; the sidecar is a convenience that can be stale
        or missing.
        """
        self.calibration = None
        self.calibration_note = ""
        if mode is None:
            sidecar = load_nn_stats(self.model_path) or {}
            block = sidecar.get("calibration")
            block = block if isinstance(block, dict) else {}
            mode = block.get("mode", "off") or "off"
        self.calibration_mode = mode
        if mode == "off":
            return True

        record = load_calibration(cfg.DATA_FOLDER, cfg.PERSON_ID)
        if record is None:
            self.status = (
                f"this net was trained on {mode!r} per-wearer calibration, "
                f"and {cfg.PERSON_ID} has none. Press CALIBRATE, or retrain the "
                f"net with:  python training/train_nn.py --calibration off")
            return False
        resolved = resolve_calibration(record, mode)
        if resolved is None:
            _ok, problems = validate_calibration(record)
            self.status = (
                f"{cfg.PERSON_ID}'s calibration cannot be applied in {mode!r} "
                f"mode: {problems[0] if problems else 'invalid record'}")
            return False
        self.calibration = resolved
        self.calibration_note = (
            f"{mode} calibration, {cfg.PERSON_ID} measured "
            f"{(record.get('created_iso') or '')[:16].replace('T', ' ')}")
        return True

    # -- the reader thread ---------------------------------------------------

    def _read_loop(self) -> None:
        """
        Drain the worker's stdout until it closes.

        Parsing happens here and the results go into a deque; interpreting
        them happens in `poll()` on the GUI's thread. That split is what
        keeps the shared state to two bounded deques behind one lock instead
        of a protocol spread across two threads.
        """
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                kind, payload = nn_parse_line(line)
                with self._lock:
                    if kind == "text":
                        if payload.get("text", "").strip():
                            self._noise.append(payload["text"])
                    else:
                        self._inbox.append((kind, payload))
                if process is not self._process:
                    return                     # stopped under us
        except (OSError, ValueError):
            pass
        with self._lock:
            self._inbox.append(("closed", {}))

    # -- sending -------------------------------------------------------------

    def submit(self, emg: np.ndarray, t: np.ndarray) -> bool:
        """
        Resample one live window onto the net's grid and send it.

        Returns False when nothing was sent, which is a normal outcome:
        not ready yet, too few samples, or the previous request has not come
        back. Dropping a window is the right answer for a diagnostic -- the
        alternative is a queue that answers questions about the past.
        """
        if not self.ready or self.win_samples <= 0:
            return False
        emg = np.asarray(emg)
        t = np.asarray(t, dtype=np.float64)
        if emg.ndim != 2 or emg.shape[0] < 4 or t.size != emg.shape[0]:
            return False
        if len(self._pending) >= max(1, cfg.NN_MAX_IN_FLIGHT):
            self.counters["dropped"] += 1
            return False

        block = resample_uniform(emg, t, self.win_samples)
        block = nn_input_block(block, self.calibration)
        self.last_window = block

        process = self._process
        if process is None or process.stdin is None:
            return False
        request_id = self._next_id
        self._next_id += 1
        try:
            process.stdin.write(nn_encode_request(request_id, block))
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.state = "error"
            self.status = f"the worker stopped accepting windows: {exc}"
            return False
        self._pending[request_id] = time.perf_counter()
        self.counters["sent"] += 1
        return True

    # -- receiving -----------------------------------------------------------

    def poll(self) -> List[NeuralPrediction]:
        """
        Called from the GUI timer. Drains the inbox and returns new answers.

        Also enforces the start timeout: a worker that never reports its
        metadata is an error with a reason, not a panel that says "starting"
        forever.
        """
        with self._lock:
            items = list(self._inbox)
            self._inbox.clear()
            noise = list(self._noise)

        produced = []
        for kind, payload in items:
            if kind == "meta":
                self._on_meta(payload)
            elif kind == "out":
                made = self._on_out(payload)
                if made is not None:
                    produced.append(made)
            elif kind == "err":
                self.counters["errors"] += 1
                reason = str(payload.get("error", "unknown error"))
                if payload.get("fatal") or not self.ready:
                    self.state, self.status = "error", reason
                else:
                    self.status = f"last window failed: {reason}"
                self._pending.pop(payload.get("id"), None)
            elif kind in ("bye", "closed"):
                if self.state == "starting":
                    tail = " | ".join(noise[-3:])
                    self.state = "error"
                    self.status = ("the worker exited before it was ready"
                                   + (f": {tail}" if tail else ""))
                elif self.state == "ready":
                    self.state, self.status = "stopped", "the worker exited"
                #  Reap it rather than just forgetting the handle. The worker
                #  has already said goodbye or closed its pipe, so this
                #  returns at once; dropping the reference instead would
                #  leave the child unwaited.
                gone, self._process = self._process, None
                if gone is not None:
                    try:
                        gone.wait(timeout=0.5)
                    except Exception:
                        pass
                self._pending.clear()

        #  Give up on a request the worker never answered. One lost reply
        #  would otherwise hold the single in-flight slot for good, and the
        #  panel would sit on its last answer looking perfectly healthy --
        #  the quietest possible failure.
        if self._pending:
            now = time.perf_counter()
            expired = [key for key, sent in self._pending.items()
                       if now - sent > cfg.NN_RESPONSE_TIMEOUT_S]
            for key in expired:
                self._pending.pop(key, None)
                self.counters["stale"] += 1
            if expired and self.ready:
                self.status = (f"{len(expired)} window(s) went unanswered for "
                               f"over {cfg.NN_RESPONSE_TIMEOUT_S:.0f} s")

        if self.state == "starting" and \
                (time.perf_counter() - self._started_at) > cfg.NN_START_TIMEOUT_S:
            tail = " | ".join(noise[-3:])
            self.state = "error"
            self.status = (f"the worker did not become ready within "
                           f"{cfg.NN_START_TIMEOUT_S:.0f} s"
                           + (f": {tail}" if tail else ""))
            self.stop_quietly()
        return produced

    def stop_quietly(self) -> None:
        """Kill the subprocess without overwriting an error already set."""
        state, status = self.state, self.status
        self.stop()
        self.state, self.status = state, status

    def _on_meta(self, payload: dict) -> None:
        if not payload.get("ok", False):
            self.state = "error"
            self.status = str(payload.get("error", "the worker refused the "
                                                   "checkpoint"))
            return
        self.meta = dict(payload)
        self.labels = [str(x) for x in payload.get("labels", [])]
        self.win_samples = int(payload.get("win_samples", 0) or 0)
        self.model_fs = float(payload.get("model_fs", cfg.NN_MODEL_FS_FALLBACK) or
                              cfg.NN_MODEL_FS_FALLBACK)
        self.window_s = float(payload.get("window_s", 0.0) or 0.0)

        #  The worker reports the mode the CHECKPOINT carries. If the sidecar
        #  said something else, the checkpoint wins and the engine refuses
        #  rather than predicting on a scale it cannot reproduce -- the
        #  sidecar is a convenience, the weights are the fact.
        worker_mode = str((payload.get("calibration") or {}).get("mode",
                                                                 "off"))
        if worker_mode != self.calibration_mode:
            if not self._load_calibration(worker_mode):
                self.state = "error"
                self.stop_quietly()
                return
        if not self.labels or self.win_samples <= 0:
            self.state = "error"
            self.status = "the worker reported no labels or no window size"
            return
        self.state = "ready"
        device = (payload.get("device") or {}).get("device", "?")
        self.status = f"ready on {device}"

    def _on_out(self, payload: dict) -> Optional[NeuralPrediction]:
        request_id = payload.get("id")
        sent_at = self._pending.pop(request_id, None)
        probabilities = payload.get("probs")
        if not isinstance(probabilities, list) or not probabilities:
            self.counters["errors"] += 1
            return None
        if len(probabilities) != len(self.labels):
            #  A width mismatch means the worker and this engine disagree
            #  about the label set, which can only end in a mislabelled
            #  answer. Refuse the whole engine rather than this one window.
            self.state = "error"
            self.status = (f"the worker returned {len(probabilities)} "
                           f"probabilities for {len(self.labels)} labels")
            return None

        raw = np.asarray(probabilities, dtype=float)
        index = int(np.argmax(raw))
        self._history.append(raw)
        smoothed = np.mean(np.vstack(self._history), axis=0)
        best = int(np.argmax(smoothed))

        now = time.perf_counter()
        self.latency_ms = ((now - sent_at) * 1000.0 if sent_at is not None
                           else float("nan"))
        try:
            self.infer_ms = float(payload.get("ms", float("nan")))
        except (TypeError, ValueError):
            self.infer_ms = float("nan")
        self.counters["answered"] += 1

        prediction = NeuralPrediction(
            label=self._name(best),
            confidence=float(smoothed[best]),
            raw_label=self._name(index),
            raw_confidence=float(raw[index]),
            probabilities=[float(p) for p in smoothed],
            raw_probabilities=[float(p) for p in raw],
            labels=list(self.labels),
            n_samples=int(self.win_samples),
            smoothed_over=len(self._history),
            latency_ms=self.latency_ms,
            infer_ms=self.infer_ms,
            device=str((self.meta.get("device") or {}).get("device", "?")),
            unix_time=time.time(),
        )
        self.last = prediction
        return prediction

    def _name(self, index: int) -> str:
        return (self.labels[index] if 0 <= index < len(self.labels)
                else str(index))

    def reset(self) -> None:
        """Forget the smoothing history, keep the worker."""
        self._history.clear()
        self.last = None

    # -- reporting -----------------------------------------------------------

    def device_badge(self) -> tuple:
        """
        `(headline, detail, severity)` for the net's own device chip.

        Reports what the WORKER is really using, which is the only honest
        source: the main process cannot import torch to find out, and the
        hardware probe in section 8c answers a different question ("is a GPU
        installed") from this one ("is the net running on it").
        """
        if not self.running:
            return ("NN DEVICE --", "the net is not running", "dim")
        info = self.meta.get("device") or {}
        if not info:
            return ("NN DEVICE ...", "waiting for the worker", "dim")
        if str(info.get("device", "cpu")).startswith("cuda"):
            return (f"NN ON GPU -- {info.get('gpu', 'CUDA device')}",
                    f"CUDA {info.get('cuda_runtime')} | torch "
                    f"{info.get('torch')} | {info.get('total_memory_mb')} MB",
                    "good")
        reason = info.get("reason") or ""
        return ("NN ON CPU",
                f"torch {info.get('torch')}"
                + (f" | {reason}" if reason else ""), "warn")

    def describe(self) -> str:
        """One line: what this net is."""
        stats = self.meta.get("stats") or load_nn_stats(self.model_path) \
            or {}
        kind = self.meta.get("kind") or stats.get("kind") or "net"
        params = self.meta.get("n_parameters") or stats.get("n_parameters")
        acc = stats.get("window_accuracy")
        cal = (f"  |  {self.calibration_mode} calibration"
               if self.calibration_mode != "off" else "")
        return (f"NN-{kind}  |  {self.window_s * 1000:.0f} ms @ "
                f"{self.model_fs:.0f} Hz  |  "
                + (f"{params} parameters" if params else "")
                + cal
                + (f"  |  {acc * 100:.0f} % per window, "
                   f"{stats.get('grouping', '?')}-grouped" if acc else ""))

    def stats_summary(self) -> dict:
        """Counters and measured latency, for the panel."""
        return {"state": self.state, "status": self.status,
                "latency_ms": self.latency_ms, "infer_ms": self.infer_ms,
                "in_flight": len(self._pending), **self.counters}

    def worker_output(self, n: int = 12) -> List[str]:
        """The worker's last non-protocol lines -- the diagnostics."""
        with self._lock:
            return list(self._noise)[-n:]

    def data_status(self, fingerprint: dict = None) -> dict:
        """
        Is a net trained, on what, and is that still what is on disk?

        Delegates to `model_data_status()` so the net answers the question
        the same way the sklearn model does -- the sidecar carries the same
        `data` block both trainers write.
        """
        sidecar = load_nn_stats(self.model_path)
        if sidecar is None:
            return model_data_status(None)
        return model_data_status(sidecar, fingerprint)

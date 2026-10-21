"""
================================================================================
 NEURAL NETWORK INFERENCE WORKER  --  the only process that imports torch
================================================================================

 Loads `models/gesture_nn.pt`, then answers one window at a time on stdin and
 stdout. `emg_dataset_tool.NeuralEngine` drives it.

 WHY A SUBPROCESS, and not a thread in the application
 -----------------------------------------------------
 Because the recorder must never import torch. That is a standing invariant
 (CLAUDE.md section 13), and it has two independent reasons:

   * the import costs about seven seconds, which would be seven seconds of a
     frozen window at startup;
   * importing torch and touching CUDA attaches a GPU context to the process,
     and this process's one job is to never drop an EMG sample. A GPU driver
     stall is not something to find out about while recording a subject.

 The GPU/CPU badge is probed the same way for the same reason, and
 `tests/test_no_torch.py` is what keeps it honest: with `import torch` blocked
 outright, the application still starts, records and predicts.

 THE PROTOCOL -- one JSON object per line, in both directions
 -----------------------------------------------------------
 Out (stdout), prefix-tagged, because torch and the CUDA driver also print
 here and a tagged line cannot be confused with their chatter:

     NNMETA {"ok": true, "labels": [...], "win_samples": 30, ...}
     NNMETA {"ok": false, "error": "..."}           fatal, then it exits
     NNOUT  {"id": 7, "probs": [...], "ms": 1.4}
     NNERR  {"id": 7, "error": "...", "fatal": false}
     NNBYE

 In (stdin):

     {"id": 7, "window": [[c1..c8], ... win rows ...]}
     {"cmd": "ping"}
     {"cmd": "quit"}

 EOF on stdin is also "quit": if the application dies, the worker goes with
 it rather than sitting on a GPU context forever.

 WHAT IT DOES NOT DO
 -------------------
 It does not resample, and it does not calibrate. Both happen on the
 application's side, in `T.nn_input_block()` and `resample_uniform()`, because
 those are the functions the TRAINER used and sharing them is what stops
 training and inference drifting apart. What this worker owns is the part that
 genuinely belongs to the model: the `log1p` + standardise scaler fitted on
 the training fold, which travels inside the checkpoint, and the forward pass.

 It also never smooths, votes or thresholds. It returns the softmax of one
 window, and the engine decides what to do with it -- so the number in the
 panel is traceable to exactly one forward pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import nn_arch


def emit(prefix: str, payload: dict) -> None:
    """One protocol line. Flushed, because the reader is blocking on it."""
    try:
        sys.stdout.write(prefix + json.dumps(payload, default=str) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def resolve_device(requested: str):
    """
    `(device, info)`.

    "cuda" means REQUIRE cuda: the worker reports the failure and exits
    rather than quietly running on the CPU, because a silent fallback looks
    exactly like success and the panel would show a green "running" next to a
    model that is twenty times slower. "auto" is the app's default and does
    fall back -- but it says so in `info["reason"]`, which the panel's device
    chip prints.
    """
    import torch

    info = {"torch": torch.__version__, "reason": ""}
    if requested == "cpu":
        info["device"] = "cpu"
        info["reason"] = "forced by --device cpu"
        return torch.device("cpu"), info

    if not torch.cuda.is_available():
        build = torch.__version__
        reason = ("CPU-only torch build" if "+cu" not in build
                  else "no usable GPU or driver")
        if requested == "cuda":
            raise RuntimeError(
                f"CUDA was required but torch.cuda.is_available() is False "
                f"({reason}, torch {build}). Install the CUDA build: "
                f"python -m pip install torch==2.9.1+cu128 --index-url "
                f"https://download.pytorch.org/whl/cu128 -- or run the net "
                f"on the CPU.")
        info["device"] = "cpu"
        info["reason"] = reason
        return torch.device("cpu"), info

    props = torch.cuda.get_device_properties(0)
    info.update(device="cuda:0", gpu=torch.cuda.get_device_name(0),
                capability=f"{props.major}.{props.minor}",
                total_memory_mb=int(props.total_memory / 1024 / 1024),
                cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version())
    #  Same determinism choice the trainer makes: autotuning picks different
    #  kernels per run, which changes the accumulation order and therefore
    #  the answer. A diagnostic that reports a different probability for the
    #  same window is not a diagnostic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0"), info


class Server:
    """Checkpoint in, probabilities out."""

    def __init__(self, model_path: str, device_request: str):
        self.model_path = model_path
        self.device_request = device_request
        self.model = None
        self.device = None
        self.device_info = {}
        self.labels = []
        self.win_samples = 0
        self.n_channels = 8
        self.mean = None
        self.std = None
        self.payload_meta = {}

    def load(self) -> dict:
        """
        Build the net from the checkpoint and report what it is.

        Everything that could go wrong is turned into one `NNMETA` with
        `ok: false` and a sentence, because the engine shows that sentence in
        the panel. A traceback on stderr would be correct and useless.
        """
        import torch

        t0 = time.perf_counter()
        self.device, self.device_info = resolve_device(self.device_request)

        #  weights_only=False: the checkpoint carries the scaler's numpy
        #  arrays and the provenance blocks alongside the tensors, and it is
        #  a file this project's own trainer wrote next to the dataset. If
        #  that ever stops being true, the file is the problem, not the flag.
        payload = torch.load(self.model_path, map_location="cpu",
                             weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise RuntimeError(
                f"{self.model_path} is not a checkpoint this worker "
                f"understands (expected a dict with a state_dict).")
        if payload.get("format") not in ("emg_nn_v1",):
            raise RuntimeError(
                f"unsupported checkpoint format "
                f"{payload.get('format')!r}; retrain with train_nn.py")

        self.labels = [str(x) for x in payload.get("labels", [])]
        self.win_samples = int(payload.get("win_samples", 0) or 0)
        self.n_channels = int(payload.get("n_channels", 8) or 8)
        if not self.labels or self.win_samples <= 0:
            raise RuntimeError("the checkpoint declares no labels or no "
                               "window length")

        scaler = payload.get("scaler") or {}
        if scaler.get("kind") != "log1p_standardise":
            raise RuntimeError(
                f"unknown scaler {scaler.get('kind')!r} in the checkpoint. "
                f"The scaler is part of the model -- refusing rather than "
                f"guessing one.")
        #  Flattened to a per-channel vector, whatever shape the checkpoint
        #  stored. A (1, 1, 8) mean -- which is what `ndarray.mean(keepdims)`
        #  hands you, and what early checkpoints carry -- broadcasts a single
        #  (win, 8) window up to (1, win, 8), which then reaches conv1d as a
        #  4-D tensor and raises. The trainer only ever scales batched
        #  `(n, win, 8)`, so that shape error exists *only* on the inference
        #  path and no training run can find it. A (8,) vector is right for
        #  both.
        self.mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(scaler["std"], dtype=np.float32).reshape(-1)
        if self.mean.size != self.n_channels or \
                self.std.size != self.n_channels:
            raise RuntimeError(
                f"the checkpoint's scaler has {self.mean.size} values for "
                f"{self.n_channels} channels")

        model = nn_arch.build(str(payload.get("kind", "tcn")),
                              self.n_channels, len(self.labels),
                              self.win_samples,
                              hidden=int(payload.get("hidden",
                                                     nn_arch.HIDDEN)),
                              dropout=float(payload.get("dropout",
                                                        nn_arch.DROPOUT)),
                              seed=0)
        #  strict=True on purpose. A shape or name mismatch means nn_arch and
        #  the checkpoint disagree about the net, and a net loaded loosely
        #  answers confidently with some layers left at their initialisation.
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().to(self.device)
        self.model = model
        self.payload_meta = {k: v for k, v in payload.items()
                             if k not in ("state_dict", "scaler")}

        #  Warm up. The first forward pass builds kernels on demand -- 2.4 s
        #  was measured for the sklearn panel's HistGradientBoosting
        #  (CLAUDE.md section 8e) and CUDA does the same thing. Paying it here
        #  means no live window eats it and the first latency figure the panel
        #  shows is a real one.
        warm = np.zeros((1, self.win_samples, self.n_channels),
                        dtype=np.float32)
        with torch.no_grad():
            self.model(torch.from_numpy(warm).to(self.device))

        stats = self.payload_meta.get("stats") or {}
        return {
            "ok": True,
            "labels": self.labels,
            "win_samples": self.win_samples,
            "n_channels": self.n_channels,
            "window_s": float(payload.get("window_s", 0.0) or 0.0),
            "model_fs": float(payload.get("model_fs", 60.0) or 60.0),
            "kind": str(payload.get("kind", "?")),
            "architecture": str(payload.get("architecture", "")),
            "hidden": int(payload.get("hidden", nn_arch.HIDDEN)),
            "n_parameters": nn_arch.count_parameters(self.model),
            "calibration": payload.get("calibration") or {"mode": "off"},
            "created_iso": payload.get("created_iso", ""),
            "device": self.device_info,
            "stats": stats,
            "data": payload.get("data"),
            "load_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

    def scale(self, block: np.ndarray) -> np.ndarray:
        """The checkpoint's own scaler. Exactly `train_nn.fold_scaler`."""
        return ((np.log1p(np.abs(block)) - self.mean) / self.std
                ).astype(np.float32)

    def answer(self, window) -> dict:
        """One window -> probabilities. Shape is checked, never reshaped."""
        import torch

        block = np.asarray(window, dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(f"expected a 2-D window, got shape "
                             f"{block.shape}")
        if block.shape != (self.win_samples, self.n_channels):
            #  A window of the wrong length lands somewhere plausible if
            #  reshaped, and means nothing. The engine resamples onto this
            #  net's own grid before sending, so a mismatch here is a bug
            #  worth seeing rather than papering over.
            raise ValueError(
                f"expected a ({self.win_samples}, {self.n_channels}) window, "
                f"got {tuple(block.shape)}")

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(self.scale(block)[None, ...]).to(self.device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {"probs": [float(p) for p in probabilities],
                "ms": round((time.perf_counter() - t0) * 1000.0, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve live predictions from a trained EMG neural net")
    parser.add_argument("--model", required=True, help="checkpoint path")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args(argv)

    server = Server(args.model, args.device)
    try:
        emit("NNMETA ", server.load())
    except Exception as exc:                       # noqa: BLE001 -- see below
        #  Everything becomes one sentence on stdout. The engine shows it in
        #  the panel; a traceback nobody reads would be worse. The failure
        #  modes here are all environmental (no CUDA, a stale checkpoint, a
        #  missing DLL), so the TYPE matters less than the message.
        emit("NNMETA ", {"ok": False,
                         "error": f"{type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:                         # EOF ends the loop
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except (ValueError, TypeError):
            emit("NNERR ", {"id": None, "error": "unparseable request",
                            "fatal": False})
            continue
        if not isinstance(request, dict):
            emit("NNERR ", {"id": None, "error": "request was not an object",
                            "fatal": False})
            continue

        command = request.get("cmd")
        if command == "quit":
            break
        if command == "ping":
            emit("NNOUT ", {"id": request.get("id"), "probs": [], "ms": 0.0,
                            "pong": True})
            continue

        request_id = request.get("id")
        try:
            emit("NNOUT ", {"id": request_id,
                            **server.answer(request.get("window"))})
        except Exception as exc:                   # noqa: BLE001
            #  One bad window must not take the worker down: the engine can
            #  recover by sending the next one, and a worker that exits on a
            #  transient means pressing START again to get the GPU context
            #  back. `fatal: false` is what tells the engine that.
            emit("NNERR ", {"id": request_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "fatal": False})

    sys.stdout.write("NNBYE\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

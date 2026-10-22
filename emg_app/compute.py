"""
The GPU/CPU probe: runs torch in a subprocess, never here.
"""

import json
import subprocess
import sys


# ------------------------------------------------------------------------------
# COMPUTE DEVICE  (is a CUDA GPU actually going to be used?)
# ------------------------------------------------------------------------------
#
# Only torch can answer this, and the recorder deliberately does not import
# torch (CLAUDE.md section 13: no torch dependency in the application -- it
# would add seconds to startup and a GPU context to a process whose job is to
# not drop EMG samples). So the probe runs in a short-lived subprocess, the
# same way training does, and the answer is cached for the session.
#
# The split: this module owns the probe snippet and the parser, so both are
# testable headless; the GUI owns running it without blocking the event loop.

COMPUTE_PROBE_TIMEOUT_S = 30.0

#  Printed as one JSON line on stdout. Deliberately tiny and dependency-free:
#  anything that raises must still produce a parseable answer.
COMPUTE_PROBE_SNIPPET = (
    "import json\n"
    "out = {'probed': True, 'torch': None, 'cuda': False, 'gpu': None,\n"
    "       'cuda_runtime': None, 'capability': None,\n"
    "       'total_memory_mb': None, 'reason': ''}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch'] = torch.__version__\n"
    "    if torch.cuda.is_available():\n"
    "        p = torch.cuda.get_device_properties(0)\n"
    "        out.update(cuda=True, gpu=torch.cuda.get_device_name(0),\n"
    "                   cuda_runtime=torch.version.cuda,\n"
    "                   capability='%d.%d' % (p.major, p.minor),\n"
    "                   total_memory_mb=int(p.total_memory / 1048576))\n"
    "    elif '+cu' not in torch.__version__:\n"
    "        out['reason'] = 'CPU-only torch build'\n"
    "    else:\n"
    "        out['reason'] = 'no usable GPU or driver'\n"
    "except ImportError:\n"
    "    out['reason'] = 'torch is not installed'\n"
    "except Exception as exc:\n"
    "    out['reason'] = '%s: %s' % (type(exc).__name__, exc)\n"
    "print('COMPUTE ' + json.dumps(out))\n"
)


def blank_compute_info(reason: str = "not checked yet",
                       failed: bool = False) -> dict:
    """
    The shape every consumer can rely on, with nothing known yet.

    `failed` separates "the probe has not run yet" from "the probe ran and
    could not tell us". Both mean the device is unknown, but only the first
    is worth waiting out, and neither may be shown as "CPU".
    """
    return {"probed": False, "failed": bool(failed), "torch": None,
            "cuda": False, "gpu": None, "cuda_runtime": None,
            "capability": None, "total_memory_mb": None, "reason": reason}


def parse_compute_probe(text: str) -> dict:
    """
    Pull the probe's answer out of its stdout.

    Never raises: a probe that printed nothing usable means "unknown", which
    is a different thing from "no GPU" and is reported as such.
    """
    for line in reversed(str(text or "").splitlines()):
        line = line.strip()
        if not line.startswith("COMPUTE "):
            continue
        try:
            found = json.loads(line[len("COMPUTE "):])
        except (ValueError, TypeError):
            break
        if not isinstance(found, dict):
            break
        info = blank_compute_info("")
        info.update({k: v for k, v in found.items() if k in info})
        info["probed"] = True
        return info
    return blank_compute_info("the device probe did not answer", failed=True)


def detect_compute_device(timeout_s: float = None) -> dict:
    """
    Blocking probe. Used by headless callers and the tests; the GUI runs the
    same snippet through QProcess instead so it never stalls the event loop.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", COMPUTE_PROBE_SNIPPET],
            capture_output=True, text=True,
            timeout=timeout_s or COMPUTE_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return blank_compute_info("the device probe timed out", failed=True)
    except OSError as exc:
        return blank_compute_info(f"could not start the device probe: {exc}",
                                  failed=True)
    return parse_compute_probe(done.stdout or "")


def compute_badge(info: dict) -> tuple:
    """
    `(headline, detail, severity)` for the GUI's compute badge.

    severity is "good" / "warn" / "dim", which the GUI maps to a colour.

    The headline says what the HARDWARE offers. The detail says what each
    trainer actually uses, because those are not the same thing: the
    scikit-learn sweep behind RETRAIN has no GPU backend and runs on the CPU
    whatever card is installed. A badge that said "GPU" next to a button that
    only ever uses the CPU would be a lie.
    """
    info = info or blank_compute_info()
    if not info.get("probed"):
        if info.get("failed"):
            return ("DEVICE UNKNOWN",
                    f"{info.get('reason') or 'the probe failed'}.\n"
                    #  train_nn.py, not train_torch.py: the latter is
                    #  retired (section 8b) and pointing a diagnostic
                    #  at a file nobody runs any more wastes the one
                    #  moment the operator is looking for an answer.
                    "Run `python training/train_nn.py --quick --no-save` in a "
                    "terminal to see what the GPU situation is.",
                    "warn")
        return ("checking compute device...", info.get("reason", ""), "dim")

    if info.get("cuda"):
        head = f"GPU READY  --  {info.get('gpu') or 'CUDA device'}"
        bits = [b for b in (
            f"CUDA {info['cuda_runtime']}" if info.get("cuda_runtime") else "",
            f"torch {info['torch']}" if info.get("torch") else "",
            f"{info['total_memory_mb']} MB" if info.get("total_memory_mb") else "",
            f"capability {info['capability']}" if info.get("capability") else "",
        ) if b]
        detail = (", ".join(bits) + "\n"
                  "RETRAIN (scikit-learn) runs on the CPU -- sklearn has no "
                  "GPU backend.\ntrain_torch.py uses this GPU.")
        return (head, detail, "good")

    reason = info.get("reason") or "no CUDA GPU found"
    head = "CPU ONLY"
    detail = f"{reason}.\nEvery trainer runs on the CPU."
    if info.get("torch"):
        detail = f"torch {info['torch']}, {detail}"
    else:
        detail += ("\nFor CUDA: python -m pip install torch==2.9.1+cu128 "
                   "--index-url https://download.pytorch.org/whl/cu128")
    return (head, detail, "warn")

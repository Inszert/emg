"""
Small helpers: paths, slugs, unique_path, time formatting.
"""

import os
import re
import time
from datetime import datetime
from typing import Tuple

from emg_app import config as cfg


# ==============================================================================
# SECTION 1 -- SMALL HELPERS
# ==============================================================================


def nominal_rate(mode: str = None) -> float:
    """Configured nominal sample rate for an EMG mode."""
    mode = mode or cfg.EMG_MODE
    return float(cfg.NOMINAL_SAMPLE_RATE.get(mode, 200))


def emg_range(mode: str = None) -> Tuple[float, float]:
    mode = mode or cfg.EMG_MODE
    return cfg.EMG_RANGE.get(mode, (-128, 127))


def mode_value(table: dict, mode: str = None, default=None):
    """Look a per-mode configuration value up with a sane fallback."""
    mode = mode or cfg.EMG_MODE
    if isinstance(table, dict):
        return table.get(mode, default if default is not None else next(iter(table.values())))
    return table


def ms_to_samples(ms: float, fs: float) -> int:
    """Convert milliseconds to a whole number of samples (at least 1)."""
    return max(1, int(round(ms * 1e-3 * fs)))


def safe_slug(text: str, fallback: str = "unnamed") -> str:
    """Make a string safe for use as a file/folder name."""
    text = (text or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def timestamp_slug(t: float = None) -> str:
    """Compact sortable timestamp: 20261006-114530-123"""
    dt = datetime.fromtimestamp(t if t is not None else time.time())
    return dt.strftime("%Y%m%d-%H%M%S-") + f"{dt.microsecond // 1000:03d}"


def iso_time(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def unique_path(path: str) -> str:
    """Never overwrite: append _2, _3, ... until the path is free."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"

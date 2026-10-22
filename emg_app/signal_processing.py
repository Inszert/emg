"""
Signal processing: moving RMS, envelope, band-pass, normalisation, activity
detection.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from emg_app import config as cfg
from emg_app.utils import emg_range, mode_value, ms_to_samples, nominal_rate


# ==============================================================================
# SECTION 2 -- SIGNAL PROCESSING
#
# Nothing in this section ever modifies its input array. Every function returns
# a new array, so the raw signal that came off the armband stays byte-identical
# from acquisition all the way to disk.
# ==============================================================================


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """
    Moving RMS along axis 0, same length as the input.

    Implemented with a cumulative sum so it stays O(n) even for long
    recordings, and centred so that peaks do not shift in time.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    n = x.shape[0]
    win = int(max(1, min(win, n)))

    sq = x * x
    # Reflect-pad so the first/last samples are not biased towards zero.
    pad_lo = win // 2
    pad_hi = win - 1 - pad_lo
    padded = np.pad(sq, ((pad_lo, pad_hi), (0, 0)), mode="reflect")
    csum = np.cumsum(padded, axis=0)
    csum = np.vstack([np.zeros((1, x.shape[1])), csum])
    out = np.sqrt((csum[win:win + n] - csum[0:n]) / float(win))

    return out[:, 0] if squeeze else out


def moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average along axis 0, same length as the input."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return moving_mean(x[:, None], win)[:, 0]

    n = x.shape[0]
    win = int(max(1, min(win, n)))
    pad_lo = win // 2
    pad_hi = win - 1 - pad_lo
    padded = np.pad(x, ((pad_lo, pad_hi), (0, 0)), mode="reflect")
    csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(padded, axis=0)])
    return (csum[win:win + n] - csum[0:n]) / float(win)


def _butter2_coeffs(fc: float, fs: float, kind: str):
    """
    One 2-pole Butterworth section as `(b, a)`, via the bilinear transform.

    Written out rather than taken from `scipy.signal`. The HAVE_SCIPY comment
    above records that `scipy.signal` costs ~7.3 s to import on a cold cache
    and that nothing here actually imports it; a filter that runs in the live
    prediction path is not the place to start, and a 2-pole Butterworth is
    four lines of algebra.

    `fc` is pre-warped with tan() so the digital cutoff lands exactly where it
    was asked for instead of where the s-plane prototype would have put it.
    """
    nyquist = fs / 2.0
    fc = float(min(max(fc, 1e-6), nyquist * 0.999))
    k = math.tan(math.pi * fc / fs)          # pre-warped cutoff
    k2 = k * k
    root2 = math.sqrt(2.0)
    norm = 1.0 / (1.0 + root2 * k + k2)

    if kind == "low":
        b = np.array([k2, 2.0 * k2, k2]) * norm
    elif kind == "high":
        b = np.array([1.0, -2.0, 1.0]) * norm
    else:                                               # pragma: no cover
        raise ValueError(f"kind must be 'low' or 'high', not {kind!r}")

    a = np.array([1.0,
                  2.0 * (k2 - 1.0) * norm,
                  (1.0 - root2 * k + k2) * norm])
    return b, a


def _lfilter(b, a, x: np.ndarray) -> np.ndarray:
    """Direct-form-II transposed IIR along axis 0. Columns are independent."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    z1 = np.zeros(x.shape[1])
    z2 = np.zeros(x.shape[1])
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        out[i] = yi
    return out


def _filtfilt(b, a, x: np.ndarray, pad: int) -> np.ndarray:
    """
    Zero-phase filtering: forward, then backward over the reversed signal.

    Phase matters here. A causal filter delays the envelope, and a delayed
    envelope moves a gesture's onset relative to the label window it was cut
    with -- so a one-directional filter would shift the very boundaries this
    dataset is built on. Running it both ways cancels the phase exactly, at
    the cost of doubling the effective order.

    Reflect-padded for the same reason `moving_rms` is: a zero-padded edge
    reads as a sudden drop to silence, and the filter rings on it.
    """
    n = x.shape[0]
    pad = int(max(0, min(pad, n - 1)))
    if pad:
        head = 2.0 * x[0] - x[pad:0:-1]
        tail = 2.0 * x[-1] - x[-2:-pad - 2:-1]
        work = np.vstack([head, x, tail])
    else:
        work = x
    work = _lfilter(b, a, work)
    work = _lfilter(b, a, work[::-1])[::-1]
    return work[pad:pad + n] if pad else work


def bandpass_filter(x: np.ndarray, fs: float, high_pass: float = None,
                    low_pass: float = None, order: int = None,
                    report: dict = None) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass. Returns a filtered COPY.

    `high_pass` / `low_pass` are cutoffs in Hz; None skips that half. Defaults
    come from `EMG_BANDPASS_HZ[EMG_MODE]`, whose comment explains why the
    "filtered" default is a low-pass only -- in short, the Myo has already
    band-passed that stream in hardware, and a high-pass on the envelope it
    returns would delete the activation level that carries most of the signal.

    **A cutoff at or above Nyquist is SKIPPED, not clamped**, and the reason is
    recorded in `report` when one is passed. Clamping 450 Hz down to 24.9 Hz on
    a 50 Hz stream would quietly apply a completely different filter from the
    one that was asked for and still report it as 20-450 Hz. At these sample
    rates that is not a corner case -- it is what happens the first time
    someone copies a textbook band from a 1 kHz setup.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return bandpass_filter(x[:, None], fs, high_pass, low_pass, order,
                               report)[:, 0]
    if report is None:
        report = {}
    notes = report.setdefault("notes", [])
    report.setdefault("applied", [])

    defaults = cfg.EMG_BANDPASS_HZ.get(cfg.EMG_MODE, (None, None))
    if high_pass is None and low_pass is None:
        high_pass, low_pass = defaults
    order = cfg.EMG_BANDPASS_ORDER if order is None else int(order)
    fs = float(fs or nominal_rate())
    nyquist = fs / 2.0

    if x.shape[0] < 9 or fs <= 0:
        notes.append(f"too short to filter ({x.shape[0]} samples)")
        return x.copy()

    #  3x the slowest cutoff's period, so the reflected edge covers the
    #  filter's own settling time.
    slowest = min([c for c in (high_pass, low_pass) if c] or [1.0])
    pad = int(min(max(9, round(3.0 * fs / slowest)), x.shape[0] - 1))

    out = x
    for cutoff, kind in ((high_pass, "high"), (low_pass, "low")):
        if not cutoff:
            continue
        if cutoff >= nyquist:
            notes.append(
                f"{kind}-pass {cutoff:.1f} Hz SKIPPED: at or above the "
                f"{nyquist:.1f} Hz Nyquist of this {fs:.1f} Hz stream")
            continue
        sections = max(1, order // 2)
        b, a = _butter2_coeffs(cutoff, fs, kind)
        for _ in range(sections):
            out = _filtfilt(b, a, out, pad)
        report["applied"].append({"kind": kind, "hz": float(cutoff),
                                  "order": int(sections * 2),
                                  "zero_phase": True})
    return np.ascontiguousarray(out)


def baseline_level(x: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Per-channel DC/offset estimate from the first n_samples.

    The median is used instead of the mean because it is not pulled around by
    an early muscle burst -- important when the subject starts too soon.
    """
    x = np.asarray(x, dtype=np.float64)
    n = min(max(1, int(n_samples)), x.shape[0])
    return np.median(x[:n], axis=0)


def envelope(x: np.ndarray, fs: float, win_ms: float = None,
             baseline_ms: float = None) -> np.ndarray:
    """
    Linear-envelope representation: moving RMS of the baseline-removed signal.

    This is a *derived* view used for display, activity detection and feature
    extraction. It is never written over the raw channels.
    """
    win_ms = cfg.ENVELOPE_MS if win_ms is None else win_ms
    baseline_ms = cfg.TRIM_BASELINE_MS if baseline_ms is None else baseline_ms

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    base = baseline_level(x, ms_to_samples(baseline_ms, fs))
    return moving_rms(x - base, ms_to_samples(win_ms, fs))


def normalize_per_channel(x: np.ndarray, mode: str = "mad",
                          ref: np.ndarray = None) -> Tuple[np.ndarray, dict]:
    """
    Produce a normalised representation WITHOUT clipping or crushing peaks.

    mode:
      "mad"    -> centre on the median, scale by the median absolute deviation.
                  Robust: a single explosive burst does not shrink the whole
                  recording the way max-normalisation would.
      "range"  -> scale by the device's full dynamic range (fully reversible).
      "zscore" -> classic mean/std.

    Returns (normalised_copy, scaling_parameters). The scaling parameters are
    stored with the data so the operation can always be undone.
    """
    x = np.asarray(x, dtype=np.float64)
    src = x if ref is None else np.asarray(ref, dtype=np.float64)

    if mode == "range":
        lo, hi = emg_range()
        centre = np.full(x.shape[1], (hi + lo) / 2.0)
        scale = np.full(x.shape[1], max(1e-9, (hi - lo) / 2.0))
    elif mode == "zscore":
        centre = src.mean(axis=0)
        scale = np.maximum(src.std(axis=0), 1e-9)
    else:  # "mad"
        centre = np.median(src, axis=0)
        mad = np.median(np.abs(src - centre), axis=0) * 1.4826
        scale = np.maximum(mad, 1e-9)

    return (x - centre) / scale, {
        "mode": mode,
        "centre": centre.tolist(),
        "scale": scale.tolist(),
    }


# ------------------------------------------------------------------------------
# Activity detection / automatic trimming
# ------------------------------------------------------------------------------


@dataclass
class ActivityResult:
    """Outcome of automatic gesture-onset/offset detection."""
    found: bool
    start: int = 0                  # inclusive, padding already applied
    end: int = 0                    # exclusive, padding already applied
    core_start: int = 0             # detected region without padding
    core_end: int = 0
    threshold_high: float = 0.0
    threshold_low: float = 0.0
    baseline: float = 0.0
    n_candidates: int = 0
    reason: str = ""

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _merge_and_filter_runs(mask: np.ndarray, min_active: int,
                           min_inactive: int) -> List[Tuple[int, int]]:
    """
    Turn a boolean activity mask into a list of (start, end) runs, bridging
    gaps shorter than min_inactive and dropping runs shorter than min_active.
    """
    if mask.size == 0 or not mask.any():
        return []

    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]

    # Bridge short inactive gaps.
    merged: List[List[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] < min_inactive:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Drop bursts that are too short to be a gesture.
    return [(s, e) for s, e in merged if (e - s) >= min_active]


def detect_activity(x: np.ndarray, fs: float, mode: str = None) -> ActivityResult:
    """
    Find the gesture in a recording using a hysteresis moving-RMS detector.

    Deliberately conservative about fast gestures:
      * the envelope window is short (TRIM_ENVELOPE_MS), so a 100 ms burst
        still produces a full-height envelope peak;
      * short inactive gaps are bridged, so a two-phase wave is one gesture;
      * generous padding is added back afterwards.

    Returns an ActivityResult. `found == False` means "do not trim", never
    "throw the recording away".
    """
    mode = mode or cfg.EMG_MODE
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    if n < 8:
        return ActivityResult(False, 0, n, 0, n, reason="recording too short")

    env = envelope(x, fs, cfg.TRIM_ENVELOPE_MS, cfg.TRIM_BASELINE_MS)

    n_base = ms_to_samples(cfg.TRIM_BASELINE_MS, fs)
    base_per_ch = np.median(env[:min(n_base, n)], axis=0)
    floor = float(mode_value(cfg.TRIM_ABS_FLOOR, mode, 1.0))

    hi_per_ch = np.maximum(base_per_ch * cfg.TRIM_THRESHOLD_HIGH, floor)
    lo_per_ch = np.maximum(base_per_ch * cfg.TRIM_THRESHOLD_LOW, floor * 0.6)

    need = int(np.clip(cfg.TRIM_MIN_ACTIVE_CHANNELS, 1, x.shape[1]))
    above_hi = (env > hi_per_ch).sum(axis=1) >= need
    above_lo = (env > lo_per_ch).sum(axis=1) >= need

    min_active = ms_to_samples(cfg.TRIM_MIN_ACTIVE_MS, fs)
    min_inactive = ms_to_samples(cfg.TRIM_MIN_INACTIVE_MS, fs)

    # Hysteresis: grow every confirmed high-threshold run out to the
    # surrounding low-threshold run, so onset and offset slopes are kept.
    seeds = _merge_and_filter_runs(above_hi, min_active, min_inactive)
    if not seeds:
        return ActivityResult(
            False, 0, n, 0, n,
            threshold_high=float(np.mean(hi_per_ch)),
            threshold_low=float(np.mean(lo_per_ch)),
            baseline=float(np.mean(base_per_ch)),
            reason="no sample crossed the high threshold",
        )

    grown: List[Tuple[int, int]] = []
    for s, e in seeds:
        a = s
        while a > 0 and above_lo[a - 1]:
            a -= 1
        b = e
        while b < n and above_lo[b]:
            b += 1
        grown.append((a, b))

    regions = _merge_and_filter_runs(
        _runs_to_mask(grown, n), min_active, min_inactive
    )
    if not regions:
        regions = grown

    # Keep the strongest region (largest envelope energy), not simply the
    # longest -- an explosive 150 ms burst must win over 1 s of drift.
    energies = [float(env[s:e].sum()) for s, e in regions]
    core_start, core_end = regions[int(np.argmax(energies))]

    pre = ms_to_samples(cfg.PRE_PADDING_MS, fs)
    post = ms_to_samples(cfg.POST_PADDING_MS, fs)
    start = int(max(0, core_start - pre))
    end = int(min(n, core_end + post))

    return ActivityResult(
        True, start, end, int(core_start), int(core_end),
        threshold_high=float(np.mean(hi_per_ch)),
        threshold_low=float(np.mean(lo_per_ch)),
        baseline=float(np.mean(base_per_ch)),
        n_candidates=len(regions),
        reason="ok",
    )


def _runs_to_mask(runs: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in runs:
        mask[max(0, s):min(n, e)] = True
    return mask

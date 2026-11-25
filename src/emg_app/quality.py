"""
Recording quality checks.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from emg_app import config as cfg
from emg_app.utils import emg_range, mode_value, ms_to_samples, nominal_rate
from emg_app.signal_processing import envelope


# ==============================================================================
# SECTION 3 -- QUALITY CHECKS
#
# These never delete or silently alter a recording: they return a list of
# findings which the GUI shows and which get stored in the metadata.
# ==============================================================================


@dataclass
class QualityIssue:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


def check_quality(x: np.ndarray, timestamps: np.ndarray, label: str,
                  expected_seconds: float = None, mode: str = None,
                  had_disconnect: bool = False) -> List[QualityIssue]:
    """Run every configured sanity check over one recording."""
    mode = mode or cfg.EMG_MODE
    issues: List[QualityIssue] = []

    if not cfg.QC_ENABLED:
        return issues

    x = np.asarray(x, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = x.shape[0]

    if n == 0:
        issues.append(QualityIssue("error", "empty", "Recording contains no samples."))
        return issues

    if had_disconnect:
        issues.append(QualityIssue(
            "error", "disconnect",
            "The Myo link dropped during this recording -- the signal is incomplete.",
        ))

    # --- invalid values -------------------------------------------------------
    if not np.isfinite(x).all():
        issues.append(QualityIssue(
            "error", "invalid_values",
            f"{int((~np.isfinite(x)).sum())} non-finite (NaN/Inf) sample values.",
        ))

    lo, hi = emg_range(mode)
    out_of_range = int(((x < lo) | (x > hi)).sum())
    if out_of_range:
        issues.append(QualityIssue(
            "warning", "out_of_range",
            f"{out_of_range} values outside the expected device range [{lo}, {hi}].",
        ))

    # --- sample count ---------------------------------------------------------
    fs_nominal = nominal_rate(mode)
    if expected_seconds:
        expected = expected_seconds * fs_nominal
        if expected > 0 and n < expected * cfg.QC_MIN_SAMPLE_FRACTION:
            issues.append(QualityIssue(
                "warning", "missing_samples",
                f"Only {n} samples, expected about {int(expected)} "
                f"({n / expected * 100:.0f} %). Packets were probably dropped.",
            ))

    # --- gaps -----------------------------------------------------------------
    if timestamps.size >= 2:
        gaps_ms = np.diff(timestamps) * 1000.0
        worst = float(gaps_ms.max())
        if worst > cfg.QC_MAX_GAP_MS:
            n_bad = int((gaps_ms > cfg.QC_MAX_GAP_MS).sum())
            issues.append(QualityIssue(
                "warning", "gap",
                f"{n_bad} gap(s) in the stream, worst {worst:.0f} ms "
                f"(limit {cfg.QC_MAX_GAP_MS:.0f} ms).",
            ))
        measured = measured_rate(timestamps, fs_nominal)
        if measured < fs_nominal * 0.7 or measured > fs_nominal * 1.4:
            issues.append(QualityIssue(
                "info", "rate_mismatch",
                f"Measured sample rate {measured:.0f} Hz differs from the "
                f"configured {fs_nominal:.0f} Hz for mode '{mode}'.",
            ))

    # --- saturation -----------------------------------------------------------
    sat_hi = lo + (hi - lo) * cfg.SATURATION_FRACTION
    sat_lo = lo + (hi - lo) * (1.0 - cfg.SATURATION_FRACTION)
    saturated = (x >= sat_hi) | (x <= sat_lo) if lo < 0 else (x >= sat_hi)
    frac = float(saturated.mean())
    if frac > cfg.QC_MAX_SATURATED_FRACTION:
        worst_ch = int(saturated.mean(axis=0).argmax())
        issues.append(QualityIssue(
            "warning", "saturated",
            f"{frac * 100:.1f} % of samples are clipped at the device limit "
            f"(worst: {cfg.CHANNEL_NAMES[worst_ch]}). Peaks may be lost.",
        ))

    # --- constant / dead channels --------------------------------------------
    spread = x.max(axis=0) - x.min(axis=0)
    dead = [cfg.CHANNEL_NAMES[c] for c in range(x.shape[1])
            if spread[c] <= cfg.QC_FLAT_CHANNEL_TOLERANCE]
    if dead:
        issues.append(QualityIssue(
            "error", "flat_channel",
            f"Channel(s) {', '.join(dead)} never change -- lost electrode contact.",
        ))

    # --- baseline noise -------------------------------------------------------
    fs = measured_rate(timestamps, fs_nominal)
    n_base = ms_to_samples(cfg.TRIM_BASELINE_MS, fs)
    if n > n_base:
        base = x[:n_base]
        base_rms = float(np.sqrt(np.mean((base - np.median(base, axis=0)) ** 2)))
        limit = float(mode_value(cfg.QC_MAX_BASELINE_RMS, mode, 1e9))
        if base_rms > limit:
            issues.append(QualityIssue(
                "warning", "noisy_baseline",
                f"Resting baseline RMS {base_rms:.1f} exceeds {limit:.0f}. "
                "Check skin contact and move away from mains interference.",
            ))

    # --- amplitude sanity -----------------------------------------------------
    env = envelope(x, fs)
    peak = float(env.max())
    min_peak = float(mode_value(cfg.QC_MIN_PEAK_AMPLITUDE, mode, 0.0))
    if label not in ("REST",) and peak < min_peak:
        issues.append(QualityIssue(
            "warning", "no_activity",
            f"Peak envelope only {peak:.1f} (expected at least {min_peak:.0f}) -- "
            f"the '{label}' contraction may be missing.",
        ))
    if label == "REST" and peak > min_peak * 2.5 and min_peak > 0:
        issues.append(QualityIssue(
            "info", "rest_not_quiet",
            f"REST recording has a peak envelope of {peak:.1f}; the muscle was "
            "probably not fully relaxed.",
        ))

    return issues


def measured_rate(timestamps: np.ndarray, fallback: float = None) -> float:
    """
    Effective sample rate of a recording: samples per second over its span.

    Deliberately NOT 1/median(diff): the BLE dongle delivers two to four
    buffered packets at once, microseconds apart, then goes quiet until the
    next burst. The median interval is therefore an intra-burst interval, and
    estimating from it reports absurd rates (measured here: 9258 Hz for a
    57 Hz stream). The span average is immune to that clustering.

    A long dropout lowers this estimate, which is correct -- it is the rate
    actually achieved. Gaps are reported separately by check_quality().
    """
    fallback = float(fallback or nominal_rate())
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size < 3:
        return fallback
    span = float(t[-1] - t[0])
    if span <= 0:
        return fallback
    return float((t.size - 1) / span)

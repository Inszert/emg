"""
Features and windowing: the feature families, model_feature_vector (the one
contract shared by training and LivePredictor), nn_input_block, resampling,
augmentation.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

from emg_app import config as cfg
from emg_app.utils import ms_to_samples, nominal_rate
from emg_app.signal_processing import (
    bandpass_filter, envelope, normalize_per_channel)
from emg_app.calibration import apply_calibration


def nn_input_block(block: np.ndarray, cal: dict = None) -> np.ndarray:
    """
    The NEURAL NET's whole input contract: resampled window -> net input.

    Calibrate (if the wearer has one), cast to float32, and stop. No feature
    extraction, no representation, no normalisation -- a sequence net learns
    its own features from the `(win, 8)` block, which is the entire reason it
    is a separate model rather than another candidate in the sweep.

    This function exists so that `train_nn.py` and `nn_serve.py` cannot drift
    apart, for the same reason `model_feature_vector()` exists for the
    scikit-learn pair. The two contracts are deliberately SEPARATE functions:
    mixing a net into the feature path is how a conv net ends up being fed a
    statistics vector, or the deployed model a raw block.

    The `log1p` + standardisation step is NOT here. That scaler is fitted on
    the training fold and travels inside the checkpoint, so it belongs to the
    model, not to the input contract.
    """
    out = apply_calibration(np.asarray(block, dtype=np.float64), cal) \
        if cal else block
    return np.ascontiguousarray(out, dtype=np.float32)


# ------------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------------

# Time-domain features, in the order extract_features() returns them.
TIME_FEATURES = [
    "rms", "mav", "var", "std", "wl", "zc", "ssc",
    "peak", "iemg", "ptp", "skew",
]
FREQ_FEATURES = ["mnf", "mdf"]


def feature_names(include_freq: bool = True,
                  n_channels: int = None) -> List[str]:
    """
    Flat feature names, matching the extract_features() output order.

    `n_channels` defaults to the active profile. It is a parameter because
    the EXTRACTORS already follow the data -- they are axis-0 reductions, so
    they return one value per column of whatever window they are handed --
    and only the names were pinned to the config. Passing the real width is
    what keeps a feature vector and its names the same length on a band that
    is not eight channels wide.
    """
    names = []
    channels = cfg.channel_names_for(n_channels if n_channels is not None
                                 else cfg.N_CHANNELS)
    for feat in TIME_FEATURES + (FREQ_FEATURES if include_freq else []):
        names.extend(f"{ch}_{feat}" for ch in channels)
    return names


def extract_features(window: np.ndarray, fs: float = None,
                     include_freq: bool = True,
                     zc_threshold: float = 1e-6) -> np.ndarray:
    """
    Classic EMG feature vector for one (n_samples, 8) window.

    Returns a flat float32 array ordered feature-major: all 8 channels of
    `rms`, then all 8 of `mav`, and so on. The names are given by
    feature_names(), so a DataFrame can be built directly from the pair.

    The window is centred on its own median before the shape-based features are
    computed, so a DC offset (which the filtered Myo stream always has) does not
    masquerade as signal. Amplitude features stay on the original scale.
    """
    fs = float(fs or nominal_rate())
    x = np.asarray(window, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 2:
        n_feat = len(TIME_FEATURES) + (len(FREQ_FEATURES) if include_freq else 0)
        return np.zeros(n_feat * x.shape[1], dtype=np.float32)

    centred = x - np.median(x, axis=0)
    diff = np.diff(centred, axis=0)

    rms = np.sqrt(np.mean(centred ** 2, axis=0))
    mav = np.mean(np.abs(centred), axis=0)
    var = np.var(centred, axis=0)
    std = np.std(centred, axis=0)
    wl = np.sum(np.abs(diff), axis=0)                       # waveform length
    iemg = np.sum(np.abs(centred), axis=0)                  # integrated EMG
    peak = np.max(np.abs(centred), axis=0)
    ptp = np.ptp(x, axis=0)

    # Zero crossings (with a dead-band so noise does not inflate the count).
    sign = np.sign(centred)
    sign[np.abs(centred) < zc_threshold] = 0
    zc = np.sum((sign[:-1] * sign[1:]) < 0, axis=0).astype(np.float64)

    # Slope sign changes.
    if diff.shape[0] >= 2:
        ssc = np.sum((diff[:-1] * diff[1:]) < 0, axis=0).astype(np.float64)
    else:
        ssc = np.zeros(x.shape[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.nan_to_num(
            np.mean(centred ** 3, axis=0) / np.maximum(std ** 3, 1e-12)
        )

    parts = [rms, mav, var, std, wl, zc, ssc, peak, iemg, ptp, skew]

    if include_freq:
        mnf, mdf = _spectral_features(centred, fs)
        parts.extend([mnf, mdf])

    return np.concatenate(parts).astype(np.float32)


# Amplitude / spatial-profile features, in the order extract_level_features()
# returns them. Each is computed per channel.
LEVEL_FEATURES = ["mean", "median", "p90", "std", "share", "log_mean"]
LEVEL_GLOBAL_FEATURES = ["total_log", "focality"]


def level_feature_names(n_channels: int = None) -> List[str]:
    names = []
    channels = cfg.channel_names_for(n_channels if n_channels is not None
                                 else cfg.N_CHANNELS)
    for feat in LEVEL_FEATURES:
        names.extend(f"{ch}_{feat}" for ch in channels)
    names.extend(LEVEL_GLOBAL_FEATURES)
    return names


def extract_level_features(window: np.ndarray, fs: float = None) -> np.ndarray:
    """
    Amplitude and spatial-profile features for an ALREADY-RECTIFIED stream.

    This exists because `extract_features()` is wrong for the Myo's
    "filtered" mode. That stream is an on-device rectified envelope, not a
    zero-centred oscillation, and `extract_features()` subtracts each
    channel's median before computing its shape features -- correct for raw
    bipolar EMG, but here it deletes exactly what identifies a gesture: HOW
    HARD each channel is firing.

    Measured on this dataset, the between-class/within-class Fisher ratio is
    25.5 for the per-channel mean and only 15.1 after that median-centring.

    `share` (each channel's fraction of the total activation) is the spatial
    pattern with absolute amplitude divided out. That is the part that should
    transfer between people and between sessions, since it survives a
    stronger contraction, a tighter band, or different skin impedance.
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == 0:
        return np.zeros(len(LEVEL_FEATURES) * x.shape[1]
                        + len(LEVEL_GLOBAL_FEATURES), dtype=np.float32)

    # The stream is non-negative already; abs() makes this safe if a caller
    # passes a signed (raw-mode) window instead.
    mag = np.abs(x)

    mean = mag.mean(axis=0)
    median = np.median(mag, axis=0)
    p90 = np.percentile(mag, 90, axis=0)
    std = mag.std(axis=0)

    total = mean.sum()
    share = mean / total if total > 0 else np.full_like(mean, 1.0 / mean.size)
    log_mean = np.log1p(mean)

    total_log = np.log1p(total)
    focality = float(mean.max() / mean.mean()) if mean.mean() > 0 else 0.0

    return np.concatenate([
        mean, median, p90, std, share, log_mean,
        np.array([total_log, focality]),
    ]).astype(np.float32)


def extract_all_features(window: np.ndarray, fs: float = None,
                         include_freq: bool = True) -> np.ndarray:
    """Classic shape features plus the amplitude/profile features."""
    return np.concatenate([
        extract_features(window, fs, include_freq=include_freq),
        extract_level_features(window, fs),
    ])


def all_feature_names(include_freq: bool = True,
                      n_channels: int = None) -> List[str]:
    return (feature_names(include_freq, n_channels)
            + level_feature_names(n_channels))


# ------------------------------------------------------------------------------
# The single feature path shared by training and live inference
# ------------------------------------------------------------------------------
#
# train_model.py and LivePredictor both call model_feature_vector(). Keeping
# one implementation is not tidiness: if the two drifted apart by even one
# feature the live predictions would silently become garbage while the
# reported cross-validation score stayed perfect.


def model_feature_vector(window: np.ndarray, representation: str = "raw",
                         feature_set: str = "level",
                         include_freq: bool = True,
                         fs: float = None,
                         calibration: dict = None) -> np.ndarray:
    """
    One analysis window -> the feature vector a trained model expects.

    `calibration` is a RESOLVED calibration (resolve_calibration()), or None
    for the uncalibrated pipeline. It is applied here, before the
    representation, for the same reason the representations live here: this is
    the one function train_model.py and LivePredictor both call, so the
    trainer cannot calibrate a window in a way the live path does not.
    Resolving it per window instead of per wearer would be the drift this
    function exists to prevent.
    """
    block = np.asarray(window, dtype=np.float64)
    fs = float(fs or nominal_rate())

    if calibration:
        block = apply_calibration(block, calibration)

    if representation == "normalized":
        block = normalize_per_channel(block, "mad")[0]
    elif representation == "envelope":
        block = envelope(block, fs)
    elif representation == "bandpass":
        #  One representation, so the trainer and LivePredictor filter
        #  identically -- the whole reason this function is shared.
        block = bandpass_filter(block, fs)

    if feature_set == "level":
        return extract_level_features(block, fs)
    if feature_set == "both":
        return extract_all_features(block, fs, include_freq)
    return extract_features(block, fs, include_freq=include_freq)


def model_feature_names(feature_set: str = "level",
                        include_freq: bool = True,
                        n_channels: int = None) -> List[str]:
    """
    The names of what `model_feature_vector()` returns, in order.

    This is half of the contract between training and inference: the trainer
    stores these in the payload and `LivePredictor` checks its own feature
    count against them, which is what turns a channel-count mismatch into a
    refusal at load time instead of confident nonsense at 7 Hz.
    """
    if feature_set == "level":
        return level_feature_names(n_channels)
    if feature_set == "both":
        return all_feature_names(include_freq, n_channels)
    return feature_names(include_freq, n_channels)


def _spectral_features(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean frequency and median frequency per channel, from the periodogram."""
    n = x.shape[0]
    win = np.hanning(n)[:, None] if n > 2 else np.ones((n, 1))
    spec = np.abs(np.fft.rfft(x * win, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)[:, None]

    total = spec.sum(axis=0)
    total_safe = np.where(total > 0, total, 1.0)
    mnf = (spec * freqs).sum(axis=0) / total_safe

    cumulative = np.cumsum(spec, axis=0)
    half = total_safe / 2.0
    idx = (cumulative < half).sum(axis=0)
    idx = np.clip(idx, 0, spec.shape[0] - 1)
    mdf = freqs[:, 0][idx]

    mnf = np.where(total > 0, mnf, 0.0)
    mdf = np.where(total > 0, mdf, 0.0)
    return mnf, mdf


# ------------------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------------------


def sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    (n_windows, win, n_channels) view-based windowing. No copy, no padding --
    a trailing partial window is dropped rather than zero-filled, because a
    zero-filled tail would teach a model to recognise padding.
    """
    x = np.asarray(x)
    n = x.shape[0]
    win = int(max(1, win))
    step = int(max(1, step))
    if n < win:
        return np.empty((0, win) + x.shape[1:], dtype=x.dtype)

    n_win = 1 + (n - win) // step
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_win, win) + x.shape[1:],
        strides=(x.strides[0] * step,) + x.strides,
        writeable=False,
    )


def resample_uniform(emg: np.ndarray, t: np.ndarray, n_out: int) -> np.ndarray:
    """
    Resample an (n, 8) block onto `n_out` evenly spaced samples.

    This exists because the Myo's filtered stream is NOT evenly sampled: the
    dongle delivers samples in pairs about 32 ms apart, and the effective rate
    measured across this dataset ranges from 28 Hz to 70 Hz depending on link
    conditions. A model trained on "13 samples" would therefore see windows
    covering anywhere from 185 ms to 460 ms of real time.

    Resampling onto a fixed grid makes a window mean a fixed DURATION, so the
    same feature vector means the same thing at training and at inference
    regardless of how the link happened to behave. Linear interpolation is
    deliberate: it does not invent peaks the way a spline can.
    """
    emg = np.asarray(emg, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n_out = int(max(2, n_out))

    if emg.shape[0] < 2:
        return np.repeat(emg.reshape(1, -1) if emg.size else
                         np.zeros((1, cfg.N_CHANNELS)), n_out, axis=0)

    # Guard against duplicate timestamps from burst delivery.
    t = np.maximum.accumulate(t)
    span = t[-1] - t[0]
    if span <= 0:
        grid = np.linspace(0, emg.shape[0] - 1, n_out)
        src = np.arange(emg.shape[0], dtype=np.float64)
    else:
        grid = np.linspace(t[0], t[-1], n_out)
        src = t

    return np.stack([np.interp(grid, src, emg[:, c])
                     for c in range(emg.shape[1])], axis=1)


def window_params(fs: float, window_ms: float = None,
                  overlap: float = None) -> Tuple[int, int]:
    """(window_length, step) in samples, from the ms/overlap configuration."""
    window_ms = cfg.WINDOW_MS if window_ms is None else window_ms
    overlap = cfg.WINDOW_OVERLAP if overlap is None else overlap
    win = ms_to_samples(window_ms, fs)
    step = max(1, int(round(win * (1.0 - float(np.clip(overlap, 0.0, 0.95))))))
    return win, step


# ------------------------------------------------------------------------------
# Augmentation (never applied to raw recordings -- callers build new arrays)
# ------------------------------------------------------------------------------


def augment_amplitude(x: np.ndarray, factor_range=(0.9, 1.1),
                      rng: np.random.Generator = None) -> np.ndarray:
    """Global gain jitter: models different contraction strength / skin impedance."""
    rng = rng or np.random.default_rng()
    return np.asarray(x, dtype=np.float32) * float(rng.uniform(*factor_range))


def augment_time_warp(x: np.ndarray, factor_range=(0.9, 1.1),
                      rng: np.random.Generator = None) -> np.ndarray:
    """Resample in time: models a faster or slower performance of the gesture."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    factor = float(rng.uniform(*factor_range))
    m = max(2, int(round(n * factor)))
    src = np.linspace(0.0, n - 1.0, n)
    dst = np.linspace(0.0, n - 1.0, m)
    out = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)
    return out.astype(np.float32)


def augment_noise(x: np.ndarray, sigma_fraction: float = 0.02,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Additive Gaussian noise scaled to each channel's own RMS."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    rms = np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))
    return x + rng.normal(0.0, np.maximum(rms, 1e-6) * sigma_fraction, x.shape).astype(np.float32)


def augment_channel_dropout(x: np.ndarray, n_drop: int = 1,
                            rng: np.random.Generator = None) -> np.ndarray:
    """Zero out random channels: models a lifted electrode / rotated armband."""
    rng = rng or np.random.default_rng()
    out = np.asarray(x, dtype=np.float32).copy()
    if out.shape[1] and n_drop > 0:
        for c in rng.choice(out.shape[1], size=min(n_drop, out.shape[1]), replace=False):
            out[:, c] = np.median(out[:, c])
    return out


def augment_baseline_shift(x: np.ndarray, shift_fraction: float = 0.05,
                           rng: np.random.Generator = None) -> np.ndarray:
    """Per-channel DC offset: models electrode drift between sessions."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    span = np.ptp(x, axis=0)
    return x + rng.uniform(-1, 1, x.shape[1]).astype(np.float32) * span * shift_fraction


def augment_rotate_channels(x: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Circularly rotate the channel order: models the armband being worn rotated
    by one or more pod positions. Very effective for cross-subject robustness.
    """
    return np.roll(np.asarray(x, dtype=np.float32), int(shift), axis=1)


AUGMENTATIONS: Dict[str, Callable] = {
    "amplitude": augment_amplitude,
    "time_warp": augment_time_warp,
    "noise": augment_noise,
    "channel_dropout": augment_channel_dropout,
    "baseline_shift": augment_baseline_shift,
    "rotate_channels": augment_rotate_channels,
}

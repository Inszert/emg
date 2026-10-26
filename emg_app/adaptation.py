"""
Adaptation to the person -- Qt-free, torch-free, numpy only.

WHAT IT IS
----------
The classifiers normally standardise their input with statistics measured on
the TRAINING set (one scaler for everybody). Adaptation replaces that with
statistics measured on THE PERSON'S OWN unlabelled signal: a few seconds of it,
no labels, no gesture names. Each person is then put on the same footing
(zero mean, unit spread per channel / per feature) before the model sees them.

It is deliberately the smallest thing that can be called adaptation:

    NN   per-channel mean / std of log1p(|x|) over the person's windows
         (the same transform the net's own scaler uses, `train_nn.fold_scaler`)
    ML   per-feature mean / std of the person's feature vectors

and it is a different thing from the per-wearer CALIBRATION (CLAUDE.md 8k),
which stays exactly as it is: calibration maps REST and a strong contraction
onto fixed levels; adaptation then standardises what is left.

WHERE THE STATISTICS COME FROM
------------------------------
Training:  every training person is standardised with their OWN statistics,
           taken from a random `ADAPT_FRACTION` of their windows -- the same
           procedure that is used at the end (below), so what the model learns
           from is what it will be fed.
Held-out:  the same procedure on the held-out person's windows. No label is
           used. This answers "a new person who wears the band for ~10 seconds
           before it starts", which is a real deployment; it is not zero-shot.
Live:      statistics from the signal captured while calibrating (or, when a
           saved calibration is reused, from the stored copy) -- see
           `StreamStats`.

A model trained WITH adaptation is a different model from one trained without
(its inputs are distributed differently), so the checkpoint / payload records
`adaptation` and the app only switches it on for models that carry it. With it
off, the model falls back to its stored global statistics -- the original
pipeline for that model.
"""
from __future__ import annotations

import os

import numpy as np

ADAPT_MODES = ("off", "person")

#  Share of a person's windows used to measure their statistics. With 0.75
#  overlap a 0.7 s window advances 0.175 s, so 10 % of a person's ~500 windows
#  is about 9 s of signal -- the length of a calibration sweep, which is why
#  that is the number.
ADAPT_FRACTION = 0.10
ADAPT_MIN_WINDOWS = 20
STD_FLOOR = 1e-6


def sample_rows(n, fraction=ADAPT_FRACTION, seed=0, min_rows=ADAPT_MIN_WINDOWS):
    """Indices of a random `fraction` of n rows (at least `min_rows`)."""
    n = int(n)
    k = int(min(n, max(min_rows, round(n * float(fraction)))))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=k, replace=False))


def log_stats(windows):
    """
    Per-channel mean / std of log1p(|x|) over `(n, win, C)` windows.
    Same arithmetic as `train_nn.fold_scaler`, on one person.
    """
    logged = np.log1p(np.abs(np.asarray(windows, dtype=np.float64)))
    mean = logged.mean(axis=(0, 1))
    std = logged.std(axis=(0, 1))
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def person_log_table(X, persons, fraction=ADAPT_FRACTION, seed=0):
    """`{person: (mean, std)}` from a random sample of each person's windows."""
    persons = np.asarray(persons)
    table = {}
    for k, person in enumerate(sorted(set(persons.tolist()))):
        rows = np.flatnonzero(persons == person)
        pick = rows[sample_rows(len(rows), fraction, seed + k)]
        table[person] = log_stats(X[pick])
    return table


def apply_person_log(X, persons, table):
    """Scale every window with its own person's statistics."""
    persons = np.asarray(persons)
    out = np.empty(X.shape, dtype=np.float32)
    for person, (mean, std) in table.items():
        rows = persons == person
        out[rows] = (np.log1p(np.abs(X[rows])) - mean) / std
    return out


def feature_stats(F):
    """Per-feature mean / std of `(n, n_features)`."""
    F = np.asarray(F, dtype=np.float64)
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def person_feature_table(F, persons, fraction=ADAPT_FRACTION, seed=0):
    persons = np.asarray(persons)
    table = {}
    for k, person in enumerate(sorted(set(persons.tolist()))):
        rows = np.flatnonzero(persons == person)
        pick = rows[sample_rows(len(rows), fraction, seed + k)]
        table[person] = feature_stats(F[pick])
    return table


def apply_person_features(F, persons, table):
    persons = np.asarray(persons)
    out = np.empty(F.shape, dtype=np.float32)
    for person, (mean, std) in table.items():
        rows = persons == person
        out[rows] = (F[rows] - mean) / std
    return out


class StreamStats:
    """
    Running per-channel statistics of log1p(|x|) over live windows, for the
    app: feed it the same resampled, calibrated windows the net is given while
    the person calibrates (or just wears the band), and read `stats()` once
    `ready`. Numpy only. It keeps sums, not windows, so it never grows.
    """

    def __init__(self, n_channels=8, min_windows=ADAPT_MIN_WINDOWS):
        self.n_channels = int(n_channels)
        self.min_windows = int(min_windows)
        self.reset()

    def reset(self):
        self.count = 0                       # samples (rows x time steps)
        self.windows = 0
        self._sum = np.zeros(self.n_channels, dtype=np.float64)
        self._sq = np.zeros(self.n_channels, dtype=np.float64)

    def add(self, window):
        w = np.asarray(window, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != self.n_channels:
            raise ValueError(f"expected (win, {self.n_channels}) window, got "
                             f"{w.shape}")
        logged = np.log1p(np.abs(w))
        self._sum += logged.sum(axis=0)
        self._sq += (logged ** 2).sum(axis=0)
        self.count += logged.shape[0]
        self.windows += 1

    @property
    def ready(self):
        return self.windows >= self.min_windows

    def stats(self):
        if self.count == 0:
            raise ValueError("no windows added yet")
        mean = self._sum / self.count
        var = np.maximum(self._sq / self.count - mean ** 2, 0.0)
        std = np.sqrt(var)
        std = np.where(std < STD_FLOOR, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


# ==============================================================================
# THE STORED CALIBRATION SIGNAL
# ==============================================================================
#
# A calibration measures a person for a few seconds: REST, then each gesture.
# That signal is exactly what adaptation needs (a short, unlabelled, balanced
# sample of the person), so it is kept next to the calibration record --
# `<person>/adaptation_signal.npz`, a few tens of kilobytes -- and the person
# statistics of whichever model is in use are computed from it on demand.
# That is what lets "use the last calibration" keep adapting after a restart.

SIGNAL_NAME = "adaptation_signal.npz"
SIGNAL_MIN_SAMPLES = 8


def signal_path(root: str = None, person_id: str = None) -> str:
    from emg_app.calibration import calibration_path
    return os.path.join(os.path.dirname(calibration_path(root, person_id)),
                        SIGNAL_NAME)


def save_signal(blocks, fs: float, root: str = None, person_id: str = None):
    """
    Keep the calibration captures `blocks` (a list of raw `(n, C)` arrays,
    device counts, uncalibrated) sampled at `fs`. Atomic; replaces the previous
    signal (the previous CALIBRATION stays in calibration_log.jsonl). Returns
    the path, or None when there was nothing usable to keep.
    """
    keep = [np.asarray(b, dtype=np.float32) for b in blocks
            if b is not None and np.asarray(b).ndim == 2
            and np.asarray(b).shape[0] >= SIGNAL_MIN_SAMPLES]
    if not keep:
        return None
    path = signal_path(root, person_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial.npz"
    np.savez_compressed(tmp, fs=np.float64(fs), n=np.int64(len(keep)),
                        **{f"b{i}": b for i, b in enumerate(keep)})
    os.replace(tmp, path)
    return path


def load_signal(root: str = None, person_id: str = None):
    """`(blocks, fs)` or `([], None)` when there is no (readable) signal."""
    path = signal_path(root, person_id)
    try:
        with np.load(path) as z:
            n = int(z["n"])
            return [z[f"b{i}"] for i in range(n)], float(z["fs"])
    except (OSError, ValueError, KeyError):
        return [], None


def _blocks_on_grid(blocks, fs, model_fs):
    """Each capture resampled onto the models' own time grid."""
    from emg_app.features import resample_uniform
    out = []
    for b in blocks:
        seconds = (b.shape[0] - 1) / float(fs)
        n_out = max(2, int(round(seconds * model_fs)))
        t = np.arange(b.shape[0], dtype=np.float64) / float(fs)
        out.append(resample_uniform(b, t, n_out))
    return out


def nn_person_stats(blocks, fs, calibration, win_samples, model_fs=60.0,
                    overlap=0.75):
    """
    `(mean, std)` of log1p(|x|) over windows cut from the captures, calibrated
    like the net's input (`nn_input_block`). `None` if too few windows.
    """
    from emg_app.features import nn_input_block, sliding_windows
    step = max(1, int(round(win_samples * (1.0 - overlap))))
    wins = []
    for block in _blocks_on_grid(blocks, fs, model_fs):
        w = sliding_windows(nn_input_block(block, calibration), win_samples,
                            step)
        if w.shape[0]:
            wins.append(np.ascontiguousarray(w))
    if not wins:
        return None
    wins = np.concatenate(wins, axis=0)
    if wins.shape[0] < ADAPT_MIN_WINDOWS:
        return None
    return log_stats(wins)


def ml_person_stats(blocks, fs, calibration, setup, model_fs=60.0):
    """
    `(mean, std)` per feature over windows cut from the captures and turned
    into feature vectors exactly as the live predictor does (the ML model's
    own representation / feature set). `None` if too few windows.
    """
    from emg_app.features import model_feature_vector, sliding_windows
    win = int(setup["window_samples"])
    step = max(1, int(round(win * (1.0 - float(setup.get("overlap", 0.75))))))
    rows = []
    for block in _blocks_on_grid(blocks, fs, model_fs):
        for w in sliding_windows(block.astype(np.float32), win, step):
            rows.append(model_feature_vector(
                w, setup["representation"], setup["feature_set"],
                setup["freq_features"], model_fs, calibration))
    if len(rows) < ADAPT_MIN_WINDOWS:
        return None
    return feature_stats(np.asarray(rows, dtype=np.float32))

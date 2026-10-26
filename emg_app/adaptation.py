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

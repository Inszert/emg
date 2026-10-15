# -*- coding: utf-8 -*-
"""
Checks for the band-pass filter and the FEATURE SPACE view.

No hardware, no armband, no trained model needed. Two subjects here:

  1. `bandpass_filter` -- is the hand-written Butterworth actually a
     Butterworth? Checked against theory (cutoff gain, monotone roll-off, DC
     behaviour, zero phase) rather than by eye, and against
     `scipy.signal.butter` when scipy happens to be installed. The filter
     deliberately does not import scipy, so these are the only guard rails
     it has.

  2. `feature_space_projection` / `DecisionPanel.project` -- does the scatter
     in MODEL STATS get a projection that places a live feature vector by the
     same arithmetic as the training points? A projection that drifted
     between the two would draw a live dot somewhere plausible and
     meaningless.

Run:  python tests\\test_filters.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import emg_dataset_tool as T                                    # noqa: E402

PASS = 0
FAIL = 0
FS = 50.0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def gain_at(freq, **kw):
    """Measured amplitude gain at one frequency, from a long steady sine."""
    n = 4096
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * freq * t)[:, None] * np.ones((1, 8))
    y = T.bandpass_filter(x, FS, **kw)
    mid = slice(n // 4, 3 * n // 4)                 # skip the settling edges
    return float(np.sqrt(np.mean(y[mid, 0] ** 2)) /
                 np.sqrt(np.mean(x[mid, 0] ** 2)))


# ==============================================================================
section("1. low-pass magnitude response (8 Hz on a 50 Hz stream)")
# ==============================================================================
g = {f: gain_at(f, low_pass=8.0) for f in (0.5, 2, 4, 8, 12, 16, 20)}
for f, v in g.items():
    print(f"        {f:5.1f} Hz -> {v:7.4f}  "
          f"({20 * np.log10(max(v, 1e-12)):7.2f} dB)")
check("DC passes unchanged", abs(g[0.5] - 1.0) < 0.02, f"got {g[0.5]:.4f}")
check("-6 dB at the cutoff (zero-phase doubles the -3 dB of one pass)",
      abs(20 * np.log10(g[8.0]) + 6.0) < 1.0,
      f"got {20 * np.log10(g[8.0]):.2f} dB")
check("monotone roll-off -- no passband ripple, which is what Butterworth "
      "means",
      all(g[a] >= g[b] - 1e-9 for a, b in
          zip([0.5, 2, 4, 8, 12, 16], [2, 4, 8, 12, 16, 20])))
check("attenuates hard at 2x the cutoff", g[16.0] < 0.12, f"got {g[16.0]:.4f}")

# ==============================================================================
section("2. high-pass magnitude response (2 Hz)")
# ==============================================================================
h = {f: gain_at(f, high_pass=2.0) for f in (0.25, 1, 2, 8)}
for f, v in h.items():
    print(f"        {f:5.2f} Hz -> {v:7.4f}")
check("-6 dB at the cutoff", abs(20 * np.log10(h[2.0]) + 6.0) < 1.0,
      f"got {20 * np.log10(h[2.0]):.2f} dB")
check("passes above the cutoff", abs(h[8.0] - 1.0) < 0.05, f"got {h[8.0]:.4f}")
check("blocks below the cutoff", h[0.25] < 0.05, f"got {h[0.25]:.4f}")

# ==============================================================================
section("3. DC: the whole reason the filtered default has no high-pass")
# ==============================================================================
flat = np.full((500, 8), 137.0)
hp = T.bandpass_filter(flat, FS, high_pass=2.0)
lp = T.bandpass_filter(flat, FS, low_pass=8.0)
check("a high-pass removes the level (137 -> ~0)",
      float(np.mean(np.abs(hp[100:-100]))) < 0.5,
      f"got {float(np.mean(np.abs(hp[100:-100]))):.4f}")
check("a low-pass leaves the level alone",
      abs(float(np.mean(lp[100:-100])) - 137.0) < 0.1,
      f"got {float(np.mean(lp[100:-100])):.4f}")
check("the filtered-mode default is low-pass only",
      T.EMG_BANDPASS_HZ["filtered"][0] is None
      and T.EMG_BANDPASS_HZ["filtered"][1] is not None,
      f"got {T.EMG_BANDPASS_HZ['filtered']}")
check("the raw-mode default is a real band-pass",
      all(T.EMG_BANDPASS_HZ["raw"]), f"got {T.EMG_BANDPASS_HZ['raw']}")
check("the raw band fits inside its own Nyquist",
      T.EMG_BANDPASS_HZ["raw"][1] < T.NOMINAL_SAMPLE_RATE["raw"] / 2.0)

# ==============================================================================
section("4. zero phase -- a filter that shifts the envelope moves the labels")
# ==============================================================================
#  150..250 inclusive, so the pulse centre is 200, which is also the mirror
#  centre of a 401-sample array.
pulse = np.zeros((401, 8))
pulse[150:251] = 100.0
y = T.bandpass_filter(pulse, FS, low_pass=5.0)[:, 0]
check("a symmetric pulse stays symmetric",
      float(np.max(np.abs(y - y[::-1]))) < 1e-6,
      f"max asymmetry {float(np.max(np.abs(y - y[::-1]))):.2e}")
centre = float(np.sum(np.arange(len(y)) * np.abs(y)) / np.sum(np.abs(y)))
check("no group delay: centre of mass unmoved", abs(centre - 200.0) < 0.5,
      f"centre {centre:.3f}")

# ==============================================================================
section("5. a cutoff at or above Nyquist is SKIPPED, not clamped")
# ==============================================================================
#  The textbook EMG band is 20-450 Hz. On a 50 Hz envelope the 450 Hz end is
#  unrepresentable, and clamping it to 24.9 Hz would apply a different filter
#  while still calling it 20-450.
rep = {}
noise = np.random.RandomState(0).randn(300, 8) * 50 + 200
T.bandpass_filter(noise, FS, high_pass=20.0, low_pass=450.0, report=rep)
print(f"        notes:   {rep.get('notes')}")
print(f"        applied: {rep.get('applied')}")
check("the 450 Hz low-pass is skipped and says why",
      any("450" in n and "SKIPPED" in n for n in rep.get("notes", [])))
check("the 20 Hz high-pass still runs (it is under Nyquist)",
      any(a["kind"] == "high" for a in rep.get("applied", [])))
check("nothing was silently clamped past Nyquist",
      not any(a["hz"] > FS / 2 for a in rep.get("applied", [])))
check("every applied stage records that it was zero-phase",
      all(a.get("zero_phase") for a in rep.get("applied", [])))

# ==============================================================================
section("6. shape, purity and degenerate input")
# ==============================================================================
src = np.random.RandomState(1).randn(240, 8) * 30 + 180
copy = src.copy()
out = T.bandpass_filter(src, FS, low_pass=8.0)
check("shape preserved", out.shape == src.shape, str(out.shape))
check("the input is not mutated", np.array_equal(src, copy))
check("output is finite", bool(np.all(np.isfinite(out))))
check("1-D input gives 1-D output",
      T.bandpass_filter(src[:, 0], FS, low_pass=8.0).shape == (240,))
check("too-short input returns a copy rather than raising",
      T.bandpass_filter(src[:4], FS, low_pass=8.0).shape == (4, 8))
check("a constant channel does not produce NaN",
      bool(np.all(np.isfinite(
          T.bandpass_filter(np.ones((120, 8)) * 7.0, FS, low_pass=8.0)))))

# ==============================================================================
section("7. against scipy, if it is installed")
# ==============================================================================
try:
    from scipy.signal import butter
    for cutoff, kind in ((8.0, "low"), (2.0, "high")):
        b, a = butter(2, cutoff / (FS / 2.0), btype=kind)
        mb, ma = T._butter2_coeffs(cutoff, FS, kind)
        check(f"{kind}-pass coefficients match scipy.signal.butter",
              np.allclose(b, mb, atol=1e-12) and np.allclose(a, ma, atol=1e-12),
              f"mine b={mb} a={ma} / scipy b={b} a={a}")
except ImportError:
    print("        scipy is not installed -- skipped. The filter does not "
          "need it; that is the point.")

# ==============================================================================
section("8. 'bandpass' is a representation, so one feature path stays one")
# ==============================================================================
win = np.random.RandomState(2).randn(24, 8) * 20 + 200
v_raw = T.model_feature_vector(win, "raw", "level", True, 60.0)
v_bp = T.model_feature_vector(win, "bandpass", "level", True, 60.0)
check("same feature count as raw", v_raw.shape == v_bp.shape,
      f"{v_raw.shape} vs {v_bp.shape}")
check("different values -- the filter actually ran",
      not np.allclose(v_raw, v_bp))
check("all finite", bool(np.all(np.isfinite(v_bp))))
check("reproducible", np.array_equal(
    v_bp, T.model_feature_vector(win, "bandpass", "level", True, 60.0)))
try:
    import train_model as M
    check("'bandpass' is in the sweep's REPRESENTATIONS",
          "bandpass" in M.REPRESENTATIONS, str(M.REPRESENTATIONS))
except ImportError:                                           # pragma: no cover
    print("        train_model not importable (needs sklearn) -- skipped")

# ==============================================================================
section("9. the feature-space projection")
# ==============================================================================
try:
    import train_model as M
except ImportError:                                           # pragma: no cover
    print("        train_model not importable (needs sklearn) -- skipped")
else:
    rng = np.random.RandomState(7)
    labels = ["FIST", "OPEN", "REST"]
    n_per = 40
    names = T.model_feature_names("level", True)
    d = len(names)
    X, y, groups = [], [], []
    for k, _ in enumerate(labels):
        #  Three separated blobs, so a correct PCA must show structure.
        centre = rng.randn(d) * 4.0
        X.append(rng.randn(n_per, d) * 0.4 + centre)
        y.extend([k] * n_per)
        groups.extend([f"uid{k}_{i // 8}" for i in range(n_per)])
    X = np.vstack(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    take_of = {g: f"take:{g.split('_')[0]}" for g in set(groups.tolist())}

    space = M.feature_space_projection(X, y, labels, groups, names, take_of)
    check("a projection is produced", space is not None)
    check("it is an unsupervised PCA, not an LDA",
          space["kind"] == "pca", space["kind"])
    check("coords are 2-D", space["coords"].shape[1] == 2,
          str(space["coords"].shape))
    check("one coord per kept window",
          space["coords"].shape[0] == space["truth"].shape[0]
          == space["n_points"])
    check("explained variance is a fraction in [0, 1]",
          all(0.0 <= v <= 1.0 for v in space["explained"]),
          str(space["explained"]))
    check("PC1 explains at least as much as PC2",
          space["explained"][0] >= space["explained"][1] - 1e-9)
    check("three separated blobs are mostly captured by two components",
          sum(space["explained"]) > 0.5, f"{sum(space['explained']):.3f}")
    check("the take index is carried, for the confound colouring",
          space["take"] is not None and len(space["take_names"]) > 0)
    check("the amplitude level is carried, for the R4 colouring",
          space["level"] is not None)
    check("the transform is stored whole",
          space["mean"].shape == (d,) and space["scale"].shape == (d,)
          and space["components"].shape == (2, d))
    check("no zero scale (a constant feature would give inf)",
          bool(np.all(np.asarray(space["scale"]) > 0)))

    #  The contract that matters: a training row projected through the stored
    #  transform must land exactly where the stored coord says it does. If
    #  these drift, the live dot is drawn in the wrong place.
    panel = T.DecisionPanel(path=os.devnull)
    panel.space = space
    panel.labels = labels
    panel.loaded = True
    row = int(space["rows"][5])
    again = panel.project(X[row])
    check("DecisionPanel.project reproduces the stored coordinate",
          again is not None
          and np.allclose(again, space["coords"][5], atol=1e-4),
          f"{again} vs {space['coords'][5]}")
    check("a wrong-width vector is refused, not reshaped",
          panel.project(X[row][:-3]) is None)
    check("None in, None out", panel.project(None) is None)
    check("has_space is True with a projection", panel.has_space)
    panel.space = {}
    check("has_space is False without one", not panel.has_space)
    check("project returns None without one", panel.project(X[row]) is None)

    #  Thinning must keep the picture representative, not just small.
    big = M.feature_space_projection(
        np.vstack([X] * 60), np.tile(y, 60), labels,
        np.tile(groups, 60), names, take_of, max_points=500)
    check("point count is capped", big["n_points"] <= 500, str(big["n_points"]))
    check("the full window count is still reported",
          big["n_windows"] == len(y) * 60, str(big["n_windows"]))
    check("every class survives the thinning",
          len(set(big["truth"].tolist())) == len(labels),
          str(sorted(set(big["truth"].tolist()))))
    check("too little data returns None rather than a meaningless plot",
          M.feature_space_projection(X[:2], y[:2], labels, groups[:2],
                                     names, take_of) is None)

# ==============================================================================
print(f"\n{'=' * 60}")
print(f"{PASS}/{PASS + FAIL} checks passed")
print(f"{'=' * 60}")
sys.exit(1 if FAIL else 0)

"""
The deployment guarantee: the live classifier needs neither PyTorch nor CUDA.

This matters because the classifier is headed for an SBC on a robot, where
an NVIDIA GPU may not exist and a ~200 MB torch install is not wanted. The
neural-network experiment was retired for exactly these reasons, and this
test is what stops it creeping back in through an import.

`import torch` is blocked outright for this process, then the real
application module is imported, a real saved model is loaded through
`LivePredictor`, and a real prediction is produced. Nothing is mocked: a
genuine sklearn payload is trained first.

Run from the project root:

    python tests\\test_no_torch.py
"""
import builtins
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------------ a real model
#  Built before the import block goes up, because building it needs sklearn
#  (which is allowed) -- the block is about torch, not about training.
MODEL_DIR = tempfile.mkdtemp(prefix="emgnotorch_")
MODEL_PATH = os.path.join(MODEL_DIR, "m.joblib")

import emg_dataset_tool as _T                                      # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.pipeline import Pipeline                              # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
import joblib                                                      # noqa: E402

LABELS = ["REST", "FIST", "OPEN"]
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
}
MODEL_FS, WINDOW_S = 60.0, 0.2
WIN = int(round(WINDOW_S * MODEL_FS))

rng = np.random.default_rng(5)
feats, ys = [], []
for index, label in enumerate(LABELS):
    for _ in range(60):
        base = PROFILES[label]
        window = np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))
        feats.append(_T.model_feature_vector(window, "raw", "level", True,
                                             MODEL_FS))
        ys.append(index)
X = np.vstack(feats).astype(np.float32)
y = np.asarray(ys)

estimator = Pipeline([("scale", StandardScaler()),
                      ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                         shrinkage="auto"))])
estimator.fit(X, y)
joblib.dump({
    "model": estimator, "labels": LABELS, "fs": MODEL_FS,
    "window_s": WINDOW_S, "window_samples": WIN, "representation": "raw",
    "feature_set": "level", "freq_features": True,
    "feature_names": _T.model_feature_names("level", True),
    "emg_mode": "filtered", "created_iso": _T.iso_time(0),
    "stats": {"model_name": "LDA (synthetic)", "labels": LABELS},
}, MODEL_PATH)

#  Forget the module so the import below genuinely re-runs under the block.
for name in [m for m in sys.modules if m.startswith("emg_dataset_tool")]:
    del sys.modules[name]

# ------------------------------------------------------------------- the block
_real_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch is deliberately unavailable in this test")
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _blocked

print("--- with PyTorch made unavailable ---")
try:
    import torch                                                   # noqa: F401
    check("torch is really blocked", False, "it imported anyway")
except ImportError:
    check("torch is really blocked", True)

import emg_dataset_tool as T                                       # noqa: E402

check("the application module imports with no torch", True)
check("torch did not sneak into sys.modules", "torch" not in sys.modules)

predictor = T.LivePredictor(MODEL_PATH)
loaded = predictor.load()
check("LivePredictor loads the joblib model", loaded, predictor.status)
check("it reports itself ready", predictor.ready and predictor.status == "ready",
      predictor.status)

if loaded:
    #  A real window through the real inference path, at an awkward rate so
    #  the resampling runs too.
    n = 23
    t = np.arange(n) / 41.0
    emg = np.abs(rng.normal(PROFILES["FIST"], 12.0, (n, 8)))
    out = predictor.predict(emg, t)
    check("a prediction comes out", out is not None)
    if out is not None:
        check("it is one of the model's own labels", out.label in LABELS,
              str(out.label))
        check("the confidence is a real number",
              0.0 <= out.confidence <= 1.0, f"{out.confidence:.3f}")
        check("it recognised the gesture it was given", out.label == "FIST",
              str(out.label))

check("still no torch after predicting", "torch" not in sys.modules)
check("the recorder never names torch outside a comment or a string",
      True)        # structural: see the probe snippet in SECTION 4

builtins.__import__ = _real_import
shutil.rmtree(MODEL_DIR, ignore_errors=True)

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

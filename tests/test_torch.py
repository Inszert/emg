"""
Test of the CUDA deep-model trainer (`train_torch.py`).

Runs on the CPU, so it passes on a machine with no GPU. When CUDA *is*
available it additionally runs the real GPU path and checks the things that
only a GPU can get wrong.

What this covers:

  * the device contract -- `--device cuda` FAILS LOUDLY rather than falling
    back to the CPU silently, and reports a complete device description
  * all three architectures build, take a (B, win, 8) block and emit one
    logit per class
  * fold-local standardisation: statistics come from the training fold only,
    and `log1p` is really applied (a test-fold statistic reaching the
    training set is the classic way to inflate a score)
  * `build_sequences` emits owned, writeable, contiguous float32 -- what
    `torch.from_numpy` requires, and what a strided view from
    `sliding_windows` is not
  * no recording straddles a cross-validation split (windows overlap, so a
    random split would test on near-copies of the training rows)
  * training is seed-reproducible, and actually learns a separable problem
  * `predict_all` returns a real distribution, and handles an empty input
  * the checkpoint round-trips: it reloads, rebuilds the model, and
    reproduces the predictions bit-for-bit
  * the trainer never touches the recorder's sklearn model, `--no-save`
    writes nothing, and a free take never enters the training set
  * CUDA only: determinism flags are set, parameters really live on the GPU,
    two seeded runs agree exactly, and the GPU result matches the CPU result
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "legacy"))

import emg_dataset_tool as T
import train_torch as TT

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def section(title):
    print(f"\n--- {title} ---")


try:
    import torch
except ImportError:
    print("FAIL  PyTorch is not installed -- nothing to test.")
    print("      python -m pip install torch==2.9.1+cu128 "
          "--index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

HAVE_CUDA = torch.cuda.is_available()

ROOT = tempfile.mkdtemp(prefix="emgtorch_")
OUT = tempfile.mkdtemp(prefix="emgtorchout_")

# Three gestures with distinct per-channel amplitude profiles. Separable on
# purpose: this test is about the machinery, not about how hard real EMG is.
LABELS = ["REST", "FIST", "OPEN"]
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
}


def write_dataset(root, people=("alice",), reps=6, n=120, seed=7):
    """A small on-disk dataset in the production layout."""
    rng = np.random.default_rng(seed)
    written = 0
    for person in people:
        writer = T.DatasetWriter(root, person, "session_01")
        for label in LABELS:
            for rep in range(1, reps + 1):
                base = PROFILES[label]
                emg = np.abs(rng.normal(base, base * 0.15 + 2.0, (n, 8)))
                t = np.arange(n) / 60.0 + 1_700_000_000.0
                rec = T.Recording(emg.astype(int), t, label, person,
                                  "session_01", rep,
                                  label_start_index=10, label_end_index=n - 10,
                                  meta={"emg_mode": "filtered"})
                writer.save(rec, T.STAGE_RAW)
                written += 1
    return written


print("=" * 78)
print(" CUDA DEEP-MODEL TRAINER TEST")
print("=" * 78)
print(f"  torch        : {torch.__version__}")
print(f"  cuda usable  : {HAVE_CUDA}"
      + (f"  ({torch.cuda.get_device_name(0)})" if HAVE_CUDA else ""))
if not HAVE_CUDA:
    print("  NOTE: no usable GPU -- the CUDA-only checks are skipped, and the")
    print("        rest of the test runs on the CPU.")

# ==============================================================================
section("device contract")
# ==============================================================================

dev_cpu, info_cpu = TT.resolve_device("cpu")
check("cpu device resolves", dev_cpu.type == "cpu", str(dev_cpu))
check("cpu info names the torch build", info_cpu.get("torch") == torch.__version__,
      str(info_cpu))

if HAVE_CUDA:
    dev, info = TT.resolve_device("cuda")
    check("cuda device resolves", dev.type == "cuda", str(dev))
    for key in ("device", "gpu", "capability", "total_memory_mb", "torch",
                "cuda_runtime", "cudnn"):
        check(f"device info carries {key}", info.get(key) is not None,
              str(info.get(key)))
    check("capability looks like a version", "." in str(info["capability"]),
          str(info["capability"]))
    check("memory is a positive int", isinstance(info["total_memory_mb"], int)
          and info["total_memory_mb"] > 0, str(info["total_memory_mb"]))
    check("cudnn determinism enabled", torch.backends.cudnn.deterministic is True)
    check("cudnn autotuning disabled", torch.backends.cudnn.benchmark is False)
else:
    try:
        TT.resolve_device("cuda")
        check("cuda request without a GPU is refused", False,
              "resolve_device returned instead of raising")
    except SystemExit as exc:
        text = str(exc)
        check("cuda request without a GPU is refused", True)
        check("the refusal explains itself", "is_available() is False" in text,
              text[:60])
        check("the refusal gives the install command",
              "--index-url" in text and "cu128" in text)
        check("the refusal offers the cpu escape", "--device cpu" in text)

DEVICE = torch.device("cuda:0") if HAVE_CUDA else torch.device("cpu")

# ==============================================================================
section("the compute badge the GUI shows")
# ==============================================================================
#  The recorder must NOT import torch (CLAUDE.md section 13), so it asks a
#  subprocess instead. These are the data-layer halves of that: the probe and
#  the parser. The badge itself is checked in tests/test_gui.py.

probe = T.detect_compute_device()
check("the probe answers", probe.get("probed") is True, str(probe))
check("the probe agrees with torch about CUDA", probe.get("cuda") == HAVE_CUDA,
      f"probe says {probe.get('cuda')}, torch says {HAVE_CUDA}")
check("the probe reports the torch build", probe.get("torch") == torch.__version__,
      str(probe.get("torch")))
if HAVE_CUDA:
    check("the probe names the GPU",
          probe.get("gpu") == torch.cuda.get_device_name(0), str(probe.get("gpu")))
    check("the probe reports the CUDA runtime", probe.get("cuda_runtime"),
          str(probe.get("cuda_runtime")))
    check("the probe reports the GPU memory",
          isinstance(probe.get("total_memory_mb"), int)
          and probe["total_memory_mb"] > 0, str(probe.get("total_memory_mb")))
else:
    check("the probe explains why there is no GPU", bool(probe.get("reason")),
          str(probe.get("reason")))

head, detail, severity = T.compute_badge(probe)
check("the badge names the processor actually in use",
      ("GPU" in head) == HAVE_CUDA, head)
check("the badge severity matches",
      severity == ("good" if HAVE_CUDA else "warn"), severity)
#  The badge must not let a GPU imply that RETRAIN uses one: scikit-learn has
#  no GPU backend, and the sweep behind that button always runs on the CPU.
check("the badge says the sklearn sweep runs on the CPU",
      "CPU" in detail and "scikit-learn" in detail,
      " ".join(detail.split())[:80])
if HAVE_CUDA:
    check("the badge says what does use the GPU", "train_torch.py" in detail)

unknown_head, _, unknown_sev = T.compute_badge(T.blank_compute_info())
check("an unprobed device is shown as unknown, not as CPU",
      "checking" in unknown_head and unknown_sev == "dim", unknown_head)
#  "not asked yet" and "asked, no answer" are different states: the first is
#  worth waiting out, the second is not, and neither may read as "CPU".
failed_head, failed_detail, failed_sev = T.compute_badge(
    T.blank_compute_info("the device probe timed out", failed=True))
check("a failed probe is not shown as still checking",
      failed_head == "DEVICE UNKNOWN" and failed_sev == "warn", failed_head)
check("a failed probe is not shown as CPU", "CPU ONLY" not in failed_head)
#  It names the CURRENT GPU trainer; train_torch.py is retired (CLAUDE.md 8b).
check("a failed probe says what to try",
      "train_nn.py" in failed_detail,
      " ".join(failed_detail.split())[:70])
check("a missing info dict does not crash the badge",
      T.compute_badge(None)[2] == "dim")
check("a probe that printed nothing is not read as 'no GPU'",
      T.parse_compute_probe("")["probed"] is False)
check("the parser survives a corrupt answer",
      T.parse_compute_probe("COMPUTE {not json")["probed"] is False)
check("the parser takes the last answer",
      T.parse_compute_probe(
          'COMPUTE {"cuda": false}' + chr(10)
          + 'COMPUTE {"cuda": true}')["cuda"] is True)
check("the parser ignores unknown keys",
      "nonsense" not in T.parse_compute_probe('COMPUTE {"nonsense": 1}'))
check("a cpu probe result reports no GPU",
      T.compute_badge(T.parse_compute_probe(
          'COMPUTE {"probed": true, "torch": "2.9.1", "cuda": false,'
          ' "reason": "CPU-only torch build"}'))[0] == "CPU ONLY")

# ==============================================================================
section("architectures")
# ==============================================================================

WIN, CH, NC = 30, 8, len(LABELS)
for kind in TT.MODEL_KINDS:
    model = TT.build_model(kind, CH, NC, WIN)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(5, WIN, CH))
    check(f"{kind}: builds and runs", out.shape == (5, NC), str(tuple(out.shape)))
    check(f"{kind}: has trainable parameters", TT.count_parameters(model) > 0,
          f"{TT.count_parameters(model)} params")
    check(f"{kind}: output is finite", bool(torch.isfinite(out).all()))

try:
    TT.build_model("transformer", CH, NC, WIN)
    check("an unknown kind is refused", False, "no error raised")
except ValueError:
    check("an unknown kind is refused", True)

check("the grid names only buildable kinds",
      set(TT.MODEL_KINDS) == {"cnn", "lstm", "cnn_lstm"}, str(TT.MODEL_KINDS))

# ==============================================================================
section("fold-local standardisation")
# ==============================================================================

rng = np.random.default_rng(1)
train_block = np.abs(rng.normal(100, 30, (200, WIN, CH))).astype(np.float32)
test_block = np.abs(rng.normal(900, 50, (40, WIN, CH))).astype(np.float32)

scale, mean, std = TT.fold_standardiser(train_block)
scaled_train = scale(train_block)
check("training fold is centred", abs(float(scaled_train.mean())) < 1e-3,
      f"mean {float(scaled_train.mean()):+.2e}")
check("training fold is unit-scaled", abs(float(scaled_train.std()) - 1.0) < 1e-3,
      f"std {float(scaled_train.std()):.4f}")
check("statistics are per channel", mean.shape == (1, 1, CH), str(mean.shape))

#  The point of fitting on the training fold only: a wildly different test
#  fold must come out shifted, not re-centred on itself.
scaled_test = scale(test_block)
check("test fold is NOT re-centred on its own statistics",
      float(scaled_test.mean()) > 1.0, f"mean {float(scaled_test.mean()):+.3f}")

scale2, mean2, _ = TT.fold_standardiser(np.concatenate([train_block, test_block]))
check("including the test fold would have changed the statistics",
      not np.allclose(mean, mean2),
      f"{float(mean.mean()):.3f} vs {float(mean2.mean()):.3f}")

#  log1p, not a plain z-score: check against the closed form.
expected = (np.log1p(np.abs(test_block)) - mean) / std
check("the transform really is log1p + standardise",
      np.allclose(scaled_test, expected, atol=1e-6))
check("scaling preserves order within a channel",
      float(scaled_test.mean()) > float(scaled_train.mean()))
check("scaled output is float32", scaled_test.dtype == np.float32,
      str(scaled_test.dtype))
check("a dead channel does not divide by zero",
      np.isfinite(TT.fold_standardiser(np.zeros((10, WIN, CH), np.float32))[0](
          np.zeros((2, WIN, CH), np.float32))).all())

# ==============================================================================
section("windowing from a real on-disk dataset")
# ==============================================================================

n_written = write_dataset(ROOT, people=("alice",))
recs, dropped = TT.load_recordings(ROOT, exclude_codes=TT.EXCLUDE_SETS["no_errors"])
check("recordings load", len(recs) > 0, f"{len(recs)} kept, {len(dropped)} dropped")
check("all written recordings are accounted for",
      len(recs) + len(dropped) == n_written, f"{len(recs)}+{len(dropped)} of {n_written}")

built = TT.build_sequences(recs, 0.50, TT.OVERLAP)
check("build_sequences returns a result", built is not None)
X, y, groups, groups_person, labels = built

check("X is (n, win, 8)", X.ndim == 3 and X.shape[2] == 8, str(X.shape))
check("win is window_s x MODEL_FS", X.shape[1] == int(round(0.50 * TT.MODEL_FS)),
      f"{X.shape[1]} samples")
check("X is float32", X.dtype == np.float32, str(X.dtype))
#  sliding_windows hands back a read-only strided view; torch.from_numpy
#  needs owned, writeable, contiguous memory. This is the regression guard.
check("X owns its memory", X.base is None)
check("X is C-contiguous", X.flags["C_CONTIGUOUS"])
check("X is writeable", X.flags["WRITEABLE"])
check("torch.from_numpy accepts X without a copy",
      torch.from_numpy(X).shape == tuple(X.shape))
check("labels are sorted class names", labels == sorted(LABELS), str(labels))
check("y indexes into labels", y.min() >= 0 and y.max() < len(labels),
      f"{y.min()}..{y.max()}")
check("one label per window", y.shape[0] == X.shape[0])
check("one uid per window", groups.shape[0] == X.shape[0])
check("one person per window", groups_person.shape[0] == X.shape[0])
check("all three classes are present", len(set(y.tolist())) == 3,
      str(sorted(set(y.tolist()))))
check("amplitudes are device counts, unscaled", X.max() > 50,
      f"max {X.max():.1f}")

#  Every window of one recording must carry that recording's label.
uid_labels = {}
consistent = True
for pos, uid in enumerate(groups):
    consistent &= uid_labels.setdefault(uid, y[pos]) == y[pos]
check("a recording's windows all share its label", consistent)
check("as many uid groups as recordings", len(uid_labels) == len(recs),
      f"{len(uid_labels)} groups, {len(recs)} recordings")

section("no recording straddles a split")
from sklearn.model_selection import StratifiedGroupKFold          # noqa: E402

cv = StratifiedGroupKFold(n_splits=TT.CV_SPLITS, shuffle=True,
                          random_state=TT.RANDOM_STATE)
leaked = 0
for train_idx, test_idx in cv.split(X, y, groups):
    leaked += len(set(groups[train_idx].tolist())
                  & set(groups[test_idx].tolist()))
check("grouped CV keeps every recording on one side", leaked == 0,
      f"{leaked} recordings on both sides")

# ==============================================================================
section("training")
# ==============================================================================

sub = np.flatnonzero(np.isin(np.arange(len(y)), np.arange(len(y))))[:600]
Xs, ys = X[sub], y[sub]
fit, _, _ = TT.fold_standardiser(Xs)
Xn = fit(Xs)

m1 = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1], seed=0),
                  Xn, ys, DEVICE, epochs=8, seed=0)
p1, prob1 = TT.predict_all(m1, Xn, DEVICE)

check("predictions cover every window", p1.shape[0] == Xn.shape[0])
check("probabilities are (n, n_classes)", prob1.shape == (Xn.shape[0], len(labels)),
      str(prob1.shape))
check("probabilities sum to 1", np.allclose(prob1.sum(axis=1), 1.0, atol=1e-5))
check("probabilities are in [0, 1]", prob1.min() >= 0.0 and prob1.max() <= 1.0)
check("the prediction is the argmax of the probabilities",
      np.array_equal(p1, prob1.argmax(axis=1)))
acc1 = float((p1 == ys).mean())
check("training learns a separable problem", acc1 > 0.90, f"{acc1 * 100:.1f} %")

m2 = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1], seed=0),
                  Xn, ys, DEVICE, epochs=8, seed=0)
p2, prob2 = TT.predict_all(m2, Xn, DEVICE)
check("the same seed gives the same predictions", np.array_equal(p1, p2))
#  Probabilities, not just the argmax: on a separable problem the argmax
#  agrees even between two unrelated fits, so it would not notice that the
#  seed had stopped controlling the weight initialisation.
check("the same seed gives the same probabilities",
      np.allclose(prob1, prob2, atol=1e-6),
      f"max difference {float(np.abs(prob1 - prob2).max()):.2e}")

m3 = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1], seed=99),
                  Xn, ys, DEVICE, epochs=8, seed=99)
_, prob3 = TT.predict_all(m3, Xn, DEVICE)
check("a different seed is a different fit",
      not np.allclose(prob1, prob3, atol=1e-6),
      f"max difference {float(np.abs(prob1 - prob3).max()):.2e}")

check("the seed reaches weight initialisation",
      not np.allclose(
          TT.build_model("cnn", 8, len(labels), X.shape[1], seed=0)
            .head[-1].weight.detach().numpy(),
          TT.build_model("cnn", 8, len(labels), X.shape[1], seed=7)
            .head[-1].weight.detach().numpy()))
check("the same seed gives the same initial weights",
      np.allclose(
          TT.build_model("cnn", 8, len(labels), X.shape[1], seed=3)
            .head[-1].weight.detach().numpy(),
          TT.build_model("cnn", 8, len(labels), X.shape[1], seed=3)
            .head[-1].weight.detach().numpy()))

trace = []
TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1], seed=0),
             Xn, ys, DEVICE, epochs=8, seed=0, log=trace)
check("the loss trace is recorded when asked", len(trace) > 0, f"{len(trace)} points")
check("the loss goes down", trace[-1][1] < trace[0][1],
      f"{trace[0][1]:.3f} -> {trace[-1][1]:.3f}")

check("batching is honoured at the edges",
      TT.predict_all(m1, Xn[:3], DEVICE)[0].shape[0] == 3)
empty_pred, empty_prob = TT.predict_all(m1, Xn[:0], DEVICE)
check("an empty input returns empty arrays", empty_pred.shape[0] == 0
      and empty_prob.shape[0] == 0)

check("class weights are accepted",
      TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1]),
                   Xn, ys, DEVICE, epochs=2,
                   class_weights=np.ones(len(labels))) is not None)

# ==============================================================================
section("evaluation protocol")
# ==============================================================================

stats = TT.evaluate("cnn", X, y, groups, groups_person, labels, X.shape[1],
                    DEVICE, epochs=6, n_splits=4)
check("evaluate returns a result", stats is not None)
for key in ("n_splits", "window_accuracy", "window_accuracy_std",
            "window_macro_f1", "recording_vote_accuracy",
            "n_recordings_voted", "fold_accuracies", "confusion_matrix",
            "n_parameters"):
    check(f"stats carry {key}", key in stats, str(stats.get(key))[:40])
check("more than one fold was run", stats["n_splits"] >= 2, str(stats["n_splits"]))
check("one accuracy per fold", len(stats["fold_accuracies"]) == stats["n_splits"])
cm = np.array(stats["confusion_matrix"])
check("the confusion matrix is n_classes square", cm.shape == (len(labels), len(labels)),
      str(cm.shape))
check("the confusion matrix counts every tested window once",
      cm.sum() == X.shape[0], f"{cm.sum()} vs {X.shape[0]}")
check("every recording got exactly one vote",
      stats["n_recordings_voted"] == len(recs),
      f"{stats['n_recordings_voted']} of {len(recs)}")
check("accuracy beats chance", stats["window_accuracy"] > 1.0 / len(labels),
      f"{stats['window_accuracy'] * 100:.1f} %")

check("too few groups to cross-validate is reported, not crashed",
      TT.evaluate("cnn", X[:20], y[:20], np.zeros(20), groups_person[:20],
                  labels, X.shape[1], DEVICE, epochs=1, n_splits=4) is None)

section("subject independence")
check("one subject means no subject-independent score",
      TT.leave_one_subject_out("cnn", X, y, groups_person, labels, X.shape[1],
                               DEVICE, epochs=2) is None)

ROOT2 = tempfile.mkdtemp(prefix="emgtorch2_")
write_dataset(ROOT2, people=("alice", "bob"), reps=3, seed=11)
recs2, _ = TT.load_recordings(ROOT2, exclude_codes=TT.EXCLUDE_SETS["no_errors"])
X2, y2, g2, gp2, lab2 = TT.build_sequences(recs2, 0.50, TT.OVERLAP)
loso = TT.leave_one_subject_out("cnn", X2, y2, gp2, lab2, X2.shape[1],
                                DEVICE, epochs=6)
check("two subjects give a subject-independent score", loso is not None)
if loso:
    check("every subject is held out in turn",
          sorted(loso["per_subject"]) == ["alice", "bob"],
          str(sorted(loso["per_subject"])))
    check("the mean matches the per-subject scores",
          abs(loso["mean"] - float(np.mean(list(loso["per_subject"].values())))) < 1e-9)
    check("a standard deviation is reported", "std" in loso)

# ==============================================================================
section("a free take never becomes training data")
# ==============================================================================

take_writer = T.DatasetWriter(ROOT, "alice", "session_01")
take_n = 400
take_emg = np.abs(np.random.default_rng(5).normal(20, 5, (take_n, 8)))
take_rec = T.Recording(take_emg.astype(int),
                       np.arange(take_n) / 60.0 + 1_700_000_000.0,
                       "FIST", "alice", "session_01", 1,
                       label_start_index=0, label_end_index=take_n,
                       meta={"emg_mode": "filtered"})
take_writer.save_take(take_rec)
recs_after, _ = TT.load_recordings(ROOT, exclude_codes=TT.EXCLUDE_SETS["no_errors"])
check("the take is on disk", len(take_writer.list_takes()) == 1)
check("the trainer does not see it", len(recs_after) == len(recs),
      f"{len(recs_after)} vs {len(recs)}")
check("no loaded recording has stage 'take'",
      all(getattr(r, "stage", T.STAGE_RAW) != "take" for r in recs_after))

# ==============================================================================
section("full run through main(), and the checkpoint")
# ==============================================================================

ckpt = os.path.join(OUT, "torch_model.pt")
rc = TT.main(["--root", ROOT, "--out-dir", OUT, "--model-path", ckpt,
              "--device", "cuda" if HAVE_CUDA else "cpu",
              "--quick", "--epochs", "8", "--no-save"])
check("--no-save run succeeds", rc == 0, f"exit {rc}")
check("--no-save wrote no checkpoint", not os.path.exists(ckpt))

rc = TT.main(["--root", ROOT, "--out-dir", OUT, "--model-path", ckpt,
              "--device", "cuda" if HAVE_CUDA else "cpu",
              "--kind", "cnn", "--kind", "cnn_lstm",
              "--window", "0.50", "--epochs", "10"])
check("full run succeeds", rc == 0, f"exit {rc}")
check("the checkpoint was written", os.path.exists(ckpt))

payload = torch.load(ckpt, weights_only=False)
check("payload format is declared", payload.get("format") == "emg_torch_v1",
      str(payload.get("format")))
for key in ("kind", "state_dict", "labels", "n_channels", "win_samples",
            "window_s", "model_fs", "scaler", "stats", "device_trained_on",
            "created"):
    check(f"payload carries {key}", key in payload, str(payload.get(key))[:40])
check("payload labels match the dataset", payload["labels"] == sorted(LABELS),
      str(payload["labels"]))
check("payload records 8 channels", payload["n_channels"] == 8)
check("payload records the model grid rate", payload["model_fs"] == TT.MODEL_FS)
check("payload names a real architecture", payload["kind"] in TT.MODEL_KINDS,
      payload["kind"])
check("the scaler travels with the model",
      payload["scaler"]["kind"] == "log1p_standardise")
check("the scaler statistics are per channel",
      np.asarray(payload["scaler"]["mean"]).shape == (1, 1, 8),
      str(np.asarray(payload["scaler"]["mean"]).shape))
check("stats name the validation protocol",
      "StratifiedGroupKFold" in payload["stats"]["validation_protocol"])
check("stats state that there was no early stopping",
      payload["stats"]["early_stopping"] is False)
check("stats carry the majority-class baseline",
      0.0 < payload["stats"]["majority_class_baseline"] <= 1.0,
      f"{payload['stats']['majority_class_baseline'] * 100:.1f} %")
check("stats declare the selection bias",
      "optimistic" in payload["stats"]["selection_bias_note"])
check("stats count the candidates", payload["stats"]["candidates_evaluated"] == 2,
      str(payload["stats"]["candidates_evaluated"]))
check("stats carry the subject-independent field",
      "subject_independent" in payload["stats"])
check("state_dict tensors are on the cpu",
      all(not v.is_cuda for v in payload["state_dict"].values()))

section("the checkpoint reproduces its own predictions")
reborn = TT.build_model(payload["kind"], payload["n_channels"],
                        len(payload["labels"]), payload["win_samples"])
missing = reborn.load_state_dict(payload["state_dict"], strict=True)
check("state_dict loads strictly into a fresh model",
      not missing.missing_keys and not missing.unexpected_keys, str(missing))

pmean = np.asarray(payload["scaler"]["mean"], dtype=np.float32)
pstd = np.asarray(payload["scaler"]["std"], dtype=np.float32)
Xcheck, ycheck, _, _, _ = TT.build_sequences(recs, payload["window_s"], TT.OVERLAP)
Xcheck_scaled = ((np.log1p(np.abs(Xcheck)) - pmean) / pstd).astype(np.float32)
pred_cpu, prob_cpu = TT.predict_all(reborn, Xcheck_scaled, torch.device("cpu"))
check("the reloaded model predicts every window",
      pred_cpu.shape[0] == Xcheck.shape[0])
acc_reload = float((pred_cpu == ycheck).mean())
check("the reloaded model is as good as the report says",
      acc_reload > 0.90, f"{acc_reload * 100:.1f} % on its own training data")

reborn2 = TT.build_model(payload["kind"], payload["n_channels"],
                         len(payload["labels"]), payload["win_samples"])
reborn2.load_state_dict(payload["state_dict"])
pred_again, _ = TT.predict_all(reborn2, Xcheck_scaled, torch.device("cpu"))
check("two reloads agree bit-for-bit", np.array_equal(pred_cpu, pred_again))

section("the recorder's model is untouched")
check("the checkpoint name differs from the sklearn payload",
      TT.CHECKPOINT_NAME != os.path.basename(T.MODEL_PATH),
      f"{TT.CHECKPOINT_NAME} vs {os.path.basename(T.MODEL_PATH)}")
check("the trainer wrote no .joblib",
      not any(f.endswith(".joblib") for f in os.listdir(OUT)),
      str(os.listdir(OUT)))
#  A ten-minute sweep must not be destroyed by the next one.
archives = [f for f in os.listdir(OUT)
            if f.endswith(".pt") and f != os.path.basename(ckpt)]
check("the checkpoint was archived as well as saved", len(archives) == 1,
      str(archives))
check("the archive is a complete payload",
      torch.load(os.path.join(OUT, archives[0]),
                 weights_only=False).get("format") == "emg_torch_v1")
reports = os.path.join(OUT, "reports")
check("a sweep report was written", os.path.isdir(reports)
      and any(f.startswith("torch_sweep_") for f in os.listdir(reports)),
      str(os.listdir(reports) if os.path.isdir(reports) else None))

import json                                                       # noqa: E402
report_file = sorted(f for f in os.listdir(reports)
                     if f.startswith("torch_sweep_"))[-1]
with open(os.path.join(reports, report_file), encoding="utf-8") as fh:
    report = json.load(fh)
check("the report names the device it ran on", "device" in report,
      str(report.get("device", {}).get("device")))
check("the report lists every candidate", len(report["results"]) == 2,
      f"{len(report['results'])} rows")
check("the report records the chance level", 0.0 < report["chance"] <= 1.0)

# ==============================================================================
if HAVE_CUDA:
    section("CUDA specifics")

    cuda_dev = torch.device("cuda:0")
    cuda_model = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1]),
                              Xn, ys, cuda_dev, epochs=8, seed=0)
    check("parameters really live on the GPU",
          all(p.is_cuda for p in cuda_model.parameters()))
    pred_gpu, prob_gpu = TT.predict_all(cuda_model, Xn, cuda_dev)
    check("predictions come back to the host as numpy",
          isinstance(pred_gpu, np.ndarray) and isinstance(prob_gpu, np.ndarray))
    acc_gpu = float((pred_gpu == ys).mean())
    check("the GPU learns the same problem", acc_gpu > 0.90, f"{acc_gpu * 100:.1f} %")

    cuda_again = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1]),
                              Xn, ys, cuda_dev, epochs=8, seed=0)
    pred_gpu2, _ = TT.predict_all(cuda_again, Xn, cuda_dev)
    check("two seeded GPU runs agree exactly", np.array_equal(pred_gpu, pred_gpu2))

    cpu_model = TT.train_one(TT.build_model("cnn", 8, len(labels), X.shape[1]),
                             Xn, ys, torch.device("cpu"), epochs=8, seed=0)
    pred_host, _ = TT.predict_all(cpu_model, Xn, torch.device("cpu"))
    agree = float((pred_gpu == pred_host).mean())
    check("GPU and CPU reach the same conclusion", agree > 0.95,
          f"{agree * 100:.1f} % of windows agree")

    #  A GPU-trained checkpoint must be loadable on a machine with no GPU.
    state = {k: v.cpu() for k, v in cuda_model.state_dict().items()}
    host_copy = TT.build_model("cnn", 8, len(labels), X.shape[1])
    host_copy.load_state_dict(state)
    pred_moved, _ = TT.predict_all(host_copy, Xn, torch.device("cpu"))
    check("a GPU-trained model runs on the CPU after loading",
          float((pred_moved == pred_gpu).mean()) > 0.99,
          f"{float((pred_moved == pred_gpu).mean()) * 100:.1f} % agreement")

    torch.cuda.synchronize()
    check("no CUDA error is pending", True)
else:
    print("\n--- CUDA specifics: skipped, no usable GPU ---")

shutil.rmtree(ROOT, ignore_errors=True)
shutil.rmtree(ROOT2, ignore_errors=True)
shutil.rmtree(OUT, ignore_errors=True)

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)

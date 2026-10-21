"""
The neural network: the SEPARATION rule first, then the machinery.

What matters most here is not that the net works -- it is that it stays out
of the deployed answer. CLAUDE.md section 8L makes three promises, and the
first block of checks is what keeps them from quietly becoming false:

    1. the application never imports torch, with the net enabled;
    2. the net's output is a different TYPE from the classifier's, so it
       cannot be appended to the classifier's history, handed to the decision
       panel or drawn in its probability bars by one misplaced line;
    3. nothing in the sklearn path reads the checkpoint, and nothing in the
       neural path writes to the predictor.

Everything else -- the protocol, the engine's state machine, the refusals,
the architectures -- is checked after that.

Most of this runs with NO torch and NO armband: the engine is driven against
a FAKE worker script that speaks the same line protocol in twenty lines of
pure Python. The torch-dependent checks are skipped with a message when
torch is absent, and the end-to-end check against the real worker is skipped
when no checkpoint has been trained yet.

Run from the project root:

    python tests\\test_nn.py
"""
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emg_dataset_tool as T                                       # noqa: E402

OK = []
SKIPPED = []
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def skip(name, why):
    SKIPPED.append(name)
    print(f"SKIP  {name}  [{why}]")


def section(title):
    print(f"\n--- {title} ---")


def source_of(obj):
    return inspect.getsource(obj)


def code_only(source):
    """
    The same source with every comment and string literal removed.

    The separation checks below ask "does this code touch that object", and
    the answer has to come from the code. Matching raw text instead reports a
    violation every time a docstring explains the rule it is enforcing --
    "never touches `self.prob_bars`" is a promise, not a breach. Comments and
    strings are therefore tokenised out first, and the few checks that really
    are about the prose read the raw text deliberately.
    """
    import io
    import tokenize
    kept = []
    try:
        for token in tokenize.generate_tokens(
                io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    except (tokenize.TokenError, IndentationError):
        #  A method's source alone is not a valid module (it is indented), so
        #  tokenize can stop early. Dedent and retry before giving up.
        import textwrap
        if not source.startswith(("def ", "class ")):
            return code_only(textwrap.dedent(source))
        return source
    return " ".join(kept)


def tight(source):
    """
    `code_only()` with all whitespace removed.

    Needed because the tokeniser joins tokens with spaces, so a dotted
    attribute path comes back as `self . predictor . last` and a plain
    substring test for `self.predictor.last` never matches. The checks that
    look for a specific attribute ACCESS use this; the ones that look for a
    bare identifier do not need it.
    """
    return "".join(code_only(source).split())


# ==============================================================================
section("1. THE SEPARATION RULE  (section 8L)")
# ==============================================================================

check("the net is enabled, so these checks mean something",
      T.ENABLE_NEURAL_NET, f"ENABLE_NEURAL_NET={T.ENABLE_NEURAL_NET}")

app_source = open(os.path.join(PROJECT, "emg_dataset_tool.py"),
                  encoding="utf-8").read()
import_lines = [line.strip() for line in app_source.splitlines()
                if line.strip().startswith(("import ", "from "))]
check("the application has no torch import at all",
      not any("torch" in line for line in import_lines),
      "; ".join(line for line in import_lines if "torch" in line) or "none")
check("'torch' survives in the module only as text (the probe snippet)",
      "torch" in app_source)
check("torch is not in sys.modules after importing the application",
      "torch" not in sys.modules)

#  The type separation. Two dataclasses with nearly the same fields, and that
#  is exactly why they must not be interchangeable.
check("NeuralPrediction is not Prediction",
      T.NeuralPrediction is not T.Prediction)
check("NeuralPrediction does not subclass Prediction",
      not issubclass(T.NeuralPrediction, T.Prediction))
check("Prediction does not subclass NeuralPrediction",
      not issubclass(T.Prediction, T.NeuralPrediction))
check("only the neural one carries a measured latency",
      "latency_ms" in T.NeuralPrediction.__dataclass_fields__
      and "latency_ms" not in T.Prediction.__dataclass_fields__)

#  The code separation, read off the source rather than asserted in prose.
live_source = code_only(source_of(T.LivePredictor))
check("LivePredictor never mentions the neural engine",
      "NeuralEngine" not in live_source and "nn_engine" not in live_source)
check("LivePredictor never mentions the neural checkpoint",
      "NN_MODEL_PATH" not in live_source and "gesture_nn" not in live_source)
check("LivePredictor never produces a NeuralPrediction",
      "NeuralPrediction" not in live_source)

engine_source = code_only(source_of(T.NeuralEngine))
engine_tight = tight(source_of(T.NeuralEngine))
check("NeuralEngine never touches LivePredictor",
      "LivePredictor" not in engine_source
      and "self.predictor" not in engine_tight)
check("NeuralEngine never touches the decision panel",
      "DecisionPanel" not in engine_source
      and "decision_panel" not in engine_source)
check("NeuralEngine never produces a Prediction",
      "Prediction(" not in engine_source.replace("NeuralPrediction(", ""))
check("NeuralEngine never calls model_feature_vector -- a net learns its own",
      "model_feature_vector" not in engine_source)

panel_source = code_only(source_of(T.DecisionPanel))
check("the decision panel knows nothing about the net",
      not any(word in panel_source for word in
              ("NeuralEngine", "NeuralPrediction", "nn_engine",
               "gesture_nn")))

trail_source = code_only(source_of(T.TrailStep)
                         + source_of(T.DecisionTrailWidget))
check("the DECISION TRAIL's fusion has no neural member",
      "Neural" not in trail_source and "nn_" not in trail_source)

#  The GUI separation: the net's draw routine must write only to its own
#  widgets, and the classifier's must write only to its own.
nn_draw = code_only(source_of(T.MainWindow._show_nn_prediction))
nn_draw_tight = tight(source_of(T.MainWindow._show_nn_prediction))
check("_show_nn_prediction never writes the classifier's bars",
      "self.prob_bars" not in nn_draw_tight)
check("_show_nn_prediction never claims the graph overlay",
      "set_prediction_overlay" not in nn_draw)
check("_show_nn_prediction never appends to the classifier's history",
      "predictor.history" not in nn_draw_tight
      and "predictor.prob_history" not in nn_draw_tight)
check("_show_nn_prediction reads the classifier only to COMPARE",
      "self.predictor.last" in nn_draw_tight
      and "self.predictor.predict" not in nn_draw_tight)

sk_draw = code_only(source_of(T.MainWindow._show_prediction))
check("_show_prediction (deployed) never writes the net's bars",
      "nn_prob_bars" not in sk_draw and "nn_label" not in sk_draw)

sk_run = code_only(source_of(T.MainWindow._run_prediction))
check("the classifier's tick never consults the net",
      "nn_engine" not in sk_run)

nn_tick = tight(source_of(T.MainWindow._tick_neural))
check("the net's tick never consults the classifier's predictor",
      "self.predictor" not in nn_tick)

#  The trainers and the on-disk artefacts.
train_model_source = open(os.path.join(PROJECT, "train_model.py"),
                          encoding="utf-8").read()
train_model_code = code_only(train_model_source)
check("train_model.py never reads the neural checkpoint",
      "gesture_nn" not in train_model_code
      and "nn_serve" not in train_model_code
      and "nn_arch" not in train_model_code)
check("the two models are written to different files",
      os.path.splitext(T.MODEL_PATH)[0] != os.path.splitext(
          T.NN_MODEL_PATH)[0],
      f"{T.MODEL_PATH} vs {T.NN_MODEL_PATH}")
check("the checkpoint is not a joblib payload LivePredictor could load",
      T.NN_MODEL_PATH.endswith(".pt") and T.MODEL_PATH.endswith(".joblib"))

train_nn_source = open(os.path.join(PROJECT, "train_nn.py"),
                       encoding="utf-8").read()
check("train_nn.py says in words that its model does not vote",
      "DOES NOT VOTE" in train_nn_source)
check("train_nn.py imports the shared protocol instead of copying it",
      "TM.dedupe_windows" in train_nn_source
      and "TM.resolve_grouping" in train_nn_source
      and "TM.load_recordings" in train_nn_source)
check("train_nn.py never writes the sklearn model path",
      "gesture_model.joblib" not in code_only(train_nn_source))


# ==============================================================================
section("2. THE INPUT CONTRACT  (nn_input_block)")
# ==============================================================================

block = np.array([[100.0, 200.0, 50.0, 10.0, 30.0, 40.0, 60.0, 70.0],
                  [120.0, 180.0, 55.0, 12.0, 28.0, 44.0, 66.0, 72.0]])

plain = T.nn_input_block(block, None)
check("with no calibration the samples are unchanged",
      np.allclose(plain, block), f"max diff "
      f"{float(np.max(np.abs(plain - block))):.3g}")
check("the result is float32 -- what torch.from_numpy wants",
      plain.dtype == np.float32, str(plain.dtype))
check("the result is contiguous and owns its memory",
      plain.flags["C_CONTIGUOUS"] and plain.flags["WRITEABLE"])

record = T.calibration_record(
    rest_levels=[20] * 8, effort_levels=[600, 700, 200, 250, 900, 950, 400,
                                         450],
    person_id="tester", n_rest=200, n_effort=200, fs=50.0)
resolved = T.resolve_calibration(record, "affine")
check("the fixture calibration validates", resolved is not None)
calibrated = T.nn_input_block(block, resolved)
check("a calibration really does change the numbers",
      not np.allclose(calibrated, block))
check("it is exactly apply_calibration, not a second implementation",
      np.allclose(calibrated, T.apply_calibration(block, resolved)))
check("nn_input_block does no log1p or standardisation -- that is the "
      "checkpoint's scaler",
      "log1p" not in code_only(source_of(T.nn_input_block)))
check("nn_input_block never looks at a label",
      "label" not in code_only(source_of(T.nn_input_block)))
check("the two contracts are separate functions",
      T.nn_input_block is not T.model_feature_vector)


# ==============================================================================
section("3. THE LINE PROTOCOL")
# ==============================================================================

window = np.arange(240, dtype=np.float32).reshape(30, 8)
line = T.nn_encode_request(42, window)
check("a request is exactly one line", line.count("\n") == 1
      and line.endswith("\n"))
decoded = json.loads(line)
check("the request carries its id", decoded["id"] == 42)
check("the request carries the whole window",
      np.asarray(decoded["window"]).shape == (30, 8))
check("the window survives the round trip to 4 decimals",
      np.allclose(np.asarray(decoded["window"]), window, atol=1e-3))

for prefix, kind, payload in (
    ("NNMETA ", "meta", {"ok": True, "labels": ["A", "B"]}),
    ("NNOUT ", "out", {"id": 1, "probs": [0.2, 0.8]}),
    ("NNERR ", "err", {"id": 1, "error": "boom"}),
):
    got_kind, got = T.nn_parse_line(prefix + json.dumps(payload))
    check(f"a {kind} line parses as {kind}", got_kind == kind, str(got_kind))
    check(f"the {kind} payload survives", got == payload)

check("NNBYE parses as bye", T.nn_parse_line("NNBYE")[0] == "bye")
check("NNBYE with a newline still parses", T.nn_parse_line("NNBYE\n")[0]
      == "bye")
check("anything else is kept as text, not dropped",
      T.nn_parse_line("CUDA warning: whatever") == (
          "text", {"text": "CUDA warning: whatever"}))
check("a malformed protocol line becomes an error, never an exception",
      T.nn_parse_line("NNOUT {not json")[0] == "err")
check("a non-object protocol payload becomes an error",
      T.nn_parse_line("NNOUT [1,2,3]")[0] == "err")
check("an empty line is text, not a crash",
      T.nn_parse_line("")[0] == "text")
check("None is tolerated", T.nn_parse_line(None)[0] == "text")


# ==============================================================================
section("4. THE SIDECAR  (statistics without torch)")
# ==============================================================================

check("the sidecar sits beside the checkpoint with a .json suffix",
      T.nn_stats_path("models/gesture_nn.pt") == "models/gesture_nn.json")
check("it follows a redirected model path",
      T.nn_stats_path("/tmp/other/net.pt") == "/tmp/other/net.json")

TEMP = tempfile.mkdtemp(prefix="emgnn_")
missing = os.path.join(TEMP, "nothing.pt")
check("a missing sidecar is None, not an exception",
      T.load_nn_stats(missing) is None)

broken = os.path.join(TEMP, "broken.pt")
with open(T.nn_stats_path(broken), "w", encoding="utf-8") as fh:
    fh.write("{not json at all")
check("an unreadable sidecar is None, not an exception",
      T.load_nn_stats(broken) is None)

FAKE_LABELS = ["REST", "FIST", "OPEN"]
FAKE_WIN = 12
fake_model = os.path.join(TEMP, "fake.pt")
with open(fake_model, "w", encoding="utf-8") as fh:
    fh.write("not a real checkpoint -- the fake worker never opens it")
FAKE_SIDECAR = {
    "format": "emg_nn_v1", "kind": "tcn", "labels": FAKE_LABELS,
    "win_samples": FAKE_WIN, "window_s": 0.2, "model_fs": 60.0,
    "architecture": "fake", "created_iso": T.iso_time(0),
    "calibration": {"mode": "off", "requested": "off", "subjects": []},
    "data": {"n_recordings": 9, "uid_digest": "deadbeef",
             "labels": {"REST": 3, "FIST": 3, "OPEN": 3},
             "subjects": ["tester"], "sessions": ["tester/session_01"],
             "newest_stamp": "20260101-000000-000"},
    "stats": {"model_name": "NN-tcn", "kind": "tcn", "window_accuracy": 0.71,
              "recording_vote_accuracy": 0.75, "window_macro_f1": 0.68,
              "grouping": "subject", "n_windows": 400, "win_samples": FAKE_WIN,
              "labels": FAKE_LABELS, "majority_class_baseline": 0.34,
              "n_parameters": 3943, "epochs": 80, "n_splits": 2,
              "fold_accuracies": [0.7, 0.72], "deduped": True,
              "confusion_matrix": [[30, 2, 1], [3, 28, 2], [1, 4, 29]],
              "dc_only_control": {"window_accuracy": 0.52,
                                  "grouping": "subject"},
              "candidates": [{"kind": "tcn", "window_s": 0.2,
                              "window_accuracy": 0.71,
                              "recording_vote_accuracy": 0.75,
                              "window_macro_f1": 0.68, "n_parameters": 3943}]},
}
with open(T.nn_stats_path(fake_model), "w", encoding="utf-8") as fh:
    json.dump(FAKE_SIDECAR, fh)
loaded = T.load_nn_stats(fake_model)
check("a written sidecar loads back", isinstance(loaded, dict))
check("its stats survive", loaded["stats"]["window_accuracy"] == 0.71)


# ==============================================================================
section("5. THE ENGINE, against a FAKE worker (no torch)")
# ==============================================================================

#  A worker in pure Python: same protocol, no torch, no GPU. It answers with
#  a deterministic function of the window so the engine's bookkeeping can be
#  checked exactly -- and it proves the protocol, not the model.
FAKE_WORKER = os.path.join(TEMP, "fake_worker.py")
with open(FAKE_WORKER, "w", encoding="utf-8") as fh:
    fh.write('''
import json, sys, argparse
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--device", default="auto")
a = p.parse_args()
labels = {labels!r}
print("some noise torch would print", flush=True)
print("NNMETA " + json.dumps({{
    "ok": True, "labels": labels, "win_samples": {win},
    "n_channels": 8, "window_s": 0.2, "model_fs": 60.0, "kind": "tcn",
    "n_parameters": 3943, "calibration": {{"mode": "off"}},
    "device": {{"device": "cpu", "torch": "fake", "reason": "fake worker"}},
    "stats": {{"window_accuracy": 0.71}}}}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("cmd") == "quit":
        break
    w = r.get("window")
    if not isinstance(w, list) or len(w) != {win}:
        print("NNERR " + json.dumps({{"id": r.get("id"),
              "error": "bad window length", "fatal": False}}), flush=True)
        continue
    #  Channel 2 louder than channel 0 -> FIST, otherwise REST.
    tot = [sum(row[c] for row in w) for c in range(8)]
    probs = [0.1, 0.8, 0.1] if tot[2] > tot[0] else [0.8, 0.1, 0.1]
    print("NNOUT " + json.dumps({{"id": r.get("id"), "probs": probs,
          "ms": 1.25}}), flush=True)
print("NNBYE", flush=True)
'''.format(labels=FAKE_LABELS, win=FAKE_WIN))


def wait_for(engine, predicate, timeout=20.0):
    """Poll the engine until `predicate(engine)`; returns what happened."""
    deadline = time.perf_counter() + timeout
    produced = []
    while time.perf_counter() < deadline:
        produced.extend(engine.poll())
        if predicate(engine):
            return produced
        time.sleep(0.02)
    return produced


engine = T.NeuralEngine(model_path=fake_model, script=FAKE_WORKER,
                        device="cpu", smoothing=3)
check("a fresh engine is stopped and says so",
      engine.state == "stopped" and not engine.ready)
check("it can see that a checkpoint exists", engine.trained())
check("the device badge does not guess before the worker answers",
      engine.device_badge()[0] == "NN DEVICE --")

started = engine.start()
check("the worker starts", started, engine.status)
check("it reports 'starting', not 'ready'", engine.state == "starting",
      engine.state)
wait_for(engine, lambda e: e.state in ("ready", "error"))
check("it becomes ready", engine.ready, engine.status)
check("it learned the worker's labels", engine.labels == FAKE_LABELS,
      str(engine.labels))
check("it learned the window length", engine.win_samples == FAKE_WIN,
      str(engine.win_samples))
check("the device badge now reports what the WORKER said",
      engine.device_badge()[0] == "NN ON CPU", engine.device_badge()[0])
check("non-protocol worker output is kept as diagnostics",
      any("noise" in line for line in engine.worker_output()),
      str(engine.worker_output()))

#  A FIST-shaped window, at an awkward sample rate so the resampling runs.
n = 19
t = np.arange(n) / 43.0
fist = np.tile(np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
               (n, 1))
rest = np.tile(np.array([12] * 8, dtype=float), (n, 1))

check("submitting a window is accepted", engine.submit(fist, t))
check("it counts as in flight", engine.stats_summary()["in_flight"] == 1)
check("a second window is DROPPED while the first is outstanding",
      not engine.submit(fist, t))
check("the drop is counted, not hidden",
      engine.stats_summary()["dropped"] == 1)

produced = wait_for(engine, lambda e: e.last is not None)
check("an answer comes back", len(produced) == 1, str(len(produced)))
first = produced[0] if produced else None
check("it is a NeuralPrediction", isinstance(first, T.NeuralPrediction))
check("it is not a Prediction", not isinstance(first, T.Prediction))
if first is not None:
    check("it recognised the window the fake worker was built to recognise",
          first.label == "FIST", str(first.label))
    check("the probabilities are the worker's own",
          np.allclose(first.raw_probabilities, [0.1, 0.8, 0.1]))
    check("the labels come with it", first.labels == FAKE_LABELS)
    check("the round trip was measured, not invented",
          first.latency_ms > 0 and not np.isnan(first.latency_ms),
          f"{first.latency_ms:.1f} ms")
    check("the forward-pass time is the worker's own figure",
          abs(first.infer_ms - 1.25) < 1e-6, f"{first.infer_ms}")
    check("the first answer is not smoothed over anything yet",
          first.smoothed_over == 1, str(first.smoothed_over))
    check("the window size reported is the net's own",
          first.n_samples == FAKE_WIN)

check("the in-flight slot is released once answered",
      engine.stats_summary()["in_flight"] == 0)

#  Smoothing: two REST windows after one FIST should pull the smoothed answer
#  across while the raw answer flips immediately.
for _ in range(2):
    engine.submit(rest, t)
    wait_for(engine, lambda e: e.stats_summary()["in_flight"] == 0)
last = engine.last
check("the smoothed answer averages the recent windows",
      last is not None and last.smoothed_over == 3, str(last.smoothed_over))
check("the raw answer follows the newest window immediately",
      last.raw_label == "REST", str(last.raw_label))
check("the smoothed answer is the mean of the stored vectors, not a "
      "majority vote",
      abs(last.probabilities[0] - (0.8 + 0.8 + 0.1) / 3) < 1e-9,
      str(last.probabilities))
check("the displayed confidence matches the displayed label",
      abs(last.confidence - max(last.probabilities)) < 1e-12)
check("the smoothing window is the engine's own, not the classifier's",
      last.smoothed_over <= 3)

answered = engine.stats_summary()["answered"]
check("every answer was counted", answered == 3, str(answered))

#  A wrong-sized window: the worker refuses it and the engine survives.
engine.win_samples = FAKE_WIN + 5           # lie to it on purpose
engine.submit(fist, t)
wait_for(engine, lambda e: e.counters["errors"] > 0, timeout=5.0)
check("a window the worker refuses raises an error count",
      engine.counters["errors"] >= 1)
check("...but does not kill the engine -- `fatal: false`",
      engine.ready, f"{engine.state}: {engine.status}")
engine.win_samples = FAKE_WIN

engine.reset()
check("reset clears the smoothing history but keeps the worker",
      engine.last is None and engine.ready)

engine.stop()
check("stop leaves it stopped", engine.state == "stopped", engine.state)
check("the worker process is gone", engine._process is None)
check("submitting after stop is refused, not queued",
      not engine.submit(fist, t))


# ==============================================================================
section("6. THE REFUSALS")
# ==============================================================================

absent = T.NeuralEngine(model_path=os.path.join(TEMP, "none.pt"),
                        script=FAKE_WORKER, device="cpu")
check("with no checkpoint it refuses to start", not absent.start())
check("...and names train_nn.py in the reason",
      "train_nn.py" in absent.status, absent.status)
check("...and stays stopped rather than going to error",
      absent.state == "stopped", absent.state)

noscript = T.NeuralEngine(model_path=fake_model,
                          script=os.path.join(TEMP, "missing_worker.py"),
                          device="cpu")
check("with no worker script it refuses to start", not noscript.start())
check("...and says which script is missing",
      "missing_worker.py" in noscript.status, noscript.status)

#  A worker that reports ok:false must put the engine in error, with the
#  worker's own sentence -- not a generic failure.
REFUSING = os.path.join(TEMP, "refusing_worker.py")
with open(REFUSING, "w", encoding="utf-8") as fh:
    fh.write('''
import json, sys, argparse
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--device", default="auto")
p.parse_args()
print("NNMETA " + json.dumps({"ok": False,
      "error": "RuntimeError: CUDA was required but is not available"}),
      flush=True)
''')
refusing = T.NeuralEngine(model_path=fake_model, script=REFUSING,
                          device="cuda")
refusing.start()
wait_for(refusing, lambda e: e.state == "error")
check("a worker that refuses the checkpoint puts the engine in error",
      refusing.state == "error", refusing.state)
check("...carrying the worker's own sentence",
      "CUDA was required" in refusing.status, refusing.status)
check("an engine in error is not ready", not refusing.ready)
refusing.stop_quietly()
check("stop_quietly does not overwrite the error",
      refusing.state == "error" and "CUDA" in refusing.status)

#  A worker that dies before reporting anything.
DYING = os.path.join(TEMP, "dying_worker.py")
with open(DYING, "w", encoding="utf-8") as fh:
    fh.write('import sys\nprint("ImportError: no module named torch")\n'
             'sys.exit(1)\n')
dying = T.NeuralEngine(model_path=fake_model, script=DYING, device="cpu")
dying.start()
wait_for(dying, lambda e: e.state == "error")
check("a worker that exits before it is ready is an error",
      dying.state == "error", dying.state)
check("...and its last words are in the reason",
      "torch" in dying.status, dying.status)
dying.stop_quietly()

#  A worker that returns the wrong NUMBER of probabilities. Silently
#  mislabelling is the one outcome that must not happen, so the whole engine
#  is refused rather than that window.
WIDE = os.path.join(TEMP, "wide_worker.py")
with open(WIDE, "w", encoding="utf-8") as fh:
    fh.write('''
import json, sys, argparse
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--device", default="auto")
p.parse_args()
print("NNMETA " + json.dumps({"ok": True, "labels": ["A", "B", "C"],
      "win_samples": 12, "n_channels": 8, "window_s": 0.2,
      "model_fs": 60.0, "device": {"device": "cpu"},
      "calibration": {"mode": "off"}}), flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("cmd") == "quit":
        break
    print("NNOUT " + json.dumps({"id": r.get("id"), "probs": [0.5, 0.5]}),
          flush=True)
''')
wide = T.NeuralEngine(model_path=fake_model, script=WIDE, device="cpu")
wide.start()
wait_for(wide, lambda e: e.state in ("ready", "error"))
check("the wide worker starts ready", wide.ready, wide.status)
wide.submit(fist, t)
wait_for(wide, lambda e: e.state == "error", timeout=5.0)
check("a probability vector of the wrong width refuses the engine",
      wide.state == "error", wide.state)
check("...and says what disagreed with what",
      "probabilities" in wide.status and "labels" in wide.status,
      wide.status)
wide.stop_quietly()

#  The calibration refusal, checked WITHOUT spawning anything: a net fitted
#  on a calibrated scale must not be fed device counts.
cal_model = os.path.join(TEMP, "calibrated.pt")
with open(cal_model, "w", encoding="utf-8") as fh:
    fh.write("fake")
cal_sidecar = dict(FAKE_SIDECAR)
cal_sidecar["calibration"] = {"mode": "affine", "requested": "affine",
                              "subjects": ["someone_else"]}
with open(T.nn_stats_path(cal_model), "w", encoding="utf-8") as fh:
    json.dump(cal_sidecar, fh)

held_person, held_root = T.PERSON_ID, T.DATA_FOLDER
T.PERSON_ID = "person_who_never_calibrated"
T.DATA_FOLDER = os.path.join(TEMP, "emptydata")
os.makedirs(T.DATA_FOLDER, exist_ok=True)
needs_cal = T.NeuralEngine(model_path=cal_model, script=FAKE_WORKER,
                           device="cpu")
check("a calibrated net refuses an uncalibrated wearer", not needs_cal.start())
check("...before any subprocess is spawned", needs_cal._process is None)
check("...and the reason names CALIBRATE and the escape hatch",
      "calibration" in needs_cal.status
      and "--calibration off" in needs_cal.status, needs_cal.status)
T.PERSON_ID, T.DATA_FOLDER = held_person, held_root

#  And with a usable calibration on disk for that wearer, it starts.
cal_root = os.path.join(TEMP, "caldata")
T.PERSON_ID = "calibrated_person"
T.DATA_FOLDER = cal_root
saved = T.save_calibration(record, cal_root, "calibrated_person")
check("the fixture calibration saves to the wearer's folder",
      os.path.exists(saved))
ok_cal = T.NeuralEngine(model_path=cal_model, script=FAKE_WORKER,
                        device="cpu")
#  The record was built for "tester"; save_calibration stores it under the
#  folder it is given, which is what the engine reads.
started_cal = ok_cal.start()
check("with a usable calibration on disk it starts", started_cal,
      ok_cal.status)
check("...and resolved the wearer's numbers rather than ignoring the mode",
      ok_cal.calibration is not None
      and ok_cal.calibration_mode == "affine",
      f"{ok_cal.calibration_mode}")
wait_for(ok_cal, lambda e: e.state in ("ready", "error"))
#  The fake worker reports mode "off" while the sidecar said "affine". The
#  checkpoint is the fact, so the engine must re-resolve to "off" rather
#  than keep calibrating.
check("a worker that disagrees with the sidecar wins -- the weights are the "
      "fact", ok_cal.calibration_mode == "off",
      f"{ok_cal.calibration_mode}")
check("...and the calibration is dropped with it",
      ok_cal.calibration is None)
ok_cal.stop()
T.PERSON_ID, T.DATA_FOLDER = held_person, held_root


# ==============================================================================
section("7. PROVENANCE  (is a net trained, on what, and is it current?)")
# ==============================================================================

provenance = T.NeuralEngine(model_path=fake_model, script=FAKE_WORKER)
status = provenance.data_status()
check("with the dataset unchecked the state is 'unknown', never 'current'",
      status.get("state") == "unknown", str(status.get("state")))
check("it still reports what it was trained on",
      "9 recordings" in status.get("detail", ""),
      status.get("detail", "")[:90])

#  Handed the matching fingerprint it must read as current -- the net is
#  asked the same question as the model and must answer it the same way.
matching = {"uid_digest": "deadbeef", "n_recordings": 9,
            "root": os.path.join(TEMP, "nowhere")}
current = provenance.data_status(matching)
check("the matching dataset reads as up to date",
      current.get("state") == "current", str(current.get("state")))
moved = dict(matching, uid_digest="something_else", n_recordings=20)
check("a changed dataset reads as out of date",
      provenance.data_status(moved).get("state") == "stale",
      str(provenance.data_status(moved).get("state")))

untrained = T.NeuralEngine(
    model_path=os.path.join(TEMP, "none.pt")).data_status()
check("with no sidecar the state is 'untrained', never 'current'",
      untrained.get("state") == "untrained", str(untrained.get("state")))


# ==============================================================================
section("8. THE ARCHITECTURES")
# ==============================================================================

try:
    import torch                                                  # noqa: F401
    import nn_arch
    HAVE_TORCH = True
except ImportError as exc:
    HAVE_TORCH = False
    skip("every architecture builds and runs", f"no torch ({exc})")

if HAVE_TORCH:
    import nn_arch

    check("the default is the dilated convnet", nn_arch.KINDS[0] == "tcn",
          nn_arch.KINDS[0])
    check("every kind has a one-line description",
          all(k in nn_arch.DESCRIPTIONS for k in nn_arch.KINDS))

    sizes = {}
    for kind in nn_arch.KINDS:
        model = nn_arch.build(kind, 8, 6, 30)
        sizes[kind] = nn_arch.count_parameters(model)
        out = model(torch.zeros(2, 30, 8))
        check(f"{kind} builds and returns one logit per class",
              tuple(out.shape) == (2, 6), str(tuple(out.shape)))
    check("the default is the SMALLEST of them -- the point, on 1000 windows",
          sizes["tcn"] == min(sizes.values()),
          ", ".join(f"{k}={v}" for k, v in sizes.items()))
    check("it is genuinely small (under 10k parameters)",
          sizes["tcn"] < 10000, str(sizes["tcn"]))

    for win in (18, 24, 30, 48):
        model = nn_arch.build("tcn", 8, 6, win)
        out = model(torch.zeros(1, win, 8))
        check(f"the dilated stack survives a {win}-sample window",
              tuple(out.shape) == (1, 6))

    #  Seeding at CONSTRUCTION, compared on probabilities. On a separable
    #  problem the argmax agrees even between unrelated fits, so comparing
    #  labels would pass for the wrong reason (CLAUDE.md section 8b).
    a = nn_arch.build("tcn", 8, 6, 30, seed=7)
    b = nn_arch.build("tcn", 8, 6, 30, seed=7)
    c = nn_arch.build("tcn", 8, 6, 30, seed=8)
    probe = torch.randn(4, 30, 8)
    a.eval(), b.eval(), c.eval()
    with torch.no_grad():
        pa = torch.softmax(a(probe), 1)
        pb = torch.softmax(b(probe), 1)
        pc = torch.softmax(c(probe), 1)
    check("the same seed gives bit-identical initial probabilities",
          torch.allclose(pa, pb),
          f"max diff {float((pa - pb).abs().max()):.3g}")
    check("a different seed really does give a different net",
          not torch.allclose(pa, pc),
          f"max diff {float((pa - pc).abs().max()):.3g}")

    #  The attention pooling keeps the activation LEVEL as well as the shape.
    #  Removing the level is the mistake section 13 records measuring twice,
    #  so a net whose output is unchanged by a 10x amplitude change would be
    #  throwing it away.
    model = nn_arch.build("tcn", 8, 6, 30, seed=3).eval()
    quiet = torch.full((1, 30, 8), 0.2)
    loud = torch.full((1, 30, 8), 2.0)
    with torch.no_grad():
        difference = float((model(quiet) - model(loud)).abs().max())
    check("the net's output depends on the amplitude, not only the shape",
          difference > 1e-4, f"max logit difference {difference:.4g}")


# ==============================================================================
section("9. THE REAL WORKER, end to end")
# ==============================================================================

real_model = os.path.join(PROJECT, T.NN_MODEL_PATH)
if not HAVE_TORCH:
    skip("the real worker answers from the real checkpoint", "no torch")
elif not os.path.exists(real_model):
    skip("the real worker answers from the real checkpoint",
         f"no checkpoint yet -- run: python train_nn.py")
else:
    real = T.NeuralEngine(model_path=real_model, device="cpu", smoothing=1)
    #  A model trained with calibration needs this machine's own wearer to
    #  have one, which is exactly the refusal checked above.
    if not real.start():
        skip("the real worker answers from the real checkpoint", real.status)
    else:
        wait_for(real, lambda e: e.state in ("ready", "error"), timeout=60.0)
        if not real.ready:
            check("the real worker becomes ready", False, real.status)
        else:
            check("the real worker becomes ready", True, real.status)
            check("it reports the labels the dataset has",
                  len(real.labels) >= 2, str(real.labels))
            check("it reports a window length", real.win_samples > 0,
                  str(real.win_samples))
            check("it reports a real parameter count",
                  int(real.meta.get("n_parameters", 0)) > 0,
                  str(real.meta.get("n_parameters")))
            check("it says which device it is on",
                  bool((real.meta.get("device") or {}).get("device")),
                  str((real.meta.get("device") or {}).get("device")))

            n = 31
            t = np.arange(n) / 49.0
            rng = np.random.default_rng(11)
            live = np.abs(rng.normal(
                [40, 300, 500, 200, 60, 50, 45, 55], 25.0, (n, 8)))
            check("a live window is accepted", real.submit(live, t))
            produced = wait_for(real, lambda e: e.last is not None,
                                timeout=20.0)
            check("the real net answers", len(produced) == 1,
                  str(len(produced)))
            if produced:
                answer = produced[0]
                check("the answer is one of its own labels",
                      answer.label in real.labels, str(answer.label))
                check("the probabilities sum to 1 -- a real softmax",
                      abs(sum(answer.raw_probabilities) - 1.0) < 1e-5,
                      f"{sum(answer.raw_probabilities):.6f}")
                check("every probability is a real number in [0, 1]",
                      all(0.0 <= p <= 1.0 for p in answer.raw_probabilities))
                check("the forward pass was measured",
                      answer.infer_ms >= 0.0, f"{answer.infer_ms} ms")
                #  The budget. It is a diagnostic, so overrunning is a
                #  warning rather than a failure -- but it must be MEASURED,
                #  the way section 12 records the decision panel's overrun.
                print(f"      measured: {answer.latency_ms:.1f} ms round "
                      f"trip, {answer.infer_ms:.2f} ms forward pass, against "
                      f"a {T.NN_INTERVAL_MS} ms budget")
                check("one window costs less than the prediction interval",
                      answer.latency_ms < T.NN_INTERVAL_MS,
                      f"{answer.latency_ms:.1f} ms vs {T.NN_INTERVAL_MS} ms")

            #  The worker must reproduce the trainer's scaler exactly. This
            #  is the one place the two halves of the contract can drift, so
            #  it is compared against a direct forward pass here in-process.
            payload = torch.load(real_model, map_location="cpu",
                                 weights_only=False)
            direct = nn_arch.build(payload["kind"], payload["n_channels"],
                                   len(payload["labels"]),
                                   payload["win_samples"],
                                   hidden=payload["hidden"],
                                   dropout=payload["dropout"], seed=0)
            direct.load_state_dict(payload["state_dict"], strict=True)
            direct.eval()
            cal = real.calibration
            grid = T.nn_input_block(
                T.resample_uniform(live, t, payload["win_samples"]), cal)
            mean = np.asarray(payload["scaler"]["mean"],
                              dtype=np.float32).reshape(-1)
            std = np.asarray(payload["scaler"]["std"],
                             dtype=np.float32).reshape(-1)
            check("the scaler is a per-channel vector, so it broadcasts "
                  "against ONE window as well as a batch",
                  mean.size == payload["n_channels"], str(mean.size))
            scaled = ((np.log1p(np.abs(grid)) - mean) / std).astype(np.float32)
            check("scaling one window keeps its shape -- the bug that only "
                  "the inference path can hit",
                  scaled.shape == (payload["win_samples"],
                                   payload["n_channels"]),
                  str(scaled.shape))
            with torch.no_grad():
                expected = torch.softmax(
                    direct(torch.from_numpy(scaled[None, ...])), 1
                )[0].numpy()
            if produced:
                check("the worker's answer matches a direct forward pass -- "
                      "one scaler, one contract",
                      np.allclose(expected, produced[0].raw_probabilities,
                                  atol=1e-5),
                      f"max diff "
                      f"{float(np.max(np.abs(expected - np.asarray(produced[0].raw_probabilities)))):.3g}")
            real.stop()
            check("the real worker shuts down cleanly",
                  real.state == "stopped", real.state)


# ==============================================================================
shutil.rmtree(TEMP, ignore_errors=True)

print(f"\n================  {sum(OK)}/{len(OK)} checks passed"
      + (f", {len(SKIPPED)} skipped" if SKIPPED else "")
      + "  ================")
if SKIPPED:
    for name in SKIPPED:
        print(f"  skipped: {name}")
sys.exit(0 if all(OK) else 1)

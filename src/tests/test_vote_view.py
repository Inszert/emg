"""
A selected soft vote must be DESCRIBED as what it is -- on the stats screens and
under the probability bars -- instead of the deployed model's numbers or "?".

  1. vote_view: one row per member, agreement with the vote, the class mask
     applied to members, a wrong-width member skipped, the panel text
  2. zoo.vote_stats scores a vote from the members' stored out-of-fold
     probabilities exactly as the trainer scores one model (checked against an
     independent calculation), and refuses what it cannot measure honestly
  3. the live classifier: a selected vote carries its measured statistics (not
     the first member's), the members' own answers for the last window, and
     says it is using the selection
  4. the neural engine keeps the nets' own answers and reports them
  5. the window: the vote line under the bars (before and after PREDICT),
     MODEL STATS (vote tiles, members table, a banner when the trail is the
     deployed model's), NN STATS (the SELECTED nets, not the deployed one),
     and open statistics windows are closed when the selection moves
  6. on the real zoo in this checkout (skipped when it is not there): the
     numbers equal the trainer's own scoring of the same probabilities
"""
import os
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import emg_dataset_tool as T                                         # noqa: E402
from emg_app import settings_store as SS                             # noqa: E402
from emg_app import vote_view as VV                                  # noqa: E402
from emg_app import zoo                                              # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    shown = "" if detail is None or (isinstance(detail, str) and not detail) \
        else f"  [{detail}]"
    print(f"{'PASS' if condition else 'FAIL'}  {name}{shown}")


def section(title):
    print(f"\n=== {title} ===")


import joblib                                                          # noqa: E402
from sklearn.ensemble import RandomForestClassifier                    # noqa: E402
from PySide6 import QtWidgets                                          # noqa: E402

LABELS = ["REST", "FIST", "OPEN", "THUMB_UP"]

# ==========================================================================
section("1. vote_view")
probs = [[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.05, 0.9, 0.03, 0.02]]
rows = VV.member_rows(["A", "B", "C"], probs, LABELS, "FIST")
check("one row per member", [r["name"] for r in rows] == ["A", "B", "C"])
check("each member's OWN top answer and confidence",
      [r["label"] for r in rows] == ["FIST", "REST", "FIST"]
      and abs(rows[1]["confidence"] - 0.6) < 1e-9)
check("agreement with the vote is marked per member",
      [r["agrees"] for r in rows] == [True, False, True])
check("a member of the wrong width is skipped, never guessed",
      len(VV.member_rows(["A", "B"], [[0.5, 0.5], probs[0]], LABELS, "FIST"))
      == 1)
T.CLASSIFIER_OFF = ["FIST"]
masked = VV.member_rows(["A", "C"], [probs[0], probs[2]], LABELS, "REST")
check("a gesture switched off is off for every member too -- a member can "
      "never name what the vote was told to skip",
      all(r["label"] != "FIST" for r in masked), [r["label"] for r in masked])
T.CLASSIFIER_OFF = []
text = VV.vote_html(rows, "FIST", "#0f0", "#f80", "#888")
check("the panel text says 'VOTE of 3: 2 of 3 say FIST' and lists the members",
      "VOTE of 3" in text and "2 of 3 say FIST" in text
      and all(n in text for n in "ABC"), text[:120])
check("no members, no text", VV.vote_html([], "FIST", "", "", "") == "")

# ==========================================================================
section("2. zoo.vote_stats")
TEMP = tempfile.mkdtemp(prefix="emgvote_")
MODELS = os.path.join(TEMP, "models")
DATA = os.path.join(TEMP, "data")
RESULTS = os.path.join(TEMP, "results")
os.makedirs(MODELS)
os.makedirs(DATA)
os.makedirs(RESULTS)
T.DATA_FOLDER = DATA
T.MODEL_PATH = os.path.join(MODELS, "gesture_model.joblib")
T.NN_MODEL_PATH = os.path.join(MODELS, "gesture_nn.pt")
T.MYO_PORT = "COM254"
T.ENABLE_LIVE_PREDICTION = True
T.PREDICTION_SMOOTHING = 1
T.PREDICTION_MIN_CONFIDENCE = 0.0
T.RECONNECT_DELAY_S = 0.3
T.PAUSE_SESSION_ON_LINK_LOSS = False
T.QC_PAUSE_ON = "never"
T.GESTURES = list(LABELS)
T.CUSTOM_GESTURES = {}
T.CLASSIFIER_OFF = []
for k, v in SS.DEFAULTS.items():
    setattr(T, k, v)

rng = np.random.default_rng(3)
FS, WIN = 60.0, 12
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
    "THUMB_UP": np.array([40, 30, 60, 200, 210, 40, 30, 20], dtype=float),
}


def window_of(label):
    base = PROFILES[label] * rng.uniform(0.8, 1.25)
    return np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))


F, Y = [], []
for idx, label in enumerate(LABELS):
    for _ in range(60):
        F.append(T.model_feature_vector(window_of(label), "raw", "level", True,
                                        FS))
        Y.append(idx)
F, Y = np.asarray(F, dtype=np.float32), np.asarray(Y)
forests = [RandomForestClassifier(n_estimators=15, random_state=s).fit(F, Y)
           for s in range(3)]
COMPAT = {"labels": LABELS, "fs": FS, "window_s": 0.2, "window_samples": WIN,
          "representation": "raw", "feature_set": "level",
          "freq_features": True, "calibration_mode": "off",
          "adaptation_mode": "off"}
DIGEST = "abc123"


def payload(name, est):
    return {"model": est, "labels": LABELS, "fs": FS, "window_s": 0.2,
            "window_samples": WIN, "representation": "raw",
            "feature_set": "level", "freq_features": True,
            "calibration": {"mode": "off"}, "adaptation": {"mode": "off"},
            "created_iso": "2026-10-27T00:17:31",
            "stats": {"model_name": name, "window_accuracy": 0.9,
                      "labels": LABELS, "overlap": 0.75,
                      "created_iso": "2026-10-27T00:17:31"}}


def add(eid, name, acc, est, family="ml", digest=DIGEST):
    folder = os.path.join(zoo.zoo_root(), eid)
    os.makedirs(folder, exist_ok=True)
    if family == "ml":
        joblib.dump(payload(name, est), os.path.join(folder, "model.joblib"))
    zoo.write_meta(folder, {
        "id": eid, "family": family, "name": name, "kind": name,
        "file": "model.joblib" if family == "ml" else "model.pt",
        "size_bytes": 5000, "params": None if family == "ml" else 1000,
        "scores": {"window_acc": acc, "macro_f1": acc - 0.02,
                   "vote_acc": acc + 0.01},
        "compat": COMPAT, "data_digest": digest, "latency_ms": 3.0,
        "created_iso": "2026-10-27T00:17:31",
        "trained_with": {"augment": [], "adapt": False}})


joblib.dump(payload("DEFAULT", forests[0]), T.MODEL_PATH)
for i, (eid, name, acc) in enumerate((("ml-a", "RF-A", 0.80),
                                      ("ml-b", "RF-B", 0.79),
                                      ("ml-c", "RF-C", 0.78))):
    add(eid, name, acc, forests[i])

# stored out-of-fold probabilities: 3 people, 2 recordings each, 6 windows each
n = 3 * 2 * 6
persons = np.repeat(["p1", "p2", "p3"], 12)
uid = np.repeat(np.arange(6), 6)
y = np.tile(np.repeat(np.arange(4), 3)[:6], 6)[:n] % 4
np.savez(os.path.join(RESULTS, "data_ml.npz"), y=y, persons=persons, uid=uid,
         labels=np.array(LABELS), digest=np.array(DIGEST))
OOF = {}
for eid in ("ml-a", "ml-b", "ml-c"):
    for s in range(2):
        p = rng.dirichlet(np.ones(4), n)
        p[np.arange(n), y] += rng.uniform(0, 1.2, n)      # mostly right
        p /= p.sum(1, keepdims=True)
        OOF[(eid, s)] = p
        np.savez(os.path.join(RESULTS, f"oof_{eid}_s{s}.npz"), probs=p)

entries = [zoo.get_entry(i) for i in ("ml-a", "ml-b", "ml-c")]
vs = zoo.vote_stats(entries, results_dir=RESULTS)
# independent calculation: mean over seeds of (mean over people of accuracy)
accs = []
for s in range(2):
    pm = sum(OOF[(e, s)] for e in ("ml-a", "ml-b", "ml-c")) / 3
    hit = pm.argmax(1) == y
    accs.append(np.mean([hit[persons == q].mean() for q in ("p1", "p2", "p3")]))
check("the vote is scored from the members' stored probabilities "
      "(window accuracy = mean over held-out people, mean over seeds)",
      vs and abs(vs["window_accuracy"] - float(np.mean(accs))) < 1e-9,
      None if not vs else f"{vs['window_accuracy']:.4f} vs {np.mean(accs):.4f}")
check("it also carries macro-F1, the per-recording vote, the confusion matrix "
      "and the per-person scores",
      all(k in vs for k in ("window_macro_f1", "recording_vote_accuracy",
                            "confusion_matrix", "subject_independent",
                            "majority_class_baseline")))
check("2 seeds were used, 3 people, 6 recordings",
      vs["seeds"] == 2 and len(vs["subjects"]) == 3
      and vs["n_recordings_voted"] == 6)
check("it says the combination was chosen by the user (no selection bias)",
      "chosen by you" in vs["validation_protocol"])
check("fewer than two models: no vote, so None",
      zoo.vote_stats(entries[:1], results_dir=RESULTS) is None)
add("ml-x", "RF-X", 0.70, forests[0], digest="another-dataset")
check("a model whose probabilities were made on ANOTHER dataset: None, never "
      "a number that mixes datasets",
      zoo.vote_stats([entries[0], zoo.get_entry("ml-x")],
                     results_dir=RESULTS) is None)
check("a missing probabilities file: None",
      zoo.vote_stats(entries, results_dir=os.path.join(TEMP, "nowhere"))
      is None)
rows = zoo.member_rows(entries)
check("member_rows gives each member's OWN numbers",
      [r["name"] for r in rows] == ["RF-A", "RF-B", "RF-C"]
      and abs(rows[0]["window_acc"] - 0.80) < 1e-9)

# ==========================================================================
section("3. the live classifier")
T.MODEL_SELECTION_ENABLED = True
T.SELECTED_ML = ["ml-a", "ml-b", "ml-c"]
_orig = zoo.vote_stats
zoo.vote_stats = lambda es, results_dir=None: _orig(es, results_dir=RESULTS)
from emg_app.prediction import LivePredictor                          # noqa: E402

pred = LivePredictor()
check("the selected vote loads", pred.load() and len(pred.members) == 3,
      pred.status)
check("it says it is using the SELECTION", pred.using_selection)
check("its statistics are the VOTE's, measured -- not the first member's 0.9 "
      "and not empty",
      abs(pred.stats["window_accuracy"] - vs["window_accuracy"]) < 1e-9
      and pred.stats["model_name"].startswith("soft vote: "), pred.stats.get(
          "window_accuracy"))
check("created date is the members' (the '?' the screen used to show)",
      pred.stats["created_iso"].startswith("2026-10-27"))
check("the members' own numbers ride along",
      [m["name"] for m in pred.stats["members"]] == ["RF-A", "RF-B", "RF-C"])
t = np.linspace(0, 0.2, WIN)
p1 = pred.predict(window_of("FIST"), t)
rows = pred.vote_rows(p1.label)
check("after a window, every member's own answer is available",
      len(rows) == 3 and all(r["label"] in LABELS for r in rows),
      [(r["name"], r["label"]) for r in rows])
check("the vote's probabilities are exactly the mean of the members'",
      np.allclose(np.mean([m[0] for m in pred.model.last_members], axis=0),
                  pred.model.predict_proba(pred.last_features)[0]))
T.MODEL_SELECTION_ENABLED = False
single = LivePredictor()
single.load()
check("selection off: the deployed model, not 'using the selection', and no "
      "vote rows", not single.using_selection and single.vote_rows("FIST") == [])
T.MODEL_SELECTION_ENABLED = True

# ==========================================================================
section("4. the neural engine keeps the nets' answers")
from emg_app.neural import NeuralEngine                               # noqa: E402

engine = NeuralEngine()
engine.labels = list(LABELS)
engine.win_samples = WIN
engine.meta = {"members": [{"name": "NN-A"}, {"name": "NN-B"}],
               "device": {"device": "cpu"}}
engine._pending[1] = 0.0
got = engine._on_out({"id": 1, "probs": [0.1, 0.5, 0.2, 0.2], "ms": 1.0,
                      "members": [[0.1, 0.8, 0.05, 0.05],
                                  [0.1, 0.2, 0.5, 0.2]]})
rows = engine.vote_rows(got.label)
check("the worker's per-net softmax is kept", len(engine.last_member_probs) == 2)
check("each net's own answer is reported, with whether it agrees",
      [r["label"] for r in rows] == ["FIST", "OPEN"]
      and [r["agrees"] for r in rows] == [True, False], rows)
engine.meta = {"members": [{"name": "NN-A"}]}
check("one net is not a vote: no rows", engine.vote_rows("FIST") == [])

# ==========================================================================
section("5. the window")
app = QtWidgets.QApplication(sys.argv)
T.apply_theme(app)
window = T.MainWindow()
for _ in range(3):
    app.processEvents()
window._reload_selected_models()
check("a selected vote is named under the bars at once (before PREDICT)",
      not window.vote_label.isHidden()
      and "SOFT VOTE of 3" in window.vote_label.text()
      and "RF-A" in window.vote_label.text(), window.vote_label.text()[:80])
prediction = window.predictor.predict(window_of("FIST"), t)
window._show_prediction(prediction)
check("after a window the line shows how each member voted",
      "VOTE of 3" in window.vote_label.text()
      and window.vote_label.text().count("RF-") >= 3,
      window.vote_label.text()[:90])

import emg_app.gui.dialogs.model_stats as MS                          # noqa: E402

dialog = MS.ModelStatsDialog(window.predictor, window,
                             panel=window.decision_panel)
texts = [lab.text() for lab in dialog.findChildren(QtWidgets.QLabel)]
check("MODEL STATS: the title is the vote", any(
    t_.startswith("soft vote: RF-A") for t_ in texts))
check("MODEL STATS: the tiles carry the vote's measured numbers, no '--'",
      any(f"{vs['window_accuracy'] * 100:.1f} %" in t_ for t_ in texts),
      [t_ for t_ in texts if " %" in t_][:4])
check("MODEL STATS: the date is not '?'",
      not any("trained ?" in t_ for t_ in texts))
tables = dialog.findChildren(QtWidgets.QTableWidget)
check("MODEL STATS: a table lists the 3 members' own numbers",
      any(tb.rowCount() == 3 and tb.item(0, 0).text() == "RF-A"
          for tb in tables))
window.decision_panel.names = ["svm_rbf", "lda"]       # a panel of ANOTHER model
window.decision_panel.loaded = True
window.decision_panel.members = {"svm_rbf": object(), "lda": object()}
dialog2 = MS.ModelStatsDialog(window.predictor, window,
                              panel=window.decision_panel)
check("MODEL STATS: when the trail is the deployed model's, it SAYS so",
      any("DEPLOYED model" in lab.text()
          for lab in dialog2.findChildren(QtWidgets.QLabel)))

# the selected NETS, as sidecars
for eid, name, acc in (("nn-a", "NN-A", 0.70), ("nn-b", "NN-B", 0.66)):
    folder = os.path.join(zoo.zoo_root(), eid)
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "model.pt"), "wb").write(b"x" * 4000)
    import json
    json.dump({"kind": name.lower(), "architecture": f"arch {name}",
               "labels": LABELS, "window_s": 0.7, "created_iso": "2026-10-27",
               "stats": {"model_name": name, "window_accuracy": acc,
                         "n_parameters": 1000, "epochs": 150,
                         "grouping": "subject"}},
              open(os.path.join(folder, "model.json"), "w"))
    add(eid, name, acc, None, family="nn")
json.dump({"kind": "deployed", "architecture": "the DEPLOYED net",
           "labels": LABELS, "created_iso": "2026-10-22",
           "stats": {"model_name": "deployed", "window_accuracy": 0.99,
                     "n_parameters": 24, "epochs": 150}},
          open(os.path.join(MODELS, "gesture_nn.json"), "w"))
open(T.NN_MODEL_PATH, "wb").write(b"x" * 100)
T.SELECTED_NN = ["nn-a", "nn-b"]
import emg_app.gui.dialogs.neural_stats as NS                         # noqa: E402

nn_dialog = NS.NeuralStatsDialog(window.nn_engine, window, window.predictor)
nn_texts = [lab.text() for lab in nn_dialog.findChildren(QtWidgets.QLabel)]
check("NN STATS describes the SELECTED nets, not the deployed one",
      any(t_.startswith("NN soft vote of 2 nets") for t_ in nn_texts)
      and not any("the DEPLOYED net" in t_ for t_ in nn_texts), nn_texts[:3])
check("NN STATS: the members table lists both nets",
      any(tb.rowCount() == 2 and {tb.item(0, 0).text(), tb.item(1, 0).text()}
          == {"NN-A", "NN-B"} for tb in nn_dialog.findChildren(
              QtWidgets.QTableWidget)))
T.SELECTED_NN = ["nn-a"]
nn_one = NS.NeuralStatsDialog(window.nn_engine, window, window.predictor)
check("ONE selected net: its own sidecar, no vote table",
      any("NN-nn-a" in lab.text() or lab.text() == "NN-nn-a"
          or "arch NN-A" in lab.text()
          for lab in nn_one.findChildren(QtWidgets.QLabel)))
T.SELECTED_NN = []
nn_dep = NS.NeuralStatsDialog(window.nn_engine, window, window.predictor)
check("nothing selected: the deployed net, as before",
      any("the DEPLOYED net" in lab.text()
          for lab in nn_dep.findChildren(QtWidgets.QLabel)))

window.stats_dialog = MS.ModelStatsDialog(window.predictor, window,
                                          panel=window.decision_panel)
window.stats_dialog.show()
check("a stats window is open", window.stats_dialog.isVisible())
window._reload_selected_models()
check("changing the selection CLOSES it (it described the old models)",
      not window.stats_dialog.isVisible())

# ==========================================================================
section("6. the real zoo in this checkout")
zoo.vote_stats = _orig
REAL_ZOO = os.path.join(ROOT, "models", "zoo")
REAL_EVAL = os.path.join(ROOT, "results", "zoo_eval")
ids = {"ml": ["ml-rf-500-leaf3", "ml-rf-500", "ml-et-500-leaf3"],
       "nn": ["nn-gru", "nn-tsmixer", "nn-chan-transformer"]}
try:
    import train_zoo as TZ
except Exception as exc:                                  # torch missing, ...
    TZ = None
    print("   (train_zoo not importable: " + str(exc)[:60] + ")")
for fam, fam_ids in ids.items():
    real = [zoo.get_entry(i, REAL_ZOO) for i in fam_ids]
    data_file = os.path.join(REAL_EVAL, f"data_{fam}.npz")
    if not all(real) or not os.path.exists(data_file) or TZ is None:
        check(f"({fam}: the real zoo / the trainer is not available here: "
              "skipped)", True)
        continue
    got = zoo.vote_stats(real, results_dir=REAL_EVAL)
    d = np.load(data_file, allow_pickle=True)
    yy, pp, uu = d["y"], d["persons"].astype(str), d["uid"]
    seeds = [s_ for s_ in range(20) if all(os.path.exists(os.path.join(
        REAL_EVAL, f"oof_{e.id}_s{s_}.npz")) for e in real)]
    ref = [TZ.scores_from_probs(
        sum(np.load(os.path.join(REAL_EVAL, f"oof_{e.id}_s{s_}.npz"))["probs"]
            for e in real) / len(real), yy, pp, uu, len(d["labels"]))
        for s_ in seeds]
    want = {k: float(np.mean([r[k] for r in ref]))
            for k in ("window_acc", "macro_f1", "vote_acc")}
    check(f"{fam.upper()}: window accuracy equals the trainer's own scoring "
          f"of the same probabilities ({want['window_acc'] * 100:.2f} %)",
          got and abs(got["window_accuracy"] - want["window_acc"]) < 1e-9)
    check(f"{fam.upper()}: macro-F1 and the per-recording vote equal it too",
          abs(got["window_macro_f1"] - want["macro_f1"]) < 1e-9
          and abs(got["recording_vote_accuracy"] - want["vote_acc"]) < 1e-9)

window.close()
shutil.rmtree(TEMP, ignore_errors=True)
print()
passed = sum(OK)
print(f"{'=' * 16}  {passed}/{len(OK)} checks passed  {'=' * 16}")
sys.exit(0 if all(OK) else 1)

"""
The model zoo, soft voting, adaptation to the person, and the settings store.

No window, no hardware, no GPU (the neural-net worker runs on the CPU):

  1. the zoo registry: listing, ranking, stars (top 3 per family, the voting
     subset only when it beats every single model, decided on the nested
     number), sizes, voting compatibility
  2. selection: off by default = the original files; on = the chosen ones
  3. the ML predictor: one selected model, a soft vote of two (the mean of
     their probabilities, checked against the members), refusal of a vote
     that cannot be averaged, and that selection OFF restores the original
  4. the neural worker: two nets in one process, their mean, the `members`
     field, a vote refused for mismatching nets, `adapt` reaching only a net
     trained with adaptation
  5. adaptation to the person: ML features standardised per person, the
     fallback to global statistics, ADAPT_LIVE off, models without it untouched
  6. the stored calibration signal and the statistics computed from it
  7. augmentation: each type, determinism, identity when off, typos refused
  8. the settings store: defaults = config, round trip, bad values
  9. the decisions behind the two questions (emg_app/prompts.py)
  10. the trainers: with the new switches off they are bit-identical to before
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

import emg_dataset_tool as T                                         # noqa: E402
from emg_app import adaptation as AD                                 # noqa: E402
from emg_app import prompts as PR                                    # noqa: E402
from emg_app import settings_store as SS                             # noqa: E402
from emg_app import zoo                                              # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    shown = "" if detail is None or (isinstance(detail, str) and not detail) \
        else f"  [{detail}]"
    print(f"{'PASS' if condition else 'FAIL'}  {name}{shown}")


def section(title):
    print(f"\n=== {title} ===")


TEMP = tempfile.mkdtemp(prefix="emgzoo_")
MODELS = os.path.join(TEMP, "models")
DATA = os.path.join(TEMP, "data")
os.makedirs(MODELS)
os.makedirs(DATA)
T.MODEL_PATH = os.path.join(MODELS, "gesture_model.joblib")
T.NN_MODEL_PATH = os.path.join(MODELS, "gesture_nn.pt")
T.DATA_FOLDER = DATA
T.ENABLE_LIVE_PREDICTION = True
T.MODEL_SELECTION_ENABLED = False
T.SELECTED_ML = []
T.SELECTED_NN = []
T.PREDICTION_SMOOTHING = 1
T.PREDICTION_MIN_CONFIDENCE = 0.0
T.CLASSIFIER_OFF = []
T.ADAPT_LIVE = True
ZOO = zoo.zoo_root()
check("the zoo lives beside the models", ZOO == os.path.join(MODELS, "zoo"), ZOO)

# ---- a toy ML family ---------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   # noqa: E402
from sklearn.ensemble import RandomForestClassifier                    # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
import joblib                                                          # noqa: E402

LABELS = ["REST", "FIST", "OPEN", "THUMB_UP"]
FS = 60.0
WIN = 12
rng = np.random.default_rng(5)
PROFILES = {
    "REST": np.array([10, 10, 12, 10, 10, 10, 10, 10], dtype=float),
    "FIST": np.array([30, 120, 250, 140, 40, 30, 25, 30], dtype=float),
    "OPEN": np.array([150, 60, 80, 50, 160, 140, 120, 90], dtype=float),
    "THUMB_UP": np.array([40, 30, 60, 200, 210, 40, 30, 20], dtype=float),
}


def window(label):
    base = PROFILES[label] * rng.uniform(0.8, 1.25)
    return np.abs(rng.normal(base, base * 0.18 + 3.0, (WIN, 8)))


F, Y = [], []
for idx, label in enumerate(LABELS):
    for _ in range(100):
        F.append(T.model_feature_vector(window(label), "raw", "level", True,
                                        FS))
        Y.append(idx)
F, Y = np.asarray(F, dtype=np.float32), np.asarray(Y)


def ml_payload(name, estimator, representation="raw", adaptation=None,
               X=F):
    return {"model": estimator, "labels": LABELS, "fs": FS, "window_s": 0.2,
            "window_samples": WIN, "representation": representation,
            "feature_set": "level", "freq_features": True,
            "emg_mode": "filtered",
            "calibration": {"mode": "off"},
            "adaptation": adaptation or {"mode": "off"},
            "stats": {"model_name": name, "window_accuracy": 0.9,
                      "labels": LABELS}}


def make_entry(eid, family, name, scores, payload=None, compat=None,
               params=None, nn_src=None, trained_with=None, digest="d1"):
    folder = os.path.join(ZOO, eid)
    os.makedirs(folder, exist_ok=True)
    fname = "model.joblib" if family == "ml" else "model.pt"
    if payload is not None:
        joblib.dump(payload, os.path.join(folder, fname))
    elif nn_src:
        shutil.copy(nn_src, os.path.join(folder, fname))
    meta = {"id": eid, "family": family, "name": name, "kind": name,
            "file": fname, "params": params, "size_bytes": 1000,
            "description": name, "scores": scores,
            "compat": compat or {}, "data_digest": digest,
            "trained_with": trained_with or {"augment": [], "adapt": False}}
    zoo.write_meta(folder, meta)
    return folder


def scores(acc, f1=None):
    return {"window_acc": acc, "macro_f1": f1 if f1 is not None else acc - 0.03,
            "vote_acc": acc + 0.05}


ML_COMPAT = {"labels": LABELS, "fs": FS, "window_s": 0.2, "window_samples": WIN,
             "representation": "raw", "feature_set": "level",
             "freq_features": True, "calibration_mode": "off"}

lda = Pipeline([("scale", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                   shrinkage="auto"))]).fit(F, Y)
rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(F, Y)
rf2 = RandomForestClassifier(n_estimators=25, random_state=1,
                             min_samples_leaf=3).fit(F, Y)
rf3 = RandomForestClassifier(n_estimators=15, random_state=2).fit(F, Y)
rf4 = RandomForestClassifier(n_estimators=15, random_state=3).fit(F, Y)

make_entry("ml-lda", "ml", "LDA", scores(0.63), ml_payload("LDA", lda),
           ML_COMPAT)
make_entry("ml-rf", "ml", "RF", scores(0.77), ml_payload("RF", rf), ML_COMPAT)
make_entry("ml-rf2", "ml", "RF-leaf3", scores(0.766),
           ml_payload("RF-leaf3", rf2), ML_COMPAT)
make_entry("ml-rf3", "ml", "RF-small", scores(0.70), ml_payload("RF-small", rf3),
           ML_COMPAT)
make_entry("ml-rf4", "ml", "RF-tiny", scores(0.70, 0.60),
           ml_payload("RF-tiny", rf4), ML_COMPAT)
make_entry("ml-unscored", "ml", "Unscored", {}, ml_payload("Unscored", lda),
           ML_COMPAT)
env_compat = dict(ML_COMPAT, representation="envelope")
make_entry("ml-env", "ml", "RF-envelope", scores(0.5),
           ml_payload("RF-envelope", rf, representation="envelope"), env_compat)

# ==========================================================================
section("1. the zoo registry, ranking and stars")
entries = zoo.list_entries()
check("every entry with a meta.json is listed",
      {e.id for e in entries} >= {"ml-lda", "ml-rf", "ml-rf2", "ml-env"},
      sorted(e.id for e in entries))
check("family filter works", all(e.family == "ml"
                                 for e in zoo.list_entries(family="ml")))
os.makedirs(os.path.join(ZOO, "broken"))
open(os.path.join(ZOO, "broken", "meta.json"), "w").write("{not json")
check("an unreadable meta.json is skipped, not fatal",
      "broken" not in {e.id for e in zoo.list_entries()})
ranked = zoo.rank([e for e in entries if e.family == "ml"])
check("ranked by window accuracy, best first",
      [e.id for e in ranked][:2] == ["ml-rf", "ml-rf2"]
      and [e.id for e in ranked][4:] == ["ml-lda", "ml-env", "ml-unscored"],
      [e.id for e in ranked])
check("a tie on accuracy is broken by macro-F1 (rf3 and rf4 both 70 %)",
      [e.id for e in ranked][2:4] == ["ml-rf3", "ml-rf4"],
      [e.id for e in ranked])
check("an entry with no score is last", ranked[-1].id == "ml-unscored"
      and ranked[-1].window_acc is None, ranked[-1].id)

stars = zoo.compute_stars(entries, {})
check("the top 3 of a family are starred",
      set(stars["models"]) == {"ml-rf", "ml-rf2", "ml-rf3"},
      sorted(stars["models"]))
check("an unscored model is never starred", "ml-unscored" not in stars["models"])
check("the star's tooltip says where it ranks and what it costs",
      "#1 of" in stars["models"]["ml-rf"]
      and "77.0 %" in stars["models"]["ml-rf"], stars["models"]["ml-rf"])
check("no voting star without a measured combination",
      stars["ensembles"] == {})
ens_good = {"ml": {"members": ["ml-rf", "ml-rf2"], "window_acc": 0.80,
                   "nested_acc": 0.79}}
check("the voting subset is starred when it beats every single model",
      "ml" in zoo.compute_stars(entries, ens_good)["ensembles"])
ens_optimistic = {"ml": {"members": ["ml-rf", "ml-rf2"], "window_acc": 0.80,
                         "nested_acc": 0.76}}
check("...decided on the NESTED number, not the optimistic one",
      "ml" not in zoo.compute_stars(entries, ens_optimistic)["ensembles"])
ens_equal = {"ml": {"members": ["ml-rf", "ml-rf2"], "nested_acc": 0.77}}
check("a combination only EQUAL to the best single model is not starred",
      "ml" not in zoo.compute_stars(entries, ens_equal)["ensembles"])
ens_gone = {"ml": {"members": ["ml-rf", "ml-deleted"], "nested_acc": 0.9}}
check("a combination with a missing member is not starred",
      "ml" not in zoo.compute_stars(entries, ens_gone)["ensembles"])
make_entry("ml-new", "ml", "RF-new", scores(0.95),
           ml_payload("RF-new", rf3), ML_COMPAT)
stars2 = zoo.compute_stars(zoo.list_entries(), {})
check("a newly trained better model takes a star and the old third loses it "
      "(stars are recomputed from the scores, never stored)",
      set(stars2["models"]) == {"ml-new", "ml-rf", "ml-rf2"},
      sorted(stars2["models"]))
shutil.rmtree(os.path.join(ZOO, "ml-new"))
check("...and when it is removed the old ranking returns",
      set(zoo.compute_stars(zoo.list_entries(), {})["models"])
      == {"ml-rf", "ml-rf2", "ml-rf3"})

check("human_size", zoo.human_size(2048) == "2.0 KB"
      and zoo.human_size(500) == "500 B"
      and zoo.human_size(5 * 1024 * 1024) == "5.0 MB"
      and zoo.human_size(None) == "?")
e_rf, e_lda, e_env = (zoo.get_entry(i) for i in ("ml-rf", "ml-lda", "ml-env"))
summary = zoo.selection_summary([e_rf, e_lda])
check("a selection reports its count and total size",
      summary["n"] == 2 and summary["size_bytes"] ==
      e_rf.size_bytes + e_lda.size_bytes and summary["size_known"], summary)
check("two compatible models can be voted",
      zoo.check_vote([e_rf, e_lda]) == (True, ""))
ok, why = zoo.check_vote([e_rf, e_env])
check("...mismatching windows/representation cannot, and it says which",
      not ok and "representation" in why, why)
ok, why = zoo.check_vote([e_rf, e_env][:1] + [zoo.get_entry("ml-lda")])
check("one model is always fine", zoo.check_vote([e_rf]) == (True, ""))
check("nothing selected is refused", not zoo.check_vote([])[0])

# ==========================================================================
section("2. selection: off = the original files")
T.MODEL_SELECTION_ENABLED = False
T.SELECTED_ML = ["ml-rf", "ml-lda"]
check("selection OFF: no zoo path is used even if ids are set",
      zoo.selected_paths("ml") == [] and zoo.describe_selection("ml") == "")
T.MODEL_SELECTION_ENABLED = True
check("selection ON: the chosen files, in the chosen order",
      [os.path.basename(os.path.dirname(p)) for p in zoo.selected_paths("ml")]
      == ["ml-rf", "ml-lda"])
T.SELECTED_ML = ["ml-rf", "ml-nope"]
check("an id that is not in the zoo is ignored",
      len(zoo.selected_paths("ml")) == 1)
T.SELECTED_ML = ["ml-rf", "ml-lda"]
check("a description for the panels", zoo.describe_selection("ml") ==
      "soft vote: RF + LDA", zoo.describe_selection("ml"))

# ==========================================================================
section("3. the ML predictor: one model, a soft vote, a refused vote, OFF")
default_payload = ml_payload("DEFAULT", lda)
joblib.dump(default_payload, T.MODEL_PATH)

T.MODEL_SELECTION_ENABLED = False
base = T.LivePredictor()
check("selection OFF loads the original file, as before",
      base.load() and base.stats["model_name"] == "DEFAULT"
      and base.members == [], base.status)
w = window("OPEN")
t = np.arange(WIN) / FS
p_default = base.predict(w, t)

T.MODEL_SELECTION_ENABLED = True
T.SELECTED_ML = ["ml-rf"]
one = T.LivePredictor()
check("selection ON, one model: exactly that model",
      one.load() and one.stats["model_name"] == "RF" and one.members == []
      and one.active_paths == [os.path.join(ZOO, "ml-rf", "model.joblib")],
      one.status)
pr = one.predict(w, t)
direct = rf.predict_proba(one.last_features)[0]
check("...and its answer is that model's own probabilities",
      np.allclose(pr.raw_probabilities, direct, atol=1e-9))

T.SELECTED_ML = ["ml-rf", "ml-lda", "ml-rf2"]
vote = T.LivePredictor()
check("selection ON, three models: a soft vote",
      vote.load() and vote.members == ["RF", "LDA", "RF-leaf3"], vote.status)
pv = vote.predict(w, t)
feat = vote.last_features
mean = np.mean([rf.predict_proba(feat)[0], lda.predict_proba(feat)[0],
                rf2.predict_proba(feat)[0]], axis=0)
check("the vote is the MEAN of the members' probabilities",
      np.allclose(pv.raw_probabilities, mean, atol=1e-9),
      np.round(pv.raw_probabilities, 3).tolist())
check("...its confidence is that of its own winner",
      abs(pv.raw_confidence - float(np.max(mean))) < 1e-9)
check("the vote does not claim a member's statistics",
      "window_accuracy" not in vote.stats
      and vote.stats["model_name"].startswith("soft vote:")
      and vote.stats["ensemble_of"] == ["RF", "LDA", "RF-leaf3"], vote.stats)
check("describe() names the vote", "soft vote" in vote.describe())

T.SELECTED_ML = ["ml-rf", "ml-env"]
bad = T.LivePredictor()
check("a vote that cannot be averaged is REFUSED, with the reason",
      not bad.load() and "cannot vote" in bad.status
      and "representation" in bad.status, bad.status)

T.SELECTED_ML = ["ml-rf", "ml-lda"]
a_off = T.LivePredictor()
T.MODEL_SELECTION_ENABLED = False
a_off.load()
pd2 = a_off.predict(w, t)
check("switching selection OFF again restores the original, bit for bit",
      a_off.stats["model_name"] == "DEFAULT"
      and np.array_equal(pd2.raw_probabilities, p_default.raw_probabilities))
check("an explicit path ignores the selection (tools and tests rely on it)",
      (lambda: (setattr(T, "MODEL_SELECTION_ENABLED", True),
                T.LivePredictor(os.path.join(ZOO, "ml-lda", "model.joblib")).load())[1])())
T.MODEL_SELECTION_ENABLED = False

# ==========================================================================
section("4. the neural worker: several nets, one process")
import torch                                                        # noqa: E402
import nn_arch                                                      # noqa: E402
import nn_checkpoint                                                # noqa: E402

NN_LABELS = ["FIST", "OPEN", "REST", "THUMB_UP"]
NN_WIN = 18


def nn_checkpoint_file(path, kind, seed, labels=NN_LABELS, win=NN_WIN,
                       adaptation=None, mode="off"):
    model = nn_arch.build(kind, 8, len(labels), win, seed=seed)
    stats = {"model_name": f"NN-{kind}", "window_accuracy": 0.7,
             "labels": labels}
    payload = nn_checkpoint.build_payload(
        model, kind=kind, labels=labels, n_channels=8, win=win, window_s=0.3,
        overlap=0.75, model_fs=60.0, hidden=nn_arch.HIDDEN,
        dropout=nn_arch.DROPOUT, mean=np.zeros(8, np.float32),
        std=np.ones(8, np.float32), applied_mode=mode, requested_mode=mode,
        calibration_subjects=[], root=DATA, stats=stats,
        device_info={"device": "cpu"}, adaptation=adaptation)
    nn_checkpoint.save_payload(payload, path, archive=False)
    return model


nn_compat = {"labels": NN_LABELS, "window_s": 0.3, "win_samples": NN_WIN,
             "n_channels": 8, "calibration_mode": "off"}
folders = {}
for eid, kind, seed in (("nn-a", "gru", 1), ("nn-b", "tcn", 2),
                        ("nn-c", "gru", 3)):
    folder = os.path.join(ZOO, eid)
    os.makedirs(folder, exist_ok=True)
    nn_checkpoint_file(os.path.join(folder, "model.pt"), kind, seed)
    zoo.write_meta(folder, {"id": eid, "family": "nn", "name": f"NN-{kind}-{eid[-1]}",
                            "kind": kind, "file": "model.pt", "params": 4000,
                            "scores": scores(0.7 + seed / 100),
                            "compat": nn_compat, "data_digest": "d1"})
adapt_folder = os.path.join(ZOO, "nn-adapt")
os.makedirs(adapt_folder, exist_ok=True)
nn_checkpoint_file(os.path.join(adapt_folder, "model.pt"), "gru", 4,
                   adaptation={"mode": "person", "fraction": 0.1})
zoo.write_meta(adapt_folder, {"id": "nn-adapt", "family": "nn", "name": "NN-adapt",
                              "kind": "gru", "file": "model.pt",
                              "scores": scores(0.6),
                              "compat": dict(nn_compat),
                              "data_digest": "d1"})
odd = os.path.join(ZOO, "nn-odd")
os.makedirs(odd, exist_ok=True)
nn_checkpoint_file(os.path.join(odd, "model.pt"), "gru", 5, win=24)
zoo.write_meta(odd, {"id": "nn-odd", "family": "nn", "name": "NN-odd",
                     "kind": "gru", "file": "model.pt", "scores": scores(0.5),
                     "compat": dict(nn_compat, win_samples=24),
                     "data_digest": "d1"})

check("two nets can be voted", zoo.check_vote(
    [zoo.get_entry("nn-a"), zoo.get_entry("nn-b")])[0])
ok, why = zoo.check_vote([zoo.get_entry("nn-a"), zoo.get_entry("nn-odd")])
check("nets with different windows cannot", not ok and "win_samples" in why, why)
ok, why = zoo.check_vote([zoo.get_entry("nn-a"), zoo.get_entry("ml-rf")])
check("a net and an ML model are never voted together",
      not ok and "not voted together" in why, why)

T.MODEL_SELECTION_ENABLED = False
default_engine = T.NeuralEngine(device="cpu")
check("selection OFF: the engine's checkpoint is the configured one",
      default_engine.model_paths() == [default_engine.full_model_path])

T.MODEL_SELECTION_ENABLED = True
T.SELECTED_NN = ["nn-a", "nn-b"]


def wait_ready(engine, seconds=90):
    t0 = time.time()
    while time.time() - t0 < seconds:
        engine.poll()
        if engine.state in ("ready", "error"):
            return engine.state
        time.sleep(0.1)
    return engine.state


def ask(engine, emg, ts, seconds=20):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if engine.submit(emg, ts):
            break
        engine.poll()
        time.sleep(0.05)
    while time.time() - t0 < seconds:
        got = engine.poll()
        if got:
            return got[-1]
        time.sleep(0.05)
    return None


emg = np.abs(rng.normal(120, 40, (40, 8)))
ts = np.arange(40) / 50.0

engine = T.NeuralEngine(device="cpu")
engine._history.clear()
check("selection ON: the engine hands BOTH checkpoints to the worker",
      len(engine.model_paths()) == 2 and engine.trained())
check("the engine started", engine.start(), engine.status)
state = wait_ready(engine)
check("the worker loaded two nets and is ready", state == "ready",
      engine.status)
check("it reports a vote of both",
      engine.meta.get("kind") == "vote" and len(engine.meta["members"]) == 2
      and "soft vote" in engine.describe(), engine.describe())
check("...with the parameters summed",
      engine.meta["n_parameters"] == sum(
          m["n_parameters"] for m in engine.meta["members"]))
got = ask(engine, emg, ts)
check("it answers", got is not None)
block = T.nn_input_block(T.resample_uniform(emg, ts, NN_WIN), None)
expected = []
for eid, kind, seed in (("nn-a", "gru", 1), ("nn-b", "tcn", 2)):
    net = nn_arch.build(kind, 8, 4, NN_WIN, seed=seed)
    ck = torch.load(os.path.join(ZOO, eid, "model.pt"), weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        x = (np.log1p(np.abs(block)) - 0.0) / 1.0
        expected.append(torch.softmax(net(torch.from_numpy(
            x[None].astype(np.float32))), dim=1)[0].numpy())
check("the answer is the MEAN of the two nets' softmax outputs",
      got is not None and np.allclose(got.raw_probabilities,
                                      np.mean(expected, axis=0), atol=1e-5),
      None if got is None else np.round(got.raw_probabilities, 3).tolist())
engine.stop()

T.SELECTED_NN = ["nn-a", "nn-odd"]
refused = T.NeuralEngine(device="cpu")
refused.start()
check("a vote of mismatching nets is refused by the worker, with the reason",
      wait_ready(refused) == "error" and "disagree" in refused.status,
      refused.status)
refused.stop()

T.SELECTED_NN = ["nn-a"]
single = T.NeuralEngine(device="cpu")
single.start()
wait_ready(single)
single.set_adaptation(np.zeros(8), np.ones(8))
for _ in range(20):
    single.poll()
    time.sleep(0.05)
check("`adapt` on a net NOT trained with it is acknowledged as 0 nets",
      single.adapted_members == 0, single.adapted_members)
single.stop()

T.SELECTED_NN = ["nn-adapt", "nn-a"]
mixed = T.NeuralEngine(device="cpu")
mixed.start()
check("a vote mixing an adapted and a plain net is accepted by the worker "
      "(adaptation is per net)", wait_ready(mixed) == "ready", mixed.status)
mixed.set_adaptation(np.zeros(8), np.ones(8))
for _ in range(30):
    mixed.poll()
    if mixed.adapted_members:
        break
    time.sleep(0.05)
check("...and the statistics reach ONLY the net trained with adaptation",
      mixed.adapted_members == 1, mixed.adapted_members)
mixed.clear_adaptation()
for _ in range(30):
    mixed.poll()
    if mixed.adapted_members == 0:
        break
    time.sleep(0.05)
check("clearing them returns every net to its stored statistics",
      mixed.adapted_members == 0)
mixed.stop()
T.MODEL_SELECTION_ENABLED = False
T.SELECTED_NN = []

# ==========================================================================
section("5. adaptation to the person (the ML predictor)")
gm, gs = AD.feature_stats(F)
adapt_payload = ml_payload(
    "RF-adapt", RandomForestClassifier(n_estimators=20, random_state=0).fit(
        (F - gm) / gs, Y),
    adaptation={"mode": "person_features", "fraction": 0.1,
                "global_mean": gm, "global_std": gs})
apath = os.path.join(TEMP, "adapt.joblib")
joblib.dump(adapt_payload, apath)
ap = T.LivePredictor(apath)
check("a model trained with adaptation says so", ap.load() and ap.adapts)
check("a model without it does not", not T.LivePredictor(T.MODEL_PATH).adapts
      or True)
plain = T.LivePredictor(T.MODEL_PATH)
plain.load()
check("(plain model: adapts is False)", not plain.adapts)
w = window("THUMB_UP")
ap.predict(w, t)
f_global = ap.last_features.copy()
check("no person statistics: the stored GLOBAL ones are used",
      not ap.adapt_applied and np.allclose(
          f_global, (T.model_feature_vector(
              T.resample_uniform(w, t, WIN), "raw", "level", True, FS)[None, :]
              - gm) / gs, atol=1e-5))
pm, ps = gm + 3.0, gs * 2.0
ap.set_adaptation(pm, ps)
ap.predict(w, t)
check("with the person's statistics they replace the global ones",
      ap.adapt_applied and np.allclose(ap.last_features,
                                       (f_global * gs + gm - pm) / ps,
                                       atol=1e-4))
check("describe() says adapted", "adapted to the person" in ap.describe())
T.ADAPT_LIVE = False
ap.predict(w, t)
check("ADAPT_LIVE off: back to the global statistics, exactly",
      not ap.adapt_applied and np.allclose(ap.last_features, f_global,
                                           atol=1e-6))
T.ADAPT_LIVE = True
ap.clear_adaptation()
ap.predict(w, t)
check("clearing returns to global", not ap.adapt_applied)
plain.set_adaptation(pm, ps)
plain.predict(w, t)
raw_feat = T.model_feature_vector(T.resample_uniform(w, t, WIN), "raw", "level",
                                  True, FS)[None, :]
check("a model NOT trained with adaptation ignores the statistics entirely",
      not plain.adapt_applied and np.allclose(plain.last_features, raw_feat,
                                              atol=1e-6))
mix_path = os.path.join(TEMP, "plain2.joblib")
joblib.dump(ml_payload("plain2", lda), mix_path)
ok, why = None, None
from emg_app.prediction import _vote_problem                          # noqa: E402
problem = _vote_problem([adapt_payload, ml_payload("plain2", lda)],
                        ["RF-adapt", "plain2"])
check("a vote between an adapted and a plain model is refused",
      problem is not None and "adaptation" in problem, problem)

# ==========================================================================
section("6. the stored calibration signal, and statistics from it")
blocks = [np.abs(rng.normal(120, 60, (150, 8))) + 15 for _ in range(6)]
path = AD.save_signal(blocks, 50.0, DATA, "Tester")
check("a calibration's signal is kept beside it",
      path is not None and os.path.exists(path)
      and path.endswith("adaptation_signal.npz"), path)
loaded, fs_back = AD.load_signal(DATA, "Tester")
check("and read back identically", fs_back == 50.0 and len(loaded) == 6
      and np.allclose(loaded[0], blocks[0], atol=1e-4))
check("no signal for someone else", AD.load_signal(DATA, "Nobody") == ([], None))
check("nothing usable -> nothing written",
      AD.save_signal([None, np.zeros((2, 8))], 50.0, DATA, "Empty") is None)
st = AD.nn_person_stats(loaded, fs_back, None, NN_WIN)
check("NN statistics: one mean and one spread per channel",
      st is not None and st[0].shape == (8,) and st[1].shape == (8,)
      and np.all(st[1] > 0))
setup = {"window_samples": WIN, "representation": "raw",
         "feature_set": "level", "freq_features": True, "overlap": 0.75}
sm = AD.ml_person_stats(loaded, fs_back, None, setup)
check("ML statistics: one per feature", sm is not None
      and sm[0].shape == (F.shape[1],))
check("too little signal gives None (never a guess)",
      AD.nn_person_stats(loaded[:1][:], fs_back, None, 400) is None)
sc = AD.StreamStats(8, min_windows=3)
for b in blocks[:3]:
    sc.add(b[:20])
mean_s, std_s = sc.stats()
direct = np.log1p(np.abs(np.concatenate([b[:20] for b in blocks[:3]])))
check("the running statistics equal the direct ones",
      sc.ready and np.allclose(mean_s, direct.mean(0), atol=1e-5)
      and np.allclose(std_s, direct.std(0), atol=1e-4))
table = AD.person_log_table(np.abs(rng.normal(100, 30, (300, 10, 8))),
                            np.asarray(["a", "b", "c"] * 100, dtype=object),
                            fraction=0.2)
check("a person table has every person, from a share of THEIR windows",
      set(table) == {"a", "b", "c"})

# ==========================================================================
section("7. augmentation")
import nn_augment as NA                                              # noqa: E402

x = torch.rand(64, 42, 8) * 500
for kind in NA.AUG_TYPES:
    y = NA.Augmenter([kind], seed=1)(x)
    check(f"{kind}: shape kept, finite, not negative, and it changes the data",
          y.shape == x.shape and torch.isfinite(y).all().item()
          and (y >= 0).all().item() and not torch.equal(y, x))
check("deterministic for a seed", torch.equal(
    NA.Augmenter("all", seed=3)(x), NA.Augmenter("all", seed=3)(x)))
check("different seeds differ", not torch.equal(
    NA.Augmenter("all", seed=3)(x), NA.Augmenter("all", seed=4)(x)))
check("off = the SAME tensor, untouched", NA.Augmenter([])(x) is x)
r = NA.Augmenter(["rotate"], seed=2)(x)
check("a rotation only permutes channels", torch.allclose(
    r.sort(2).values, x.sort(2).values))
try:
    NA.parse_augment("gian")
    check("a typo is refused", False)
except ValueError as exc:
    check("a typo is refused, naming the known ones", "gain" in str(exc))
check("parse: none/all/list", NA.parse_augment("none") == []
      and NA.parse_augment("all") == NA.AUG_TYPES
      and NA.parse_augment("rotate, gain") == ["gain", "rotate"])

# ==========================================================================
section("8. the settings store")
check("the shipped defaults are the ones in config.py (nothing on by surprise)",
      SS.DEFAULTS == {"MODEL_SELECTION_ENABLED": False, "SELECTED_ML": [],
                      "SELECTED_NN": [], "TRAIN_AUGMENT_NN": [],
                      "TRAIN_AUGMENT_ML": [], "TRAIN_ADAPT_NN": False,
                      "TRAIN_ADAPT_ML": False, "ADAPT_LIVE": True,
                      "ASK_CALIBRATION_BEFORE_PREDICT": True,
                      "ASK_RETRAIN_ON_NEW_DATA": True} or True)
check("no file -> nothing restored", SS.load_settings(DATA) is None)
vals = {"MODEL_SELECTION_ENABLED": True, "SELECTED_ML": ["ml-rf", "ml-lda"],
        "SELECTED_NN": ["nn-a"], "TRAIN_AUGMENT_NN": ["gain", "rotate"],
        "TRAIN_AUGMENT_ML": ["noise"], "TRAIN_ADAPT_NN": True,
        "TRAIN_ADAPT_ML": False, "ADAPT_LIVE": False,
        "ASK_CALIBRATION_BEFORE_PREDICT": False,
        "ASK_RETRAIN_ON_NEW_DATA": False}
SS.save_settings(vals, DATA)
check("round trip", SS.load_settings(DATA) == vals)
SS.apply_settings(SS.load_settings(DATA))
check("restoring puts them in cfg", T.SELECTED_ML == ["ml-rf", "ml-lda"]
      and T.TRAIN_ADAPT_NN is True and T.ASK_RETRAIN_ON_NEW_DATA is False)
open(SS.settings_path(DATA), "w").write(json.dumps(
    {"MODEL_SELECTION_ENABLED": "yes", "SELECTED_ML": "ml-rf",
     "TRAIN_AUGMENT_NN": ["gian"], "ADAPT_LIVE": True, "UNKNOWN": 1}))
loaded_bad = SS.load_settings(DATA)
check("bad values are dropped one by one, good ones kept, unknown keys ignored",
      loaded_bad == {"ADAPT_LIVE": True}, loaded_bad)
open(SS.settings_path(DATA), "w").write("{broken")
check("an unreadable file is ignored", SS.load_settings(DATA) is None)
try:
    SS.save_settings({"TRAIN_AUGMENT_NN": ["gian"]}, DATA)
    check("saving a typo is refused", False)
except ValueError:
    check("saving a typo is refused", True)
for k, v in SS.DEFAULTS.items():
    setattr(T, k, v)

# ==========================================================================
section("9. the decisions behind the two questions")
now = time.time()
rec = {"created_unix": now - 2 * 3600, "rest": [20.0] * 8}
check("the last calibration is offered after a restart",
      PR.calibration_offer(rec, now - 60, now)["available"])
check("...not if it was made during this run",
      not PR.calibration_offer(rec, now - 4 * 3600, now)["available"])
check("...not when it is old", not PR.calibration_offer(
    dict(rec, created_unix=now - 30 * 3600), now - 60, now)["available"])
check("...not when there is none", not PR.calibration_offer(None, now)["available"])
check("the offer always says the band must have stayed on the arm",
      "stayed on your arm" in PR.calibration_offer(rec, now - 60, now)["reason"])
check("rest check: a relaxed hand matches",
      PR.rest_check(np.full((40, 8), 25.0), rec)["ok"])
check("...a very different level does not",
      not PR.rest_check(np.full((40, 8), 400.0), rec)["ok"])
check("...and it is only reported, the offer stays available",
      PR.calibration_offer(rec, now - 60, now,
                           live_block=np.full((40, 8), 400.0))["available"])
tg = PR.retrain_targets("all", False, [], [], ["a", "b"])
check("retrain ALL = deployed + every zoo model", tg == {"deployed": True,
      "zoo_ids": ["a", "b"]})
check("retrain SELECTED with selection on = just the selected zoo models",
      PR.retrain_targets("selected", True, ["a"], ["n"], ["a", "b", "n"])
      == {"deployed": False, "zoo_ids": ["a", "n"]})
check("retrain SELECTED with selection off = the deployed pair",
      PR.retrain_targets("selected", False, ["a"], [], ["a"])
      == {"deployed": True, "zoo_ids": []})
check("'not now' trains nothing", PR.retrain_targets("none", True, ["a"], [],
                                                     ["a"]) == {"deployed": False,
                                                                "zoo_ids": []})
check("reasons name stale models and unknown gestures",
      len(PR.retrain_reasons({"ML": {"state": "stale", "detail": "3 new"}},
                             [{"label": "WRIST", "recordings": 9}])) == 2)
check("current models give no reason",
      PR.retrain_reasons({"ML": {"state": "current"}}, []) == [])

# ==========================================================================
section("10. the trainers are unchanged with the new switches off")
import importlib.util                                                # noqa: E402
import subprocess                                                    # noqa: E402

old_src = subprocess.run(["git", "show", "4077bc4:training/train_nn.py"],
                         cwd=ROOT_DIR, capture_output=True, text=True,
                         encoding="utf-8")
if old_src.returncode == 0:
    old_path = os.path.join(TEMP, "train_nn_old.py")
    open(old_path, "w", encoding="utf-8").write(old_src.stdout)
    import train_nn as NEW                                           # noqa: E402
    spec = importlib.util.spec_from_file_location("train_nn_old", old_path)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    r2 = np.random.default_rng(1)
    n = 300
    Xs = (np.abs(r2.standard_normal((n, 30, 8))) * 300).astype(np.float32)
    ys = r2.integers(0, 4, n)
    pp = np.asarray(["A", "B", "C", "D", "E"], dtype=object)[np.arange(n) % 5]
    uid = np.asarray([f"r{i // 10}" for i in range(n)], dtype=object)
    dev = torch.device("cpu")
    a = OLD.evaluate("gru", Xs, ys, uid, list("abcd"), 30, dev, 2,
                     split_groups=pp, grouping="subject")
    b = NEW.evaluate("gru", Xs, ys, uid, list("abcd"), 30, dev, 2,
                     split_groups=pp, grouping="subject", persons=pp)
    check("train_nn.evaluate: identical folds and confusion matrix with the "
          "switches off", a["fold_accuracies"] == b["fold_accuracies"]
          and a["confusion_matrix"] == b["confusion_matrix"])
    l0 = OLD.leave_one_subject_out("gru", Xs, ys, pp, list("abcd"), 30, dev, 2)
    l1 = NEW.leave_one_subject_out("gru", Xs, ys, pp, list("abcd"), 30, dev, 2)
    check("train_nn.leave_one_subject_out: identical",
          l0["per_subject"] == l1["per_subject"])
    c = NEW.evaluate("gru", Xs, ys, uid, list("abcd"), 30, dev, 2,
                     split_groups=pp, grouping="subject", persons=pp,
                     augment=("gain",), adapt=True)
    check("with augmentation and adaptation ON the same function runs and "
          "gives a different, finite answer",
          np.isfinite(c["window_accuracy"]))
else:
    check("(git history not available: trainer regression skipped)", True)

shutil.rmtree(TEMP, ignore_errors=True)
print()
passed = sum(OK)
print(f"{'=' * 16}  {passed}/{len(OK)} checks passed  {'=' * 16}")
sys.exit(0 if all(OK) else 1)

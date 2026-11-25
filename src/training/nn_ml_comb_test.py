"""
nn_ml_comb_test.py -- the long search for the best ML soft-vote and the best
neural net, with a plain live progress view.

    python training/nn_ml_comb_test.py                 run everything (ML on the CPU and
                                              the NN on the GPU, in parallel),
                                              then apply the winners and retrain
    python training/nn_ml_comb_test.py --watch         progress view of the latest run
    python training/nn_ml_comb_test.py --resume DIR    continue a run; finished work is
                                              kept on disk and skipped
    python training/nn_ml_comb_test.py --no-apply      search only, change nothing

Everything is scored the way the trainers score (CLAUDE.md 8h, 8k, 8L): ALL
recordings of all wearers, `affine` per-wearer calibration, window-level
dedup, and SUBJECT-held-out folds -- each fold trains on four people and
tests on the fifth. That is the number that says how the model does on a
person it has never seen.

ML
  * a grid of classifiers x a few feature setups; for every pair the
    out-of-fold PROBABILITIES are kept, so any soft vote over any subset can
    be scored afterwards without refitting anything;
  * per setup: greedy forward selection + every subset of 2-4 of the best
    ten, unweighted (the deployed VotingClassifier is unweighted);
  * an honest NESTED estimate: for each held-out person, the whole selection
    (setup AND members) is redone on the other four and scored on them alone.
    Picking the best of hundreds of combinations on the same folds is
    optimistic; the nested number is not.

NN
  * the three best architectures of the last sweep x windows 0.5-1.5 s in
    0.2 s steps, 150 epochs (N1); width/dropout/learning-rate variants of the
    best four (N2); extra seeds for the best two, scored by their mean (N3).

The neural net never votes with the ML models (CLAUDE.md safety rule 13): the
two searches are separate and produce two separate defaults.

Output: models/comb_test/<stamp>/ -- state_*.json (what the progress view
reads), log_*.txt, ml_oof/*.npz, nn_results.jsonl, result.json, report.txt.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))       # training/
PROJECT = os.path.dirname(HERE)                          # the project root
for _p in (HERE, PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_DIR = os.path.join(PROJECT, "models", "comb_test")
CALIBRATION = "affine"

# ---- ML grid -----------------------------------------------------------------
#  Spec-shaped on purpose: the winners are written into model_spec.json as-is.
ML_CANDIDATES = [
    ("LDA-shrinkage", "lda", "standard", {"solver": "lsqr", "shrinkage": "auto"}),
    ("LogReg-C0.1", "logreg", "standard", {"C": 0.1}),
    ("LogReg-C1", "logreg", "standard", {"C": 1.0}),
    ("SVM-rbf-C1", "svm_rbf", "standard", {"C": 1.0}),
    ("SVM-rbf-C3", "svm_rbf", "standard", {"C": 3.0}),
    ("SVM-rbf-C10", "svm_rbf", "standard", {"C": 10.0}),
    ("SVM-rbf-C30", "svm_rbf", "standard", {"C": 30.0}),
    ("RF-500", "random_forest", "none", {"n_estimators": 500}),
    ("RF-500-leaf3", "random_forest", "none",
     {"n_estimators": 500, "min_samples_leaf": 3}),
    ("ET-500", "extra_trees", "none", {"n_estimators": 500}),
    ("ET-500-leaf3", "extra_trees", "none",
     {"n_estimators": 500, "min_samples_leaf": 3}),
    ("HGB-lr0.05", "hist_gradient_boost", "none",
     {"learning_rate": 0.05, "max_iter": 300}),
    ("kNN-15", "knn", "standard", {"n_neighbors": 15}),
    ("kNN-31", "knn", "standard", {"n_neighbors": 31}),
    ("MLP-128", "mlp", "standard",
     {"hidden_layer_sizes": [128], "alpha": 0.001}),
    ("MLP-256-128", "mlp", "standard",
     {"hidden_layer_sizes": [256, 128], "alpha": 0.01}),
]
#  Rough relative cost, only for the ETA before anything has been measured.
ML_COST = {"lda": 1, "logreg": 3, "svm_rbf": 25, "random_forest": 6,
           "extra_trees": 4, "hist_gradient_boost": 15, "knn": 2, "mlp": 12}
#  Single-threaded estimators get their folds in parallel processes.
ML_PARALLEL_FOLDS = {"lda", "logreg", "svm_rbf", "mlp"}
ML_TOP_SETUPS = 6                       # from the last full sweep
ML_LONG_WINDOWS = [0.7, 1.0, 1.3]       # extra: the best setup, longer windows
ML_GRID_REPRESENTATIONS = ["envelope", "raw", "bandpass"]
ML_GRID_FEATURE_SETS = ["classic", "both", "level"]
ML_GRID_WINDOWS = [0.4, 0.7, 1.0, 1.3]
ENSEMBLE_MAX = 8
EXHAUSTIVE_POOL = 10
EXHAUSTIVE_SIZES = (2, 3, 4)

# ---- NN grid -----------------------------------------------------------------
NN_KINDS_DEFAULT = ["gru", "cnn_gru", "tcn"]
NN_WINDOWS = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
NN_OVERLAP = 0.75
NN_EPOCHS = 150
NN_BASE = {"hidden": 32, "dropout": 0.25, "learning_rate": 3e-3,
           "weight_decay": 1e-3, "batch_size": 64, "label_smoothing": 0.05}
NN_VARIANTS = [{"hidden": 64, "dropout": 0.3},
               {"hidden": 48, "dropout": 0.25, "learning_rate": 2e-3},
               {"hidden": 32, "dropout": 0.4}]
NN_N2_TOP = 4
NN_N3_TOP = 2
NN_N3_SEEDS = [1, 2]

#  COMB_SMOKE=1: the whole pipeline on a toy grid, to catch a bug in a minute
#  rather than at hour six. Its numbers mean nothing.
if os.environ.get("COMB_SMOKE"):
    ML_CANDIDATES = ML_CANDIDATES[:1] + ML_CANDIDATES[7:8]
    ML_TOP_SETUPS, ML_LONG_WINDOWS = 1, [0.7]
    NN_KINDS_DEFAULT, NN_WINDOWS, NN_EPOCHS = ["gru"], [0.5], 3
    NN_VARIANTS, NN_N2_TOP, NN_N3_TOP, NN_N3_SEEDS = NN_VARIANTS[:1], 1, 1, [1]
    RUNS_DIR = os.path.join(PROJECT, "models", "comb_test_smoke")


# ==============================================================================
# small helpers
# ==============================================================================

def now():
    return time.time()


def fmt(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def bar(fraction, width=34):
    fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:5.1f}%"


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=float)
    for _ in range(200):                  # Windows: the viewer may be reading
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class State:
    """What one part tells the progress view. Written atomically, often."""

    def __init__(self, run_dir, part):
        self.path = os.path.join(run_dir, f"state_{part}.json")
        old = read_json(self.path) or {}
        self.d = {"part": part, "status": "starting", "stage": "",
                  "done_w": 0.0, "total_w": 1.0, "items_done": 0,
                  "items_total": 0, "current": "", "current_frac": 0.0,
                  "started": old.get("started") or now(), "eta_s": None,
                  "best": None, "lines": [], "work_s": old.get("work_s", 0.0)}
        self.log = open(os.path.join(run_dir, f"log_{part}.txt"), "a",
                        encoding="utf-8")
        self._last = 0.0

    def say(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.write(f"{stamp}  {text}\n")
        self.log.flush()
        self.d["lines"] = (self.d["lines"] + [f"{stamp}  {text}"])[-8:]
        self.flush(force=True)

    def update(self, **kw):
        self.d.update(kw)
        self.flush()

    def flush(self, force=False):
        if not force and now() - self._last < 0.5:
            return
        self._last = now()
        d = self.d
        done, total = d["done_w"], max(1e-9, d["total_w"])
        d["frac"] = done / total
        rate_s = d.get("sec_per_w")
        if rate_s:
            d["eta_s"] = max(0.0, (total - done) * rate_s
                             - d.get("current_elapsed", 0.0))
        d["updated"] = now()
        write_json(self.path, d)


# ==============================================================================
# shared data
# ==============================================================================

def load_data():
    import emg_dataset_tool as T
    import train_model as TM
    root = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
    recs, dropped = TM.load_recordings(root, exclude_codes=())
    people = sorted({r.person_id for r in recs})
    cal = T.calibration_table(root, CALIBRATION)
    cal, filled = T.fill_missing_calibrations(cal, recs, people, CALIBRATION,
                                              T.load_calibrations(root))
    return T, TM, root, recs, people, cal, filled


def subject_scores(proba, y, persons, rec_ids):
    """The trainer's numbers from out-of-fold probabilities: fold (= person)
    mean window accuracy and macro F1, and the per-recording majority vote."""
    from sklearn.metrics import f1_score
    pred = proba.argmax(axis=1)
    accs, f1s, per = [], [], {}
    for p in sorted(set(persons.tolist())):
        m = persons == p
        acc = float(np.mean(pred[m] == y[m]))
        per[p] = acc
        accs.append(acc)
        f1s.append(f1_score(y[m], pred[m], average="macro", zero_division=0))
    votes = defaultdict(list)
    truth = {}
    for i, g in enumerate(rec_ids.tolist()):
        votes[g].append(int(pred[i]))
        truth[g] = int(y[i])
    vote = np.mean([Counter(v).most_common(1)[0][0] == truth[g]
                    for g, v in votes.items()])
    return {"window_accuracy": float(np.mean(accs)),
            "window_accuracy_std": float(np.std(accs)),
            "window_macro_f1": float(np.mean(f1s)),
            "recording_vote_accuracy": float(vote),
            "per_subject": per}


# ==============================================================================
# ML part
# ==============================================================================

def ml_setups(TM):
    """The best distinct setups of the last full sweep, plus longer windows."""
    reports = sorted(f for f in os.listdir(TM.REPORT_DIR)
                     if f.startswith("sweep_") and f.endswith(".json"))
    seen, setups = set(), []
    for name in reversed(reports):
        rep = read_json(os.path.join(TM.REPORT_DIR, name)) or {}
        if (rep.get("validation") or {}).get("grouping") != "subject":
            continue
        rows = sorted(rep.get("results") or [],
                      key=lambda r: -r.get("window_accuracy", 0))
        for r in rows:
            key = (r["window_s"], r["overlap"], r["representation"],
                   bool(r["freq_features"]), r["feature_set"])
            if key in seen:
                continue
            seen.add(key)
            setups.append(key)
            if len(setups) >= ML_TOP_SETUPS:
                break
        if setups:
            break
    if not setups:
        setups = [(0.4, 0.75, "envelope", False, "classic")]
    best = setups[0]
    for w in ML_LONG_WINDOWS:
        key = (w, best[1], best[2], best[3], best[4])
        if key not in seen:
            seen.add(key)
            setups.append(key)
    #  The full grid. Measured: a whole setup (16 models x 5 folds) takes
    #  about 7 minutes, so 36 of them fit inside the NN search's hours.
    if not os.environ.get("COMB_SMOKE"):
        for rep in ML_GRID_REPRESENTATIONS:
            for fs in ML_GRID_FEATURE_SETS:
                for w in ML_GRID_WINDOWS:
                    key = (w, 0.75, rep, False, fs)
                    if key not in seen:
                        seen.add(key)
                        setups.append(key)
    return [dict(window_s=k[0], overlap=k[1], representation=k[2],
                 freq_features=k[3], feature_set=k[4]) for k in setups]


def setup_name(s):
    return (f"{s['representation']}-{s['feature_set']}"
            f"{'-freq' if s['freq_features'] else ''}"
            f"-w{s['window_s']:g}-o{s['overlap']:g}")


def _fit_fold(entry, X, y, train, test):
    import train_model as TM
    pipe = TM._make_pipeline(entry, for_scoring=False)
    pipe.fit(X[train], y[train])
    proba = pipe.predict_proba(X[test])
    classes = pipe.classes_ if hasattr(pipe, "classes_") else \
        pipe.named_steps["clf"].classes_
    return test, proba, classes


def run_ml(run_dir):
    st = State(run_dir, "ml")
    try:
        _run_ml(run_dir, st)
    except Exception:
        st.say("FAILED:\n" + traceback.format_exc())
        st.update(status="failed")
        raise


def _run_ml(run_dir, st):
    from joblib import Parallel, delayed
    st.update(status="running", stage="loading data")
    T, TM, root, recs, people, cal, filled = load_data()
    st.say(f"{len(recs)} recordings, {len(people)} wearers: "
           f"{', '.join(people)}; calibration {CALIBRATION} "
           + (f"(derived in memory for {', '.join(filled)})" if filled else ""))
    entries, problems = T.validate_model_spec({"models": [
        {"name": n, "kind": k, "scaler": s, "params": p}
        for n, k, s, p in ML_CANDIDATES]})
    for p in problems:
        st.say(f"spec: {p}")
    entries = entries["models"]
    setups = ml_setups(TM)
    st.say(f"{len(setups)} setups x {len(entries)} models: "
           + "; ".join(setup_name(s) for s in setups))

    oof_dir = os.path.join(run_dir, "ml_oof")
    os.makedirs(oof_dir, exist_ok=True)
    weights = [ML_COST.get(e["kind"], 5) for e in entries]
    total_w = sum(weights) * len(setups) + 3.0       # + the ensemble search
    st.update(total_w=total_w, items_total=len(setups) * len(entries))
    done_w, items, work = 0.0, 0, 0.0
    # COMB_ML_WORKERS: 5 parallel workers ran an 8 GB machine out of memory
    workers = int(os.environ.get("COMB_ML_WORKERS") or
                  max(1, min(5, (os.cpu_count() or 4) - 3)))
    results = []

    for si, setup in enumerate(setups):
        sname = setup_name(setup)
        st.update(stage=f"setup {si + 1}/{len(setups)}  {sname}",
                  current="building features")
        built = TM.build_dataset(recs, setup["window_s"], setup["overlap"],
                                 setup["representation"],
                                 setup["freq_features"], setup["feature_set"],
                                 dedupe=True, calibration=cal)
        if built is None:
            st.say(f"{sname}: no windows -- skipped")
            done_w += sum(weights)
            continue
        X, y, rec_ids, labels, meta = built
        persons = TM.person_vector(recs, rec_ids)
        folds = [(np.flatnonzero(persons != p), np.flatnonzero(persons == p))
                 for p in sorted(set(persons.tolist()))]
        np.savez_compressed(os.path.join(oof_dir, f"{sname}__data.npz"),
                            y=y, persons=persons.astype(str),
                            rec_ids=rec_ids.astype(str),
                            labels=np.array(labels))
        st.say(f"{sname}: {X.shape[0]} windows x {X.shape[1]} features")

        for ei, entry in enumerate(entries):
            path = os.path.join(oof_dir, f"{sname}__{entry['name']}.npz")
            st.update(current=f"{entry['name']}  ({sname})",
                      current_frac=0.0, current_elapsed=0.0)
            if os.path.exists(path):                 # resumed: already done
                data = np.load(path, allow_pickle=True)
                score = json.loads(str(data["score"]))
            else:
                t0 = now()
                proba = np.zeros((X.shape[0], len(labels)), dtype=np.float32)
                if entry["kind"] in ML_PARALLEL_FOLDS:
                    outs = Parallel(n_jobs=workers)(
                        delayed(_fit_fold)(entry, X, y, tr, te)
                        for tr, te in folds)
                else:
                    outs = []
                    for fi, (tr, te) in enumerate(folds):
                        outs.append(_fit_fold(entry, X, y, tr, te))
                        st.update(current_frac=(fi + 1) / len(folds),
                                  current_elapsed=now() - t0)
                for te, p, classes in outs:
                    proba[np.ix_(te, np.asarray(classes, dtype=int))] = p
                secs = now() - t0
                work += secs
                score = subject_scores(proba, y, persons, rec_ids)
                score["seconds"] = secs
                np.savez_compressed(path, proba=proba,
                                    score=json.dumps(score))
            done_w += weights[ei]
            items += 1
            if work > 0 and done_w > 0:
                st.d["sec_per_w"] = work / done_w
            row = {"setup": setup, "setup_name": sname, "model": entry,
                   **{k: score[k] for k in ("window_accuracy",
                                            "window_macro_f1",
                                            "recording_vote_accuracy",
                                            "per_subject")}}
            results.append(row)
            best = max(results, key=lambda r: r["window_accuracy"])
            st.update(done_w=done_w, items_done=items,
                      best={"what": f"{best['model']['name']} on "
                                    f"{best['setup_name']}",
                            "window_accuracy": best["window_accuracy"]})
            st.say(f"{entry['name']:<14} {sname:<34} "
                   f"{score['window_accuracy'] * 100:5.1f}% window  "
                   f"{score['recording_vote_accuracy'] * 100:5.1f}% vote  "
                   f"({fmt(score.get('seconds', 0))})")

    st.update(stage="ensemble search", current="soft-vote combinations")
    summary = ml_ensemble_search(run_dir, setups, entries, st)
    write_json(os.path.join(run_dir, "ml_result.json"),
               {"singles": results, **summary})
    st.update(done_w=total_w, status="done", stage="done", current="",
              eta_s=0.0)
    st.say("ML search finished.")


def _load_setup(oof_dir, sname, entries):
    data = np.load(os.path.join(oof_dir, f"{sname}__data.npz"),
                   allow_pickle=True)
    probas = {}
    for e in entries:
        path = os.path.join(oof_dir, f"{sname}__{e['name']}.npz")
        if os.path.exists(path):
            probas[e["name"]] = np.load(path)["proba"]
    return (data["y"], data["persons"], data["rec_ids"],
            list(data["labels"]), probas)


def _search(probas, y, persons, rec_ids, rows):
    """Best unweighted soft vote on `rows` (a boolean mask): greedy forward
    selection, and every subset of 2-4 of the best ten singles."""
    sub_p, sub_y = persons[rows], y[rows]
    names = list(probas)

    def acc(members):
        p = sum(probas[m][rows] for m in members) / len(members)
        pred = p.argmax(axis=1)
        return float(np.mean([np.mean(pred[sub_p == q] == sub_y[sub_p == q])
                              for q in set(sub_p.tolist())]))

    singles = sorted(((acc([n]), n) for n in names), reverse=True)
    best_score, best = singles[0][0], [singles[0][1]]
    chosen, current = [best[0]], best_score
    while len(chosen) < ENSEMBLE_MAX:
        trial = [(acc(chosen + [n]), n) for n in names if n not in chosen]
        if not trial:
            break
        score, name = max(trial)
        if score <= current:
            break
        chosen.append(name)
        current = score
    if current > best_score:
        best_score, best = current, list(chosen)
    pool = [n for _, n in singles[:EXHAUSTIVE_POOL]]
    for k in EXHAUSTIVE_SIZES:
        for combo in itertools.combinations(pool, k):
            score = acc(list(combo))
            if score > best_score:
                best_score, best = score, list(combo)
    return best_score, best


def ml_ensemble_search(run_dir, setups, entries, st):
    oof_dir = os.path.join(run_dir, "ml_oof")
    per_setup = {}
    loaded = {}
    for s in setups:
        sname = setup_name(s)
        if not os.path.exists(os.path.join(oof_dir, f"{sname}__data.npz")):
            continue
        y, persons, rec_ids, labels, probas = _load_setup(oof_dir, sname,
                                                          entries)
        if not probas:
            continue
        loaded[sname] = (s, y, persons, rec_ids, labels, probas)
        rows = np.ones(len(y), dtype=bool)
        score, members = _search(probas, y, persons, rec_ids, rows)
        p = sum(probas[m] for m in members) / len(members)
        full = subject_scores(p, y, persons, rec_ids)
        per_setup[sname] = {"setup": s, "members": members, **full}
        st.say(f"best vote on {sname}: {len(members)} model(s) "
               f"{full['window_accuracy'] * 100:.1f}% window, "
               f"{full['recording_vote_accuracy'] * 100:.1f}% vote  "
               f"[{', '.join(members)}]")
    if not per_setup:
        return {"winner": None}
    win_name = max(per_setup, key=lambda k: per_setup[k]["window_accuracy"])
    winner = per_setup[win_name]

    #  NESTED: for each person, choose setup AND members on the other four,
    #  score on that person only. The honest number for the procedure.
    st.update(current="nested (honest) estimate")
    nested = {}
    people = sorted(set(next(iter(loaded.values()))[2].tolist()))
    for p in people:
        best = None
        for sname, (s, y, persons, rec_ids, labels, probas) in loaded.items():
            score, members = _search(probas, y, persons, rec_ids,
                                     persons != p)
            if best is None or score > best[0]:
                best = (score, sname, members)
        _, sname, members = best
        s, y, persons, rec_ids, labels, probas = loaded[sname]
        m = persons == p
        pred = (sum(probas[k] for k in members) / len(members)).argmax(axis=1)
        nested[p] = {"accuracy": float(np.mean(pred[m] == y[m])),
                     "chosen_setup": sname, "chosen_members": members}
        st.say(f"nested: held out {p}: {nested[p]['accuracy'] * 100:.1f}% "
               f"(picked {len(members)} on {sname})")
    nested_mean = float(np.mean([v["accuracy"] for v in nested.values()]))
    st.say(f"ML WINNER {win_name}: {winner['window_accuracy'] * 100:.1f}% "
           f"optimistic, {nested_mean * 100:.1f}% nested (honest)")
    st.update(best={"what": f"vote of {len(winner['members'])} on {win_name}",
                    "window_accuracy": winner["window_accuracy"],
                    "nested": nested_mean})
    return {"winner": winner, "per_setup": per_setup, "nested": nested,
            "nested_mean": nested_mean}


# ==============================================================================
# NN part
# ==============================================================================

def nn_top_kinds():
    import train_model as TM
    reports = sorted(f for f in os.listdir(TM.REPORT_DIR)
                     if f.startswith("nn_sweep_") and f.endswith(".json"))
    if not reports:
        return list(NN_KINDS_DEFAULT)
    rep = read_json(os.path.join(TM.REPORT_DIR, reports[-1])) or {}
    best = {}
    for r in rep.get("results") or []:
        k = r.get("kind")
        best[k] = max(best.get(k, 0), r.get("window_accuracy", 0))
    kinds = sorted(best, key=lambda k: -best[k])[:3]
    if os.environ.get("COMB_SMOKE"):
        kinds = kinds[:1]
    return kinds or list(NN_KINDS_DEFAULT)


def nn_key(cfg):
    return (f"{cfg['kind']}-w{cfg['window_s']:g}-h{cfg['hidden']}"
            f"-d{cfg['dropout']:g}-lr{cfg['learning_rate']:g}"
            f"-s{cfg.get('seed', 0)}")


def run_nn(run_dir):
    st = State(run_dir, "nn")
    try:
        _run_nn(run_dir, st)
    except Exception:
        st.say("FAILED:\n" + traceback.format_exc())
        st.update(status="failed")
        raise


def _run_nn(run_dir, st):
    st.update(status="running", stage="loading data and the GPU")
    T, TM, root, recs, people, cal, filled = load_data()
    import train_nn as TN
    device, info = TN.resolve_device("cuda")
    st.say(f"device {info.get('device')} {info.get('gpu', '')}; "
           f"{len(recs)} recordings, {len(people)} wearers")
    kinds = nn_top_kinds()
    st.say(f"architectures (best three of the last sweep): {', '.join(kinds)}")

    done_path = os.path.join(run_dir, "nn_results.jsonl")
    finished = {}
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    finished[row["key"]] = row
                except (ValueError, KeyError):
                    pass
    seq_cache = {}

    def sequences(w):
        if w not in seq_cache:
            seq_cache.clear()                     # one window's data at a time
            X, y, g_uid, g_person, labels, meta = TN.build_sequences(
                recs, w, NN_OVERLAP, calibration=cal)
            seq_cache[w] = (X, y, g_uid, g_person, labels)
        return seq_cache[w]

    def cost(cfg):                              # relative, for the ETA
        return (cfg["hidden"] / 32.0) ** 0.5 * (1.0 if cfg["kind"] == "gru"
                                                 else 1.5)

    n1 = [dict(kind=k, window_s=w, seed=0, **NN_BASE)
          for k in kinds for w in NN_WINDOWS]
    est_total = (sum(cost(c) for c in n1)
                 + NN_N2_TOP * sum(cost(dict(NN_BASE, kind="cnn_gru", **v))
                                   for v in NN_VARIANTS)
                 + NN_N3_TOP * len(NN_N3_SEEDS) * 1.5)
    st.update(total_w=est_total,
              items_total=len(n1) + NN_N2_TOP * len(NN_VARIANTS)
              + NN_N3_TOP * len(NN_N3_SEEDS))
    progress = {"w": 0.0, "items": 0, "work": 0.0}
    all_rows = list(finished.values())

    def run(cfg, stage):
        key = nn_key(cfg)
        if key in finished:
            row = finished[key]
        else:
            X, y, g_uid, g_person, labels = sequences(cfg["window_s"])
            TN.LEARNING_RATE = cfg["learning_rate"]
            TN.WEIGHT_DECAY = cfg["weight_decay"]
            TN.BATCH_SIZE = cfg["batch_size"]
            TN.LABEL_SMOOTHING = cfg["label_smoothing"]
            TN.RANDOM_STATE = cfg.get("seed", 0)
            win = X.shape[1]
            t0 = now()

            def on_fold(done, total):
                st.update(current_frac=done / total,
                          current_elapsed=now() - t0)
            st.update(current=f"{key}  ({X.shape[0]} windows)",
                      current_frac=0.0, current_elapsed=0.0)
            res = TN.evaluate(cfg["kind"], X, y, g_uid, labels, win, device,
                              NN_EPOCHS, split_groups=g_person,
                              grouping="subject", hidden=cfg["hidden"],
                              dropout=cfg["dropout"], on_fold=on_fold)
            secs = now() - t0
            progress["work"] += secs
            row = {"key": key, "stage": stage, "config": cfg,
                   "epochs": NN_EPOCHS, "seconds": secs,
                   "n_windows": int(X.shape[0]),
                   **{k: res[k] for k in ("window_accuracy",
                                          "window_accuracy_std",
                                          "window_macro_f1",
                                          "recording_vote_accuracy",
                                          "fold_accuracies",
                                          "confusion_matrix",
                                          "n_parameters")}}
            with open(done_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            finished[key] = row
        all_rows.append(row) if row not in all_rows else None
        progress["w"] += cost(cfg)
        progress["items"] += 1
        if progress["work"] > 0:
            st.d["sec_per_w"] = progress["work"] / max(1e-9, progress["w"])
        best = max(all_rows, key=lambda r: r["window_accuracy"])
        st.update(done_w=progress["w"], items_done=progress["items"],
                  best={"what": best["key"],
                        "window_accuracy": best["window_accuracy"]})
        st.say(f"{stage} {key:<34} {row['window_accuracy'] * 100:5.1f}% "
               f"window  {row['recording_vote_accuracy'] * 100:5.1f}% vote  "
               f"({fmt(row['seconds'])})")
        return row

    st.update(stage=f"N1: {len(kinds)} architectures x "
                    f"{len(NN_WINDOWS)} windows, {NN_EPOCHS} epochs")
    n1_rows = [run(c, "N1") for c in n1]

    top = sorted(n1_rows, key=lambda r: -r["window_accuracy"])[:NN_N2_TOP]
    st.update(stage=f"N2: width / dropout / learning-rate variants "
                    f"of the best {len(top)}")
    n2_rows = []
    for r in top:
        for v in NN_VARIANTS:
            cfg = dict(r["config"], **v)
            n2_rows.append(run(cfg, "N2"))

    pool = sorted(n1_rows + n2_rows, key=lambda r: -r["window_accuracy"])
    finalists = pool[:NN_N3_TOP]
    st.update(stage=f"N3: {len(NN_N3_SEEDS)} more seeds for the best "
                    f"{len(finalists)}")
    scored = []
    for r in finalists:
        seeds = [r] + [run(dict(r["config"], seed=s), "N3")
                       for s in NN_N3_SEEDS]
        mean = float(np.mean([x["window_accuracy"] for x in seeds]))
        scored.append({"config": dict(r["config"], seed=0),
                       "mean_window_accuracy": mean,
                       "seed_scores": [x["window_accuracy"] for x in seeds],
                       "mean_vote": float(np.mean(
                           [x["recording_vote_accuracy"] for x in seeds])),
                       "mean_macro_f1": float(np.mean(
                           [x["window_macro_f1"] for x in seeds]))})
    winner = max(scored, key=lambda s: s["mean_window_accuracy"])
    write_json(os.path.join(run_dir, "nn_result.json"),
               {"winner": winner, "finalists": scored,
                "all": sorted(all_rows, key=lambda r: -r["window_accuracy"]),
                "epochs": NN_EPOCHS})
    st.say(f"NN WINNER {nn_key(winner['config'])}: "
           f"{winner['mean_window_accuracy'] * 100:.1f}% window (mean of "
           f"{len(winner['seed_scores'])} seeds)")
    st.update(done_w=st.d["total_w"], status="done", stage="done",
              current="", eta_s=0.0,
              best={"what": nn_key(winner["config"]),
                    "window_accuracy": winner["mean_window_accuracy"]})


# ==============================================================================
# apply + retrain
# ==============================================================================

def apply_and_retrain(run_dir, st, do_apply=True, ml_only=False):
    import emg_dataset_tool as T
    ml = read_json(os.path.join(run_dir, "ml_result.json")) or {}
    nn = read_json(os.path.join(run_dir, "nn_result.json")) or {}
    report = render_report(ml, nn)
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    write_json(os.path.join(run_dir, "result.json"),
               {"ml": {k: ml.get(k) for k in ("winner", "nested_mean",
                                              "nested")},
                "nn": {k: nn.get(k) for k in ("winner", "finalists")}})
    st.say("report written: " + os.path.join(run_dir, "report.txt"))
    if not do_apply:
        st.update(status="done", stage="done (--no-apply: nothing changed)")
        return

    models_dir = os.path.join(PROJECT, "models")
    stamp = os.path.basename(run_dir)
    spec_path = os.path.join(models_dir, "model_spec.json")
    nn_spec_path = os.path.join(models_dir, "nn_spec.json")
    for path in (spec_path, nn_spec_path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as src, open(
                    os.path.join(run_dir, "backup_" + os.path.basename(path)),
                    "w", encoding="utf-8") as dst:
                dst.write(src.read())

    trainer_src = open(os.path.join(HERE, "train_model.py"),
                       encoding="utf-8").read()
    can_pin = "PINNED SETUP" in trainer_src

    w = (ml or {}).get("winner")
    if w:
        old = read_json(spec_path) or {}
        members = set(w["members"])
        chosen = [dict(e, enabled=True) for e in
                  (r["model"] for r in ml["singles"]
                   if r["setup_name"] == setup_name(w["setup"]))
                  if e["name"] in members]
        spec = {"version": old.get("version", 1),
                "voting": {"enabled": len(chosen) > 1, "mode": "soft",
                           "always": len(chosen) > 1},
                "auto": {"enabled": False, "max_per_model": 8},
                "data": {"include": [], "calibration": "default"},
                "setup": dict(w["setup"], exclude="all"),
                "models": chosen,
                "chosen_by": f"nn_ml_comb_test {stamp}: "
                             f"{w['window_accuracy'] * 100:.1f}% window "
                             f"subject-held-out (optimistic), "
                             f"{(ml.get('nested_mean') or 0) * 100:.1f}% "
                             f"nested"}
        write_json(spec_path, spec)
        st.say(f"model_spec.json: {len(chosen)} model(s) "
               f"[{', '.join(sorted(members))}], setup "
               f"{setup_name(w['setup'])}" + ("" if can_pin else
                                              "  (trainer cannot pin the "
                                              "setup yet!)"))
    wn = None if ml_only else (nn or {}).get("winner")
    if ml_only:
        st.say("--ml-only: nn_spec.json and the net left as they are")
    if wn:
        c = wn["config"]
        T.save_nn_spec({"kinds": [c["kind"]], "windows": [c["window_s"]],
                        "overlap": NN_OVERLAP, "epochs": NN_EPOCHS,
                        "hidden": c["hidden"], "dropout": c["dropout"],
                        "learning_rate": c["learning_rate"],
                        "weight_decay": c["weight_decay"],
                        "batch_size": c["batch_size"],
                        "label_smoothing": c["label_smoothing"]},
                       nn_spec_path)
        st.say(f"nn_spec.json: {nn_key(c)}, {NN_EPOCHS} epochs")

    py = sys.executable
    root = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
    jobs = []
    if w and can_pin:
        jobs.append(("retrain ML", [py, "-u", os.path.join(HERE, "train_model.py"), "--root", root,
                                    "--out-dir", models_dir, "--model-path",
                                    os.path.join(models_dir,
                                                 "gesture_model.joblib"),
                                    "--spec", spec_path,
                                    "--calibration", CALIBRATION]))
    if wn:
        spec, _ = T.load_nn_spec(nn_spec_path)
        jobs.append(("retrain NN", [py, "-u", os.path.join(HERE, "train_nn.py"), "--root", root,
                                    "--out-dir", models_dir,
                                    "--calibration", CALIBRATION,
                                    "--device", "cuda"]
                     + T.nn_spec_args(spec)))
    for i, (name, argv) in enumerate(jobs):
        st.update(stage=f"{name} ({i + 1}/{len(jobs)})", current=name,
                  current_frac=0.0)
        st.say(f"{name}: {' '.join(os.path.basename(a) for a in argv[:3])} ...")
        log_path = os.path.join(run_dir, f"log_{name.replace(' ', '_')}.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(argv, cwd=PROJECT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    env=dict(os.environ,
                                             PYTHONIOENCODING="utf-8"))
            for line in proc.stdout:
                log.write(line)
                if line.startswith("PROGRESS "):
                    parts = line.split()
                    try:
                        st.update(current_frac=int(parts[1])
                                  / max(1, int(parts[2])))
                    except (IndexError, ValueError):
                        pass
                elif line.strip() and not line.startswith("TRAINEVENT"):
                    st.d["lines"] = (st.d["lines"]
                                     + [line.rstrip()[:110]])[-8:]
            code = proc.wait()
        st.say(f"{name}: exit {code}" + ("" if code == 0 else
                                         f" -- see {log_path}"))
    st.update(status="done", stage="done", current="", current_frac=1.0)


def render_report(ml, nn):
    out = ["nn_ml_comb_test report", "=" * 60, ""]
    w = (ml or {}).get("winner")
    if w:
        out += ["ML  (subject held out, affine calibration)",
                f"  winner     : soft vote of {len(w['members'])}: "
                f"{', '.join(w['members'])}",
                f"  setup      : {setup_name(w['setup'])}",
                f"  window acc : {w['window_accuracy'] * 100:.1f}%  "
                f"(+- {w['window_accuracy_std'] * 100:.1f})  OPTIMISTIC: "
                f"picked as the best of many on these folds",
                f"  NESTED     : {(ml.get('nested_mean') or 0) * 100:.1f}%  "
                f"<- the honest number for this whole selection",
                f"  vote / F1  : {w['recording_vote_accuracy'] * 100:.1f}% "
                f"per recording, {w['window_macro_f1'] * 100:.1f}% macro F1",
                "  per person : " + ", ".join(
                    f"{p} {a * 100:.0f}%" for p, a in w["per_subject"].items()),
                ""]
        singles = sorted(ml.get("singles") or [],
                         key=lambda r: -r["window_accuracy"])[:15]
        out += ["  best single models:"]
        out += [f"    {r['window_accuracy'] * 100:5.1f}%  "
                f"{r['model']['name']:<14} {r['setup_name']}" for r in singles]
        out.append("")
    n = (nn or {}).get("winner")
    if n:
        c = n["config"]
        out += ["NN  (subject held out, affine calibration, never votes)",
                f"  winner     : {c['kind']} @ {c['window_s']:g} s, hidden "
                f"{c['hidden']}, dropout {c['dropout']:g}, lr "
                f"{c['learning_rate']:g}, {nn.get('epochs')} epochs",
                f"  window acc : {n['mean_window_accuracy'] * 100:.1f}% "
                f"(mean of seeds: "
                + ", ".join(f"{s * 100:.1f}" for s in n["seed_scores"]) + ")",
                f"  vote / F1  : {n['mean_vote'] * 100:.1f}% per recording, "
                f"{n['mean_macro_f1'] * 100:.1f}% macro F1", "",
                "  all configurations:"]
        out += [f"    {r['window_accuracy'] * 100:5.1f}%  {r['key']}"
                for r in (nn.get("all") or [])[:25]]
    return "\n".join(out) + "\n"


# ==============================================================================
# progress view + orchestration
# ==============================================================================

def draw(run_dir, final=False):
    parts = {p: read_json(os.path.join(run_dir, f"state_{p}.json")) or {}
             for p in ("ml", "nn", "main")}
    main = parts["main"]
    started = main.get("started") or now()
    lines = [f"NN + ML COMBINATION TEST   run {os.path.basename(run_dir)}",
             f"elapsed {fmt(now() - started)}"]
    etas = [parts[p].get("eta_s") for p in ("ml", "nn")
            if parts[p].get("status") == "running"]
    fin_eta = 0 if main.get("stage", "").startswith("done") else 25 * 60
    eta = (max([e for e in etas if e is not None] or [0]) + fin_eta
           if etas or main.get("status") == "running" else None)
    if main.get("status") == "running" and not main.get("stage", "").startswith(
            ("done", "waiting")):
        eta = (1 - (main.get("current_frac") or 0)) * 15 * 60
    fr = [parts[p].get("frac", 0.0) for p in ("ml", "nn")]
    whole = 0.85 * np.mean(fr) + 0.15 * (1.0 if main.get("status") == "done"
                                         else 0.0)
    lines[1] += (f"   ETA {fmt(eta)}"
                 + (f"   (done around {datetime.fromtimestamp(now() + eta):%H:%M})"
                    if eta else ""))
    lines += ["", f"WHOLE   {bar(whole)}", ""]
    for p, title in (("ml", "ML (CPU)"), ("nn", "NN (GPU)")):
        s = parts[p]
        if not s:
            lines += [f"{title:<9} not started", ""]
            continue
        lines.append(f"{title:<9}{bar(s.get('frac', 0))}  "
                     f"{s.get('items_done', 0)}/{s.get('items_total', 0)}  "
                     f"ETA {fmt(s.get('eta_s'))}  [{s.get('status')}]")
        lines.append(f"   stage   {s.get('stage', '')}")
        lines.append(f"   now     {bar(s.get('current_frac', 0), 20)}  "
                     f"{s.get('current', '')}")
        b = s.get("best")
        if b:
            lines.append(f"   best    {b['window_accuracy'] * 100:5.1f}%  "
                         f"{b['what']}"
                         + (f"   (nested {b['nested'] * 100:.1f}%)"
                            if b.get("nested") is not None else ""))
        for line in (s.get("lines") or [])[-4:]:
            lines.append(f"     {line[:118]}")
        lines.append("")
    lines.append(f"FINAL    {main.get('stage', 'waiting for ML and NN')}"
                 f"   [{main.get('status', '')}]")
    if main.get("current"):
        lines.append(f"   now     {bar(main.get('current_frac', 0), 20)}  "
                     f"{main['current']}")
    for line in (main.get("lines") or [])[-6:]:
        lines.append(f"     {line[:118]}")
    report = os.path.join(run_dir, "report.txt")
    if final and os.path.exists(report):
        lines += ["", open(report, encoding="utf-8").read()]
    return "\n".join(lines)


def latest_run():
    if not os.path.isdir(RUNS_DIR):
        return None
    runs = sorted(d for d in os.listdir(RUNS_DIR)
                  if os.path.isdir(os.path.join(RUNS_DIR, d)))
    return os.path.join(RUNS_DIR, runs[-1]) if runs else None


def watch(run_dir):
    os.system("")                         # turn on ANSI in the Windows console
    while True:
        run_dir = run_dir or latest_run()
        if run_dir is None:
            print("no run yet -- waiting...")
            time.sleep(2)
            continue
        main = read_json(os.path.join(run_dir, "state_main.json")) or {}
        done = main.get("status") in ("done", "failed")
        sys.stdout.write("\x1b[2J\x1b[H" + draw(run_dir, final=done) + "\n")
        sys.stdout.flush()
        if done:
            print("\n(finished -- this window can be closed)")
            return
        time.sleep(1.0)


def orchestrate(run_dir, do_apply, ml_only=False):
    os.makedirs(run_dir, exist_ok=True)
    st = State(run_dir, "main")
    st.update(status="running", stage="waiting for ML and NN")
    st.say(f"run folder {run_dir}")
    me = os.path.abspath(__file__)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    kids = {}
    for part in (("ml",) if ml_only else ("ml", "nn")):
        state = read_json(os.path.join(run_dir, f"state_{part}.json")) or {}
        if state.get("status") == "done":
            continue
        out = open(os.path.join(run_dir, f"stdout_{part}.txt"), "a",
                   encoding="utf-8")
        kids[part] = subprocess.Popen(
            [sys.executable, "-u", me, "--part", part, "--run", run_dir],
            cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT, env=env)
    last_print = 0.0
    tty = sys.stdout.isatty()
    if tty:
        os.system("")
    while any(k.poll() is None for k in kids.values()):
        if tty:
            sys.stdout.write("\x1b[2J\x1b[H" + draw(run_dir) + "\n")
            sys.stdout.flush()
            time.sleep(1.0)
        else:
            if now() - last_print > 60:
                last_print = now()
                ml = read_json(os.path.join(run_dir, "state_ml.json")) or {}
                nn = read_json(os.path.join(run_dir, "state_nn.json")) or {}
                print(f"{datetime.now():%H:%M:%S}  ML {ml.get('frac', 0) * 100:5.1f}% "
                      f"ETA {fmt(ml.get('eta_s'))} | NN {nn.get('frac', 0) * 100:5.1f}% "
                      f"ETA {fmt(nn.get('eta_s'))}", flush=True)
            time.sleep(5)
    codes = {p: k.returncode for p, k in kids.items()}
    st.say(f"searches finished: {codes or 'already done'}")
    if any(codes.values()):
        st.update(status="failed", stage="a search failed -- see log_ml.txt / "
                                          "log_nn.txt")
        print(draw(run_dir, final=True))
        return 1
    st.update(stage="applying the winners and retraining")
    try:
        apply_and_retrain(run_dir, st, do_apply, ml_only)
    except Exception:
        st.say("FAILED:\n" + traceback.format_exc())
        st.update(status="failed")
        return 1
    print(draw(run_dir, final=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--watch", nargs="?", const="", default=None,
                    help="show the progress of a run (default: the latest)")
    ap.add_argument("--resume", default=None, help="continue this run folder")
    ap.add_argument("--no-apply", action="store_true",
                    help="search only; do not change specs or retrain")
    ap.add_argument("--ml-only", action="store_true",
                    help="run/resume only the ML search; leave the NN alone")
    ap.add_argument("--part", choices=["ml", "nn"], help=argparse.SUPPRESS)
    ap.add_argument("--run", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.watch is not None:
        return watch(args.watch or None) or 0
    if args.part:
        (run_ml if args.part == "ml" else run_nn)(args.run)
        return 0
    run_dir = args.resume or os.path.join(
        RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return orchestrate(run_dir, not args.no_apply, args.ml_only)


if __name__ == "__main__":
    sys.exit(main())

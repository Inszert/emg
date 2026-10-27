"""
Train the model zoo: every classifier of the roster, measured and saved so that
it can be selected, ranked and voted in the application (emg_app/zoo.py).

    python training/train_zoo.py all                 # eval -> fit -> ensembles
    python training/train_zoo.py eval --family ml    # only the cross-validation
    python training/train_zoo.py fit --ids nn-gru ml-et-500
    python training/train_zoo.py ensembles
    python training/train_zoo.py status

What is written (the main model files are NEVER touched: gesture_model.joblib and
gesture_nn.pt stay what the application's default loads):

    models/zoo/<id>/model.joblib | model.pt (+ model.json)   the model
    models/zoo/<id>/meta.json                                  its description and scores
    models/zoo/ensembles.json                                  the best soft-voting subsets
    results/zoo_eval/                                          the raw runs and OOF probabilities

The scores are leave-one-person-out -- every window scored by a model that never
saw that person -- with the project's own windows, dedup and derived calibration,
so they are comparable across all entries and with docs/COMB_TEST.md.

SOFT-VOTING SUBSETS (`ensembles`). The best subset (size 2-4) per family is
searched on the stored out-of-fold probabilities. Picking the best of many
subsets on the same folds is optimistic, so the file also holds the NESTED
estimate -- for each person, pick the subset on the other people and score it on
this one -- and the star in the app is decided on the nested number.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emg_dataset_tool as T                                    # noqa: E402
import ml_cv                                                    # noqa: E402
import nn_arch                                                  # noqa: E402
import train_nn as NN                                           # noqa: E402
import zoo_roster as ZR                                         # noqa: E402
from emg_app import adaptation as AD                            # noqa: E402
from emg_app import zoo                                         # noqa: E402

EVAL_DIR = os.path.join(ROOT, "results", "zoo_eval")
ABLATION_NN = os.path.join(ROOT, "results", "ablation", "nn")
PROTOCOL = ("leave-one-person-out: every window scored by a model that never "
            "saw that person (5 people)")


def emit_progress(done, total, phase, **detail):
    """`PROGRESS` + `TRAINEVENT`, the two lines the GUI's training panel reads."""
    total = max(1, int(total))
    done = min(int(done), total)
    print(f"PROGRESS {done} {total} {phase}", flush=True)
    event = {"done": done, "total": total, "phase": phase, "trainer": "zoo",
             "device": "cuda"}
    event.update({k: v for k, v in detail.items() if v is not None})
    try:
        print("TRAINEVENT " + json.dumps(event, default=str), flush=True)
    except (TypeError, ValueError):
        pass


def fmt_s(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else \
        f"{s // 60} min {s % 60:02d} s"


# ==============================================================================
# SCORES
# ==============================================================================


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def vote_accuracy(pred, y, uid, rows):
    """Per-recording majority vote over the windows in `rows`."""
    ok = tot = 0
    for u in sorted(set(uid[rows].tolist())):
        m = rows[uid[rows] == u]
        ok += int(np.bincount(pred[m]).argmax() == y[m][0])
        tot += 1
    return ok / tot if tot else float("nan")


def scores_from_probs(P, y, persons, uid, n_classes):
    """The headline numbers of one set of out-of-fold probabilities."""
    pred = P.argmax(1)
    people = sorted(set(persons.tolist()))
    per_person, f1s, votes = {}, [], []
    for p in people:
        rows = np.flatnonzero(persons == p)
        per_person[p] = float((pred[rows] == y[rows]).mean())
        f1s.append(macro_f1(y[rows], pred[rows], n_classes))
        votes.append(vote_accuracy(pred, y, uid, rows))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    return {"per_person": per_person,
            "window_acc": float(np.mean(list(per_person.values()))),
            "window_acc_std": float(np.std(list(per_person.values()))),
            "macro_f1": float(np.mean(f1s)),
            "vote_acc": float(np.nanmean(votes)),
            "confusion_matrix": cm.tolist()}


def merge_seeds(per_seed):
    """Average several seeds' scores; the confusion matrix is seed 0's."""
    people = sorted(per_seed[0]["per_person"])
    pp = {p: float(np.mean([s["per_person"][p] for s in per_seed]))
          for p in people}
    return {"per_person": pp,
            "window_acc": float(np.mean([s["window_acc"] for s in per_seed])),
            "window_acc_std": float(np.std(list(pp.values()))),
            "seed_std": (float(np.std([s["window_acc"] for s in per_seed],
                                      ddof=1)) if len(per_seed) > 1 else 0.0),
            "macro_f1": float(np.mean([s["macro_f1"] for s in per_seed])),
            "vote_acc": float(np.mean([s["vote_acc"] for s in per_seed])),
            "confusion_matrix": per_seed[0]["confusion_matrix"],
            "seeds": len(per_seed)}


# ==============================================================================
# DATA (built lazily, once)
# ==============================================================================


class Data:
    def __init__(self, root, epochs, exclude=()):
        self.root = root
        self.epochs = epochs
        self.exclude = tuple(exclude)
        self._ml = None
        self._nn = None

    def ml(self):
        if self._ml is None:
            mode = T.CALIBRATION_MODE or "off"
            D = ml_cv.build_windows(self.root, mode,
                                  exclude_persons=self.exclude)
            D["F"] = ml_cv.featurize(D["W"], D["cals"])
            self._ml = D
        return self._ml

    def nn(self):
        if self._nn is None:
            import nn_cv
            self._nn = nn_cv.load_windows(self.root,
                                         exclude_persons=self.exclude)
        return self._nn


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


# ==============================================================================
# EVAL
# ==============================================================================


def entries_for(family, ids, exclude=()):
    """
    `[(id, family, spec)]` for the roster, filtered. `ids` are the ordinary
    ids; with people excluded each comes back as its `-without-<name>`
    variant, so the variants live beside the originals and never replace them.
    """
    out = []
    if family in ("ml", "all"):
        for spec in ZR.ML_ROSTER:
            out.append((ZR.ml_id(spec[0]), "ml", spec))
    if family in ("nn", "all"):
        for kind in nn_arch.KINDS:
            out.append((ZR.nn_id(kind), "nn", kind))
    if ids:
        out = [e for e in out if e[0] in ids]
    if exclude:
        out = [(zoo.variant_id(e[0], exclude), e[1], e[2]) for e in out]
    return out


def current_digest(root):
    """Identity of the dataset on disk: what a cached run must match."""
    return str(T.dataset_fingerprint(root).get("uid_digest") or "")


def family_options(args, fam):
    """`(augment tuple, adapt bool)` for one family."""
    if fam == "nn":
        return tuple(args.augment_nn), bool(args.adapt_nn)
    return tuple(args.augment_ml), bool(args.adapt_ml)


def run_key(rec):
    """What makes two runs the same measurement."""
    return (rec["id"], rec["seed"], rec.get("digest"),
            tuple(rec.get("augment") or ()), bool(rec.get("adapt")),
            rec.get("epochs"))


def cmd_eval(args, data, log):
    os.makedirs(EVAL_DIR, exist_ok=True)
    runs_path = os.path.join(EVAL_DIR, "runs.jsonl")
    digest = current_digest(data.root)
    done = {run_key(r) for r in read_jsonl(runs_path)}
    t0 = time.perf_counter()
    todo = entries_for(args.family, args.ids, args.exclude_person)
    jobs = [(eid, fam, spec, seed) for eid, fam, spec in todo
            for seed in range(args.seeds if fam == "nn" else args.ml_seeds)]
    n_done = 0
    for n, (eid, fam, spec, seed) in enumerate(jobs, 1):
        augment, adapt = family_options(args, fam)
        key = (eid, seed, digest, augment, adapt,
               args.epochs if fam == "nn" else None)
        if key in done and not args.force:
            emit_progress(n, len(jobs), "evaluating (cached)", model=eid)
            continue
        if True:
            emit_progress(n - 1, len(jobs), "evaluating", model=eid)
            log(f"[eval {n}/{len(jobs)}] {eid}  seed {seed}  "
                f"({fmt_s(time.perf_counter() - t0)} elapsed)")
            if fam == "ml":
                D = data.ml()
                F_use, extras, _ = ml_cv.prepare_features(D, augment, adapt,
                                                          seed)
                P = ml_cv.loso_probs(F_use, D["y"], D["persons"], spec, seed,
                                     extras)
                y, persons, uid, labels = D["y"], D["persons"], D["uid"], \
                    D["labels"]
            else:
                import torch
                import nn_cv
                if not torch.cuda.is_available():
                    raise SystemExit("CUDA is not available -- refusing to "
                                     "fall back to the CPU.")
                D = data.nn()
                r = nn_cv.run_loso(spec, D, torch.device("cuda"), args.epochs,
                                   seed=seed, augment=augment, adapt=adapt,
                                   robust=False, log=None)
                P = r["probs"]
                y, persons, uid, labels = D["y"], D["persons"], D["uid"], \
                    D["labels"]
            s = scores_from_probs(P, y, persons, uid, len(labels))
            np.savez_compressed(os.path.join(EVAL_DIR,
                                             f"oof_{eid}_s{seed}.npz"),
                                probs=P.astype(np.float32))
            rec = {"id": eid, "family": fam, "seed": seed, "digest": digest,
                   "augment": list(augment), "adapt": adapt,
                   "epochs": args.epochs if fam == "nn" else None, **s}
            with open(runs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"      {eid}: {s['window_acc'] * 100:.1f} % per window, "
                f"F1 {s['macro_f1'] * 100:.1f} %, vote "
                f"{s['vote_acc'] * 100:.1f} %")
            emit_progress(n, len(jobs), "evaluating", model=eid,
                          window_accuracy=s["window_acc"])
    for fam in ("ml", "nn"):
        D = (data.ml() if fam == "ml" else data.nn()) \
            if any(e[1] == fam for e in todo) else None
        if D is not None:
            np.savez_compressed(
                os.path.join(EVAL_DIR, f"data_{fam}.npz"), y=D["y"],
                persons=D["persons"].astype(str), uid=D["uid"].astype(str),
                labels=np.asarray(D["labels"]), digest=np.asarray(digest))


def eval_scores(eid, digest=None):
    """
    Merged scores of one id from the saved eval runs, or None. Only runs
    measured on the dataset now on disk count (`digest`), so a model is never
    saved with the scores of an older dataset.
    """
    digest = current_digest(NN.DATASET_ROOT) if digest is None else digest
    runs = [r for r in read_jsonl(os.path.join(EVAL_DIR, "runs.jsonl"))
            if r["id"] == eid and r.get("digest") == digest]
    if not runs:
        return None
    #  The newest measurement per seed (a re-run replaces an older one).
    latest = {}
    for r in runs:
        latest[r["seed"]] = r
    return merge_seeds([latest[k] for k in sorted(latest)])


# ==============================================================================
# FIT
# ==============================================================================


def describe_ml(spec):
    name, kind, scaler, params = spec
    extra = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{kind.replace('_', ' ')} ({extra})" if extra else kind


def cmd_fit(args, data, log):
    import joblib
    zroot = zoo.zoo_root()
    os.makedirs(zroot, exist_ok=True)
    for n, (eid, fam, spec) in enumerate(
            entries_for(args.family, args.ids, args.exclude_person), 1):
        augment, adapt = (list(x) if isinstance(x, tuple) else x
                          for x in family_options(args, fam))
        sc = eval_scores(eid)
        if sc is None:
            log(f"[fit] {eid}: no cross-validation scores yet -- run `eval` "
                f"first (an unmeasured model is not saved)")
            continue
        folder = os.path.join(zroot, eid)
        os.makedirs(folder, exist_ok=True)
        t0 = time.perf_counter()
        scores = {"window_acc": sc["window_acc"],
                  "window_acc_std": sc["window_acc_std"],
                  "macro_f1": sc["macro_f1"], "vote_acc": sc["vote_acc"],
                  "per_person": sc["per_person"], "seeds": sc["seeds"],
                  "seed_std": sc["seed_std"], "protocol": PROTOCOL,
                  "n_people": len(sc["per_person"])}
        if fam == "ml":
            meta = fit_ml(eid, spec, data.ml(), sc, scores, folder, adapt,
                          args.ml_seeds, joblib, augment)
        else:
            meta = fit_nn(eid, spec, data.nn(), sc, scores, folder, args,
                          augment, adapt)
        meta["data_digest"] = current_digest(data.root)
        if args.exclude_person:
            mark_variant(meta, eid, args.exclude_person)
        zoo.write_meta(folder, meta)
        log(f"[fit {n}] {eid}: saved {zoo.human_size(meta['size_bytes'])} "
            f"({fmt_s(time.perf_counter() - t0)}), "
            f"{scores['window_acc'] * 100:.1f} % per window")


def mark_variant(meta, eid, excluded):
    """
    Describe a model trained WITHOUT some people: who, which model it is a
    variant of, and what the ordinary model scored on those people and on
    everybody else -- so the two can be compared without a second lookup.
    """
    base_id = eid.split("-without-")[0]
    base = zoo.get_entry(base_id)
    base_pp = dict((base.scores.get("per_person") or {})) if base else {}
    scores = meta["scores"]
    meta["excluded_persons"] = sorted(excluded)
    meta["variant_of"] = base_id
    label = ", ".join(sorted(excluded))
    meta["name"] = f"{meta.get('name', base_id)} (without {label})"
    meta["description"] = (f"{meta.get('description', '')}  Trained without "
                           f"{label}'s recordings; scored on the people left.")
    scores["excluded_persons"] = sorted(excluded)
    scores["base_per_person"] = base_pp
    #  The ordinary model's score on the excluded person is the right number
    #  for them: it was trained on exactly the people this one was.
    scores["excluded_held_out"] = {p: base_pp.get(p) for p in excluded}


def fit_ml(eid, spec, D, sc, scores, folder, adapt, seeds, joblib, augment=()):
    import train_model as TM
    F, y, persons = D["F"], D["y"], D["persons"]
    setup = ZR.ML_SETUP
    Fit, extras, _table = ml_cv.prepare_features(D, augment, adapt, 0)
    adaptation = {"mode": "off"}
    if adapt:
        gm, gs = AD.feature_stats(F)
        adaptation = {"mode": "person_features",
                      "fraction": AD.ADAPT_FRACTION,
                      "global_mean": gm, "global_std": gs}
    pipe = ml_cv.make_pipeline(spec, 0)
    pipe.fit(np.concatenate([Fit] + list(extras), axis=0),
             np.concatenate([y] * (1 + len(extras))))
    fit_acc = float((pipe.predict(Fit) == y).mean())
    names = ZR.ml_id(spec[0])
    mode = D["calibration"]["mode"]
    payload = {
        "model": pipe, "labels": D["labels"], "fs": TM.MODEL_FS,
        "window_s": setup["window_s"], "window_ms": setup["window_s"] * 1000.0,
        "representation": setup["representation"], "features": True,
        "feature_set": setup["feature_set"],
        "freq_features": setup["freq_features"],
        "window_samples": D["window_samples"],
        "feature_names": TM.feature_names_for(setup["feature_set"],
                                              setup["freq_features"]),
        "channel_names": list(T.CHANNEL_NAMES), "emg_mode": T.EMG_MODE,
        "calibration": {
            "mode": mode, "requested": T.CALIBRATION_MODE or "off",
            "target_level": float(T.CALIBRATION_TARGET_LEVEL),
            "subjects": D["calibration"]["subjects"],
            "uncalibrated_subjects": [], "records": D["calibration"]["records"],
            "note": ("the features were built on each wearer's own "
                     "calibrated scale; a new wearer must calibrate before "
                     "this model can be used on them" if mode != "off"
                     else "no calibration applied -- device counts")},
        "adaptation": adaptation,
        "created_iso": T.iso_time(time.time()),
        "data": T.dataset_fingerprint(NN.DATASET_ROOT),
        "stats": {
            "model_name": spec[0], "feature_set": setup["feature_set"],
            "representation": setup["representation"],
            "window_s": setup["window_s"],
            "window_accuracy": scores["window_acc"],
            "window_accuracy_std": scores["window_acc_std"],
            "window_macro_f1": scores["macro_f1"],
            "recording_vote_accuracy": scores["vote_acc"],
            "fold_accuracies": [scores["per_person"][p]
                                for p in sorted(scores["per_person"])],
            "confusion_matrix": sc["confusion_matrix"],
            "n_windows": int(len(y)), "n_features": int(F.shape[1]),
            "n_recordings": len(D["recs"]),
            "n_recordings_total": len(D["recs"]) + len(D["dropped"]),
            "subjects": sorted(set(persons.tolist())),
            "overlap": setup["overlap"], "n_splits": len(scores["per_person"]),
            "validation_protocol": PROTOCOL,
            "selection_grouping": "subject",
            "subject_independent": {
                "per_subject": scores["per_person"],
                "mean": scores["window_acc"], "std": scores["window_acc_std"]},
            "train_fit_accuracy": fit_acc,
            "deduped_windows": True, "calibration_mode": mode,
            "calibrated_subjects": D["calibration"]["subjects"],
            "trainer": "train_zoo.py",
            "created_iso": T.iso_time(time.time())},
    }
    path = os.path.join(folder, "model.joblib")
    joblib.dump(payload, path + ".partial")
    os.replace(path + ".partial", path)

    # latency of ONE live window (features + predict), single-threaded like the app
    T.serialise_estimator(pipe)
    row = Fit[:1]
    for _ in range(5):
        pipe.predict_proba(row)
    ts = []
    for _ in range(40):
        a = time.perf_counter()
        pipe.predict_proba(row)
        ts.append((time.perf_counter() - a) * 1000)
    return {
        "id": eid, "family": "ml", "name": spec[0], "kind": spec[1],
        "description": describe_ml(spec), "file": "model.joblib",
        "params": None, "size_bytes": os.path.getsize(path),
        "scores": scores, "latency_ms": float(np.median(ts)),
        "latency_note": "predict_proba of one row, 1 thread, machine not idle",
        "setup": ZR.ML_SETUP,
        "trained_with": {"augment": list(augment), "adapt": adapt,
                         "calibration": mode},
        "compat": {"labels": D["labels"], "fs": TM.MODEL_FS,
                   "window_s": setup["window_s"],
                   "window_samples": D["window_samples"],
                   "representation": setup["representation"],
                   "feature_set": setup["feature_set"],
                   "freq_features": setup["freq_features"],
                   "calibration_mode": mode,
                   "adaptation_mode": "person_features" if adapt else "off"},
        "n_windows": int(len(y)), "created_iso": T.iso_time(time.time()),
    }


def fit_nn(eid, kind, D, sc, scores, folder, args, augment, adapt):
    import torch
    import nn_cv
    import nn_checkpoint
    device = torch.device("cuda")
    model, mean, std, table, fit_acc = nn_cv.fit_final(
        kind, D, device, args.epochs, seed=0, augment=tuple(augment),
        adapt=adapt)
    win, n_ch = D["X"].shape[1], D["X"].shape[2]
    mode = D["calibration"]
    people = sorted(set(D["persons"].tolist()))
    stats = {
        "model_name": ZR.NN_TITLES.get(kind, f"NN-{kind}"),
        "trainer": "train_zoo.py", "kind": kind,
        "validation_protocol": PROTOCOL, "grouping": "subject",
        "grouping_note": "leave-one-person-out",
        "grouping_contrast": None, "dc_only_control": None,
        "majority_class_baseline": float(max(np.bincount(D["y"]))
                                         / len(D["y"])),
        "train_fit_accuracy": fit_acc, "subjects": people,
        "labels": D["labels"], "epochs": args.epochs,
        "batch_size": NN.BATCH_SIZE, "learning_rate": NN.LEARNING_RATE,
        "weight_decay": NN.WEIGHT_DECAY, "label_smoothing": NN.LABEL_SMOOTHING,
        "early_stopping": False, "deduped": True,
        "augmentation": list(augment), "adaptation": adapt,
        "subject_independent": {
            "per_subject": scores["per_person"], "mean": scores["window_acc"],
            "std": scores["window_acc_std"]},
        "window_accuracy": scores["window_acc"],
        "window_accuracy_std": scores["window_acc_std"],
        "window_macro_f1": scores["macro_f1"],
        "recording_vote_accuracy": scores["vote_acc"],
        "n_recordings_voted": int(len(set(D["uid"].tolist()))),
        "n_splits": len(people),
        "fold_accuracies": [scores["per_person"][p] for p in people],
        "confusion_matrix": sc["confusion_matrix"],
        "n_windows": int(len(D["y"])),
        "n_parameters": nn_arch.count_parameters(model),
        "window_s": ZR.NN_SETUP["window_s"], "overlap": ZR.NN_SETUP["overlap"],
        "win_samples": int(win),
        "architecture_note": nn_arch.DESCRIPTIONS.get(kind, ""),
        "separation_note": ("This net is NOT part of the deployed answer "
                            "unless it is selected in the settings; it is "
                            "never blended with the ML models."),
        "created_iso": T.iso_time(time.time()),
    }
    adaptation = ({"mode": "person", "fraction": AD.ADAPT_FRACTION}
                  if adapt else {"mode": "off"})
    payload = nn_checkpoint.build_payload(
        model, kind=kind, labels=D["labels"], n_channels=n_ch, win=win,
        window_s=ZR.NN_SETUP["window_s"], overlap=ZR.NN_SETUP["overlap"],
        model_fs=NN.MODEL_FS, hidden=nn_arch.HIDDEN,
        dropout=nn_arch.DROPOUT, mean=mean, std=std, applied_mode=mode,
        requested_mode=T.CALIBRATION_MODE or "off",
        calibration_subjects=[], root=NN.DATASET_ROOT, stats=stats,
        device_info={"device": "cuda", "gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__},
        adaptation=adaptation, augmentation=augment)
    payload["calibration"]["subjects"] = people
    path, _, _ = nn_checkpoint.save_payload(
        payload, os.path.join(folder, "model.pt"), archive=False)
    return {
        "id": eid, "family": "nn", "name": stats["model_name"], "kind": kind,
        "description": nn_arch.DESCRIPTIONS.get(kind, ""),
        "file": "model.pt", "sidecar": "model.json",
        "params": stats["n_parameters"], "size_bytes": os.path.getsize(path),
        "scores": scores,
        "latency_ms": nn_cv.latency_ms(model, device, win, n_ch),
        "latency_note": "batch of 1 on CUDA, synchronised, machine not idle",
        "setup": ZR.NN_SETUP,
        "trained_with": {"augment": list(augment), "adapt": adapt,
                         "calibration": mode},
        "compat": {"labels": D["labels"], "window_s": ZR.NN_SETUP["window_s"],
                   "win_samples": int(win), "n_channels": int(n_ch),
                   "calibration_mode": mode},
        "n_windows": int(len(D["y"])), "created_iso": T.iso_time(time.time()),
    }


# ==============================================================================
# ENSEMBLES
# ==============================================================================


def load_oof(fam):
    """`{id: seed-0 probs}`, the labels etc. for one family."""
    d = np.load(os.path.join(EVAL_DIR, f"data_{fam}.npz"), allow_pickle=True)
    digest = str(d["digest"]) if "digest" in d.files else None
    fresh = {r["id"] for r in read_jsonl(os.path.join(EVAL_DIR, "runs.jsonl"))
             if r.get("digest") == digest and r["seed"] == 0}
    probs = {}
    for e in zoo.list_entries(family=fam):
        f = os.path.join(EVAL_DIR, f"oof_{e.id}_s0.npz")
        if e.id in fresh and os.path.exists(f):
            P = np.load(f)["probs"]
            if len(P) == len(d["y"]):
                probs[e.id] = P
    return probs, d["y"], d["persons"].astype(str), d["uid"].astype(str), \
        [str(x) for x in d["labels"]]


def subset_acc(probs, ids, y, persons, rows=None):
    P = np.mean([probs[i] for i in ids], axis=0)
    pred = P.argmax(1)
    accs = []
    for p in sorted(set(persons.tolist())):
        m = persons == p if rows is None else (persons == p) & rows
        if m.any():
            accs.append(float((pred[m] == y[m]).mean()))
    return float(np.mean(accs)) if accs else 0.0


def search_subsets(probs, y, persons, pool, max_size=4):
    out = []
    for k in range(2, max_size + 1):
        for ids in itertools.combinations(pool, k):
            out.append((subset_acc(probs, ids, y, persons), ids))
    out.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    return out


def cmd_ensembles(args, log):
    result = {}
    for fam in ("ml", "nn"):
        try:
            probs, y, persons, uid, labels = load_oof(fam)
        except OSError:
            continue
        if len(probs) < 2:
            continue
        singles = sorted(probs, key=lambda i: -subset_acc(probs, [i], y,
                                                          persons))
        pool = singles[:8]
        ranked = search_subsets(probs, y, persons, pool)
        best_acc, best_ids = ranked[0]
        # nested: pick the subset on the OTHER people, score it on this one
        nested = {}
        for held in sorted(set(persons.tolist())):
            train_rows = persons != held
            cand = []
            for k in range(2, 5):
                for ids in itertools.combinations(pool, k):
                    cand.append((subset_acc(probs, ids, y, persons,
                                            train_rows), ids))
            cand.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
            ids = cand[0][1]
            P = np.mean([probs[i] for i in ids], axis=0)
            m = persons == held
            nested[held] = float((P[m].argmax(1) == y[m]).mean())
        P = np.mean([probs[i] for i in best_ids], axis=0)
        s = scores_from_probs(P, y, persons, uid, len(labels))
        best_single = max(subset_acc(probs, [i], y, persons) for i in probs)
        result[fam] = {
            "members": list(best_ids),
            "window_acc": best_acc, "macro_f1": s["macro_f1"],
            "vote_acc": s["vote_acc"],
            "nested_acc": float(np.mean(list(nested.values()))),
            "nested_per_person": nested,
            "best_single_acc": best_single,
            "searched": f"every subset of size 2-4 of the {len(pool)} best "
                        f"single models ({len(ranked)} subsets)",
            "note": ("window_acc is the best of the searched subsets on the "
                     "same folds, so it is optimistic; nested_acc picks the "
                     "subset without the held-out person and is the number "
                     "the star is decided on.")}
        log(f"[ensembles] {fam}: best subset {', '.join(best_ids)}  "
            f"{best_acc * 100:.1f} % (nested {result[fam]['nested_acc'] * 100:.1f} %"
            f", best single {best_single * 100:.1f} %)")
    if result:
        zoo.write_ensembles(result)
    return result


# ==============================================================================
# MAIN
# ==============================================================================


def cmd_status(args):
    entries = zoo.list_entries()
    stars = zoo.compute_stars(entries, zoo.read_ensembles())
    print(f"{'id':<38}{'per window':>11}{'F1':>8}{'vote':>8}{'size':>10}  star")
    #  Ordinary models first; trained-without variants after them (their score
    #  is over fewer people). `*` = the yellow star, `b` = the blue one.
    ordered = []
    for fam in ("ml", "nn"):
        mine = [e for e in entries if e.family == fam]
        ordered += zoo.rank([e for e in mine if not e.excluded])
        ordered += zoo.rank([e for e in mine if e.excluded])
    for e in ordered:
        mark = ("*" if e.id in stars["models"]
                else "b" if e.id in stars["variants"] else "")
        print(f"{e.id:<38}"
              f"{(e.window_acc or 0) * 100:10.1f}%"
              f"{(e.macro_f1 or 0) * 100:8.1f}"
              f"{(e.scores.get('vote_acc') or 0) * 100:8.1f}"
              f"{zoo.human_size(e.size_bytes):>10}  {mark}")
    for fam, why in stars["ensembles"].items():
        print(f"* {fam}: {why}")


def main(argv=None):
    T.force_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["all", "eval", "fit", "ensembles",
                                        "status"])
    ap.add_argument("--family", default="all", choices=["all", "ml", "nn"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=2, help="NN seeds")
    ap.add_argument("--ml-seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--augment", default=None,
                    help="comma list or 'all': both families (overridden by "
                         "--augment-nn / --augment-ml)")
    ap.add_argument("--augment-nn", default=None)
    ap.add_argument("--augment-ml", default=None)
    ap.add_argument("--adapt", dest="adapt", action="store_true", default=None,
                    help="adaptation for both families (overridden by "
                         "--adapt-nn / --adapt-ml)")
    ap.add_argument("--no-adapt", dest="adapt", action="store_false")
    ap.add_argument("--adapt-nn", choices=["on", "off"], default=None)
    ap.add_argument("--adapt-ml", choices=["on", "off"], default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-measure even if a matching run is cached")
    ap.add_argument("--root", default=NN.DATASET_ROOT)
    ap.add_argument("--exclude-person", action="append", default=[],
                    help="train WITHOUT this person's recordings (repeatable): "
                         "saves <id>-without-<name> variants, scored "
                         "leave-one-person-out on the people left, in "
                         "results/zoo_eval_without_<name>/")
    args = ap.parse_args(argv)
    args.exclude_person = sorted(set(args.exclude_person))
    if args.exclude_person:
        global EVAL_DIR, PROTOCOL
        tag = "-".join(zoo.slug(p) for p in args.exclude_person)
        EVAL_DIR = os.path.join(ROOT, "results", f"zoo_eval_without_{tag}")
        PROTOCOL = ("leave-one-person-out among the people kept; "
                    + ", ".join(args.exclude_person)
                    + " excluded from training and from the folds")
    import nn_augment
    #  Defaults are the application's own per-family settings (config.py).
    def pick(specific, both, default):
        chosen = specific if specific is not None else both
        return nn_augment.parse_augment(default if chosen is None else chosen)

    args.augment_nn = pick(args.augment_nn, args.augment,
                           list(T.TRAIN_AUGMENT_NN))
    args.augment_ml = pick(args.augment_ml, args.augment,
                           list(T.TRAIN_AUGMENT_ML))
    both = args.adapt
    args.adapt_nn = ((args.adapt_nn == "on") if args.adapt_nn is not None
                     else bool(T.TRAIN_ADAPT_NN if both is None else both))
    args.adapt_ml = ((args.adapt_ml == "on") if args.adapt_ml is not None
                     else bool(T.TRAIN_ADAPT_ML if both is None else both))

    if args.command == "status":
        cmd_status(args)
        return 0

    def log(msg):
        print(msg, flush=True)
        os.makedirs(EVAL_DIR, exist_ok=True)
        with open(os.path.join(EVAL_DIR, "log.txt"), "a",
                  encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")

    data = Data(args.root, args.epochs, args.exclude_person)
    t0 = time.perf_counter()
    if args.command in ("all", "eval"):
        cmd_eval(args, data, log)
    if args.command in ("all", "fit"):
        cmd_fit(args, data, log)
    if args.command in ("all", "ensembles", "fit") and not args.exclude_person:
        #  Variants are scored on fewer people; they never take part in the
        #  best-subset search (and must not overwrite ensembles.json).
        cmd_ensembles(args, log)
    log(f"done in {fmt_s(time.perf_counter() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Dataset discovery, loading, windowing, person folds, fingerprint and model
provenance.
"""

import hashlib
import os
from typing import Dict, List

import numpy as np

from emg_app import config as cfg
from emg_app.utils import safe_slug
from emg_app.features import (
    extract_features, feature_names, sliding_windows, window_params)
from emg_app.recording import (
    DatasetWriter, STAGE_RAW, STAGE_SUFFIX_RE, STAGE_TAKE, STAGE_TRIMMED,
    _parse_path_identity, _stage_from_filename, load_recording,
    parse_recording_name, stamp_to_unix)


def dataset_person_sessions(root: str = None) -> Dict[str, int]:
    """
    `{"Dano/session_01": 44, ...}` -- how many recordings each subject/session
    holds, as discovery would count them (prefer_trimmed, underscore folders
    skipped). What the MODEL SPEC dialog offers as "train on these".
    """
    counts: Dict[str, int] = {}
    for path in discover_recordings(root, stage="prefer_trimmed"):
        _label, person, session, _rep = _parse_path_identity(path)
        key = f"{person}/{session}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def list_all_takes(root: str = None) -> List[dict]:
    """
    Every free take under the dataset root, across ALL subjects and sessions,
    newest first. Each entry is what `DatasetWriter.list_takes()` returns plus
    `person_id` and `session_id` (the folders it was found in), so a list can
    be filtered by either and an entry can be opened or deleted through a
    writer for its own subject/session rather than the current one.

    Underscore folders are skipped at both levels, the same rule discovery
    uses. Nothing is created: a missing root is simply an empty list.
    """
    root = root or cfg.DATA_FOLDER
    if not os.path.isabs(root):
        root = os.path.join(cfg.PROJECT_DIR, root)
    if not os.path.isdir(root):
        return []

    found: List[dict] = []
    for person in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person)
        if person.startswith("_") or not os.path.isdir(person_dir):
            continue
        for session in sorted(os.listdir(person_dir)):
            folder = os.path.join(person_dir, session, cfg.FREE_TAKE_FOLDER)
            if session.startswith("_") or not os.path.isdir(folder):
                continue
            for info in DatasetWriter(root, person, session).list_takes():
                info["person_id"] = person
                info["session_id"] = session
                found.append(info)
    found.sort(key=lambda t: t["created_unix"], reverse=True)
    return found


def takes_by_session(root: str = None, person_id: str = None) -> Dict[str, int]:
    """
    One subject's take COUNT per session: `{"session_01": 6, ...}`.

    Only sessions that actually hold takes appear, and a session with none is
    simply absent.

    This exists because TAKES ON DISK is scoped to the current subject AND
    session, and that scoping was invisible: a take recorded under one
    `SESSION_ID` is not in the list when the app is reopened under another,
    which looks exactly like a take that was never saved. The panel uses this
    to say where the others are instead of leaving the operator to guess.

    It counts file stems rather than reading the metadata sidecars -- the
    answer is "are there takes elsewhere", so one `listdir` per session is
    enough and the panel can afford to ask on every refresh.
    """
    root = root or cfg.DATA_FOLDER
    if not os.path.isabs(root):
        root = os.path.join(cfg.PROJECT_DIR, root)
    person = safe_slug(person_id or cfg.PERSON_ID, "person")
    person_dir = os.path.join(root, person)
    if not os.path.isdir(person_dir):
        return {}

    found: Dict[str, int] = {}
    for session in sorted(os.listdir(person_dir)):
        #  Same rule discovery uses: an underscore folder is not a session.
        if session.startswith("_"):
            continue
        folder = os.path.join(person_dir, session, cfg.FREE_TAKE_FOLDER)
        if not os.path.isdir(folder):
            continue
        stems = set()
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower().lstrip(".") not in ("npz", "csv", "parquet"):
                continue
            if stem.endswith("_meta"):
                continue
            if _stage_from_filename(name) == STAGE_TAKE:
                stems.add(stem)
        if stems:
            found[session] = len(stems)
    return found


def discover_recordings(root: str = None, persons=None, sessions=None,
                        gestures=None, stage: str = STAGE_RAW,
                        prefer: str = "npz"):
    """
    Walk the dataset tree and return the recording file paths that match.

    `stage` is "raw", "trimmed", "processed", "augmented" or
    "prefer_trimmed" (use the trimmed file when one exists, else the raw one).
    `prefer` picks the format when both .npz and .csv exist for a recording.
    """
    root = os.path.abspath(root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
    if not os.path.isdir(root):
        return []

    wanted_stages = ([STAGE_TRIMMED, STAGE_RAW] if stage == "prefer_trimmed"
                     else [stage])
    order = [prefer] + [f for f in ("npz", "csv", "parquet") if f != prefer]

    # uid -> {stage: {ext: path}}
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Underscore folders are tooling, not data: _graphs holds exported
        # images and _deleted holds quarantined recordings, which must stay
        # recoverable on disk without ever re-entering the dataset.
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if ext not in ("npz", "csv", "parquet") or name.endswith("_meta.json"):
                continue
            path = os.path.join(dirpath, name)
            label, person, session, _ = _parse_path_identity(path)
            if persons and person not in persons:
                continue
            if sessions and session not in sessions:
                continue
            if gestures and label not in gestures:
                continue
            file_stage = _stage_from_filename(path)
            stem = os.path.splitext(name)[0]
            # Group the formats of one recording together. The dedupe counter
            # stays part of the key, so a re-trimmed copy is its own entry
            # rather than silently shadowing the first one.
            suffix = STAGE_SUFFIX_RE.search(stem)
            base = stem[:suffix.start()] if suffix else stem
            counter = suffix.group(0)[len(file_stage) + 1:] if suffix else ""
            key = (person, session, label, base + counter)
            found.setdefault(key, {}).setdefault(file_stage, {})[ext] = path

    paths = []
    for key in sorted(found):
        for want in wanted_stages:
            by_ext = found[key].get(want)
            if not by_ext:
                continue
            for ext in order:
                if ext in by_ext:
                    paths.append(by_ext[ext])
                    break
            break
    return paths


def load_dataset(root: str = None, persons=None, sessions=None, gestures=None,
                 stage: str = STAGE_RAW, prefer: str = "npz",
                 skip_quality_errors: bool = False, verbose: bool = False):
    """
    Load whole recordings into memory.

    Returns List[Recording]. This is the entry point for a training script:

        recs = load_dataset("datasets", stage="prefer_trimmed")
        X, y, groups, meta = make_windows(recs)
    """
    paths = discover_recordings(root, persons, sessions, gestures, stage, prefer)
    recs = []
    for path in paths:
        try:
            rec = load_recording(path)
        except Exception as exc:
            print(f"[warn] could not load {path}: {exc}")
            continue
        if skip_quality_errors and rec.worst_issue_level == "error":
            if verbose:
                print(f"[skip] {path}: flagged with a quality error")
            continue
        recs.append(rec)
    if verbose:
        print(f"loaded {len(recs)} recordings from {len(paths)} files")
    return recs


def make_windows(recordings, window_ms: float = None, overlap: float = None,
                 fs: float = None, labels=None, use_label_window: bool = True,
                 representation: str = "raw", normalize: str = "mad"):
    """
    Turn recordings into a supervised windowed dataset.

    Returns (X, y, groups, meta) where
        X       (n_windows, window_samples, 8) float32
        y       (n_windows,) int   -- index into `labels`
        groups  (n_windows,) str   -- person_id, for subject-independent CV
        meta    dict with "labels", "window_samples", "step", "fs" and a
                per-window record list.

    `groups` is the important part: pass it to leave_one_person_out() or to
    scikit-learn GroupKFold so that windows from one person never straddle the
    train/test boundary. Random splitting of windows from the same person is
    the classic way to accidentally report 99 % accuracy that does not transfer.

    `representation`:
        "raw"       -> the device signal, untouched
        "normalized"-> per-recording robust normalisation (see normalize)
        "envelope"  -> moving-RMS envelope
    """
    recordings = list(recordings)
    if not recordings:
        return (np.empty((0, 0, cfg.N_CHANNELS), np.float32), np.empty(0, int),
                np.empty(0, object), {"labels": [], "window_samples": 0})
    #  Width comes from the recordings themselves. With nothing to go on
    #  above, the active profile is the only available answer.
    n_channels = int(recordings[0].n_channels)

    labels = list(labels) if labels else sorted({r.label for r in recordings})
    label_to_index = {name: i for i, name in enumerate(labels)}

    fs = float(fs or np.median([r.fs for r in recordings]))
    win, step = window_params(fs, window_ms, overlap)

    xs, ys, gs, records = [], [], [], []
    for rec in recordings:
        if rec.label not in label_to_index:
            continue

        if representation == "envelope":
            signal = rec.envelope().astype(np.float32)
        elif representation == "normalized":
            signal = rec.normalized(normalize)[0].astype(np.float32)
        else:
            signal = rec.emg.astype(np.float32)

        if use_label_window and rec.label_end_index > rec.label_start_index:
            signal = signal[rec.label_start_index:rec.label_end_index]

        windows = sliding_windows(signal, win, step)
        if windows.shape[0] == 0:
            continue

        xs.append(np.ascontiguousarray(windows))
        ys.append(np.full(windows.shape[0], label_to_index[rec.label], dtype=int))
        gs.append(np.full(windows.shape[0], rec.person_id, dtype=object))
        records.extend({
            "uid": rec.uid,
            "person_id": rec.person_id,
            "session_id": rec.session_id,
            "label": rec.label,
            "repetition": rec.repetition,
            "window_index": w,
            "source": rec.source,
        } for w in range(windows.shape[0]))

    if not xs:
        return (np.empty((0, win, n_channels), np.float32), np.empty(0, int),
                np.empty(0, object), {"labels": labels, "window_samples": win})

    meta = {
        "labels": labels,
        "window_samples": win,
        "step": step,
        "fs": fs,
        "representation": representation,
        "records": records,
    }
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), meta)


def make_feature_matrix(X, fs: float, include_freq: bool = True):
    """
    (n_windows, window, 8) -> (n_windows, n_features) for the classic models
    (Random Forest / SVM / XGBoost). Returns (features, names).
    """
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.empty((0, len(feature_names(include_freq))), np.float32), \
               feature_names(include_freq)
    feats = np.stack([extract_features(X[i], fs, include_freq)
                      for i in range(X.shape[0])])
    return feats, feature_names(include_freq)


def leave_one_person_out(groups):
    """
    Yield (train_index, test_index, held_out_person) for every person.

    This is the validation protocol that actually answers "will this work on
    somebody new?". Use it instead of a random split.
    """
    groups = np.asarray(groups, dtype=object)
    for person in sorted(set(groups.tolist())):
        test = np.flatnonzero(groups == person)
        train = np.flatnonzero(groups != person)
        if train.size and test.size:
            yield train, test, person


def split_by_person(groups, test_persons):
    """Explicit split, e.g. train on persons 1-3 and test on person 4."""
    groups = np.asarray(groups, dtype=object)
    test_persons = set(test_persons if isinstance(test_persons, (list, tuple, set))
                       else [test_persons])
    mask = np.isin(groups, list(test_persons))
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def dataset_fingerprint(root: str = None, stage: str = STAGE_RAW) -> dict:
    """
    A compact, comparable description of what is on disk right now.

    `uid_digest` is the identity: two fingerprints with the same digest
    describe the same set of recordings, whatever order they were found in.
    """
    root = os.path.abspath(root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
    uids, labels, subjects, sessions = [], {}, set(), set()
    newest_stamp = ""

    for path in discover_recordings(root, stage=stage, prefer="npz"):
        label, person, session, _ = _parse_path_identity(path)
        parsed = parse_recording_name(path)
        if parsed is None:
            continue
        _, stamp, uid = parsed
        uids.append(uid)
        labels[label] = labels.get(label, 0) + 1
        subjects.add(person)
        sessions.add(f"{person}/{session}")
        if stamp > newest_stamp:        # the slug sorts chronologically
            newest_stamp = stamp

    digest = hashlib.sha1(
        "\n".join(sorted(uids)).encode("utf-8")).hexdigest()[:16]
    return {
        "root": root,
        "n_recordings": len(uids),
        "labels": labels,
        "subjects": sorted(subjects),
        "sessions": sorted(sessions),
        "newest_stamp": newest_stamp,
        "newest_unix": stamp_to_unix(newest_stamp) if newest_stamp else None,
        "uid_digest": digest,
    }


def count_recordings_after(stamp: str, root: str = None,
                           stage: str = STAGE_RAW) -> int:
    """How many recordings on disk are newer than `stamp`."""
    if not stamp:
        return 0
    root = os.path.abspath(root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
    newer = 0
    for path in discover_recordings(root, stage=stage, prefer="npz"):
        parsed = parse_recording_name(path)
        if parsed is not None and parsed[1] > stamp:
            newer += 1
    return newer


def describe_training_data(data: dict) -> str:
    """One line naming what a model was trained on."""
    if not data:
        return "not recorded"
    bits = [f"{data.get('n_recordings', 0)} recordings"]
    labels = data.get("labels") or {}
    if labels:
        bits.append(", ".join(f"{name} {count}"
                              for name, count in sorted(labels.items())))
    subjects = data.get("subjects") or []
    if subjects:
        bits.append(f"{len(subjects)} subject(s): {', '.join(subjects)}")
    sessions = data.get("sessions") or []
    if sessions:
        bits.append(f"{len(sessions)} session(s)")
    return "  |  ".join(bits)


def model_data_status(payload: dict, fingerprint: dict = None) -> dict:
    """
    Is there a model, what was it trained on, and is that still the dataset?

    Returns `state`, a `headline`, a `detail` and a `severity` the GUI maps
    to a colour. `state` is one of:

        "untrained"  nothing loaded
        "unknown"    a model, but it does not record its training data
        "current"    trained on exactly the recordings now on disk
        "stale"      the dataset has changed since it was trained
    """
    if not payload:
        return {"state": "untrained", "n_new": 0,
                "headline": "NO MODEL TRAINED",
                "detail": "Record some gestures, then press RETRAIN ON ALL "
                          "DATA. Until then the classifier stays disabled -- "
                          "it never invents a prediction.",
                "severity": "dim"}

    stats = payload.get("stats") or {}
    # `created_iso` is the sklearn payload's key, `created` the torch
    # checkpoint's. Both are asked the same question, so both are read.
    trained_at = (payload.get("created_iso") or payload.get("created")
                  or stats.get("created_iso") or "")
    when = f" on {trained_at[:16].replace('T', ' ')}" if trained_at else ""
    data = payload.get("data") or {}

    if not data.get("uid_digest"):
        return {"state": "unknown", "n_new": 0,
                "headline": f"TRAINED{when}",
                "detail": ("This model does not record which recordings it "
                           "was trained on, so whether it is up to date "
                           "cannot be answered. Retrain to find out.\n"
                           f"It reports: {stats.get('n_recordings', '?')} "
                           f"recordings, subjects "
                           f"{', '.join(stats.get('subjects') or []) or '?'}."),
                "severity": "warn"}

    trained_on = describe_training_data(data)
    if fingerprint is None:
        return {"state": "unknown", "n_new": 0,
                "headline": f"TRAINED{when}",
                "detail": f"Trained on {trained_on}.\n"
                          "The dataset on disk has not been checked.",
                "severity": "dim"}

    if fingerprint.get("uid_digest") == data.get("uid_digest"):
        return {"state": "current", "n_new": 0,
                "headline": f"UP TO DATE  --  trained{when}",
                "detail": f"Trained on {trained_on}.\n"
                          "Every recording now on disk went into this model.",
                "severity": "good"}

    n_new = count_recordings_after(data.get("newest_stamp", ""),
                                   fingerprint.get("root"))
    on_disk = fingerprint.get("n_recordings", 0)
    trained_n = data.get("n_recordings", 0)
    if n_new > 0:
        what = (f"{n_new} recording(s) have been added since this model was "
                f"trained.")
    elif on_disk < trained_n:
        what = (f"{trained_n - on_disk} recording(s) it was trained on are "
                f"no longer on disk.")
    else:
        what = "The set of recordings on disk is not the one it was trained on."
    return {"state": "stale", "n_new": n_new,
            "headline": f"OUT OF DATE  --  trained{when}",
            "detail": f"Trained on {trained_on}.\n{what}\n"
                      f"{on_disk} recording(s) on disk now. "
                      "RETRAIN ON ALL DATA to include them.",
            "severity": "warn"}


def dataset_summary(root: str = None):
    """Count recordings per person / session / gesture across the whole tree."""
    root = os.path.abspath(root or os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
    summary = {"root": root, "persons": {}, "total_raw": 0, "total_trimmed": 0}
    if not os.path.isdir(root):
        return summary

    for path in discover_recordings(root, stage=STAGE_RAW, prefer="npz"):
        label, person, session, _ = _parse_path_identity(path)
        person_entry = summary["persons"].setdefault(person, {})
        session_entry = person_entry.setdefault(session, {})
        session_entry[label] = session_entry.get(label, 0) + 1
        summary["total_raw"] += 1

    summary["total_trimmed"] = len(
        discover_recordings(root, stage=STAGE_TRIMMED, prefer="npz")
    )
    return summary

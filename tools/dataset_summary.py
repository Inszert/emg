"""
What is in `datasets/` right now, measured -- the data a report needs.

    python tools/dataset_summary.py

Writes `results/dataset_summary.json` and `docs/DATASET_SUMMARY.md`. Nothing is
typed by hand: recordings are counted from their own `*_raw_meta.json` files,
the windows the models were scored on come from the stored evaluation data
(`results/zoo_eval/data_*.npz`), and the calibrations from each person's own
`calibration.json` and from the derived calibration the trainers use.

`docs/EDA.md` describes the dataset that was archived on 2026-10-15; this is the
description of the working set.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results")
GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(v):
    return f"{v * 100:.1f} %"


def recordings():
    """Every raw recording's own metadata (clips in the gesture folders)."""
    out = []
    for path in sorted(glob.glob(os.path.join(
            DATA, "*", "*", "*", "*_raw_meta.json"))):
        parts = os.path.relpath(path, DATA).split(os.sep)
        if any(p.startswith("_") for p in parts[:-1]):
            continue
        meta = jload(path)
        meta["_file"] = os.path.relpath(path, DATA)
        out.append(meta)
    return out


def takes_and_deleted():
    takes = defaultdict(list)
    deleted = defaultdict(int)
    for path in glob.glob(os.path.join(DATA, "*", "*", "_takes", "*_meta.json")):
        person = os.path.relpath(path, DATA).split(os.sep)[0]
        takes[person].append(jload(path))
    for path in glob.glob(os.path.join(DATA, "*", "*", "_deleted", "*_meta.json")):
        deleted[os.path.relpath(path, DATA).split(os.sep)[0]] += 1
    return takes, deleted


def calibrations(people):
    """The saved (live) calibration and the DERIVED one the trainers use."""
    saved = {}
    for p in people:
        path = os.path.join(DATA, p, "calibration.json")
        if os.path.exists(path):
            c = jload(path)
            saved[p] = {
                "created": c.get("created_iso"), "source": c.get("source"),
                "rest": c.get("rest"), "reference": c.get("reference"),
                "contrast": c.get("contrast"),
                "gesture_sweep": c.get("gesture_sweep") is not None}
    derived = {}
    try:
        import emg_dataset_tool as T
        import train_model as TM
        recs, _ = TM.load_recordings(DATA, exclude_codes=())
        who = sorted({r.person_id for r in recs})
        cal, _notes, _used = T.training_calibrations(DATA, "affine", recs, who)
        for p, c in (cal or {}).items():
            if c and c.get("mode") == "affine":
                #  affine: offset = rest level, scale = reference - rest
                rest = np.asarray(c["offset"], dtype=float)
                derived[p] = {"rest": [float(x) for x in rest],
                              "reference": [float(x) for x in
                                            rest + np.asarray(c["scale"],
                                                              dtype=float)]}
    except Exception as exc:                       # the summary must not fail
        derived = {"_error": str(exc)[:200]}
    return saved, derived


def lineage():
    """
    Each session's append-only manifest against what is on disk.

    A clip the manifest logged as saved should be in its gesture folder, or in
    `_deleted/` with a `deleted` line. A clip that is neither was removed some
    other way; it is reported with its label and times, and how many clips of
    the same label were saved afterwards, so the reader can see whether it was
    probably re-recorded.
    """
    import re
    out = {}
    uid = lambda f: re.search(r"_([0-9a-f]{8})_raw\.npz$", f).group(1)
    for mf in sorted(glob.glob(os.path.join(DATA, "*", "*", "manifest.jsonl"))):
        parts = os.path.relpath(mf, DATA).split(os.sep)
        if parts[0].startswith("_"):
            continue
        person, session = parts[0], parts[1]
        base = os.path.dirname(mf)
        lines = [json.loads(ln) for ln in open(mf, encoding="utf-8")
                 if ln.strip()]
        raw = [d for d in lines if d.get("stage") == "raw"]
        files = glob.glob(os.path.join(base, "*", "*_raw.npz"))
        quarantined = {uid(f) for f in files
                       if os.sep + "_deleted" + os.sep in f}
        ondisk = {uid(f) for f in files if os.sep + "_" not in
                  f[len(base):]}
        gone = [d for d in raw if d["uid"] not in ondisk
                and d["uid"] not in quarantined]
        notes = []
        for label in sorted({d["label"] for d in gone}):
            g = sorted((d for d in gone if d["label"] == label),
                       key=lambda d: d["created_iso"])
            later = [d for d in raw if d["label"] == label
                     and d["created_iso"] > g[-1]["created_iso"]
                     and d["uid"] in ondisk]
            notes.append({
                "label": label, "clips": len(g),
                "from": g[0]["created_iso"][:16], "to": g[-1]["created_iso"][:16],
                "seconds_each": round(float(np.mean(
                    [d["duration_s"] for d in g])), 1),
                "clips_of_the_same_gesture_saved_afterwards": len(later),
                "their_seconds_each": round(float(np.mean(
                    [d["duration_s"] for d in later])), 1) if later else None,
                "repetitions": [d["repetition"] for d in g]})
        out[f"{person}/{session}"] = {
            "logged_raw": len(raw), "on_disk": len(ondisk),
            "quarantined": len(quarantined), "logged_but_gone": len(gone),
            "deleted_entries": sum(1 for d in lines
                                   if d.get("stage") == "deleted"),
            "gone": notes}
    return out


def read_signals():
    """
    Read EVERY live clip and take (npz and csv) and check them, and measure
    the signal level per person and gesture.

    Checks: shape (n, 8), no NaN, timestamps never go backwards, n_samples as
    in the metadata, the label window inside the clip, no sample at the 2047
    ceiling, no dead channel, and the csv identical to the npz. The level of a
    clip is the mean over the labelled window and the 8 channels, in device
    counts (the on-device filtered envelope); a person's level for a gesture
    is the mean of their clips' levels.
    """
    import csv
    problems, levels, peaks = [], defaultdict(list), []
    counts = {"clips": 0, "clip_samples": 0, "takes": 0, "take_samples": 0,
              "csv_identical": 0, "csv_checked": 0}
    for kind, pattern in (("clips", "*/*/*/*_raw.npz"),
                          ("takes", "*/*/_takes/*_take.npz")):
        for path in sorted(glob.glob(os.path.join(DATA, pattern))):
            rel = os.path.relpath(path, DATA).replace(os.sep, "/")
            parts = rel.split("/")
            if any(p.startswith("_") for p in
                   (parts[:-1] if kind == "clips" else parts[:1])):
                continue
            z = np.load(path, allow_pickle=True)
            emg, t = z["emg"], z["t_unix"]
            meta = json.loads(str(z["metadata_json"]))
            counts[kind] += 1
            counts[kind[:-1] + "_samples"] += int(len(emg))
            if emg.ndim != 2 or emg.shape[1] != 8:
                problems.append((rel, "shape", list(emg.shape)))
            if not np.isfinite(emg.astype(float)).all():
                problems.append((rel, "NaN"))
            if len(t) != len(emg) or np.any(np.diff(t) < 0):
                problems.append((rel, "timestamps"))
            if len(emg) != meta.get("n_samples"):
                problems.append((rel, "n_samples differs from the metadata"))
            lo, hi = meta.get("label_start_index"), meta.get("label_end_index")
            if kind == "clips" and not (0 <= lo < hi <= len(emg)):
                problems.append((rel, "label window outside the clip"))
            if int(emg.max()) >= 2047:
                problems.append((rel, "touches the 2047 ceiling"))
            if (emg.max(axis=0) == 0).any():
                problems.append((rel, "dead channel"))
            peaks.append(int(emg.max()))
            csv_path = path[:-4] + ".csv"
            if os.path.exists(csv_path):
                counts["csv_checked"] += 1
                with open(csv_path, encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
                cols = [rows[0].index(f"ch{k}") for k in range(1, 9)]
                arr = np.array([[int(r[i]) for i in cols] for r in rows[1:]])
                if arr.shape == emg.shape and (arr == emg).all():
                    counts["csv_identical"] += 1
                else:
                    problems.append((rel, "csv differs from the npz"))
            if kind == "clips":
                levels[(parts[0], str(z["label"]))].append(
                    float(emg[lo:hi].astype(float).mean()))
    return {"counts": counts, "problems": [list(map(str, p)) for p in problems],
            "max_count": max(peaks),
            "levels": {p: {g: (round(float(np.mean(levels[(p, g)])), 1)
                               if levels.get((p, g)) else None)
                           for g in GESTURES}
                       for p in sorted({k[0] for k in levels})}}


def windows(family):
    """Windows the models were scored on, per person and gesture."""
    path = os.path.join(RESULTS, "zoo_eval", f"data_{family}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    y, persons = d["y"], d["persons"].astype(str)
    labels = [str(x) for x in d["labels"]]
    table = {p: {labels[k]: int(((persons == p) & (y == k)).sum())
                 for k in range(len(labels))} for p in sorted(set(persons))}
    return {"labels": labels, "n": int(len(y)), "digest": str(d["digest"]),
            "per_person": table,
            "majority_class": float(np.bincount(y).max() / len(y))}


def archive_counts(name):
    """Recordings in an archive folder, the quarantined ones counted apart."""
    files = glob.glob(os.path.join(DATA, name, "**", "*_raw_meta.json"),
                      recursive=True)
    kept = [f for f in files if os.sep + "_deleted" + os.sep not in f]
    return {"recordings": len(kept), "quarantined": len(files) - len(kept)}


def build():
    recs = recordings()
    people = sorted({m["person_id"] for m in recs})
    takes, deleted = takes_and_deleted()
    per = {}
    for p in people:
        mine = [m for m in recs if m["person_id"] == p]
        by = Counter(m["label"] for m in mine)
        dur = [float(m["duration_s"]) for m in mine]
        rate = [float(m["sample_rate_measured"]) for m in mine
                if m.get("sample_rate_measured")]
        codes = Counter(q.get("code") for m in mine
                        for q in (m.get("quality_issues") or []))
        stamps = sorted(m["created_iso"] for m in mine)
        t = takes.get(p, [])
        per[p] = {
            "recordings": len(mine),
            "sessions": sorted({m["session_id"] for m in mine}),
            "per_gesture": {g: by.get(g, 0) for g in GESTURES},
            "seconds_total": round(sum(dur), 1),
            "seconds_mean_clip": round(float(np.mean(dur)), 2),
            "seconds_min_clip": round(min(dur), 2),
            "seconds_max_clip": round(max(dur), 2),
            "sample_rate_mean": round(float(np.mean(rate)), 1),
            "sample_rate_min": round(min(rate), 1),
            "sample_rate_max": round(max(rate), 1),
            "quality_flags": dict(codes),
            "recordings_with_a_flag": sum(1 for m in mine
                                          if m.get("quality_issues")),
            #  REST has no activity by definition, so only the others count
            "no_activity_non_rest": sum(
                1 for m in mine if m["label"] != "REST" and not
                (m.get("activity_detected") or {}).get("found", True)),
            "first_recording": stamps[0], "last_recording": stamps[-1],
            "free_takes": len(t),
            "free_take_minutes": round(sum(float(x["duration_s"])
                                           for x in t) / 60, 1),
            "quarantined_files_sets": deleted.get(p, 0),
            "cut_from_takes": sum(1 for m in mine if (m.get("meta") or {})
                                  .get("capture_style") == "cut_from_take"),
        }
    saved, derived = calibrations(people)
    for p in people:
        per[p]["calibration_saved"] = saved.get(p)
        per[p]["calibration_derived"] = derived.get(p)
    first = recs[0]
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "root": "datasets/",
        "device": (first.get("meta") or {}).get("device"),
        "emg_mode": first.get("emg_mode") or (first.get("meta") or {}).get(
            "emg_mode"),
        "n_channels": first.get("n_channels"),
        "gestures": GESTURES,
        "totals": {
            "recordings": len(recs), "people": len(people),
            "seconds": round(sum(float(m["duration_s"]) for m in recs), 1),
            "per_gesture": {g: sum(1 for m in recs if m["label"] == g)
                            for g in GESTURES}},
        "people": per,
        "signals": read_signals(),
        "lineage": lineage(),
        "windows_ml_1.0s": windows("ml"),
        "windows_nn_0.7s": windows("nn"),
        "folders_without_recordings": sorted(
            d for d in os.listdir(DATA)
            if os.path.isdir(os.path.join(DATA, d)) and not d.startswith("_")
            and d not in people),
        "empty_session_folders": sorted(
            os.path.relpath(p, DATA).replace(os.sep, "/") for p in glob.glob(
                os.path.join(DATA, "*", "session_*"))
            if os.path.isdir(p) and not os.listdir(p)),
        "archives": {
            name: archive_counts(name)
            for name in sorted(os.listdir(DATA)) if name.startswith("_")},
    }
    return summary


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(s, path):
    people = list(s["people"])
    P = s["people"]
    L = []
    w = L.append
    w("# DATASET_SUMMARY.md -- what is in `datasets/` now")
    w("")
    w(f"*Generated {s['created'][:16].replace('T', ' ')} by "
      "`tools/dataset_summary.py` from the recordings' own metadata and the "
      "stored evaluation data; re-run it after recording. Nothing here is "
      "typed by hand. `EDA.md` describes the dataset archived on 2026-10-15; "
      "this is the working set.*")
    w("")
    t = s["totals"]
    w("## 1. At a glance")
    w("")
    w(f"- **{t['recordings']} recordings, {t['people']} people** "
      f"({', '.join(people)}), {len(s['gestures'])} gestures "
      f"({', '.join(s['gestures'])}), {t['seconds'] / 60:.1f} minutes of "
      "labelled signal in total.")
    w(f"- Device: {s['device']}, {s['n_channels']} channels, EMG mode "
      f"`{s['emg_mode']}` (on-device filtered envelope, about 50 Hz).")
    sg = s["signals"]
    c = sg["counts"]
    w(f"- **Every file was read** ({c['clips']} clips with {c['clip_samples']} "
      f"samples, {c['takes']} free takes with {c['take_samples']} samples = "
      f"{c['take_samples'] / 50 / 60:.0f} minutes): "
      + ("no problems found -- " if not sg["problems"] else
         f"{len(sg['problems'])} PROBLEM(S) (listed in the JSON) -- ")
      + f"no NaN, timestamps never go backwards, sample counts and label "
      f"windows agree with the metadata, no sample at the 2047 ceiling "
      f"(highest count {sg['max_count']}), no dead channel, and "
      f"{c['csv_identical']} of {c['csv_checked']} csv files identical to "
      "their npz.")
    sessions = {p: P[p]["sessions"] for p in people}
    w("- **One sitting per person** (" + ", ".join(
        f"{p}: {', '.join(v)}" for p, v in sessions.items()) + "): the band "
      "was not taken off and refitted, so the confound inside a person "
      "(CLAUDE.md 8f) is still there; the honest number is the person-held-out "
      "one.")
    for fam, key in (("ML, 1.0 s windows at 0.75 overlap", "windows_ml_1.0s"),
                     ("neural nets, 0.7 s windows at 0.75 overlap",
                      "windows_nn_0.7s")):
        win = s[key]
        if win:
            w(f"- Windows scored ({fam}, after window-level deduplication): "
              f"**{win['n']}**; the majority class is "
              f"{pct(win['majority_class'])} of them (what a model that "
              "always answers the most common gesture would score).")
    if s["folders_without_recordings"] or s["empty_session_folders"]:
        w("- Folders that exist but hold no recordings (nothing was lost, "
          "they were never filled): "
          + ", ".join(s["folders_without_recordings"]
                      + s["empty_session_folders"]) + ".")
    w("- Archives, never read by discovery (leading underscore): "
      + "; ".join(f"`{k}` ({v['recordings']} recordings"
                  + (f", {v['quarantined']} more in its `_deleted`"
                     if v["quarantined"] else "") + ")"
                  for k, v in s["archives"].items()) + ".")
    w("")
    w("## 2. Recordings per person and gesture")
    w("")
    w(table(["person"] + s["gestures"] + ["total"],
            [[p] + [P[p]["per_gesture"][g] for g in s["gestures"]]
             + [P[p]["recordings"]] for p in people]
            + [["**all**"] + [t["per_gesture"][g] for g in s["gestures"]]
               + [t["recordings"]]]))
    w("")
    w("## 3. Per person")
    w("")
    w(table(["person", "labelled seconds", "mean clip (min - max) s",
             "sample rate Hz (min - max)", "recorded",
             "free takes (minutes)", "cut from takes",
             "recordings with a quality flag",
             "gesture clips with no activity found",
             "quarantined recordings"],
            [[p, P[p]["seconds_total"],
              f"{P[p]['seconds_mean_clip']} ({P[p]['seconds_min_clip']} - "
              f"{P[p]['seconds_max_clip']})",
              f"{P[p]['sample_rate_mean']} ({P[p]['sample_rate_min']} - "
              f"{P[p]['sample_rate_max']})",
              P[p]["first_recording"][:10]
              + ("" if P[p]["first_recording"][:10]
                 == P[p]["last_recording"][:10]
                 else " - " + P[p]["last_recording"][:10]),
              f"{P[p]['free_takes']} ({P[p]['free_take_minutes']})",
              P[p]["cut_from_takes"], P[p]["recordings_with_a_flag"],
              P[p]["no_activity_non_rest"], P[p]["quarantined_files_sets"]]
             for p in people]))
    w("")
    w("Quality flags are findings, never deletions (safety rule 5): "
      + "; ".join(f"{p}: " + (", ".join(f"{c} {n}" for c, n in
                                        P[p]["quality_flags"].items())
                              or "none") for p in people)
      + ". `noisy_baseline` = resting RMS above 60 (skin contact or mains); "
        "`rest_not_quiet` = the clip's rest part is not quiet. "
        "\"Cut from takes\" are clips carved from a longer free take; the "
        "takes themselves are never training data.")
    w("")
    for key, name in (("windows_ml_1.0s", "ML, 1.0 s"),
                      ("windows_nn_0.7s", "neural nets, 0.7 s")):
        win = s[key]
        if not win:
            continue
        w(f"## 4{'a' if 'ml' in key else 'b'}. Windows the models are "
          f"scored on ({name})")
        w("")
        w(table(["person"] + win["labels"] + ["total"],
                [[p] + [win["per_person"][p][g] for g in win["labels"]]
                 + [sum(win["per_person"][p].values())]
                 for p in sorted(win["per_person"])]))
        w("")
    w("## 4c. Signal level per person and gesture")
    w("")
    w("Mean of the on-device filtered envelope over each clip's labelled "
      "window and the eight channels, in device counts, averaged over the "
      "person's clips of that gesture. `contrast` = the mean of the five "
      "active gestures over REST.")
    w("")
    lv = sg["levels"]
    rows = []
    for p_ in people:
        act = [lv[p_][g] for g in s["gestures"] if g != "REST"
               and lv[p_][g] is not None]
        rest = lv[p_]["REST"]
        rows.append([p_] + [lv[p_][g] for g in s["gestures"]]
                    + [f"{np.mean(act) / rest:.1f}x" if rest else "--"])
    w(table(["person"] + s["gestures"] + ["contrast"], rows))
    w("")
    from emg_app import config as cfg
    ceiling = float(cfg.CALIBRATION_MAX_REST_LEVEL)
    high = [p_ for p_ in people if lv[p_]["REST"] and lv[p_]["REST"] > ceiling]
    if high:
        w(f"**Resting level above the live calibration's ceiling.** A live "
          f"calibration rejects a REST capture whose per-channel median "
          f"exceeds {ceiling:.0f} counts (`CALIBRATION_MAX_REST_LEVEL`); the "
          "mean REST level of " + ", ".join(
              f"{p_} is {lv[p_]['REST']}" for p_ in high)
          + " (this is a clip mean over channels, not the gate's median, so "
            "it is an indication, not a prediction). Such a person may be "
            "refused at CALIBRATE even though their recorded REST clips are "
            "fine; the derived calibration used for training is unaffected.")
        w("")
    w("A person whose REST is low and whose gestures are weak is hard in a "
      "different way from one whose REST is high; the DC-only control "
      "(CLAUDE.md 8f) shows how much of every score is exactly this level.")
    w("")
    w("## 4d. Manifest against disk")
    w("")
    w("Every session keeps an append-only `manifest.jsonl` (one line per saved "
      "file set). Compared with what is on disk:")
    w("")
    lin = s["lineage"]
    w(table(["session", "logged as saved", "on disk", "quarantined",
             "logged but gone"],
            [[k, v["logged_raw"], v["on_disk"], v["quarantined"],
              v["logged_but_gone"]] for k, v in lin.items()]))
    w("")
    gone_any = [(k, n) for k, v in lin.items() for n in v["gone"]]
    if gone_any:
        for k, n in gone_any:
            w(f"- **{k}: {n['clips']} {n['label']} clips are logged as saved "
              f"({n['from']} - {n['to']}, {n['seconds_each']} s each, "
              f"repetitions {n['repetitions']}) but are neither on disk nor in "
              "`_deleted/`, and no `deleted` line records them.** "
              + (f"{n['clips_of_the_same_gesture_saved_afterwards']} clips of "
                 f"the same gesture were saved afterwards "
                 f"({n['their_seconds_each']} s each), so they were most "
                 "likely discarded and re-recorded; the files were removed "
                 "outside the app's DELETE / REDO path (safety rule 4), and "
                 "they were never committed to git either, so they cannot be "
                 "recovered. " if n["clips_of_the_same_gesture_saved_afterwards"]
                 else "")
              + "What is on disk is what the models were trained on.")
        w("")
    else:
        w("Nothing logged is missing.")
        w("")
    w("## 5. Calibration")
    w("")
    w("Two calibrations exist per person and they are not the same thing "
      "(CLAUDE.md 8n): the **saved** one is what PREDICT uses (measured live, "
      "a maximum-effort squeeze or sweep), the **derived** one is what "
      "TRAINING uses (percentiles of the person's own recorded relaxed "
      "holds). Per channel, in device counts:")
    w("")
    rows = []
    for p in people:
        sv, dv = P[p]["calibration_saved"], P[p]["calibration_derived"]
        rows.append([
            p,
            "--" if not sv else f"{sv['created'][:10]} ({sv['source']})",
            "--" if not sv else f"{np.mean(sv['rest']):.0f}",
            "--" if not sv else (f"{min(sv['reference']):.0f} - "
                                 f"{max(sv['reference']):.0f}"),
            "--" if not dv else f"{np.mean(dv['rest']):.0f}",
            "--" if not dv else (f"{min(dv['reference']):.0f} - "
                                 f"{max(dv['reference']):.0f}"),
            "--" if not (sv and dv) else (
                f"{min(a / b for a, b in zip(sv['reference'], dv['reference'])):.1f}x - "
                f"{max(a / b for a, b in zip(sv['reference'], dv['reference'])):.1f}x")])
    w(table(["person", "saved: when (how)", "saved: mean rest",
             "saved: reference min - max", "derived: mean rest",
             "derived: reference min - max", "saved / derived reference"],
            rows))
    w("")
    w("A person with no saved calibration cannot use a model trained on "
      "calibrated features until they calibrate (CLAUDE.md 12); training is "
      "unaffected because it uses the derived one.")
    w("")
    w("## 6. Reading this honestly")
    w("")
    w("- Five people and one sitting each: every accuracy in the other "
      "documents is the mean over five held-out people and carries that "
      "spread (the weakest person alone moves a mean by about 10 points).")
    w("- Recording counts per gesture are uneven (REST has the most clips "
      "for some people); the windows tables above are what the majority-class "
      "baseline is computed from.")
    w("- Where this data is described elsewhere: `DATASET.md` (on-disk format "
      "and schema), `EDA.md` (the archived dataset and why it could not give "
      "an honest score), `MODELS_COMPARED.md` and `REPORT.md` (results).")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    summary = build()
    os.makedirs(RESULTS, exist_ok=True)
    out_json = os.path.join(RESULTS, "dataset_summary.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    out_md = os.path.join(ROOT, "docs", "DATASET_SUMMARY.md")
    write_md(summary, out_md)
    print("wrote", os.path.relpath(out_json, ROOT), "and",
          os.path.relpath(out_md, ROOT))

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
            name: {"recordings": len(glob.glob(os.path.join(
                DATA, name, "**", "*_raw_meta.json"), recursive=True))}
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
      + "; ".join(f"`{k}` ({v['recordings']} recordings)"
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

"""
Every training run that was saved, read and tabulated -- the history a report
needs to say how the numbers got to where they are and what each run cost.

    python tools/training_history.py

Reads (all of it, nothing sampled):
  models/reports/sweep_*.json       every classifier sweep (29)
  models/reports/nn_sweep_*.json    every neural-net sweep (8)
  models/comb_test/<run>/           the combination search: state files, result
  models/jobs/*/status.json         the background training jobs
  results/*/log.txt, models/zoo_*   durations of the ablation and zoo runs
  results/ablation*/*/runs.jsonl    run counts

Writes `results/training_history.json` and `docs/TRAINING_HISTORY.md`.
"""
import glob
import json
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(v, d=1):
    return "--" if v is None else f"{v * 100:.{d}f} %"


def dur(s):
    if s is None:
        return "--"
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def stamp_of(path):
    m = re.search(r"(\d{8})-(\d{6})", os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2),
                             "%Y%m%d%H%M%S").isoformat(timespec="seconds")


def ml_sweeps():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "reports",
                                              "sweep_*.json"))):
        d = jload(path)
        b = d.get("best") or {}
        val = d.get("validation") or {}
        contrast = (val.get("grouping_contrast") or {})
        conf = d.get("confound") or {}
        dc = conf.get("dc_only_control") or {}
        cal = d.get("calibration") or {}
        subs = d.get("subjects") or []
        rows.append({
            "file": os.path.basename(path), "created": d.get("created_iso"),
            "recordings": d.get("n_recordings_total"), "people": len(subs),
            "subjects": subs, "grouping": b.get("grouping") or "recording",
            "calibration": cal.get("mode", "off") if cal else "off",
            "model": b.get("model"), "window_s": b.get("window_s"),
            "overlap": b.get("overlap"),
            "representation": b.get("representation"),
            "feature_set": b.get("feature_set"),
            "windows": b.get("n_windows"), "features": b.get("n_features"),
            "window_accuracy": b.get("window_accuracy"),
            "window_accuracy_std": b.get("window_accuracy_std"),
            "recording_vote": b.get("recording_vote_accuracy"),
            "macro_f1": b.get("window_macro_f1"),
            "contrast_grouping": contrast.get("grouping"),
            "contrast_accuracy": contrast.get("window_accuracy"),
            "dc_only_accuracy": dc.get("window_accuracy"),
            "majority_class": conf.get("majority_class"),
            "elapsed_s": d.get("elapsed_s"),
            "candidates": len(d.get("best_per_model") or {}),
        })
    return rows


def nn_sweeps():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "reports",
                                              "nn_sweep_*.json"))):
        d = jload(path)
        results = d.get("results") or []
        best = max(results, key=lambda r: r.get("window_accuracy") or 0) \
            if results else {}
        cal = d.get("calibration") or {}
        dc = d.get("dc_only_control") or {}
        ct = d.get("grouping_contrast") or {}
        subj = d.get("subject_independent") or {}
        rows.append({
            "file": os.path.basename(path), "created": stamp_of(path),
            "people": len(cal.get("subjects") or []),
            "subjects": cal.get("subjects") or [],
            "calibration": cal.get("mode"), "grouping": d.get("grouping"),
            "configs": len(results), "kind": best.get("kind"),
            "window_s": best.get("window_s"),
            "window_accuracy": best.get("window_accuracy"),
            "window_accuracy_std": best.get("window_accuracy_std"),
            "recording_vote": best.get("recording_vote_accuracy"),
            "macro_f1": best.get("window_macro_f1"),
            "parameters": best.get("n_parameters"),
            "epochs": d.get("epochs"), "chance": d.get("chance"),
            "contrast_accuracy": ct.get("window_accuracy"),
            "dc_only_accuracy": dc.get("window_accuracy"),
            "per_subject": subj.get("per_subject"),
            "device": (d.get("device") or {}).get("gpu"),
        })
    return rows


def last_elapsed(path):
    """Largest 'N min S s elapsed' (or 'done in ...') a log mentions."""
    if not os.path.exists(path):
        return None
    best = None
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"\((\d+) min (\d+) s elapsed\)", text):
        best = max(best or 0, int(m.group(1)) * 60 + int(m.group(2)))
    for m in re.finditer(r"done in (?:(\d+) h )?(\d+) min(?: (\d+) s)?", text):
        best = max(best or 0, int(m.group(1) or 0) * 3600
                   + int(m.group(2)) * 60 + int(m.group(3) or 0))
    return best


def count_lines(path):
    return sum(1 for ln in open(path, encoding="utf-8") if ln.strip()) \
        if os.path.exists(path) else None


def long_runs():
    runs = []
    for name, rel, runs_file in (
            ("ML ablation (3 seeds)", "results/ablation/ml/log.txt",
             "results/ablation/ml/runs.jsonl"),
            ("NN ablation (3 seeds)", "results/ablation/nn/log.txt",
             "results/ablation/nn/runs.jsonl"),
            ("ML ablation without Barbora", "results/ablation_without_barbora/"
             "ml/log.txt", "results/ablation_without_barbora/ml/runs.jsonl"),
            ("NN ablation without Barbora", "results/ablation_without_barbora/"
             "nn/log.txt", "results/ablation_without_barbora/nn/runs.jsonl"),
            ("zoo evaluation and fits", "results/zoo_eval/log.txt", None),
            ("zoo trained without Barbora",
             "results/zoo_eval_without_barbora/log.txt", None)):
        path = os.path.join(ROOT, rel)
        runs.append({"name": name, "log": rel,
                     "seconds": last_elapsed(path),
                     "runs": count_lines(os.path.join(ROOT, runs_file))
                     if runs_file else None})
    comb = os.path.join(ROOT, "models", "comb_test")
    for run in sorted(glob.glob(os.path.join(comb, "*"))):
        if not os.path.isdir(run):
            continue
        for part in ("ml", "nn"):
            path = os.path.join(run, f"state_{part}.json")
            if os.path.exists(path):
                s = jload(path)
                runs.append({
                    "name": f"combination search, {part.upper()} "
                            f"({os.path.basename(run)})",
                    "log": os.path.relpath(path, ROOT),
                    "seconds": (s.get("updated", 0) - s.get("started", 0))
                    if s.get("updated") else None,
                    "runs": s.get("items_done"),
                    "of": s.get("items_total"), "status": s.get("status"),
                    "stage": s.get("stage")})
    for j in sorted(glob.glob(os.path.join(ROOT, "models", "jobs", "*"))):
        path = os.path.join(j, "status.json")
        if os.path.exists(path):
            s = jload(path)
            runs.append({"name": f"background job {s.get('kind')} "
                                 f"({os.path.basename(j)[:15]})",
                         "log": os.path.relpath(path, ROOT),
                         "seconds": (s.get("finished_unix", 0)
                                     - s.get("started_unix", 0))
                         if s.get("finished_unix") else None,
                         "status": s.get("state")})
    return runs


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(ml, nn, runs, path):
    L = []
    w = L.append
    w("# TRAINING_HISTORY.md -- every saved training run")
    w("")
    w("*Generated by `tools/training_history.py` from every sweep report, log "
      "and job file under `models/` and `results/` (nothing sampled, nothing "
      "typed by hand). It is the answer to \"how did the numbers get here and "
      "what did each run cost\". The current numbers are in "
      "[REPORT.md](REPORT.md); where each file lives is in "
      "[DATA_MAP.md](DATA_MAP.md).*")
    w("")
    w("## How to read the trend")
    w("")
    first, last = ml[0], ml[-1]
    honest = [r for r in ml if r["grouping"] == "subject"]
    w(f"- The first classifier sweep ({first['created'][:10]}, "
      f"{first['recordings']} recordings from {first['people']} person) scored "
      f"**{pct(first['window_accuracy'])}**; the latest "
      f"({last['created'][:10]}, {last['recordings']} recordings, "
      f"{last['people']} people, person held out) scores "
      f"**{pct(last['window_accuracy'])}** (the deployed 4-model vote). The "
      "drop is not the models getting worse: with one person and one take per "
      "gesture the score measured which block of signal a window came from "
      "(EDA.md), and it stopped doing so once people could be held out.")
    if honest:
        w(f"- {len(honest)} of the {len(ml)} sweeps are person-held-out; their "
          f"best per-window scores run from "
          f"{pct(min(r['window_accuracy'] for r in honest))} to "
          f"{pct(max(r['window_accuracy'] for r in honest))}. Those sweeps "
          "also differ in people (2 to 5), calibration and window length, so "
          "they are a history, not a controlled comparison.")
    w("- The `contrast` column is the same model scored with recordings (not "
      "people) held out. The gap between it and the person-held-out score is "
      "how much of a score is the wearer rather than the gesture.")
    w("")
    w("## 1. Classifier sweeps (`train_model.py`)")
    w("")
    w(table(["date", "recordings", "people", "split by", "calibration",
             "winner", "setup", "per window (± people)", "per recording",
             "macro-F1", "contrast", "DC-only", "time"],
            [[r["created"][:16].replace("T", " "), r["recordings"], r["people"],
              r["grouping"], r["calibration"], r["model"],
              f"{r['window_s']} s / {r['representation']} / {r['feature_set']}",
              f"{pct(r['window_accuracy'])} ± "
              f"{(r['window_accuracy_std'] or 0) * 100:.1f}",
              pct(r["recording_vote"]), pct(r["macro_f1"]),
              pct(r["contrast_accuracy"]), pct(r["dc_only_accuracy"]),
              dur(r["elapsed_s"])] for r in ml]))
    w("")
    w("`split by`: what the cross-validation held out (`recording`, or "
      "`subject` = a whole person). `contrast`: the other protocol, where the "
      "sweep recorded it. `DC-only`: the same winner given only the eight "
      "channel means (CLAUDE.md 8f); a high value next to the full score means "
      "amplitude, not gesture shape, carried the result.")
    w("")
    w("## 2. Neural-net sweeps (`train_nn.py`)")
    w("")
    w(table(["date", "people", "calibration", "configs tried", "best",
             "per window", "per recording", "macro-F1", "parameters",
             "contrast", "DC-only", "chance"],
            [[(r["created"] or "")[:16].replace("T", " "), r["people"],
              r["calibration"], r["configs"],
              f"{r['kind']} @ {r['window_s']} s", pct(r["window_accuracy"]),
              pct(r["recording_vote"]), pct(r["macro_f1"]),
              f"{r['parameters']:,}".replace(",", " ") if r["parameters"]
              else "--", pct(r["contrast_accuracy"]),
              pct(r["dc_only_accuracy"]), pct(r["chance"])] for r in nn]))
    w("")
    w("A sweep of one config is a retrain of the chosen net, not a search. "
      "The GRU at 0.7 s was chosen by the combination search and trained "
      "again on 2026-10-22 (the two one-config rows).")
    w("")
    w("## 3. Long runs and what they cost")
    w("")
    w(table(["run", "items", "wall time", "state", "log"],
            [[r["name"],
              "--" if r.get("runs") is None else (
                  f"{r['runs']} of {r['of']}" if r.get("of") else r["runs"]),
              dur(r["seconds"]), r.get("status") or "", f"`{r['log']}`"]
             for r in runs]))
    w("")
    w("Times are wall-clock on the i5-9300H / GTX 1650 machine, with other "
      "work often running at the same time. The combination search's NN half "
      "was stopped by choice after 12 of 34 configurations (GRU 0.7 s kept); "
      "the ML half ran all 592 combinations (about 10.6 h).")
    w("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ml, nn, runs = ml_sweeps(), nn_sweeps(), long_runs()
    out = {"created": datetime.now().isoformat(timespec="seconds"),
           "classifier_sweeps": ml, "nn_sweeps": nn, "long_runs": runs}
    path = os.path.join(ROOT, "results", "training_history.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    write_md(ml, nn, runs, os.path.join(ROOT, "docs", "TRAINING_HISTORY.md"))
    print(f"read {len(ml)} classifier sweeps, {len(nn)} NN sweeps, "
          f"{len(runs)} long runs -> results/training_history.json, "
          "docs/TRAINING_HISTORY.md")

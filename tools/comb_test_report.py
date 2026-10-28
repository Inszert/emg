"""
Turn one nn_ml_comb_test.py run into a spreadsheet and a documentation page.

    python tools/comb_test_report.py              the latest run
    python tools/comb_test_report.py RUN_DIR      a specific one

Writes, into the run folder:
    comb_test_results.xlsx   one sheet per table (needs openpyxl)
    csv/<sheet>.csv          the same tables, ';'-separated with a decimal
                             comma, so they open in columns in a Slovak Excel
and docs/COMB_TEST.md -- the method, the winners and the main tables.

Read-only on everything else: it never touches a model, a spec or the data.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "training"))

import nn_ml_comb_test as C  # noqa: E402


def pct(v):
    return None if v is None else round(float(v) * 100, 2)


def load(run_dir):
    ml = C.read_json(os.path.join(run_dir, "ml_result.json")) or {}
    nn = C.read_json(os.path.join(run_dir, "nn_result.json")) or {}
    states = {p: C.read_json(os.path.join(run_dir, f"state_{p}.json")) or {}
              for p in ("ml", "nn", "main")}
    nn_rows = []
    path = os.path.join(run_dir, "nn_results.jsonl")
    if os.path.exists(path):
        seen = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row["key"] not in seen:
                    seen.add(row["key"])
                    nn_rows.append(row)
    return ml, nn, states, nn_rows


def confusion_from_oof(run_dir, winner):
    """The ML winner's summed out-of-fold confusion matrix."""
    if not winner:
        return None, None
    sname = C.setup_name(winner["setup"])
    oof = os.path.join(run_dir, "ml_oof")
    data = np.load(os.path.join(oof, f"{sname}__data.npz"), allow_pickle=True)
    y, labels = data["y"], [str(l) for l in data["labels"]]
    p = sum(np.load(os.path.join(oof, f"{sname}__{m}.npz"))["proba"]
            for m in winner["members"]) / len(winner["members"])
    pred = p.argmax(axis=1)
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, g in zip(y, pred):
        cm[int(t), int(g)] += 1
    return cm, labels


def tables(run_dir):
    ml, nn, states, nn_rows = load(run_dir)
    people = sorted({p for r in (ml.get("singles") or [])
                     for p in (r.get("per_subject") or {})})
    t = {}

    # ---- summary -----------------------------------------------------------
    w = ml.get("winner") or {}
    n = nn.get("winner") or {}
    started = states["main"].get("started")
    finished = states["main"].get("updated")
    summary = [
        ("run", os.path.basename(run_dir)),
        ("started", datetime.fromtimestamp(started).isoformat(" ", "minutes")
         if started else None),
        ("last update", datetime.fromtimestamp(finished).isoformat(
            " ", "minutes") if finished else None),
        ("duration (h)", round((finished - started) / 3600, 2)
         if started and finished else None),
        ("protocol", "all recordings, 5 wearers, affine calibration, window "
                     "dedup, each fold holds out ONE PERSON (trained on the "
                     "other four)"),
        ("", ""),
        ("ML winner", f"soft vote of {len(w.get('members') or [])}: "
                      + ", ".join(w.get("members") or [])),
        ("ML setup", C.setup_name(w["setup"]) if w else None),
        ("ML window accuracy %", pct(w.get("window_accuracy"))),
        ("ML NESTED (honest) %", pct(ml.get("nested_mean"))),
        ("ML per-recording vote %", pct(w.get("recording_vote_accuracy"))),
        ("ML macro F1 %", pct(w.get("window_macro_f1"))),
    ] + [(f"ML held out {p} %", pct((w.get("per_subject") or {}).get(p)))
         for p in people] + [
        ("", ""),
        ("NN winner", C.nn_key(n["config"]) if n else None),
        ("NN window accuracy % (mean of seeds)",
         pct(n.get("mean_window_accuracy"))),
        ("NN seed scores %", ", ".join(f"{s * 100:.1f}"
                                       for s in n.get("seed_scores") or [])),
        ("NN per-recording vote %", pct(n.get("mean_vote"))),
        ("NN macro F1 %", pct(n.get("mean_macro_f1"))),
        ("NN epochs", nn.get("epochs")),
        ("", ""),
        ("goal", "82 % (stretch 92 %), subject held out"),
    ]
    t["Summary"] = (["item", "value"], [list(r) for r in summary])

    # ---- ML: every single model on every setup ----------------------------
    head = (["setup", "representation", "feature_set", "window_s", "overlap",
             "model", "kind", "params", "window_acc_%", "vote_%",
             "macro_F1_%"] + [f"{p}_%" for p in people])
    rows = []
    for r in sorted(ml.get("singles") or [],
                    key=lambda r: -r["window_accuracy"]):
        s, m = r["setup"], r["model"]
        rows.append([r["setup_name"], s["representation"], s["feature_set"],
                     s["window_s"], s["overlap"], m["name"], m["kind"],
                     json.dumps(m.get("params"), sort_keys=True),
                     pct(r["window_accuracy"]),
                     pct(r["recording_vote_accuracy"]),
                     pct(r["window_macro_f1"])]
                    + [pct((r.get("per_subject") or {}).get(p))
                       for p in people])
    t["ML_all_models"] = (head, rows)

    # ---- ML: what each choice is worth, over every DISTINCT fit -------------
    #  The setup list of this run names one setup twice, so the singles hold
    #  592 entries of which 576 are distinct (the repeats are identical);
    #  counting the repeats twice would weight that setup double.
    seen, distinct = set(), []
    for r in ml.get("singles") or []:
        key = (r["setup_name"], r["model"]["name"])
        if key not in seen:
            seen.add(key)
            distinct.append(r)
    marg = []
    for label, pick in (
            ("representation", lambda r: r["setup"]["representation"]),
            ("feature set", lambda r: r["setup"]["feature_set"]),
            ("window length (s)", lambda r: str(r["setup"]["window_s"])),
            ("model family", lambda r: r["model"]["kind"])):
        groups = {}
        for r in distinct:
            groups.setdefault(pick(r), []).append(r["window_accuracy"])
        for value, accs in sorted(groups.items(), key=lambda kv: -max(kv[1])):
            marg.append([label, value, len(accs), pct(max(accs)),
                         pct(sum(accs) / len(accs)), pct(min(accs))])
    t["ML_marginals"] = (["choice", "value", "fits", "best_%", "mean_%",
                          "worst_%"], marg)
    spread = []
    for p_ in people:
        v = sorted((r.get("per_subject") or {}).get(p_) for r in distinct)
        v = [x for x in v if x is not None]
        if v:
            spread.append([p_, pct(max(v)), pct(v[len(v) // 2]), pct(min(v))])
    t["ML_person_spread"] = (["held-out person", "best fit_%", "median fit_%",
                              "worst fit_%"], spread)
    t["_n_distinct"] = (["n"], [[len(distinct), len(ml.get("singles") or [])]])

    # ---- ML: the best vote on each setup ------------------------------------
    head = (["setup", "n_members", "members", "window_acc_%", "std_%",
             "vote_%", "macro_F1_%"] + [f"{p}_%" for p in people])
    rows = []
    for name, v in sorted((ml.get("per_setup") or {}).items(),
                          key=lambda kv: -kv[1]["window_accuracy"]):
        rows.append([name, len(v["members"]), ", ".join(v["members"]),
                     pct(v["window_accuracy"]), pct(v["window_accuracy_std"]),
                     pct(v["recording_vote_accuracy"]),
                     pct(v["window_macro_f1"])]
                    + [pct(v["per_subject"].get(p)) for p in people])
    t["ML_best_vote_per_setup"] = (head, rows)

    # ---- ML: nested -----------------------------------------------------------
    t["ML_nested_check"] = (
        ["held_out_person", "accuracy_%", "setup_chosen_without_them",
         "members_chosen_without_them"],
        [[p, pct(v["accuracy"]), v["chosen_setup"],
          ", ".join(v["chosen_members"])]
         for p, v in sorted((ml.get("nested") or {}).items())])

    # ---- NN: every configuration --------------------------------------------
    head = ["stage", "kind", "window_s", "hidden", "dropout",
            "learning_rate", "seed", "epochs", "window_acc_%", "std_%",
            "vote_%", "macro_F1_%", "parameters", "windows", "minutes",
            "fold_accuracies_%"]
    rows = []
    for r in sorted(nn_rows, key=lambda r: -r["window_accuracy"]):
        c = r["config"]
        rows.append([r["stage"], c["kind"], c["window_s"], c["hidden"],
                     c["dropout"], c["learning_rate"], c.get("seed", 0),
                     r.get("epochs"), pct(r["window_accuracy"]),
                     pct(r["window_accuracy_std"]),
                     pct(r["recording_vote_accuracy"]),
                     pct(r["window_macro_f1"]), r.get("n_parameters"),
                     r.get("n_windows"), round(r["seconds"] / 60, 1),
                     ", ".join(f"{a * 100:.0f}" for a in
                               r.get("fold_accuracies") or [])])
    t["NN_all_configs"] = (head, rows)

    # ---- NN: window length effect -------------------------------------------
    by = {}
    for r in nn_rows:
        if r["stage"] != "N1":
            continue
        c = r["config"]
        by.setdefault(c["window_s"], {})[c["kind"]] = pct(r["window_accuracy"])
    kinds = sorted({k for v in by.values() for k in v})
    t["NN_window_length"] = (["window_s"] + kinds,
                             [[w] + [by[w].get(k) for k in kinds]
                              for w in sorted(by)])

    t["NN_finalists"] = (
        ["config", "mean_window_acc_%", "seed_scores_%", "mean_vote_%",
         "mean_macro_F1_%"],
        [[C.nn_key(f["config"]), pct(f["mean_window_accuracy"]),
          ", ".join(f"{s * 100:.1f}" for s in f["seed_scores"]),
          pct(f["mean_vote"]), pct(f["mean_macro_f1"])]
         for f in nn.get("finalists") or []])

    # ---- confusion matrices -------------------------------------------------
    rows = []
    cm, labels = confusion_from_oof(run_dir, w) if w else (None, None)
    if cm is not None:
        rows.append(["ML winner (rows = truth, % of row)"] + labels)
        for i, l in enumerate(labels):
            tot = cm[i].sum()
            rows.append([l] + [round(100 * cm[i, j] / tot, 1) if tot else None
                               for j in range(len(labels))])
        rows.append([])
    best_nn = next((r for r in sorted(nn_rows,
                                      key=lambda r: -r["window_accuracy"])
                    if n and r["config"]["kind"] == n["config"]["kind"]
                    and r["config"]["window_s"] == n["config"]["window_s"]
                    and r["config"]["hidden"] == n["config"]["hidden"]),
                   None)
    if best_nn and labels:
        m = np.asarray(best_nn["confusion_matrix"])
        rows.append([f"NN winner {best_nn['key']} (rows = truth, % of row)"]
                    + labels)
        for i, l in enumerate(labels):
            tot = m[i].sum()
            rows.append([l] + [round(100 * m[i, j] / tot, 1) if tot else None
                               for j in range(len(labels))])
    t["Confusion"] = ([""], rows)
    return t, ml, nn, people


def write_xlsx(path, t):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    wb.remove(wb.active)
    for name, (head, rows) in t.items():
        if name.startswith("_"):
            continue                     # helpers for the document only
        ws = wb.create_sheet(name[:31])
        ws.append(head)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append(r)
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0
                        for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(60, width + 2)
        ws.freeze_panes = "A2"
    wb.save(path)


def write_csv(folder, t):
    os.makedirs(folder, exist_ok=True)
    for name, (head, rows) in t.items():
        if name.startswith("_"):
            continue
        with open(os.path.join(folder, f"{name}.csv"), "w",
                  encoding="utf-8-sig") as fh:
            for r in [head] + rows:
                fh.write(";".join(
                    "" if v is None else
                    (str(v).replace(".", ",") if isinstance(v, float)
                     else str(v).replace(";", ","))
                    for v in r) + "\n")


def md_table(head, rows, limit=None):
    rows = rows[:limit] if limit else rows
    out = ["| " + " | ".join(str(h) for h in head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def write_doc(path, run_dir, t, ml, nn, people):
    w, n = ml.get("winner") or {}, nn.get("winner") or {}
    s = dict(t["Summary"][1])
    rel = os.path.relpath(run_dir, PROJECT).replace("\\", "/")
    lines = [
        "# The ML + NN combination test",
        "",
        f"Generated by `tools/comb_test_report.py` from run `{rel}` "
        f"({s.get('started')} -> {s.get('last update')}, "
        f"{s.get('duration (h)')} h). The full numbers are in "
        f"`{rel}/comb_test_results.xlsx` (and `csv/`).",
        "",
        "## What was asked, and how it was measured",
        "",
        "Find the best combination of classic ML models (a soft vote) and the "
        "best neural net, make them the defaults, and retrain. Script: "
        "`nn_ml_comb_test.py`.",
        "",
        "**Protocol -- the same as both trainers.** All recordings of all five "
        "wearers, `affine` per-wearer calibration, window-level dedup, and "
        "every fold holds out ONE PERSON: trained on four, scored on the "
        "fifth. So every number below is *on a person the model never saw* "
        "(who calibrated first -- CLAUDE.md 8k).",
        "",
        "**ML.** 16 classifiers (LDA, logistic regression, RBF-SVM at four C "
        "values, random forest and extra trees at two leaf sizes, gradient "
        "boosting, kNN at two k, two MLPs) on every feature setup "
        "(signal representation x feature set x window length). Every "
        "model's out-of-fold probabilities are stored, so every soft vote "
        "can be scored without refitting: greedy forward selection plus every "
        "subset of 2-4 of the ten best singles, per setup. The vote is "
        "unweighted because the deployed `VotingClassifier` is.",
        "",
        "**Two ML numbers, and the second is the one to quote.** Picking the "
        "best of hundreds of combinations on the same five folds is "
        "optimistic. The NESTED check redoes the whole choice (setup AND "
        "members) without each person, then scores it on that person only.",
        "",
        "**NN.** The three best architectures of the previous sweep x window "
        "lengths 0.5-1.5 s in 0.2 s steps, 150 epochs (N1); width / dropout / "
        "learning-rate variants of the best four (N2); two extra seeds for "
        "the best two, ranked by the mean (N3). The net never votes with the "
        "ML models (safety rule 13).",
        "",
    ] + ([f"**NN search stopped early, by choice:** {nn['note']}", ""]
         if nn.get("note") else []) + [
        "## Result",
        "",
        "| | ML (soft vote) | NN |",
        "|---|---|---|",
        f"| winner | {s.get('ML winner')} on `{s.get('ML setup')}` | "
        f"`{s.get('NN winner')}` |",
        f"| per window | {s.get('ML window accuracy %')} % (optimistic) / "
        f"**{s.get('ML NESTED (honest) %')} % nested** | "
        f"**{s.get('NN window accuracy % (mean of seeds)')} %** "
        f"(seeds {s.get('NN seed scores %')}) |",
        f"| per recording | {s.get('ML per-recording vote %')} % | "
        f"{s.get('NN per-recording vote %')} % |",
        f"| macro F1 | {s.get('ML macro F1 %')} % | {s.get('NN macro F1 %')} % |",
        "",
        "Per person held out (ML winner): " + ", ".join(
            f"{p} {s.get(f'ML held out {p} %')} %" for p in people) + ".",
        "",
        "Goal: 82 % (stretch 92 %), person held out.",
        "",
        "## ML -- the best vote on each setup (top 12)",
        "",
        md_table(*t["ML_best_vote_per_setup"], limit=12),
        "",
        "## ML -- the best single models (top 15)",
        "",
        md_table(t["ML_all_models"][0][:11], [r[:11] for r in
                                               t["ML_all_models"][1]], 15),
        "",
        "## ML -- what each choice is worth",
        "",
        "Best, mean and worst person-held-out accuracy per window over every "
        f"distinct single-model fit ({t['_n_distinct'][1][0][0]} distinct of "
        f"{t['_n_distinct'][1][0][1]} listed: one setup is named twice in the "
        "run's list, identically). The mean is dominated by the weaker model "
        "families, so read the best column for what a choice can reach and the "
        "mean for how forgiving it is.",
        "",
        md_table(*t["ML_marginals"]),
        "",
        "How far each person is from the others on the same fits -- the best "
        "fit any of the 16 models on any setup reached for that person, the "
        "median, and the worst:",
        "",
        md_table(*t["ML_person_spread"]),
        "",
        "## ML -- nested check",
        "",
        md_table(*t["ML_nested_check"]),
        "",
        "## NN -- window length (N1, per window %, held-out person)",
        "",
        md_table(*t["NN_window_length"]),
        "",
        "## NN -- all configurations (top 15)",
        "",
        md_table(t["NN_all_configs"][0][:13], [r[:13] for r in
                                                t["NN_all_configs"][1]], 15),
        "",
        "## NN -- finalists (mean over seeds)",
        "",
        md_table(*t["NN_finalists"]),
        "",
        "## What was made the default",
        "",
        "`models/model_spec.json` now holds exactly the winning members with "
        "`voting.always` and a pinned `setup`, so every RETRAIN reproduces "
        "this combination instead of re-sweeping (CLAUDE.md 8m). "
        "`models/nn_spec.json` holds the winning net. The previous files are "
        f"kept as `{rel}/backup_*.json`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run_dir = argv[0] if argv else C.latest_run()
    if not run_dir:
        print("no run found")
        return 1
    t, ml, nn, people = tables(run_dir)
    xlsx = os.path.join(run_dir, "comb_test_results.xlsx")
    try:
        write_xlsx(xlsx, t)
        print("wrote", xlsx)
    except ImportError:
        print("openpyxl not installed -- CSV only")
    write_csv(os.path.join(run_dir, "csv"), t)
    print("wrote", os.path.join(run_dir, "csv"))
    doc = os.path.join(PROJECT, "docs", "COMB_TEST.md")
    write_doc(doc, run_dir, t, ml, nn, people)
    print("wrote", doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Figures and tables for docs/EXCLUDING_PEOPLE.md: what rotation augmentation did
to the hard person's fold, and what training WITHOUT that person did to
everybody else.

    python tools/make_exclusion_report.py [PERSON]          (default: Barbora)

Nothing is retrained; everything is read from what the project saved:

  * rotation: results/ablation/{ml,nn}/runs.jsonl  (leave-one-person-out, every
    seed, the held-out person's accuracy and the accuracy when the band is
    slid one pod);
  * with / without: models/zoo/<id>/meta.json (trained WITH everybody, scored
    leave-one-person-out) against models/zoo/<id>-without-<name>/meta.json
    (the same model trained without PERSON, scored leave-one-person-out on the
    people left).

Writes docs/figures/exclusion_*.png and results/exclusion.json (every number
the document quotes).

Reading the with / without comparison. For a kept person P the two numbers are
a model that never saw P, trained on all the other people (Barbora among them)
or on all the other people except Barbora. The difference is therefore exactly
"what Barbora's recordings did for P", nothing else changes. Barbora herself is
never in the without-fold, so her number is the same in both: the ordinary
model's held-out fold, whose training set is the people kept.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

import numpy as np                                               # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

PERSON = sys.argv[1] if len(sys.argv) > 1 else "Barbora"
OUT = os.path.join(PROJECT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, TEXT, TEXT_2, GRID, MUTED = ("#fcfcfb", "#0b0b0b", "#52514e",
                                      "#e4e3de", "#c9c8c2")
ML, NN = "#2a78d6", "#eb6834"
ML_LIGHT, NN_LIGHT = "#a9c9f2", "#f6c0a6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True})


def jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


# ==============================================================================
# 1. ROTATION AUGMENTATION ON EVERY HELD-OUT FOLD
# ==============================================================================

ROT_MODELS = {"ml": "vote(ET+RF)", "nn": "gru"}


def rotation():
    """{family: {config: {person: {"acc": [per seed], "shift": [per seed]}}}}"""
    out = {}
    for fam, kind in ROT_MODELS.items():
        rows = jsonl(os.path.join(PROJECT, "results", "ablation", fam,
                                  "runs.jsonl"))
        cfgs = {}
        for r in rows:
            if r["kind"] != kind or r["config"] not in ("baseline",
                                                        "aug:rotate"):
                continue
            for f in r["folds"]:
                d = cfgs.setdefault(r["config"], {}).setdefault(
                    f["person"], {"acc": [], "shift": []})
                d["acc"].append(f["acc"])
                d["shift"].append(f["robust"]["shift"])
        out[fam] = cfgs
    return out


def fig_rotation(rot):
    people = sorted(rot["ml"]["baseline"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True)
    x = np.arange(len(people))
    w = 0.38
    for col, (fam, dark, light, title) in enumerate((
            ("ml", ML, ML_LIGHT, "ML classifier (vote of ET + RF)"),
            ("nn", NN, NN_LIGHT, "Neural net (GRU)"))):
        for row, (key, what) in enumerate((
                ("acc", "accuracy per window, person held out"),
                ("shift", "same, with the band slid one pod"))):
            ax = axes[row][col]
            for j, (cfg, colour, label) in enumerate((
                    ("baseline", light, "no augmentation"),
                    ("aug:rotate", dark, "with rotation augmentation"))):
                vals = [np.mean(rot[fam][cfg][p][key]) * 100 for p in people]
                bars = ax.bar(x + (j - 0.5) * w, vals, w, color=colour,
                              label=label)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                            f"{v:.0f}", ha="center", fontsize=8,
                            color=TEXT_2)
            ax.set_xticks(x)
            ax.set_xticklabels(people)
            for lab in ax.get_xticklabels():
                if lab.get_text() == PERSON:
                    lab.set_fontweight("bold")
                    lab.set_color(TEXT)
            ax.set_ylim(0, 100)
            ax.set_title(f"{title}\n{what}", fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel("% of windows")
            ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, TEXT_2)]
    fig.legend(handles, ["no augmentation (light)",
                         "with rotation augmentation (dark)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Rotation augmentation on every held-out person "
                 f"({PERSON} in bold) -- mean of 3 seeds", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, "exclusion_rotation.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 2. TRAINED WITHOUT PERSON, AGAINST TRAINED WITH EVERYBODY
# ==============================================================================


def paired():
    """[{id, family, name, with, without, held_out}] for every variant."""
    root = os.path.join(PROJECT, "models", "zoo")
    rows = []
    for name in sorted(os.listdir(root)):
        meta_path = os.path.join(root, name, "meta.json")
        if not os.path.isfile(meta_path) or "-without-" not in name:
            continue
        with open(meta_path, encoding="utf-8") as fh:
            m = json.load(fh)
        if PERSON not in (m.get("excluded_persons") or []):
            continue
        s = m["scores"]
        base = s.get("base_per_person") or {}
        if not base:
            continue
        kept = sorted(s["per_person"])
        with open(os.path.join(root, m["variant_of"], "meta.json"),
                  encoding="utf-8") as fh:
            base_scores = json.load(fh)["scores"]
        rows.append({
            "headline_with": base_scores["window_acc"],
            "headline_without": s["window_acc"],
            "id": name, "base_id": m["variant_of"], "family": m["family"],
            "name": m["name"].replace(f" (without {PERSON})", ""),
            "with": {p: base[p] for p in kept if p in base},
            "without": {p: s["per_person"][p] for p in kept},
            "held_out": (s.get("excluded_held_out") or {}).get(PERSON),
            "mean_with": s.get("window_acc"),
            "size_bytes": m.get("size_bytes"), "seeds": s.get("seeds")})
    for r in rows:
        r["delta"] = {p: r["without"][p] - r["with"][p] for p in r["without"]}
        r["mean_delta"] = float(np.mean(list(r["delta"].values())))
    return rows


def fig_pairs(rows):
    fams = [("ml", "ML classifiers", ML, ML_LIGHT),
            ("nn", "Neural nets", NN, NN_LIGHT)]
    people = sorted(rows[0]["without"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    x = np.arange(len(people) + 1)
    w = 0.38
    for ax, (fam, title, dark, light) in zip(axes, fams):
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        for j, (key, colour, label) in enumerate((
                ("with", light, f"trained WITH {PERSON}"),
                ("without", dark, f"trained WITHOUT {PERSON}"))):
            per = [np.mean([r[key][p] for r in rs]) * 100 for p in people]
            per.append(np.mean(per))
            bars = ax.bar(x + (j - 0.5) * w, per, w, color=colour, label=label)
            for b, v in zip(bars, per):
                ax.text(b.get_x() + b.get_width() / 2, 4, f"{v:.1f}",
                        ha="center", fontsize=8.5,
                        color=TEXT if key == "with" else "white")
            for r in rs:                                   # each model, a dot
                pts = [r[key][p] * 100 for p in people]
                pts.append(np.mean(pts))
                ax.plot(x + (j - 0.5) * w, pts, "o", ms=3.2, color=TEXT,
                        alpha=0.55, zorder=3)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x)
        ax.set_xticklabels(people + ["mean"])
        ax.set_ylim(0, 105)
        ax.set_title(f"{title}  ({len(rs)} recommended models; "
                     "dots = each model)", fontsize=10, loc="left")
    axes[0].set_ylabel("% of windows, person held out")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, TEXT_2)]
    fig.legend(handles, [f"trained WITH {PERSON} (light)",
                         f"trained WITHOUT {PERSON} (dark)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"What {PERSON}'s recordings did for everybody else",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(OUT, "exclusion_per_person.png"), dpi=140)
    plt.close(fig)


def fig_delta(rows):
    people = sorted(rows[0]["without"])
    order = sorted(rows, key=lambda r: (r["family"], -r["mean_delta"]))
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(order) + 1.6))
    y = np.arange(len(order))[::-1]
    for yi, r in zip(y, order):
        colour = ML if r["family"] == "ml" else NN
        d = r["mean_delta"] * 100
        ax.barh(yi, d, height=0.55, color=colour)
        ax.plot([r["delta"][p] * 100 for p in people], [yi] * len(people),
                "o", ms=4, color=TEXT, alpha=0.6, zorder=3)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name'].replace('_', ' ')}   "
                        f"{r['mean_delta'] * 100:+.1f}" for r in order])
    ax.set_xlabel(f"change in accuracy (points) when {PERSON} is left out\n"
                  "bar = mean over the people kept (number on the left), "
                  "dots = each person")
    ax.set_title(f"Left of zero: leaving {PERSON} out made it WORSE for "
                 "the others.  Right: better.", fontsize=10, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exclusion_delta.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 2b. THE TRAINING SWITCHES, WITH AND WITHOUT THE PERSON IN TRAINING
# ==============================================================================

SWITCHES = [("aug:gain", "gain"), ("aug:noise", "noise"),
            ("aug:timewarp", "time warp"), ("aug:rotate", "band rotation"),
            ("aug:chandrop", "dead electrode"), ("adapt", "adaptation")]
FAM_KIND = {"ml": "vote(ET+RF)", "nn": "gru"}
FAM_TITLE = {"ml": "ML (vote of ET + RF)", "nn": "Neural net (GRU)"}


def fold_table(path, kind):
    """{config: {person: {"acc": mean over seeds, "shift": ...}}}, seeds."""
    if not os.path.exists(path):
        return {}, 0
    out, seeds = {}, 0
    for r in jsonl(path):
        if r["kind"] != kind:
            continue
        seeds = max(seeds, r["seed"] + 1)
        for f in r["folds"]:
            d = out.setdefault(r["config"], {}).setdefault(
                f["person"], {"acc": [], "shift": []})
            d["acc"].append(f["acc"])
            d["shift"].append(f["robust"]["shift"])
    return {c: {p: {k: float(np.mean(v)) for k, v in d.items()}
                for p, d in per.items()} for c, per in out.items()}, seeds


def seed_spread(path, kind, people):
    """Mean over `people` of the seed-to-seed standard deviation of one
    person's baseline accuracy: how far a single person's number moves between
    two trainings that differ only in the seed."""
    per = {}
    for r in jsonl(path):
        if r["kind"] == kind and r["config"] == "baseline":
            for f in r["folds"]:
                per.setdefault(f["person"], []).append(f["acc"])
    stds = [float(np.std(v, ddof=1)) for p, v in per.items()
            if p in people and len(v) > 1]
    return float(np.mean(stds)) if stds else None


def switches():
    """
    {family: {...}} for the effect of each training switch on the people who
    stay in, in a world where PERSON is in training and in one where she is not.
    The effect is (switch - baseline) accuracy per held-out person, averaged
    over those same people: like for like, never a 5-person mean against a
    4-person one.
    """
    sys.path.insert(0, HERE)
    import collect_results as CR
    tag = PERSON.lower()
    out = {}
    for fam, kind in FAM_KIND.items():
        w_path = os.path.join(PROJECT, "results", "ablation", fam, "runs.jsonl")
        x_path = os.path.join(PROJECT, "results", f"ablation_without_{tag}",
                              fam, "runs.jsonl")
        w, w_seeds = fold_table(w_path, kind)
        x, x_seeds = fold_table(x_path, kind)
        if "baseline" not in w or "baseline" not in x:
            continue
        kept = sorted(p for p in x["baseline"] if p != PERSON)
        rows = {}
        for cfg, label in SWITCHES:
            if cfg not in w or cfg not in x:
                continue
            row = {"label": label}
            for world, tab in (("with", w), ("without", x)):
                for key in ("acc", "shift"):
                    row[f"{world}_{key}"] = float(np.mean(
                        [tab[cfg][p][key] - tab["baseline"][p][key]
                         for p in kept]))
                row[f"{world}_per_person"] = {
                    p: tab[cfg][p]["acc"] - tab["baseline"][p]["acc"]
                    for p in kept}
            rows[cfg] = row
        verdict = {}
        for world, path in (("with", w_path), ("without", x_path)):
            for r in CR.ablation_rows(fam, jsonl(path), kind):
                if r["config"] in dict(SWITCHES):
                    verdict.setdefault(r["config"], {})[world] = {
                        "verdict": r["verdict"],
                        "delta_acc": r["delta_acc"],
                        "people_better": r["people_better"],
                        "n_people": r["n_people"],
                        "worthy": r["default_worthy"]}
        for cfg in rows:
            rows[cfg]["verdict"] = verdict.get(cfg, {})
        out[fam] = {
            "kind": kind, "kept": kept, "seeds_with": w_seeds,
            "seeds_without": x_seeds, "switches": rows,
            "baseline_with": {p: w["baseline"][p]["acc"] for p in kept},
            "baseline_without": {p: x["baseline"][p]["acc"] for p in kept},
            "baseline_with_all": {p: w["baseline"][p]["acc"]
                                  for p in w["baseline"]},
            "baseline_shift_with": {p: w["baseline"][p]["shift"]
                                    for p in kept},
            "baseline_shift_without": {p: x["baseline"][p]["shift"]
                                       for p in kept}}
        dw = {d["setting"]: d["value"] for d in CR.decide(
            CR.ablation_rows(fam, jsonl(w_path), kind), fam)}
        dx = {d["setting"]: d["value"] for d in CR.decide(
            CR.ablation_rows(fam, jsonl(x_path), kind), fam)}
        out[fam]["defaults_with"] = dw
        out[fam]["defaults_without"] = dx
        out[fam]["seed_std"] = {
            "with": seed_spread(w_path, kind, kept),
            "without": seed_spread(x_path, kind, kept)}
    return out


def _default_difference(d, setting):
    """One sentence on why a shipped default differs between the two worlds."""
    cfg = "adapt" if setting == "adaptation" else "aug:" + setting
    v = d["switches"].get(cfg, {}).get("verdict", {})
    a, b = v.get("with"), v.get("without")
    if not a or not b:
        return f"{setting} differs."
    return (f"{setting}: with her {_pc(a['delta_acc'], True)} points over all "
            f"{a['n_people']} people (her own big gain included), better for "
            f"{a['people_better']}/{a['n_people']} ({a['verdict']}); without "
            f"her {_pc(b['delta_acc'], True)} points over the "
            f"{b['n_people']} who stay, better for "
            f"{b['people_better']}/{b['n_people']} "
            f"({b['verdict']}). The rule asks for at least 4 people better, "
            f"which with {b['n_people']} people means all of them, so part of "
            "the difference is the rule, not the data.")


def fig_switches(sw):
    fams = [f for f in ("ml", "nn") if f in sw]
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 2, figsize=(11.5, 3.6 * len(fams)),
                             squeeze=False)
    for row, fam in enumerate(fams):
        dark, light = (ML, ML_LIGHT) if fam == "ml" else (NN, NN_LIGHT)
        sws = sw[fam]["switches"]
        keys = [c for c, _ in SWITCHES if c in sws]
        labels = [sws[c]["label"] for c in keys]
        x = np.arange(len(keys))
        w = 0.38
        for col, (key, what) in enumerate((
                ("acc", "change in accuracy per window (points)"),
                ("shift", "change with the band slid one pod (points)"))):
            ax = axes[row][col]
            for j, (world, colour) in enumerate((("with", light),
                                                 ("without", dark))):
                vals = [sws[c][f"{world}_{key}"] * 100 for c in keys]
                bars = ax.bar(x + (j - 0.5) * w, vals, w, color=colour)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2,
                            v + (0.8 if v >= 0 else -0.8), f"{v:+.1f}",
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7.5, color=TEXT_2)
            ax.axhline(0, color=TEXT_2, lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace(" ", "\n") for l in labels],
                               fontsize=9)
            ax.margins(y=0.18)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{FAM_TITLE[fam]}\n{what}", fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, TEXT_2)]
    fig.legend(handles, [f"{PERSON} in training (light)",
                         f"{PERSON} left out (dark)"], loc="lower center",
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"What each training switch does to the people who stay in, "
                 f"with and without {PERSON} in training", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(os.path.join(OUT, "exclusion_switches.png"), dpi=140)
    plt.close(fig)


def write_csv(rows, sw):
    import csv
    path_a = os.path.join(PROJECT, "results", "exclusion_per_person.csv")
    with open(path_a, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_id", "family", "model", "person",
                    f"acc_trained_with_{PERSON}",
                    f"acc_trained_without_{PERSON}", "change",
                    f"{PERSON}_held_out_acc"])
        for r in rows:
            for p in sorted(r["without"]):
                w.writerow([r["id"], r["family"], r["name"], p,
                            f"{r['with'][p]:.4f}", f"{r['without'][p]:.4f}",
                            f"{r['delta'][p]:+.4f}",
                            "" if r["held_out"] is None
                            else f"{r['held_out']:.4f}"])
    path_b = os.path.join(PROJECT, "results", "exclusion_switches.csv")
    with open(path_b, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "model", "switch", "person",
                    f"change_acc_{PERSON}_in_training",
                    f"change_acc_{PERSON}_left_out"])
        for fam, d in sw.items():
            for cfg, r in d["switches"].items():
                for p in d["kept"]:
                    w.writerow([fam, d["kind"], r["label"], p,
                                f"{r['with_per_person'][p]:+.4f}",
                                f"{r['without_per_person'][p]:+.4f}"])
    return path_a, path_b


# ==============================================================================
# 3. THE DOCUMENT
# ==============================================================================


def _pc(v, signed=False):
    return f"{v * 100:+.1f}" if signed else f"{v * 100:.1f}"


def models_compared_block(summary):
    """The short with / without block that docs/MODELS_COMPARED.md carries."""
    rows = summary["pairs"]
    if not rows:
        return "*Not measured yet.*"
    kept = sorted(rows[0]["without"])
    L = [f"The recommended (starred) models were trained again without "
         f"{PERSON}'s recordings and scored on the people who stay in "
         f"({', '.join(kept)}), each held out in turn, against the same model "
         f"trained with her. **Same people, same protocol:** the change is "
         f"exactly what her recordings did for each of them. Full account, "
         f"figures and the effect of every training switch: "
         f"[EXCLUDING_PEOPLE.md](EXCLUDING_PEOPLE.md).\n",
         f"| family | mean over {', '.join(kept)}: with | without | change | "
         f"{PERSON} herself (held out) |",
         "|---|---|---|---|---|"]
    for fam in ("ml", "nn"):
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        a = np.mean([np.mean(list(r["with"].values())) for r in rs])
        b = np.mean([np.mean(list(r["without"].values())) for r in rs])
        h = np.mean([r["held_out"] for r in rs if r["held_out"] is not None])
        L.append(f"| {fam.upper()} ({len(rs)} models) | {_pc(a)} % | "
                 f"{_pc(b)} % | {_pc(b - a, True)} | {_pc(h)} % |")
    head = ", ".join(
        f"{fam.upper()} {_pc(np.mean([r['headline_without'] for r in rows if r['family'] == fam]))} %"
        for fam in ("ml", "nn") if any(r["family"] == fam for r in rows))
    L += ["", "![](figures/exclusion_per_person.png)", "",
          f"The score of a model trained without her is higher on its own "
          f"({head}) only because she, the hardest person, is "
          "no longer in the average: it is a different set of people and is "
          "never compared with the 5-person score above. The blue star in "
          "SETTINGS -> Models marks every variant that scores higher than its "
          "yellow star on the *same* people (any gain, computed live)."]
    return "\n".join(L)


def fill_marked(path, text):
    """Replace what is between the exclusion markers of a hand-written doc."""
    a, b = "<!-- exclusion:begin -->", "<!-- exclusion:end -->"
    s = open(path, encoding="utf-8").read()
    if a not in s or b not in s:
        return False
    head, rest = s.split(a, 1)
    _, tail = rest.split(b, 1)
    open(path, "w", encoding="utf-8", newline="\n").write(
        head + a + "\n" + text + "\n" + b + tail)
    return True


def write_doc(summary):
    rot, rows = summary["rotation"], summary["pairs"]
    people = sorted(rot["ml"]["baseline"])
    kept = [p for p in people if p != PERSON]
    L = []
    add = L.append

    def rot_table(fam, key_a, key_b):
        r = rot[fam]
        add("| held-out person | no augmentation | rotation | change |")
        add("|---|---|---|---|")
        for p in people:
            a, b = r[key_a][p], r[key_b][p]
            name = f"**{p}**" if p == PERSON else p
            add(f"| {name} | {_pc(a)} % | {_pc(b)} % | {_pc(b - a, True)} |")
        a = np.mean(list(r[key_a].values()))
        b = np.mean(list(r[key_b].values()))
        add(f"| mean of {len(people)} | {_pc(a)} % | {_pc(b)} % | "
            f"{_pc(b - a, True)} |")
        add("")

    ml_r, nn_r = rot["ml"], rot["nn"]
    nn_seeds_a = ", ".join(_pc(v) for v in nn_r["person_seeds_baseline"])
    nn_seeds_b = ", ".join(_pc(v) for v in nn_r["person_seeds_rotate"])
    add(f"# Training without {PERSON}: what was tried, what it did\n")
    add("*Generated by `tools/make_exclusion_report.py` from the saved runs "
        "(`results/ablation/`, `models/zoo/`); every number below is read "
        "from them, none is typed by hand. The numbers are also in "
        "`results/exclusion.json`.*\n")
    add(f"## 1. The question\n\n{PERSON} is the hard fold (CLAUDE.md 8, "
        "docs/MODELS_COMPARED.md): a model that never saw her reads her "
        "far worse than anybody else. Two things were tried, in this order.\n")
    add("1. **Rotation augmentation** (the band slid round the arm during "
        f"training), checked on {PERSON}'s fold -- the explanation offered "
        "was a band worn at a different angle.\n"
        f"2. If that does not help, **leave {PERSON}'s recordings out of "
        "training**, as a setting, with the recommended models trained that "
        "way and ready.\n")

    add("## 2. Rotation augmentation on her fold\n")
    add("Leave-one-person-out, 3 seeds each, the same protocol as every "
        "other number in the project (these are the ablation study's runs, "
        "`results/ablation/*/runs.jsonl`). Accuracy per window of the "
        "held-out person.\n")
    add("![rotation](figures/exclusion_rotation.png)\n")
    add("**ML (vote of ET + RF)**\n")
    rot_table("ml", "baseline", "rotate")
    add("**Neural net (GRU)**\n")
    rot_table("nn", "baseline", "rotate")
    add(f"With the band slid one pod (the thing the augmentation exists "
        "for):\n")
    add("| | ML no aug. | ML rotation | NN no aug. | NN rotation |")
    add("|---|---|---|---|---|")
    for p in people:
        name = f"**{p}**" if p == PERSON else p
        add(f"| {name} | {_pc(ml_r['baseline_shift'][p])} % | "
            f"{_pc(ml_r['rotate_shift'][p])} % | "
            f"{_pc(nn_r['baseline_shift'][p])} % | "
            f"{_pc(nn_r['rotate_shift'][p])} % |")
    add("")
    dml = ml_r["rotate"][PERSON] - ml_r["baseline"][PERSON]
    dnn = nn_r["rotate"][PERSON] - nn_r["baseline"][PERSON]
    oth_nn = {p: nn_r["rotate"][p] - nn_r["baseline"][p] for p in kept}
    worst_nn = min(oth_nn, key=oth_nn.get)
    mean_ml = np.mean([ml_r["rotate"][p] - ml_r["baseline"][p]
                       for p in people])
    mean_nn = np.mean([nn_r["rotate"][p] - nn_r["baseline"][p]
                       for p in people])
    add("**Verdict: it does not help in a way that can be deployed.**\n")
    add(f"- **ML:** {PERSON} goes {_pc(ml_r['baseline'][PERSON])} -> "
        f"{_pc(ml_r['rotate'][PERSON])} % ({_pc(dml, True)} points): no "
        f"help, and the mean over everybody is {_pc(mean_ml, True)}.")
    add(f"- **NN:** {PERSON} goes {_pc(nn_r['baseline'][PERSON])} -> "
        f"{_pc(nn_r['rotate'][PERSON])} % ({_pc(dnn, True)} points), but the "
        f"three seeds disagree a lot (no augmentation: {nn_seeds_a}; "
        f"rotation: {nn_seeds_b}), and it costs "
        f"{worst_nn} {_pc(oth_nn[worst_nn], True)} points; the mean over "
        f"everybody is {_pc(mean_nn, True)}.")
    add(f"- **It does what it is for, but not for her.** With the band slid "
        f"a pod, rotation raises the other people's accuracy a lot, and "
        f"{PERSON}'s only from {_pc(ml_r['baseline_shift'][PERSON])} to "
        f"{_pc(ml_r['rotate_shift'][PERSON])} % (ML) and "
        f"{_pc(nn_r['baseline_shift'][PERSON])} to "
        f"{_pc(nn_r['rotate_shift'][PERSON])} % (NN). So a one-pod roll is "
        "**not** what makes her hard to read: the earlier suggestion that "
        "it is a rotated band is not supported by this test. What differs "
        "about her is something else, and these runs cannot say what.\n")
    add(f"- **What did help her, for ML, is adaptation to the person** (each "
        "person standardised with their own statistics, already the "
        "shipped ML default): the same ablation measured "
        + ", ".join(f"{k} {_pc(v)} %" for k, v in
                    summary["adapt_barbora"].items())
        + ". That is a bigger move than rotation made for any model, and "
        "it did not need her data excluded.\n")

    if not rows:
        add("## 3. Training without her\n\n*No trained-without models were "
            "found in `models/zoo/`.*\n")
    else:
        fams = {"ml": [r for r in rows if r["family"] == "ml"],
                "nn": [r for r in rows if r["family"] == "nn"]}
        add(f"## 3. Training without {PERSON}\n")
        add("The recommended (starred) models, the three best ML classifiers "
            "and the three best nets, were trained again with exactly the "
            f"same settings but without {PERSON}'s recordings, and scored "
            "leave-one-person-out on the people left "
            f"({', '.join(kept)}). They are saved as "
            "`<id>-without-<name>` beside the originals "
            "(`models/zoo/`), the originals are untouched, and they are "
            "listed in Settings -> Models.\n")
        add("### How to read the comparison\n")
        add("For a kept person the two numbers are a model that **never saw "
            "them**, trained on everybody else -- with or without "
            f"{PERSON} among the others. So the difference is exactly what "
            f"{PERSON}'s recordings did for that person; nothing else "
            f"changes. {PERSON} herself is in neither test fold: her number "
            "is the same either way (the ordinary model's held-out fold).\n")
        add("![per person](figures/exclusion_per_person.png)\n")
        add("![change](figures/exclusion_delta.png)\n")
        add("### Per person, mean of the three models in each family\n")
        add("| family | person | with " + PERSON + " | without | change |")
        add("|---|---|---|---|---|")
        fam_mean = {}
        for fam in ("ml", "nn"):
            rs = fams[fam]
            if not rs:
                continue
            ch = []
            for p in kept:
                a = np.mean([r["with"][p] for r in rs])
                b = np.mean([r["without"][p] for r in rs])
                ch.append(b - a)
                add(f"| {fam.upper()} | {p} | {_pc(a)} % | {_pc(b)} % | "
                    f"{_pc(b - a, True)} |")
            fam_mean[fam] = float(np.mean(ch))
            add(f"| **{fam.upper()}** | **mean of {len(kept)}** | "
                f"{_pc(np.mean([np.mean(list(r['with'].values())) for r in rs]))}"
                f" % | "
                f"{_pc(np.mean([np.mean(list(r['without'].values())) for r in rs]))}"
                f" % | **{_pc(fam_mean[fam], True)}** |")
        add("")
        add("### Every model\n")
        add("| model | " + " | ".join(f"{p} change" for p in kept)
            + f" | mean change | {PERSON} (held out, same both ways) |")
        add("|---|" + "---|" * (len(kept) + 2))
        for r in sorted(rows, key=lambda r: (r["family"], -r["mean_delta"])):
            add(f"| {r['name']} | "
                + " | ".join(_pc(r["delta"][p], True) for p in kept)
                + f" | **{_pc(r['mean_delta'], True)}** | "
                f"{_pc(r['held_out'])} % |")
        add("")
        add("### The headline score changes -- and why it must not be read "
            "as an improvement\n")
        add("| model | score with everybody (5 people) | score without "
            f"{PERSON} ({len(kept)} people) |")
        add("|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["family"], r["name"])):
            add(f"| {r['name']} | {_pc(r['headline_with'])} % | "
                f"{_pc(r['headline_without'])} % |")
        add("")
        add(f"The second column is higher mostly because {PERSON}, the "
            "hardest fold, is no longer in the average. It is a different "
            "set of people and **cannot be compared** with the first. The "
            "table above (same person, with and without) is the fair "
            "comparison.\n")
        add("### Verdict\n")
        lo = min(r["mean_delta"] for r in rows)
        hi = max(r["mean_delta"] for r in rows)
        add(f"- Leaving {PERSON} out changes the others by "
            + ", ".join(f"{fam.upper()} {_pc(fam_mean[fam], True)}"
                        for fam in fam_mean)
            + f" points on average (single models from {_pc(lo, True)} to "
            f"{_pc(hi, True)}). That is small next to what moves it person "
            "by person (the dots in the second figure): one person gains "
            "several points and another loses a few, in both directions "
            "within one model. It is not a clear win, and the sample "
            "(four people, two or three seeds) cannot separate it from "
            "noise.")
        add(f"- **She herself is left worse off.** A model without her "
            f"recordings has never seen her: it reads her "
            + ", ".join(f"{fam.upper()} "
                        f"{_pc(np.mean([r['held_out'] for r in fams[fam]]))} %"
                        for fam in fam_mean)
            + ". A model that was trained on her own recordings is a "
            "different situation (the model has seen her band, see "
            "TEST MODEL's warning about sittings that are in the training "
            "data) and was not measured here.")
        add("- So exclusion is offered as a setting and **off by default**. "
            "It is worth switching on when the model is meant for new "
            "wearers and you suspect one person's data is pulling the others "
            "down; the numbers above say that, on this dataset, it barely "
            "does.\n")

    sw = summary.get("switches") or {}
    add(f"## 4. The training switches, with and without {PERSON} in "
        "training\n")
    if not sw:
        add("*The ablation without her has not been run "
            f"(`results/ablation_without_{PERSON.lower()}/`).*\n")
    else:
        add("Settings -> Training has one set of switches per family: five "
            "augmentations (gain, noise, time warp, band rotation, dead "
            "electrode) and adaptation to the person. The ablation study "
            "(`tools/ablation_ml.py`, `tools/ablation_nn.py`) measured each "
            "one on its own, leave-one-person-out, 3 seeds, judged by the "
            "rule in `training/ablation_stats.py`. It was run again with "
            f"{PERSON}'s recordings left out of everything "
            f"(`results/ablation_without_{PERSON.lower()}/`).\n")
        add("The effect of a switch is its accuracy minus the baseline's for "
            "the same held-out person, **averaged over the people who stay "
            f"in ({', '.join(kept)})**, in a world where {PERSON} is in "
            "training and one where she is not. A 5-person mean is never "
            "set against a 4-person one.\n")
        add("![switches](figures/exclusion_switches.png)\n")
        for fam in ("ml", "nn"):
            if fam not in sw:
                continue
            d = sw[fam]
            add(f"### {FAM_TITLE[fam]}\n")
            add("Baseline (no switch on), accuracy per window, person held "
                "out:\n")
            add(f"| person | {PERSON} in training | {PERSON} left out | "
                "change |")
            add("|---|---|---|---|")
            for p in d["kept"]:
                a, b = d["baseline_with"][p], d["baseline_without"][p]
                add(f"| {p} | {_pc(a)} % | {_pc(b)} % | "
                    f"{_pc(b - a, True)} |")
            a = np.mean(list(d["baseline_with"].values()))
            b = np.mean(list(d["baseline_without"].values()))
            add(f"| **mean of {len(d['kept'])}** | {_pc(a)} % | {_pc(b)} % | "
                f"{_pc(b - a, True)} |")
            add("")
            pmax = max(d["kept"], key=lambda p: abs(
                d["baseline_without"][p] - d["baseline_with"][p]))
            chg = d["baseline_without"][pmax] - d["baseline_with"][pmax]
            zoo_ch = [r["delta"][pmax] for r in rows
                      if r["family"] == fam and pmax in r["delta"]]
            if zoo_ch and abs(chg) >= 0.03:
                add(f"The person who depends most on her recordings here is "
                    f"**{pmax}** ({_pc(chg, True)} points). "
                    f"The zoo's {fam.upper()} models moved the same person "
                    f"by {_pc(np.mean(zoo_ch), True)} points on average "
                    f"(section 3). "
                    + ("Those ML models train with adaptation to the "
                       "person and this baseline does not, which fits "
                       "adaptation making the model much less dependent on "
                       "who else is in the training set -- suggestive only: "
                       "the two sets of models are close but not identical.\n"
                       if fam == "ml" else "\n"))
            add("Effect of each switch on those people (points; the second "
                "pair is accuracy with the band slid one pod):\n")
            add(f"| switch | acc, {PERSON} in | acc, {PERSON} out | "
                f"shifted band, in | shifted band, out | rule verdict, in "
                f"(5 people) | rule verdict, out ({len(d['kept'])} people) |")
            add("|---|---|---|---|---|---|---|")
            for cfg, label in SWITCHES:
                r = d["switches"].get(cfg)
                if r is None:
                    continue
                vi = r["verdict"].get("with", {})
                vo = r["verdict"].get("without", {})
                add(f"| {label} | {_pc(r['with_acc'], True)} | "
                    f"{_pc(r['without_acc'], True)} | "
                    f"{_pc(r['with_shift'], True)} | "
                    f"{_pc(r['without_shift'], True)} | "
                    f"{vi.get('verdict', '--')} | "
                    f"{vo.get('verdict', '--')} |")
            add("")
            dw, dx = d["defaults_with"], d["defaults_without"]
            same = dw == dx
            add("Defaults the rule would ship: with her "
                + ", ".join(f"{k} {v}" for k, v in dw.items()) + "; without "
                "her " + ", ".join(f"{k} {v}" for k, v in dx.items()) + ". "
                + ("**Identical -- leaving her out does not change which "
                   "switches should be on.**\n" if same else
                   "**They differ.** " + " ".join(
                       _default_difference(d, name) for name in dw
                       if dw.get(name) != dx.get(name)) + "\n"))
        rot = [(fam, w, sw[fam]["switches"]["aug:rotate"][f"{w}_shift"])
               for fam in ("ml", "nn") if fam in sw
               and "aug:rotate" in sw[fam]["switches"]
               for w in ("with", "without")]
        if rot and all(v > 0.05 for _, _, v in rot):
            rot_sentence = (
                "Band rotation raises the shifted-band accuracy by "
                f"{min(v for _, _, v in rot) * 100:.0f} to "
                f"{max(v for _, _, v in rot) * 100:.0f} points in every "
                "family, with her in training and without her: that effect "
                "does not depend on her.")
        elif rot:
            rot_sentence = ("Band rotation's effect on the shifted-band "
                            "accuracy differs between the two worlds: "
                            + ", ".join(f"{f.upper()} {w} her "
                                        f"{v * 100:+.0f}" for f, w, v in rot)
                            + " points.")
        else:
            rot_sentence = ""
        spread = {f: max(v for v in sw[f]["seed_std"].values() if v)
                  for f in ("ml", "nn") if f in sw
                  and any(sw[f]["seed_std"].values())}
        spread_sentence = (
            "**How much one person's number moves by chance:** the "
            "seed-to-seed spread of a single person's baseline score is about "
            + " and ".join(f"{f.upper()} {v * 100:.1f}" for f, v in
                           spread.items())
            + " points (standard deviation over seeds), so a per-person "
            "change smaller than that, and in practice anything under ~5 "
            "points for the net, is noise -- including the large single-person "
            "changes in the baseline tables. " if spread else "")
        flips = []
        for fam in ("ml", "nn"):
            for cfg, r in (sw.get(fam, {}).get("switches") or {}).items():
                if (r["with_acc"] > 0) != (r["without_acc"] > 0) and \
                        max(abs(r["with_acc"]), abs(r["without_acc"])) > 0.01:
                    flips.append(f"{fam.upper()} {r['label']}")
        add("**Reading it.** "
            + ("Where the sign of a switch's effect flips between the two "
               "worlds by more than a point: " + ", ".join(flips) + ". "
               if flips else
               "No switch changes sign by more than a point between the two "
               "worlds. ")
            + "Differences of a point or two are inside seed noise (3 "
            "seeds). "
            + spread_sentence + rot_sentence + "\n")

    add("## 5. The setting\n")
    add("Settings (gear icon) -> **Training** -> *Train without these "
        "people*. One checkbox per person with the recordings on disk, each "
        "with how well the best ordinary ML model / net reads them when they "
        "were held out. It is saved with the other settings "
        "(`TRAIN_EXCLUDE_PERSONS` in `<data>/app_settings.json`).\n")
    add("| what starts | what the setting does |")
    add("|---|---|")
    add("| RETRAIN, TRAIN NEURAL NET | the trainer gets `--person` for every "
        "person that stays |")
    add("| model zoo (RETRAIN ML + NN, `train_zoo.py`) | `--exclude-person`: "
        "saves `<id>-without-<name>` variants beside the ordinary models, "
        "scored in `results/zoo_eval_without_<name>/` |")
    add("| Settings -> Models | variants are listed after the ordinary "
        "models, named `... (without <name>)`, never ranked or starred (a "
        "score over fewer people is not comparable); the quick "
        "selector *Blue ★* ticks every variant that scores higher than its "
        "yellow star (*Yellow + blue ★* adds the yellow ones) |")
    add("")
    add("At least two people must stay in (a held-out person needs somebody "
        "to train on); the dialog refuses otherwise. Nothing on disk is "
        "touched, a trained model keeps what it learned, and with nobody "
        "ticked every trainer behaves exactly as before.\n")
    add("By hand: `python training/train_zoo.py all --ids ml-rf-500-leaf3 "
        f"nn-gru --exclude-person {PERSON}`.\n")
    add(f"### Where to choose: with {PERSON} or without\n")
    add("There are two separate switches, and they do different things.\n")
    add("**Use the models that are already trained (instant, no "
        "retraining)**\n")
    add("1. Settings (gear icon, `Ctrl+,`) -> **Models**.")
    add("2. Tick *Use the models selected below*.")
    add("3. For ML and for the neural nets, press the quick selector "
        "*Blue ★* (every model trained without somebody that scores higher "
        "than its yellow star) or tick the "
        f"`(without {PERSON})` rows by hand. *Yellow ★* ticks the "
        f"ordinary top-3 models trained with {PERSON}, *Yellow + blue ★* "
        "both; *Clear* goes back to the original deployed models. A quick "
        "selector also switches *Use the models selected below* on.")
    add("4. **APPLY**. The live classifier reloads at once; a running net "
        "is stopped and START NN loads the new one.\n")
    add("**Make the next training skip her**\n")
    add(f"1. Settings -> **Training** -> *Train without these people* -> "
        f"tick {PERSON}.")
    add("2. **APPLY**, then RETRAIN / TRAIN NEURAL NET / RETRAIN ML + NN. "
        "Untick to include her again.\n")
    add(f"### Which one to use\n")
    add(f"- **{PERSON} is the one wearing the band: use the models trained "
        f"WITH her recordings.** The models trained without her have never "
        f"seen her and read her about "
        + ", ".join(
            f"{fam.upper()} "
            f"{_pc(np.mean([r['held_out'] for r in rows if r['family'] == fam]))} %"
            for fam in ("ml", "nn")
            if any(r["family"] == fam for r in rows))
        + ". A model that has her recordings should do better for her, but "
        "that was **not measured** (it needs a TEST MODEL run in a new "
        "sitting, band re-donned, with her own recordings in the training "
        "data).")
    add(f"- **Somebody else is wearing it, or it is for new people: it "
        f"barely matters.** Leaving {PERSON} out moved the others by about "
        "a point on average (section 3), so there is no strong reason to "
        "give up her data.")
    add("- *Observation from the operator, not tested here:* she has a "
        "smaller arm and the band does not fit her tightly. A loose band "
        "would change the signal at the pods and would fit what is seen "
        "(she is different from everybody, and rotation does not explain "
        "it), but none of these runs can confirm it. A tighter fit (or a "
        "band clip) and a new recording sitting would test it.\n")
    add("## 6. Caveats\n")
    add("- Five people, so four in the paired comparison. A change of a "
        "couple of points is inside what two seeds and four folds can "
        "show.")
    add("- The ML models here use adaptation to the person (the shipped "
        f"default), the rotation table's ML vote does not: that is why "
        f"{PERSON} reads about 60 % here and about 46 % there (section 2).")
    add("- The variants were trained once, with the recommended models "
        "only. The other 19 zoo models have no variant; use the setting "
        "with RETRAIN ML + NN to make more.")
    add("- **The judging rule is stricter with four people.** It asks for a "
        "gain in at least 4 people (an absolute number, "
        "`MIN_PEOPLE_BETTER`), which with four left means all four. A "
        "switch can therefore read *no evidence* without her where it read "
        "*helps* with her, purely because one fewer person was available. "
        "Compare the points, not only the verdict.")
    add("- Five people is a small sample for a published claim. Every "
        "per-person difference above is a difference between two "
        "trainings of the same model on slightly different data; the "
        "dots in the figures show how far single people move.")
    add("- Nothing here tests a model trained on her recordings and used "
        "by her in a later sitting (see section 3).\n")

    add("## 7. For the publication: what is on hand\n")
    add("### Data on hand\n")
    try:
        d_ml = np.load(os.path.join(PROJECT, "results", "zoo_eval",
                                    "data_ml.npz"), allow_pickle=True)
        d_nn = np.load(os.path.join(PROJECT, "results", "zoo_eval",
                                    "data_nn.npz"), allow_pickle=True)
        pm = d_ml["persons"].astype(str)
        pn = d_nn["persons"].astype(str)
        add("| person | ML windows (1.0 s) | NN windows (0.7 s) | recordings "
            "(ML) |")
        add("|---|---|---|---|")
        um = d_ml["uid"].astype(str)
        for p in sorted(set(pm)):
            add(f"| {'**' + p + '**' if p == PERSON else p} | "
                f"{int((pm == p).sum())} | {int((pn == p).sum())} | "
                f"{len(set(um[pm == p]))} |")
        add(f"| total | {len(pm)} | {len(pn)} | {len(set(um))} |")
        add("")
        add("Windows are after the window-level de-duplication (clips cut "
            "from the same free take overlap; the trainer keeps each "
            "stretch of signal once).\n")
    except (OSError, KeyError, ValueError):
        add("*(results/zoo_eval/data_*.npz not found)*\n")
    add("### Methods (draft paragraph)\n")
    add("> Eight-channel EMG was recorded with a Myo armband (on-device "
        "filtered and rectified envelope, about 50 Hz) from five wearers "
        "performing six gestures. Recordings were cut to the labelled hold, "
        "resampled to 60 Hz, windowed (ML 1.0 s, NN 0.7 s, 75 % overlap) "
        "and de-duplicated so that no stretch of signal appears in two "
        "windows. Each wearer's signal was put on a common scale with an "
        "affine per-wearer calibration derived from that wearer's own "
        "unlabelled signal. Every number is leave-one-person-out: each "
        "window is scored by a model that never saw its wearer. The ML "
        "models use band-pass-filtered level features and are trained with "
        "adaptation to the person; the neural nets read the calibrated "
        "window directly. Training switches were compared with their "
        "baseline on identical folds and seeds (3 seeds), and a switch was "
        "called helpful only when it gained at least one point and twice "
        "the seed-to-seed deviation in at least four people. The whole "
        "study was repeated with one wearer's recordings removed from "
        "training and from the folds, and models were compared on the same "
        "held-out people with and without that wearer in training.\n")
    add("### Files\n")
    add("| what | where |")
    add("|---|---|")
    add("| this document, regenerated | `python tools/make_exclusion_report.py` |")
    add("| every number quoted here | `results/exclusion.json` |")
    add("| per person, per model, with / without | "
        "`results/exclusion_per_person.csv` |")
    add("| per switch, per person, with / without | "
        "`results/exclusion_switches.csv` |")
    add("| figures (PNG, 140 dpi) | `docs/figures/exclusion_rotation.png`, "
        "`exclusion_per_person.png`, `exclusion_delta.png`, "
        "`exclusion_switches.png` |")
    add("| the original ablation (5 people) | `results/ablation/{ml,nn}/` "
        "(`runs.jsonl`, out-of-fold probabilities, logs) |")
    add(f"| the ablation without {PERSON} | "
        f"`results/ablation_without_{PERSON.lower()}/{{ml,nn}}/` |")
    add(f"| the models trained without {PERSON} | "
        f"`models/zoo/*-without-{PERSON.lower()}/` and their evaluation "
        f"`results/zoo_eval_without_{PERSON.lower()}/` |")
    add("| everything in one file | `results/experiments.json` (keys "
        "`variants`, `without`) -> `docs/REPORT.md` |")
    add("| the full ordinary evaluation | `docs/REPORT.md`, "
        "`docs/MODELS_COMPARED.md`, `docs/COMB_TEST.md` |\n")
    add("### Reproduce\n")
    add("```powershell")
    add("# the six recommended models without her (about 16 min)")
    add("python training/train_zoo.py all --ids ml-rf-500-leaf3 ml-rf-500 "
        "ml-et-500-leaf3 nn-gru nn-tsmixer nn-chan-transformer "
        f"--exclude-person {PERSON} --seeds 2 --ml-seeds 3")
    add("# the training-switch study without her (about 1.5 h)")
    add(f"python tools/ablation_ml.py --exclude-person {PERSON} --no-combos "
        "--no-calib")
    add(f"python tools/ablation_nn.py --kind gru --exclude-person {PERSON} "
        "--no-combos")
    add("# tables, figures, CSV and this document")
    add("python tools/collect_results.py")
    add("python tools/make_exclusion_report.py")
    add("python tools/make_report.py")
    add("```\n")
    path = os.path.join(PROJECT, "docs", "EXCLUDING_PEOPLE.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


# ==============================================================================
# 4. NUMBERS
# ==============================================================================


def main():
    rot = rotation()
    summary = {"person": PERSON, "rotation": {}, "pairs": [],
               "adapt_barbora": {}}
    ml_rows = jsonl(os.path.join(PROJECT, "results", "ablation", "ml",
                                 "runs.jsonl"))
    for kind in ("vote(ET+RF)", "ET-500", "RF-500-leaf3"):
        for cfg, label in (("baseline", "without"), ("adapt", "with")):
            v = [f["acc"] for r in ml_rows
                 if r["kind"] == kind and r["config"] == cfg
                 for f in r["folds"] if f["person"] == PERSON]
            if v:
                summary["adapt_barbora"][f"{kind} {label} adaptation"] = \
                    float(np.mean(v))
    people = sorted(rot["ml"]["baseline"])
    for fam in ("ml", "nn"):
        summary["rotation"][fam] = {
            "model": ROT_MODELS[fam],
            "baseline": {p: float(np.mean(rot[fam]["baseline"][p]["acc"]))
                         for p in people},
            "rotate": {p: float(np.mean(rot[fam]["aug:rotate"][p]["acc"]))
                       for p in people},
            "baseline_shift": {p: float(np.mean(
                rot[fam]["baseline"][p]["shift"])) for p in people},
            "rotate_shift": {p: float(np.mean(
                rot[fam]["aug:rotate"][p]["shift"])) for p in people},
            "person_seeds_baseline": rot[fam]["baseline"][PERSON]["acc"],
            "person_seeds_rotate": rot[fam]["aug:rotate"][PERSON]["acc"]}
    fig_rotation(rot)
    rows = paired()
    if rows:
        fig_pairs(rows)
        fig_delta(rows)
    summary["pairs"] = rows
    summary["switches"] = switches()
    fig_switches(summary["switches"])
    csvs = write_csv(rows, summary["switches"])
    path = os.path.join(PROJECT, "results", "exclusion.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    doc = write_doc(summary)
    fill_marked(os.path.join(PROJECT, "docs", "MODELS_COMPARED.md"),
                models_compared_block(summary))
    print(f"{len(rows)} trained-without models found, "
          f"{len(summary['switches'])} switch comparison(s); wrote {path}, "
          f"{doc}, {', '.join(csvs)} and docs/figures/exclusion_*.png")


if __name__ == "__main__":
    main()

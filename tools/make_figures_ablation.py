"""
Figures for docs/REPORT.md: the ablation study, voting, the zoo, the tests.

    python tools/make_figures_ablation.py [OUT_DIR]        (default: docs/figures)

Reads results/experiments.json and the raw runs (results/ablation/*/runs.jsonl);
trains nothing. Same palette as tools/make_figures.py: the ML family is blue, the
neural family orange, everything else neutral; verdicts are written out, never
carried by colour alone.

  ablation_effects.png   per switch: change in accuracy against the baseline,
                         each dot one held-out person, the bar the mean, the grey
                         band the seed-to-seed noise it had to clear
  ablation_robustness.png  accuracy under noise / a band shifted one pod / a
                         weaker signal, baseline against every switch
  voting_vs_members.png  each ML member alone against the soft votes
  zoo_overview.png       every model: accuracy against size on disk, stars marked
  tests_overview.png     the automatic suites: checks passed / total
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
import ablation_stats as ST                                       # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, TEXT, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, MUTED = "#e4e3de", "#c9c8c2"
ML, NN, GOOD, BAD = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
FAMILY = {"ml": ML, "nn": NN}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 170})

DOC = json.load(open(os.path.join(ROOT, "results", "experiments.json"),
                     encoding="utf-8"))


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def groups(runs, kind):
    g = {}
    for r in runs:
        if r["kind"] == kind:
            g.setdefault(r["config"], []).append(r)
    return g


NN_RUNS = read_jsonl(os.path.join(ROOT, "results", "ablation", "nn", "runs.jsonl"))
ML_RUNS = read_jsonl(os.path.join(ROOT, "results", "ablation", "ml", "runs.jsonl"))
PANELS = []
if NN_RUNS:
    PANELS.append(("Neural net (GRU)", "nn", groups(NN_RUNS, "gru")))
if ML_RUNS:
    PANELS.append(("ML (soft vote ET-500 + RF-500-leaf3)", "ml",
                   groups(ML_RUNS, "vote(ET+RF)")))


def short(name):
    return (name.replace("aug:", "aug ").replace("|", " + ")
            .replace("calib:", "calibration "))


# ---- 1. effects ------------------------------------------------------------------
if PANELS:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 5.2 * len(PANELS)))
    axes = np.atleast_1d(axes)
    for ax, (title, fam, g) in zip(axes, PANELS):
        base = g.get("baseline")
        if not base:
            continue
        rows = [n for n in g if n != "baseline"]
        rows.sort(key=lambda n: -ST.judge(base, g[n])["delta_acc"])
        for y, name in enumerate(rows[::-1]):
            j = ST.judge(base, g[name])
            people, d = ST.paired(base, g[name])
            ax.barh(y, j["delta_acc"] * 100, color=FAMILY[fam], alpha=0.85,
                    height=0.55, zorder=2)
            ax.plot(d * 100, [y] * len(d), "o", color=TEXT, ms=4, alpha=0.6,
                    zorder=3)
            verdict = j["verdict"]
            ax.text(1.01, y, f"{j['delta_acc'] * 100:+.1f} pts   "
                    f"{j['people_better']}/{j['n_people']} people   {verdict}",
                    transform=ax.get_yaxis_transform(), va="center",
                    fontsize=9, color=GOOD if j["default_worthy"] else TEXT_2,
                    fontweight="bold" if j["default_worthy"] else None)
        ax.axvline(0, color=TEXT_2, lw=1)
        #  The bar a switch has to clear differs per row: twice the LARGER of
        #  the baseline's and its own seed spread (at least one point). Drawn as
        #  two grey ticks per row, the judge's own number.
        for y, name in enumerate(rows[::-1]):
            need = max(ST.MIN_GAIN, 2 * max(ST.seed_std(base),
                                            ST.seed_std(g[name]))) * 100
            ax.plot([-need, need], [y, y], "|", color=TEXT_2, ms=16, mew=2,
                    zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([short(n) for n in rows[::-1]])
        ax.set_xlabel("change in accuracy per window against the baseline, points\n"
                      "dots: each held-out person    ticks: the gain to clear "
                      "(2 x the larger seed spread, at least 1)")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        b = ST.summarise(base)
        ax.set_title(f"{title}   baseline {b['acc_mean'] * 100:.1f} % "
                     f"± {b['acc_std'] * 100:.1f} over {b['seeds']} seeds")
    fig.subplots_adjust(right=0.62, hspace=0.45)
    save(fig, "ablation_effects.png")

# ---- 2. robustness ------------------------------------------------------------------
if PANELS:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 5.0 * len(PANELS)),
                             squeeze=False)
    kinds = [("robust_noise", "noise"), ("robust_shift", "band shifted\nby one pod"),
             ("robust_gain", "25 % weaker")]
    #  One hue per switch, so every bar can be matched to the legend. The
    #  baseline keeps its family colour (blue ML / orange NN) and is the only
    #  bar drawn with a dark outline. Every configuration is shown -- an earlier
    #  version cut the list at six and silently dropped "adapt" and the
    #  calibration modes.
    SWITCH = {"aug:gain": "#8e6bbf", "aug:noise": "#8a8a85", "aug:timewarp": "#2aa5a0",
              "aug:rotate": "#d9a21b", "aug:chandrop": "#b5527d", "adapt": GOOD,
              "calib:off": "#a37b5c", "calib:gain": "#6f9ed8",
              "calib:per_channel": "#c46a6a"}
    for ax, (title, fam, g) in zip(axes[:, 0], PANELS):
        names = ["baseline"] + sorted(n for n in g if n != "baseline")
        w = 0.88 / len(names)
        for i, name in enumerate(names):
            s = ST.summarise(g[name])
            vals = [(s.get(k) or 0) * 100 for k, _ in kinds]
            base = name == "baseline"
            x = np.arange(3) + i * w
            ax.bar(x, vals, w * 0.9,
                   color=FAMILY[fam] if base else SWITCH.get(name, MUTED),
                   edgecolor=TEXT if base else SURFACE, linewidth=1.2 if base else 0.8,
                   label=short(name), zorder=2)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=TEXT_2)
        ax.set_xticks(np.arange(3) + w * (len(names) - 1) / 2)
        ax.set_xticklabels([lab for _, lab in kinds])
        ax.set_ylabel("accuracy on the perturbed\nheld-out person, %")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8.5, ncol=min(len(names), 5), loc="upper center",
                  bbox_to_anchor=(0.5, -0.17))
        ax.set_title(title)
    fig.subplots_adjust(hspace=0.55)
    save(fig, "ablation_robustness.png")

# ---- 3. voting --------------------------------------------------------------------------
if DOC.get("voting"):
    rows = sorted(DOC["voting"], key=lambda r: -r["acc_mean"])
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6))
    for i, r in enumerate(rows[::-1]):
        is_vote = r["model"].startswith("vote")
        ax.barh(i, r["acc_mean"] * 100, color=ML if is_vote else MUTED,
                height=0.6, zorder=2)
        ax.errorbar(r["acc_mean"] * 100, i, xerr=r["acc_std"] * 100,
                    color=TEXT_2, capsize=3, zorder=3)
        ax.text(r["acc_mean"] * 100 + r["acc_std"] * 100 + 1, i,
                f"{r['acc_mean'] * 100:.1f} %", va="center", fontsize=9.5,
                fontweight="bold" if is_vote else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows[::-1]])
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("accuracy per window, person held out (mean ± seed spread)")
    ax.set_title("Soft voting against its members (ML, baseline setup)")
    save(fig, "voting_vs_members.png")

# ---- 4. the zoo ------------------------------------------------------------------------------
models = [m for m in DOC.get("models", []) if m.get("window_acc") is not None]
if models:
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.scatter(max(m["size_bytes"] or 1, 1) / 1024, m["window_acc"] * 100,
                   s=70 if m["starred"] else 40, color=FAMILY[m["family"]],
                   edgecolor=TEXT if m["starred"] else SURFACE, lw=1.4,
                   zorder=3)
        ax.annotate(("★ " if m["starred"] else "") + m["name"],
                    (max(m["size_bytes"] or 1, 1) / 1024, m["window_acc"] * 100),
                    xytext=(5, 4), textcoords="offset points", fontsize=8,
                    color=TEXT if m["starred"] else TEXT_2)
    ax.set_xscale("log")
    ax.set_xlabel("size on disk, KB (log scale)")
    ax.set_ylabel("accuracy per window, person held out, %")
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.scatter([], [], color=ML, label="ML classifier")
    ax.scatter([], [], color=NN, label="neural net")
    ax.legend(loc="lower right")
    ax.set_title("The model zoo: accuracy against size  (★ = starred)")
    save(fig, "zoo_overview.png")

# ---- 5. the tests ---------------------------------------------------------------------------------
tests = (DOC.get("tests") or {}).get("results")
if tests:
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(tests) + 1.6))
    for i, t in enumerate(tests[::-1]):
        ax.barh(i, t["total"], color=MUTED, height=0.6, zorder=1)
        ax.barh(i, t["passed"], color=GOOD if t["ok"] else BAD, height=0.6,
                zorder=2)
        ax.text(t["total"] + 4, i, f"{t['passed']}/{t['total']}"
                + ("" if t["ok"] else "   FAILED"), va="center", fontsize=9,
                color=TEXT if t["ok"] else BAD)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels([t["suite"] for t in tests[::-1]])
    ax.set_xlabel("checks")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    done = sum(t["passed"] for t in tests)
    ax.set_title(f"Automatic tests: {done}/{sum(t['total'] for t in tests)} "
                 "checks passed")
    save(fig, "tests_overview.png")

"""
One figure for the newer-models experiment (tools/new_models_test.py):

    python tools/make_figure_new_models.py [RUN_DIR] [OUT_PNG]
        RUN_DIR  default: the newest models/reports/new_models_test_* folder
        OUT_PNG  default: docs/figures/novesie_modely_test.png

Reads only what the run saved (results.jsonl + the out-of-fold probabilities),
plus -- as the reference row -- the comb test's stored out-of-fold probabilities
for the deployed ML vote (ET-500 + RF-500-leaf3). Trains nothing.

The ML reference was scored with the same rule (each person held out, mean over
people) but on 1.0 s windows, not 0.7 s, so its windows are not the NN
windows. The caption says so.
"""
import glob
import json
import os
import sys
import textwrap
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors                                        # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap            # noqa: E402
from matplotlib.gridspec import GridSpec                         # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

RUNS = sorted(glob.glob(os.path.join(PROJECT, "models", "reports",
                                     "new_models_test_*")))
RUN = sys.argv[1] if len(sys.argv) > 1 else (RUNS[-1] if RUNS else None)
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    PROJECT, "docs", "figures", "novesie_modely_test.png")
if not RUN or not os.path.isdir(RUN):
    raise SystemExit("No run folder. Run tools/new_models_test.py first.")
COMB = os.path.join(PROJECT, "models", "comb_test", "20261022-003918",
                    "ml_oof", "bandpass-level-w1-o0.75__")

# ---- palette (same family as make_figures.py) --------------------------------
SURFACE, TEXT, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, MUTED = "#e4e3de", "#c9c8c2"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
#  One hue per model, no two close: the two greens (ROCKET / TSMixer) and the
#  two warm ones (GRU orange / InceptionTime amber) used to be hard to tell
#  apart. ML stays blue and the GRU orange, as in every other figure.
COLOR = {"ml": "#2a78d6", "gru": "#eb6834", "circ_gru": "#6250d6",
         "chan_transformer": "#e87ba4", "inception": "#17a2b8",
         "tsmixer": "#8a6d3b", "rocket": "#1baf7a"}


def darker(c, k=0.72):
    """Text version of a model colour: the light ones (pink, cyan) are fine as
    bars but unreadable as coloured text on the cream background."""
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r * k, g * k, b * k)


def model_cmap(m):
    """White -> the model's own colour -> a dark shade of it, so a model's
    confusion matrix is drawn in the model's colour (ML blue, GRU orange)."""
    return LinearSegmentedColormap.from_list(
        m, ["#fcfcfb", COLOR[m], darker(COLOR[m], 0.45)])


def ink(cmap, value, vmin=0.0, vmax=100.0):
    r, g, b, _a = cmap((value - vmin) / (vmax - vmin))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    return "white" if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.18         else TEXT
REF = "ml"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 170,
})


# ---- load ---------------------------------------------------------------------
meta = json.load(open(os.path.join(RUN, "run.json"), encoding="utf-8"))
TITLES = dict(meta["titles"])
TITLES[REF] = "ML vote ET+RF (combination-test winner)"
LABELS = meta["labels"]
SHORT = {"WAVE_LEFT": "WAVE_L", "WAVE_RIGHT": "WAVE_R", "THUMB_UP": "THUMB"}
data = np.load(os.path.join(RUN, "data.npz"), allow_pickle=True)
Y, UID, PERSONS = data["y"], data["uid"].astype(str), data["persons"].astype(str)
PEOPLE = sorted(set(PERSONS))
N = len(LABELS)

records = defaultdict(list)
with open(os.path.join(RUN, "results.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        records[(r["model"], r["protocol"])].append(r)


def oof(model, protocol, seed):
    f = os.path.join(RUN, f"oof_{model}_{protocol}_s{seed}.npz")
    return np.load(f)["probs"] if os.path.exists(f) else None


def macro_f1(y, p):
    f = []
    for c in range(N):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def vote_acc(pred, y, uid, mask):
    ok = tot = 0
    for u in set(uid[mask]):
        m = mask & (uid == u)
        v = np.bincount(pred[m], minlength=N).argmax()
        ok += int(v == y[m][0])
        tot += 1
    return ok / tot if tot else float("nan")


def summarise(probs_list, protocol):
    """Metrics from one or more seeds' out-of-fold probabilities."""
    per_person = defaultdict(list)
    f1s, votes, cms = [], [], np.zeros((N, N))
    for P in probs_list:
        pred = P.argmax(1)
        groups = PEOPLE if protocol != "contrast" else [None]
        accs, f1, vv = [], [], []
        for g in groups:
            mask = (PERSONS == g) if g else np.ones(len(Y), bool)
            a = float((pred[mask] == Y[mask]).mean())
            accs.append(a)
            if g:
                per_person[g].append(a)
            f1.append(macro_f1(Y[mask], pred[mask]))
            vv.append(vote_acc(pred, Y, UID, mask))
        f1s.append(np.mean(f1))
        votes.append(np.mean(vv))
        for t, p in zip(Y, pred):
            cms[t, p] += 1
    pp = {k: float(np.mean(v)) for k, v in per_person.items()}
    fold_acc = list(pp.values()) if pp else [
        float(np.mean([(P.argmax(1) == Y).mean() for P in probs_list]))]
    return {"acc": float(np.mean(fold_acc)), "sd": float(np.std(fold_acc)),
            "per_person": pp, "f1": float(np.mean(f1s)),
            "vote": float(np.mean(votes)), "cm": cms}


S = {}                              # model -> protocol -> summary
for m in meta["models"]:
    for p in meta["protocols"]:
        runs = records.get((m, p))
        if not runs:
            continue
        probs = [oof(m, p, r["seed"]) for r in runs]
        probs = [x for x in probs if x is not None]
        if probs:
            S.setdefault(m, {})[p] = summarise(probs, p)
            S[m][p]["raw"] = runs
            S[m][p]["seconds_fold"] = float(np.mean(
                [f["seconds"] for r in runs for f in r["folds"]]))
            S[m][p]["params"] = runs[0]["n_parameters"]
            S[m][p]["latency"] = next(
                (r["latency_ms"] for r in runs if "latency_ms" in r), None)

# ---- the deployed ML vote, from the comb test's stored out-of-fold probs ----
if os.path.exists(COMB + "data.npz"):
    d = np.load(COMB + "data.npz", allow_pickle=True)
    ml_y, ml_p = d["y"], d["persons"].astype(str)
    ml_rec = d["rec_ids"].astype(str)
    ml_labels = [str(x) for x in d["labels"]]
    P = (np.load(COMB + "ET-500.npz")["proba"]
         + np.load(COMB + "RF-500-leaf3.npz")["proba"]) / 2
    pred = P.argmax(1)
    pp = {k: float((pred[ml_p == k] == ml_y[ml_p == k]).mean())
          for k in sorted(set(ml_p))}
    cms = np.zeros((N, N))
    assert ml_labels == LABELS, (ml_labels, LABELS)
    for t, q in zip(ml_y, pred):
        cms[t, q] += 1
    f1s, votes = [], []
    for k in sorted(set(ml_p)):
        m_ = ml_p == k
        f1s.append(macro_f1(ml_y[m_], pred[m_]))
        ok = tot = 0
        for u in set(ml_rec[m_]):
            mm = m_ & (ml_rec == u)
            ok += int(np.bincount(pred[mm], minlength=N).argmax()
                      == ml_y[mm][0])
            tot += 1
        votes.append(ok / tot)
    S[REF] = {"loso": {"acc": float(np.mean(list(pp.values()))),
                       "sd": float(np.std(list(pp.values()))),
                       "per_person": pp, "f1": float(np.mean(f1s)),
                       "vote": float(np.mean(votes)), "cm": cms,
                       "params": None, "latency": 53.0,
                       "seconds_fold": None, "raw": []}}

order = sorted([m for m in S if "loso" in S[m]],
               key=lambda m: -S[m]["loso"]["acc"])
new_order = [m for m in order if m in meta["models"]]


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x * 100:.{d}f} %"


# ---- figure ---------------------------------------------------------------------
fig = plt.figure(figsize=(21, 34))
gs = GridSpec(8, 4, figure=fig, height_ratios=[0.55, 3.0, 3.4, 3.7, 3.5, 3.4,
                                               3.0, 0.9], hspace=0.5, wspace=0.28,
              left=0.07, right=0.985, top=0.985, bottom=0.01)

# -- title + headline
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
gru = S["gru"]["loso"]["acc"] if "gru" in S else None
cands = [m for m in new_order if m != "gru"]
best_new = max(cands, key=lambda m: S[m]["loso"]["acc"]) if cands else None
ax.text(0, 0.95, "Newer models on this project's data", fontsize=22,
        fontweight="bold", va="top")
head = []
if gru is not None and best_new and best_new != "gru":
    b = S[best_new]["loso"]
    head.append(f"Best of the new ones: {TITLES[best_new]} "
                f"{b['acc'] * 100:.1f} % per window "
                f"({(b['acc'] - gru) * 100:+.1f} points against the current GRU, "
                f"{gru * 100:.1f} %).")
if REF in S:
    head.append(f"ML vote ET+RF (the combination test's winner, stored result, "
                f"1.0 s windows -- not the file in models/), same people held out: "
                f"{S[REF]['loso']['acc'] * 100:.1f} %.")
head.append(f"{len(PEOPLE)} people, {len(Y)} windows of 0.7 s, every window "
            f"scored by a model that never saw that person (leave-one-person-"
            f"out), {meta['epochs']} epochs fixed, CUDA, "
            f"{len(meta['seeds'])} seed(s).")
ax.text(0, 0.40, "\n".join(head), fontsize=11.5, color=TEXT_2, va="top",
        linespacing=1.5)

# -- A: ranking with per-person dots
ax = fig.add_subplot(gs[1, :2])
ys = np.arange(len(order))[::-1]
for yy, m in zip(ys, order):
    s = S[m]["loso"]
    ax.barh(yy, s["acc"] * 100, color=COLOR[m], height=0.62, alpha=0.9,
            zorder=2)
    for person, a in s["per_person"].items():
        ax.plot(a * 100, yy, "o", ms=4.2, color=TEXT, alpha=0.55, zorder=3)
    ax.text(102, yy, f"{s['acc'] * 100:.1f} %", va="center", fontsize=10.5,
            fontweight="bold", clip_on=False)
    if "dc_only" in S[m]:
        ax.plot(S[m]["dc_only"]["acc"] * 100, yy - 0.38, marker="|", ms=11,
                color=TEXT_2, zorder=4)
ax.set_yticks(ys)
ax.set_yticklabels([TITLES[m] for m in order], fontsize=10.5)
ax.set_xlim(0, 100)
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
maj = max(np.bincount(Y)) / len(Y) * 100
ax.axvline(maj, color=MUTED, lw=1.4, ls="--", zorder=1)
ax.text(maj + 0.6, len(order) - 0.45, f"majority class {maj:.0f} %",
        fontsize=8.5, color=TEXT_2, va="bottom")
ax.set_xlabel("accuracy per window, person held out   (bar = mean of the 5 "
              "people, dots = each person, | = same model with time removed)")
ax.set_title("A   Accuracy, person held out")

# -- B: model x person heatmap
ax = fig.add_subplot(gs[1, 2:])
M = np.array([[S[m]["loso"]["per_person"].get(p, np.nan) * 100
               for p in PEOPLE] for m in order])
im = ax.imshow(M, cmap=CMAP, vmin=20, vmax=100, aspect="auto",
               interpolation="nearest")
ax.set_xticks(range(len(PEOPLE)))
ax.set_xticklabels(PEOPLE)
ax.set_yticks([])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=ink(CMAP, M[i, j], 20, 100))
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("B   Per person held out (%), same row order as A")

# -- C: table
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
cols = ["model", "params", "per window\n(± across people)", "per recording\n(vote)",
        "macro F1", "time removed\n(DC-only)", "what time adds", "within-person\n(optimistic)",
        "ms per live\nwindow", "s per fold\n(training)"]
rows = []
for m in order:
    s = S[m]["loso"]
    dc = S[m].get("dc_only")
    ct = S[m].get("contrast")
    rows.append([
        TITLES[m],
        "2 x 500 trees" if m == REF else f"{s['params']:,}".replace(",", " ")
        + (" (+ read-out)" if m == "rocket" else ""),
        f"{s['acc'] * 100:.1f} % ± {s['sd'] * 100:.1f}",
        pct(s["vote"]),
        pct(s["f1"]),
        pct(dc["acc"]) if dc else ("77.8 %*" if m == REF else "—"),
        (f"{(s['acc'] - dc['acc']) * 100:+.1f}" if dc
         else ("-0.2*" if m == REF else "—")),
        pct(ct["acc"]) if ct else "—",
        "—" if s["latency"] is None else
        (f"{s['latency']:.1f}" if s["latency"] >= 10 else f"{s['latency']:.2f}"),
        "—" if s["seconds_fold"] is None else f"{s['seconds_fold']:.0f}",
    ])
tab = ax.table(cellText=rows, colLabels=cols, loc="upper center",
               cellLoc="center", colLoc="center",
               colWidths=[0.19, 0.115, 0.11, 0.085, 0.065, 0.085, 0.08, 0.09, 0.095,
                          0.07])
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1, 2.05)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor("#efeeea")
        cell.set_text_props(fontweight="bold", color=TEXT)
        cell.set_height(cell.get_height() * 1.25)
    else:
        cell.set_facecolor(SURFACE)
        if c == 0:
            cell.set_text_props(ha="left", fontweight="bold",
                                color=darker(COLOR[order[r - 1]]))
            cell._loc = "left"
        if c == 2:
            cell.set_text_props(fontweight="bold")
ax.set_title("C   Scores", pad=2)

# -- D: confusion matrices (row-normalised), reference + all
panels = order
for k, m in enumerate(panels[:7]):
    ax = fig.add_subplot(gs[3 + k // 4, k % 4])
    if k == 0:
        FIRST_CM = ax
    cm = S[m]["loso"]["cm"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    cm_map = model_cmap(m)
    ax.imshow(norm, cmap=cm_map, vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([SHORT.get(l, l) for l in LABELS], rotation=40,
                       ha="right", fontsize=8.5)
    ax.set_yticklabels([SHORT.get(l, l) for l in LABELS] if k % 4 == 0
                       else [""] * N, fontsize=8.5)
    for i in range(N):
        for j in range(N):
            v = norm[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold" if i == j else None,
                        color=ink(cm_map, v))
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{TITLES[m]}\n{S[m]['loso']['acc'] * 100:.1f} %",
                 fontsize=10.5, color=darker(COLOR[m]))
    if k % 4 == 0:
        ax.set_ylabel("true gesture")
    if k >= 3:
        ax.set_xlabel("predicted")

# -- 8th cell: biggest mistakes per model
ax = fig.add_subplot(gs[4, 3])
ax.axis("off")
ax.text(0, 1.0, "Most common mistake\n(share of that gesture's windows)",
        fontsize=10.5, fontweight="bold", va="top")
y0 = 0.80
for m in order:
    cm = S[m]["loso"]["cm"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    off = norm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    ax.text(0, y0, TITLES[m], fontsize=9.2, color=darker(COLOR[m]),
            fontweight="bold", va="top")
    ax.text(0, y0 - 0.052, f"{LABELS[i]}  ->  {LABELS[j]}   {off[i, j]:.0f} %",
            fontsize=9.2, color=TEXT_2, va="top")
    y0 -= 0.125
_pos = FIRST_CM.get_position()
fig.text(_pos.x0, _pos.y1 + 0.028, "D   Confusion matrices — each row sums to "
         "100 % (true gesture, all held-out windows pooled)", fontsize=12,
         fontweight="bold", ha="left")

# -- E: learning curves
nets = [m for m in new_order if "loso" in S[m] and S[m]["loso"]["raw"]
        and S[m]["loso"]["raw"][0]["folds"][0]["losses"]]
ax = fig.add_subplot(gs[5, 0:2])
for m in nets:
    folds = [f for r in S[m]["loso"]["raw"] for f in r["folds"]]
    L = np.mean([f["losses"] for f in folds], axis=0)
    ax.plot(np.arange(1, len(L) + 1), L, color=COLOR[m], lw=2,
            label=TITLES[m])
ax.set_xlim(0, meta["epochs"] * 1.02)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
          fontsize=8.5)
ax.set_xlabel("epoch")
ax.set_ylabel("training loss (mean over the 5 folds)")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.set_title("E   Training loss")

ax = fig.add_subplot(gs[5, 2:])
for m in nets:
    folds = [f for r in S[m]["loso"]["raw"] for f in r["folds"]]
    eps = [e for e, _ in folds[0]["monitor"]]
    A = np.mean([[a for _, a in f["monitor"]] for f in folds], axis=0) * 100
    ax.plot(eps, A, color=COLOR[m], lw=2, marker="o", ms=3.5)
    ax.text(eps[-1] + 2, A[-1], f"{A[-1]:.0f}", color=darker(COLOR[m]), fontsize=9,
            va="center", fontweight="bold")
ax.set_xlim(0, meta["epochs"] * 1.08)
ax.set_xlabel("epoch")
ax.set_ylabel("held-out person accuracy, %")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.set_title("F   Held-out accuracy while training (a monitor, never used to choose)")
handles = [plt.Line2D([0], [0], color=COLOR[m], lw=2, label=TITLES[m])
           for m in nets]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          fontsize=8.5, ncol=3)

# -- G: three protocols side by side
ax = fig.add_subplot(gs[6, :])
x = np.arange(len(order))
w = 0.26
series = [("loso", "person held out  (quote this)", TEXT_2, None),
          ("dc_only", "same, time removed (channel means only)", MUTED, "///"),
          ("contrast", "within-person windows (optimistic)", "#86b6ef", None)]
for k, (key, label, col, hatch) in enumerate(series):
    vals = [S[m][key]["acc"] * 100 if key in S[m] else np.nan for m in order]
    bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, hatch=hatch,
                  edgecolor=SURFACE, label=label, zorder=2)
    for xi, v in zip(x + (k - 1) * w, vals):
        if not np.isnan(v):
            ax.text(xi, v + 1.0, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([TITLES[m] for m in order], fontsize=10)
for tick, m in zip(ax.get_xticklabels(), order):
    tick.set_color(darker(COLOR[m]))
    tick.set_fontweight("bold")
ax.set_ylim(0, 108)
ax.set_ylabel("accuracy per window, %")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.0))
ax.set_title("G   Validation: the same models under three protocols", pad=8)

# -- caption
ax = fig.add_subplot(gs[7, :])
ax.axis("off")
cap = ("How to read it.  Person held out = the honest number; the 'within-person' "
       "column (windows of the same person in training and test) is what a "
       "naive split would report, and the gap between the two is how much of a "
       "score is memorising the wearer.  'Time removed' feeds the same model "
       "each window flattened to its 8 channel means: a model that does not beat "
       "it is reading amplitude, not the gesture's shape in time (CLAUDE.md "
       "8f/8k).  One seed and five people: differences of a couple of points are "
       "inside the spread between people shown in the table.  ROCKET-GPU is "
       "random convolutional kernels with only a ridge read-out fitted; its "
       "parameter count is the fixed kernels plus the read-out.  *The ML vote's "
       "reference row is its stored comb-test result (1.0 s windows, not the "
       "0.7 s windows used here; DC-only 77.8 % and the 53 ms live cost are from "
       "docs/MODELS_COMPARED.md).  ms / live window: one window, batch of 1, "
       "device synchronised, on this machine's GTX 1650 (ROCKET includes the "
       "feature transform; the ML vote is CPU).")
ax.text(0, 1.0, textwrap.fill(cap, 215), fontsize=8.8, color=TEXT_2, va="top",
        linespacing=1.4)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.3)
print("wrote", os.path.relpath(OUT_PNG, PROJECT), "from", os.path.relpath(RUN, PROJECT))

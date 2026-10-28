"""
Behaviour figures for docs/MODELS_COMPARED.md: how the models fail, not just
how often.

    python tools/make_figures_behaviour.py [OUT_DIR]      (default: docs/figures)

Companion to `make_figures.py` (same palette, same colour per model, same
data) and, like it, reads only what is already saved -- nothing is retrained:

  * ML: the comb-test run's stored out-of-fold probabilities for the winning
    setup (models/comb_test/<run>/ml_oof). Every window was scored by a model
    that never saw that person, so everything per person / per window /
    per recording below is honest.
  * NN: the comb test and the stats sidecar keep the NN's SUMMED confusion
    matrices, its fold accuracies and its per-person scores, but NOT its
    per-window probabilities. So the figures that need a probability per
    window (per-person confusion, confidence, timelines, agreement with the
    ML vote) are ML-only, and every caption says so. Re-running the net's
    cross-validation while keeping its probabilities would add them.

Loads no dataset and imports no project code, so it runs in seconds.
"""
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap            # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
RUN = os.path.join(PROJECT, "models", "comb_test", "20261022-003918")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT, "docs",
                                                          "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (same as make_figures.py) ------------------------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
MUTED = "#c9c8c2"
ML = "#2a78d6"          # the ML vote, everywhere
NN = "#eb6834"          # the neural net, everywhere
ML4 = "#1baf7a"
LIGHT_BLUE = "#86b6ef"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
#  Diverging with the two MODELS as the poles and the neutral grey between:
#  blue = the ML vote has the larger share, orange = the net has.
DIV = LinearSegmentedColormap.from_list("ml_nn", [ML, "#f0efec", NN])
#  The net is orange everywhere, so its heat maps are orange too.
ORANGE_RAMP = ["#fcfcfb", "#fbe0d2", "#f8c4a8", "#f4a27a", "#ee8049",
               "#d25a25", "#a4401a", "#6f2a0f"]
CMAP_NN = LinearSegmentedColormap.from_list("orange", ORANGE_RAMP)


def ink(cmap, value, vmin=0.0, vmax=100.0):
    """White or near-black text, whichever reads better on the cell colour
    (WCAG relative luminance) -- the switch follows the ramp instead of a
    hand-picked percentage."""
    r, g, b, _a = cmap((value - vmin) / (vmax - vmin))
    return "white" if _lum((r, g, b)) < 0.18 else TEXT


def _lum(rgb):
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

#  Gestures use the application's own colours (cfg.label_color: "the one
#  colour of gesture X, everywhere it is drawn"), not a palette of the
#  figures' own -- the old one painted FIST green where the app paints it red.
from emg_app import config as _cfg                                   # noqa: E402
GESTURE_COLOR = dict(_cfg.LABEL_COLORS)    # the colours the application draws
GESTURE_LETTER = {"REST": "R", "FIST": "F", "OPEN": "O", "THUMB_UP": "T",
                  "WAVE_LEFT": "L", "WAVE_RIGHT": "W"}
SHORT = {"WAVE_LEFT": "WAVE_L", "WAVE_RIGHT": "WAVE_R", "THUMB_UP": "THUMB"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, PROJECT))


def note(fig, text, y=-0.02):
    #  Wrapped by hand: matplotlib's wrap=True cuts a long caption at the
    #  figure edge once the saved figure is cropped tight.
    chars = max(40, int(fig.get_figwidth() * 12.5))
    fig.text(0.0, y, textwrap.fill(text, chars), ha="left", va="top",
             fontsize=8.5, color=TEXT_2, linespacing=1.35)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---- data -------------------------------------------------------------------
SETUP = "bandpass-level-w1-o0.75"
OOF = os.path.join(RUN, "ml_oof", SETUP + "__")
data = np.load(OOF + "data.npz", allow_pickle=True)
Y = data["y"]
PERSONS = data["persons"].astype(str)
RECS = data["rec_ids"].astype(str)
LABELS = [str(x) for x in data["labels"]]
PEOPLE = sorted(set(PERSONS))
N = len(LABELS)


def proba(name):
    return np.load(OOF + name + ".npz")["proba"]


P_VOTE = (proba("ET-500") + proba("RF-500-leaf3")) / 2
PRED = P_VOTE.argmax(1)
RIGHT = PRED == Y
nn_side = json.load(open(os.path.join(PROJECT, "models", "gesture_nn.json"),
                         encoding="utf-8"))
NN_STATS = nn_side["stats"]
nn_labels = list(NN_STATS["labels"])
assert nn_labels == LABELS, "label order differs between ML and NN"
CM_NN = np.asarray(NN_STATS["confusion_matrix"])
CM_NN_DC = np.asarray(NN_STATS["dc_only_control"]["confusion_matrix"])
nn_rows = [json.loads(l) for l in open(os.path.join(RUN, "nn_results.jsonl"),
                                       encoding="utf-8") if l.strip()]


def confusion(y, pred):
    cm = np.zeros((N, N), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


CM_ML = confusion(Y, PRED)


def row_pct(cm):
    return cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100


def draw_matrix(ax, cm, title, small=False, show_y=True, cmap=CMAP):
    pct = row_pct(cm)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    fontsize=8 if small else 10,
                    fontweight="bold" if i == j else "normal",
                    color=ink(cmap, pct[i, j]))
    names = [SHORT.get(l, l) for l in LABELS]
    ax.set_xticks(range(N), names, rotation=45 if small else 0,
                  ha="right" if small else "center",
                  fontsize=7.5 if small else 8.5)
    ax.set_yticks(range(N), names if show_y else [], fontsize=7.5 if small
                  else 8.5)
    ax.set_title(title, pad=8, fontsize=10.5 if small else 12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, N), minor=True)
    ax.set_yticks(np.arange(-.5, N), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    #  The grid lines that draw the gaps between cells are clipped to the axes,
    #  so on the outer edge the colour of the last cell showed as a thin dark
    #  sliver beyond them. Unclipped, they cover it.
    for _tick in list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_minor_ticks()):
        _tick.gridline.set_clip_on(False)


# ==========================================================================
# 1. The ML vote's confusion matrix for each person held out
# ==========================================================================
fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.4))
axes = axes.ravel()
for k, (ax, person) in enumerate(zip(axes, PEOPLE)):
    m = PERSONS == person
    acc = RIGHT[m].mean() * 100
    draw_matrix(ax, confusion(Y[m], PRED[m]),
                f"{person}  {acc:.0f}%", small=True, show_y=(k % 3 == 0))
    if k % 3 == 0:
        ax.set_ylabel("actual gesture")
    ax.set_xlabel("predicted")
guide = axes[len(PEOPLE)]
guide.axis("off")
guide.text(0.0, 0.95, "How to read", fontsize=11, fontweight="bold",
           va="top")
guide.text(0.0, 0.84,
           "Each panel is the same ML vote,\ntrained on the OTHER four people\n"
           "and scored on the one named.\n\nRow = the gesture actually made;\n"
           "number = % of that row given\neach answer. Bold = correct.\n\n"
           "A cell can rest on few windows\n(see the person x gesture figure).",
           fontsize=9.5, color=TEXT_2, va="top", linespacing=1.4)
fig.suptitle("ML vote (ET + RF): the same model, each person held out",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=3.2)
note(fig, "The four others make different, scattered mistakes: Betka's "
          "THUMB_UP goes to FIST, Bruno's to OPEN and WAVE_L, Dano's OPEN to "
          "THUMB_UP. Barbora's are a consistent SUBSTITUTION: FIST -> OPEN "
          "(81 %), OPEN -> WAVE_R (100 %), THUMB_UP -> OPEN (83 %), while "
          "REST and both waves are right. That is the pattern of her "
          "gestures landing on other gestures' places, not of a noisy "
          "signal. (The net's per-window answers were not saved, so this "
          "is ML only.)", y=0.03)
save(fig, "confusion_per_person_ml.png")

# ==========================================================================
# 2. Person x gesture: where exactly is it wrong
# ==========================================================================
grid = np.zeros((len(PEOPLE), N))
count = np.zeros((len(PEOPLE), N), int)
for i, person in enumerate(PEOPLE):
    for g in range(N):
        m = (PERSONS == person) & (Y == g)
        count[i, g] = m.sum()
        grid[i, g] = RIGHT[m].mean() * 100 if m.any() else np.nan
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.imshow(grid, cmap=CMAP, vmin=0, vmax=100, aspect="auto",
          interpolation="nearest")
for i in range(len(PEOPLE)):
    for g in range(N):
        v = grid[i, g]
        ax.text(g, i - 0.1, f"{v:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=ink(CMAP, v))
        ax.text(g, i + 0.27, f"n={count[i, g]}", ha="center", va="center",
                fontsize=7, color=ink(CMAP, v), alpha=0.85)
ax.set_xticks(range(N), [SHORT.get(l, l) for l in LABELS], fontsize=9)
ax.set_yticks(range(len(PEOPLE)), PEOPLE, fontsize=9)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, N), minor=True)
ax.set_yticks(np.arange(-.5, len(PEOPLE)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
for _tick in list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_minor_ticks()):
    _tick.gridline.set_clip_on(False)
ax.set_title("ML vote: % of windows right, person x gesture", pad=28)
note(fig, "Person held out; n = windows. The trouble is the FIST / OPEN / "
          "THUMB_UP trio, unevenly: Barbora gets 19 / 0 / 4 % of them while "
          "reading REST and both waves at 94-98 %; Bruno never gets "
          "THUMB_UP (0 %), Betka 24 %. Cells with a small n rest on only a "
          "few recordings.")
save(fig, "person_gesture_heatmap.png")

# ==========================================================================
# 3. precision / recall / F1 per gesture, both models
# ==========================================================================
def prf(cm):
    tp = np.diag(cm).astype(float)
    recall = tp / cm.sum(1).clip(min=1)
    precision = tp / cm.sum(0).clip(min=1)
    f1 = 2 * precision * recall / (precision + recall).clip(min=1e-9)
    return precision * 100, recall * 100, f1 * 100


p_ml, r_ml, f_ml = prf(CM_ML)
p_nn, r_nn, f_nn = prf(CM_NN)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
ys = np.arange(N)
for ax, title, a, b in ((axes[0], "precision", p_ml, p_nn),
                        (axes[1], "recall", r_ml, r_nn),
                        (axes[2], "F1", f_ml, f_nn)):
    ax.barh(ys - 0.19, a, height=0.34, color=ML, label="ML vote (ET + RF)")
    ax.barh(ys + 0.19, b, height=0.34, color=NN, label="Neural net (GRU)")
    for yy, v in zip(ys - 0.19, a):
        ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=8, color=TEXT_2)
    for yy, v in zip(ys + 0.19, b):
        ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_title(title)
    ax.set_xlim(0, 112)
    ax.set_xlabel("%")
    vgrid(ax)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ys, LABELS)
axes[0].invert_yaxis()
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, fontsize=9)
fig.tight_layout()
note(fig, "Pooled over the five held-out people. Recall = of the windows of "
          "this gesture, how many were found. Precision = of the windows the "
          "model called this gesture, how many were. The net's weak spots are "
          "FIST recall (49 % against 68 %) and THUMB_UP precision (52 % "
          "against 73 %). The ML vote's are OPEN precision (54 %: many "
          "windows of other gestures are called OPEN) and THUMB_UP recall "
          "(59 %). The net is slightly more precise on the two waves.")
save(fig, "precision_recall_f1.png")

# ==========================================================================
# 4. Where do the two models differ (cell by cell)
# ==========================================================================
diff = row_pct(CM_NN) - row_pct(CM_ML)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
lim = 25
ax.imshow(diff, cmap=DIV, vmin=-lim, vmax=lim)
for i in range(N):
    for j in range(N):
        v = diff[i, j]
        txt = "0" if round(v) == 0 else f"{v:+.0f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                fontweight="bold" if i == j else "normal",
                color="white" if abs(v) > 17 else TEXT)
names = [SHORT.get(l, l) for l in LABELS]
ax.set_xticks(range(N), names, fontsize=8.5)
ax.set_yticks(range(N), names, fontsize=8.5)
ax.set_xlabel("predicted")
ax.set_ylabel("actual gesture")
ax.set_title("Where the two models differ", pad=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, N), minor=True)
ax.set_yticks(np.arange(-.5, N), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
ax.text(0.0, -0.17, "blue: the ML vote puts a larger share of the row here",
        transform=ax.transAxes, fontsize=9, color=ML)
ax.text(0.0, -0.215, "orange: the net puts a larger share of the row here",
        transform=ax.transAxes, fontsize=9, color=NN)
note(fig, "Cell = net's % of the row minus the ML vote's % of the row, in "
          "percentage points. On the diagonal a bigger share is better: the "
          "ML vote is ahead on every gesture but OPEN (a tie), by 20 points "
          "on FIST. Off the diagonal a bigger share is more mistakes of that "
          "kind: the net sends FIST to THUMB_UP (+14) and THUMB_UP to OPEN "
          "(+8) more often; the ML vote sends OPEN to WAVE_R (13 points more) "
          "and THUMB_UP to WAVE_L (7 more). The windows differ (1.0 s vs "
          "0.7 s), so this compares shares, not the same windows.",
     y=-0.1)
save(fig, "confusion_difference.png")

# ==========================================================================
# 5. ML vote: confidence, calibration, near misses
# ==========================================================================
conf = P_VOTE.max(1)
fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
axes = axes.ravel()
ax = axes[0]
bins = np.linspace(0.2, 1.0, 17)
ax.hist([conf[~RIGHT], conf[RIGHT]], bins=bins, stacked=True,
        color=[TEXT_2, ML], label=["wrong", "right"])
ax.set_xlabel("probability the vote gave its top answer")
ax.set_ylabel("windows")
ax.set_title("How sure it is")
ax.legend(fontsize=9)
hgrid(ax)

ax = axes[1]
edges = np.array([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
mid, acc, cnt = [], [], []
ece = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum() < 10:
        continue
    mid.append(conf[m].mean())
    acc.append(RIGHT[m].mean())
    cnt.append(int(m.sum()))
    ece += m.sum() / len(conf) * abs(RIGHT[m].mean() - conf[m].mean())
ax.plot([0.3, 1], [0.3, 1], color=MUTED, linewidth=1.2, linestyle="--")
ax.plot(mid, acc, "-o", color=ML, linewidth=2, markersize=6)
for x, a, c in zip(mid, acc, cnt):
    late = x > 0.8
    ax.annotate(f"n={c}", (x, a), xytext=(-6, -14) if late else (0, -13),
                textcoords="offset points", ha="right" if late else "center",
                va="center" if late else "baseline", fontsize=7.5,
                color=TEXT_2)
ax.set_xlim(0.3, 1.0)
ax.set_ylim(0.3, 1.0)
ax.set_xlabel("mean stated confidence")
ax.set_ylabel("fraction actually right")
ax.set_title(f"Calibration  (average gap {ece * 100:.1f} pts)")
ax.text(0.97, 0.34, "dashed = perfectly calibrated", fontsize=8,
        color=TEXT_2, ha="right")
hgrid(ax)

ax = axes[2]
order = np.argsort(-P_VOTE, axis=1)
ranks = [(order[:, :k] == Y[:, None]).any(1).mean() * 100 for k in (1, 2, 3)]
bars = ax.bar(["top 1", "top 2", "top 3"], ranks, color=ML, width=0.55)
for b, v in zip(bars, ranks):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center",
            fontsize=10, color=TEXT_2)
ax.set_ylim(0, 108)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("% of windows, true gesture in the top k")
ax.set_title("Near misses")
hgrid(ax)
ax.tick_params(axis="x", length=0)

#  Act only when sure: accuracy of what is kept against how much is kept.
ax = axes[3]
thresholds = np.linspace(0.3, 0.95, 66)
coverage = [(conf >= t).mean() * 100 for t in thresholds]
kept_acc = [RIGHT[conf >= t].mean() * 100 for t in thresholds]
ax.plot(coverage, kept_acc, color=ML, linewidth=2)
marks = {}
for t in (0.5, 0.7, 0.9):
    cv, ac = (conf >= t).mean() * 100, RIGHT[conf >= t].mean() * 100
    marks[t] = (cv, ac)
    ax.scatter([cv], [ac], s=34, color=ML, zorder=3)
    ax.annotate(f"sure >= {t:.1f}:\n{ac:.0f}% right, {cv:.0f}% kept",
                (cv, ac), xytext=(10, 8), textcoords="offset points",
                fontsize=8, color=TEXT_2)
ax.set_xlabel("windows answered, % (the rest: 'not sure')")
ax.set_ylabel("accuracy of the answered windows, %")
ax.set_title("Answer only when sure")
ax.set_xlim(0, 105)
ax.set_ylim(70, 108)
ax.set_yticks(range(70, 101, 5))
hgrid(ax)
fig.tight_layout(rect=(0, 0.09, 1, 1))
note(fig, "ML vote, person held out, all 1,787 windows. Top left: the votes "
          "of wrong windows (dark, stacked under the right ones) are spread over low confidence, while the "
          "right ones pile up at 1.0. Top right: the curve sits ABOVE the "
          "diagonal -- the vote is UNDER-confident (a forest's probability "
          "is the share of trees that agree, which is cautious): when it "
          "says 0.36 it is right 53 % of the time. Read the number as an "
          "ordering, not as a percentage. Bottom left: in about half of the "
          "wrong windows the true gesture is the runner-up. Bottom right: "
          "ignoring windows the vote is unsure about buys accuracy at the "
          "price of silence. (The net's per-window probabilities were not "
          "saved, so this is ML only.)", y=0.05)
save(fig, "ml_confidence.png")
CONF_RIGHT, CONF_WRONG = conf[RIGHT].mean(), conf[~RIGHT].mean()
print(f"  confidence right {CONF_RIGHT:.3f} wrong {CONF_WRONG:.3f}  "
      f"ECE {ece * 100:.1f}  top1/2/3 {[round(r, 1) for r in ranks]}")
print("  accuracy-coverage:", {t: (round(c, 1), round(a, 1))
                               for t, (c, a) in marks.items()})
for p in PEOPLE:
    m = PERSONS == p
    print(f"   {p}: mean confidence {conf[m].mean():.2f}, "
          f"when wrong {conf[m & ~RIGHT].mean():.2f}, "
          f"share of windows sure>=0.9: {(conf[m] >= 0.9).mean() * 100:.0f}%")

# ==========================================================================
# 6. do the ML models make the same mistakes (why a vote adds little)
# ==========================================================================
members = ["ET-500", "RF-500-leaf3", "HGB-lr0.05", "MLP-128", "kNN-31",
           "SVM-rbf-C10", "LDA-shrinkage", "LogReg-C1"]
nice = {"ET-500": "ExtraTrees", "RF-500-leaf3": "RandomForest",
        "HGB-lr0.05": "HistGradBoost", "MLP-128": "MLP-128", "kNN-31": "kNN-31",
        "SVM-rbf-C10": "SVM", "LDA-shrinkage": "LDA", "LogReg-C1": "LogReg"}
wrong = {m: proba(m).argmax(1) != Y for m in members}
share = np.zeros((len(members), len(members)))
for i, a in enumerate(members):
    for j, b in enumerate(members):
        share[i, j] = (wrong[a] & wrong[b]).sum() / max(1, wrong[a].sum()) * 100
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0),
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
ax.imshow(share, cmap=CMAP, vmin=40, vmax=100, interpolation="nearest")
for i in range(len(members)):
    for j in range(len(members)):
        ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center",
                fontsize=9, color=ink(CMAP, share[i, j], 40, 100))
ax.set_xticks(range(len(members)), [nice[m] for m in members], rotation=40,
              ha="right", fontsize=8.5)
ax.set_yticks(range(len(members)), [nice[m] for m in members], fontsize=8.5)
ax.set_xlabel("... is ALSO wrong on the same window (%)")
ax.set_ylabel("of the windows where this model is wrong")
ax.set_title("Do two models make the same mistakes?")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax = axes[1]
n_wrong = np.sum([wrong[m] for m in members], axis=0)
counts = [int((n_wrong == k).sum()) for k in range(len(members) + 1)]
ax.bar(range(len(members) + 1), counts,
       color=[ML4] + [MUTED] * (len(members) - 1) + [TEXT_2], width=0.7)
for k, c in enumerate(counts):
    ax.text(k, c + 12, str(c), ha="center", fontsize=8.5, color=TEXT_2)
ax.set_xlabel(f"how many of these {len(members)} models are wrong on a window")
ax.set_ylabel("windows")
ax.set_title("Easy, hard and impossible windows")
hgrid(ax)
fig.tight_layout()
share_trees = share[0, 1]
note(fig, "Winning setup, person held out. Left: read along a row -- of "
          "the windows ExtraTrees gets wrong, "
          f"{share_trees:.0f} % are also wrong for RandomForest, so averaging "
          "the two changes few answers. SVM, LDA and the rest overlap less "
          "with the forests but are wrong in more places. Right: "
          f"{counts[0]} windows are right for all eight, {counts[-1]} are "
          "wrong for all eight -- no vote of these eight can fix those; "
          "they need different information, not another model of the same "
          "features.", y=-0.04)
save(fig, "ml_error_overlap.png")
print(f"  ET errors shared with RF: {share_trees:.1f} %; all right "
      f"{counts[0]}, all wrong {counts[-1]} of {len(Y)}")

# ==========================================================================
# 7. a recording as the live app sees it: strips of window answers
# ==========================================================================
starts = np.r_[0, np.flatnonzero(RECS[1:] != RECS[:-1]) + 1, len(RECS)]
records = []
for a, b in zip(starts[:-1], starts[1:]):
    records.append((RECS[a], PERSONS[a], int(Y[a]), a, b))


def pick(person, gesture):
    """The recording at the MEDIAN accuracy for that person and gesture."""
    cand = [(RIGHT[a:b].mean(), a, b) for r, p, g, a, b in records
            if p == person and g == gesture]
    cand.sort()
    return cand[len(cand) // 2]


show = ["Dano", "Barbora"]
longest = max(pick(person, g)[2] - pick(person, g)[1]
              for person in show for g in range(N))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
for ax, person in zip(axes, show):
    for row, g in enumerate(range(N)):
        acc, a, b = pick(person, g)
        for k, idx in enumerate(range(a, b)):
            p = PRED[idx]
            ax.add_patch(Rectangle((k, row), 0.94, 0.86,
                                   facecolor=GESTURE_COLOR[LABELS[p]],
                                   edgecolor=TEXT if p != g else "none",
                                   linewidth=1.8 if p != g else 0))
            ax.text(k + 0.47, row + 0.43, GESTURE_LETTER[LABELS[p]],
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white" if _lum(matplotlib.colors.to_rgb(
                        GESTURE_COLOR[LABELS[p]])) < 0.3 else TEXT)
        ax.text(-0.35, row + 0.43, LABELS[g], ha="right", va="center",
                fontsize=9, color=TEXT)
        ax.text(b - a + 0.1, row + 0.43, f"{acc * 100:.0f}%", ha="left",
                va="center", fontsize=8.5, color=TEXT_2)
    ax.set_xlim(0, longest + 4)
    ax.set_ylim(N, -0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{person} held out", loc="left")
handles = [Rectangle((0, 0), 1, 1, facecolor=GESTURE_COLOR[l])
           for l in LABELS]
fig.legend(handles, [f"{GESTURE_LETTER[l]} = {l}" for l in LABELS],
           loc="lower center", ncol=N, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
note(fig, "One typical recording (median accuracy) per gesture. Each cell is "
          "one 1.0 s window, in time order, coloured and lettered by what the "
          "ML vote answered; a heavy border = a wrong answer. Row label = "
          "the gesture actually made; % = windows right. Dano's are right "
          "apart from one OPEN recording, where a stretch of the hold was "
          "read as FIST and then THUMB_UP before returning to OPEN. "
          "Barbora's FIST, OPEN and THUMB_UP recordings are answered with "
          "the same wrong gesture from the first window to the last, which "
          "averaging more windows cannot fix.", y=-0.07)
save(fig, "ml_timeline.png")

# ==========================================================================
# 8. live behaviour: smoothing, flicker, where in a hold errors fall
# ==========================================================================
def smoothed_pred(k):
    out = np.empty_like(PRED)
    for a, b in zip(starts[:-1], starts[1:]):
        cs = np.cumsum(np.vstack([np.zeros((1, N)), P_VOTE[a:b]]), axis=0)
        for i in range(b - a):
            lo = max(0, i + 1 - k)
            out[a + i] = ((cs[i + 1] - cs[lo]) / (i + 1 - lo)).argmax()
    return out


def flips(pred):
    n = f = 0
    for a, b in zip(starts[:-1], starts[1:]):
        n += b - a - 1
        f += int((pred[a + 1:b] != pred[a:b - 1]).sum())
    return f / max(1, n) * 100


ks = [1, 2, 3, 5, 8]
acc_all, acc_bar, acc_rest, flip = [], [], [], []
bar = PERSONS == "Barbora"
for k in ks:
    pk = smoothed_pred(k)
    r = pk == Y
    acc_all.append(r.mean() * 100)
    acc_bar.append(r[bar].mean() * 100)
    acc_rest.append(r[~bar].mean() * 100)
    flip.append(flips(pk))
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
ax = axes[0]
ax.plot(ks, acc_rest, "-o", color=ML, linewidth=2, label="other four people")
ax.plot(ks, acc_bar, "-o", color=TEXT_2, linewidth=2, label="Barbora")
for x, v in zip(ks, acc_rest):
    ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8, color=TEXT_2)
for x, v in zip(ks, acc_bar):
    ax.annotate(f"{v:.0f}", (x, v), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8,
                color=TEXT_2)
ax.set_xlabel("windows averaged (the latest k)")
ax.set_ylabel("accuracy per window, %")
ax.set_title("Smoothing: +2 points, none for Barbora")
ax.set_xticks(ks)
ax.set_ylim(30, 100)
ax.legend(fontsize=9, loc="center right")
hgrid(ax)

ax = axes[1]
ax.plot(ks, flip, "-o", color=ML, linewidth=2)
for x, v in zip(ks, flip):
    ax.annotate(f"{v:.0f}%", (x, v), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8,
                color=TEXT_2)
ax.set_xlabel("windows averaged (the latest k)")
ax.set_ylabel("answer changes between consecutive windows, %")
ax.set_title("... but halves the flicker")
ax.set_xticks(ks)
ax.set_ylim(0, max(flip) * 1.25)
hgrid(ax)

ax = axes[2]
pos_edges = np.linspace(0, 1, 6)
err = np.zeros(5)
cnt = np.zeros(5)
for a, b in zip(starts[:-1], starts[1:]):
    n = b - a
    rel = (np.arange(n) + 0.5) / n
    which = np.clip(np.digitize(rel, pos_edges) - 1, 0, 4)
    for q in range(5):
        m = which == q
        err[q] += (~RIGHT[a:b][m]).sum()
        cnt[q] += m.sum()
err_pct = err / cnt * 100
bars = ax.bar(["first\nfifth", "2nd", "middle", "4th", "last\nfifth"],
              err_pct, color=ML, width=0.6)
for bb, v in zip(bars, err_pct):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.6, f"{v:.0f}%",
            ha="center", fontsize=9.5, color=TEXT_2)
ax.set_ylabel("windows wrong, %")
ax.set_title("Where in the hold the errors fall")
ax.set_ylim(0, max(err_pct) * 1.3)
hgrid(ax)
ax.tick_params(axis="x", length=0)
fig.tight_layout()
note(fig, "ML vote, person held out. Smoothing = the mean of the latest k "
          "windows' probabilities (what the panel does), in time order inside "
          "each recording. These windows are 0.25 s apart; the live app "
          "predicts every 0.15 s, so k there covers a shorter time. Right: "
          "errors do not cluster at the start of a hold, i.e. they are not "
          "just the onset; the model is wrong through the hold.")
save(fig, "ml_live_behaviour.png")
print("  smoothing k:", ks)
print("   others", [round(v, 1) for v in acc_rest])
print("   barbora", [round(v, 1) for v in acc_bar])
print("   flips", [round(v, 1) for v in flip])
print("   error by position fifth", [round(v, 1) for v in err_pct])

# ==========================================================================
# 9. the price of a new person: same person vs never-seen person
# ==========================================================================
rows = [("ML vote\nET + RF", ML, 77.6, 97.4),
        ("ML vote\nof 4", ML4, 69.5, 96.8),
        ("Neural net\nGRU", NN, 72.4,
         NN_STATS["grouping_contrast"]["window_accuracy"] * 100)]
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for k, (name, color, new, same) in enumerate(rows):
    ax.plot([new, same], [k, k], color=MUTED, linewidth=4,
            solid_capstyle="round", zorder=1)
    ax.scatter([same], [k], s=120, color=SURFACE, edgecolor=color,
               linewidth=2.5, zorder=3)
    ax.scatter([new], [k], s=120, color=color, zorder=3)
    ax.text(same + 1.6, k, f"{same:.1f}", va="center", fontsize=9.5,
            color=TEXT_2)
    ax.text(new - 1.6, k, f"{new:.1f}", va="center", ha="right", fontsize=9.5,
            color=TEXT_2)
    ax.text((new + same) / 2, k - 0.26, f"-{same - new:.0f} points",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_2)
ax.set_yticks(range(len(rows)), [r[0].replace("\n", " ") for r in rows])
ax.invert_yaxis()
ax.set_xlim(55, 105)
ax.set_ylim(len(rows) - 0.4, -0.7)
ax.set_xlabel("accuracy per window, %")
ax.set_title("What not having seen the person costs")
vgrid(ax)
ax.tick_params(axis="y", length=0)
ax.text(0.99, 0.04, "filled = a new person, never in training\n"
        "ring = the person IS in training, other recordings held out",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        color=TEXT_2)
note(fig, "Ring: what you see on yourself live (still optimistic -- clips "
          "from one sitting are close relatives, CLAUDE.md 8f). Filled: the "
          "person is held out entirely. The 20-27 point gap is the part of "
          "the signal that belongs to the individual (band placement, "
          "anatomy); more wearers shrink it. Axis starts at 55 %.")
save(fig, "generalisation_gap.png")

# ==========================================================================
# 10. the net: window length, spread across folds, what timing buys
# ==========================================================================
gru = sorted((r for r in nn_rows if r["config"]["kind"] == "gru"
              and r["stage"] == "N1"), key=lambda r: r["config"]["window_s"])
wl = [r["config"]["window_s"] for r in gru]
recall_by_w = np.array([row_pct(np.asarray(r["confusion_matrix"])).diagonal()
                        for r in gru])
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6),
                         gridspec_kw={"width_ratios": [1.15, 1, 1.1]})
ax = axes[0]
ax.imshow(recall_by_w.T, cmap=CMAP_NN, vmin=0, vmax=100, aspect="auto",
          interpolation="nearest")
for i in range(len(wl)):
    for g in range(N):
        v = recall_by_w[i, g]
        ax.text(i, g, f"{v:.0f}", ha="center", va="center", fontsize=9,
                color=ink(CMAP_NN, v))
ax.set_xticks(range(len(wl)), [f"{w_:.1f}" for w_ in wl])
ax.set_yticks(range(N), [SHORT.get(l, l) for l in LABELS], fontsize=8.5)
ax.set_xlabel("window length, s")
ax.set_title("Net: recall per gesture, by window")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

ax = axes[1]
for i, r in enumerate(gru):
    f = np.asarray(r["fold_accuracies"]) * 100
    ax.scatter([i] * len(f), f, s=26, color=NN, alpha=0.85, zorder=3)
    ax.plot([i - 0.25, i + 0.25], [f.mean()] * 2, color=TEXT, linewidth=2,
            zorder=4)
ax.set_xticks(range(len(wl)), [f"{w_:.1f}" for w_ in wl])
ax.set_xlabel("window length, s")
ax.set_ylabel("accuracy of one fold, %")
ax.set_ylim(20, 100)
ax.set_title("Net: spread across folds")
hgrid(ax)

ax = axes[2]
full = row_pct(CM_NN).diagonal()
dc = row_pct(CM_NN_DC).diagonal()
ys = np.arange(N)
ax.barh(ys - 0.19, full, height=0.34, color=NN, label="full window")
ax.barh(ys + 0.19, dc, height=0.34, color=MUTED, label="channel averages only")
for yy, v in zip(ys - 0.19, full):
    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=8, color=TEXT_2)
for yy, v in zip(ys + 0.19, dc):
    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=8, color=TEXT_2)
ax.set_yticks(ys, [SHORT.get(l, l) for l in LABELS], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("recall, %")
ax.set_title("Net: what the timing buys", pad=26)
ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
vgrid(ax)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
note(fig, "Neural net (GRU), person held out. Left and middle: the six window "
          "lengths of the comb test, ONE seed each. 0.7 s stands out "
          "(72 % against 65-68 % for every other length, and a weakest fold "
          "of 49 % against 34-43 %), but a jump that sharp between "
          "neighbouring lengths is also what seed-to-seed luck looks like; "
          "the seed spread was not measured. Right: the same net with every "
          "window replaced by its 8 channel averages. The timing is worth "
          "most on THUMB_UP (+20 points), REST (+26) and OPEN (+14): "
          "without it REST drifts to THUMB_UP and OPEN to WAVE_R. The waves "
          "are no better with it (WAVE_R is 8 points worse).")
save(fig, "nn_behaviour.png")
print("  net recall full", [round(v) for v in full], "dc", [round(v) for v in dc])
print("  net fold spread by window", [(w_, round(min(r["fold_accuracies"]) * 100),
                                      round(max(r["fold_accuracies"]) * 100))
                                     for w_, r in zip(wl, gru)])
print("done")

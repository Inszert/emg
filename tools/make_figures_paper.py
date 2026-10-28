"""
Figures added for publication.tex (new files; no existing figure is touched).

    python tools/make_figures_paper.py [OUT_DIR]          (default: docs/figures)

  channel_profiles.png   mean level of every channel for every gesture, one panel
                         per wearer, read from the recordings themselves
  generalisation_gap_paper.png
                         a person who is in the training data against a person
                         who is not, for the two recommended models; read from
                         results/training_history.json (the sweeps' own numbers)

Same palette as tools/make_figures.py: blue = the ML family, orange = the net.
The footnote of the older `generalisation_gap.png` points at CLAUDE.md, which a
paper cannot do, so this version has none.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, TEXT, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, MUTED = "#e4e3de", "#c9c8c2"
ML, NN = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]
SHORT = {"REST": "rest", "FIST": "fist", "OPEN": "open", "THUMB_UP": "thumb up",
         "WAVE_LEFT": "wave L", "WAVE_RIGHT": "wave R"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200})


def ink(value, norm):
    r, g, b, _a = CMAP(norm(value))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if lum < 0.18 else TEXT


# ---- 1. channel profiles --------------------------------------------------
levels = {}
for path in sorted(glob.glob(os.path.join(ROOT, "datasets", "*", "*", "*",
                                          "*_raw.npz"))):
    parts = os.path.relpath(path, os.path.join(ROOT, "datasets")).split(os.sep)
    if any(p.startswith("_") for p in parts[:-1]):
        continue
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["metadata_json"]))
    lo, hi = meta["label_start_index"], meta["label_end_index"]
    levels.setdefault((parts[0], str(z["label"])), []).append(
        z["emg"][lo:hi].astype(float))
people = sorted({k[0] for k in levels})
prof = {(p, g): np.concatenate(levels[(p, g)]).mean(0)
        for (p, g) in levels}
vals = np.array(list(prof.values()))
norm = LogNorm(vmin=10, vmax=float(vals.max()))

fig, axes = plt.subplots(1, len(people), figsize=(11.5, 2.9), sharey=True)
for ax, p in zip(axes, people):
    grid = np.array([prof[(p, g)] for g in GESTURES])
    ax.imshow(grid, cmap=CMAP, norm=norm, aspect="auto",
              interpolation="nearest")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.3, color=ink(grid[i, j], norm))
    ax.set_xticks(range(8), [str(k) for k in range(1, 9)], fontsize=7.5)
    ax.set_xlabel("channel")
    ax.set_title(p, pad=4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 8), minor=True)
    ax.set_yticks(np.arange(-.5, len(GESTURES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for tick in list(ax.xaxis.get_minor_ticks()) + list(
            ax.yaxis.get_minor_ticks()):
        tick.gridline.set_clip_on(False)
axes[0].set_yticks(range(len(GESTURES)), [SHORT[g] for g in GESTURES])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "channel_profiles.png"), bbox_inches="tight",
            pad_inches=0.1)
plt.close(fig)
print("wrote docs/figures/channel_profiles.png")

# ---- 2. the price of a new person ----------------------------------------
hist = json.load(open(os.path.join(ROOT, "results", "training_history.json"),
                      encoding="utf-8"))
ml_row = next(r for r in hist["classifier_sweeps"]
              if r["created"].startswith("2026-10-22T11:2")
              and r["model"] == "Vote-soft-2")
nn_row = next(r for r in hist["nn_sweeps"]
              if r["created"].startswith("2026-10-22T10:43"))
rows = [("ML vote (ET-500 + RF-500-leaf3)", ml_row["window_accuracy"],
         ml_row["contrast_accuracy"], ML),
        ("GRU, 0.7 s", nn_row["window_accuracy"], nn_row["contrast_accuracy"],
         NN)]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for k, (name, new, same, colour) in enumerate(rows):
    y = len(rows) - 1 - k
    ax.plot([new * 100, same * 100], [y, y], color=MUTED, linewidth=3,
            zorder=1)
    ax.scatter([new * 100], [y], s=70, color=colour, zorder=3)
    ax.scatter([same * 100], [y], s=70, color=SURFACE, edgecolor=colour,
               linewidth=2, zorder=3)
    ax.text(new * 100 - 1.5, y + 0.17, f"{new * 100:.1f}", ha="right",
            va="bottom", fontsize=8, color=TEXT_2)
    ax.text(same * 100 + 1.5, y + 0.17, f"{same * 100:.1f}", ha="left",
            va="bottom", fontsize=8, color=TEXT_2)
    ax.text(new * 100 + (same - new) * 50, y - 0.2,
            f"{(same - new) * 100:.0f} points", ha="center", va="top",
            fontsize=7.5, color=TEXT_2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7.5)
ax.set_xlim(60, 104)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("accuracy per window, %")
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.scatter([], [], s=40, color=TEXT_2, label="new wearer (held out)")
ax.scatter([], [], s=40, color=SURFACE, edgecolor=TEXT_2, linewidth=1.5,
           label="wearer in training (other recordings held out)")
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=6.5,
          ncol=1, handletextpad=0.2, labelspacing=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "generalisation_gap_paper.png"),
            bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("wrote docs/figures/generalisation_gap_paper.png",
      f"(ML {ml_row['window_accuracy']:.4f} / {ml_row['contrast_accuracy']:.4f},"
      f" GRU {nn_row['window_accuracy']:.4f} / {nn_row['contrast_accuracy']:.4f})")

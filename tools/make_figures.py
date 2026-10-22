"""
Report figures: confusion matrices, the PCA feature space, and the ML vs NN
comparison charts.

    python tools/make_figures.py [OUT_DIR]          (default: docs/figures)

Everything is read from what the project already saved -- nothing is
retrained:

  * the ML vote's numbers come from the comb-test run's stored out-of-fold
    probabilities (models/comb_test/<run>/ml_oof), the windows each model
    was scored on with that person held out;
  * the NN's from its stats sidecar (models/gesture_nn.json) and the
    comb-test NN log (nn_results.jsonl);
  * speed from tools/bench_models.py's output (models/reports/bench_*.json);
  * the PCA is computed here from the dataset, at the deployed setup and on
    the training calibration -- unsupervised, labels only colour the dots.

Light theme for print. One fixed colour per model across every figure.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "training"))

import numpy as np                                               # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap            # noqa: E402

RUN = os.path.join(PROJECT, "models", "comb_test", "20261022-003918")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT, "docs",
                                                          "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (dataviz reference instance, light mode) -------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
MUTED = "#c9c8c2"
ML = "#2a78d6"          # slot 1, blue   -- the ML vote, everywhere
NN = "#eb6834"          # slot 2, orange -- the neural net, everywhere
ML4 = "#1baf7a"         # slot 3, aqua   -- the 4-model vote in use
LIGHT_BLUE = "#86b6ef"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

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


def note(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8.5,
             color=TEXT_2, wrap=True)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---- data -----------------------------------------------------------------
SETUP = "bandpass-level-w1-o0.75"
oof = os.path.join(RUN, "ml_oof", SETUP + "__")
data = np.load(oof + "data.npz", allow_pickle=True)
Y = data["y"]
PERSONS = data["persons"].astype(str)
LABELS = [str(x) for x in data["labels"]]
SHORT = {"WAVE_LEFT": "WAVE_L", "WAVE_RIGHT": "WAVE_R", "THUMB_UP": "THUMB"}


def proba(name):
    return np.load(oof + name + ".npz")["proba"]


P_VOTE = (proba("ET-500") + proba("RF-500-leaf3")) / 2
ml_result = json.load(open(os.path.join(RUN, "ml_result.json"),
                           encoding="utf-8"))
nn_side = json.load(open(os.path.join(PROJECT, "models", "gesture_nn.json"),
                         encoding="utf-8"))
NN_STATS = nn_side["stats"]
nn_rows = [json.loads(l) for l in open(os.path.join(RUN, "nn_results.jsonl"),
                                       encoding="utf-8") if l.strip()]


def confusion(y, pred, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


# ==========================================================================
# 1. confusion matrices
# ==========================================================================
def draw_confusion(ax, cm, labels, title):
    rows = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / rows * 100
    ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100)
    n = len(labels)
    for i in range(n):
        for j in range(n):
            dark = pct[i, j] > 55
            ax.text(j, i - 0.08, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i == j else "normal",
                    color="white" if dark else TEXT)
            ax.text(j, i + 0.25, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=7, color="white" if dark else TEXT_2)
    names = [SHORT.get(l, l) for l in labels]
    ax.set_xticks(range(n), names, rotation=0, fontsize=8.5)
    ax.set_yticks(range(n), names, fontsize=8.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual gesture")
    ax.set_title(title, pad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


cm_ml = confusion(Y, P_VOTE.argmax(1), len(LABELS))
nn_labels = list(NN_STATS["labels"])
cm_nn = np.asarray(NN_STATS["confusion_matrix"])

for cm, labels, title, name in (
        (cm_ml, LABELS, "ML soft vote (ExtraTrees + RandomForest)",
         "confusion_ml_vote.png"),
        (cm_nn, nn_labels, "Neural net (GRU, 0.7 s)", "confusion_nn.png")):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    draw_confusion(ax, cm, labels, title)
    note(fig, "Rows: what was made; columns: what the model said. Big number "
              "= % of that row, small = window count. Each person held out "
              "in turn (trained on the other four).")
    save(fig, name)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
draw_confusion(axes[0], cm_ml, LABELS, "ML soft vote (ET + RF)")
draw_confusion(axes[1], cm_nn, nn_labels, "Neural net (GRU 0.7 s)")
note(fig, "Both scored with each person held out. FIST, OPEN and THUMB_UP "
          "are where both models lose their points; the waves and REST are "
          "easy for both.")
save(fig, "confusion_both.png")

# ==========================================================================
# 2. PCA feature space (unsupervised; colour = label, after the fact)
# ==========================================================================
print("  building the deployed setup's feature matrix for the PCA ...")
import emg_dataset_tool as T                                     # noqa: E402
import train_model as TM                                         # noqa: E402

root = os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
recs, _dropped = TM.load_recordings(root, exclude_codes=())
people = sorted({r.person_id for r in recs})
cal, _notes, _used = T.training_calibrations(root, "affine", recs, people)
built = TM.build_dataset(recs, 1.0, 0.75, "bandpass", False, "level",
                         dedupe=True, calibration=cal)
X, y_pca, rec_ids, pca_labels, _meta = built
persons_pca = TM.person_vector(recs, rec_ids).astype(str)
Z = (X - X.mean(0)) / X.std(0).clip(min=1e-9)
_u, s, vt = np.linalg.svd(Z, full_matrices=False)
XY = Z @ vt[:2].T
var = s ** 2 / np.sum(s ** 2) * 100
lim_x = np.percentile(XY[:, 0], [0.5, 99.5])
lim_y = np.percentile(XY[:, 1], [0.5, 99.5])


def pca_grid(groups, values, title, name, ncols, caption):
    rows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(3.4 * ncols, 3.2 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, groups):
        m = values == g
        ax.scatter(XY[~m, 0], XY[~m, 1], s=3, color=MUTED, alpha=0.35,
                   linewidths=0, rasterized=True)
        ax.scatter(XY[m, 0], XY[m, 1], s=6, color=ML, alpha=0.75,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{g}  ({m.sum()})", fontsize=10.5)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.supxlabel(f"PC 1 ({var[0]:.0f} % of variance)", color=TEXT_2,
                  fontsize=9.5)
    fig.supylabel(f"PC 2 ({var[1]:.0f} % of variance)", color=TEXT_2,
                  fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    note(fig, caption)
    save(fig, name)


pca_grid(pca_labels, np.asarray(pca_labels)[y_pca],
         "Feature space (PCA) -- by gesture", "pca_by_gesture.png", 3,
         f"Every training window ({len(XY)}) as one dot: the deployed setup's "
         "50 features (bandpass, level, 1.0 s, calibrated), projected to 2-D "
         "by PCA, which never sees the labels. Each panel highlights one "
         "gesture in blue over all the others in grey. Separate clusters = "
         "the features tell the gestures apart.")
pca_grid(people, persons_pca, "Feature space (PCA) -- by person",
         "pca_by_person.png", 3,
         "The same projection, one person highlighted per panel. Everyone "
         "spreads over the same shared clusters -- calibration puts people on "
         "one scale -- and the differences are in where each person's "
         "gestures land within them. 2-D shows only 61 % of the variance, so "
         "Barbora's difficulty is not visible here.")

# ==========================================================================
# 3. accuracy per person and per gesture
# ==========================================================================
ml_person = ml_result["winner"]["per_subject"]
nn_person = {}
for line in open(os.path.join(RUN, "log_final_nn_gru07_2.txt"),
                 encoding="utf-8", errors="replace"):
    if "held out " in line and "%" in line:
        bits = line.split()
        nn_person[bits[2]] = float(bits[3].rstrip("%")) / 100


def grouped_hbars(ax, cats, series, xmax=100):
    n = len(series)
    h = 0.78 / n
    for k, (label, color, vals) in enumerate(series):
        ys = np.arange(len(cats)) + (k - (n - 1) / 2) * h
        bars = ax.barh(ys, [v * 100 for v in vals], height=h * 0.86,
                       color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    f"{v * 100:.0f}%", va="center", fontsize=8.5,
                    color=TEXT_2)
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("accuracy per window, %")
    vgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series),
              fontsize=9)


fig, ax = plt.subplots(figsize=(7.5, 4.6))
grouped_hbars(ax, people, [
    ("ML vote (ET + RF)", ML, [ml_person[p] for p in people]),
    ("Neural net (GRU)", NN, [nn_person.get(p, np.nan) for p in people])],
    xmax=105)
ax.set_title("Accuracy on a person the model never saw", pad=28)
note(fig, "Each person held out in turn; trained on the other four, who "
          "calibrated first. Barbora is the hard case for every model.")
save(fig, "accuracy_per_person.png")

recall_ml = np.diag(cm_ml) / cm_ml.sum(1)
recall_nn = np.diag(cm_nn) / cm_nn.sum(1)
order = [nn_labels.index(l) for l in LABELS]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
grouped_hbars(ax, LABELS, [
    ("ML vote (ET + RF)", ML, list(recall_ml)),
    ("Neural net (GRU)", NN, list(recall_nn[order]))], xmax=108)
ax.set_title("Accuracy per gesture (person held out)", pad=28)
ax.set_xlabel("% of that gesture's windows recognised")
note(fig, "FIST, OPEN and THUMB_UP use overlapping forearm muscles and sit "
          "close together on an 8-pod band; the waves and REST are easy.")
save(fig, "accuracy_per_gesture.png")

# ==========================================================================
# 4. every ML model on the winning setup
# ==========================================================================
rows = sorted((r for r in ml_result["singles"] if r["setup_name"] == SETUP),
              key=lambda r: r["window_accuracy"])
fig, ax = plt.subplots(figsize=(7.5, 5.8))
names = [r["model"]["name"] for r in rows]
vals = [r["window_accuracy"] * 100 for r in rows]
forest = [r["model"]["kind"] in ("extra_trees", "random_forest") for r in rows]
bars = ax.barh(range(len(rows)), vals, height=0.7,
               color=[ML if f else LIGHT_BLUE for f in forest])
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8.5, color=TEXT_2)
ax.set_yticks(range(len(rows)), names, fontsize=9)
ax.set_xlim(50, 82)
ax.set_xlabel("accuracy per window, % (person held out)")
ax.set_title("Single ML models on the winning setup")
vgrid(ax)
ax.tick_params(axis="y", length=0)
note(fig, "bandpass signal, level features, 1.0 s windows. Dark blue = the "
          "tree forests, about 10 points ahead of everything else. Axis "
          "starts at 50 %.")
save(fig, "ml_single_models.png")

# ==========================================================================
# 5. soft voting -- what it adds and what it costs
# ==========================================================================
def pooled(names_):
    p = sum(proba(n) for n in names_) / len(names_)
    return float((p.argmax(1) == Y).mean() * 100)


combos = [("ExtraTrees alone", ["ET-500"]),
          ("ET + RF  (the pick)", ["ET-500", "RF-500-leaf3"]),
          ("ET + RF + LDA", ["ET-500", "RF-500-leaf3", "LDA-shrinkage"]),
          ("ET + RF + SVM + LDA  (in use)",
           ["ET-500", "RF-500-leaf3", "SVM-rbf-C10", "LDA-shrinkage"])]
vals = [pooled(c) for _, c in combos]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
colors = [LIGHT_BLUE, ML, ML4, ML4]
bars = ax.barh(range(len(combos)), vals, height=0.62, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=9, color=TEXT_2)
ax.set_yticks(range(len(combos)), [c for c, _ in combos])
ax.invert_yaxis()
ax.set_xlim(60, 84)
ax.set_xlabel("accuracy, % of all windows (person held out)")
ax.set_title("Soft voting: two forests = one forest; weaker members cost ~8 pts")
vgrid(ax)
ax.tick_params(axis="y", length=0)
note(fig, "Unweighted soft vote = mean of the members' probabilities. "
          "Scored on the same stored out-of-fold windows. Axis starts at 60 %.")
save(fig, "soft_voting.png")

# ==========================================================================
# 6. window length
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
for fs, color in (("level", ML), ("both", NN), ("classic", ML4)):
    xs = sorted({r["setup"]["window_s"] for r in ml_result["singles"]
                 if r["setup"]["feature_set"] == fs})
    ys = [max(r["window_accuracy"] for r in ml_result["singles"]
              if r["setup"]["feature_set"] == fs
              and r["setup"]["window_s"] == w) * 100 for w in xs]
    ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=fs)
    ax.annotate(fs, (xs[-1], ys[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=TEXT_2)
ax.set_title("ML: best model per window length")
ax.set_xlabel("window length, s")
ax.set_ylabel("accuracy per window, %")
ax.legend(title="feature set", fontsize=8.5, title_fontsize=8.5)
hgrid(ax)
ax = axes[1]
for kind, color in (("gru", NN), ("cnn_gru", LIGHT_BLUE)):
    pts = sorted((r["config"]["window_s"], r["window_accuracy"] * 100)
                 for r in nn_rows if r["stage"] == "N1"
                 and r["config"]["kind"] == kind)
    if pts:
        ax.plot(*zip(*pts), "-o", color=color, linewidth=2, markersize=6,
                label=kind)
ax.set_title("Neural net: per window length (150 epochs)")
ax.set_xlabel("window length, s")
ax.set_ylabel("accuracy per window, %")
ax.legend(fontsize=8.5)
hgrid(ax)
fig.tight_layout()
note(fig, "Person held out. The ML models want long windows (1.0-1.3 s); "
          "the GRU peaks at 0.7 s.")
save(fig, "window_length.png")

# ==========================================================================
# 7. the two models side by side: accuracy, speed, size
# ==========================================================================
bench_files = sorted(glob.glob(os.path.join(PROJECT, "models", "reports",
                                            "bench_models_*.json")))
bench = json.load(open(bench_files[-1], encoding="utf-8")) if bench_files \
    else None
nn_gpu = next((n for n in (bench or {}).get("nn", [])
               if n.get("device") == "cuda"), None)
w = ml_result["winner"]
panels = [
    ("accuracy per window, %", [w["window_accuracy"] * 100, 69.5,
                                NN_STATS["window_accuracy"] * 100], False),
    ("accuracy per gesture, %", [w["recording_vote_accuracy"] * 100, 73.7,
                                 NN_STATS["recording_vote_accuracy"] * 100],
     False),
]
if bench:
    members = bench["ml"]["members"]
    pair_ms = (members["RF-500-leaf3"]["median_ms"]
               + members["ET-500"]["median_ms"]
               + bench["ml"]["features_only"]["median_ms"])
    panels.append(("time per live window, ms (log)",
                   [pair_ms, bench["ml"]["whole_prediction"]["median_ms"],
                    nn_gpu["round_trip"]["median_ms"] if nn_gpu else np.nan],
                   True))
size_ml = os.path.getsize(os.path.join(PROJECT, "models",
                                       "gesture_model.joblib")) / 1024
size_nn = os.path.getsize(os.path.join(PROJECT, "models",
                                       "gesture_nn.pt")) / 1024
panels.append(("size on disk, KB (log)", [size_ml, size_ml, size_nn], True))
names3 = ["ML vote\nET + RF", "ML vote\nof 4", "NN\nGRU"]
fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.8))
for ax, (title, vals, log) in zip(axes, panels):
    bars = ax.bar(range(3), vals, width=0.62, color=[ML, ML4, NN])
    for b, v in zip(bars, vals):
        txt = (f"{v:.1f}" if v < 100 else f"{v:,.0f}")
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() * (1.12 if log else 1.0) + (0 if log else 1),
                txt, ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_xticks(range(3), names3, fontsize=8.5)
    ax.set_title(title, fontsize=10)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min(vals) / 3, max(vals) * 6)
    else:
        ax.set_ylim(0, 100)
    hgrid(ax)
    ax.tick_params(axis="x", length=0)
fig.suptitle("ML vote vs neural net", x=0.01, ha="left", fontweight="bold",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
note(fig, "Accuracy with a person held out. Time: median per live window on "
          "this PC (ML on the CPU, NN round trip to its GPU worker). Sizes "
          "measured from the files; the ET + RF pair is within 1 % of the "
          "4-model file. Log scales on the right.")
save(fig, "ml_vs_nn.png")

# ==========================================================================
# 8. DC-only control
# ==========================================================================
dc_nn = NN_STATS.get("dc_only_control") or {}
dc_nn_acc = (dc_nn.get("window_accuracy") if isinstance(dc_nn, dict)
             else None) or 0.633
fig, ax = plt.subplots(figsize=(7.5, 3.2))
cats = ["ML vote (ET + RF)", "Neural net (GRU)"]
full = [w["window_accuracy"] * 100, NN_STATS["window_accuracy"] * 100]
dc = [77.8, dc_nn_acc * 100]
ys = np.arange(2)
ax.barh(ys - 0.18, full, height=0.32, color=[ML, NN], label="full input")
ax.barh(ys + 0.18, dc, height=0.32, color=MUTED,
        label="only the 8 channel averages")
for yy, v in zip(ys - 0.18, full):
    ax.text(v + 0.6, yy, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_2)
for yy, v in zip(ys + 0.18, dc):
    ax.text(v + 0.6, yy, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_2)
ax.set_yticks(ys, cats)
ax.invert_yaxis()
ax.set_xlim(50, 85)
ax.set_xlabel("accuracy per window, % (person held out)")
ax.set_title("What each model reads: full input vs channel averages only")
ax.legend(loc="lower right", fontsize=8.5)
vgrid(ax)
ax.tick_params(axis="y", length=0)
note(fig, "DC-only control: the same model given each window's 8 channel "
          "means only. ML: no difference -- it reads activation levels. NN: "
          "+9 points from the signal's timing. Axis starts at 50 %.")
save(fig, "dc_only_control.png")

print("done")

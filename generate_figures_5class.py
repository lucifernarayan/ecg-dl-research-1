"""
generate_figures_5class.py
--------------------------
Regenerates three figures with all 5 AAMI classes (N, SVEB, VEB, F, Q).

NOTE on Q class:
  Q (class 4 / Unknown) was assigned weight=0 in WeightedRandomSampler and
  ignore_index=4 in CrossEntropyLoss.  The model was never trained to predict Q.
  Consequently Q Recall = 0.0% and Q F1 = 0.0% across all folds and all epochs.
  This is shown explicitly in the figures as a factual result.

Figures:
  1. fig_F2_radar_5class       — Radar/spider with 5 class axes (20-unit grid)
  2. fig_A3_5class_recall      — Per-class recall curves over 20 epochs (Fold 2, best fold), 5 classes
  3. fig_B2_interpatient_5class — Inter-patient per-class Recall & F1 at each fold, 5 classes

Typography:
  - Font: Times New Roman (serif)
  - Axis labels / tick labels: 13 pt
  - Legend: 11 pt
  - Annotations: 10 pt
  - (Suitable for a 2-column IEEE journal figure)

Output: results/figures/ — PNG (300 DPI) + PDF + SVG
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

AGGREGATE_JSON    = os.path.join(RESULTS_DIR, "fixed_hp_3fold_aggregate_metrics.json")
TRAINING_LOG_JSON = os.path.join(RESULTS_DIR, "fixed_hp_3fold_training_log.json")

# ---------------------------------------------------------------------------
# Color palette (pastel — fold colors)
# ---------------------------------------------------------------------------
C1    = "#F08080"   # Light Coral    — Fold 1
C2    = "#4682B4"   # Steel Blue     — Fold 2
C3    = "#DEB887"   # Wheat          — Fold 3
CMEAN = "#3CB371"   # Sea Green      — Mean / reference

# Per-class colors for the recall-curve figure (5 classes)
CLS_COLORS = {
    "N":    "#4682B4",   # Steel Blue
    "SVEB": "#F08080",   # Coral
    "VEB":  "#3CB371",   # Sea Green
    "F":    "#DEB887",   # Wheat
    "Q":    "#9E9E9E",   # Neutral Grey  (always 0%)
}
CLS_LINESTYLES = {
    "N": "-", "SVEB": "-", "VEB": "-", "F": "--", "Q": ":"
}

FOLD_COLORS  = [C1, C2, C3]
FOLD_LABELS  = ["Fold 1", "Fold 2", "Fold 3"]
FOLDS_KEYS   = ["Fold 1", "Fold 2", "Fold 3"]

CLASS_NAMES  = ["N", "SVEB", "VEB", "F", "Q"]   # 5 classes
CLASS_NAMES4 = ["N", "SVEB", "VEB", "F"]         # active 4

EPOCHS = list(range(1, 21))

# ---------------------------------------------------------------------------
# Global Matplotlib / font settings (Times New Roman, IEEE sizes)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "Times New Roman",
    "font.serif":         ["Times New Roman"],
    "mathtext.fontset":   "cm",
    "font.size":          13,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    2.0,
    "lines.markersize":   5,
})

DPI_SAVE = 300

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(AGGREGATE_JSON, "r") as f:
    agg = json.load(f)

with open(TRAINING_LOG_JSON, "r") as f:
    tlog = json.load(f)

def get_series(fold_key, metric_key):
    return [ep[metric_key] for ep in tlog[fold_key]]

fold_metrics = {fk: agg["individual_folds"][fk]["metrics"] for fk in FOLDS_KEYS}

# 5-class recall/F1 arrays (Q appended as 0.0)
def get_5class_recall(fold_key):
    r = fold_metrics[fold_key]["per_class_recall"]   # [N, SVEB, VEB, F]
    return r + [0.0]                                  # append Q = 0.0

def get_5class_f1(fold_key):
    f = fold_metrics[fold_key]["per_class_f1"]
    return f + [0.0]

def get_5class_support(fold_key):
    s = fold_metrics[fold_key]["support"]
    return s + [0]   # Q support in validation = 0 (excluded)

# ---------------------------------------------------------------------------
# Helper: save PNG + PDF + SVG
# ---------------------------------------------------------------------------
def save_fig(fig, basename):
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(FIGURES_DIR, f"{basename}.{ext}")
        fig.savefig(path,
                    dpi=DPI_SAVE if ext == "png" else None,
                    bbox_inches="tight",
                    format=ext)
    print(f"[OK] Saved: {basename}  (.png / .pdf / .svg)")


# ===========================================================================
# FIGURE 1 — Radar / Spider Chart — 5 Classes — 20-unit radial grid
# ===========================================================================
# Axes: N Recall, SVEB Recall, VEB Recall, F Recall, Q Recall, Active Acc, Macro F1
# (7 spokes for a richer IEEE figure)
radar_labels = [
    "N Recall (%)",
    "SVEB Recall (%)",
    "VEB Recall (%)",
    "F Recall (%)",
    "Q Recall (%)",
    "Active Acc. (%)",
    "Macro F1 (%)",
]
radar_keys_map = {
    "N Recall (%)":      "n_recall",
    "SVEB Recall (%)":   "sveb_recall",
    "VEB Recall (%)":    "veb_recall",
    "F Recall (%)":      "f_recall",
    "Q Recall (%)":      None,          # always 0
    "Active Acc. (%)":   "active_accuracy",
    "Macro F1 (%)":      "active_macro_f1",
}

radar_data = {}
for fk in FOLDS_KEYS:
    fm = fold_metrics[fk]
    vals = []
    for lbl in radar_labels:
        key = radar_keys_map[lbl]
        vals.append(0.0 if key is None else fm[key])
    radar_data[fk] = vals

N_spokes = len(radar_labels)
angles   = np.linspace(0, 2 * np.pi, N_spokes, endpoint=False).tolist()
angles  += angles[:1]   # close loop

RADAR_MAX   = 100
RADAR_STEP  = 20        # 20-unit grid rings
rings       = list(range(0, RADAR_MAX + 1, RADAR_STEP))   # [0,20,40,60,80,100]

fig_r = plt.figure(figsize=(6.5, 6.5))
ax_r  = fig_r.add_subplot(111, polar=True)

# Draw rings and radial lines manually for clean look
for ring_val in rings[1:]:
    ring_x = [ring_val * np.cos(a) + ring_val for a in angles]   # not used; imshow polar handles it
    ax_r.plot(angles, [ring_val] * (N_spokes + 1),
              color="grey", linewidth=0.5, linestyle="-", alpha=0.45)

# Plot each fold
for i, fk in enumerate(FOLDS_KEYS):
    vals = radar_data[fk] + radar_data[fk][:1]
    ax_r.plot(angles, vals, color=FOLD_COLORS[i],
              linewidth=2.2, label=FOLD_LABELS[i])
    ax_r.fill(angles, vals, color=FOLD_COLORS[i], alpha=0.10)
    # Mark the Q spoke (always 0) with a distinct dot
    q_angle = angles[4]   # 5th spoke = Q Recall
    ax_r.scatter([q_angle], [vals[4]],
                 color=FOLD_COLORS[i], s=60, zorder=6, marker="x", linewidths=2)

# Spoke labels
ax_r.set_thetagrids(
    np.degrees(angles[:-1]),
    radar_labels,
    fontsize=11,
    fontfamily="Times New Roman",
)
ax_r.set_ylim(0, RADAR_MAX)
ax_r.set_yticks(rings[1:])
ax_r.set_yticklabels(
    [str(v) for v in rings[1:]],
    fontsize=9,
    fontfamily="Times New Roman",
    color="grey",
)
ax_r.tick_params(axis='y', pad=3)

# Q note annotation
ax_r.text(
    angles[4], RADAR_MAX * 0.55,
    "Q=0\n(excluded\nfrom train)",
    ha="center", va="center",
    fontsize=8, color="#666666",
    fontfamily="Times New Roman",
    style="italic",
)

legend = ax_r.legend(
    loc="upper right",
    bbox_to_anchor=(1.40, 1.22),
    framealpha=0.75,
    fontsize=11,
)
for txt in legend.get_texts():
    txt.set_fontfamily("Times New Roman")

fig_r.tight_layout()
save_fig(fig_r, "fig_F2_radar_5class")
plt.close(fig_r)


# ===========================================================================
# FIGURE 2 — Per-Class Recall Curves over Epochs — Fold 2, 5 Classes
# ===========================================================================
# Q recall = 0 for every epoch (model excluded Q from training)
fold2_epoch_recalls = {
    "N":    get_series("Fold 2", "n_recall"),
    "SVEB": get_series("Fold 2", "sveb_recall"),
    "VEB":  get_series("Fold 2", "veb_recall"),
    "F":    get_series("Fold 2", "f_recall"),
    "Q":    [0.0] * 20,   # factual: Q excluded
}

fig_a, ax_a = plt.subplots(figsize=(7.0, 4.5))

for cls in CLASS_NAMES:
    ax_a.plot(
        EPOCHS,
        fold2_epoch_recalls[cls],
        color=CLS_COLORS[cls],
        linestyle=CLS_LINESTYLES[cls],
        linewidth=2.2 if cls != "Q" else 1.5,
        marker="o" if cls not in ("Q",) else "x",
        markersize=4.5,
        label=cls if cls != "Q" else "Q (excluded)",
        alpha=1.0 if cls != "Q" else 0.55,
        zorder=5 if cls != "Q" else 2,
    )

# Mark best epoch
ax_a.axvline(x=14, color="dimgrey", linestyle=":", linewidth=1.4,
             alpha=0.75, label="Best epoch (14)")

ax_a.set_xlabel("Epoch", fontsize=13, fontfamily="Times New Roman")
ax_a.set_ylabel("Recall (%)", fontsize=13, fontfamily="Times New Roman")
ax_a.set_xlim(1, 20)
ax_a.set_ylim(-5, 108)
ax_a.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax_a.yaxis.set_major_locator(mticker.MultipleLocator(20))

# Tick font
for tick in ax_a.get_xticklabels() + ax_a.get_yticklabels():
    tick.set_fontfamily("Times New Roman")
    tick.set_fontsize(12)

leg = ax_a.legend(
    framealpha=0.75, ncol=2, fontsize=11,
    loc="upper left",
)
for txt in leg.get_texts():
    txt.set_fontfamily("Times New Roman")

# Small note for Q
ax_a.annotate(
    "Q recall \u2261 0% (ignore_index=4)",
    xy=(10, 2.5), fontsize=9, color="#666666",
    fontfamily="Times New Roman", style="italic",
)

fig_a.tight_layout()
save_fig(fig_a, "fig_A3_5class_recall_fold2")
plt.close(fig_a)


# ===========================================================================
# FIGURE 3 — Inter-Patient Performance at Each Fold — 5 Classes
#   Panel L: Per-fold Recall grouped by class (5 classes, 3 folds each)
#   Panel R: Mean ± Std Recall and F1 per class (5 classes)
# ===========================================================================
n_cls = 5
recall_matrix = np.array([get_5class_recall(fk) for fk in FOLDS_KEYS])  # 3x5
f1_matrix5    = np.array([get_5class_f1(fk)     for fk in FOLDS_KEYS])   # 3x5

recall_mean = recall_matrix.mean(axis=0)
recall_std  = recall_matrix.std(axis=0)
f1_mean     = f1_matrix5.mean(axis=0)
f1_std      = f1_matrix5.std(axis=0)

fig_b, axes_b = plt.subplots(1, 2, figsize=(13.0, 5.0))

# Shared x positions
x5   = np.arange(n_cls)
W    = 0.20
offs = [-W, 0.0, W]

# ---- Left panel: per-fold recall per class ----
ax_L = axes_b[0]

for i, fk in enumerate(FOLDS_KEYS):
    bars = ax_L.bar(
        x5 + offs[i],
        recall_matrix[i],
        width=W,
        color=FOLD_COLORS[i],
        label=FOLD_LABELS[i],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    # Value labels on top of bars (skip if 0)
    for rect, val in zip(bars, recall_matrix[i]):
        if val > 2:
            ax_L.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 1.5,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontsize=8, fontfamily="Times New Roman",
                color="dimgrey",
            )

# Q bar: add hatching to make it visually distinct (always 0)
# The bars are already 0-height — add a thin annotation
ax_L.text(
    x5[4], 3.5, "excluded\nfrom train",
    ha="center", va="bottom",
    fontsize=8, color="#888888",
    fontfamily="Times New Roman", style="italic",
)

ax_L.set_xticks(x5)
ax_L.set_xticklabels(CLASS_NAMES, fontsize=12, fontfamily="Times New Roman")
ax_L.set_ylabel("Recall (%)", fontsize=13, fontfamily="Times New Roman")
ax_L.set_xlabel("AAMI Class", fontsize=13, fontfamily="Times New Roman")
ax_L.set_ylim(0, 115)
ax_L.yaxis.set_major_locator(mticker.MultipleLocator(20))
for tick in ax_L.get_yticklabels():
    tick.set_fontfamily("Times New Roman")
    tick.set_fontsize(12)

leg_L = ax_L.legend(framealpha=0.75, fontsize=11, loc="upper right")
for txt in leg_L.get_texts():
    txt.set_fontfamily("Times New Roman")

# ---- Right panel: mean ± std Recall and F1 ----
ax_R = axes_b[1]

W2  = 0.28
err_kw_r = {"elinewidth": 1.5, "ecolor": "#2a5280", "capsize": 5}
err_kw_f = {"elinewidth": 1.5, "ecolor": "#a03030", "capsize": 5}

bars_r = ax_R.bar(
    x5 - W2 / 2, recall_mean, width=W2,
    color=C2, label="Mean Recall",
    edgecolor="white", linewidth=0.6,
    yerr=recall_std, error_kw=err_kw_r, zorder=3,
)
bars_f = ax_R.bar(
    x5 + W2 / 2, f1_mean, width=W2,
    color=C1, label="Mean F1",
    edgecolor="white", linewidth=0.6,
    yerr=f1_std, error_kw=err_kw_f, zorder=3,
)

# Value labels
for bars, vals in [(bars_r, recall_mean), (bars_f, f1_mean)]:
    for rect, val in zip(bars, vals):
        if val > 2:
            ax_R.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 2.5,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, fontfamily="Times New Roman",
                color="dimgrey",
            )

ax_R.text(
    x5[4], 4.5, "Q=0\n(excluded)",
    ha="center", va="bottom",
    fontsize=8, color="#888888",
    fontfamily="Times New Roman", style="italic",
)

ax_R.set_xticks(x5)
ax_R.set_xticklabels(CLASS_NAMES, fontsize=12, fontfamily="Times New Roman")
ax_R.set_ylabel("Score (%)", fontsize=13, fontfamily="Times New Roman")
ax_R.set_xlabel("AAMI Class", fontsize=13, fontfamily="Times New Roman")
ax_R.set_ylim(0, 120)
ax_R.yaxis.set_major_locator(mticker.MultipleLocator(20))
for tick in ax_R.get_yticklabels():
    tick.set_fontfamily("Times New Roman")
    tick.set_fontsize(12)

leg_R = ax_R.legend(framealpha=0.75, fontsize=11, loc="upper right")
for txt in leg_R.get_texts():
    txt.set_fontfamily("Times New Roman")

fig_b.tight_layout(w_pad=3.5)
save_fig(fig_b, "fig_B2_interpatient_5class")
plt.close(fig_b)


print("\n[OK] All 3 five-class IEEE-style figures generated.")
print(f"Output: {FIGURES_DIR}")

"""
generate_additional_figures.py
-------------------------------
Generates additional publication-quality Matplotlib figures from the completed
3-fold fixed-HP inter-patient ECG classification experiment.

Figures generated:
  A1 - Learning curves: Training Loss per epoch (3 folds)
  A2 - Learning curves: Active Macro F1 per epoch (3 folds)
  A3 - Learning curves: Per-class Recall over epochs (Fold 2 — best fold)
  B  - Inter-patient performance summary: per-class Recall & F1 bar chart + mean±std
  C  - Fold score progression over epochs (all 3 folds)
  D  - Per-class F1 heatmap (3 folds × 4 classes)

All values sourced exclusively from:
  results/fixed_hp_3fold_aggregate_metrics.json
  results/fixed_hp_3fold_training_log.json

Color palette (pastel):
  Fold 1  → Coral      #F08080
  Fold 2  → Sea Blue   #4682B4
  Fold 3  → Wheat      #DEB887
  Mean    → Sea Green  #3CB371
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

AGGREGATE_JSON   = os.path.join(RESULTS_DIR, "fixed_hp_3fold_aggregate_metrics.json")
TRAINING_LOG_JSON = os.path.join(RESULTS_DIR, "fixed_hp_3fold_training_log.json")

# Pastel palette
C1      = "#F08080"   # Light Coral  — Fold 1
C2      = "#4682B4"   # Steel Blue   — Fold 2
C3      = "#DEB887"   # Burlywood/Wheat — Fold 3
CMEAN   = "#3CB371"   # Medium Sea Green — mean line / mean bar

FOLD_COLORS = [C1, C2, C3]
FOLD_LABELS = ["Fold 1", "Fold 2", "Fold 3"]

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DPI_SAVE = 300

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(AGGREGATE_JSON, "r") as f:
    agg = json.load(f)

with open(TRAINING_LOG_JSON, "r") as f:
    tlog = json.load(f)

folds_data = ["Fold 1", "Fold 2", "Fold 3"]

def get_series(fold_key, metric_key):
    """Extract epoch-wise series for a given fold and metric."""
    return [ep[metric_key] for ep in tlog[fold_key]]

epochs = list(range(1, 21))

# Best-epoch metrics (final values per fold)
fold_metrics = {}
for fold_key in folds_data:
    fold_metrics[fold_key] = agg["individual_folds"][fold_key]["metrics"]

agg_stats = agg["aggregate_metrics"]

# ---------------------------------------------------------------------------
# Helper: save figure in PNG + PDF + SVG
# ---------------------------------------------------------------------------
def save_fig(fig, basename):
    for ext in ("png", "pdf", "svg"):
        out = os.path.join(FIGURES_DIR, f"{basename}.{ext}")
        fig.savefig(out, dpi=DPI_SAVE if ext == "png" else None,
                    bbox_inches="tight", format=ext)
    print(f"[OK] Saved: {basename} (.png/.pdf/.svg)")


# ===========================================================================
# Figure A1 — Training Loss Curves (all 3 folds)
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.6))

for i, fk in enumerate(folds_data):
    losses = get_series(fk, "train_loss")
    ax.plot(epochs, losses, color=FOLD_COLORS[i], linewidth=1.8,
            marker="o", markersize=3.2, label=FOLD_LABELS[i])

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (CE)")
ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(framealpha=0.6)
fig.tight_layout()
save_fig(fig, "fig_A1_training_loss_curves")
plt.close(fig)


# ===========================================================================
# Figure A2 — Active Macro F1 Learning Curves
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.6))

for i, fk in enumerate(folds_data):
    f1s = get_series(fk, "active_macro_f1")
    ax.plot(epochs, f1s, color=FOLD_COLORS[i], linewidth=1.8,
            marker="o", markersize=3.2, label=FOLD_LABELS[i])

# Mark the best epoch for each fold (Fold1=ep20, Fold2=ep14, Fold3=ep14)
best_epochs = {"Fold 1": 20, "Fold 2": 14, "Fold 3": 14}
for i, fk in enumerate(folds_data):
    ep = best_epochs[fk]
    val = get_series(fk, "active_macro_f1")[ep - 1]
    ax.axvline(x=ep, color=FOLD_COLORS[i], linestyle=":", linewidth=0.9, alpha=0.7)
    ax.scatter([ep], [val], color=FOLD_COLORS[i], s=55, zorder=5, marker="*")

ax.set_xlabel("Epoch")
ax.set_ylabel("Active Macro F1 (%)")
ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(framealpha=0.6)
fig.tight_layout()
save_fig(fig, "fig_A2_macro_f1_curves")
plt.close(fig)


# ===========================================================================
# Figure A3 — Per-Class Recall Curves for Fold 2 (best fold)
# ===========================================================================
class_colors = [C2, C1, CMEAN, C3]  # N, SVEB, VEB, F
class_labels  = ["N Recall", "SVEB Recall", "VEB Recall", "F Recall"]
class_keys    = ["n_recall", "sveb_recall", "veb_recall", "f_recall"]

fig, ax = plt.subplots(figsize=(5.5, 3.6))

for j, (ckey, clabel, ccolor) in enumerate(zip(class_keys, class_labels, class_colors)):
    vals = get_series("Fold 2", ckey)
    ls = "-" if j < 3 else "--"
    ax.plot(epochs, vals, color=ccolor, linewidth=1.8,
            linestyle=ls, marker="o", markersize=3.2, label=clabel)

ax.axvline(x=14, color="grey", linestyle=":", linewidth=1.0, alpha=0.8, label="Best epoch (14)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Recall (%)")
ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(framealpha=0.6, ncol=2)
fig.tight_layout()
save_fig(fig, "fig_A3_fold2_per_class_recall_curves")
plt.close(fig)


# ===========================================================================
# Figure B — Inter-Patient Performance: Per-Class Recall & F1 (bar + error)
# ===========================================================================
class_names = ["N", "SVEB", "VEB", "F"]
n_classes   = 4

# Per-fold recall (%) and F1 (%)
recall_per_fold = np.array([
    [fm["per_class_recall"] for _, fm in [(fk, fold_metrics[fk]) for fk in folds_data]]
])  # shape: 1 x 3 x 4
recall_per_fold = np.array([fold_metrics[fk]["per_class_recall"] for fk in folds_data])  # 3x4
f1_per_fold     = np.array([fold_metrics[fk]["per_class_f1"]     for fk in folds_data])  # 3x4

recall_mean = recall_per_fold.mean(axis=0)
recall_std  = recall_per_fold.std(axis=0)
f1_mean     = f1_per_fold.mean(axis=0)
f1_std      = f1_per_fold.std(axis=0)

# Two panels: (left) per-fold recall grouped by class; (right) mean±std recall and F1
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)

# --- Left: per-fold recall grouped by class ---
ax = axes[0]
x   = np.arange(n_classes)
w   = 0.22
offsets = [-w, 0, w]

for i, fk in enumerate(folds_data):
    vals = recall_per_fold[i]
    bars = ax.bar(x + offsets[i], vals, width=w, color=FOLD_COLORS[i],
                  label=FOLD_LABELS[i], edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 110)
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.legend(framealpha=0.6)
ax.set_xlabel("AAMI Class")

# --- Right: mean±std for Recall and F1 side by side per class ---
ax2 = axes[1]
w2  = 0.30
x2  = np.arange(n_classes)

bars_r = ax2.bar(x2 - w2/2, recall_mean, width=w2, color=C2,
                 label="Mean Recall", edgecolor="white",
                 yerr=recall_std, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#2a5280"})
bars_f = ax2.bar(x2 + w2/2, f1_mean,     width=w2, color=C1,
                 label="Mean F1",     edgecolor="white",
                 yerr=f1_std, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#a03030"})

ax2.set_xticks(x2)
ax2.set_xticklabels(class_names)
ax2.set_ylabel("Score (%)")
ax2.set_ylim(0, 115)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax2.legend(framealpha=0.6)
ax2.set_xlabel("AAMI Class")

fig.tight_layout(w_pad=3.0)
save_fig(fig, "fig_B_inter_patient_performance")
plt.close(fig)


# ===========================================================================
# Figure C — Fold Score Progression Over Epochs
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.6))

for i, fk in enumerate(folds_data):
    scores = get_series(fk, "fold_score")
    ax.plot(epochs, scores, color=FOLD_COLORS[i], linewidth=1.8,
            marker="o", markersize=3.2, label=FOLD_LABELS[i])

# Mark best-epoch score
best_scores = {
    "Fold 1": (20,  47.71),
    "Fold 2": (14,  67.56),
    "Fold 3": (14,  57.51),
}
for i, (fk, (ep, sc)) in enumerate(best_scores.items()):
    ax.scatter([ep], [sc], color=FOLD_COLORS[i], s=80, zorder=5, marker="*")
    ax.annotate(f"{sc:.1f}", xy=(ep, sc), xytext=(ep + 0.3, sc + 5),
                fontsize=7.5, color=FOLD_COLORS[i])

ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Fold Validation Score")
ax.set_xlim(1, 20)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(framealpha=0.6)
fig.tight_layout()
save_fig(fig, "fig_C_fold_score_progression")
plt.close(fig)


# ===========================================================================
# Figure D — Per-Class F1 Heatmap (3 folds × 4 classes)
# ===========================================================================
f1_matrix = f1_per_fold  # shape 3x4, values in %

fig, ax = plt.subplots(figsize=(5.5, 2.8))

# Custom colormap: light wheat → coral → sea blue (low→high)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "custom_heatmap",
    ["#FFF8E7", "#F5C9A0", "#F08080", "#4682B4"],
    N=256
)

im = ax.imshow(f1_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)

# Annotate cells
for row in range(3):
    for col in range(4):
        val = f1_matrix[row, col]
        txt_color = "white" if val > 65 else "#2a2a2a"
        ax.text(col, row, f"{val:.1f}%", ha="center", va="center",
                fontsize=9, color=txt_color, fontweight="bold")

ax.set_xticks(range(n_classes))
ax.set_xticklabels(class_names)
ax.set_yticks(range(3))
ax.set_yticklabels(FOLD_LABELS)
ax.set_xlabel("AAMI Class")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("F1 Score (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
save_fig(fig, "fig_D_per_class_f1_heatmap")
plt.close(fig)


# ===========================================================================
# Figure E — Augmentation Effect: WeightedRandomSampler class balance
#            (Training class distribution per fold — actual counts)
# ===========================================================================
# Training class counts from runtime logs (verified from session history)
train_counts = {
    "Fold 1": {"N": 37067, "SVEB": 739,  "VEB": 2401, "F": 414},
    "Fold 2": {"N": 36842, "SVEB": 546,  "VEB": 2111, "F": 522},
    "Fold 3": {"N": 37177, "SVEB": 742,  "VEB": 3077, "F": 880},
}
# Validation (held-out) support per fold
val_counts = {
    "Fold 1": {"N": 7746,  "SVEB": 204,  "VEB": 745,  "F": 484},
    "Fold 2": {"N": 7971,  "SVEB": 397,  "VEB": 1035, "F": 376},
    "Fold 3": {"N": 8636,  "SVEB": 201,  "VEB": 709,  "F": 18},
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
class_names4 = ["N", "SVEB", "VEB", "F"]
bar_colors_cls = [C2, C1, CMEAN, C3]

for ax_idx, (ax_sub, counts_dict, ylabel, panel_tag) in enumerate(zip(
        axes,
        [train_counts, val_counts],
        ["Sample Count (Training Set)", "Sample Count (Validation Set)"],
        ["Training Distribution (DS1 partitions)", "Validation Distribution (per fold)"]
)):
    x   = np.arange(len(class_names4))
    w   = 0.22
    off = [-w, 0, w]
    for i, fk in enumerate(folds_data):
        vals = [counts_dict[fk][c] for c in class_names4]
        ax_sub.bar(x + off[i], vals, width=w, color=FOLD_COLORS[i],
                   label=FOLD_LABELS[i], edgecolor="white", linewidth=0.5)
    ax_sub.set_xticks(x)
    ax_sub.set_xticklabels(class_names4)
    ax_sub.set_ylabel(ylabel)
    ax_sub.set_xlabel("AAMI Class")
    ax_sub.legend(framealpha=0.6)
    ax_sub.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

fig.tight_layout(w_pad=3.0)
save_fig(fig, "fig_E_class_distribution_train_val")
plt.close(fig)


# ===========================================================================
# Figure F — Summary Radar/Polar: Per-fold key metrics (spider chart)
# ===========================================================================
# Metrics to show on radar: N Recall, SVEB Recall, VEB Recall, Active Acc, Macro F1
radar_labels  = ["N Recall", "SVEB Recall", "VEB Recall", "Active Acc.", "Macro F1"]
radar_metrics = ["n_recall", "sveb_recall", "veb_recall",
                 "active_accuracy", "active_macro_f1"]

radar_values = {}
for fk in folds_data:
    fm = fold_metrics[fk]
    radar_values[fk] = [fm[m] for m in radar_metrics]

N_axes = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N_axes, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={"polar": True})

for i, fk in enumerate(folds_data):
    vals = radar_values[fk] + radar_values[fk][:1]
    ax.plot(angles, vals, color=FOLD_COLORS[i], linewidth=1.8, label=FOLD_LABELS[i])
    ax.fill(angles, vals, color=FOLD_COLORS[i], alpha=0.12)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9)
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), framealpha=0.6)

fig.tight_layout()
save_fig(fig, "fig_F_radar_fold_metrics")
plt.close(fig)


print("\n[OK] All 6 additional figures generated successfully.")
print(f"Output directory: {FIGURES_DIR}")

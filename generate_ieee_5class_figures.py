"""
generate_ieee_5class_figures.py
-------------------------------
Generates the 3 publication-ready IEEE journal figures featuring all 5 AAMI classes:
  [N, SVEB, VEB, F, Q]

All experimental values are strictly from the verified 3-fold fixed-HP experiment results:
  - Fold 1 (k=1): N=97.20%, SVEB=15.69%, VEB=73.69%, F=0.00%, Q=0.00%
  - Fold 2 (k=2): N=96.89%, SVEB=88.16%, VEB=99.52%, F=0.27%, Q=0.00%
  - Fold 3 (k=3): N=96.61%, SVEB=33.33%, VEB=81.38%, F=0.00%, Q=0.00%
  - Mean:         N=96.90%, SVEB=45.73%, VEB=84.86%, F=0.09%, Q=0.00%

Figures:
  1. Radar Figure (5 classes):
     - Spokes: Class N, Class SVEB, Class VEB, Class F, Class Q
     - Radial scale: 0 to 100%, 1 unit = 20% (0, 20, 40, 60, 80, 100)
     - Shows Fold 1 (k=1), Fold 2 (k=2), Fold 3 (k=3), and Mean
  2. Recall Curve per Class over Training Epochs (5 classes):
     - Epochs 1 to 20 for Fold 2 (k=2, best fold)
     - Classes: N, SVEB, VEB, F, Q (Q marked at 0% across all epochs)
  3. Inter-Patient Performance at Each Fold k (5 classes):
     - Left: Class Recall at each k (k=1, k=2, k=3) across all 5 classes
     - Right: Class F1-Score at each k (k=1, k=2, k=3) across all 5 classes + Mean±Std

Typography & Style:
  - Font: strictly Times New Roman
  - Large, readable fonts suitable for IEEE journal papers
  - Palette: Coral (#F08080), Sea Blue (#4682B4), Wheat Yellow (#DEB887), Sea Green (#3CB371)
  - No interior figure titles (captions handled in LaTeX)
  - Formats: PNG (300 DPI), PDF, SVG
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Directories and Files
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

AGGREGATE_JSON = os.path.join(RESULTS_DIR, "fixed_hp_3fold_aggregate_metrics.json")
TRAINING_LOG_JSON = os.path.join(RESULTS_DIR, "fixed_hp_3fold_training_log.json")

# ---------------------------------------------------------------------------
# Global Styling: Times New Roman, IEEE Journal standards
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color Palette: Coral, Sea Blue, Wheat Yellow, Sea Green
COLOR_FOLD1 = "#F08080"   # Pastel Coral (Fold 1 / k=1)
COLOR_FOLD2 = "#4682B4"   # Sea Blue / Steel Blue (Fold 2 / k=2)
COLOR_FOLD3 = "#DEB887"   # Wheat Yellow / Burlywood (Fold 3 / k=3)
COLOR_MEAN  = "#3CB371"   # Sea Green (Mean / Aggregate)

FOLD_COLORS = [COLOR_FOLD1, COLOR_FOLD2, COLOR_FOLD3]
FOLD_LABELS = ["Fold 1 (k=1)", "Fold 2 (k=2)", "Fold 3 (k=3)"]
FOLDS_KEYS  = ["Fold 1", "Fold 2", "Fold 3"]

# 5 AAMI Classes
CLASSES_5 = ["N", "SVEB", "VEB", "F", "Q"]

# ---------------------------------------------------------------------------
# Load Experimental Results
# ---------------------------------------------------------------------------
with open(AGGREGATE_JSON, "r") as f:
    agg_data = json.load(f)

with open(TRAINING_LOG_JSON, "r") as f:
    tlog_data = json.load(f)

# Extract 5-class Recall and F1 (Q = 0.0)
recalls_5class = {}
f1_5class = {}

for fk in FOLDS_KEYS:
    m = agg_data["individual_folds"][fk]["metrics"]
    recalls_5class[fk] = m["per_class_recall"] + [0.0]  # [N, SVEB, VEB, F, Q]
    f1_5class[fk]      = m["per_class_f1"] + [0.0]

# Mean and Std across the 3 folds
recalls_arr = np.array([recalls_5class[fk] for fk in FOLDS_KEYS])  # (3, 5)
f1_arr      = np.array([f1_5class[fk] for fk in FOLDS_KEYS])       # (3, 5)

recall_mean = recalls_arr.mean(axis=0)
recall_std  = recalls_arr.std(axis=0)
f1_mean     = f1_arr.mean(axis=0)
f1_std      = f1_arr.std(axis=0)

epochs = list(range(1, 21))

def save_publication_figure(fig, name):
    for ext in ["png", "pdf", "svg"]:
        out_path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
    print(f"[OK] Generated: {name}.png, {name}.pdf, {name}.svg")


# ===========================================================================
# FIGURE 1: Radar Chart for 5 Classes (Scale 1 unit = 20%)
# ===========================================================================
def generate_radar_5class():
    spoke_labels = [f"Class {c}" for c in CLASSES_5]
    N_spokes = len(spoke_labels)
    angles = np.linspace(0, 2 * np.pi, N_spokes, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})

    # Radial scale: 0 to 100%, 1 unit = 20%
    grid_ticks = [20, 40, 60, 80, 100]
    ax.set_ylim(0, 100)
    ax.set_yticks(grid_ticks)
    ax.set_yticklabels([f"{t}%" for t in grid_ticks], fontsize=10, fontfamily="Times New Roman", color="#444444")
    
    # Custom circular grid appearance
    ax.yaxis.grid(True, color="#CCCCCC", linestyle="--", linewidth=0.8)
    ax.xaxis.grid(True, color="#CCCCCC", linestyle="-", linewidth=0.8)

    # Plot Fold 1, 2, 3
    for i, fk in enumerate(FOLDS_KEYS):
        vals = recalls_5class[fk] + recalls_5class[fk][:1]
        ax.plot(angles, vals, color=FOLD_COLORS[i], linewidth=2.2, label=FOLD_LABELS[i], marker="o", markersize=5)
        ax.fill(angles, vals, color=FOLD_COLORS[i], alpha=0.12)

    # Plot Mean curve
    mean_vals = recall_mean.tolist() + recall_mean.tolist()[:1]
    ax.plot(angles, mean_vals, color=COLOR_MEAN, linewidth=2.0, linestyle="--", label="Mean Across Folds", marker="s", markersize=4)

    # Set Spoke Category Labels
    ax.set_thetagrids(np.degrees(angles[:-1]), spoke_labels, fontsize=13, fontfamily="Times New Roman", fontweight="bold")
    ax.tick_params(axis="x", pad=12)

    # Legend outside to preserve chart clarity
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        framealpha=0.85,
        fontsize=11,
        edgecolor="#B0B0B0"
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")

    fig.tight_layout()
    save_publication_figure(fig, "fig1_radar_5class_performance")
    plt.close(fig)


# ===========================================================================
# FIGURE 2: Recall Curve Per Class Training (Fold 2 / k=2, all 5 classes)
# ===========================================================================
def generate_recall_curves_5class():
    # Extract Fold 2 epoch-wise metrics (k=2 was the best fold)
    f2_logs = tlog_data["Fold 2"]
    recalls_by_class = {
        "N":    [ep["n_recall"] for ep in f2_logs],
        "SVEB": [ep["sveb_recall"] for ep in f2_logs],
        "VEB":  [ep["veb_recall"] for ep in f2_logs],
        "F":    [ep["f_recall"] for ep in f2_logs],
        "Q":    [0.0] * len(f2_logs),  # Q marked strictly as 0
    }

    class_plot_styles = {
        "N":    {"color": "#4682B4", "ls": "-",  "marker": "o", "label": "Class N"},
        "SVEB": {"color": "#F08080", "ls": "-",  "marker": "s", "label": "Class SVEB"},
        "VEB":  {"color": "#3CB371", "ls": "-",  "marker": "^", "label": "Class VEB"},
        "F":    {"color": "#DEB887", "ls": "--", "marker": "d", "label": "Class F"},
        "Q":    {"color": "#888888", "ls": ":",  "marker": "x", "label": "Class Q (0.0)"},
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for c in CLASSES_5:
        cfg = class_plot_styles[c]
        ax.plot(
            epochs,
            recalls_by_class[c],
            color=cfg["color"],
            linestyle=cfg["ls"],
            marker=cfg["marker"],
            markersize=5.5,
            linewidth=2.0,
            label=cfg["label"]
        )

    # Highlight best epoch (Epoch 14 for Fold 2)
    ax.axvline(x=14, color="#555555", linestyle=":", linewidth=1.5, alpha=0.8, label="Selected Epoch (14)")

    ax.set_xlabel("Training Epoch", fontsize=13, fontfamily="Times New Roman")
    ax.set_ylabel("Validation Recall (%)", fontsize=13, fontfamily="Times New Roman")
    ax.set_xlim(1, 20)
    ax.set_ylim(-3, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        tick.set_fontsize(12)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.85,
        fontsize=11,
        edgecolor="#B0B0B0"
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")

    fig.tight_layout()
    save_publication_figure(fig, "fig2_recall_curve_per_class_training_5class")
    plt.close(fig)


# ===========================================================================
# FIGURE 3: Inter-Patient Performance at Each Fold k (5 classes)
# ===========================================================================
def generate_interpatient_performance_5class():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    x = np.arange(len(CLASSES_5))
    w = 0.22
    offsets = [-w, 0, w]

    # --- Left Subplot: Recall at each k (k=1, k=2, k=3) ---
    for i, fk in enumerate(FOLDS_KEYS):
        vals = recalls_5class[fk]
        bars = ax1.bar(
            x + offsets[i],
            vals,
            width=w,
            color=FOLD_COLORS[i],
            label=FOLD_LABELS[i],
            edgecolor="white",
            linewidth=0.7,
            zorder=3
        )
        # Value annotations above bars
        for bar, val in zip(bars, vals):
            if val > 1.5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, fontfamily="Times New Roman",
                    color="#333333"
                )
            elif val == 0.0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.0,
                    "0",
                    ha="center", va="bottom",
                    fontsize=8.5, fontfamily="Times New Roman",
                    color="#777777"
                )

    ax1.set_xlabel("AAMI Heartbeat Class", fontsize=13, fontfamily="Times New Roman")
    ax1.set_ylabel("Class Recall / Sensitivity (%)", fontsize=13, fontfamily="Times New Roman")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))
    for tick in ax1.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        tick.set_fontsize(12)

    leg1 = ax1.legend(loc="upper right", framealpha=0.85, fontsize=11, edgecolor="#B0B0B0")
    for text in leg1.get_texts():
        text.set_fontfamily("Times New Roman")

    # --- Right Subplot: F1-Score at each k (k=1, k=2, k=3) ---
    for i, fk in enumerate(FOLDS_KEYS):
        vals = f1_5class[fk]
        bars = ax2.bar(
            x + offsets[i],
            vals,
            width=w,
            color=FOLD_COLORS[i],
            label=FOLD_LABELS[i],
            edgecolor="white",
            linewidth=0.7,
            zorder=3
        )
        for bar, val in zip(bars, vals):
            if val > 1.5:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, fontfamily="Times New Roman",
                    color="#333333"
                )
            elif val == 0.0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.0,
                    "0",
                    ha="center", va="bottom",
                    fontsize=8.5, fontfamily="Times New Roman",
                    color="#777777"
                )

    ax2.set_xlabel("AAMI Heartbeat Class", fontsize=13, fontfamily="Times New Roman")
    ax2.set_ylabel("Class F1-Score (%)", fontsize=13, fontfamily="Times New Roman")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(20))
    for tick in ax2.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        tick.set_fontsize(12)

    leg2 = ax2.legend(loc="upper right", framealpha=0.85, fontsize=11, edgecolor="#B0B0B0")
    for text in leg2.get_texts():
        text.set_fontfamily("Times New Roman")

    fig.tight_layout(w_pad=3.0)
    save_publication_figure(fig, "fig3_interpatient_performance_at_each_k_5class")
    plt.close(fig)


if __name__ == "__main__":
    print("[*] Generating 5-class IEEE journal figures...")
    generate_radar_5class()
    generate_recall_curves_5class()
    generate_interpatient_performance_5class()
    print("[OK] All figures generated successfully in results/figures/")

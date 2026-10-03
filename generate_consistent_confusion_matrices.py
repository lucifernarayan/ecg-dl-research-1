"""
generate_consistent_confusion_matrices.py
-----------------------------------------
Generates 3 perfectly consistent, publication-ready 5x5 Confusion Matrices
for Fold 2 (k=2, best fold) with all 5 AAMI classes [N, SVEB, VEB, F, Q]:

1. Numbers Matrix (Raw Sample Counts):
   - Color: Classic Academic Sea Blue
   - Colorbar label: "Number of Samples"
   - Numbers inside cells: formatted integers (e.g. 7,723, 350, 1,030, etc.)

2. Percentage Matrix (Row-Normalized / Class Recall %):
   - Color: Classic Academic Sea Blue
   - Colorbar label: "Percentage (%)"
   - Values inside cells: percentage (e.g. 96.89%, 88.16%, etc.) where each row sums to 100%

3. Accuracy Matrix (Magenta Colormap):
   - Color: Beautiful publication-grade Magenta gradient (#FFFFFF -> #E91E63 -> #880E4F)
   - Colorbar label: "Accuracy (%)"
   - Values inside cells: class accuracy percentage (%)

Styling:
   - 100% consistent labeling across all 3 matrices:
     * X-axis label: "Predicted Class"
     * Y-axis label: "True Class"
     * Category ticks: "N", "SVEB", "VEB", "F", "Q"
     * Typography: strictly Times New Roman
   - High-resolution: PNG (300 DPI), vector PDF, vector SVG in results/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Typography & IEEE Styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
})

CLASSES_5 = ["N", "SVEB", "VEB", "F", "Q"]

# ---------------------------------------------------------------------------
# Data: Fold 2 (Best Fold, Epoch 14)
# ---------------------------------------------------------------------------
CM_COUNTS = np.array([
    [7723,  239,    7,    2,  0],  # True N    (Support: 7971)
    [  47,  350,    0,    0,  0],  # True SVEB (Support: 397)
    [   4,    1, 1030,    0,  0],  # True VEB  (Support: 1035)
    [ 143,   23,  209,    1,  0],  # True F    (Support: 376)
    [   0,    0,    0,    0,  0]   # True Q    (Support: 0)
], dtype=float)

# Row-normalized matrix (percentages summing to 100% per row)
row_sums = CM_COUNTS.sum(axis=1, keepdims=True)
CM_PERCENT = np.zeros_like(CM_COUNTS)
for i in range(5):
    if row_sums[i, 0] > 0:
        CM_PERCENT[i, :] = (CM_COUNTS[i, :] / row_sums[i, 0]) * 100.0
    else:
        CM_PERCENT[i, :] = 0.0

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
# 1. Pastel Sea Blue colormap (for Numbers and Percentage)
cmap_blue = LinearSegmentedColormap.from_list(
    "PastelSeaBlue",
    ["#FFFFFF", "#E6F0FA", "#B8D5E5", "#78ACD9", "#4682B4", "#204D74"],
    N=256
)

# 2. Academic Magenta colormap (for Accuracy in Magenta)
cmap_magenta = LinearSegmentedColormap.from_list(
    "AcademicMagenta",
    ["#FFFFFF", "#FCE4EC", "#F48FB1", "#E91E63", "#C2185B", "#700836"],
    N=256
)

def save_fig(fig, basename):
    for ext in ["png", "pdf", "svg"]:
        out_fig = os.path.join(FIGURES_DIR, f"{basename}.{ext}")
        fig.savefig(out_fig, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
        out_root = os.path.join(RESULTS_DIR, f"{basename}.{ext}")
        fig.savefig(out_root, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
    print(f"[OK] Saved: {basename}.png, .pdf, .svg")


# ===========================================================================
# 1. NUMBERS MATRIX (Sample Counts)
# ===========================================================================
def generate_matrix_numbers():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(CM_COUNTS, interpolation="nearest", cmap=cmap_blue, vmin=0, vmax=CM_COUNTS.max())

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Number of Samples", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    thresh = CM_COUNTS.max() / 2.0
    for i in range(5):
        for j in range(5):
            val = int(CM_COUNTS[i, j])
            color = "white" if val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    color=color, fontsize=11, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "cm_fold2_numbers_5class")
    plt.close(fig)


# ===========================================================================
# 2. PERCENTAGE MATRIX (Normalized %)
# ===========================================================================
def generate_matrix_percentage():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(CM_PERCENT, interpolation="nearest", cmap=cmap_blue, vmin=0, vmax=100)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Percentage (%)", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    thresh = 50.0
    for i in range(5):
        for j in range(5):
            val = CM_PERCENT[i, j]
            color = "white" if val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    color=color, fontsize=11, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "cm_fold2_percentage_5class")
    plt.close(fig)


# ===========================================================================
# 3. ACCURACY MATRIX IN MAGENTA (%)
# ===========================================================================
def generate_matrix_accuracy_magenta():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(CM_PERCENT, interpolation="nearest", cmap=cmap_magenta, vmin=0, vmax=100)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Accuracy (%)", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    thresh = 50.0
    for i in range(5):
        for j in range(5):
            val = CM_PERCENT[i, j]
            color = "white" if val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    color=color, fontsize=11, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "cm_fold2_accuracy_magenta_5class")
    plt.close(fig)


if __name__ == "__main__":
    print("[*] Generating consistent confusion matrices...")
    generate_matrix_numbers()
    generate_matrix_percentage()
    generate_matrix_accuracy_magenta()
    print("[OK] All 3 consistent confusion matrices generated successfully.")

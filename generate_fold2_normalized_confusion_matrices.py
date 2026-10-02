"""
generate_fold2_normalized_confusion_matrices.py
------------------------------------------------
Generates publication-quality 5x5 Confusion Matrices for Fold 2 (k=2, best fold):
  1. Recall Matrix (Row-Normalized / Class Sensitivity %):
     Each row i is divided by True Class support sum(C_i), so rows sum to 100%.
     The diagonal displays the exact Class Recall (Sensitivity):
     N = 96.89%, SVEB = 88.16%, VEB = 99.52%, F = 0.27%, Q = 0.00%.

  2. Total Normalized Matrix (Overall % of total evaluated beats):
     Each cell (i, j) is divided by the total number of beats (N = 9,779).
     All cells sum to 100%.

  3. Dual-Annotated Matrix (Raw Count + Recall % in each cell):
     Shows both exact heartbeat counts and class recall percentages simultaneously.

Features:
  - All 5 AAMI Classes: [N, SVEB, VEB, F, Q]
  - Class Q shown cleanly with 0 values
  - Typography: strictly Times New Roman
  - Large, readable fonts for IEEE journal paper
  - Output: PNG (300 DPI), vector PDF, vector SVG in results/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global Matplotlib Styling: Times New Roman, IEEE Journal
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

# Raw Confusion Matrix for Fold 2 (Best Epoch 14, 5x5)
# True classes (rows) vs Predicted classes (cols)
CM_RAW_5X5 = np.array([
    [7723,  239,    7,    2,  0],  # True N    (Support: 7971)
    [  47,  350,    0,    0,  0],  # True SVEB (Support: 397)
    [   4,    1, 1030,    0,  0],  # True VEB  (Support: 1035)
    [ 143,   23,  209,    1,  0],  # True F    (Support: 376)
    [   0,    0,    0,    0,  0]   # True Q    (Support: 0)
], dtype=float)

TOTAL_SAMPLES = np.sum(CM_RAW_5X5)  # 9779.0

# 1. Recall Matrix (Row-Normalized, in %)
row_sums = CM_RAW_5X5.sum(axis=1, keepdims=True)
CM_RECALL = np.zeros_like(CM_RAW_5X5)
for i in range(5):
    if row_sums[i, 0] > 0:
        CM_RECALL[i, :] = (CM_RAW_5X5[i, :] / row_sums[i, 0]) * 100.0
    else:
        CM_RECALL[i, :] = 0.0  # Q class has 0 support

# 2. Total Normalized Matrix (in %)
CM_TOTAL_NORM = (CM_RAW_5X5 / TOTAL_SAMPLES) * 100.0

# Custom soft pastel blue/teal colormap suitable for IEEE journals
cmap_pastel_blue = LinearSegmentedColormap.from_list(
    "PastelBlues",
    ["#FFFFFF", "#E3EEF8", "#B5D4EE", "#78ACD9", "#4682B4", "#2B5C8F"],
    N=256
)

def save_fig(fig, basename):
    for ext in ["png", "pdf", "svg"]:
        out_path = os.path.join(FIGURES_DIR, f"{basename}.{ext}")
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
    # Also save in results/ root for convenience
    for ext in ["png", "pdf", "svg"]:
        out_path = os.path.join(RESULTS_DIR, f"{basename}.{ext}")
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
    print(f"[OK] Saved: {basename}.png, .pdf, .svg")


# ===========================================================================
# 1. RECALL MATRIX (Row-Normalized / Class Sensitivity %)
# ===========================================================================
def plot_recall_matrix():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    
    im = ax.imshow(CM_RECALL, interpolation="nearest", cmap=cmap_pastel_blue, vmin=0, vmax=100)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Recall / Class Sensitivity (%)", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True AAMI Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    # Annotate cells with percentage values
    thresh = 50.0
    for i in range(5):
        for j in range(5):
            val = CM_RECALL[i, j]
            text_color = "white" if val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    color=text_color, fontsize=11, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "fold2_confusion_matrix_recall_normalized_5class")
    plt.close(fig)


# ===========================================================================
# 2. TOTAL NORMALIZED MATRIX (Normalized by total evaluated beats %)
# ===========================================================================
def plot_total_normalized_matrix():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    
    im = ax.imshow(CM_TOTAL_NORM, interpolation="nearest", cmap=cmap_pastel_blue, vmin=0, vmax=80)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Proportion of Total Dataset (%)", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True AAMI Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    thresh = 40.0
    for i in range(5):
        for j in range(5):
            val = CM_TOTAL_NORM[i, j]
            text_color = "white" if val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    color=text_color, fontsize=11, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "fold2_confusion_matrix_total_normalized_5class")
    plt.close(fig)


# ===========================================================================
# 3. DUAL-ANNOTATED MATRIX (Count + Recall % in each cell)
# ===========================================================================
def plot_dual_annotated_matrix():
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    
    im = ax.imshow(CM_RECALL, interpolation="nearest", cmap=cmap_pastel_blue, vmin=0, vmax=100)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Recall / Sensitivity (%)", fontsize=12, fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=12, fontfamily="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)
    ax.set_ylabel("True AAMI Class", fontsize=13, fontfamily="Times New Roman", fontweight="bold", labelpad=8)

    thresh = 50.0
    for i in range(5):
        for j in range(5):
            raw_cnt = int(CM_RAW_5X5[i, j])
            pct_val = CM_RECALL[i, j]
            text_color = "white" if pct_val > thresh else "#1A1A1A"
            weight = "bold" if (i == j and raw_cnt > 0) else "normal"
            
            if raw_cnt > 0:
                cell_text = f"{raw_cnt:,}\n({pct_val:.1f}%)"
            else:
                cell_text = "0\n(0.0%)"
                
            ax.text(j, i, cell_text, ha="center", va="center",
                    color=text_color, fontsize=10, fontfamily="Times New Roman", fontweight=weight)

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_fig(fig, "fold2_confusion_matrix_dual_annotated_5class")
    plt.close(fig)


if __name__ == "__main__":
    print("[*] Generating Fold 2 5-class normalized confusion matrices...")
    plot_recall_matrix()
    plot_total_normalized_matrix()
    plot_dual_annotated_matrix()
    print("[OK] All normalized confusion matrices generated successfully.")

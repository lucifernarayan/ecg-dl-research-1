"""
generate_fold2_accuracy_bluewhite_cm.py
---------------------------------------
Generates the publication-grade Accuracy Confusion Matrix for Fold 2 (k=2)
using the standard academic Blue & White colormap with large, highly legible
Times New Roman typography for IEEE journal papers.

Classes: ['N', 'SVEB', 'VEB', 'F', 'Q']
Metrics source: results/fixed_hp_fold2_metrics.json
Format: Strictly percentages with 14pt bold text, 16pt axis labels.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Directories & Files
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FOLD2_METRICS_PATH = os.path.join(RESULTS_DIR, "fixed_hp_fold2_metrics.json")

# ---------------------------------------------------------------------------
# Global Matplotlib Styling: Times New Roman, Large IEEE Font
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
# Data Ingestion & Calculation
# ---------------------------------------------------------------------------
with open(FOLD2_METRICS_PATH, "r") as f:
    metrics = json.load(f)

raw_4x4 = np.array(metrics["confusion_matrix"], dtype=float)
raw_5x5 = np.zeros((5, 5), dtype=float)
raw_5x5[:4, :4] = raw_4x4

# Class accuracy / row-normalized percentages (%)
row_sums = raw_5x5.sum(axis=1, keepdims=True)
pct_matrix = np.divide(raw_5x5, row_sums, out=np.zeros_like(raw_5x5), where=row_sums > 0) * 100.0

# ---------------------------------------------------------------------------
# Standard Blue and White Colormap (Pure White -> Light Blue -> Deep Ocean Blue)
# ---------------------------------------------------------------------------
cmap_blue_white = LinearSegmentedColormap.from_list(
    "StandardBlueWhite",
    ["#FFFFFF", "#E3EEF8", "#B5D4EE", "#70A6D6", "#3A7EBA", "#1E4C7A"],
    N=256
)

def plot_accuracy_bluewhite_cm():
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    
    im = ax.imshow(pct_matrix, interpolation="nearest", cmap=cmap_blue_white, vmin=0.0, vmax=100.0)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Accuracy (%)", fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        
    # Tick marks & category labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    
    # Axes titles
    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    
    # Cell text annotations (14pt large font for paper visibility)
    threshold = 50.0
    for i in range(5):
        for j in range(5):
            val = pct_matrix[i, j]
            text_color = "white" if val > threshold else "#1A1A1A"
            text_weight = "bold" if (i == j and val > 0.0) else "normal"
            ax.text(
                j, i, f"{val:.2f}%",
                ha="center", va="center",
                color=text_color,
                fontsize=14,
                fontfamily="Times New Roman",
                fontweight=text_weight
            )
            
    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    
    # Save as both names for convenience
    for name in ["cm_fold2_accuracy_bluewhite_5class", "cm_fold2_accuracy_standard_blue_5class"]:
        for target_dir in [FIGURES_DIR, RESULTS_DIR]:
            for ext in ["png", "pdf", "svg"]:
                filepath = os.path.join(target_dir, f"{name}.{ext}")
                fig.savefig(filepath, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
                
    plt.close(fig)
    print("[OK] Saved Accuracy Confusion Matrix in Standard Blue & White (.png, .pdf, .svg)")


if __name__ == "__main__":
    plot_accuracy_bluewhite_cm()

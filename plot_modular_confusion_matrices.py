"""
plot_modular_confusion_matrices.py
----------------------------------
Modular, standardized confusion matrix generator for IEEE journal publication.

Guarantees:
  1. Single unified plotting engine for 100% structural consistency across all figures.
  2. Dynamically loads raw evaluation metrics directly from results/fixed_hp_fold2_metrics.json.
  3. Strict percentage display: Every cell shows only percentage (e.g. 96.89%, 0.00%).
  4. Standardized Y-axis label: "Class Sensitivity / Recall (%)"
  5. Standardized X-axis label: "Predicted Class"
  6. Standardized 5 AAMI classes: ['N', 'SVEB', 'VEB', 'F', 'Q']
  7. Typography: Strictly Times New Roman
  8. Two colormap variations:
     - Classic Academic Sea Blue
     - Vibrant Academic Magenta
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
# Global Matplotlib Configuration (Strict Times New Roman, IEEE standard)
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
# Data Ingestion & Programmatic Normalization
# ---------------------------------------------------------------------------
def load_and_normalize_cm():
    with open(FOLD2_METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    raw_mat = np.array(metrics["confusion_matrix"], dtype=float)
    if raw_mat.shape == (5, 5):
        raw_5x5 = raw_mat
    else:
        raw_5x5 = np.zeros((5, 5), dtype=float)
        raw_5x5[:raw_mat.shape[0], :raw_mat.shape[1]] = raw_mat
    
    # Programmatic row-normalization: Recall / Sensitivity percentage
    row_sums = raw_5x5.sum(axis=1, keepdims=True)
    pct_matrix = np.divide(raw_5x5, row_sums, out=np.zeros_like(raw_5x5), where=row_sums > 0) * 100.0
    
    return raw_5x5, pct_matrix

# ---------------------------------------------------------------------------
# Colormap Definitions
# ---------------------------------------------------------------------------
CMAP_SEA_BLUE = LinearSegmentedColormap.from_list(
    "AcademicSeaBlue",
    ["#FFFFFF", "#E8F1F8", "#B5D4EE", "#78ACD9", "#4682B4", "#2B5C8F"],
    N=256
)

CMAP_MAGENTA = LinearSegmentedColormap.from_list(
    "AcademicMagenta",
    ["#FFFFFF", "#FCE4EC", "#F8BBD0", "#F06292", "#E91E63", "#880E4F"],
    N=256
)

# ---------------------------------------------------------------------------
# Unified Modular Plotting Engine
# ---------------------------------------------------------------------------
def plot_confusion_matrix_percentage(pct_matrix, colormap, colorbar_title, output_basename):
    """
    Renders and saves a 5x5 confusion matrix strictly in percentage format.
    Every single parameter, label, font, and layout is 100% standardized with large publication-grade fonts.
    """
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    
    im = ax.imshow(pct_matrix, interpolation="nearest", cmap=colormap, vmin=0.0, vmax=100.0)
    
    # Standardized colorbar with large font
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(colorbar_title, fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        
    # Standardized tick marks & labels with large bold font
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    
    # Standardized axes titles (prominent size for IEEE journal)
    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("Class Sensitivity / Recall (%)", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    
    # Strict percentage annotation in every cell with large font
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
    
    # Save in both figures/ directory and results/ root in PNG (300 DPI), PDF, and SVG
    for target_dir in [FIGURES_DIR, RESULTS_DIR]:
        for ext in ["png", "pdf", "svg"]:
            filepath = os.path.join(target_dir, f"{output_basename}.{ext}")
            fig.savefig(filepath, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
            
    plt.close(fig)
    print(f"[OK] Saved standardized figure: {output_basename} (.png, .pdf, .svg)")


# ---------------------------------------------------------------------------
# Execution Driver
# ---------------------------------------------------------------------------
def main():
    print("[*] Ingesting Fold 2 ground truth metrics and calculating percentages via code...")
    raw_5x5, pct_matrix = load_and_normalize_cm()
    
    print("\n--- Programmatically Computed 5x5 Recall / Sensitivity Matrix (%) ---")
    for idx, c in enumerate(CLASSES_5):
        row_str = "  ".join([f"{pct_matrix[idx, j]:6.2f}%" for j in range(5)])
        print(f"  Class {c:4s} -> [{row_str}]")
        
    print("\n[*] Generating standardized figures via unified modular engine...")
    
    # 1. Sea Blue Version
    plot_confusion_matrix_percentage(
        pct_matrix=pct_matrix,
        colormap=CMAP_SEA_BLUE,
        colorbar_title="Class Sensitivity / Recall (%)",
        output_basename="cm_fold2_recall_percentage_seablue_5class"
    )
    
    # 2. Magenta Version
    plot_confusion_matrix_percentage(
        pct_matrix=pct_matrix,
        colormap=CMAP_MAGENTA,
        colorbar_title="Class Sensitivity / Recall (%)",
        output_basename="cm_fold2_recall_percentage_magenta_5class"
    )
    
    print("\n[OK] Both modular confusion matrices successfully generated with 100% consistency.")

if __name__ == "__main__":
    main()

"""
generate_ds2_confusion_matrices.py
-----------------------------------
Generates publication-grade Confusion Matrices for the Unseen AAMI EC57 DS2
Independent Test Set (22 Patients, 49,659 Heartbeats) evaluated using the
optimal Fold 2 checkpoint (257k parameters, alpha=0.7320, lr=0.001184).

Matrices generated:
  1. cm_ds2_accuracy_bluewhite_5class: Blue & White colormap, Row-Normalized (Recall/Accuracy %)
  2. cm_ds2_recall_percentage_magenta_5class: Academic Magenta colormap, Row-Normalized (%)
  3. cm_ds2_raw_counts_5class: Sea Blue colormap, Raw Sample Counts
  4. cm_ds2_dual_annotated_5class: Dual Annotation (Raw Count + Recall %)

Styling:
  - Strict Times New Roman font (IEEE Transaction standard)
  - 14pt-16pt bold labels and annotations
  - 300 DPI PNG, Vector PDF, and Vector SVG outputs
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Directories & Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS_JSON = os.path.join(RESULTS_DIR, "ds2_test_confusion_matrix_metrics.json")

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
# Load Metrics Data
# ---------------------------------------------------------------------------
with open(METRICS_JSON, "r") as f:
    data = json.load(f)

CM_RAW = np.array(data["confusion_matrix_raw"], dtype=float)
CM_RECALL = np.array(data["confusion_matrix_recall_pct"], dtype=float)
TOTAL_SAMPLES = int(data["total_beats"])

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
cmap_blue_white = LinearSegmentedColormap.from_list(
    "StandardBlueWhite",
    ["#FFFFFF", "#E3EEF8", "#B5D4EE", "#70A6D6", "#3A7EBA", "#1E4C7A"],
    N=256
)

cmap_magenta = LinearSegmentedColormap.from_list(
    "AcademicMagenta",
    ["#FFFFFF", "#FCE4EC", "#F48FB1", "#E91E63", "#C2185B", "#700836"],
    N=256
)

cmap_sea_blue = LinearSegmentedColormap.from_list(
    "PastelSeaBlue",
    ["#FFFFFF", "#E6F0FA", "#B8D5E5", "#78ACD9", "#4682B4", "#204D74"],
    N=256
)

def save_multiformat(fig, basename):
    for target_dir in [FIGURES_DIR, RESULTS_DIR]:
        for ext in ["png", "pdf", "svg"]:
            filepath = os.path.join(target_dir, f"{basename}.{ext}")
            fig.savefig(filepath, dpi=300 if ext == "png" else None, bbox_inches="tight", format=ext)
    print(f"[OK] Saved {basename} (.png, .pdf, .svg) to figures/ and results/")


# ---------------------------------------------------------------------------
# 1. Blue & White Accuracy / Recall Matrix (Percentages)
# ---------------------------------------------------------------------------
def plot_ds2_accuracy_bluewhite():
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(CM_RECALL, interpolation="nearest", cmap=cmap_blue_white, vmin=0.0, vmax=100.0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Accuracy (%)", fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)

    threshold = 50.0
    for i in range(5):
        for j in range(5):
            val = CM_RECALL[i, j]
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
    save_multiformat(fig, "cm_ds2_accuracy_bluewhite_5class")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Academic Magenta Recall Percentage Matrix
# ---------------------------------------------------------------------------
def plot_ds2_recall_magenta():
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(CM_RECALL, interpolation="nearest", cmap=cmap_magenta, vmin=0.0, vmax=100.0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Class Recall (%)", fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)

    threshold = 50.0
    for i in range(5):
        for j in range(5):
            val = CM_RECALL[i, j]
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
    save_multiformat(fig, "cm_ds2_recall_percentage_magenta_5class")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Raw Counts Matrix
# ---------------------------------------------------------------------------
def plot_ds2_raw_counts():
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(CM_RAW, interpolation="nearest", cmap=cmap_sea_blue, vmin=0, vmax=CM_RAW.max())

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Number of Heartbeats", fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)

    threshold = CM_RAW.max() * 0.4
    for i in range(5):
        for j in range(5):
            val = int(CM_RAW[i, j])
            text_color = "white" if val > threshold else "#1A1A1A"
            text_weight = "bold" if (i == j and val > 0) else "normal"
            ax.text(
                j, i, f"{val:,}",
                ha="center", va="center",
                color=text_color,
                fontsize=13,
                fontfamily="Times New Roman",
                fontweight=text_weight
            )

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_multiformat(fig, "cm_ds2_raw_counts_5class")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Dual Annotated Matrix (Count + Recall %)
# ---------------------------------------------------------------------------
def plot_ds2_dual_annotated():
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(CM_RECALL, interpolation="nearest", cmap=cmap_blue_white, vmin=0.0, vmax=100.0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Class Recall (%)", fontsize=15, fontfamily="Times New Roman", fontweight="bold", labelpad=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")
    ax.set_yticklabels(CLASSES_5, fontsize=15, fontfamily="Times New Roman", fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=16, fontfamily="Times New Roman", fontweight="bold", labelpad=10)

    threshold = 50.0
    for i in range(5):
        for j in range(5):
            cnt = int(CM_RAW[i, j])
            pct = CM_RECALL[i, j]
            text_color = "white" if pct > threshold else "#1A1A1A"
            text_weight = "bold" if (i == j and cnt > 0) else "normal"
            ax.text(
                j, i, f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color,
                fontsize=11.5,
                fontfamily="Times New Roman",
                fontweight=text_weight
            )

    ax.set_ylim(4.5, -0.5)
    fig.tight_layout()
    save_multiformat(fig, "cm_ds2_dual_annotated_5class")
    plt.close(fig)


if __name__ == "__main__":
    print("[*] Generating all DS2 publication-grade confusion matrices...")
    plot_ds2_accuracy_bluewhite()
    plot_ds2_recall_magenta()
    plot_ds2_raw_counts()
    plot_ds2_dual_annotated()
    print("[*] All DS2 confusion matrices generated successfully!")

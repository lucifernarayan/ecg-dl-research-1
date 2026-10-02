import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# Set Matplotlib publication rendering options
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 9.5
mpl.rcParams['axes.labelsize'] = 10.5
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Colors - Academic restrained palette
COLOR_FOLD1 = '#2b5c8f'  # Deep Navy Blue
COLOR_FOLD2 = '#d95f02'  # Muted Orange/Coral (Highlighting Fold 2)
COLOR_FOLD3 = '#7570b3'  # Slate Purple

CLASSES = ['Normal (N)', 'SVEB/A', 'VEB/PVC', 'Fusion (F/VT)']

# =====================================================================
# OUTPUT 1 — FOLD 2 CONFIGURATION TABLE
# =====================================================================
def create_fold2_table():
    table_data = [
        ["Target Dataset & Split", "Global / Fixed", "AAMI EC57 Inter-Patient DS1 Partition"],
        ["CV Strategy", "Global / Fixed", "3-Fold Patient-Level Cross-Validation"],
        ["Fold 2 Validation Patients", "Fold 2 Specific", "['208', '209', '114', '115'] (4 Patients)"],
        ["Fold 2 Training Patients", "Fold 2 Specific", "18 remaining DS1 Patients"],
        ["Sampling Alpha (α)", "Global / Fixed", "0.7320"],
        ["Learning Rate (LR)", "Global / Fixed", "0.001184"],
        ["Weight Decay (WD)", "Global / Fixed", "7.114476e-04"],
        ["Batch Size", "Global / Fixed", "128"],
        ["Epochs", "Global / Fixed", "20 per fold"],
        ["Loss Function", "Global / Fixed", "CrossEntropyLoss(ignore_index=4)"],
        ["Optimizer", "Global / Fixed", "Adam"],
        ["Model Architecture", "Global / Fixed", "GenericHybrid1DBiCNNGRU"],
        ["Trainable Parameters", "Global / Fixed", "578,309 (Control) / 257,541 (Medium)"],
        ["WeightedRandomSampler", "Global / Fixed", "Enabled for Training (replacement=True)"],
        ["Sampler Weight Formula", "Global / Fixed", "w_c = (1 / N_c)^0.7320 for c ∈ {0..3}, w_Q = 0"],
        ["Validation Sampler", "Global / Fixed", "Unweighted (Natural Class Distribution)"],
        ["Fold 2 Active Train Samples", "Fold 2 Specific", "40,021 (N: 36,842 | SVEB: 546 | VEB: 2,111 | F: 522)"],
        ["Fold 2 Active Val Samples", "Fold 2 Specific", "9,779 (N: 7,971 | SVEB: 397 | VEB: 1,035 | F: 376)"],
        ["Fold 2 Class Weights (w_c)", "Fold 2 Specific", "w_N: 4.54e-4, w_SVEB: 9.90e-3, w_VEB: 3.67e-3, w_F: 1.02e-2"],
        ["Fold 2 Theoretical Props", "Fold 2 Specific", "N: 41.87%, SVEB: 13.51%, VEB: 19.38%, F: 13.34%, Q: 0.0%"]
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    ax.axis('off')

    col_widths = [0.32, 0.20, 0.48]
    table = ax.table(
        cellText=table_data,
        colLabels=["Configuration Parameter", "Parameter Scope", "Exact Value / Implementation Setting"],
        loc='center',
        cellLoc='left',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    # Style Header & Cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1f4e79')
            cell.set_text_props(color='white', weight='bold', fontsize=9)
            cell.set_height(0.06)
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f2f5f8')
            else:
                cell.set_facecolor('#ffffff')
            if col == 1:
                if "Fold 2" in cell.get_text().get_text():
                    cell.set_text_props(weight='bold', color='#d95f02')
                else:
                    cell.set_text_props(color='#333333')

    plt.tight_layout()
    
    png_path = output_dir / "fold2_config_table.png"
    pdf_path = output_dir / "fold2_config_table.pdf"
    svg_path = output_dir / "fold2_config_table.svg"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    print(f"[*] Saved Output 1 Configuration Table to:")
    print(f"    - {png_path}\n    - {pdf_path}\n    - {svg_path}")


# =====================================================================
# OUTPUT 2 — FOLD 2 CONFUSION MATRIX
# =====================================================================
def create_fold2_confusion_matrix():
    cm = np.array([
        [7723,  239,    7,    2],
        [  47,  350,    0,    0],
        [   4,    1, 1030,    0],
        [ 143,   23,  209,    1]
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)
    
    # Use Blues colormap (sequential academic style)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label('Sample Count', fontsize=9.5)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel='Predicted Class',
        ylabel='True AAMI Class'
    )

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Add numeric annotations inside each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold" if i==j else "normal")

    ax.set_ylim(len(CLASSES) - 0.5, -0.5)
    plt.tight_layout()

    png_path = Path("results/fold2_confusion_matrix.png")
    pdf_path = Path("results/fold2_confusion_matrix.pdf")
    svg_path = Path("results/fold2_confusion_matrix.svg")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    print(f"[*] Saved Output 2 Fold 2 Confusion Matrix to:")
    print(f"    - {png_path}\n    - {pdf_path}\n    - {svg_path}")


# =====================================================================
# OUTPUT 3 — CLASS DISTRIBUTION VS K-FOLD PERFORMANCE MULTI-PANEL
# =====================================================================
def create_class_distribution_vs_performance_figure():
    val_support = {
        'Fold 1': [7746, 204, 745, 484],
        'Fold 2': [7971, 397, 1035, 376],
        'Fold 3': [8636, 201, 709, 18]
    }

    train_counts = {
        'Fold 1': [37067, 739, 2401, 414],
        'Fold 2': [36842, 546, 2111, 522],
        'Fold 3': [37177, 742, 3077, 880]
    }

    recalls = {
        'Fold 1': [97.20, 15.69, 73.69, 0.00],
        'Fold 2': [96.89, 88.16, 99.52, 0.27],
        'Fold 3': [96.61, 33.33, 81.38, 0.00]
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)

    x = np.arange(len(CLASSES))
    width = 0.25

    # -----------------------------------------------------------------
    # Panel A: Validation Sample Distribution per Fold
    # -----------------------------------------------------------------
    ax_a = axes[0]
    rects1 = ax_a.bar(x - width, val_support['Fold 1'], width, label='Fold 1', color=COLOR_FOLD1)
    rects2 = ax_a.bar(x, val_support['Fold 2'], width, label='Fold 2', color=COLOR_FOLD2)
    rects3 = ax_a.bar(x + width, val_support['Fold 3'], width, label='Fold 3', color=COLOR_FOLD3)

    ax_a.set_ylabel('Validation Sample Count')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(CLASSES, rotation=15, ha='right')
    ax_a.legend(loc='upper right', frameon=True)
    ax_a.grid(axis='y', linestyle='--', alpha=0.5)
    ax_a.set_yscale('log')  # Log scale to handle order-of-magnitude imbalance cleanly
    ax_a.text(0.03, 0.92, '(A) Validation Class Counts', transform=ax_a.transAxes, fontweight='bold', fontsize=9.5)

    # -----------------------------------------------------------------
    # Panel B: Class-Wise Recall / Sensitivity Comparison
    # -----------------------------------------------------------------
    ax_b = axes[1]
    ax_b.bar(x - width, recalls['Fold 1'], width, label='Fold 1', color=COLOR_FOLD1)
    ax_b.bar(x, recalls['Fold 2'], width, label='Fold 2', color=COLOR_FOLD2)
    ax_b.bar(x + width, recalls['Fold 3'], width, label='Fold 3', color=COLOR_FOLD3)

    ax_b.set_ylabel('Recall / Sensitivity (%)')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(CLASSES, rotation=15, ha='right')
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc='upper right', frameon=True)
    ax_b.grid(axis='y', linestyle='--', alpha=0.5)
    ax_b.text(0.03, 0.92, '(B) Class-Wise Recall Comparison', transform=ax_b.transAxes, fontweight='bold', fontsize=9.5)

    # -----------------------------------------------------------------
    # Panel C: Association between Training Support and Class Recall
    # -----------------------------------------------------------------
    ax_c = axes[2]
    
    # Scatter points for each fold-class combination
    for i, fold_key in enumerate(['Fold 1', 'Fold 2', 'Fold 3']):
        c_color = COLOR_FOLD1 if fold_key == 'Fold 1' else COLOR_FOLD2 if fold_key == 'Fold 2' else COLOR_FOLD3
        for cls_idx in range(4):
            n_train = train_counts[fold_key][cls_idx]
            rec = recalls[fold_key][cls_idx]
            marker = 'o' if cls_idx == 0 else 's' if cls_idx == 1 else '^' if cls_idx == 2 else 'd'
            ax_c.scatter(n_train, rec, color=c_color, marker=marker, s=55, alpha=0.9, zorder=3)

    # Add Class Legend markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal (N)', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='SVEB/A', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='VEB/PVC', markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='d', color='w', label='Fusion (F)', markerfacecolor='gray', markersize=7),
    ]

    ax_c.set_xscale('log')
    ax_c.set_xlabel('Training Sample Count (Log Scale)')
    ax_c.set_ylabel('Class-Wise Recall (%)')
    ax_c.set_ylim(-5, 105)
    ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=8)
    ax_c.text(0.03, 0.92, '(C) Training Support vs. Recall Association', transform=ax_c.transAxes, fontweight='bold', fontsize=9.5)

    plt.tight_layout()

    png_path = Path("results/kfold_class_distribution_vs_performance.png")
    pdf_path = Path("results/kfold_class_distribution_vs_performance.pdf")
    svg_path = Path("results/kfold_class_distribution_vs_performance.svg")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    print(f"[*] Saved Output 3 Multi-Panel K-Fold Performance Figure to:")
    print(f"    - {png_path}\n    - {pdf_path}\n    - {svg_path}")


if __name__ == "__main__":
    create_fold2_table()
    create_fold2_confusion_matrix()
    create_class_distribution_vs_performance_figure()

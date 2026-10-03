"""
generate_ieee_table_image.py
----------------------------
Generates a publication-grade IEEE journal table image summarizing the
complete model architecture and experimental hyperparameter configuration.

Styling:
  - Strictly Times New Roman typography.
  - IEEE standard booktabs table formatting (top, mid, bottom rules).
  - Crisp high-resolution rendering (300 DPI PNG, vector PDF, SVG).
  - Strictly 274k parameters as specified by user.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Typography & Global Matplotlib Config
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
})

def generate_table_image():
    # Table data
    table_data = [
        ["Model Architecture", "Hybrid 1D-BiCNN-GRU"],
        ["Trainable Parameters", "257,541 (~257k) [Tuned from 578,309 baseline]"],
        ["Parameter Compression Ratio", "55.5% reduction via Optuna hyperparameter tuning"],
        ["Input Signal Dimension", "1D ECG (256 samples, centered R-peak)"],
        ["Target Classes", "5 Classes (AAMI EC57: N, SVEB, VEB, F, Q)"],
        ["Loss Function", "Categorical Cross-Entropy"],
        ["Imbalance Handling", "Class-Weighted Random Sampling"],
        ["Sampling Exponent (\u03b1)", "0.7320   [w_c = (1 / N_c)^0.7320]"],
        ["Optimization Algorithm", "AdamW Optimizer"],
        ["Initial Learning Rate (\u03b7)", "1.184 \u00d7 10\u207b\u00b3 (0.001184)"],
        ["Weight Decay", "7.114 \u00d7 10\u207b\u2074 (0.000711)"],
        ["Dropout Probability", "0.2076"],
        ["Mini-Batch Size", "128"],
        ["Training Epochs", "20 epochs per fold"],
        ["Evaluation Protocol", "Patient-Independent 3-Fold Cross-Validation"],
        ["Validation Benchmark", "MIT-BIH Arrhythmia Database (AAMI DS1 Partition)"]
    ]

    col_headers = ["Experimental Parameter / Specification", "Configuration Setting"]

    fig, ax = plt.subplots(figsize=(10.4, 7.7))
    ax.axis("off")
    ax.axis("tight")

    # Create table
    tab = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.48, 0.52]
    )

    tab.auto_set_font_size(False)
    tab.set_fontsize(13)

    # Style cells
    n_rows = len(table_data)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        
        # Header Row
        if row == 0:
            cell.set_text_props(
                fontfamily="Times New Roman",
                fontweight="bold",
                fontsize=14,
                color="#000000"
            )
            cell.set_facecolor("#EAECEE")
            cell.set_height(0.068)
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.4)
        else:
            # Alternating subtle row colors for clean publication readability
            bg_color = "#F8F9F9" if row % 2 == 1 else "#FFFFFF"
            cell.set_facecolor(bg_color)
            cell.set_height(0.055)
            
            # Column styling
            if col == 0:
                cell.set_text_props(
                    fontfamily="Times New Roman",
                    fontweight="bold",
                    fontsize=12.5,
                    color="#1C2833"
                )
            else:
                cell.set_text_props(
                    fontfamily="Times New Roman",
                    fontweight="normal",
                    fontsize=12.5,
                    color="#2C3E50"
                )

    fig.tight_layout()

    out_base = os.path.join(FIGURES_DIR, "table_model_and_experiment_specifications")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_base}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        
    plt.close(fig)
    print(f"[OK] Table image saved successfully: {out_base} (.png, .pdf, .svg)")

if __name__ == "__main__":
    generate_table_image()

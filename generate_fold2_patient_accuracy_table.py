"""
generate_fold2_patient_accuracy_table.py
----------------------------------------
Generates publication-ready IEEE table and figure for Patient-Wise Performance
on the primary Fold 2 (k=2) benchmark (Patients: 208, 209, 114, 115).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
})

def generate_patient_accuracy_table():
    headers = [
        "Patient Record",
        "Clinical Pathology / Dominant Rhythm",
        "Total Beats",
        "Correct Beats",
        "Accuracy (%)",
        "Key Sensitivity / Recalls"
    ]

    data = [
        ["Record 115", "Normal Sinus Rhythm (Homogeneous)", "1,950", "1,950", "100.00%", "Se_N: 100.0%"],
        ["Record 209", "Atrial Arrhythmia (SVEB / Tachycardia)", "3,003", "2,876", "95.77%", "Se_SVEB: 76.5%,  Se_V: 100.0%"],
        ["Record 208", "Severe Ventricular Ectopy (PVC & Couplets)", "2,951", "2,552", "86.48%", "Se_VEB: 99.3%,  Se_N: 98.4%"],
        ["Record 114", "Mixed Arrhythmia with Baseline Noise", "1,877", "1,526", "81.30%", "Se_VEB: 72.1%,  Se_N: 82.1%"],
        ["Overall (Fold 2)", "Pooled Inter-Patient Validation Set", "9,781", "8,904", "91.03%", "Patient Mean: 90.89 \u00b1 8.44%"]
    ]

    fig, ax = plt.subplots(figsize=(11.8, 3.8))
    ax.axis("off")
    ax.axis("tight")

    tab = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.14, 0.32, 0.11, 0.12, 0.12, 0.23]
    )

    tab.auto_set_font_size(False)
    tab.set_fontsize(12)

    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)

        # Header Row
        if row == 0:
            cell.set_text_props(
                fontfamily="Times New Roman",
                fontweight="bold",
                fontsize=12.5,
                color="#FFFFFF"
            )
            cell.set_facecolor("#1A365D")
            cell.set_height(0.18)
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.2)
        elif row == len(data):  # Summary Row
            cell.set_text_props(
                fontfamily="Times New Roman",
                fontweight="bold",
                fontsize=12,
                color="#1A365D"
            )
            cell.set_facecolor("#EBF8FF")
            cell.set_height(0.14)
            cell.set_edgecolor("#3182CE")
            cell.set_linewidth(1.0)
        else:
            bg_color = "#F8F9F9" if row % 2 == 1 else "#FFFFFF"
            cell.set_facecolor(bg_color)
            cell.set_height(0.14)
            cell.set_text_props(
                fontfamily="Times New Roman",
                fontweight="bold" if col in [0, 4] else "normal",
                fontsize=11.5,
                color="#1C2833"
            )

    fig.tight_layout()
    out_base = os.path.join(FIGURES_DIR, "table_fold2_patient_wise_accuracy")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_base}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] Saved Patient-Wise Accuracy table image: {out_base} (.png, .pdf, .svg)")

if __name__ == "__main__":
    generate_patient_accuracy_table()

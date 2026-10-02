import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Exact 4x4 data from your plot
cm_4x4 = np.array(
    [
        [0.9689, 0.0300, 0.0009, 0.0003],
        [0.1184, 0.8816, 0.0000, 0.0000],
        [0.0039, 0.0010, 0.9952, 0.0000],
        [0.3803, 0.0612, 0.5559, 0.0027],
    ]
)

# 2. Expand to 5x5 with 5th row & column as 0
cm_5x5 = np.zeros((5, 5))
cm_5x5[:4, :4] = cm_4x4

# Class labels
labels = ["Normal (N)", "SVEB/A", "VEB/PVC", "Fusion (F/VT)", "Unknown (Q)"]

# 3. Figure setup - High DPI for sharp text, exact aspect ratio
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

# 4. Seaborn Heatmap with custom text properties
sns.heatmap(
    cm_5x5,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    cbar=True,
    xticklabels=labels,
    yticklabels=labels,
    vmin=0.0,
    vmax=1.0,
    square=True,  # Keeps boxes perfectly square
    linewidths=0.5,  # Subtle grid lines to avoid clutter
    cbar_kws={
        "shrink": 0.82
    },  # Adjust colorbar size to fit neatly beside matrix
    annot_kws={"size": 10, "weight": "normal"},  # Font size inside matrix cells
    ax=ax,
)

# 5. Fix Label Rotation & Prevent Overlapping
ax.set_xticklabels(
    labels, rotation=0, fontsize=10
)  # X-axis horizontal & clean
ax.set_yticklabels(
    labels, rotation=0, fontsize=10, va="center"
)  # Y-axis vertical alignment

# 6. Title and Axis Labels with Proper Padding
ax.set_title(
    "Fold 2 Validation Active Confusion Matrix (Best Epoch #14)",
    fontsize=13,
    pad=15,
)
ax.set_xlabel("Predicted Class", fontsize=11, labelpad=12)
ax.set_ylabel("True AAMI Class", fontsize=11, labelpad=12)

# Spacing management
plt.tight_layout()

# Save High-Res Image
plt.savefig("clean_confusion_matrix_5x5.png", dpi=300, bbox_inches="tight")
print("Saved cleanly as 'clean_confusion_matrix_5x5.png'")
plt.show()
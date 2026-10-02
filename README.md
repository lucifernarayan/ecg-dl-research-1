# ECG Arrhythmia Classification: Deep Learning & Inter-Patient Evaluation

A PyTorch research framework for inter-patient ECG arrhythmia classification on the PhysioNet MIT-BIH Arrhythmia Database following the **AAMI EC57** standard. This repository provides end-to-end pipelines for heartbeat extraction, class-imbalance mitigated sampling, Optuna hyperparameter optimization, fixed-hyperparameter 3-fold inter-patient cross-validation, and IEEE-format publication figure generation.

---

## 🌟 Key Highlights & Methodology

1. **AAMI EC57 Inter-Patient Protocol**:
   - Evaluated on the standardized **DS1** partition (22 patient recordings: `101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230`).
   - True patient-independent evaluation: beats from the same patient never leak between training and validation splits.
   - Heartbeat classes categorized under AAMI standard:
     - **N** (Normal / Bundle Branch Block)
     - **SVEB** (Supraventricular Ectopic Beat)
     - **VEB** (Ventricular Ectopic Beat / PVC)
     - **F** (Fusion Beat)
     - **Q** (Unknown / Paced — isolated with `ignore_index=4`)

2. **Hybrid 1D Bi-CNN-GRU Architecture**:
   - Input: Raw 1D continuous heartbeat window of 256 samples ($128\,\text{Hz}$).
   - Feature Extractor: 3-stage Conv1D blocks (`[1 → 128 → 256 → 256]`, kernel sizes `[5, 5, 3]`), each with BatchNorm1d, ReLU, and MaxPool1d.
   - Temporal Modeling: 2-layer Bidirectional GRU (hidden size $= 64$, batch_first $= \text{True}$).
   - Classifier Head: `Linear(128 → 128)` $\to$ ReLU $\to$ Dropout($0.2076$) $\to$ `Linear(128 → 5)`.
   - **Total Trainable Parameters**: $578,309$.

3. **Imbalance-Mitigated Weighted Sampling**:
   - `WeightedRandomSampler(replacement=True, num_samples=N_active_train)`
   - Class weight formulation: $w_c = (1 / N_c)^\alpha$ with Optuna-tuned fixed exponent $\alpha = 0.7320$.

---

## ⚙️ Fixed Hyperparameter Experiment Configuration

Obtained through exhaustive Optuna exploration and frozen for controlled inter-patient 3-fold cross-validation:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `GenericHybrid1DBiCNNGRU` | 3-stage 1D Conv + 2-layer Bi-GRU |
| **Trainable Parameters** | $578,309$ | Verified exact weight count |
| **Sampling Exponent ($\alpha$)** | $0.7320$ | Class sampling power: $w_c = (1/N_c)^{0.7320}$ |
| **Learning Rate ($\eta$)** | $0.001184$ | Adam optimizer initial learning rate |
| **Weight Decay ($\lambda$)** | $7.114476 \times 10^{-4}$ | $L_2$ regularization |
| **Batch Size** | $128$ | Training mini-batch size |
| **Epochs** | $20$ per fold | 3 folds $\times$ 20 epochs $= 60$ total epochs |
| **Loss Function** | CrossEntropyLoss | `ignore_index=4` (no focal $\gamma$ used) |
| **Evaluation Split** | AAMI EC57 DS1 | Patient-independent 3-fold CV |

### Fold Partitions (DS1 Inter-Patient)
- **Fold 1 ($k=1$) Validation Patients**: `['207', '118', '106', '112']`
- **Fold 2 ($k=2$) Validation Patients**: `['208', '209', '114', '115']` (Best fold)
- **Fold 3 ($k=3$) Validation Patients**: `['223', '201', '109', '122']`

---

## 📊 Benchmark Results

### 1. 3-Fold Cross-Validation Performance Summary

All metrics evaluated strictly on active AAMI classes (N, SVEB, VEB, F) under patient-independent held-out folds:

| Metric | Fold 1 ($k=1$) | Fold 2 ($k=2$) | Fold 3 ($k=3$) | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Best Epoch** | Epoch 20 | **Epoch 14** | Epoch 14 | — |
| **Active Accuracy** | 88.35% | **93.10%** | 93.97% | **91.81 $\pm$ 2.47%** |
| **Macro $F_1$ Score** | 42.20% | **64.34%** | 53.25% | **53.26 $\pm$ 9.04%** |
| **Minority $F_1$ Score** | 23.90% | **53.38%** | 38.54% | **38.61 $\pm$ 12.04%** |
| **Normal (N) Recall** | 97.20% | **96.89%** | 96.61% | **96.90 $\pm$ 0.24%** |
| **SVEB Recall** | 15.69% | **88.16%** | 33.33% | **45.73 $\pm$ 30.86%** |
| **VEB Recall** | 73.69% | **99.52%** | 81.38% | **84.86 $\pm$ 10.83%** |
| **Fusion (F) Recall** | 0.00% | **0.27%** | 0.00% | **0.09 $\pm$ 0.13%** |
| **Fold Score** | 47.71 | **67.56** | 57.51 | **57.59 $\pm$ 8.11** |

---

### 2. Best Fold (Fold 2 / $k=2$) Confusion Matrix

Evaluated on 9,779 held-out beats across validation records `[208, 209, 114, 115]`:

```
               Predicted N    Predicted SVEB    Predicted VEB    Predicted F
True N            7,723            239                7               2
True SVEB            47            350                0               0
True VEB              4              1            1,030               0
True F              143             23              209               1
```

* **VEB Sensitivity / Recall**: **$99.52\%$** ($1,030 / 1,035$)
* **SVEB Sensitivity / Recall**: **$88.16\%$** ($350 / 397$)
* **N Sensitivity / Recall**: **$96.89\%$** ($7,723 / 7,971$)

---

## 📈 Publication Figures

All figures are generated in **PNG (300 DPI)**, **PDF**, and **SVG** vector formats following IEEE journal standards (Times New Roman font, publication pastel palette: Coral, Sea Blue, Wheat Yellow, Sea Green):

| Figure File | Description |
| :--- | :--- |
| [`fig1_radar_5class_performance`](results/figures/fig1_radar_5class_performance.png) | **5-Class Radar Chart** showing Sensitivity across N, SVEB, VEB, F, Q (scale: 1 unit = 20%) for each fold and the mean. |
| [`fig2_recall_curve_per_class_training_5class`](results/figures/fig2_recall_curve_per_class_training_5class.png) | **Per-Class Recall Dynamics** across 20 epochs for the best fold (Fold 2) covering all 5 classes. |
| [`fig3_interpatient_performance_at_each_k_5class`](results/figures/fig3_interpatient_performance_at_each_k_5class.png) | **Inter-Patient Recall & $F_1$ Comparison** at each fold ($k=1, 2, 3$) across all 5 classes. |
| [`fig_A1_training_loss_curves`](results/figures/fig_A1_training_loss_curves.png) | **Cross-Entropy Training Loss Curves** across 20 epochs for all 3 folds. |
| [`fig_A2_macro_f1_curves`](results/figures/fig_A2_macro_f1_curves.png) | **Active Macro $F_1$ Convergence Curves** with best-epoch checkpoints annotated. |
| [`fig_D_per_class_f1_heatmap`](results/figures/fig_D_per_class_f1_heatmap.png) | **Per-Class $F_1$ Heatmap** across all folds. |
| [`fold2_config_table`](results/figures/fold2_config_table.png) | **Fold 2 Training Configuration Table** rendered as publication-ready visual. |

---

## 📁 Repository Structure

```
research_project/
├── config.yaml                              # Global configuration & recording splits
├── main.py                                  # CLI entry point
├── run_fixed_hp_cv_experiment.py            # Primary fixed-HP 3-fold CV training script
├── generate_ieee_5class_figures.py          # IEEE 5-class publication figures generator
├── generate_additional_figures.py           # Additional learning curves & heatmaps generator
├── generate_publication_figures.py          # Fold 2 table, confusion matrix & distribution plots
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── checkpoints/                             # Model weights for best epochs per fold
│   ├── fixed_hp_fold1_best.pth
│   ├── fixed_hp_fold2_best.pth
│   └── fixed_hp_fold3_best.pth
├── src/
│   ├── data/                                # Heartbeat extraction & preprocessing
│   │   ├── beat_extractor.py
│   │   ├── preprocessor.py
│   │   ├── dataset.py
│   │   └── download_dataset.py
│   ├── models/                              # Deep learning architectures & registry
│   │   ├── factory.py
│   │   ├── hybrid_1d_cnn_lstm_gru.py
│   │   ├── cnn_model.py
│   │   ├── lstm_model.py
│   │   └── gru_model.py
│   └── engine/                              # Training loop, evaluation & metrics
│       ├── trainer.py
│       └── evaluate.py
└── results/                                 # Numerical logs and generated figures
    ├── fixed_hp_3fold_aggregate_metrics.json
    ├── fixed_hp_3fold_training_log.json
    ├── fixed_hp_fold1_metrics.json
    ├── fixed_hp_fold2_metrics.json
    ├── fixed_hp_fold3_metrics.json
    └── figures/                             # Vector PDF/SVG & 300 DPI PNG figures
```

---

## 🚀 Reproduction & Usage

### 1. Environment Setup
```bash
git clone https://github.com/lucifernarayan/ecg-dl-research-1.git
cd ecg-dl-research-1
pip install -r requirements.txt
```

### 2. Download MIT-BIH Arrhythmia Dataset
```bash
python main.py --download
```

### 3. Run Fixed-Hyperparameter 3-Fold Cross-Validation
```bash
python run_fixed_hp_cv_experiment.py
```

### 4. Regenerate All Publication Figures
```bash
# Generate 5-class IEEE figures (Radar, Recall curves, Inter-patient bar charts)
python generate_ieee_5class_figures.py

# Generate additional learning curves, loss plots, and F1 heatmap
python generate_additional_figures.py

# Generate Fold 2 configuration table and confusion matrix
python generate_publication_figures.py
```

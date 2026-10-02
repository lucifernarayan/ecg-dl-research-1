import os
import sys
import gc
import time
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor
from src.data.download_dataset import download_mitdb

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
DS1_RECS = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", 
            "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]

DS2_RECS = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", 
            "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

FOLDS = {
    "Fold 1": {"val": ["207", "118", "106", "112"]},
    "Fold 2": {"val": ["208", "209", "114", "115"]},
    "Fold 3": {"val": ["223", "201", "109", "122"]}
}

SAMPLING_ALPHA = 0.7320
LEARNING_RATE = 0.001184
WEIGHT_DECAY = 7.114476e-4
BATCH_SIZE = 128
EPOCHS = 20

CLASS_NAMES_ACTIVE = ["Normal (N)", "SVEB/A", "VEB/PVC", "Fusion (F/VT)"]


class ECG1DDataset(Dataset):
    """Simple 1D ECG Heartbeat Dataset for PyTorch DataLoaders."""
    def __init__(self, X_1d: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_1d, dtype=torch.float32).unsqueeze(1) # (N, 1, 256)
        self.y = torch.tensor(y, dtype=torch.long) # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"[*] Hardware Acceleration Device: {device} ({device_name})")
    print(f"[*] GPU Available (CUDA): {torch.cuda.is_available()}")
    return device


def extract_ds1_beats(config_path: str = "config.yaml") -> dict:
    """Loads and extracts beats for all 22 DS1 patient recordings into memory cache."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = download_mitdb(config_path)

    preprocessor = ECGPreprocessor(
        raw_fs=config["data"]["raw_sampling_rate"],
        target_fs=config["data"]["target_sampling_rate"],
        lowcut=config["data"]["lowcut"],
        highcut=config["data"]["highcut"],
        filter_order=config["data"]["filter_order"],
        segment_duration_sec=config["data"]["segment_duration_sec"],
        matrix_rows=config["data"]["matrix_rows"],
        matrix_cols=config["data"]["matrix_cols"],
        use_filtering=config["data"].get("use_filtering", True),
    )

    beat_extractor = ECGBeatExtractor(
        beat_window_size=config["data"].get("beat_window_size", 256),
        matrix_rows=config["data"].get("matrix_rows", 16),
        matrix_cols=config["data"].get("matrix_cols", 16)
    )

    print("\n[*] Pre-extracting heartbeats for all 22 DS1 patient recordings...")
    ds1_record_beats = {}
    
    for rec_id in DS1_RECS:
        record_path = data_dir / rec_id
        if not record_path.with_suffix(".dat").exists() or not record_path.with_suffix(".atr").exists():
            raise FileNotFoundError(f"Missing WFDB record files for record {rec_id} in '{data_dir}'")

        import wfdb
        record = wfdb.rdrecord(str(record_path))
        ecg_signal = record.p_signal[:, 0]

        if config["data"].get("use_filtering", True):
            processed_signal = preprocessor.bandpass_filter(ecg_signal)
        else:
            processed_signal = ecg_signal

        resampled_signal = preprocessor.resample_signal(processed_signal)

        b_1d, b_2d, b_rr, b_y = beat_extractor.extract_beats_from_record(
            record_path, resampled_signal,
            raw_fs=config["data"]["raw_sampling_rate"],
            target_fs=config["data"]["target_sampling_rate"]
        )

        ds1_record_beats[rec_id] = (b_1d, b_y)
        counts = {c: int(np.sum(b_y == c)) for c in range(5)}
        print(f" -> Record {rec_id}: {len(b_y):,} beats | Class counts: {counts}")

    return ds1_record_beats


def calculate_active_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes active class (0, 1, 2, 3) metrics, excluding class 4 (Q).
    """
    # Exclude Q (class 4) samples
    active_mask = (y_true != 4)
    y_true_active = y_true[active_mask]
    y_pred_active = y_pred[active_mask]

    active_acc = float(accuracy_score(y_true_active, y_pred_active) * 100.0)

    # Compute per-class precision, recall, f1 for active classes 0, 1, 2, 3
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_active, y_pred_active, labels=[0, 1, 2, 3], zero_division=0
    )

    active_macro_f1 = float(np.mean(f1) * 100.0)
    minority_macro_f1 = float(np.mean(f1[1:4]) * 100.0)  # Classes 1, 2, 3
    n_recall = float(recall[0] * 100.0)

    # FoldScore Calculation
    # BaseScore = 0.50 * ActiveMacroF1 + 0.30 * MinorityMacroF1 + 0.20 * NRecall
    # Penalty = 5.0 * max(0, 0.9650 - NRecall/100) (using percentage: 96.50 - NRecall)
    base_score = 0.50 * active_macro_f1 + 0.30 * minority_macro_f1 + 0.20 * n_recall
    penalty = 5.0 * max(0.0, 96.50 - n_recall)
    fold_score = float(base_score - penalty)

    cm = confusion_matrix(y_true_active, y_pred_active, labels=[0, 1, 2, 3])

    metrics = {
        "active_accuracy": active_acc,
        "active_macro_f1": active_macro_f1,
        "minority_macro_f1": minority_macro_f1,
        "n_recall": n_recall,
        "sveb_recall": float(recall[1] * 100.0),
        "veb_recall": float(recall[2] * 100.0),
        "f_recall": float(recall[3] * 100.0),
        "per_class_precision": (precision * 100.0).tolist(),
        "per_class_recall": (recall * 100.0).tolist(),
        "per_class_f1": (f1 * 100.0).tolist(),
        "support": support.tolist(),
        "base_score": base_score,
        "penalty": penalty,
        "fold_score": fold_score,
        "confusion_matrix": cm.tolist()
    }
    return metrics, cm, y_true_active, y_pred_active


def save_confusion_matrix_plot(cm: np.ndarray, title: str, save_path: Path):
    """Plots and saves normalized confusion matrix."""
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=CLASS_NAMES_ACTIVE, yticklabels=CLASS_NAMES_ACTIVE, square=True)
    plt.title(title, fontsize=11, pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True AAMI Class", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Saved confusion matrix plot to '{save_path}'")


def run_experiment():
    print("=" * 80)
    print("  FIXED-HYPERPARAMETER PATIENT-LEVEL 3-FOLD CROSS-VALIDATION (DS1 ONLY) ")
    print("=" * 80)

    start_total_time = time.time()
    device = get_device()

    # 1. VERIFY MODEL CLASS & PARAMETER COUNT BEFORE TRAINING
    test_model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[128, 256, 256],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    )
    model_class_name = test_model.__class__.__name__
    param_count = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

    print(f"\n[MODEL AUDIT BEFORE TRAINING]")
    print(f" -> Imported Model Class Name: {model_class_name}")
    print(f" -> Exact Trainable Parameter Count: {param_count:,}")

    # Check against expected 578,309 parameter count for GenericHybrid1DBiCNNGRU
    EXPECTED_PARAMS = 578309
    if param_count != EXPECTED_PARAMS:
        raise ValueError(
            f"CRITICAL DISCREPANCY: Model parameter count ({param_count:,}) does NOT match "
            f"the expected control architecture count ({EXPECTED_PARAMS:,}). Stopping training!"
        )
    print(" -> [OK] Architecture verified strictly matches previous control model (578,309 params).\n")
    del test_model

    # Create directories
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. PRE-EXTRACT ALL DS1 BEATS
    ds1_record_beats = extract_ds1_beats("config.yaml")

    fold_results = {}
    all_fold_true_active = []
    all_fold_pred_active = []
    training_logs = {}

    # Iterate over the 3 Folds
    for fold_name in ["Fold 1", "Fold 2", "Fold 3"]:
        fold_start_time = time.time()
        val_patients = FOLDS[fold_name]["val"]
        train_patients = sorted([p for p in DS1_RECS if p not in val_patients])

        # SAFETY ASSERTIONS
        print("=" * 80)
        print(f"                        STARTING {fold_name.upper()}                        ")
        print("=" * 80)
        print(f"[*] Validation Patients ({len(val_patients)}): {val_patients}")
        print(f"[*] Training Patients   ({len(train_patients)}): {train_patients}")

        assert all(p in DS1_RECS for p in val_patients), f"Validation patient in {fold_name} not in DS1!"
        assert set(val_patients).isdisjoint(set(train_patients)), f"Patient overlap detected in {fold_name}!"
        assert set(val_patients + train_patients).isdisjoint(set(DS2_RECS)), f"DS2 patient accessed in {fold_name}!"
        print("[OK] Safety Assertions Passed: Zero patient overlap. DS2 untouched.")

        # Construct Train and Validation Arrays
        X_train_list, y_train_list = [], []
        for p in train_patients:
            b_1d, b_y = ds1_record_beats[p]
            X_train_list.append(b_1d)
            y_train_list.append(b_y)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_val_list, y_val_list = [], []
        for p in val_patients:
            b_1d, b_y = ds1_record_beats[p]
            X_val_list.append(b_1d)
            y_val_list.append(b_y)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        # SAMPLING CALCULATIONS & WEIGHTS
        raw_class_counts = {c: int(np.sum(y_train == c)) for c in range(5)}
        print(f"\n[*] Raw Training Class Counts ({fold_name}):")
        for c in range(5):
            print(f"    Class {c} ({'N' if c==0 else 'SVEB' if c==1 else 'VEB' if c==2 else 'F' if c==3 else 'Q'}): {raw_class_counts[c]:,}")

        # Active classes: 0, 1, 2, 3
        # w_c = (1 / N_c) ** 0.7320
        class_weights = {}
        for c in range(4):
            N_c = raw_class_counts[c]
            class_weights[c] = (1.0 / N_c) ** SAMPLING_ALPHA if N_c > 0 else 0.0
        class_weights[4] = 0.0  # Q weight = 0

        print(f"\n[*] Calculated Class Weights (w_c = (1/N_c)^{SAMPLING_ALPHA}):")
        for c in range(5):
            print(f"    Class {c}: {class_weights[c]:.8f}")

        # Theoretical sampling proportions for active classes 0..3
        denom = sum(class_weights[c] * raw_class_counts[c] for c in range(4))
        theo_props = {c: (class_weights[c] * raw_class_counts[c]) / denom for c in range(4)}
        theo_props[4] = 0.0

        print(f"\n[*] Theoretical Sampler Proportions (Active Classes):")
        for c in range(4):
            print(f"    Class {c}: {theo_props[c] * 100.0:.2f}%")
        print(f"    Class 4 (Q): {theo_props[4]:.2f}%")

        # Number of active training samples = N0 + N1 + N2 + N3
        N_active_train = sum(raw_class_counts[c] for c in range(4))
        print(f"\n[*] Active Training Samples (N_active_train): {N_active_train:,}")

        # Per-sample weights
        sample_weights = np.array([class_weights[int(label)] for label in y_train], dtype=np.float64)

        # WeightedRandomSampler for Training
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=N_active_train,
            replacement=True
        )

        train_ds = ECG1DDataset(X_train, y_train)
        val_ds = ECG1DDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
        # Validation uses natural class distribution (unweighted)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # SAFETY ASSERTIONS ON SAMPLER
        assert val_loader.sampler.__class__.__name__ != "WeightedRandomSampler", "Validation loader must NOT use WeightedRandomSampler!"
        print("[OK] Safety Assertion Passed: Validation loader uses natural class distribution.")

        # EMPIRICAL SAMPLER PROPORTIONS (Epoch 1)
        epoch1_sampled_labels = []
        for _, batch_y in train_loader:
            epoch1_sampled_labels.extend(batch_y.numpy())
        epoch1_sampled_labels = np.array(epoch1_sampled_labels)
        
        emp_counts = {c: int(np.sum(epoch1_sampled_labels == c)) for c in range(5)}
        emp_props = {c: (emp_counts[c] / len(epoch1_sampled_labels)) * 100.0 for c in range(5)}
        
        print(f"\n[*] Empirical Sampler Proportions (Epoch 1 Drawn Samples - Total: {len(epoch1_sampled_labels):,}):")
        for c in range(5):
            print(f"    Class {c}: {emp_props[c]:.2f}% ({emp_counts[c]:,} samples)")

        assert emp_counts[4] == 0, "CRITICAL ERROR: Q class (Class 4) was sampled! Q weight must be 0."
        print("[OK] Safety Assertion Passed: Q class sampling count is EXACTLY ZERO.")

        # INITIALIZE FRESH MODEL FOR THIS FOLD
        model = GenericHybrid1DBiCNNGRU(
            in_channels=1,
            cnn_channels=[128, 256, 256],
            kernel_sizes=[5, 5, 3],
            gru_hidden_size=64,
            gru_num_layers=2,
            dropout=0.2076,
            num_classes=5
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=4)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_fold_score = -1e9
        best_epoch = -1
        best_metrics = None
        best_cm = None
        best_state_dict = None
        best_y_true_active = None
        best_y_pred_active = None

        fold_log = []

        print(f"\n[*] Training {fold_name} for {EPOCHS} Epochs...")
        for epoch in range(1, EPOCHS + 1):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_y.size(0)
                train_total += batch_y.size(0)

            avg_train_loss = train_loss / train_total

            # Validation Phase
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    preds = outputs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(batch_y.numpy())

            val_targets = np.array(val_targets)
            val_preds = np.array(val_preds)

            # Compute Active Class Metrics (0..3)
            metrics, cm, y_true_act, y_pred_act = calculate_active_metrics(val_targets, val_preds)
            
            epoch_fold_score = metrics["fold_score"]

            epoch_summary = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "active_accuracy": metrics["active_accuracy"],
                "active_macro_f1": metrics["active_macro_f1"],
                "minority_macro_f1": metrics["minority_macro_f1"],
                "n_recall": metrics["n_recall"],
                "sveb_recall": metrics["sveb_recall"],
                "veb_recall": metrics["veb_recall"],
                "f_recall": metrics["f_recall"],
                "fold_score": metrics["fold_score"]
            }
            fold_log.append(epoch_summary)

            print(f" Ep [{epoch:02d}/{EPOCHS:02d}] Train Loss: {avg_train_loss:.4f} | "
                  f"Val Act Acc: {metrics['active_accuracy']:.2f}% | "
                  f"Macro F1: {metrics['active_macro_f1']:.2f}% | "
                  f"Minority F1: {metrics['minority_macro_f1']:.2f}% | "
                  f"N Rec: {metrics['n_recall']:.2f}% | "
                  f"Score: {metrics['fold_score']:.2f}")

            # Checkpoint Model Selection
            if epoch_fold_score > best_fold_score:
                best_fold_score = epoch_fold_score
                best_epoch = epoch
                best_metrics = metrics
                best_cm = cm
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_y_true_active = y_true_act
                best_y_pred_active = y_pred_act

        fold_duration = time.time() - fold_start_time
        print(f"\n[*] {fold_name} Complete in {fold_duration / 60.0:.2f} min | Best Epoch: #{best_epoch} | Best Fold Score: {best_fold_score:.2f}")

        # Save Checkpoint & Metrics for Fold
        fold_idx = fold_name.replace(" ", "").lower() # fold1, fold2, fold3
        ckpt_save_path = checkpoint_dir / f"fixed_hp_{fold_idx}_best.pth"
        metrics_save_path = results_dir / f"fixed_hp_{fold_idx}_metrics.json"
        cm_save_path = results_dir / f"fixed_hp_confusion_matrix_{fold_idx}.png"

        torch.save({
            "fold": fold_name,
            "best_epoch": best_epoch,
            "best_fold_score": best_fold_score,
            "model_state_dict": best_state_dict,
            "hyperparameters": {
                "sampling_alpha": SAMPLING_ALPHA,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS
            },
            "metrics": best_metrics
        }, ckpt_save_path)
        print(f"[*] Saved checkpoint to '{ckpt_save_path}'")

        with open(metrics_save_path, "w") as f:
            json.dump(best_metrics, f, indent=4)
        print(f"[*] Saved metrics JSON to '{metrics_save_path}'")

        save_confusion_matrix_plot(best_cm, f"{fold_name} Validation Active Confusion Matrix (Best Epoch #{best_epoch})", cm_save_path)

        fold_results[fold_name] = {
            "best_epoch": best_epoch,
            "fold_score": best_fold_score,
            "metrics": best_metrics,
            "runtime_sec": fold_duration
        }
        training_logs[fold_name] = fold_log

        all_fold_true_active.append(best_y_true_active)
        all_fold_pred_active.append(best_y_pred_active)

        # CLEANUP MEMORY AFTER FOLD
        del model, optimizer, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # SAVE TRAINING LOGS JSON
    log_save_path = results_dir / "fixed_hp_3fold_training_log.json"
    with open(log_save_path, "w") as f:
        json.dump(training_logs, f, indent=4)
    print(f"\n[*] Saved training logs to '{log_save_path}'")

    # AGGREGATE RESULTS ACROSS ALL 3 FOLDS
    print("\n" + "=" * 80)
    print("        COMPUTING AGGREGATE RESULTS ACROSS ALL 3 PATIENT FOLDS        ")
    print("=" * 80)

    accs = [fold_results[f]["metrics"]["active_accuracy"] for f in FOLDS]
    macro_f1s = [fold_results[f]["metrics"]["active_macro_f1"] for f in FOLDS]
    min_f1s = [fold_results[f]["metrics"]["minority_macro_f1"] for f in FOLDS]
    n_recs = [fold_results[f]["metrics"]["n_recall"] for f in FOLDS]
    sveb_recs = [fold_results[f]["metrics"]["sveb_recall"] for f in FOLDS]
    veb_recs = [fold_results[f]["metrics"]["veb_recall"] for f in FOLDS]
    f_recs = [fold_results[f]["metrics"]["f_recall"] for f in FOLDS]
    scores = [fold_results[f]["metrics"]["fold_score"] for f in FOLDS]

    aggregate_summary = {
        "individual_folds": fold_results,
        "aggregate_metrics": {
            "active_accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "active_macro_f1": {"mean": float(np.mean(macro_f1s)), "std": float(np.std(macro_f1s))},
            "minority_macro_f1": {"mean": float(np.mean(min_f1s)), "std": float(np.std(min_f1s))},
            "n_recall": {"mean": float(np.mean(n_recs)), "std": float(np.std(n_recs))},
            "sveb_recall": {"mean": float(np.mean(sveb_recs)), "std": float(np.std(sveb_recs))},
            "veb_recall": {"mean": float(np.mean(veb_recs)), "std": float(np.std(veb_recs))},
            "f_recall": {"mean": float(np.mean(f_recs)), "std": float(np.std(f_recs))},
            "fold_score": {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        }
    }

    agg_json_path = results_dir / "fixed_hp_3fold_aggregate_metrics.json"
    with open(agg_json_path, "w") as f:
        json.dump(aggregate_summary, f, indent=4)
    print(f"[*] Saved aggregate metrics JSON to '{agg_json_path}'")

    # AGGREGATE CONFUSION MATRIX
    concat_true = np.concatenate(all_fold_true_active, axis=0)
    concat_pred = np.concatenate(all_fold_pred_active, axis=0)
    agg_cm = confusion_matrix(concat_true, concat_pred, labels=[0, 1, 2, 3])
    
    agg_cm_path = results_dir / "fixed_hp_confusion_matrix_aggregate.png"
    save_confusion_matrix_plot(agg_cm, "Aggregate 3-Fold Validation Active Confusion Matrix", agg_cm_path)

    total_duration = time.time() - start_total_time

    # FINAL CONCISE TABLE SUMMARY
    print("\n" + "=" * 105)
    print("                   FIXED-HYPERPARAMETER 3-FOLD CV FINAL SUMMARY TABLE                   ")
    print("=" * 105)
    print(f"{'Fold':<8} | {'Best Ep':<7} | {'Active Acc':<10} | {'Macro F1':<9} | {'Minority F1':<11} | {'N Recall':<9} | {'SVEB Rec':<9} | {'VEB Rec':<8} | {'F Rec':<7} | {'Score':<7}")
    print("-" * 105)

    for fname in FOLDS:
        res = fold_results[fname]
        m = res["metrics"]
        print(f"{fname:<8} | #{res['best_epoch']:<6} | {m['active_accuracy']:>9.2f}% | {m['active_macro_f1']:>8.2f}% | {m['minority_macro_f1']:>10.2f}% | {m['n_recall']:>8.2f}% | {m['sveb_recall']:>8.2f}% | {m['veb_recall']:>7.2f}% | {m['f_recall']:>6.2f}% | {m['fold_score']:>6.2f}")

    print("-" * 105)
    am = aggregate_summary["aggregate_metrics"]
    print(f"{'Mean±Std':<8} | {'-':<7} | "
          f"{am['active_accuracy']['mean']:.2f}±{am['active_accuracy']['std']:.2f}% | "
          f"{am['active_macro_f1']['mean']:.2f}±{am['active_macro_f1']['std']:.2f}% | "
          f"{am['minority_macro_f1']['mean']:.2f}±{am['minority_macro_f1']['std']:.2f}% | "
          f"{am['n_recall']['mean']:.2f}±{am['n_recall']['std']:.2f}% | "
          f"{am['sveb_recall']['mean']:.2f}±{am['sveb_recall']['std']:.2f}% | "
          f"{am['veb_recall']['mean']:.2f}±{am['veb_recall']['std']:.2f}% | "
          f"{am['f_recall']['mean']:.2f}±{am['f_recall']['std']:.2f}% | "
          f"{am['fold_score']['mean']:.2f}±{am['fold_score']['std']:.2f}")
    print("=" * 105 + "\n")

    print(f"[*] Total Experiment Runtime: {total_duration / 60.0:.2f} minutes")
    print(f"[*] Saved Checkpoints: checkpoints/fixed_hp_fold[1-3]_best.pth")
    print(f"[*] Saved Results:     results/fixed_hp_fold[1-3]_metrics.json")
    print(f"[*] Saved Aggregate:   results/fixed_hp_3fold_aggregate_metrics.json")
    print(f"[*] Saved Plots:       results/fixed_hp_confusion_matrix_*.png")
    print(f"[*] Saved Logs:        results/fixed_hp_3fold_training_log.json")


if __name__ == "__main__":
    run_experiment()

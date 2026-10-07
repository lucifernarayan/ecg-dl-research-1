import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.count_parameters_compression_models import GenericHybrid1DBiCNNGRU
from src.data.preprocessor import ECGPreprocessor
from src.data.beat_extractor import ECGBeatExtractor
from src.data.download_dataset import download_mitdb

DS2_RECS = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", 
            "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

CLASS_LABELS = ["N", "SVEB", "VEB", "F", "Q"]
CLASS_FULL_NAMES = [
    "Normal (N)",
    "Supraventricular (SVEB)",
    "Ventricular Ectopic (VEB)",
    "Fusion (F)",
    "Unclassifiable (Q)"
]

def run_ds2_evaluation():
    print("[*] Starting DS2 Test Evaluation with Fold 2 Checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # 1. Load Checkpoint
    ckpt_path = Path("checkpoints/fixed_hp_fold2_best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[*] Checkpoint loaded successfully from {ckpt_path}. Fold: {ckpt.get('fold')}, Epoch: {ckpt.get('best_epoch')}")

    model = GenericHybrid1DBiCNNGRU(
        in_channels=1,
        cnn_channels=[128, 256, 256],
        kernel_sizes=[5, 5, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.2076,
        num_classes=5
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Extract DS2 Heartbeats
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_dir = download_mitdb("config.yaml")

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

    all_y_true = []
    all_y_pred = []
    patient_records = {}

    import wfdb

    for rec_id in DS2_RECS:
        record_path = data_dir / rec_id
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

        # Batch inference
        rec_preds = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(b_1d), batch_size):
                batch_x = torch.tensor(b_1d[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                rec_preds.extend(preds)

        rec_preds = np.array(rec_preds)
        rec_correct = int(np.sum(rec_preds == b_y))
        rec_total = len(b_y)
        rec_acc = float(rec_correct / rec_total * 100.0) if rec_total > 0 else 0.0

        patient_records[rec_id] = {
            "total_beats": rec_total,
            "correct_beats": rec_correct,
            "accuracy": rec_acc,
            "y_true": b_y,
            "y_pred": rec_preds
        }

        all_y_true.extend(b_y)
        all_y_pred.extend(rec_preds)
        print(f" -> Patient {rec_id}: {rec_correct}/{rec_total} ({rec_acc:.2f}%)")

    y_true_total = np.array(all_y_true)
    y_pred_total = np.array(all_y_pred)

    total_beats = len(y_true_total)
    correct_beats = int(np.sum(y_true_total == y_pred_total))
    global_acc = float(correct_beats / total_beats * 100.0)

    print("\n=======================================================")
    print(f"[*] DS2 EVALUATION COMPLETE:")
    print(f"[*] Total Beats: {total_beats:,}")
    print(f"[*] Correct Beats: {correct_beats:,}")
    print(f"[*] Global Accuracy: {global_acc:.2f}%")
    print("=======================================================\n")

    # Confusion Matrix
    cm_raw = confusion_matrix(y_true_total, y_pred_total, labels=[0, 1, 2, 3, 4])
    cm_recall_pct = (cm_raw.astype(float) / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-12)) * 100.0
    cm_prec_pct = (cm_raw.astype(float) / (cm_raw.sum(axis=0)[np.newaxis, :] + 1e-12)) * 100.0
    cm_total_pct = (cm_raw.astype(float) / total_beats) * 100.0

    # Per class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_total, y_pred_total, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    per_class_stats = {}
    for i, cls_name in enumerate(CLASS_LABELS):
        tp = int(cm_raw[i, i])
        fn = int(np.sum(cm_raw[i, :]) - tp)
        fp = int(np.sum(cm_raw[:, i]) - tp)
        tn = int(total_beats - tp - fn - fp)
        spec = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

        per_class_stats[cls_name] = {
            "name": CLASS_FULL_NAMES[i],
            "support": int(support[i]),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "sensitivity_recall": float(recall[i] * 100.0),
            "specificity": spec,
            "precision": float(precision[i] * 100.0),
            "f1_score": float(f1[i] * 100.0)
        }

    results_data = {
        "benchmark": "AAMI EC57 DS2 Independent Test Partition",
        "num_patients": len(DS2_RECS),
        "total_beats": total_beats,
        "correct_beats": correct_beats,
        "global_accuracy": global_acc,
        "macro_f1": float(np.mean(f1) * 100.0),
        "macro_precision": float(np.mean(precision) * 100.0),
        "macro_recall": float(np.mean(recall) * 100.0),
        "class_labels": CLASS_LABELS,
        "class_full_names": CLASS_FULL_NAMES,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_recall_pct": cm_recall_pct.tolist(),
        "confusion_matrix_precision_pct": cm_prec_pct.tolist(),
        "confusion_matrix_total_pct": cm_total_pct.tolist(),
        "per_class_metrics": per_class_stats
    }

    # Save to JSON
    out_json = Path("results/ds2_test_confusion_matrix_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"[*] Saved exact metrics to {out_json}")

    return results_data, cm_raw, cm_recall_pct

if __name__ == "__main__":
    run_ds2_evaluation()

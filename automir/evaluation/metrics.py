from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch


def compute_tempo_metrics(
    pred_bpms: Union[np.ndarray, torch.Tensor, List[float]],
    target_bpms: Union[np.ndarray, torch.Tensor, List[float]],
) -> Dict[str, float]:
    """Compute comprehensive tempo evaluation metrics including octave ambiguities."""
    if isinstance(pred_bpms, torch.Tensor):
        pred_bpms = pred_bpms.detach().cpu().numpy()
    if isinstance(target_bpms, torch.Tensor):
        target_bpms = target_bpms.detach().cpu().numpy()

    preds = np.asarray(pred_bpms, dtype=np.float64)
    targets = np.asarray(target_bpms, dtype=np.float64)

    if len(preds) == 0:
        return {
            "mae_bpm": 0.0,
            "median_ae_bpm": 0.0,
            "tempo_acc_4": 0.0,
            "tempo_acc_8": 0.0,
            "octave_aware_acc": 0.0,
            "half_tempo_rate": 0.0,
            "double_tempo_rate": 0.0,
        }

    abs_errors = np.abs(preds - targets)
    rel_errors = abs_errors / np.maximum(targets, 1e-6)

    mae = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))

    # Standard percentage tolerances
    acc_4 = float(np.mean(rel_errors <= 0.04)) * 100.0
    acc_8 = float(np.mean(rel_errors <= 0.08)) * 100.0

    # Octave error checks (half / double tempo)
    half_targets = 0.5 * targets
    double_targets = 2.0 * targets

    half_rel_errors = np.abs(preds - half_targets) / half_targets
    double_rel_errors = np.abs(preds - double_targets) / double_targets

    is_half = (half_rel_errors <= 0.04) & (rel_errors > 0.04)
    is_double = (double_rel_errors <= 0.04) & (rel_errors > 0.04)

    half_tempo_rate = float(np.mean(is_half)) * 100.0
    double_tempo_rate = float(np.mean(is_double)) * 100.0

    # Octave-aware accuracy: within 4% of 1x, 0.5x, or 2x ground truth
    octave_aware = (rel_errors <= 0.04) | (half_rel_errors <= 0.04) | (double_rel_errors <= 0.04)
    octave_aware_acc = float(np.mean(octave_aware)) * 100.0

    return {
        "mae_bpm": mae,
        "median_ae_bpm": median_ae,
        "tempo_acc_4": acc_4,
        "tempo_acc_8": acc_8,
        "octave_aware_acc": octave_aware_acc,
        "half_tempo_rate": half_tempo_rate,
        "double_tempo_rate": double_tempo_rate,
    }


def compute_classification_metrics(
    pred_labels: Union[np.ndarray, torch.Tensor, List[int]],
    target_labels: Union[np.ndarray, torch.Tensor, List[int]],
    prefix: str = "style",
) -> Dict[str, Any]:
    """Compute classification metrics: Macro F1, Weighted F1, Accuracy, and Confusion Matrix."""
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    if isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.detach().cpu().numpy()

    preds = np.asarray(pred_labels, dtype=np.int64)
    targets = np.asarray(target_labels, dtype=np.int64)

    if len(preds) == 0:
        return {
            f"{prefix}_macro_f1": 0.0,
            f"{prefix}_weighted_f1": 0.0,
            f"{prefix}_accuracy": 0.0,
            f"{prefix}_confusion_matrix": [],
        }

    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0)) * 100.0
    acc = float(accuracy_score(targets, preds)) * 100.0
    cm = confusion_matrix(targets, preds).tolist()

    return {
        f"{prefix}_macro_f1": macro_f1,
        f"{prefix}_weighted_f1": weighted_f1,
        f"{prefix}_accuracy": acc,
        f"{prefix}_confusion_matrix": cm,
    }

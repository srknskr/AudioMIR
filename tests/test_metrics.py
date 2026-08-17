import numpy as np
from automir.evaluation.metrics import (
    compute_tempo_metrics,
    compute_classification_metrics,
)


def test_tempo_metrics_perfect_prediction():
    preds = [120.0, 90.0, 140.0]
    targets = [120.0, 90.0, 140.0]

    metrics = compute_tempo_metrics(preds, targets)
    assert metrics["mae_bpm"] == 0.0
    assert metrics["median_ae_bpm"] == 0.0
    assert metrics["tempo_acc_4"] == 100.0
    assert metrics["tempo_acc_8"] == 100.0
    assert metrics["octave_aware_acc"] == 100.0
    assert metrics["half_tempo_rate"] == 0.0
    assert metrics["double_tempo_rate"] == 0.0


def test_tempo_metrics_octave_ambiguity():
    # Ground truth is 120 BPM, predicted is half (60 BPM) and double (240 BPM)
    preds = [60.0, 240.0]
    targets = [120.0, 120.0]

    metrics = compute_tempo_metrics(preds, targets)
    assert metrics["tempo_acc_4"] == 0.0  # Fails strict 4%
    assert metrics["octave_aware_acc"] == 100.0  # Passes octave-aware!
    assert metrics["half_tempo_rate"] == 50.0
    assert metrics["double_tempo_rate"] == 50.0


def test_classification_metrics():
    preds = [0, 1, 2, 0]
    targets = [0, 1, 2, 1]

    metrics = compute_classification_metrics(preds, targets, prefix="style")
    assert metrics["style_accuracy"] == 75.0
    assert metrics["style_macro_f1"] > 0.0
    assert len(metrics["style_confusion_matrix"]) == 3

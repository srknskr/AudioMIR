import tempfile
import torch

from automir.automl import RandomSearch, OptunaTPESearch, EvolutionaryParetoSearch
from automir.datasets import SyntheticRhythmDataset
from automir.models.factory import build_model
from automir.training.multi_fidelity import FidelityConfig, FidelityLevel
from automir.training.trainer import Trainer
from automir.utils.device import get_device


def test_tiny_training_loop_synthetic():
    device = torch.device("cpu")
    train_ds = SyntheticRhythmDataset(num_samples=8, representation="logmel", segment_duration=4.0, is_training=True)
    val_ds = SyntheticRhythmDataset(num_samples=4, representation="logmel", segment_duration=4.0, is_training=False)

    config = {
        "representation": "logmel",
        "conv_blocks": 2,
        "base_channels": 16,
        "kernel_size": 3,
        "use_gru": False,
        "dropout": 0.1,
    }
    model = build_model(config, num_styles=4, num_meters=2, enable_meter=True)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        batch_size=4,
        learning_rate=1e-3,
        enable_meter=True,
    )

    metrics = trainer.fit(epochs=2, patience=2)
    assert "val_loss" in metrics
    assert "mae_bpm" in metrics
    assert "style_macro_f1" in metrics


def test_tiny_automl_search_synthetic():
    device = torch.device("cpu")
    train_ds = SyntheticRhythmDataset(num_samples=6, representation="logmel", segment_duration=4.0, is_training=True)
    val_ds = SyntheticRhythmDataset(num_samples=4, representation="logmel", segment_duration=4.0, is_training=False)

    fidelity_cfg = FidelityConfig(
        level=FidelityLevel.QUICK,
        quick_fraction=0.5,
        quick_epochs=1,
    )

    objectives = [
        {"name": "tempo_acc_4", "direction": "maximize"},
        {"name": "latency_ms", "direction": "minimize"},
    ]

    # Test Evolutionary search
    searcher = EvolutionaryParetoSearch(
        population_size=2,
        objectives=objectives,
        fidelity_config=fidelity_cfg,
        num_styles=4,
        num_meters=2,
    )

    history = searcher.search(
        evaluations=2,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
    )

    assert len(history) == 2
    pareto_front = searcher.get_pareto_front()
    assert len(pareto_front) >= 1
    assert "latency_ms" in pareto_front[0].metrics

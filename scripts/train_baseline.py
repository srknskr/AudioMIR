import argparse
from pathlib import Path
import yaml
import torch

from automir.datasets import (
    SyntheticRhythmDataset,
    GrooveDataset,
    SerkanLoopsDataset,
    group_aware_split,
)
from automir.models.factory import build_model
from automir.training.trainer import Trainer
from automir.utils.device import get_device, get_device_name
from automir.utils.seed import seed_everything


def train_baseline(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    print(f"Using compute device: {get_device_name(device)}")

    ds_cfg = config.get("dataset", {})
    ds_name = ds_cfg.get("name", "synthetic")

    num_styles = ds_cfg.get("num_classes", 4)
    num_meters = ds_cfg.get("meter_classes", 2)
    enable_meter = ds_cfg.get("enable_meter", True)

    print(f"Loading dataset: {ds_name}...")
    if ds_name == "synthetic":
        train_ds = SyntheticRhythmDataset(num_samples=40, is_training=True)
        val_ds = SyntheticRhythmDataset(num_samples=20, is_training=False)
        test_ds = SyntheticRhythmDataset(num_samples=20, is_training=False)
    elif ds_name == "groove":
        manifest = ds_cfg.get("manifest_path", "data/groove/info.csv")
        train_ds = GrooveDataset.from_info_csv(manifest, split="train", is_training=True)
        val_ds = GrooveDataset.from_info_csv(manifest, split="validation", is_training=False)
        test_ds = GrooveDataset.from_info_csv(manifest, split="test", is_training=False)
    elif ds_name == "serkan_loops":
        manifest = ds_cfg.get("manifest_path", "data/custom_manifest.csv")
        train_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="train", is_training=True)
        val_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="val", is_training=False)
        test_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="test", is_training=False)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    print(f"Split sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    baseline_config = {
        "representation": "logmel",
        "segment_duration": 8.0,
        "n_mels": 96,
        "conv_blocks": 3,
        "base_channels": 32,
        "kernel_size": 3,
        "use_gru": False,
        "gru_hidden": 64,
        "dropout": 0.20,
    }

    model = build_model(
        baseline_config,
        num_styles=num_styles,
        num_meters=num_meters,
        enable_meter=enable_meter,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        loss_weights=config.get("loss_weights"),
        enable_meter=enable_meter,
    )

    epochs = config.get("fidelity", {}).get("full_epochs", 10)
    print(f"Training baseline model for {epochs} epochs...")
    val_results = trainer.fit(epochs=epochs, patience=config.get("fidelity", {}).get("patience", 4))

    print(f"\n--- Baseline Validation Results ---")
    print(f"Val Loss:           {val_results.get('val_loss', 0.0):.4f}")
    print(f"Tempo MAE (BPM):    {val_results.get('mae_bpm', 0.0):.2f}")
    print(f"Tempo Acc ±4%:      {val_results.get('tempo_acc_4', 0.0):.2f}%")
    print(f"Octave-Aware Acc:   {val_results.get('octave_aware_acc', 0.0):.2f}%")
    print(f"Style Macro-F1:     {val_results.get('style_macro_f1', 0.0):.2f}%")

    out_dir = Path("results/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "baseline_model.pt")
    print(f"Saved baseline checkpoint to {out_dir / 'baseline_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline AutoMIR model")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_baseline(args.config)

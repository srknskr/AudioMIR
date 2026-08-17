import argparse
import json
from pathlib import Path
import torch

from automir.automl.candidate import CandidateConfig
from automir.datasets import (
    SyntheticRhythmDataset,
    GrooveDataset,
    SerkanLoopsDataset,
)
from automir.experiments.sqlite_store import ExperimentStore
from automir.models.factory import build_model
from automir.training.trainer import Trainer
from automir.utils.device import get_device, get_device_name
from automir.utils.seed import seed_everything


def retrain_pareto_models(run_id: str, epochs: int = 20, patience: int = 5) -> None:
    store = ExperimentStore("results/experiments.sqlite")
    run_data = store.get_run(run_id)
    if not run_data:
        print(f"Error: Run '{run_id}' not found in database.")
        return

    run_dir = Path(f"results/{run_id}")
    pareto_json_path = run_dir / "pareto.json"
    if not pareto_json_path.exists():
        print(f"Error: {pareto_json_path} does not exist.")
        return

    with open(pareto_json_path, "r") as f:
        pareto_list = json.load(f)

    if not pareto_list:
        print(f"No Pareto candidates found in {pareto_json_path}")
        return

    config = run_data["config"]
    seed_everything(config.get("experiment", {}).get("seed", 42))
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    print(f"Retraining {len(pareto_list)} Pareto candidates on device: {get_device_name(device)}")

    ds_cfg = config.get("dataset", {})
    ds_name = ds_cfg.get("name", "synthetic")
    num_styles = ds_cfg.get("num_classes", 4)
    num_meters = ds_cfg.get("meter_classes", 2)
    enable_meter = ds_cfg.get("enable_meter", True)

    if ds_name == "synthetic":
        train_ds = SyntheticRhythmDataset(num_samples=40, is_training=True)
        val_ds = SyntheticRhythmDataset(num_samples=20, is_training=False)
    elif ds_name == "groove":
        manifest = ds_cfg.get("manifest_path", "data/groove/info.csv")
        train_ds = GrooveDataset.from_info_csv(manifest, split="train", is_training=True)
        val_ds = GrooveDataset.from_info_csv(manifest, split="validation", is_training=False)
    elif ds_name == "serkan_loops":
        manifest = ds_cfg.get("manifest_path", "data/custom_manifest.csv")
        train_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="train", is_training=True)
        val_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="val", is_training=False)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    retrained_results = []
    for idx, c_dict in enumerate(pareto_list):
        cand = CandidateConfig.from_dict(c_dict)
        print(f"\n[{idx+1}/{len(pareto_list)}] Retraining Pareto candidate `{cand.candidate_id}` (Rep: {cand.representation}, Blocks: {cand.conv_blocks})...")

        for ds in [train_ds, val_ds]:
            ds.representation = cand.representation
            ds.n_mels = cand.n_mels
            ds.segment_duration = cand.segment_duration

        model = build_model(
            cand,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            batch_size=cand.batch_size,
            learning_rate=cand.learning_rate,
            weight_decay=cand.weight_decay,
            loss_weights=config.get("loss_weights"),
            enable_meter=enable_meter,
        )

        metrics = trainer.fit(epochs=epochs, patience=patience)
        ckpt_path = ckpt_dir / f"{cand.candidate_id}.pt"
        torch.save(model.state_dict(), ckpt_path)

        cand.metrics.update(metrics)
        cand.metrics["checkpoint_path"] = str(ckpt_path)
        retrained_results.append(cand.to_dict())

        print(f"-> Saved Checkpoint: {ckpt_path}")
        print(f"-> Final Val Tempo Acc ±4%: {metrics.get('tempo_acc_4', 0.0):.2f}% | Style F1: {metrics.get('style_macro_f1', 0.0):.2f}%")

    with open(run_dir / "retrained_pareto.json", "w", encoding="utf-8") as f:
        json.dump(retrained_results, f, indent=2, default=str)
    print(f"\nAll Pareto models retrained successfully. Updated results in {run_dir / 'retrained_pareto.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Pareto-optimal models on full fidelity")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the search experiment")
    parser.add_argument("--epochs", type=int, default=15, help="Number of full training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    retrain_pareto_models(run_id=args.run_id, epochs=args.epochs, patience=args.patience)

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

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


def run_benchmark(run_id: str) -> None:
    store = ExperimentStore("results/experiments.sqlite")
    run_data = store.get_run(run_id)
    if not run_data:
        print(f"Error: Run '{run_id}' not found in database.")
        return

    run_dir = Path(f"results/{run_id}")
    retrained_path = run_dir / "retrained_pareto.json"
    pareto_path = run_dir / "pareto.json"

    if retrained_path.exists():
        with open(retrained_path, "r") as f:
            candidates_data = json.load(f)
    elif pareto_path.exists():
        with open(pareto_path, "r") as f:
            candidates_data = json.load(f)
    else:
        print(f"No Pareto candidate JSON found in {run_dir}")
        return

    config = run_data["config"]
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    print(f"=== AutoMIR Test Set Benchmark ===")
    print(f"Evaluating {len(candidates_data)} Pareto candidates on the Test Set (Untouched during AutoML search).")
    print(f"Device: {get_device_name(device)}")

    ds_cfg = config.get("dataset", {})
    ds_name = ds_cfg.get("name", "synthetic")
    num_styles = ds_cfg.get("num_classes", 4)
    num_meters = ds_cfg.get("meter_classes", 2)
    enable_meter = ds_cfg.get("enable_meter", True)

    if ds_name == "synthetic":
        test_ds = SyntheticRhythmDataset(num_samples=20, is_training=False)
    elif ds_name == "groove":
        manifest = ds_cfg.get("manifest_path", "data/groove/info.csv")
        test_ds = GrooveDataset.from_info_csv(manifest, split="test", is_training=False)
    elif ds_name == "serkan_loops":
        manifest = ds_cfg.get("manifest_path", "data/custom_manifest.csv")
        test_ds = SerkanLoopsDataset.from_manifest_csv(manifest, split="test", is_training=False)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    print(f"Test Set Size: {len(test_ds)} examples")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    benchmark_records = []

    for idx, c_dict in enumerate(candidates_data):
        cand = CandidateConfig.from_dict(c_dict)
        test_ds.representation = cand.representation
        test_ds.n_mels = cand.n_mels
        test_ds.segment_duration = cand.segment_duration
        curr_test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        model = build_model(
            cand,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )

        ckpt_path = run_dir / "checkpoints" / f"{cand.candidate_id}.pt"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

        trainer = Trainer(
            model=model,
            train_dataset=test_ds,
            val_dataset=test_ds,
            device=device,
            enable_meter=enable_meter,
        )

        test_metrics = trainer.evaluate(dataloader=curr_test_loader)

        record = {
            "candidate_id": cand.candidate_id,
            "representation": cand.representation,
            "conv_blocks": cand.conv_blocks,
            "base_channels": cand.base_channels,
            "use_gru": cand.use_gru,
            "params": cand.metrics.get("params", 0),
            "model_size_mb": cand.metrics.get("model_size_mb", 0.0),
            "latency_ms": cand.metrics.get("latency_ms", 0.0),
            "test_loss": test_metrics.get("val_loss", 0.0),
            "test_mae_bpm": test_metrics.get("mae_bpm", 0.0),
            "test_tempo_acc_4": test_metrics.get("tempo_acc_4", 0.0),
            "test_tempo_acc_8": test_metrics.get("tempo_acc_8", 0.0),
            "test_octave_aware_acc": test_metrics.get("octave_aware_acc", 0.0),
            "test_half_tempo_rate": test_metrics.get("half_tempo_rate", 0.0),
            "test_double_tempo_rate": test_metrics.get("double_tempo_rate", 0.0),
            "test_style_macro_f1": test_metrics.get("style_macro_f1", 0.0),
            "test_style_acc": test_metrics.get("style_accuracy", 0.0),
        }
        benchmark_records.append(record)

    df = pd.DataFrame(benchmark_records)
    out_csv = run_dir / "test_benchmark.csv"
    out_json = run_dir / "test_benchmark.json"

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("FINAL TEST SET BENCHMARK RESULTS")
    print("=" * 80)
    for r in benchmark_records:
        print(
            f"Model [{r['candidate_id']}] | Rep: {r['representation']:16s} | "
            f"Tempo ±4%: {r['test_tempo_acc_4']:5.1f}% | Octave Acc: {r['test_octave_aware_acc']:5.1f}% | "
            f"Style F1: {r['test_style_macro_f1']:5.1f}% | Latency: {r['latency_ms']:5.2f}ms | Size: {r['model_size_mb']:4.2f}MB"
        )
    print("=" * 80)
    print(f"Saved benchmark results to {out_csv} and {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pareto models on untouched test set")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to benchmark")
    args = parser.parse_args()
    run_benchmark(args.run_id)

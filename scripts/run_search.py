import argparse
import uuid
from datetime import datetime
from pathlib import Path
import yaml
import torch

from automir.automl import (
    RandomSearch,
    OptunaTPESearch,
    EvolutionaryParetoSearch,
)
from automir.datasets import (
    SyntheticRhythmDataset,
    GrooveDataset,
    SerkanLoopsDataset,
)
from automir.experiments.tracker import ExperimentTracker
from automir.training.multi_fidelity import FidelityConfig, FidelityLevel
from automir.utils.device import get_device, get_device_name
from automir.utils.seed import seed_everything


def run_automl_search(
    strategy_name: str,
    evaluations: int,
    config_path: str,
    dataset_name: str = None,
    manifest_path: str = None,
) -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    print(f"Using compute device: {get_device_name(device)}")

    ds_name = dataset_name or config.get("dataset", {}).get("name", "synthetic")
    ds_cfg = config.get("dataset", {})
    num_styles = ds_cfg.get("num_classes", 4)
    num_meters = ds_cfg.get("meter_classes", 2)
    enable_meter = ds_cfg.get("enable_meter", True)

    print(f"Initializing Dataset: {ds_name}...")
    if ds_name == "synthetic":
        train_ds = SyntheticRhythmDataset(num_samples=40, is_training=True)
        val_ds = SyntheticRhythmDataset(num_samples=20, is_training=False)
    elif ds_name == "groove":
        man = manifest_path or ds_cfg.get("manifest_path", "data/groove/info.csv")
        train_ds = GrooveDataset.from_info_csv(man, split="train", is_training=True)
        val_ds = GrooveDataset.from_info_csv(man, split="validation", is_training=False)
    elif ds_name == "serkan_loops":
        man = manifest_path or ds_cfg.get("manifest_path", "data/custom_manifest.csv")
        train_ds = SerkanLoopsDataset.from_manifest_csv(man, split="train", is_training=True)
        val_ds = SerkanLoopsDataset.from_manifest_csv(man, split="val", is_training=False)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    print(f"Search dataset splits - Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # Setup fidelity
    fid_cfg = config.get("fidelity", {})
    fidelity_config = FidelityConfig(
        level=FidelityLevel(fid_cfg.get("level", "QUICK")),
        quick_fraction=fid_cfg.get("quick_fraction", 0.25),
        quick_epochs=fid_cfg.get("quick_epochs", 2),
        screen_fraction=fid_cfg.get("screen_fraction", 0.50),
        screen_epochs=fid_cfg.get("screen_epochs", 5),
        full_fraction=fid_cfg.get("full_fraction", 1.0),
        full_epochs=fid_cfg.get("full_epochs", 10),
        patience=fid_cfg.get("patience", 3),
    )

    objectives = config.get("search", {}).get("objectives", [
        {"name": "tempo_acc_4", "direction": "maximize"},
        {"name": "style_macro_f1", "direction": "maximize"},
        {"name": "latency_ms", "direction": "minimize"},
        {"name": "model_size_mb", "direction": "minimize"},
    ])

    search_cfg = config.get("search", {})
    strategy_str = strategy_name.lower()
    
    if strategy_str == "random":
        strategy = RandomSearch(
            objectives=objectives,
            fidelity_config=fidelity_config,
            loss_weights=config.get("loss_weights"),
            enable_meter=enable_meter,
            num_styles=num_styles,
            num_meters=num_meters,
        )
    elif strategy_str == "tpe":
        strategy = OptunaTPESearch(
            objectives=objectives,
            fidelity_config=fidelity_config,
            loss_weights=config.get("loss_weights"),
            enable_meter=enable_meter,
            num_styles=num_styles,
            num_meters=num_meters,
        )
    elif strategy_str in ["evolutionary", "nsga2"]:
        strategy = EvolutionaryParetoSearch(
            population_size=search_cfg.get("population_size", 8),
            crossover_prob=search_cfg.get("crossover_prob", 0.7),
            mutation_prob=search_cfg.get("mutation_prob", 0.35),
            tournament_size=search_cfg.get("tournament_size", 2),
            objectives=objectives,
            fidelity_config=fidelity_config,
            loss_weights=config.get("loss_weights"),
            enable_meter=enable_meter,
            num_styles=num_styles,
            num_meters=num_meters,
        )
    else:
        raise ValueError(f"Unknown search strategy: {strategy_name}")

    # Generate Run ID
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{strategy_str}_{timestamp_str}_{str(uuid.uuid4())[:4]}"
    print(f"\n🚀 Starting AutoML Search Run: {run_id}")
    print(f"Strategy:    {strategy_str.upper()}")
    print(f"Budget:      {evaluations} candidates")
    print(f"Fidelity:    {fidelity_config.level.value} ({fidelity_config.get_epochs()} epochs, {fidelity_config.get_data_fraction()*100:.0f}% data)")

    tracker = ExperimentTracker(
        run_id=run_id,
        strategy=strategy_str,
        dataset=ds_name,
        config=config,
    )

    def on_candidate_evaluated(candidate):
        tracker.log_candidate(candidate)
        m = candidate.metrics
        status = "❌ Failed" if candidate.failed else "✅ OK"
        print(
            f"[{len(tracker.history):02d}/{evaluations}] Candidate {candidate.candidate_id} ({status}) | "
            f"Rep: {candidate.representation:16s} | Blocks: {candidate.conv_blocks} | "
            f"Tempo ±4%: {m.get('tempo_acc_4', 0.0):5.1f}% | "
            f"Style F1: {m.get('style_macro_f1', 0.0):5.1f}% | "
            f"Latency: {m.get('latency_ms', 0.0):5.2f} ms | "
            f"Size: {m.get('model_size_mb', 0.0):5.2f} MB"
        )

    strategy.search(
        evaluations=evaluations,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        callback=on_candidate_evaluated,
    )

    summary = tracker.finalize(objectives=objectives)
    print(f"\n✨ Search Finished!")
    print(f"Discovered {summary['pareto_candidates_count']} Pareto-optimal models out of {summary['successful_candidates']} successful evaluations.")
    print(f"Results, charts and summaries stored in: {summary['run_dir']}\n")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoMIR AutoML Architecture Search")
    parser.add_argument("--strategy", type=str, default="evolutionary", choices=["random", "tpe", "evolutionary"], help="Search strategy")
    parser.add_argument("--evaluations", type=int, default=10, help="Candidate evaluation budget")
    parser.add_argument("--config", type=str, default="configs/quick.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest path")
    args = parser.parse_args()

    run_automl_search(
        strategy_name=args.strategy,
        evaluations=args.evaluations,
        config_path=args.config,
        dataset_name=args.dataset,
        manifest_path=args.manifest,
    )

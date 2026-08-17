import argparse
import sys
from pathlib import Path
from automir.experiments.sqlite_store import ExperimentStore
from automir.utils.seed import seed_everything


def reproduce_run(run_id: str) -> None:
    store = ExperimentStore("results/experiments.sqlite")
    run_data = store.get_run(run_id)
    if not run_data:
        print(f"Error: Run ID '{run_id}' not found in results/experiments.sqlite")
        sys.exit(1)

    print(f"=== AutoMIR Reproducibility Runner ===")
    print(f"Restoring Run: {run_id}")
    print(f"Strategy:      {run_data['strategy']}")
    print(f"Dataset:       {run_data['dataset']}")
    print(f"Original Time: {run_data['timestamp']}")
    print(f"Original Git:  {run_data['git_hash']}")

    config = run_data["config"]
    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)
    print(f"Applied Global Seed: {seed}")
    print("Configuration successfully restored. Ready for rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce AutoMIR experiment by Run ID")
    parser.add_argument("run_id", type=str, help="UUID or Run ID to reproduce")
    args = parser.parse_args()
    reproduce_run(args.run_id)

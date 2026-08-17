import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from automir.automl.candidate import CandidateConfig
from automir.evaluation.pareto import extract_pareto_front
from automir.experiments.sqlite_store import ExperimentStore
from automir.utils.device import get_device_info


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


class ExperimentTracker:
    """Tracks, persists, and visualizes AutoML experiment runs and Pareto sets."""

    def __init__(
        self,
        run_id: str,
        strategy: str,
        dataset: str,
        config: Dict[str, Any],
        results_dir: str = "results",
    ):
        self.run_id = run_id
        self.strategy = strategy
        self.dataset = dataset
        self.config = config
        self.timestamp = datetime.now().isoformat()
        self.git_hash = get_git_commit_hash()
        self.device_info = get_device_info()

        self.run_dir = Path(results_dir) / self.run_id
        self.candidates_dir = self.run_dir / "candidates"
        self.plots_dir = self.run_dir / "plots"

        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.store = ExperimentStore(f"{results_dir}/experiments.sqlite")
        self.store.record_run(
            run_id=self.run_id,
            timestamp=self.timestamp,
            git_hash=self.git_hash,
            strategy=self.strategy,
            dataset=self.dataset,
            config=self.config,
            device_info=self.device_info,
        )

        self.history: List[CandidateConfig] = []
        self.start_time = time.perf_counter()

    def log_candidate(self, candidate: CandidateConfig) -> None:
        """Save candidate record to memory, JSON file, and SQLite."""
        self.history.append(candidate)

        cand_file = self.candidates_dir / f"{candidate.candidate_id}.json"
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(candidate.to_dict(), f, indent=2, default=str)

        self.store.record_candidate(
            run_id=self.run_id,
            candidate_id=candidate.candidate_id,
            generation=candidate.generation,
            config_dict=candidate.to_dict(),
            metrics_dict=candidate.metrics,
            pareto_rank=candidate.rank,
            crowding_distance=candidate.crowding_distance,
            failed=candidate.failed,
            failure_reason=candidate.failure_reason,
        )

    def finalize(self, objectives: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate final pareto.csv, pareto.json, plots, and summary.md."""
        wall_clock = round(time.perf_counter() - self.start_time, 2)
        valid_candidates = [c for c in self.history if not c.failed]

        self.store.record_run(
            run_id=self.run_id,
            timestamp=self.timestamp,
            git_hash=self.git_hash,
            strategy=self.strategy,
            dataset=self.dataset,
            config=self.config,
            device_info=self.device_info,
            total_evaluations=len(self.history),
            wall_clock_s=wall_clock,
        )

        # Extract Pareto front
        pareto_candidates: List[CandidateConfig] = []
        if valid_candidates:
            dict_list = [c.to_dict()["metrics"] | {"_obj": c} for c in valid_candidates]
            front_dicts = extract_pareto_front(dict_list, objectives)
            pareto_candidates = [d["_obj"] for d in front_dicts]

        # 1. Save all candidates to CSV
        all_records = []
        for c in self.history:
            flat = {
                "candidate_id": c.candidate_id,
                "generation": c.generation,
                "failed": c.failed,
                "representation": c.representation,
                "segment_duration": c.segment_duration,
                "n_mels": c.n_mels,
                "conv_blocks": c.conv_blocks,
                "base_channels": c.base_channels,
                "kernel_size": c.kernel_size,
                "use_gru": c.use_gru,
                "gru_hidden": c.gru_hidden,
                "dropout": c.dropout,
                "learning_rate": c.learning_rate,
                "weight_decay": c.weight_decay,
                "batch_size": c.batch_size,
                "pareto_rank": c.rank,
                **c.metrics,
            }
            all_records.append(flat)

        df_all = pd.DataFrame(all_records)
        df_all.to_csv(self.run_dir / "all_candidates.csv", index=False)

        # 2. Save Pareto CSV & JSON
        pareto_records = [
            c.to_dict() for c in pareto_candidates
        ]
        with open(self.run_dir / "pareto.json", "w", encoding="utf-8") as f:
            json.dump(pareto_records, f, indent=2, default=str)

        df_pareto = pd.DataFrame([
            r for r in all_records if r["candidate_id"] in {c.candidate_id for c in pareto_candidates}
        ])
        df_pareto.to_csv(self.run_dir / "pareto.csv", index=False)

        # 3. Generate Trade-off Plots
        self._generate_plots(valid_candidates, pareto_candidates)

        # 4. Generate summary.md
        self._write_summary_markdown(wall_clock, valid_candidates, pareto_candidates)

        return {
            "run_id": self.run_id,
            "wall_clock_s": wall_clock,
            "total_evaluated": len(self.history),
            "successful_candidates": len(valid_candidates),
            "pareto_candidates_count": len(pareto_candidates),
            "run_dir": str(self.run_dir),
        }

    def _generate_plots(
        self,
        all_candidates: List[CandidateConfig],
        pareto_candidates: List[CandidateConfig],
    ) -> None:
        if not all_candidates:
            return

        pareto_ids = {c.candidate_id for c in pareto_candidates}

        pairs = [
            ("latency_ms", "tempo_acc_4", "Inference Latency (ms)", "Tempo Accuracy ±4% (%)", "latency_vs_tempo_acc.png"),
            ("model_size_mb", "tempo_acc_4", "Model Size (MB)", "Tempo Accuracy ±4% (%)", "size_vs_tempo_acc.png"),
            ("latency_ms", "style_macro_f1", "Inference Latency (ms)", "Style Macro-F1 (%)", "latency_vs_style_f1.png"),
            ("model_size_mb", "style_macro_f1", "Model Size (MB)", "Style Macro-F1 (%)", "size_vs_style_f1.png"),
        ]

        for x_key, y_key, x_label, y_label, filename in pairs:
            plt.figure(figsize=(8, 5))
            
            x_all = [c.metrics.get(x_key, 0.0) for c in all_candidates]
            y_all = [c.metrics.get(y_key, 0.0) for c in all_candidates]
            plt.scatter(x_all, y_all, c="#94a3b8", alpha=0.6, label="Evaluated Candidates", s=40)

            if pareto_candidates:
                x_par = [c.metrics.get(x_key, 0.0) for c in pareto_candidates]
                y_par = [c.metrics.get(y_key, 0.0) for c in pareto_candidates]
                plt.scatter(x_par, y_par, c="#0284c7", edgecolors="#0f172a", linewidths=1.5, s=90, label="Pareto-Optimal")

            plt.xlabel(x_label, fontsize=11)
            plt.ylabel(y_label, fontsize=11)
            plt.title(f"AutoMIR Trade-off: {y_label} vs {x_label}\nRun: {self.run_id} ({self.strategy.upper()})", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / filename, dpi=200)
            plt.close()

    def _write_summary_markdown(
        self,
        wall_clock: float,
        valid_candidates: List[CandidateConfig],
        pareto_candidates: List[CandidateConfig],
    ) -> None:
        summary_file = self.run_dir / "summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# AutoMIR Experiment Run Summary: `{self.run_id}`\n\n")
            f.write(f"- **Strategy**: `{self.strategy}`\n")
            f.write(f"- **Dataset**: `{self.dataset}`\n")
            f.write(f"- **Date & Time**: `{self.timestamp}`\n")
            f.write(f"- **Git Commit**: `{self.git_hash}`\n")
            f.write(f"- **Device**: `{self.device_info.get('device_name')}`\n")
            f.write(f"- **Total Evaluations**: {len(self.history)}\n")
            f.write(f"- **Successful Evaluations**: {len(valid_candidates)}\n")
            f.write(f"- **Pareto Candidates Discovered**: {len(pareto_candidates)}\n")
            f.write(f"- **Wall-clock Time**: {wall_clock} seconds\n\n")

            f.write("## Discovered Pareto-Optimal Models\n\n")
            if pareto_candidates:
                f.write("| ID | Rep | Blocks | Chans | GRU | Tempo Acc ±4% | Style Macro-F1 | Latency (ms) | Size (MB) |\n")
                f.write("|---|---|---|---|---|---|---|---|---|\n")
                for c in pareto_candidates:
                    m = c.metrics
                    f.write(
                        f"| `{c.candidate_id}` | {c.representation} | {c.conv_blocks} | {c.base_channels} | "
                        f"{'Yes' if c.use_gru else 'No'} | {m.get('tempo_acc_4', 0.0):.1f}% | "
                        f"{m.get('style_macro_f1', 0.0):.1f}% | {m.get('latency_ms', 0.0):.2f} ms | "
                        f"{m.get('model_size_mb', 0.0):.2f} MB |\n"
                    )
            else:
                f.write("No valid Pareto candidates discovered in this run.\n")

            f.write("\n## Generated Visualizations\n\n")
            f.write("- `plots/latency_vs_tempo_acc.png`\n")
            f.write("- `plots/size_vs_tempo_acc.png`\n")
            f.write("- `plots/latency_vs_style_f1.png`\n")
            f.write("- `plots/size_vs_style_f1.png`\n")

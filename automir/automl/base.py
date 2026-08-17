from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import time
import torch
from torch.utils.data import Dataset

from automir.automl.candidate import CandidateConfig
from automir.models.factory import build_model
from automir.training.trainer import Trainer
from automir.training.multi_fidelity import FidelityConfig, create_fidelity_subset
from automir.evaluation.efficiency import benchmark_efficiency
from automir.evaluation.pareto import extract_pareto_front


class BaseSearchStrategy(ABC):
    """Abstract base class for AutoML search algorithms."""

    def __init__(
        self,
        objectives: List[Dict[str, str]],
        fidelity_config: Optional[FidelityConfig] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        enable_meter: bool = True,
        num_styles: int = 9,
        num_meters: int = 3,
    ):
        self.objectives = objectives or [
            {"name": "tempo_acc_4", "direction": "maximize"},
            {"name": "style_macro_f1", "direction": "maximize"},
            {"name": "latency_ms", "direction": "minimize"},
            {"name": "model_size_mb", "direction": "minimize"},
        ]
        self.fidelity_config = fidelity_config or FidelityConfig()
        self.loss_weights = loss_weights
        self.enable_meter = enable_meter
        self.num_styles = num_styles
        self.num_meters = num_meters

        self.history: List[CandidateConfig] = []
        self.best_trajectory: List[Dict[str, Any]] = []

    def evaluate_candidate(
        self,
        candidate: CandidateConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
    ) -> CandidateConfig:
        """Evaluate a single candidate with multi-fidelity training and efficiency benchmarking."""
        t0 = time.perf_counter()
        try:
            # 1. Instantiate model
            model = build_model(
                candidate,
                num_styles=self.num_styles,
                num_meters=self.num_meters,
                enable_meter=self.enable_meter,
            )

            # 2. Benchmark efficiency before training
            sample_dur = candidate.segment_duration
            sr = 22050
            if candidate.representation == "logmel_tempogram":
                # (1, 1, n_mels, T) and (1, 1, 193, T)
                T = int(sample_dur * sr / 512) + 1
                sample_in = (
                    torch.randn(1, 1, candidate.n_mels, T),
                    torch.randn(1, 1, 193, T),
                )
            elif candidate.representation == "tempogram":
                T = int(sample_dur * sr / 512) + 1
                sample_in = torch.randn(1, 1, 193, T)
            else:
                T = int(sample_dur * sr / 512) + 1
                sample_in = torch.randn(1, 1, candidate.n_mels, T)

            eff_metrics = benchmark_efficiency(
                model=model,
                sample_input=sample_in,
                device=device,
                warmup_runs=5,
                measured_runs=15,
            )

            # 3. Configure dataset representations for candidate
            for ds in [train_dataset, val_dataset]:
                target = ds.dataset if hasattr(ds, "dataset") else ds
                target.representation = candidate.representation
                target.n_mels = candidate.n_mels
                target.segment_duration = candidate.segment_duration

            # 4. Create fidelity subset
            fidelity_train_ds = create_fidelity_subset(
                train_dataset, fraction=self.fidelity_config.get_data_fraction()
            )

            # 5. Multi-task training
            trainer = Trainer(
                model=model,
                train_dataset=fidelity_train_ds,
                val_dataset=val_dataset,
                device=device,
                batch_size=candidate.batch_size,
                learning_rate=candidate.learning_rate,
                weight_decay=candidate.weight_decay,
                loss_weights=self.loss_weights,
                enable_meter=self.enable_meter,
            )

            train_metrics = trainer.fit(
                epochs=self.fidelity_config.get_epochs(),
                patience=self.fidelity_config.patience,
            )

            candidate.metrics = {
                **eff_metrics,
                **train_metrics,
                "wall_clock_s": round(time.perf_counter() - t0, 2),
            }
            candidate.failed = False
            candidate.failure_reason = None

        except Exception as e:
            candidate.failed = True
            candidate.failure_reason = str(e)
            candidate.metrics = {
                "params": 0,
                "model_size_mb": 999.0,
                "latency_ms": 9999.0,
                "tempo_acc_4": 0.0,
                "style_macro_f1": 0.0,
                "wall_clock_s": round(time.perf_counter() - t0, 2),
                "error": str(e),
            }

        return candidate

    @abstractmethod
    def search(
        self,
        evaluations: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        callback: Optional[Callable[[CandidateConfig], None]] = None,
    ) -> List[CandidateConfig]:
        pass

    def get_pareto_front(self) -> List[CandidateConfig]:
        """Return non-dominated candidates from search history."""
        valid = [c for c in self.history if not c.failed]
        if not valid:
            return []
        dict_list = [c.to_dict()["metrics"] | {"_candidate_ref": c} for c in valid]
        front_dicts = extract_pareto_front(dict_list, self.objectives)
        return [d["_candidate_ref"] for d in front_dicts]

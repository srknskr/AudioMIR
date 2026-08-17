from typing import Callable, List, Optional
import optuna
import torch
from torch.utils.data import Dataset

from automir.automl.base import BaseSearchStrategy
from automir.automl.candidate import CandidateConfig
from automir.automl.search_space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPESearch(BaseSearchStrategy):
    """Bayesian Optimization Baseline using Optuna Tree-structured Parzen Estimator (TPE)."""

    def search(
        self,
        evaluations: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        callback: Optional[Callable[[CandidateConfig], None]] = None,
    ) -> List[CandidateConfig]:
        self.history = []

        directions = [
            obj.get("direction", "maximize").lower() for obj in self.objectives
        ]

        study = optuna.create_study(
            directions=directions,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> List[float]:
            rep = trial.suggest_categorical("representation", SearchSpace.REPRESENTATIONS)
            dur = trial.suggest_categorical("segment_duration", SearchSpace.SEGMENT_DURATIONS)
            n_mels = trial.suggest_categorical("n_mels", SearchSpace.N_MELS_CHOICES)
            conv_blocks = trial.suggest_categorical("conv_blocks", SearchSpace.CONV_BLOCKS_CHOICES)
            base_channels = trial.suggest_categorical("base_channels", SearchSpace.BASE_CHANNELS_CHOICES)
            kernel_size = trial.suggest_categorical("kernel_size", SearchSpace.KERNEL_SIZES)
            use_gru = trial.suggest_categorical("use_gru", SearchSpace.USE_GRU_CHOICES)
            gru_hidden = trial.suggest_categorical("gru_hidden", SearchSpace.GRU_HIDDEN_CHOICES)
            batch_size = trial.suggest_categorical("batch_size", SearchSpace.BATCH_SIZES)

            dropout = trial.suggest_float("dropout", SearchSpace.DROPOUT_MIN, SearchSpace.DROPOUT_MAX)
            lr = trial.suggest_float("learning_rate", SearchSpace.LR_MIN, SearchSpace.LR_MAX, log=True)
            wd = trial.suggest_float("weight_decay", SearchSpace.WD_MIN, SearchSpace.WD_MAX, log=True)

            candidate = CandidateConfig(
                candidate_id=f"tpe_{trial.number:03d}",
                generation=trial.number,
                representation=rep,
                segment_duration=dur,
                n_mels=n_mels,
                conv_blocks=conv_blocks,
                base_channels=base_channels,
                kernel_size=kernel_size,
                use_gru=use_gru,
                gru_hidden=gru_hidden,
                dropout=round(dropout, 3),
                learning_rate=lr,
                weight_decay=wd,
                batch_size=batch_size,
            )

            candidate = self.evaluate_candidate(
                candidate=candidate,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
            )
            self.history.append(candidate)
            if callback is not None:
                callback(candidate)

            # Return objective values in order
            values = []
            for obj in self.objectives:
                name = obj["name"]
                direction = obj.get("direction", "maximize").lower()
                val = candidate.metrics.get(
                    name, -1e9 if direction == "maximize" else 1e9
                )
                values.append(float(val))

            return values

        study.optimize(objective, n_trials=evaluations, catch=(Exception,))
        return self.history

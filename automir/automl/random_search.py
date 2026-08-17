from typing import Callable, List, Optional
import torch
from torch.utils.data import Dataset

from automir.automl.base import BaseSearchStrategy
from automir.automl.candidate import CandidateConfig
from automir.automl.search_space import SearchSpace


class RandomSearch(BaseSearchStrategy):
    """Uniform Random Search Baseline under equal candidate evaluation budget."""

    def search(
        self,
        evaluations: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        callback: Optional[Callable[[CandidateConfig], None]] = None,
    ) -> List[CandidateConfig]:
        self.history = []

        for i in range(evaluations):
            candidate = SearchSpace.sample_candidate(generation=0)
            candidate = self.evaluate_candidate(
                candidate=candidate,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
            )
            self.history.append(candidate)
            if callback is not None:
                callback(candidate)

        return self.history

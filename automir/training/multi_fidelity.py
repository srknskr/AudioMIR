from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from torch.utils.data import Subset, Dataset


class FidelityLevel(str, Enum):
    QUICK = "QUICK"
    SCREEN = "SCREEN"
    FULL = "FULL"


@dataclass
class FidelityConfig:
    level: FidelityLevel = FidelityLevel.QUICK
    quick_fraction: float = 0.25
    quick_epochs: int = 2
    screen_fraction: float = 0.50
    screen_epochs: int = 5
    full_fraction: float = 1.0
    full_epochs: int = 30
    patience: int = 5

    def get_epochs(self) -> int:
        if self.level == FidelityLevel.QUICK:
            return self.quick_epochs
        elif self.level == FidelityLevel.SCREEN:
            return self.screen_epochs
        else:
            return self.full_epochs

    def get_data_fraction(self) -> float:
        if self.level == FidelityLevel.QUICK:
            return self.quick_fraction
        elif self.level == FidelityLevel.SCREEN:
            return self.screen_fraction
        else:
            return self.full_fraction


def create_fidelity_subset(
    dataset: Dataset,
    fraction: float,
    seed: int = 42,
) -> Dataset:
    """Subsample dataset deterministically according to fidelity level fraction."""
    total_len = len(dataset)
    if fraction >= 1.0 or total_len <= 4:
        return dataset

    num_samples = max(2, int(np.floor(total_len * fraction)))
    rng = np.random.RandomState(seed)
    indices = rng.choice(total_len, size=num_samples, replace=False)
    return Subset(dataset, indices)

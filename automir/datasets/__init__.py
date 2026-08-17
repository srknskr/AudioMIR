"""Dataset loaders and splitting logic with zero-leakage guarantees."""

from automir.datasets.base import BaseRhythmDataset, group_aware_split
from automir.datasets.groove import GrooveDataset
from automir.datasets.serkan_loops import SerkanLoopsDataset
from automir.datasets.synthetic import (
    SyntheticRhythmDataset,
    generate_synthetic_drum_audio,
    create_synthetic_manifest,
)

__all__ = [
    "BaseRhythmDataset",
    "group_aware_split",
    "GrooveDataset",
    "SerkanLoopsDataset",
    "SyntheticRhythmDataset",
    "generate_synthetic_drum_audio",
    "create_synthetic_manifest",
]

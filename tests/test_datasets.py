import pandas as pd
import pytest
import torch

from automir.datasets import (
    SyntheticRhythmDataset,
    SerkanLoopsDataset,
    group_aware_split,
    create_synthetic_manifest,
)


def test_group_aware_split_no_leakage():
    manifest = create_synthetic_manifest(num_samples=40, seed=42)
    train_df, val_df, test_df = group_aware_split(
        manifest,
        group_col="source_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_groups = set(train_df["source_id"])
    val_groups = set(val_df["source_id"])
    test_groups = set(test_df["source_id"])

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Strict assertion: zero group overlap
    assert train_groups.isdisjoint(val_groups), "Leakage detected between train and val!"
    assert train_groups.isdisjoint(test_groups), "Leakage detected between train and test!"
    assert val_groups.isdisjoint(test_groups), "Leakage detected between val and test!"


def test_synthetic_rhythm_dataset_loading():
    ds = SyntheticRhythmDataset(num_samples=8, representation="logmel", segment_duration=4.0)
    assert len(ds) == 8

    item = ds[0]
    assert "bpm" in item
    assert "log2_bpm" in item
    assert "style_id" in item
    assert "feature" in item
    assert item["feature"].ndim == 3  # (1, n_mels, T)


def test_synthetic_dual_representation():
    ds = SyntheticRhythmDataset(num_samples=4, representation="logmel_tempogram", segment_duration=4.0)
    item = ds[0]
    assert "logmel" in item
    assert "tempogram" in item
    assert item["logmel"].shape[0] == 1
    assert item["tempogram"].shape[0] == 1


def test_serkan_loops_manifest_validation():
    # Missing source_id should raise ValueError
    invalid_df = pd.DataFrame([{"audio_path": "a.wav", "bpm": 120}])
    with pytest.raises(ValueError):
        SerkanLoopsDataset(df=invalid_df)

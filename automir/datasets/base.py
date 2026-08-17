from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from automir.audio.transforms import preprocess_audio, extract_features
from automir.audio.cache import FeatureCache


def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "source_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into Train, Validation, and Test subsets such that all records

    with the same group_col (e.g. source_id) belong strictly to ONE split.
    Guarantees 0% group leakage across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    
    unique_groups = df[group_col].unique()
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    n_train = int(np.floor(n_groups * train_ratio))
    n_val = int(np.floor(n_groups * val_ratio))

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train : n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val :])

    train_df = df[df[group_col].isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].copy().reset_index(drop=True)

    # Double-check assertion for zero group overlap
    train_set = set(train_df[group_col])
    val_set = set(val_df[group_col])
    test_set = set(test_df[group_col])
    assert train_set.isdisjoint(val_set), "Fatal: Leakage between train and val groups!"
    assert train_set.isdisjoint(test_set), "Fatal: Leakage between train and test groups!"
    assert val_set.isdisjoint(test_set), "Fatal: Leakage between val and test groups!"

    return train_df, val_df, test_df


class BaseRhythmDataset(Dataset, ABC):
    """Base dataset class for rhythm understanding tasks."""

    def __init__(
        self,
        df: pd.DataFrame,
        representation: str = "logmel",
        segment_duration: float = 8.0,
        sample_rate: int = 22050,
        n_mels: int = 96,
        is_training: bool = False,
        cache: Optional[FeatureCache] = None,
        style_to_id: Optional[Dict[str, int]] = None,
        meter_to_id: Optional[Dict[str, int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.representation = representation
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.is_training = is_training
        self.cache = cache

        # Build / set style mapping
        if style_to_id is not None:
            self.style_to_id = style_to_id
        else:
            unique_styles = sorted(self.df["genre"].dropna().unique()) if "genre" in self.df.columns else []
            self.style_to_id = {s: i for i, s in enumerate(unique_styles)}

        # Build / set meter mapping
        if meter_to_id is not None:
            self.meter_to_id = meter_to_id
        else:
            unique_meters = sorted(self.df["meter"].dropna().unique()) if "meter" in self.df.columns else []
            self.meter_to_id = {m: i for i, m in enumerate(unique_meters)}

    def __len__(self) -> int:
        return len(self.df)

    @abstractmethod
    def load_audio(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Return (waveform_tensor, sample_rate, audio_id)."""
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        waveform, sr, audio_id = self.load_audio(index)

        # Preprocess waveform
        proc_wave = preprocess_audio(
            waveform=waveform,
            sample_rate=sr,
            target_sample_rate=self.sample_rate,
            target_duration=self.segment_duration,
            is_training=self.is_training,
        )

        # Feature extraction with cache lookup
        feature_params = {
            "rep": self.representation,
            "sr": self.sample_rate,
            "dur": self.segment_duration,
            "n_mels": self.n_mels,
            "training": self.is_training,
        }

        features = None
        if self.cache is not None:
            features = self.cache.get(audio_id, feature_params)

        if features is None:
            features = extract_features(
                waveform=proc_wave,
                representation=self.representation,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
            )
            if self.cache is not None:
                self.cache.put(audio_id, feature_params, features)

        # Targets
        bpm = float(row["bpm"])
        # Log2(BPM) regression target
        log2_bpm = float(np.log2(max(bpm, 1.0)))

        style_id = self.style_to_id.get(row.get("genre", ""), 0)
        meter_id = self.meter_to_id.get(row.get("meter", ""), 0)

        item = {
            "bpm": torch.tensor(bpm, dtype=torch.float32),
            "log2_bpm": torch.tensor(log2_bpm, dtype=torch.float32),
            "style_id": torch.tensor(style_id, dtype=torch.long),
            "meter_id": torch.tensor(meter_id, dtype=torch.long),
            "audio_id": audio_id,
        }

        if isinstance(features, tuple):
            item["logmel"] = features[0]
            item["tempogram"] = features[1]
        else:
            item["feature"] = features

        return item

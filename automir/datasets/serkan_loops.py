from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import torch
import torchaudio
from automir.datasets.base import BaseRhythmDataset, group_aware_split


class SerkanLoopsDataset(BaseRhythmDataset):
    """Custom Drum-Loop Dataset with strict group-aware splitting based on source_id."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Optional[str] = None,
        **kwargs,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self._validate_manifest(df)
        super().__init__(df=df, **kwargs)

    @staticmethod
    def _validate_manifest(df: pd.DataFrame) -> None:
        """Validate required columns in manifest."""
        required_cols = {"audio_path", "bpm", "source_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Custom manifest is missing mandatory columns: {missing}")

    @classmethod
    def from_manifest_csv(
        cls,
        manifest_path: str,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        root_dir: Optional[str] = None,
        **kwargs,
    ) -> "SerkanLoopsDataset":
        """Load manifest CSV and perform group-aware split if 'split' column is absent."""
        df = pd.read_csv(manifest_path)
        cls._validate_manifest(df)

        if "split" not in df.columns:
            train_df, val_df, test_df = group_aware_split(
                df,
                group_col="source_id",
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            if split.lower() == "train":
                target_df = train_df
            elif split.lower() in ["val", "validation"]:
                target_df = val_df
            elif split.lower() == "test":
                target_df = test_df
            else:
                target_df = df
        else:
            target_df = df[df["split"].str.lower() == split.lower()].copy()

        return cls(df=target_df, root_dir=root_dir, **kwargs)

    def load_audio(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        audio_rel_path = str(row["audio_path"])
        full_path = self.root_dir / audio_rel_path

        if not full_path.exists():
            full_path = Path(audio_rel_path)

        waveform, sr = torchaudio.load(str(full_path))
        return waveform, sr, audio_rel_path

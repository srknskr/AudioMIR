from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import torch
import torchaudio
from automir.datasets.base import BaseRhythmDataset


class GrooveDataset(BaseRhythmDataset):
    """Dataset adapter for the Magenta Groove MIDI Dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Optional[str] = None,
        **kwargs,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path("data/groove")
        super().__init__(df=df, **kwargs)

    @classmethod
    def from_info_csv(
        cls,
        info_csv_path: str,
        split: str = "train",
        root_dir: Optional[str] = None,
        **kwargs,
    ) -> "GrooveDataset":
        """Load Groove dataset preserving official train/validation/test split."""
        df = pd.read_csv(info_csv_path)
        
        # Standardize column names if needed
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if "bpm" in lower:
                col_map[col] = "bpm"
            elif "style" in lower:
                col_map[col] = "genre"
            elif "time_signature" in lower:
                col_map[col] = "meter"
            elif "audio_filename" in lower:
                col_map[col] = "audio_path"
            elif "split" in lower:
                col_map[col] = "split"
            elif "id" in lower and "source" not in lower:
                col_map[col] = "source_id"
        
        df = df.rename(columns=col_map)
        
        if "source_id" not in df.columns:
            # Fallback to drummer id + beat name if available
            df["source_id"] = df.get("drummer", df.index.astype(str))
            
        if "split" in df.columns and split:
            df = df[df["split"].str.lower() == split.lower()].copy()

        return cls(df=df, root_dir=root_dir, **kwargs)

    def load_audio(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        audio_rel_path = str(row["audio_path"])
        full_path = self.root_dir / audio_rel_path
        
        if not full_path.exists():
            # If full path doesn't exist, try resolving directly
            full_path = Path(audio_rel_path)

        waveform, sr = torchaudio.load(str(full_path))
        return waveform, sr, audio_rel_path

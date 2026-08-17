import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class FeatureCache:
    """Disk-based feature cache ensuring stale features are never silently reused.

    Cache key is generated via SHA-256 over the audio identifier and all feature parameters.
    """

    def __init__(self, cache_dir: Union[str, Path] = ".cache/features", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        audio_id: str,
        params: Dict[str, Any],
    ) -> str:
        """Create deterministic SHA-256 hash from audio identity and feature parameters."""
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        hash_input = f"{audio_id}::{sorted_params}".encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    def get(
        self,
        audio_id: str,
        params: Dict[str, Any],
    ) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve cached features if available."""
        if not self.enabled:
            return None

        key = self.generate_cache_key(audio_id, params)
        file_path = self.cache_dir / f"{key}.pt"
        if file_path.exists():
            try:
                data = torch.load(file_path, weights_only=True)
                return data
            except Exception:
                # If file corrupted, return None to trigger recomputation
                return None
        return None

    def put(
        self,
        audio_id: str,
        params: Dict[str, Any],
        features: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Save extracted features to disk cache."""
        if not self.enabled:
            return

        key = self.generate_cache_key(audio_id, params)
        file_path = self.cache_dir / f"{key}.pt"
        try:
            torch.save(features, file_path)
        except Exception:
            pass

    def clear(self) -> int:
        """Clear all cached feature files and return count of deleted files."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for f in self.cache_dir.glob("*.pt"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return count

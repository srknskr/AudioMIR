from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from automir.datasets.base import BaseRhythmDataset


def generate_synthetic_drum_audio(
    duration: float = 8.0,
    sample_rate: int = 22050,
    bpm: float = 120.0,
    style: str = "rock",
    meter: str = "4/4",
    seed: int = 42,
) -> np.ndarray:
    """Deterministically synthesize rhythmic drum audio with kick and snare hits."""
    rng = np.random.RandomState(seed)
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    beat_duration = 60.0 / bpm
    beats_per_bar = 3 if meter == "3/4" else 4

    time_vec = np.arange(total_samples) / sample_rate
    num_beats = int(duration / beat_duration)

    for b in range(num_beats):
        beat_time = b * beat_duration
        sample_idx = int(beat_time * sample_rate)
        if sample_idx >= total_samples:
            break

        bar_beat = b % beats_per_bar

        # Kick on downbeat (0) and beat 2 in 4/4
        if bar_beat == 0 or (beats_per_bar == 4 and bar_beat == 2 and style in ["rock", "funk"]):
            hit_len = min(int(0.15 * sample_rate), total_samples - sample_idx)
            t = np.arange(hit_len) / sample_rate
            # Pitch drop sine
            freq = np.linspace(120, 45, hit_len)
            env = np.exp(-t * 25)
            audio[sample_idx : sample_idx + hit_len] += 0.8 * np.sin(2 * np.pi * freq * t) * env

        # Snare on backbeats (1 and 3 in 4/4, or beat 1 in 3/4)
        if (beats_per_bar == 4 and bar_beat in [1, 3]) or (beats_per_bar == 3 and bar_beat == 1):
            hit_len = min(int(0.12 * sample_rate), total_samples - sample_idx)
            t = np.arange(hit_len) / sample_rate
            env = np.exp(-t * 20)
            noise = rng.normal(0, 0.4, hit_len)
            tone = np.sin(2 * np.pi * 180 * t) * 0.3
            audio[sample_idx : sample_idx + hit_len] += (noise + tone) * env

        # Hi-hat on 8th notes
        eighth_time = beat_time + beat_duration / 2.0
        eighth_idx = int(eighth_time * sample_rate)
        if eighth_idx < total_samples:
            hh_len = min(int(0.04 * sample_rate), total_samples - eighth_idx)
            env_hh = np.exp(-np.arange(hh_len) / (0.01 * sample_rate))
            audio[eighth_idx : eighth_idx + hh_len] += rng.normal(0, 0.15, hh_len) * env_hh

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    return audio


def create_synthetic_manifest(
    num_samples: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a balanced synthetic rhythm manifest with metadata and source_id groups."""
    rng = np.random.RandomState(seed)
    styles = ["rock", "funk", "jazz", "latin"]
    meters = ["4/4", "3/4"]

    records = []
    # Create 10 source groups with 4 variations each
    num_groups = max(1, num_samples // 4)
    for g in range(num_groups):
        source_id = f"synth_pack_{g:02d}"
        base_bpm = float(rng.randint(60, 181))
        genre = styles[g % len(styles)]
        meter = meters[g % len(meters)]

        for v in range(4):
            if len(records) >= num_samples:
                break
            bpm_jitter = base_bpm + float(rng.choice([-2, 0, 2]))
            records.append({
                "audio_path": f"synthetic://{source_id}_var_{v}.wav",
                "bpm": round(bpm_jitter, 1),
                "genre": genre,
                "meter": meter,
                "source_id": source_id,
                "variation_id": v,
                "seed": seed + g * 10 + v,
            })

    return pd.DataFrame(records)


class SyntheticRhythmDataset(BaseRhythmDataset):
    """Synthetic dataset for offline testing and continuous integration."""

    def __init__(self, df: Optional[pd.DataFrame] = None, num_samples: int = 40, **kwargs):
        if df is None:
            df = create_synthetic_manifest(num_samples=num_samples)
        super().__init__(df=df, **kwargs)
        self._audio_cache: dict[int, tuple[torch.Tensor, int, str]] = {}

    def load_audio(self, index: int) -> tuple[torch.Tensor, int, str]:
        if index in self._audio_cache:
            return self._audio_cache[index]

        row = self.df.iloc[index]
        seed = int(row.get("seed", 42 + index))
        bpm = float(row["bpm"])
        genre = str(row["genre"])
        meter = str(row["meter"])
        audio_id = str(row["audio_path"])

        audio_np = generate_synthetic_drum_audio(
            duration=self.segment_duration + 2.0,  # extra margin
            sample_rate=self.sample_rate,
            bpm=bpm,
            style=genre,
            meter=meter,
            seed=seed,
        )
        waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
        result = (waveform, self.sample_rate, audio_id)
        self._audio_cache[index] = result
        return result

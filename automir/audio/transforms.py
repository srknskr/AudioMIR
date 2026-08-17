import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torchaudio
import librosa


def preprocess_audio(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    target_sample_rate: int = 22050,
    target_duration: Optional[float] = 8.0,
    is_training: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Standardize raw audio waveform:
    - Convert to mono
    - Resample to target_sample_rate
    - Deterministic center crop (eval) or random crop (training) or pad to target_duration
    - Peak amplitude normalization
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    else:
        waveform = waveform.float()

    # Ensure 2D (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.t()

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # Peak normalization
    if normalize:
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform / max_val

    # Duration cropping / padding
    if target_duration is not None and target_duration > 0:
        target_samples = int(target_sample_rate * target_duration)
        current_samples = waveform.shape[-1]

        if current_samples < target_samples:
            # Zero pad
            pad_amount = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif current_samples > target_samples:
            if is_training:
                # Random crop
                max_start = current_samples - target_samples
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                # Deterministic center crop
                start = (current_samples - target_samples) // 2
            waveform = waveform[:, start : start + target_samples]

    return waveform


def compute_logmel(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_mels: int = 96,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 20.0,
    f_max: Optional[float] = 11025.0,
    top_db: float = 80.0,
) -> torch.Tensor:
    """Compute Log-Mel Spectrogram with shape (1, n_mels, time_frames)."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max if f_max is not None else float(sample_rate // 2),
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec = mel_transform(waveform)
    # Log compression (dB scale)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
    log_mel = amplitude_to_db(mel_spec)
    # Normalize between ~0 and 1 or standardize
    log_mel = (log_mel + top_db) / top_db
    return log_mel


def compute_tempogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    hop_length: int = 512,
    win_length: int = 384,
    bpm_min: float = 30.0,
    bpm_max: float = 300.0,
) -> torch.Tensor:
    """Compute Tempogram matrix using librosa Fourier tempogram."""
    audio_np = waveform.squeeze().cpu().numpy()

    onset_env = librosa.onset.onset_strength(
        y=audio_np, sr=sample_rate, hop_length=hop_length
    )

    if len(onset_env) == 0:
        return torch.zeros(1, win_length // 2 + 1, 10, dtype=torch.float32)

    # Pad onset envelope if shorter than win_length
    if len(onset_env) < win_length:
        pad_width = win_length - len(onset_env)
        onset_env = np.pad(onset_env, (0, pad_width), mode="constant")

    tempogram_np = librosa.feature.fourier_tempogram(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
    )
    tempogram_mag = np.abs(tempogram_np).astype(np.float32)

    # Normalize tempogram
    max_val = np.max(tempogram_mag)
    if max_val > 1e-6:
        tempogram_mag = tempogram_mag / max_val

    tempogram_tensor = torch.from_numpy(tempogram_mag).unsqueeze(0)
    return tempogram_tensor


def extract_features(
    waveform: torch.Tensor,
    representation: str = "logmel",
    sample_rate: int = 22050,
    n_mels: int = 96,
    hop_length: int = 512,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Extract audio features according to the specified representation:
    - 'logmel': Log-Mel Spectrogram (1, n_mels, T)
    - 'tempogram': Tempogram (1, tempo_bins, T)
    - 'logmel_tempogram': Tuple of (logmel_tensor, tempogram_tensor)
    """
    if representation == "logmel":
        return compute_logmel(
            waveform, sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length
        )
    elif representation == "tempogram":
        return compute_tempogram(
            waveform, sample_rate=sample_rate, hop_length=hop_length
        )
    elif representation == "logmel_tempogram":
        mel = compute_logmel(
            waveform, sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length
        )
        tempo = compute_tempogram(
            waveform, sample_rate=sample_rate, hop_length=hop_length
        )
        return mel, tempo
    else:
        raise ValueError(f"Unknown audio representation: {representation}")

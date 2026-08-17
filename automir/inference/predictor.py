import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio

import librosa

from automir.audio.transforms import preprocess_audio, extract_features
from automir.models.factory import build_model, count_trainable_parameters, get_model_size_mb
from automir.models.heads import TempoRegressionHead
from automir.utils.device import get_device


class AutoMIRPredictor:
    """Production and demo inference engine for rhythm understanding models."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        style_classes: Optional[List[str]] = None,
        meter_classes: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or get_device()
        self.style_classes = list(style_classes or ["rock", "funk", "jazz", "latin"])
        self.meter_classes = list(meter_classes or ["4/4", "3/4"])

        state_dict = None
        num_styles = len(self.style_classes)
        num_meters = len(self.meter_classes)
        enable_meter = True

        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            # Detect exact trained output head dimensions
            if "style_head.net.3.weight" in state_dict:
                num_styles = state_dict["style_head.net.3.weight"].shape[0]
            if "meter_head.net.3.weight" in state_dict:
                num_meters = state_dict["meter_head.net.3.weight"].shape[0]
            elif not any("meter_head" in k for k in state_dict.keys()):
                enable_meter = False

        # Adjust class labels length
        if len(self.style_classes) < num_styles:
            self.style_classes.extend([f"style_{i}" for i in range(len(self.style_classes), num_styles)])
        else:
            self.style_classes = self.style_classes[:num_styles]

        if len(self.meter_classes) < num_meters:
            self.meter_classes.extend([f"meter_{i}" for i in range(len(self.meter_classes), num_meters)])
        else:
            self.meter_classes = self.meter_classes[:num_meters]

        self.model = build_model(
            config=self.config,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.param_count = count_trainable_parameters(self.model)
        self.model_size_mb = get_model_size_mb(self.model)

    def predict(
        self,
        audio_path_or_waveform: Union[str, Path, torch.Tensor, np.ndarray],
        sample_rate: int = 22050,
    ) -> Dict[str, Any]:
        """Run single-item inference on input audio and return rhythm predictions."""
        # 1. Load waveform
        if isinstance(audio_path_or_waveform, (str, Path)):
            try:
                y, sr = librosa.load(str(audio_path_or_waveform), sr=sample_rate, mono=True)
                waveform = torch.from_numpy(y).unsqueeze(0).float()
            except Exception:
                waveform, sr = torchaudio.load(str(audio_path_or_waveform))
        elif isinstance(audio_path_or_waveform, np.ndarray):
            waveform = torch.from_numpy(audio_path_or_waveform).float()
            sr = sample_rate
        else:
            waveform = audio_path_or_waveform
            sr = sample_rate

        # 2. Preprocess audio
        dur = float(self.config.get("segment_duration", 8.0))
        proc_wave = preprocess_audio(
            waveform=waveform,
            sample_rate=sr,
            target_sample_rate=22050,
            target_duration=dur,
            is_training=False,
        )

        # 3. Extract features
        rep = self.config.get("representation", "logmel")
        n_mels = int(self.config.get("n_mels", 96))
        features = extract_features(
            waveform=proc_wave,
            representation=rep,
            sample_rate=22050,
            n_mels=n_mels,
        )

        # 4. Measure inference latency
        t0 = time.perf_counter()
        with torch.no_grad():
            if isinstance(features, tuple):
                mel = features[0].unsqueeze(0).to(self.device)
                tempo = features[1].unsqueeze(0).to(self.device)
                preds = self.model(mel, tempo)
            else:
                feat = features.unsqueeze(0).to(self.device)
                preds = self.model(feat)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Decode predictions
        log2_bpm = preds["log2_bpm"].cpu()
        predicted_bpm = float(TempoRegressionHead.to_linear_bpm(log2_bpm).item())

        style_probs = torch.softmax(preds["style_logits"], dim=-1).cpu().numpy()[0]
        top_style_idx = int(np.argmax(style_probs))
        predicted_style = self.style_classes[top_style_idx] if top_style_idx < len(self.style_classes) else str(top_style_idx)
        style_confidence = float(style_probs[top_style_idx])

        predicted_meter = None
        if "meter_logits" in preds:
            meter_probs = torch.softmax(preds["meter_logits"], dim=-1).cpu().numpy()[0]
            top_meter_idx = int(np.argmax(meter_probs))
            predicted_meter = self.meter_classes[top_meter_idx] if top_meter_idx < len(self.meter_classes) else str(top_meter_idx)

        return {
            "predicted_bpm": round(predicted_bpm, 1),
            "predicted_style": predicted_style,
            "style_confidence": round(style_confidence, 4),
            "style_probabilities": {
                self.style_classes[i] if i < len(self.style_classes) else str(i): round(float(p), 4)
                for i, p in enumerate(style_probs)
            },
            "predicted_meter": predicted_meter,
            "latency_ms": round(latency_ms, 2),
            "params": self.param_count,
            "model_size_mb": round(self.model_size_mb, 3),
            "representation": rep,
            "segment_duration": dur,
            "waveform": proc_wave.squeeze().cpu().numpy(),
        }

from typing import Optional
import torch
import torch.nn as nn


class TempoRegressionHead(nn.Module):
    """Tempo prediction head predicting log2(BPM).

    Log-space regression handles octave jumps and wide tempo ranges smoothly.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns log2(BPM) predictions of shape (B,)
        return self.net(x).squeeze(-1)

    @staticmethod
    def to_linear_bpm(log2_bpm: torch.Tensor) -> torch.Tensor:
        """Convert log2(BPM) prediction back to linear BPM (2^log2_bpm)."""
        return torch.pow(2.0, log2_bpm)


class StyleClassificationHead(nn.Module):
    """Rhythm style/genre classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns raw classification logits of shape (B, num_classes)
        return self.net(x)


class MeterClassificationHead(nn.Module):
    """Time signature / meter classification head (e.g., 4/4, 3/4, 6/8)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits of shape (B, num_classes)
        return self.net(x)

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from automir.models.backbones import ConvBlock, CNNBackbone
from automir.models.heads import (
    TempoRegressionHead,
    StyleClassificationHead,
    MeterClassificationHead,
)


class TinyCNN(nn.Module):
    """Pure lightweight CNN multi-task rhythm model."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_styles: int = 9,
        num_meters: int = 3,
        enable_meter: bool = True,
    ):
        super().__init__()
        self.backbone = CNNBackbone(
            in_channels=in_channels,
            conv_blocks=conv_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        in_features = self.backbone.out_channels

        self.tempo_head = TempoRegressionHead(in_features=in_features, dropout=dropout)
        self.style_head = StyleClassificationHead(
            in_features=in_features, num_classes=num_styles, dropout=dropout
        )
        self.enable_meter = enable_meter
        self.meter_head = (
            MeterClassificationHead(
                in_features=in_features, num_classes=num_meters, dropout=dropout
            )
            if enable_meter
            else None
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        out = {
            "log2_bpm": self.tempo_head(features),
            "style_logits": self.style_head(features),
        }
        if self.enable_meter and self.meter_head is not None:
            out["meter_logits"] = self.meter_head(features)
        return out


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for temporal rhythm modeling."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_styles: int = 9,
        num_meters: int = 3,
        enable_meter: bool = True,
    ):
        super().__init__()
        # Build convolutional blocks without pooling time completely
        layers = []
        curr_in = in_channels
        curr_out = base_channels

        for i in range(conv_blocks):
            # Pool frequency axis more than time axis to preserve temporal beats
            pool = (2, 2)
            layers.append(
                ConvBlock(
                    in_channels=curr_in,
                    out_channels=curr_out,
                    kernel_size=kernel_size,
                    pool_size=pool,
                    dropout=dropout,
                )
            )
            curr_in = curr_out
            curr_out = min(curr_out * 2, 256)

        self.conv = nn.Sequential(*layers)
        self.conv_out_channels = curr_in

        # GRU Layer
        self.gru = nn.GRU(
            input_size=curr_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)

        self.tempo_head = TempoRegressionHead(in_features=gru_out_dim, dropout=dropout)
        self.style_head = StyleClassificationHead(
            in_features=gru_out_dim, num_classes=num_styles, dropout=dropout
        )
        self.enable_meter = enable_meter
        self.meter_head = (
            MeterClassificationHead(
                in_features=gru_out_dim, num_classes=num_meters, dropout=dropout
            )
            if enable_meter
            else None
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, C, F, T)
        conv_feat = self.conv(x)  # (B, C_out, F', T')
        # Average pool over frequency dimension F'
        time_seq = torch.mean(conv_feat, dim=2)  # (B, C_out, T')
        time_seq = time_seq.permute(0, 2, 1)  # (B, T', C_out) for batch_first GRU

        gru_out, _ = self.gru(time_seq)  # (B, T', gru_out_dim)
        # Temporal mean pooling
        pooled = torch.mean(gru_out, dim=1)  # (B, gru_out_dim)

        out = {
            "log2_bpm": self.tempo_head(pooled),
            "style_logits": self.style_head(pooled),
        }
        if self.enable_meter and self.meter_head is not None:
            out["meter_logits"] = self.meter_head(pooled)
        return out


class DualInputNet(nn.Module):
    """Dual-tower neural network consuming both Log-Mel and Tempogram simultaneously."""

    def __init__(
        self,
        conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        use_gru: bool = False,
        gru_hidden: int = 64,
        dropout: float = 0.2,
        num_styles: int = 9,
        num_meters: int = 3,
        enable_meter: bool = True,
    ):
        super().__init__()
        self.use_gru = use_gru

        if use_gru:
            self.mel_tower = CRNN(
                in_channels=1,
                conv_blocks=conv_blocks,
                base_channels=base_channels,
                kernel_size=kernel_size,
                gru_hidden=gru_hidden,
                dropout=dropout,
                enable_meter=False,
            )
            self.tempo_tower = CRNN(
                in_channels=1,
                conv_blocks=conv_blocks,
                base_channels=base_channels,
                kernel_size=kernel_size,
                gru_hidden=gru_hidden,
                dropout=dropout,
                enable_meter=False,
            )
            tower_out_dim = (gru_hidden * 2) * 2  # concat both bidirectional GRUs
        else:
            self.mel_tower = CNNBackbone(
                in_channels=1,
                conv_blocks=conv_blocks,
                base_channels=base_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.tempo_tower = CNNBackbone(
                in_channels=1,
                conv_blocks=conv_blocks,
                base_channels=base_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            tower_out_dim = self.mel_tower.out_channels + self.tempo_tower.out_channels

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(tower_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.tempo_head = TempoRegressionHead(in_features=128, dropout=dropout)
        self.style_head = StyleClassificationHead(
            in_features=128, num_classes=num_styles, dropout=dropout
        )
        self.enable_meter = enable_meter
        self.meter_head = (
            MeterClassificationHead(
                in_features=128, num_classes=num_meters, dropout=dropout
            )
            if enable_meter
            else None
        )

    def forward(
        self,
        mel: torch.Tensor,
        tempogram: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if tempogram is None:
            raise ValueError("DualInputNet requires both 'mel' and 'tempogram' tensors!")

        if self.use_gru:
            # When towers are CRNN, extract pre-head features
            conv_mel = self.mel_tower.conv(mel)
            seq_mel = torch.mean(conv_mel, dim=2).permute(0, 2, 1)
            out_mel, _ = self.mel_tower.gru(seq_mel)
            feat_mel = torch.mean(out_mel, dim=1)

            conv_tempo = self.tempo_tower.conv(tempogram)
            seq_tempo = torch.mean(conv_tempo, dim=2).permute(0, 2, 1)
            out_tempo, _ = self.tempo_tower.gru(seq_tempo)
            feat_tempo = torch.mean(out_tempo, dim=1)
        else:
            feat_mel = self.mel_tower(mel)
            feat_tempo = self.tempo_tower(tempogram)

        combined = torch.cat([feat_mel, feat_tempo], dim=1)
        fused = self.fusion(combined)

        out = {
            "log2_bpm": self.tempo_head(fused),
            "style_logits": self.style_head(fused),
        }
        if self.enable_meter and self.meter_head is not None:
            out["meter_logits"] = self.meter_head(fused)
        return out

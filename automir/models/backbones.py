from typing import List, Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm, ReLU, MaxPool and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: tuple[int, int] = (2, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBackbone(nn.Module):
    """Configurable 2D CNN feature extractor."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_blocks_count = conv_blocks
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []
        curr_in = in_channels
        curr_out = base_channels

        for i in range(conv_blocks):
            layers.append(
                ConvBlock(
                    in_channels=curr_in,
                    out_channels=curr_out,
                    kernel_size=kernel_size,
                    pool_size=(2, 2),
                    dropout=dropout,
                )
            )
            curr_in = curr_out
            curr_out = min(curr_out * 2, 256)

        self.features = nn.Sequential(*layers)
        self.out_channels = curr_in
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T)
        feat = self.features(x)
        pooled = self.adaptive_pool(feat)
        return torch.flatten(pooled, 1)  # (B, out_channels)

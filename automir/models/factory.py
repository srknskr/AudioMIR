import io
from typing import Any, Dict, Union
import torch
import torch.nn as nn

from automir.models.architectures import TinyCNN, CRNN, DualInputNet


def build_model(
    config: Union[Dict[str, Any], Any],
    num_styles: int = 9,
    num_meters: int = 3,
    enable_meter: bool = True,
) -> nn.Module:
    """Instantiate and configure model based on architecture configuration."""
    cfg = config if isinstance(config, dict) else config.to_dict()

    rep = cfg.get("representation", "logmel")
    conv_blocks = int(cfg.get("conv_blocks", 3))
    base_channels = int(cfg.get("base_channels", 32))
    kernel_size = int(cfg.get("kernel_size", 3))
    use_gru = bool(cfg.get("use_gru", False))
    gru_hidden = int(cfg.get("gru_hidden", 64))
    dropout = float(cfg.get("dropout", 0.2))

    # Input validation
    if conv_blocks < 1 or conv_blocks > 5:
        raise ValueError(f"Invalid conv_blocks: {conv_blocks}")
    if base_channels not in [16, 32, 64, 128]:
        raise ValueError(f"Invalid base_channels: {base_channels}")
    if kernel_size not in [3, 5, 7]:
        raise ValueError(f"Invalid kernel_size: {kernel_size}")

    if rep == "logmel_tempogram":
        model = DualInputNet(
            conv_blocks=conv_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            use_gru=use_gru,
            gru_hidden=gru_hidden,
            dropout=dropout,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )
    elif use_gru:
        model = CRNN(
            in_channels=1,
            conv_blocks=conv_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            gru_hidden=gru_hidden,
            dropout=dropout,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )
    else:
        model = TinyCNN(
            in_channels=1,
            conv_blocks=conv_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_styles=num_styles,
            num_meters=num_meters,
            enable_meter=enable_meter,
        )

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Return total number of trainable weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Compute serialized state_dict size in Megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes / (1024.0 * 1024.0)

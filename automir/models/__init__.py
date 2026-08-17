"""Model architectures, backbones, multi-task heads, and factory."""

from automir.models.backbones import ConvBlock, CNNBackbone
from automir.models.heads import (
    TempoRegressionHead,
    StyleClassificationHead,
    MeterClassificationHead,
)
from automir.models.architectures import TinyCNN, CRNN, DualInputNet
from automir.models.factory import (
    build_model,
    count_trainable_parameters,
    get_model_size_mb,
)

__all__ = [
    "ConvBlock",
    "CNNBackbone",
    "TempoRegressionHead",
    "StyleClassificationHead",
    "MeterClassificationHead",
    "TinyCNN",
    "CRNN",
    "DualInputNet",
    "build_model",
    "count_trainable_parameters",
    "get_model_size_mb",
]

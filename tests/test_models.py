import pytest
import torch

from automir.automl.candidate import CandidateConfig
from automir.models.factory import (
    build_model,
    count_trainable_parameters,
    get_model_size_mb,
)


def test_tiny_cnn_forward():
    config = {
        "representation": "logmel",
        "conv_blocks": 2,
        "base_channels": 16,
        "kernel_size": 3,
        "use_gru": False,
    }
    model = build_model(config, num_styles=4, num_meters=2, enable_meter=True)
    x = torch.randn(2, 1, 64, 100)  # (batch, channel, freq, time)
    out = model(x)

    assert "log2_bpm" in out
    assert "style_logits" in out
    assert "meter_logits" in out
    assert out["log2_bpm"].shape == (2,)
    assert out["style_logits"].shape == (2, 4)
    assert out["meter_logits"].shape == (2, 2)


def test_crnn_forward():
    config = {
        "representation": "tempogram",
        "conv_blocks": 3,
        "base_channels": 32,
        "kernel_size": 3,
        "use_gru": True,
        "gru_hidden": 64,
    }
    model = build_model(config, num_styles=9, num_meters=3, enable_meter=True)
    x = torch.randn(2, 1, 193, 80)
    out = model(x)

    assert out["log2_bpm"].shape == (2,)
    assert out["style_logits"].shape == (2, 9)


def test_dual_input_net_forward():
    config = {
        "representation": "logmel_tempogram",
        "conv_blocks": 2,
        "base_channels": 16,
        "kernel_size": 3,
        "use_gru": False,
    }
    model = build_model(config, num_styles=4, num_meters=2, enable_meter=False)
    mel = torch.randn(2, 1, 64, 100)
    tempo = torch.randn(2, 1, 193, 100)
    out = model(mel, tempo)

    assert "log2_bpm" in out
    assert "style_logits" in out
    assert "meter_logits" not in out  # disabled


def test_model_size_and_param_count():
    config = {"representation": "logmel", "conv_blocks": 2, "base_channels": 16}
    model = build_model(config, num_styles=4)
    params = count_trainable_parameters(model)
    size_mb = get_model_size_mb(model)

    assert params > 0
    assert size_mb > 0.0
    assert size_mb < 50.0  # reasonable size


def test_invalid_candidate_config_validation():
    # Invalid conv_blocks
    invalid_cand = CandidateConfig(conv_blocks=9)
    assert not invalid_cand.validate()

    # Invalid learning rate
    invalid_lr = CandidateConfig(learning_rate=100.0)
    assert not invalid_lr.validate()

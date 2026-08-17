import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from automir.models.factory import count_trainable_parameters, get_model_size_mb
from automir.utils.device import get_device_info


def benchmark_efficiency(
    model: nn.Module,
    sample_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    warmup_runs: int = 10,
    measured_runs: int = 50,
) -> Dict[str, Any]:
    """Benchmark inference latency (batch_size=1) and model footprint.

    Includes warm-up passes, GPU/MPS synchronization, and calculates median & p95 latency.
    """
    model = model.to(device)
    model.eval()

    # Prepare input tensor(s)
    if isinstance(sample_input, tuple):
        inputs = tuple(x.to(device) for x in sample_input)
    else:
        inputs = sample_input.to(device)

    def _forward():
        if isinstance(inputs, tuple):
            return model(inputs[0], inputs[1])
        else:
            return model(inputs)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    with torch.no_grad():
        # Warmup passes
        for _ in range(warmup_runs):
            _ = _forward()
            _sync()

        # Measurement passes
        latencies = []
        for _ in range(measured_runs):
            _sync()
            t0 = time.perf_counter()
            _ = _forward()
            _sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))

    param_count = count_trainable_parameters(model)
    model_size_mb = get_model_size_mb(model)

    return {
        "params": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "latency_ms": round(median_latency, 3),
        "latency_p95_ms": round(p95_latency, 3),
        "latency_mean_ms": round(mean_latency, 3),
        "device_info": get_device_info(),
    }

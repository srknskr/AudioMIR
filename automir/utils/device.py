import os
import platform
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Automatically select and return the best available compute device.

    Priority hierarchy:
    CUDA -> Apple MPS -> CPU
    """
    if preference and preference.lower() != "auto":
        pref = preference.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return descriptive string for the device."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return f"CPU ({platform.processor() or platform.machine()})"


def get_device_info() -> dict:
    """Collect hardware metadata for experiment tracking and benchmark comparisons."""
    device = get_device()
    info = {
        "device_type": device.type,
        "device_name": get_device_name(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_capability"] = torch.cuda.get_device_capability()
    return info

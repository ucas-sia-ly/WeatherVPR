"""Strict weather model loading for inference entry points.

The day model is loaded from the original SALAD checkpoint.  A ``full``
weather artifact contains a second DINOv2 backbone only; the day model and
its frozen SALAD aggregator remain the source of gallery descriptors.
"""
from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file

from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.models.io import adapter_model_kwargs


def load_weather_model(base, weather: str, directory: Path, metadata: dict,
                       *, device: str, adapter_scale: float = 1.0) -> WeatherAdapterSALAD:
    """Construct one weather branch and strictly restore its saved tensors."""
    configuration = adapter_model_kwargs(metadata)
    kind = configuration["adapter_type"]
    if kind != "lora" and adapter_scale != 1.0:
        raise ValueError("adapter-scale applies only to LoRA adapters")
    net = WeatherAdapterSALAD(base, (weather,), **configuration).to(device).eval()
    weights = load_file(str(Path(directory) / "adapter.safetensors"), device="cpu")
    net.adapters[weather].load_state_dict(weights, strict=True)
    if kind == "lora":
        net.set_lora_scale(weather, adapter_scale)
    return net

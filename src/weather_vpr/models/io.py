"""Small adapter metadata contract shared by training and inference.

The frozen daytime DINOv2-SALAD checkpoint is stored once. A weather directory
contains either lightweight adaptation tensors or a full weather-specific
DINOv2 backbone, plus this configuration. Old output-adapter runs predate the
``adapter_type`` field and remain loadable.
"""
from __future__ import annotations


def adapter_model_kwargs(metadata: dict) -> dict:
    kind = metadata.get("adapter_type", "output")
    if kind not in {"output", "lora", "full"}:
        raise ValueError(f"Unsupported adapter_type in metadata: {kind}")
    return {
        "adapter_type": kind,
        "bottleneck": int(metadata.get("bottleneck", 64)),
        "lora_rank": int(metadata.get("lora_rank", 8)),
        "lora_alpha": float(metadata.get("lora_alpha", 16)),
        "lora_dropout": float(metadata.get("lora_dropout", 0)),
        "lora_blocks": int(metadata.get("lora_blocks", 4)),
    }

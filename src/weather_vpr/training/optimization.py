"""Optimization helpers for the independent, fully trainable weather backbone.

These helpers never inspect or modify the frozen daytime model. Pass the
selected ``net.adapters[weather]`` DINOBackbone, rather than its owning model.
"""
from __future__ import annotations

import math
import re

from torch import nn


_BLOCK_PARAMETER = re.compile(r"model\.blocks\.(\d+)\.")
_TOKEN_PARAMETERS = {"cls_token", "mask_token", "register_tokens", "pos_embed"}


def full_backbone_parameter_groups(
    weather_backbone: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    layer_decay: float = 1.0,
) -> list[dict]:
    """Build AdamW groups covering every selected full-backbone parameter once.

    For N transformer blocks, the embedding group uses ``layer_decay ** N``;
    block i uses ``layer_decay ** (N - 1 - i)``. The last block and final norm
    therefore both use ``learning_rate``. No layers are frozen. Biases, scalar
    and vector parameters, positional embeddings, and tokens have zero weight
    decay. Returned ``initial_lr`` values can be multiplied by a schedule.

    This intentionally rejects wrappers containing other models, chunked blocks,
    unknown parameter paths, and frozen parameters instead of silently skipping
    them. The supported layout is the project's unchunked full DINOBackbone.
    """
    if (not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if (not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool)
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (not isinstance(layer_decay, (int, float)) or isinstance(layer_decay, bool)
            or not math.isfinite(layer_decay) or not 0 < layer_decay <= 1):
        raise ValueError("layer_decay must be finite and in (0, 1]")

    model = getattr(weather_backbone, "model", None)
    blocks = getattr(model, "blocks", None)
    if (not isinstance(weather_backbone, nn.Module)
            or getattr(weather_backbone, "full_finetune", False) is not True
            or not isinstance(model, nn.Module)
            or not isinstance(blocks, nn.ModuleList) or len(blocks) == 0
            or bool(getattr(model, "chunked_blocks", False))
            or not isinstance(getattr(model, "patch_embed", None), nn.Module)
            or not isinstance(getattr(model, "norm", None), nn.Module)
            or not isinstance(getattr(model, "cls_token", None), nn.Parameter)
            or not isinstance(getattr(model, "pos_embed", None), nn.Parameter)):
        raise ValueError("Expected a full DINOBackbone with unchunked model.blocks")

    block_count = len(blocks)
    groups: dict[tuple[int, bool], dict] = {}
    seen: dict[int, tuple[int, bool]] = {}
    for name, parameter in weather_backbone.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            raise ValueError(f"Full weather backbone parameter is frozen: {name}")
        match = _BLOCK_PARAMETER.match(name)
        if match:
            block_index = int(match.group(1))
            if block_index >= block_count:
                raise ValueError(f"Parameter refers to an unknown DINOv2 block: {name}")
            layer_id = block_index + 1
            exponent = block_count - 1 - block_index
        elif name.startswith("model.patch_embed.") or name in {
                f"model.{token}" for token in _TOKEN_PARAMETERS}:
            layer_id, exponent = 0, block_count
        elif name.startswith("model.norm."):
            layer_id, exponent = block_count, 0
        else:
            raise ValueError(f"Unsupported full DINOBackbone parameter path: {name}")

        no_decay = (parameter.ndim <= 1 or name.endswith(".bias")
                    or name.rsplit(".", 1)[-1] in _TOKEN_PARAMETERS)
        key = (layer_id, no_decay)
        identity = id(parameter)
        if identity in seen:
            if seen[identity] != key:
                raise ValueError(f"Shared parameter has conflicting layer assignments: {name}")
            continue
        seen[identity] = key
        if key not in groups:
            scale = float(layer_decay) ** exponent
            rate = float(learning_rate) * scale
            groups[key] = {
                "params": [], "param_names": [], "layer_id": layer_id,
                "lr": rate, "initial_lr": rate, "lr_scale": scale,
                "weight_decay": 0.0 if no_decay else float(weight_decay),
            }
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    if not seen:
        raise ValueError("Full weather backbone has no parameters")
    expected = {id(parameter) for parameter in weather_backbone.parameters()}
    if set(seen) != expected:
        raise RuntimeError("Full weather backbone parameter coverage is incomplete")
    return [groups[key] for key in sorted(groups)]


def warmup_cosine_multiplier(
    optimizer_step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    min_multiplier: float = 0.0,
) -> float:
    """Return the LR multiplier before a zero-based optimizer update.

    Warmup updates use ``(step + 1) / warmup_steps``. The subsequent cosine
    starts at 1 and reaches ``min_multiplier`` on the final update. With only
    one post-warmup update, that update uses the minimum. A one-update run with
    no warmup uses 1. Warmup must leave at least one post-warmup update; invalid
    steps are rejected, so an accumulation microbatch is never a scheduler step.
    """
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(optimizer_step) is not int or not 0 <= optimizer_step < total_steps:
        raise ValueError("optimizer_step must be in [0, total_steps)")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if (not isinstance(min_multiplier, (int, float)) or isinstance(min_multiplier, bool)
            or not math.isfinite(min_multiplier) or not 0 <= min_multiplier <= 1):
        raise ValueError("min_multiplier must be finite and in [0, 1]")
    if total_steps == 1:
        return 1.0
    if optimizer_step < warmup_steps:
        return (optimizer_step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps == 1:
        return float(min_multiplier)
    progress = (optimizer_step - warmup_steps) / (decay_steps - 1)
    return float(min_multiplier) + (1.0 - float(min_multiplier)) * (
        1.0 + math.cos(math.pi * progress)) / 2.0

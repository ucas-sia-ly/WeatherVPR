"""Optimization helpers for the independent weather backbone.

These helpers never inspect or modify the frozen daytime model. Pass the
selected ``net.adapters[weather]`` DINOBackbone, rather than its owning model.
"""
from __future__ import annotations

import math
import re

from torch import nn


_BLOCK_PARAMETER = re.compile(r"model\.blocks\.(\d+)\.")
_TOKEN_PARAMETERS = {"cls_token", "mask_token", "register_tokens", "pos_embed"}


def _full_backbone_blocks(weather_backbone: nn.Module) -> nn.ModuleList:
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
    return blocks


def _parameter_layer(name: str, block_count: int) -> tuple[int, int]:
    match = _BLOCK_PARAMETER.match(name)
    if match:
        block_index = int(match.group(1))
        if block_index >= block_count:
            raise ValueError(f"Parameter refers to an unknown DINOv2 block: {name}")
        return block_index + 1, block_count - 1 - block_index
    if name.startswith("model.patch_embed.") or name in {
            f"model.{token}" for token in _TOKEN_PARAMETERS}:
        return 0, block_count
    if name.startswith("model.norm."):
        return block_count, 0
    raise ValueError(f"Unsupported full DINOBackbone parameter path: {name}")


def configure_full_backbone_scope(
    weather_backbone: nn.Module,
    scope: str = "all",
    block_count: int = 6,
) -> dict:
    """Select full-weight fine-tuning layers in an independent weather copy.

    ``all`` enables every parameter, including embeddings, tokens and final
    norm. ``early`` and ``late`` enable only the first or last ``block_count``
    transformer blocks; all other parameters are frozen. The complete state
    dict is retained for inference regardless of the training scope.

    ``full_finetune`` remains enabled and the backend's differentiable block
    path covers every block. Frozen later blocks and final norm must propagate
    gradients to early trainable blocks; a frozen prefix naturally builds no
    graph when its images and parameters do not require gradients. This is
    full-weight block selection, not LoRA.
    """
    blocks = _full_backbone_blocks(weather_backbone)
    if not isinstance(scope, str) or scope not in {"all", "early", "late"}:
        raise ValueError("scope must be all, early, or late")
    if type(block_count) is not int or block_count < 1:
        raise ValueError("block_count must be a positive integer")
    if scope != "all" and block_count > len(blocks):
        raise ValueError("block_count exceeds the backbone block count")

    selected_blocks = (list(range(len(blocks))) if scope == "all" else
                       list(range(block_count)) if scope == "early" else
                       list(range(len(blocks) - block_count, len(blocks))))
    selected_ids = set(selected_blocks)
    assignments: dict[int, tuple[int, bool]] = {}
    # Validate before mutating, including tied parameters spanning two layers.
    for name, parameter in weather_backbone.named_parameters(remove_duplicate=False):
        layer_id, _ = _parameter_layer(name, len(blocks))
        match = _BLOCK_PARAMETER.match(name)
        trainable = scope == "all" or (match is not None and int(match.group(1)) in selected_ids)
        assignment = (layer_id, trainable)
        if id(parameter) in assignments and assignments[id(parameter)] != assignment:
            raise ValueError(f"Shared parameter has conflicting layer assignments: {name}")
        assignments[id(parameter)] = assignment
    if not assignments:
        raise ValueError("Full weather backbone has no parameters")
    if not any(trainable for _, trainable in assignments.values()):
        raise ValueError("Selected weather backbone blocks have no parameters")

    for parameter in weather_backbone.parameters():
        parameter.requires_grad_(assignments[id(parameter)][1])
        # A scope may be reconfigured between experiments or training phases.
        # Stale gradients on newly frozen parameters must not survive.
        if not parameter.requires_grad:
            parameter.grad = None
    weather_backbone.num_trainable_blocks = len(blocks)
    return {
        "scope": scope, "block_indices": selected_blocks,
        "block_count": len(selected_blocks), "total_blocks": len(blocks),
        "trainable_parameters": sum(p.numel() for p in weather_backbone.parameters()
                                    if p.requires_grad),
        "total_parameters": sum(p.numel() for p in weather_backbone.parameters()),
    }


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
    therefore both use ``learning_rate``. Frozen parameters are excluded, while
    layer rates retain their original absolute block positions. Biases, scalar
    and vector parameters, positional embeddings, and tokens have zero weight
    decay. Returned ``initial_lr`` values can be multiplied by a schedule.

    This intentionally rejects wrappers containing other models, chunked blocks,
    and unknown parameter paths, including frozen unknown parameters. The
    supported layout is the project's unchunked full DINOBackbone.
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

    blocks = _full_backbone_blocks(weather_backbone)
    block_count = len(blocks)
    groups: dict[tuple[int, bool], dict] = {}
    seen: dict[int, tuple[int, bool]] = {}
    selected: set[int] = set()
    for name, parameter in weather_backbone.named_parameters(remove_duplicate=False):
        layer_id, exponent = _parameter_layer(name, block_count)
        no_decay = (parameter.ndim <= 1 or name.endswith(".bias")
                    or name.rsplit(".", 1)[-1] in _TOKEN_PARAMETERS)
        key = (layer_id, no_decay)
        identity = id(parameter)
        if identity in seen:
            if seen[identity] != key:
                raise ValueError(f"Shared parameter has conflicting layer assignments: {name}")
            continue
        seen[identity] = key
        if not parameter.requires_grad:
            continue
        selected.add(identity)
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
    if not selected:
        raise ValueError("Full weather backbone has no trainable parameters")
    expected = {id(parameter) for parameter in weather_backbone.parameters()
                if parameter.requires_grad}
    if selected != expected:
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

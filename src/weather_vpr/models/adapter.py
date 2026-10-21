"""One frozen DINOv2-SALAD with independent lightweight adapters per weather.

``output`` keeps the original post-backbone Patch/CLS adapter. ``lora`` adapts
qkv/proj in the last DINOv2 blocks. ``hybrid`` enables both. A direct call to
the base model, including the daytime gallery path, always has LoRA disabled.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import weakref

import torch
import torch.nn.functional as F
from torch import nn


class PatchCLSAdapter(nn.Module):
    """Residual spatial patch adaptation and CLS adaptation; identity at init."""

    def __init__(self, channels: int = 768, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.spatial = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1,
                                 groups=bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1)
        self.cls_down = nn.Linear(channels, bottleneck)
        self.cls_up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.cls_up.weight)
        nn.init.zeros_(self.cls_up.bias)

    def forward(self, patches: torch.Tensor, cls: torch.Tensor):
        delta = self.up(self.spatial(F.gelu(self.down(patches))))
        patch_out = patches + delta
        cls_out = cls + self.cls_up(F.gelu(self.cls_down(cls)))
        return patch_out, cls_out


class LoRADelta(nn.Module):
    """Zero-at-init rank-r update with alpha/r scaling for one Linear layer."""

    def __init__(self, in_features: int, out_features: int, *, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA rank/alpha/dropout must be positive, positive, and in [0,1)")
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class WeatherLoRAAdapter(nn.Module):
    """All LoRA weights for one weather, saved as a self-contained state_dict."""

    def __init__(self, targets: dict[str, tuple[int, int]], *, rank: int,
                 alpha: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleDict({
            name: LoRADelta(in_features, out_features, rank=rank, alpha=alpha,
                            dropout=dropout)
            for name, (in_features, out_features) in targets.items()
        })


class WeatherHybridAdapter(nn.Module):
    """Weather-specific LoRA plus the existing output Patch/CLS adapter."""

    def __init__(self, channels: int, bottleneck: int,
                 targets: dict[str, tuple[int, int]], *, rank: int,
                 alpha: float, dropout: float):
        super().__init__()
        self.output = PatchCLSAdapter(channels, bottleneck)
        self.lora = WeatherLoRAAdapter(targets, rank=rank, alpha=alpha, dropout=dropout)


class _LoRALinear(nn.Module):
    """Frozen original Linear plus the currently selected external LoRA delta.

    The delta is owned by ``net.adapters[weather]``, not registered a second
    time under the shared base. This keeps per-weather state_dicts independent.
    """

    def __init__(self, original: nn.Linear, owner: WeatherAdapterSALAD,
                 key: str):
        super().__init__()
        self.original = original
        self._owner = weakref.ref(owner)
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        owner = self._owner()
        if owner is None or owner._active_weather is None:
            return result
        adapter = owner.adapters[owner._active_weather]
        lora = adapter.lora if isinstance(adapter, WeatherHybridAdapter) else adapter
        return result + owner._lora_scales[owner._active_weather] * lora.layers[self.key](x)


class WeatherAdapterSALAD(nn.Module):
    """Reuse one strictly loaded, frozen DINOv2-SALAD for teacher and student.

    ``adapter_type='output'`` preserves the original output-adapter interface.
    ``'lora'`` targets qkv/proj of the final four DINOv2 blocks by default.
    ``'hybrid'`` combines them. Each ``net.adapters[weather]`` is a complete
    lightweight module suitable for strict ``state_dict`` save/load.
    """

    def __init__(self, base_salad: nn.Module, weather_names=("night",),
                 bottleneck: int = 64, *, adapter_type: str = "lora",
                 lora_rank: int = 8, lora_alpha: float = 16.0,
                 lora_dropout: float = 0.0, lora_blocks: int = 4):
        super().__init__()
        if adapter_type not in {"output", "lora", "hybrid"}:
            raise ValueError("adapter_type must be output, lora, or hybrid")
        if bottleneck < 1:
            raise ValueError("bottleneck must be positive")
        if adapter_type != "output" and (
                type(lora_rank) is not int or lora_rank < 1
                or not isinstance(lora_alpha, (int, float))
                or not math.isfinite(lora_alpha) or lora_alpha <= 0
                or not isinstance(lora_dropout, (int, float))
                or not 0 <= lora_dropout < 1):
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        self.base = base_salad
        self.base.requires_grad_(False)
        self.base.eval()
        self.adapter_type = adapter_type
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_blocks = lora_blocks
        # Context-local selection prevents concurrent inference requests from
        # activating another request's weather adapter on this shared base.
        self._active_weather_context: ContextVar[str | None] = ContextVar(
            f"weather_vpr_lora_{id(self)}", default=None)
        channels = self.base.config["agg_config"]["num_channels"]
        names = tuple(weather_names)
        if len(names) != len(set(names)) or any(not name or name == "day" for name in names):
            raise ValueError("Weather names must be unique, nonempty, and not 'day'")
        self._lora_scales = {name: 1.0 for name in names}

        targets = self._install_lora(lora_blocks) if adapter_type != "output" else {}
        if adapter_type == "output":
            self.adapters = nn.ModuleDict({name: PatchCLSAdapter(channels, bottleneck)
                                           for name in names})
        elif adapter_type == "lora":
            self.adapters = nn.ModuleDict({
                name: WeatherLoRAAdapter(targets, rank=lora_rank, alpha=lora_alpha,
                                         dropout=lora_dropout)
                for name in names
            })
        else:
            self.adapters = nn.ModuleDict({
                name: WeatherHybridAdapter(channels, bottleneck, targets,
                                           rank=lora_rank, alpha=lora_alpha,
                                           dropout=lora_dropout)
                for name in names
            })

    def _install_lora(self, blocks_to_adapt: int) -> dict[str, tuple[int, int]]:
        backbone = self.base.backbone
        blocks = getattr(getattr(backbone, "model", None), "blocks", None)
        if (blocks is None or type(blocks_to_adapt) is not int or blocks_to_adapt < 1
                or blocks_to_adapt > len(blocks)):
            raise ValueError("LoRA requires DINOv2 model.blocks and a valid lora_blocks count")
        targets = {}
        replacements = []
        for index in range(len(blocks) - blocks_to_adapt, len(blocks)):
            attention = getattr(blocks[index], "attn", None)
            if attention is None:
                raise ValueError(f"DINOv2 block {index} has no attn module")
            for projection in ("qkv", "proj"):
                layer = getattr(attention, projection, None)
                if not isinstance(layer, nn.Linear):
                    raise ValueError(f"DINOv2 block {index}.attn.{projection} is not nn.Linear")
                key = f"block_{index}_{projection}"
                targets[key] = (layer.in_features, layer.out_features)
                replacements.append((attention, projection, layer, key))
        for attention, projection, layer, key in replacements:
            setattr(attention, projection, _LoRALinear(layer, self, key))
        # The backend's prefix no_grad boundary must end before every LoRA
        # target. Its final norm remains outside no_grad, preserving input grads.
        backbone.num_trainable_blocks = blocks_to_adapt
        self.base.eval()
        return targets

    @contextmanager
    def _using_weather(self, weather: str | None):
        token = self._active_weather_context.set(weather)
        try:
            yield
        finally:
            self._active_weather_context.reset(token)

    @property
    def _active_weather(self) -> str | None:
        return self._active_weather_context.get()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # SALAD and DINOv2 dropout remain disabled
        return self

    @torch.no_grad()
    def teacher(self, images: torch.Tensor, return_aux=True):
        self.base.eval()
        with self._using_weather(None):
            return self.base(images, return_aux=return_aux)

    def student(self, images: torch.Tensor, weather: str, return_aux=True):
        if weather not in self.adapters:
            raise ValueError(f"Weather adapter not loaded: {weather}")
        if self.adapter_type == "output":
            # Output adapter: do not retain any DINOv2 backward graph.
            with torch.no_grad():
                patch_features, cls = self.base.backbone(images)
            patch_features, cls = self.adapters[weather](patch_features, cls)
        else:
            with self._using_weather(weather):
                patch_features, cls = self.base.backbone(images)
            if self.adapter_type == "hybrid":
                patch_features, cls = self.adapters[weather].output(patch_features, cls)
        return self.base.aggregator((patch_features, cls), return_aux=return_aux)

    def adapter_parameters(self, weather: str):
        return self.adapters[weather].parameters()

    def set_lora_scale(self, weather: str, scale: float) -> None:
        """Set a weather LoRA inference scale; 1.0 is the trained update."""
        if self.adapter_type not in {"lora", "hybrid"}:
            raise ValueError("LoRA scale is available only for lora or hybrid adapters")
        if weather not in self.adapters:
            raise ValueError(f"Weather adapter not loaded: {weather}")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError("LoRA scale must be finite and nonnegative")
        self._lora_scales[weather] = float(scale)

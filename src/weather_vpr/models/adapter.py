"""One frozen DINOv2-SALAD and one lightweight output adapter per weather.

The teacher path and student path share EXACTLY the same base module.
Only adapter parameters receive gradients. No transformer-block LoRA here.
"""
from __future__ import annotations
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

class WeatherAdapterSALAD(nn.Module):
    """Reuse one *strictly loaded* and frozen DINOv2+SALAD for teacher/student.

    The baseline teacher uses no adapter. The student applies only the selected
    weather output adapter, then the same frozen SALAD aggregator.
    """

    def __init__(self, base_salad: nn.Module, weather_names=("night",), bottleneck=64):
        super().__init__()
        self.base = base_salad
        self.base.requires_grad_(False)
        self.base.eval()
        channels = self.base.config["agg_config"]["num_channels"]
        self.adapters = nn.ModuleDict({name: PatchCLSAdapter(channels, bottleneck)
                                       for name in weather_names})

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # SALAD dropout must remain disabled for both branches
        return self

    @torch.no_grad()
    def teacher(self, images: torch.Tensor, return_aux=True):
        self.base.eval()
        return self.base(images, return_aux=return_aux)

    def student(self, images: torch.Tensor, weather: str, return_aux=True):
        if weather not in self.adapters:
            raise ValueError(f"Weather adapter not loaded: {weather}")
        # Output adapter: do not retain *any* DINOv2 backward graph.
        with torch.no_grad():
            patch_features, cls = self.base.backbone(images)
        patch_features, cls = self.adapters[weather](patch_features, cls)
        return self.base.aggregator((patch_features, cls), return_aux=return_aux)

    def adapter_parameters(self, weather: str):
        return self.adapters[weather].parameters()

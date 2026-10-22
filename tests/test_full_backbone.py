"""Full Night DINO fine-tuning tests without downloading model weights."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.utils.backend import enable_backend_imports

enable_backend_imports()
from workflow.model import DINOBackbone  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, tokens):
        return tokens + torch.tanh(self.proj(tokens))


class TinyDINO(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, width, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width) * 0.1)
        self.pos_embed = nn.Parameter(torch.randn(1, 5, width) * 0.1)
        self.blocks = nn.ModuleList([TinyBlock(width), TinyBlock(width)])
        self.norm = nn.LayerNorm(width)

    def prepare_tokens_with_masks(self, images):
        patch = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls, patch), dim=1) + self.pos_embed


def tiny_backbone() -> DINOBackbone:
    # Bypass torch.hub while exercising the backend's actual DINOBackbone.forward.
    backbone = DINOBackbone.__new__(DINOBackbone)
    nn.Module.__init__(backbone)
    backbone.model = TinyDINO()
    backbone.num_channels = 4
    backbone.num_trainable_blocks = 0
    backbone.norm_layer = True
    backbone.full_finetune = False
    return backbone


class TinySALAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tiny_backbone()
        self.aggregator = TinyAggregator()
        self.config = {"agg_config": {"num_channels": 4}}

    def forward(self, images, return_aux=True):
        return self.aggregator(self.backbone(images), return_aux=return_aux)


class TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, features, return_aux=True):
        patches, cls = features
        descriptor = self.projection(patches.mean((2, 3)) + cls)
        if return_aux:
            return descriptor, {"backbone_local_features": patches,
                                "assignment_keep": torch.sigmoid(patches.mean(1).flatten(1))}
        return descriptor


class FullBackboneTests(unittest.TestCase):
    def test_night_copy_is_exact_and_all_dino_parameters_receive_gradients(self):
        torch.manual_seed(112)
        base = TinySALAD().eval()
        initial = copy.deepcopy(base)
        net = WeatherAdapterSALAD(base, ("night",), adapter_type="full")
        images = torch.randn(2, 3, 28, 28)

        self.assertIsNot(net.base.backbone, net.adapters["night"])
        self.assertIsNot(net.base.backbone.model.patch_embed.weight,
                         net.adapters["night"].model.patch_embed.weight)
        self.assertFalse(hasattr(net.adapters["night"], "aggregator"))
        self.assertEqual(net.adapters["night"].num_trainable_blocks, 2)
        self.assertTrue(net.adapters["night"].full_finetune)
        self.assertFalse(net.base.backbone.full_finetune)
        self.assertFalse(any(p.requires_grad for p in net.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.adapter_parameters("night")))

        expected = initial(images, return_aux=False)
        torch.testing.assert_close(net.teacher(images, return_aux=False), expected)
        torch.testing.assert_close(net.student(images, "night", return_aux=False), expected)

        net.train()
        self.assertFalse(net.base.training)
        self.assertFalse(net.base.aggregator.training)
        self.assertTrue(net.adapters["night"].training)
        self.assertTrue(net.adapters["night"].model.training)
        student, aux = net.student(images, "night")
        self.assertTrue(aux["backbone_local_features"].requires_grad)
        student.square().sum().backward()
        night = net.adapters["night"].model
        for parameter in (night.patch_embed.weight, night.cls_token, night.pos_embed,
                          night.blocks[0].proj.weight, night.blocks[1].proj.weight,
                          night.norm.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertFalse(any(parameter.grad is not None for parameter in net.base.parameters()))

        before = {name: tensor.detach().clone()
                  for name, tensor in net.base.state_dict().items()}
        optimizer = torch.optim.SGD(net.adapter_parameters("night"), lr=1e-3)
        optimizer.step()
        self.assertFalse(torch.allclose(net.student(images, "night", return_aux=False),
                                        expected))
        torch.testing.assert_close(net.teacher(images, return_aux=False), expected)
        for name, tensor in net.base.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_strict_night_only_state_and_metadata(self):
        torch.manual_seed(113)
        day = TinySALAD()
        net = WeatherAdapterSALAD(day, ("night",), adapter_type="full")
        state = {name: value.detach().clone()
                 for name, value in net.adapters["night"].state_dict().items()}
        self.assertTrue(state)
        self.assertTrue(all(name.startswith("model.") for name in state))
        self.assertFalse(any("aggregator" in name for name in state))
        self.assertEqual(adapter_model_kwargs({"adapter_type": "full"})["adapter_type"],
                         "full")

        restored = WeatherAdapterSALAD(copy.deepcopy(day), ("night",),
                                       adapter_type="full")
        restored.adapters["night"].load_state_dict(state, strict=True)
        image = torch.randn(1, 3, 28, 28)
        net.eval()
        restored.eval()
        torch.testing.assert_close(net.student(image, "night", return_aux=False),
                                   restored.student(image, "night", return_aux=False))

    def test_default_selects_full_and_rejects_non_dino(self):
        net = WeatherAdapterSALAD(TinySALAD(), ("night",))
        self.assertEqual(net.adapter_type, "full")
        malformed = TinySALAD()
        malformed.backbone.model = nn.Identity()
        with self.assertRaisesRegex(ValueError, "DINOv2 backbone"):
            WeatherAdapterSALAD(malformed, ("night",), adapter_type="full")


if __name__ == "__main__":
    unittest.main()

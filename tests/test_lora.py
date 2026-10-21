"""LoRA tests with a tiny DINO-shaped backbone; no checkpoint or download."""
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import torch
from torch import nn

from weather_vpr.models.adapter import WeatherAdapterSALAD


class FakeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v)


class FakeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = FakeAttention(width)

    def forward(self, x):
        return x + self.attn(x)


class FakeBackbone(nn.Module):
    def __init__(self, width=4, blocks=6):
        super().__init__()
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList(FakeBlock(width) for _ in range(blocks))
        self.model.norm = nn.LayerNorm(width)
        self.num_trainable_blocks = 0

    def forward(self, images):
        batch, width, height, length = images.shape
        patch = images.flatten(2).transpose(1, 2)
        tokens = torch.cat((patch.mean(1, keepdim=True), patch), dim=1)
        split = len(self.model.blocks) - self.num_trainable_blocks
        with torch.no_grad():
            for block in self.model.blocks[:split]:
                tokens = block(tokens)
        for block in self.model.blocks[split:]:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 1:].transpose(1, 2).reshape(batch, width, height, length), tokens[:, 0]


class FakeSALAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.config = {"agg_config": {"num_channels": 4}}
        self.projection = nn.Linear(4, 4)

    def aggregator(self, value, return_aux=True):
        patch, cls = value
        descriptor = self.projection(patch.mean(dim=(-2, -1)) + cls)
        if return_aux:
            return descriptor, {"backbone_local_features": patch}
        return descriptor

    def forward(self, images, return_aux=True):
        return self.aggregator(self.backbone(images), return_aux=return_aux)


class LoRATests(unittest.TestCase):
    def test_teacher_baseline_and_weather_gradients(self):
        torch.manual_seed(7)
        original = FakeSALAD().eval()
        frozen_copy = copy.deepcopy(original)
        net = WeatherAdapterSALAD(original, ("night", "rain"), adapter_type="lora",
                                  lora_rank=2, lora_alpha=4, lora_dropout=0,
                                  lora_blocks=4)
        x = torch.randn(2, 4, 2, 2)
        expected = frozen_copy(x, return_aux=False)
        teacher = net.teacher(x, return_aux=False)
        torch.testing.assert_close(teacher, expected)
        torch.testing.assert_close(net.base(x, return_aux=False), expected)
        torch.testing.assert_close(net.student(x, "night", return_aux=False), expected)
        self.assertEqual(len(net.adapters["night"].layers), 8)
        self.assertEqual(set(net.adapters["night"].layers), {
            f"block_{index}_{projection}" for index in range(2, 6)
            for projection in ("qkv", "proj")
        })
        self.assertEqual(net.base.backbone.num_trainable_blocks, 4)

        net.train()
        self.assertFalse(net.base.training)
        descriptor, aux = net.student(x, "night")
        self.assertTrue(aux["backbone_local_features"].requires_grad)
        aux["backbone_local_features"].retain_grad()
        descriptor.square().sum().backward()
        self.assertIsNotNone(aux["backbone_local_features"].grad)
        self.assertFalse(any(parameter.grad is not None for parameter in net.base.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in net.adapters["night"].parameters()))
        self.assertFalse(any(parameter.grad is not None for parameter in net.adapters["rain"].parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in net.base.parameters()))
        self.assertIsNone(net._active_weather)

        optimizer = torch.optim.SGD(net.adapter_parameters("night"), lr=0.1)
        optimizer.step()
        self.assertFalse(torch.allclose(net.student(x, "night", return_aux=False), expected))
        net.set_lora_scale("night", 0.0)
        torch.testing.assert_close(net.student(x, "night", return_aux=False), expected)
        net.set_lora_scale("night", 1.0)
        torch.testing.assert_close(net.student(x, "rain", return_aux=False), expected)
        torch.testing.assert_close(net.teacher(x, return_aux=False), expected)
        torch.testing.assert_close(net.base(x, return_aux=False), expected)

        state = {key: value.detach().clone() for key, value in
                 net.adapters["night"].state_dict().items()}
        self.assertEqual(len(state), 16)
        self.assertIn("layers.block_2_qkv.lora_A.weight", state)
        self.assertIn("layers.block_5_proj.lora_B.weight", state)
        restored = WeatherAdapterSALAD(copy.deepcopy(frozen_copy), ("night",),
                                        adapter_type="lora", lora_rank=2,
                                        lora_alpha=4, lora_blocks=4)
        restored.adapters["night"].load_state_dict(state, strict=True)
        torch.testing.assert_close(restored.student(x, "night", return_aux=False),
                                   net.student(x, "night", return_aux=False))

    def test_hybrid_and_invalid_target(self):
        torch.manual_seed(9)
        base = FakeSALAD()
        expected = base(torch.randn(1, 4, 2, 2), return_aux=False)
        net = WeatherAdapterSALAD(base, ("night",), bottleneck=2,
                                  adapter_type="hybrid", lora_rank=2,
                                  lora_alpha=4, lora_blocks=4)
        state = net.adapters["night"].state_dict()
        self.assertTrue(any(key.startswith("output.") for key in state))
        self.assertTrue(any(key.startswith("lora.layers.") for key in state))
        self.assertTrue(torch.isfinite(expected).all())
        with self.assertRaisesRegex(ValueError, "lora_blocks"):
            WeatherAdapterSALAD(FakeSALAD(), ("night",), adapter_type="lora",
                                lora_blocks=7)
        malformed = FakeSALAD()
        malformed.backbone.model.blocks[5].attn.proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, "attn.proj"):
            WeatherAdapterSALAD(malformed, ("night",), adapter_type="lora")
        self.assertIsInstance(malformed.backbone.model.blocks[2].attn.qkv, nn.Linear)
        with self.assertRaises(ValueError):
            net.set_lora_scale("night", -1)

    def test_weather_selection_is_context_local(self):
        net = WeatherAdapterSALAD(FakeSALAD(), ("night", "rain"),
                                  adapter_type="lora", lora_blocks=2)
        barrier = Barrier(2)

        def selected(weather):
            with net._using_weather(weather):
                barrier.wait(timeout=2)
                return net._active_weather

        with ThreadPoolExecutor(max_workers=2) as workers:
            night = workers.submit(selected, "night")
            rain = workers.submit(selected, "rain")
            self.assertEqual(night.result(), "night")
            self.assertEqual(rain.result(), "rain")
        self.assertIsNone(net._active_weather)


if __name__ == "__main__":
    unittest.main()

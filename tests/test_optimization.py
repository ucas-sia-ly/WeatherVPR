"""Full-weather optimizer grouping and update-based schedule tests."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from weather_vpr.training.optimization import (
    full_backbone_parameter_groups, warmup_cosine_multiplier,
)


class TinyFullBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_finetune = True
        self.model = nn.Module()
        self.model.chunked_blocks = False
        self.model.patch_embed = nn.Conv2d(3, 4, kernel_size=2)
        self.model.cls_token = nn.Parameter(torch.randn(1, 1, 4))
        self.model.pos_embed = nn.Parameter(torch.randn(1, 5, 4))
        self.model.mask_token = nn.Parameter(torch.randn(1, 4))
        self.model.register_tokens = nn.Parameter(torch.randn(1, 2, 4))
        self.model.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)) for _ in range(3)
        ])
        self.model.norm = nn.LayerNorm(4)


def named_group_values(groups, field):
    return {name: group[field] for group in groups for name in group["param_names"]}


class FullBackboneParameterGroupTests(unittest.TestCase):
    def test_all_weather_parameters_exactly_once_and_teacher_excluded(self):
        torch.manual_seed(13)
        teacher = TinyFullBackbone().requires_grad_(False)
        night = copy.deepcopy(teacher).requires_grad_(True)
        teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
        groups = full_backbone_parameter_groups(
            night, learning_rate=1e-3, weight_decay=0.1, layer_decay=0.5)
        actual = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(actual), len({id(parameter) for parameter in actual}))
        self.assertEqual({id(p) for p in actual}, {id(p) for p in night.parameters()})
        self.assertTrue({id(p) for p in actual}.isdisjoint({id(p) for p in teacher.parameters()}))
        self.assertTrue(all(p.requires_grad for p in actual))
        self.assertEqual(set(named_group_values(groups, "lr")), dict(night.named_parameters()).keys())

        optimizer = torch.optim.AdamW(groups)
        before = {name: p.clone() for name, p in night.named_parameters()}
        sum(parameter.sum() for parameter in night.parameters()).backward()
        optimizer.step()
        for name, parameter in night.named_parameters():
            self.assertFalse(torch.equal(parameter, before[name]), name)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)

    def test_layer_rates_embedding_smallest_and_last_block_norm_base(self):
        groups = full_backbone_parameter_groups(
            TinyFullBackbone(), learning_rate=1e-3, weight_decay=0.1, layer_decay=0.5)
        rates = named_group_values(groups, "lr")
        self.assertAlmostEqual(rates["model.patch_embed.weight"], 1e-3 * 0.5 ** 3)
        self.assertAlmostEqual(rates["model.cls_token"], 1e-3 * 0.5 ** 3)
        self.assertAlmostEqual(rates["model.pos_embed"], 1e-3 * 0.5 ** 3)
        for index in range(3):
            self.assertAlmostEqual(rates[f"model.blocks.{index}.0.weight"],
                                   1e-3 * 0.5 ** (2 - index))
        self.assertEqual(rates["model.norm.weight"], 1e-3)
        self.assertEqual(rates["model.norm.bias"], 1e-3)
        for group in groups:
            self.assertEqual(group["lr"], group["initial_lr"])
            self.assertEqual(group["lr"], 1e-3 * group["lr_scale"])

    def test_decay_one_preserves_identical_learning_rates(self):
        groups = full_backbone_parameter_groups(
            TinyFullBackbone(), learning_rate=7e-7, weight_decay=0.01)
        self.assertEqual({group["lr"] for group in groups}, {7e-7})
        self.assertEqual({group["lr_scale"] for group in groups}, {1.0})

    def test_bias_vectors_tokens_and_positions_have_no_weight_decay(self):
        night = TinyFullBackbone()
        groups = full_backbone_parameter_groups(
            night, learning_rate=1e-3, weight_decay=0.1)
        decay = named_group_values(groups, "weight_decay")
        for name, parameter in night.named_parameters():
            token = name in {"model.cls_token", "model.mask_token", "model.register_tokens",
                             "model.pos_embed"}
            expected = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") or token else 0.1
            self.assertEqual(decay[name], expected, name)
        self.assertEqual(decay["model.patch_embed.weight"], 0.1)
        self.assertEqual(decay["model.blocks.0.0.weight"], 0.1)

    def test_same_layer_alias_cannot_duplicate_a_parameter(self):
        night = TinyFullBackbone()
        night.model.patch_embed.alias = night.model.patch_embed.weight
        groups = full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        parameters = [p for group in groups for p in group["params"]]
        self.assertEqual(len(parameters), len(list(night.parameters())))

    def test_rejects_frozen_unknown_and_non_full_structures(self):
        night = TinyFullBackbone()
        night.model.cls_token.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "frozen"):
            full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        self.assertFalse(night.model.cls_token.requires_grad)
        for mutation in (lambda net: setattr(net, "full_finetune", False),
                         lambda net: setattr(net.model, "chunked_blocks", True),
                         lambda net: setattr(net.model, "blocks", nn.ModuleList()),
                         lambda net: setattr(net.model, "norm", None)):
            night = TinyFullBackbone()
            mutation(night)
            with self.assertRaisesRegex(ValueError, "full DINOBackbone"):
                full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        night = TinyFullBackbone()
        night.other_model = nn.Linear(4, 4)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        with self.assertRaisesRegex(ValueError, "full DINOBackbone"):
            full_backbone_parameter_groups(nn.Linear(4, 4), learning_rate=1e-3, weight_decay=0.1)

    def test_rejects_cross_layer_tied_parameters(self):
        night = TinyFullBackbone()
        night.model.blocks[1][0].weight = night.model.blocks[0][0].weight
        with self.assertRaisesRegex(ValueError, "conflicting layer"):
            full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)

    def test_rejects_invalid_group_hyperparameters(self):
        for changes in ({"learning_rate": 0}, {"learning_rate": float("nan")},
                        {"learning_rate": True}, {"weight_decay": -1},
                        {"weight_decay": float("inf")}, {"layer_decay": 0},
                        {"layer_decay": 1.01}, {"layer_decay": False}):
            kwargs = {"learning_rate": 1e-3, "weight_decay": 0.1, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                full_backbone_parameter_groups(TinyFullBackbone(), **kwargs)


class WarmupCosineTests(unittest.TestCase):
    def test_no_warmup_has_exact_endpoints_and_monotonic_decay(self):
        values = [warmup_cosine_multiplier(step, 9, min_multiplier=0.1)
                  for step in range(9)]
        self.assertEqual(values[0], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_warmup_is_update_based_then_cosine(self):
        values = [warmup_cosine_multiplier(step, 8, warmup_steps=3, min_multiplier=0.2)
                  for step in range(8)]
        self.assertEqual(values[:3], [1 / 3, 2 / 3, 1.0])
        self.assertEqual(values[3], 1.0)
        self.assertAlmostEqual(values[-1], 0.2)
        self.assertTrue(all(a >= b for a, b in zip(values[3:], values[4:])))

    def test_single_update_and_single_decay_update_are_explicit(self):
        self.assertEqual(warmup_cosine_multiplier(0, 1), 1.0)
        self.assertEqual(warmup_cosine_multiplier(0, 1, min_multiplier=0.3), 1.0)
        self.assertEqual([warmup_cosine_multiplier(s, 2) for s in range(2)], [1.0, 0.0])
        self.assertEqual([warmup_cosine_multiplier(s, 3, warmup_steps=2) for s in range(3)],
                         [0.5, 1.0, 0.0])

    def test_invalid_schedule_boundaries_rejected(self):
        for step, total, kwargs in (
                (0, 0, {}), (-1, 3, {}), (3, 3, {}), (True, 3, {}),
                (0, 3.0, {}), (0, 1, {"warmup_steps": 1}),
                (0, 3, {"warmup_steps": 3}), (0, 3, {"warmup_steps": 4}),
                (0, 3, {"warmup_steps": -1}), (0, 3, {"min_multiplier": -0.1}),
                (0, 3, {"min_multiplier": float("nan")}),
                (0, 3, {"min_multiplier": 1.1})):
            with self.subTest(step=step, total=total, kwargs=kwargs), self.assertRaises(ValueError):
                warmup_cosine_multiplier(step, total, **kwargs)


if __name__ == "__main__":
    unittest.main()

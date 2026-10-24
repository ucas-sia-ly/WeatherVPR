"""Full-weather optimizer grouping and update-based schedule tests."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from weather_vpr.training.optimization import (
    configure_full_backbone_scope, full_backbone_parameter_groups,
    warmup_cosine_multiplier,
)
from weather_vpr.utils.backend import enable_backend_imports

enable_backend_imports()
from workflow.model import DINOBackbone  # noqa: E402


class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunked_blocks = False
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 4))
        self.pos_embed = nn.Parameter(torch.randn(1, 5, 4))
        self.mask_token = nn.Parameter(torch.randn(1, 4))
        self.register_tokens = nn.Parameter(torch.randn(1, 2, 4))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)) for _ in range(3)
        ])
        self.norm = nn.LayerNorm(4)

    def prepare_tokens_with_masks(self, images):
        patch = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls, patch), dim=1) + self.pos_embed


class TinyFullBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_finetune = True
        self.model = TinyDINO()
        self.num_channels = 4
        self.norm_layer = True
        self.num_trainable_blocks = 3

    # Exercise the production forward, including its gradient/no_grad boundary.
    forward = DINOBackbone.forward


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

    def test_excludes_frozen_parameters_and_rejects_unknown_non_full_structures(self):
        night = TinyFullBackbone()
        night.model.cls_token.requires_grad_(False)
        groups = full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        self.assertNotIn("model.cls_token", named_group_values(groups, "lr"))
        self.assertFalse(night.model.cls_token.requires_grad)
        night.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        for mutation in (lambda net: setattr(net, "full_finetune", False),
                         lambda net: setattr(net.model, "chunked_blocks", True),
                         lambda net: setattr(net.model, "blocks", nn.ModuleList()),
                         lambda net: setattr(net.model, "norm", None)):
            night = TinyFullBackbone()
            mutation(night)
            with self.assertRaisesRegex(ValueError, "full DINOBackbone"):
                full_backbone_parameter_groups(night, learning_rate=1e-3, weight_decay=0.1)
        night = TinyFullBackbone()
        night.other_model = nn.Linear(4, 4).requires_grad_(False)
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


class FullBackboneScopeTests(unittest.TestCase):
    def test_early_and_late_gradients_survive_the_production_forward(self):
        for scope, selected_indices in (("early", [0, 1]), ("late", [1, 2])):
            for checkpointing in (False, True):
                with self.subTest(scope=scope, checkpointing=checkpointing):
                    torch.manual_seed(99)
                    night = TinyFullBackbone().train()
                    state_before = copy.deepcopy(night.state_dict())
                    metadata = configure_full_backbone_scope(night, scope, block_count=2)
                    night.gradient_checkpointing = checkpointing
                    self.assertEqual(metadata["scope"], scope)
                    self.assertEqual(metadata["block_indices"], selected_indices)
                    self.assertEqual(metadata["block_count"], 2)
                    self.assertEqual(metadata["total_blocks"], 3)
                    self.assertEqual(metadata["trainable_parameters"],
                                     sum(p.numel() for p in night.parameters() if p.requires_grad))
                    self.assertEqual(metadata["total_parameters"],
                                     sum(p.numel() for p in night.parameters()))
                    self.assertTrue(night.full_finetune)
                    self.assertEqual(night.num_trainable_blocks, 3)

                    # If an early scope used a frozen-prefix no_grad boundary,
                    # the frozen later block or norm would sever this gradient.
                    patches, cls = night(torch.randn(2, 3, 28, 28))
                    self.assertTrue(patches.requires_grad)
                    self.assertTrue(cls.requires_grad)
                    loss = (patches * torch.randn_like(patches)).sum()
                    loss = loss + (cls * torch.randn_like(cls)).sum()
                    loss.backward()
                    for name, parameter in night.named_parameters():
                        selected = any(name.startswith(f"model.blocks.{i}.")
                                       for i in selected_indices)
                        self.assertEqual(parameter.requires_grad, selected, name)
                        if selected:
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                        else:
                            self.assertIsNone(parameter.grad, name)

                    groups = full_backbone_parameter_groups(
                        night, learning_rate=1e-3, weight_decay=0.1, layer_decay=0.5)
                    parameters = [p for group in groups for p in group["params"]]
                    self.assertEqual(len(parameters), len({id(p) for p in parameters}))
                    self.assertEqual({id(p) for p in parameters},
                                     {id(p) for p in night.parameters() if p.requires_grad})
                    rates = named_group_values(groups, "lr")
                    for i in selected_indices:
                        self.assertAlmostEqual(rates[f"model.blocks.{i}.0.weight"],
                                               1e-3 * 0.5 ** (2 - i))
                    optimizer = torch.optim.AdamW(groups)
                    optimizer.step()
                    for name, parameter in night.named_parameters():
                        if parameter.requires_grad:
                            self.assertFalse(torch.equal(parameter, state_before[name]), name)
                        else:
                            torch.testing.assert_close(parameter, state_before[name], rtol=0, atol=0)
                    self.assertEqual(night.state_dict().keys(), state_before.keys())

    def test_late_scope_frozen_prefix_builds_no_graph(self):
        night = TinyFullBackbone()
        configure_full_backbone_scope(night, "late", block_count=1)
        outputs_require_grad = []
        handle = night.model.blocks[1].register_forward_hook(
            lambda module, inputs, output: outputs_require_grad.append(output.requires_grad))
        patches, _ = night(torch.randn(1, 3, 28, 28))
        handle.remove()
        self.assertEqual(outputs_require_grad, [False])
        self.assertTrue(patches.requires_grad)

    def test_all_restores_every_parameter_and_does_not_change_outputs(self):
        torch.manual_seed(100)
        night = TinyFullBackbone()
        images = torch.randn(1, 3, 28, 28)
        expected = night(images)
        configure_full_backbone_scope(night, "early", block_count=1)
        actual = night(images)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        metadata = configure_full_backbone_scope(night, "all")
        self.assertEqual(metadata["block_indices"], [0, 1, 2])
        self.assertEqual(metadata["trainable_parameters"], metadata["total_parameters"])
        self.assertTrue(all(p.requires_grad for p in night.parameters()))
        self.assertEqual(night.num_trainable_blocks, 3)

    def test_reconfiguration_clears_stale_gradients_and_invalid_input_is_atomic(self):
        night = TinyFullBackbone()
        for parameter in night.parameters():
            parameter.grad = torch.ones_like(parameter)
        configure_full_backbone_scope(night, "late", block_count=1)
        for parameter in night.parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
        trainability = {name: p.requires_grad for name, p in night.named_parameters()}
        for scope, count in (("other", 1), (None, 1), ([], 1), ("early", 0),
                             ("late", 4), ("early", True), ("all", 1.2)):
            with self.subTest(scope=scope, count=count), self.assertRaises(ValueError):
                configure_full_backbone_scope(night, scope, count)
            self.assertEqual({name: p.requires_grad for name, p in night.named_parameters()},
                             trainability)
        malformed = TinyFullBackbone()
        malformed.model.blocks[2][0].weight = malformed.model.blocks[0][0].weight
        with self.assertRaisesRegex(ValueError, "conflicting layer"):
            configure_full_backbone_scope(malformed, "early", 1)


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

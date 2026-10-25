"""Partial Token style statistics and call-local model insertion contracts."""
from __future__ import annotations

import copy
import json
import math
import random
import unittest

import numpy as np
import torch
from torch import nn

from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.training.token_style import TokenStyleAugment
from weather_vpr.utils.backend import enable_backend_imports

enable_backend_imports()
from workflow.model import DINOBackbone  # noqa: E402


def patterned_tokens(dtype=torch.float32):
    # Distinct image means/stds and distinct channel scales expose wrong axes.
    patches = torch.arange(1, 9, dtype=torch.float32)[None, :, None]
    channels = torch.tensor([1.0, 3.0, 7.0])[None, None, :]
    means = torch.tensor([0.0, 20.0, -30.0, 4.0])[:, None, None]
    scales = torch.tensor([1.0, 2.0, 0.5, 4.0])[:, None, None]
    patches = patches * channels * scales + means
    cls = torch.tensor([100.0, -500.0, 2000.0])[None, None, :].expand(4, -1, -1)
    return torch.cat((cls, patches), dim=1).to(dtype)


class TokenStyleTests(unittest.TestCase):
    def test_patch_axes_detached_statistics_partial_masks_and_cls(self):
        tokens = patterned_tokens()
        original = tokens.clone()
        aug = TokenStyleAugment(143, probability=1, fraction=0.375, strength=0.8)
        result = aug.transform(tokens, epoch=0, step=7)
        record = aug.last_record
        self.assertEqual(record["applied_images"], 4)
        self.assertEqual(record["applied_patches"], 12)
        self.assertEqual(sorted(record["donor_indices"]), list(range(4)))
        self.assertTrue(all(i != j for i, j in enumerate(record["donor_indices"])))
        self.assertEqual(result.shape, tokens.shape)
        self.assertTrue(torch.equal(tokens, original))
        self.assertTrue(torch.equal(result[:, 0], tokens[:, 0]))
        mean = tokens[:, 1:].mean(dim=1, keepdim=True)
        std = (tokens[:, 1:].var(dim=1, correction=0, keepdim=True) + aug.eps).sqrt()
        for image, patch_ids in zip(record["selected_images"], record["patch_indices"]):
            self.assertEqual(len(set(patch_ids)), math.floor(8 * 0.375))
            donor = record["donor_indices"][image]
            mix = record["lambdas"][image]
            mixed_mean = mean[image] + mix * (mean[donor] - mean[image])
            mixed_std = std[image] + mix * (std[donor] - std[image])
            expected = ((tokens[image, 1:][patch_ids] - mean[image]) / std[image]) * mixed_std + mixed_mean
            torch.testing.assert_close(result[image, 1:][patch_ids], expected, rtol=0, atol=0)
            untouched = [index for index in range(8) if index not in patch_ids]
            self.assertTrue(torch.equal(result[image, 1:][untouched], tokens[image, 1:][untouched]))
        json.dumps(record, allow_nan=False)
        json.dumps(aug.configuration(), allow_nan=False)
        delta = (result[:, 1:] - tokens[:, 1:]).square().mean().sqrt()
        rms = tokens[:, 1:].square().mean().sqrt()
        self.assertAlmostEqual(record["relative_rms_delta"], float(delta / rms), places=6)

    def test_randomness_is_deterministic_local_and_call_order_independent(self):
        aug = TokenStyleAugment(53, probability=0.5, fraction=0.5)
        x = patterned_tokens()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        first = aug.transform(x, epoch=1, step=4)
        first_record = copy.deepcopy(aug.last_record)
        self.assertEqual(random.getstate(), python_before)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_before[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_before[1])
        self.assertEqual(after_numpy[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        aug.transform(x, epoch=88, step=999)
        again = aug.transform(x, epoch=1, step=4)
        self.assertTrue(torch.equal(first, again))
        self.assertEqual(first_record, aug.last_record)
        self.assertTrue(torch.equal(first, TokenStyleAugment(53, fraction=0.5).transform(x, epoch=1, step=4)))

    def test_each_sample_bernoulli_and_exact_mask_count(self):
        aug = TokenStyleAugment(2026, probability=0.5, fraction=0.3)
        counts = []
        per_image = [0] * 4
        for step in range(40):
            x = patterned_tokens()
            y = aug.transform(x, epoch=0, step=step)
            record = aug.last_record
            counts.append(record["applied_images"])
            selected = record.get("selected_images", [])
            for image in range(4):
                per_image[image] += image in selected
                if image not in selected:
                    self.assertTrue(torch.equal(x[image], y[image]))
            if selected:
                self.assertTrue(all(len(ids) == 2 for ids in record["patch_indices"]))
        # A batch-wide Bernoulli would only produce 0 and 4.
        self.assertTrue(any(0 < count < 4 for count in counts))
        self.assertTrue(all(0 < count < 40 for count in per_image))

    def test_disabled_and_batch_one_are_tensor_identity_before_rng_or_finite_checks(self):
        invalid = torch.tensor(float("nan"))
        for changes in ({"probability": 0}, {"fraction": 0}, {"strength": 0}):
            aug = TokenStyleAugment(9, **changes)
            self.assertIs(aug.transform(invalid, epoch=-10, step=None), invalid)
            self.assertEqual(aug.last_record["skip_reason"], "disabled")
        x = torch.full((1, 9, 3), float("nan"))
        aug = TokenStyleAugment(9)
        self.assertIs(aug.transform(x, epoch=0, step=0), x)
        self.assertEqual(aug.last_record["skip_reason"], "batch_smaller_than_two")
        aug = TokenStyleAugment(9, fraction=0.01)
        x = patterned_tokens()
        self.assertIs(aug.transform(x, epoch=0, step=0), x)
        self.assertEqual(aug.last_record["skip_reason"], "no_selected_patches")

    def test_gradient_uses_content_only_and_never_donor_statistics(self):
        x = patterned_tokens().requires_grad_()
        aug = TokenStyleAugment(12, probability=1, fraction=0.5, strength=0.7)
        y = aug.transform(x, epoch=0, step=0)
        record = aug.last_record
        image = record["selected_images"][0]
        ids = record["patch_indices"][0]
        donor = record["donor_indices"][image]
        y[image, torch.tensor(ids) + 1].sum().backward()
        self.assertEqual(float(x.grad[donor].abs().sum()), 0.0)
        self.assertEqual(float(x.grad[image, 0].abs().sum()), 0.0)
        std = (x.detach()[:, 1:].var(dim=1, correction=0) + aug.eps).sqrt()
        mix = record["lambdas"][image]
        gain = (std[image] + mix * (std[donor] - std[image])) / std[image]
        torch.testing.assert_close(x.grad[image, torch.tensor(ids) + 1],
                                   gain.expand(len(ids), -1), rtol=1e-6, atol=1e-6)
        untouched = [index + 1 for index in range(8) if index not in ids]
        self.assertEqual(float(x.grad[image, untouched].abs().sum()), 0.0)

    def test_bfloat16_extreme_std_finite_and_untouched_bits(self):
        x = patterned_tokens()
        x[0, 1:] = 3.0  # epsilon protects an exactly constant recipient.
        x[1, 1:] *= 1000
        x[2, 1:] *= 1e-6
        x = x.to(torch.bfloat16).requires_grad_()
        aug = TokenStyleAugment(4, probability=1, fraction=0.5, strength=1)
        y = aug.transform(x, epoch=0, step=0)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertTrue(torch.equal(x[:, 0], y[:, 0]))
        for image, ids in zip(aug.last_record["selected_images"], aug.last_record["patch_indices"]):
            untouched = [index + 1 for index in range(8) if index not in ids]
            self.assertTrue(torch.equal(x[image, untouched], y[image, untouched]))
        y.float().square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(aug.last_record["raw_gain_max"], 1)
        json.dumps(aug.last_record, allow_nan=False)

    def test_active_validation_and_configuration_validation(self):
        for changes in ({"seed": True}, {"probability": -1}, {"fraction": 1.1},
                        {"strength": float("nan")}, {"beta": 0}, {"eps": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TokenStyleAugment(**{"seed": 1, **changes})
        aug = TokenStyleAugment(1, probability=1)
        for invalid in (torch.ones(2, 3), torch.ones(2, 1, 3),
                        torch.ones(2, 8, 3, dtype=torch.int64),
                        torch.full((2, 9, 3), float("inf"))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                aug.transform(invalid, epoch=0, step=0)
        with self.assertRaises(ValueError):
            aug.transform(patterned_tokens(), epoch=True, step=0)


class RecordingBlock(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = offset
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x * self.scale + self.offset


class RecordingDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([RecordingBlock(i + 1) for i in range(3)])
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, images):
        return images.mean((2, 3))[:, None, :].expand(-1, 5, -1)


def recording_backbone(*, full=True, trainable=3):
    backbone = DINOBackbone.__new__(DINOBackbone)
    nn.Module.__init__(backbone)
    backbone.model = RecordingDINO()
    backbone.num_channels = 3
    backbone.num_trainable_blocks = trainable
    backbone.norm_layer = True
    backbone.full_finetune = full
    return backbone


class RecordingSALAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = recording_backbone(full=False, trainable=0)
        self.config = {"agg_config": {"num_channels": 3}}
        self.aggregator = RecordingAggregator()

    def forward(self, x, return_aux=True):
        return self.aggregator(self.backbone(x), return_aux=return_aux)


class RecordingAggregator(nn.Module):
    def forward(self, features, return_aux=True):
        patch, cls = features
        desc = patch.mean((2, 3)) + cls
        return (desc, {}) if return_aux else desc


class TokenStyleInsertionTests(unittest.TestCase):
    def test_absolute_after_block_in_prefix_suffix_and_before_norm(self):
        images = torch.ones(2, 3, 28, 28)
        for full, trainable in ((True, 3), (False, 1)):
            for index in range(3):
                with self.subTest(full=full, index=index):
                    model = recording_backbone(full=full, trainable=trainable)
                    model.model.norm = RecordingBlock(100)
                    seen = []
                    def transform(x):
                        seen.append(x.detach().clone())
                        return x * 10
                    patches, cls = model(images, token_transform=transform,
                                         token_transform_after_block=index)
                    before = 1 + sum(range(1, index + 2))
                    self.assertEqual(len(seen), 1)
                    self.assertTrue(torch.equal(seen[0], torch.full_like(seen[0], before)))
                    expected = before * 10 + sum(range(index + 2, 4)) + 100
                    self.assertTrue(torch.equal(cls, torch.full_like(cls, expected)))
                    self.assertTrue(torch.equal(patches, torch.full_like(patches, expected)))
                    if index < 2:
                        self.assertTrue(torch.equal(model.model.blocks[index + 1].seen,
                                                    torch.full_like(seen[0], before * 10)))

    def test_transform_is_differentiable_and_keeps_frozen_state(self):
        model = recording_backbone(full=True)
        model.model.blocks[0].requires_grad_(False)
        before = copy.deepcopy(model.state_dict())
        images = torch.ones(2, 3, 28, 28, requires_grad=True)
        patches, cls = model(images, token_transform=lambda x: x * 2,
                             token_transform_after_block=0)
        (patches.sum() + cls.sum()).backward()
        self.assertGreater(float(images.grad.abs().sum()), 0)
        self.assertIsNone(model.model.blocks[0].scale.grad)
        self.assertIsNotNone(model.model.blocks[1].scale.grad)
        self.assertIsNotNone(model.model.blocks[2].scale.grad)
        self.assertEqual(list(before), list(model.state_dict()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_checkpoint_recomputation_does_not_repeat_callback_or_diagnostics(self):
        model = recording_backbone(full=True).train()
        model.gradient_checkpointing = True
        x = torch.ones(2, 3, 28, 28, requires_grad=True)
        calls = []
        def transform(tokens):
            calls.append(True)
            return tokens * 2
        patches, cls = model(x, token_transform=transform, token_transform_after_block=1)
        (patches.sum() + cls.sum()).backward()
        self.assertEqual(len(calls), 1)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_default_exact_path_and_validation_only_when_enabled(self):
        model = recording_backbone()
        x = torch.ones(2, 3, 28, 28)
        expected = model(x)
        actual = model(x, token_transform=None, token_transform_after_block=-999)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        for index in (-1, 3, True, 0.0):
            with self.assertRaises(ValueError):
                model(x, token_transform=lambda t: t, token_transform_after_block=index)
        with self.assertRaises(ValueError):
            model(x, token_transform=7, token_transform_after_block=0)
        model.eval()
        with self.assertRaisesRegex(ValueError, "only during training"):
            model(x, token_transform=lambda t: t, token_transform_after_block=0)

    def test_student_callback_is_local_and_teacher_and_state_are_untouched(self):
        net = WeatherAdapterSALAD(RecordingSALAD(), ("night", "rain"), adapter_type="full").train()
        x = torch.ones(2, 3, 28, 28)
        teacher = net.teacher(x, return_aux=False)
        before = copy.deepcopy(net.state_dict())
        calls = []
        def transform(tokens):
            calls.append(True)
            return tokens * 2
        transformed = net.student(x, "night", return_aux=False,
                                  token_transform=transform, token_transform_after_block=1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(torch.equal(teacher, transformed))
        self.assertTrue(torch.equal(teacher, net.teacher(x, return_aux=False)))
        self.assertTrue(torch.equal(teacher, net.student(x, "night", return_aux=False)))
        self.assertTrue(torch.equal(teacher, net.student(x, "rain", return_aux=False)))
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "Night student"):
            net.student(x, "rain", token_transform=transform)
        self.assertEqual(list(before), list(net.state_dict()))
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        output_adapter = WeatherAdapterSALAD(RecordingSALAD(), ("night",),
                                             bottleneck=2, adapter_type="output")
        with self.assertRaisesRegex(ValueError, "full weather backbone"):
            output_adapter.student(x, "night", token_transform=transform)


if __name__ == "__main__":
    unittest.main()

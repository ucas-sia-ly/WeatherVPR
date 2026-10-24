"""Differentiable CLS attention extraction without a second DINOv2 forward."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from weather_vpr.models.attention import capture_cls_attention

# Use the repository's actual attention/block implementations, without hub
# downloads or construction of a full ViT. No changes to these modules needed.
_DINO_SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "dinov2"
if str(_DINO_SOURCE) not in sys.path:
    sys.path.insert(0, str(_DINO_SOURCE))
from dinov2.layers.attention import Attention, MemEffAttention  # noqa: E402
from dinov2.layers.block import Block  # noqa: E402


class TinyDINO(nn.Module):
    def __init__(self, attention_class=Attention, *, registers=0):
        super().__init__()
        self.num_register_tokens = registers
        self.chunked_blocks = False
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.register_tokens = nn.Parameter(torch.randn(1, registers, 8))
        self.blocks = nn.ModuleList([
            Block(8, 2, mlp_ratio=2, qkv_bias=True, attn_class=attention_class)
            for _ in range(8)
        ])
        self.norm = nn.LayerNorm(8)
        self.checkpointing = False

    def forward(self, images):
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        batch = len(images)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1),
                            self.register_tokens.expand(batch, -1, -1), patches), dim=1)
        for block in self.blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False) if self.checkpointing else block(tokens)
        return self.norm(tokens)


def explicit_cls_attention(qkv, heads, prefix):
    batch, count, tripled_width = qkv.shape
    width = tripled_width // 3
    full = qkv.float().reshape(batch, count, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
    query, key, _ = full.unbind(0)
    matrix = (query @ key.transpose(-2, -1)) * ((width // heads) ** -0.5)
    return matrix.softmax(-1)[:, :, 0, prefix:].mean(1)


class CLSAttentionCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(89)
        self.images = torch.randn(2, 3, 4, 4)
        # xFormers kernels require CUDA. Exercise MemEffAttention's production
        # CPU fallback on this tiny, device-independent test fixture.
        fallback = mock.patch("dinov2.layers.attention.XFORMERS_AVAILABLE", False)
        fallback.start()
        self.addCleanup(fallback.stop)

    def test_matches_full_attention_for_standard_memory_efficient_and_register_tokens(self):
        for attention_class in (Attention, MemEffAttention):
            for registers in (0, 2):
                with self.subTest(attention=attention_class.__name__, registers=registers):
                    model = TinyDINO(attention_class, registers=registers).eval()
                    baseline = model(self.images)
                    with capture_cls_attention(model) as capture:
                        actual = model(self.images)
                        expected = explicit_cls_attention(capture._qkv, 2, 1 + registers)
                    attention = capture.attention
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                    torch.testing.assert_close(attention, expected)
                    self.assertEqual(tuple(attention.shape), (2, 4))
                    self.assertEqual(attention.dtype, torch.float32)
                    self.assertIs(capture.attention, attention)
                    self.assertTrue((attention.sum(-1) < 1).all())

    def test_teacher_has_no_graph_student_has_gradients_only_in_last_four_blocks(self):
        teacher = TinyDINO().eval().requires_grad_(False)
        student = copy.deepcopy(teacher)
        for block in student.blocks[-4:]:
            block.requires_grad_(True)
        teacher_before = copy.deepcopy(teacher.state_dict())
        with torch.no_grad(), capture_cls_attention(teacher) as target_capture:
            teacher(self.images)
        target = target_capture.attention
        self.assertFalse(target.requires_grad)
        with capture_cls_attention(student) as student_capture:
            student(self.images + torch.randn_like(self.images) * 0.2)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ((student_capture.attention - target.detach()) * weights).sum().backward()
        for name, parameter in student.named_parameters():
            if any(name.startswith(f"blocks.{index}.") for index in range(4, 8)):
                # The attention auxiliary depends on Q/K and previous blocks;
                # final V/proj/MLP are downstream and need the descriptor loss.
                if ".attn.qkv.weight" in name:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)

    def test_nonreentrant_checkpoint_backward_matches_without_checkpoint_after_hook_removal(self):
        reference = TinyDINO().train().requires_grad_(False)
        for block in reference.blocks[-4:]:
            block.requires_grad_(True)
        checkpointed = copy.deepcopy(reference)
        checkpointed.checkpointing = True
        for model in (reference, checkpointed):
            with capture_cls_attention(model) as capture:
                features = model(self.images)
            self.assertFalse(model.blocks[-1].attn.qkv._forward_hooks)
            # Attention is evaluated after hooks are removed, outside the block
            # checkpoint. Include an ordinary retrieval-feature objective too.
            loss = (capture.attention * torch.arange(1, 5)).sum() + features.square().mean()
            loss.backward()
        for (name, left), (_, right) in zip(reference.named_parameters(), checkpointed.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(right.grad, left.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_autocast_attention_stays_fp32_and_only_materializes_one_query_row(self):
        model = TinyDINO().eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with capture_cls_attention(model) as capture:
                model(self.images)
            original_matmul = torch.matmul
            with mock.patch("torch.matmul", wraps=original_matmul) as multiply:
                attention = capture.attention
        self.assertEqual(attention.dtype, torch.float32)
        self.assertEqual(multiply.call_count, 1)
        query, key = multiply.call_args.args
        self.assertEqual(tuple(query.shape), (2, 2, 1, 4))
        self.assertEqual(tuple(key.shape), (2, 2, 4, 5))
        self.assertEqual(query.dtype, torch.float32)
        self.assertEqual(key.dtype, torch.float32)

    def test_frozen_parameters_do_not_block_input_gradients(self):
        model = TinyDINO().eval().requires_grad_(False)
        images = self.images.detach().requires_grad_(True)
        with capture_cls_attention(model) as capture:
            model(images)
        (capture.attention * torch.arange(1, 5)).sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_hooks_removed_on_failure_and_unexecuted_context_rejected(self):
        model = TinyDINO()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with capture_cls_attention(model):
                raise RuntimeError("interrupted")
        self.assertFalse(model.blocks[-1].attn._forward_pre_hooks)
        self.assertFalse(model.blocks[-1].attn.qkv._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "not executed"):
            with capture_cls_attention(model):
                pass
        self.assertFalse(model.blocks[-1].attn.qkv._forward_hooks)

    def test_read_inside_context_supported_but_multiple_forwards_and_context_reuse_rejected(self):
        model = TinyDINO()
        context = capture_cls_attention(model)
        with context as capture:
            model(self.images)
            first = capture.attention
        self.assertIs(first, capture.attention)
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            with context:
                pass
        with self.assertRaisesRegex(RuntimeError, "per backbone forward"):
            with capture_cls_attention(model) as capture:
                model(self.images)
                _ = capture.attention
                model(self.images)
        self.assertFalse(model.blocks[-1].attn.qkv._forward_hooks)

    def test_unsupported_attention_bias_causal_dropout_and_stochastic_depth_fail_closed(self):
        for kind in ("unknown", "chunked", "scale", "dropout", "stochastic"):
            model = TinyDINO()
            if kind == "unknown":
                model.blocks[-1].attn = nn.Identity()
            elif kind == "chunked":
                model.chunked_blocks = True
            elif kind == "scale":
                model.blocks[-1].attn.scale = 1.0
            elif kind == "dropout":
                model.blocks[-1].attn.attn_drop = 0.1
            else:
                model.blocks[-1].sample_drop_ratio = 0.05
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                with capture_cls_attention(model):
                    model(self.images)
            if kind not in {"unknown", "chunked", "scale"}:
                self.assertFalse(model.blocks[-1].attn.qkv._forward_hooks)
        for attention_class, option in ((Attention, True), (MemEffAttention, object())):
            model = TinyDINO(attention_class)
            with self.assertRaisesRegex(ValueError, "bias or causal"):
                with capture_cls_attention(model):
                    model.blocks[-1].attn(torch.randn(2, 5, 8), option)


if __name__ == "__main__":
    unittest.main()

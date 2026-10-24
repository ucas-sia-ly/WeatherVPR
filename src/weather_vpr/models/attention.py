"""Read the last DINOv2 CLS attention row without changing model parameters.

The ordinary (possibly memory efficient) attention forward is left untouched.
Only the final QKV projection is retained; the auxiliary CLS row is calculated
outside the block forward, which also makes hook removal safe with nonreentrant
gradient checkpointing. No patch-by-patch attention matrix is constructed.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CLSAttentionCapture:
    """One-forward context exposing FP32, head-averaged CLS-to-patch attention.

    ``attention`` has shape ``[batch, patch_count]`` and contains the original
    attention probabilities: softmax includes CLS/register tokens, then those
    nonpatch keys are removed. Thus patch probabilities need not sum to one.
    Normalize on trusted correspondences in the distillation loss, if needed.

    Example::

        with capture_cls_attention(night_backbone) as capture:
            output = night_backbone(images)
        attention = capture.attention
        loss = criterion(attention, frozen_teacher_attention)
        loss.backward()

    Teacher forwards must still be performed under ``torch.no_grad()``. Student
    captures preserve gradients through Q/K and preceding trainable blocks.
    Do not use reentrant checkpointing (its first forward disables gradients).
    """

    def __init__(self, backbone: nn.Module):
        model = getattr(backbone, "model", backbone)
        blocks = getattr(model, "blocks", None)
        if (getattr(model, "chunked_blocks", False) or blocks is None
                or len(blocks) == 0):
            raise ValueError("CLS attention requires unchunked DINOv2 blocks")
        self._block = blocks[-1]
        attention = getattr(self._block, "attn", None)
        # The extractor assumes exactly the QKV layout and attention definition
        # of the vendored DINOv2 implementation. Unknown variants fail closed.
        if (attention is None
                or type(attention).__module__ != "dinov2.layers.attention"
                or type(attention).__name__ not in {"Attention", "MemEffAttention"}):
            raise ValueError("Unsupported DINOv2 attention implementation")
        heads = getattr(attention, "num_heads", None)
        width = getattr(attention, "dim", None)
        if (type(heads) is not int or heads < 1 or type(width) is not int
                or width < 1 or width % heads):
            raise ValueError("Invalid DINOv2 attention heads or width")
        scale = (width // heads) ** -0.5
        if (not isinstance(getattr(attention, "scale", None), (float, int))
                or not math.isclose(float(attention.scale), scale, rel_tol=1e-12)):
            raise ValueError("Unsupported DINOv2 attention scaling")
        if not isinstance(getattr(attention, "qkv", None), nn.Module):
            raise ValueError("DINOv2 attention has no QKV projection")
        register_count = getattr(model, "num_register_tokens", 0)
        if type(register_count) is not int or register_count < 0:
            raise ValueError("Invalid DINOv2 register token count")
        self._attention = attention
        self._heads = heads
        self._width = width
        self._scale = scale
        self._prefix_count = 1 + register_count
        self._qkv: torch.Tensor | None = None
        self._projection_seen = False
        self._captured_attention: torch.Tensor | None = None
        self._expected_shape: tuple[int, int] | None = None
        self._handles = []
        self._entered = False

    def _check_forward(self, module, args, kwargs):
        allowed = ({"x", "is_causal"} if type(module).__name__ == "Attention"
                   else {"x", "attn_bias"})
        if len(args) > 2 or set(kwargs) - allowed:
            raise ValueError("Unsupported DINOv2 attention forward arguments")
        images = args[0] if args else kwargs.get("x")
        option_name = "is_causal" if type(module).__name__ == "Attention" else "attn_bias"
        option = args[1] if len(args) == 2 else kwargs.get(option_name)
        if option is not None and option is not False:
            raise ValueError("CLS attention capture does not support bias or causal masks")
        if (not isinstance(images, torch.Tensor) or images.ndim != 3
                or images.shape[-1] != self._width
                or images.shape[1] <= self._prefix_count):
            raise ValueError("CLS attention requires a dense batch of CLS/patch tokens")
        dropout = module.attn_drop
        dropout = dropout.p if isinstance(dropout, nn.Dropout) else dropout
        if module.training and dropout != 0:
            raise ValueError("CLS attention capture requires zero attention dropout")
        if self._block.training and getattr(self._block, "sample_drop_ratio", 0) != 0:
            raise ValueError("CLS attention capture requires zero stochastic depth")
        self._expected_shape = tuple(images.shape[:2])

    def _capture_projection(self, module, args, output):
        if self._projection_seen:
            raise RuntimeError("Use one CLS attention capture context per backbone forward")
        if (not isinstance(output, torch.Tensor) or output.ndim != 3
                or tuple(output.shape[:2]) != self._expected_shape
                or output.shape[-1] != 3 * self._width
                or not output.is_floating_point()):
            raise ValueError("Unsupported DINOv2 QKV output layout")
        # Do not transform tensors here. Operations added inside a checkpointed
        # block would also have to run during its backward recomputation.
        self._projection_seen = True
        self._qkv = output

    def __enter__(self):
        if self._entered:
            raise RuntimeError("CLS attention capture contexts cannot be reused")
        self._entered = True
        try:
            self._handles.append(self._attention.register_forward_pre_hook(
                self._check_forward, with_kwargs=True))
            self._handles.append(self._attention.qkv.register_forward_hook(
                self._capture_projection))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()
        if exc_type is None and not self._projection_seen:
            raise RuntimeError("The final DINOv2 attention block was not executed")
        return False

    @property
    def attention(self) -> torch.Tensor:
        if self._captured_attention is not None:
            return self._captured_attention
        if self._qkv is None:
            raise RuntimeError("Read CLS attention after the backbone forward")
        if self._captured_attention is None:
            qkv = self._qkv
            batch, tokens, _ = qkv.shape
            qkv = qkv.reshape(batch, tokens, 3, self._heads,
                              self._width // self._heads)
            # Explicitly disable outer AMP: both dot products and probabilities
            # must be FP32 even when the backbone projection used BF16/FP16.
            with torch.autocast(device_type=qkv.device.type, enabled=False):
                cls_query = qkv[:, 0, 0].float().unsqueeze(-2)
                keys = qkv[:, :, 1].transpose(1, 2).float()
                logits = torch.matmul(cls_query, keys.transpose(-2, -1)) * self._scale
                probabilities = logits.softmax(dim=-1).squeeze(-2)
                self._captured_attention = probabilities[:, :, self._prefix_count:].mean(dim=1)
            self._qkv = None  # the cached attention graph owns required tensors
        return self._captured_attention


def capture_cls_attention(backbone: nn.Module) -> CLSAttentionCapture:
    """Return a one-forward, parameter-free last-block CLS attention capture."""
    return CLSAttentionCapture(backbone)

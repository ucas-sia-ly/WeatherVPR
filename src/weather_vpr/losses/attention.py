"""DINO CLS-to-patch attention alignment on externally verified matches.

This aligns a distribution over trusted Night patches, rather than treating
attention or SALAD keep mass as correspondence or semantic labels. No identity
correspondence is inferred. Many-to-one matches retain repeated Teacher mass
on the matched Night support; callers needing unique matches must validate
that property when building their correspondence cache.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch


def _attention_values(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or min(value.shape) < 1):
        raise ValueError(f"{name} must be nonempty [B,N]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")
    if (value < 0).any() or (value > 1 + 1e-5).any():
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return value


def trusted_attention_alignment(student_attention, teacher_attention,
                                teacher_keep, match_index, confidence, *,
                                importance_mode="saliency", eps=1e-8):
    """Return FP32 KL(Teacher || Student) and detached coverage statistics.

    For Night patch ``i``, both attention maps receive the same fixed weight
    ``confidence[i] * teacher_keep[match_index[i]]`` (or confidence alone in
    uniform mode). Each map is normalized on this restricted support. Only
    images with at least two positive-weight matches and nonzero weighted
    Teacher attention contribute to the batch mean. Invalid match indices
    receive zero weight even if their input confidence was positive.

    Teacher tensors, match indices, and confidence are detached. Therefore a
    Student cannot reduce its loss by changing its own saliency mask. Small
    additive smoothing on Student attention ensures finite KL and gradients
    even when its probability underflows to zero on a trusted patch.
    """
    if importance_mode not in ("saliency", "uniform"):
        raise ValueError("importance_mode must be saliency or uniform")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(student_attention, torch.Tensor):
        raise TypeError("student_attention must be a Tensor")
    precision = (torch.autocast(device_type=student_attention.device.type, enabled=False)
                 if student_attention.device.type in ("cpu", "cuda") else nullcontext())
    with precision:
        student = _attention_values(student_attention, "student attention")
        if not isinstance(teacher_attention, torch.Tensor):
            raise TypeError("teacher_attention must be a Tensor")
        if not isinstance(teacher_keep, torch.Tensor):
            raise TypeError("teacher_keep must be a Tensor")
        teacher = _attention_values(teacher_attention.detach(), "teacher attention")
        keep = _attention_values(teacher_keep.detach(), "teacher keep")
        if (student.shape[0] != teacher.shape[0] or keep.shape != teacher.shape
                or student.device != teacher.device or keep.device != teacher.device):
            raise ValueError("Teacher/Student batch, keep grid, or device differs")
        if (not isinstance(match_index, torch.Tensor)
                or match_index.dtype != torch.long):
            raise TypeError("match_index must be int64 [B,N]")
        if match_index.shape != student.shape:
            raise ValueError("match_index must match Student [B,N]")
        if not isinstance(confidence, torch.Tensor) or not confidence.is_floating_point():
            raise TypeError("confidence must be floating point [B,N]")
        if confidence.shape != student.shape:
            raise ValueError("confidence must match Student [B,N]")
        if match_index.device != student.device or confidence.device != student.device:
            raise ValueError("Correspondences must share the attention device")
        q = confidence.detach().float()
        if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
            raise ValueError("confidence must be finite in [0,1]")
        index = match_index.detach()
        in_range = (index >= 0) & (index < teacher.shape[1])
        safe_index = index.clamp(0, teacher.shape[1] - 1)
        target = teacher.gather(1, safe_index)
        weights = q * in_range
        if importance_mode == "saliency":
            weights = weights * keep.gather(1, safe_index).clamp(0, 1)
        support = weights > 0
        target_weighted = target * weights
        teacher_mass = target_weighted.sum(1)
        student_mass = (student * weights).sum(1)
        valid = (support.sum(1) >= 2) & (teacher_mass > eps)

        target_probability = target_weighted / teacher_mass.clamp_min(eps)[:, None]
        student_weighted = (student + eps) * weights
        student_probability = (student_weighted
                               / student_weighted.sum(1).clamp_min(
                                   torch.finfo(torch.float32).tiny)[:, None])
        # x log x is defined as zero at x=0. Finite logs also keep inactive
        # images safe before their differentiable zero contribution is taken.
        terms = target_probability * (
            target_probability.clamp_min(torch.finfo(torch.float32).tiny).log()
            - student_probability.clamp_min(torch.finfo(torch.float32).tiny).log())
        per_query = terms.sum(1).clamp_min(0)
        loss = (per_query * valid).sum() / valid.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite FP32 attention KL")
        return {
            "loss": loss,
            "attention_loss": loss.detach(),
            "attention_valid_queries": valid.sum().detach(),
            "attention_trusted_patches": support.sum().detach(),
            "attention_coverage": support.float().mean().detach(),
            "attention_weight_mass": weights.sum().detach(),
            "attention_teacher_mass": (teacher_mass * valid).sum().detach(),
            "attention_student_mass": (student_mass * valid).sum().detach(),
        }

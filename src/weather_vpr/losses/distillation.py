"""Independent SGPD and global losses for a frozen DINOv2-SALAD teacher.

Local confidence must come from Stage 2B A-class trusted correspondences.
SALAD keep mass is an importance weight, not a correspondence label.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F


def _fp32(tensor):
    if tensor.device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _tokens(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4) or min(value.shape) < 1:
        raise ValueError(f"{name} must be [B,C,N] or [B,C,H,W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    result = value.flatten(2).float()
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite {name}")
    return result


def _patch_values(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3) or min(value.shape) < 1:
        raise ValueError(f"{name} must be [B,N] or [B,H,W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    result = value.flatten(1).float()
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite {name}")
    return result


def _match(index, confidence, batch, student_tokens, teacher_tokens, device):
    if not isinstance(index, torch.Tensor) or index.dtype != torch.long or index.ndim not in (2, 3):
        raise TypeError("match_index must be int64 [B,N] or [B,H,W]")
    if not isinstance(confidence, torch.Tensor) or confidence.shape != index.shape:
        raise ValueError("confidence shape must equal match_index shape")
    if index.device != device or confidence.device != device:
        raise ValueError("Correspondences must share the feature device")
    index = index.detach().flatten(1)
    q = confidence.detach().flatten(1).float()
    if index.shape != (batch, student_tokens):
        raise ValueError("Correspondence grid differs from student patches")
    if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("confidence must be finite in [0,1]")
    valid = (index >= 0) & (index < teacher_tokens)
    if ((q > 0) & ~valid).any():
        raise ValueError("Positive confidence requires an in-range teacher patch")
    return index.clamp(0, teacher_tokens - 1), q * valid


def _feature_inputs(student_features, teacher_features, teacher_keep, match_index, confidence):
    student = _tokens(student_features, "student features")
    teacher = _tokens(teacher_features.detach(), "teacher features")
    keep = _patch_values(teacher_keep.detach(), "teacher keep")
    if (student.shape[:2] != teacher.shape[:2] or student.device != teacher.device
            or keep.shape != (teacher.shape[0], teacher.shape[-1]) or keep.device != teacher.device):
        raise ValueError("Teacher/student feature channels, keep grid, batch, or device differ")
    if (keep < -1e-4).any() or (keep > 1 + 1e-4).any():
        raise ValueError("Teacher keep must be a probability")
    index, q = _match(match_index, confidence, student.shape[0], student.shape[-1],
                      teacher.shape[-1], student.device)
    target = teacher.gather(2, index[:, None].expand(-1, teacher.shape[1], -1))
    target_keep = keep.gather(1, index).clamp(0, 1)
    # A zero target has no cosine direction and gives no patch supervision.
    q = q * (target.norm(dim=1) > torch.finfo(torch.float32).tiny)
    return student, target, target_keep, q


def _keep_inputs(student_keep, teacher_keep, match_index, confidence):
    student = _patch_values(student_keep, "student keep")
    teacher = _patch_values(teacher_keep.detach(), "teacher keep")
    if student.shape[0] != teacher.shape[0] or student.device != teacher.device:
        raise ValueError("Student/teacher keep batch or device differs")
    if (teacher < -1e-4).any() or (teacher > 1 + 1e-4).any():
        raise ValueError("Teacher keep must be a probability")
    index, q = _match(match_index, confidence, student.shape[0], student.shape[1],
                      teacher.shape[1], student.device)
    return student, teacher.gather(1, index).clamp(0, 1), q


def _importance(q, teacher_keep, mode):
    if mode not in ("saliency", "uniform"):
        raise ValueError("importance_mode must be saliency or uniform")
    return q * teacher_keep.detach() if mode == "saliency" else q


def _weighted(values, q, eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    result = (values * q).sum() / (q.sum() + eps)
    if not torch.isfinite(result):
        raise ValueError("Nonfinite local loss")
    return result


def sgpd_patch_loss(student_features, teacher_features, teacher_keep,
                    match_index, confidence, *, importance_mode="saliency", eps=1e-8):
    """Trusted-patch weighted 1-cosine. Teacher and weights are detached."""
    with _fp32(student_features):
        student, target, keep, q = _feature_inputs(
            student_features, teacher_features, teacher_keep, match_index, confidence)
        weights = _importance(q, keep, importance_mode)
        cosine = (F.normalize(student, dim=1, eps=1e-6)
                  * F.normalize(target, dim=1, eps=1e-6)).sum(dim=1).clamp(-1, 1)
        return _weighted(1 - cosine, weights, eps)


def keep_alignment_loss(student_keep, teacher_keep, match_index, confidence, *, eps=1e-8):
    """Optional trusted-patch MSE of SALAD assignment_keep."""
    with _fp32(student_keep):
        student, target, q = _keep_inputs(
            student_keep, teacher_keep, match_index, confidence)
        return _weighted((student - target).square(), q, eps)


def patch_magnitude_loss(student_features, teacher_features, teacher_keep,
                         match_index, confidence, *, importance_mode="saliency",
                         beta=1.0, eps=1e-8):
    """Optional Smooth L1 of patch feature norms, complementary to cosine.

    Divide norms by sqrt(channels), making beta comparable across backbones.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Smooth L1 beta must be finite and nonnegative")
    with _fp32(student_features):
        student, target, keep, q = _feature_inputs(
            student_features, teacher_features, teacher_keep, match_index, confidence)
        weights = _importance(q, keep, importance_mode)
        scale = math.sqrt(student.shape[1])
        values = F.smooth_l1_loss(
            student.norm(dim=1) / scale, target.norm(dim=1) / scale,
            reduction="none", beta=beta)
        return _weighted(values, weights, eps)


def _conditional_ot_values(student_log_assignment, teacher_log_assignment,
                           match_index, confidence, min_teacher_keep):
    student = _tokens(student_log_assignment, "student log assignment")
    teacher = _tokens(teacher_log_assignment.detach(), "teacher log assignment")
    if (student.shape[:2] != teacher.shape[:2] or student.shape[1] < 2
            or student.device != teacher.device):
        raise ValueError("OT tensors need matching batch/clusters/device and a dustbin")
    index, q = _match(match_index, confidence, student.shape[0], student.shape[-1],
                      teacher.shape[-1], student.device)
    target = teacher.gather(2, index[:, None].expand(-1, teacher.shape[1], -1))
    floor = torch.finfo(torch.float32).eps if min_teacher_keep is None else min_teacher_keep
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("min_teacher_keep must be finite in [0,1)")
    # Last row is dustbin. Evaluate teacher keep in log space, then condition
    # both distributions on valid clusters only.
    log_keep = torch.logsumexp(target[:, :-1], dim=1) - torch.logsumexp(target, dim=1)
    low_keep = log_keep <= math.log(max(floor, torch.finfo(torch.float32).tiny))
    weights = q * ~low_keep
    log_student = F.log_softmax(student[:, :-1], dim=1)
    log_teacher = F.log_softmax(target[:, :-1], dim=1)
    values = F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(1).clamp_min(0)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite FP32 conditional OT KL")
    return values, weights, low_keep & (q > 0)


def conditional_ot_distillation_loss(student_log_assignment, teacher_log_assignment,
                                     match_index, confidence, *, min_teacher_keep=None,
                                     eps=1e-8):
    """FP32 KL(teacher || student) over valid clusters on A-class patches.

    Only the teacher's low keep mass can mask a trusted match. Keep alignment
    is a separate term; conditional KL alone does not constrain dustbin mass.
    """
    with _fp32(student_log_assignment):
        values, weights, _ = _conditional_ot_values(
            student_log_assignment, teacher_log_assignment,
            match_index, confidence, min_teacher_keep)
        return _weighted(values, weights, eps)


def weighted_patch_losses(student_aux: dict, teacher_aux: dict,
                          match_index: torch.Tensor, confidence: torch.Tensor, *,
                          importance_mode: str = "saliency",
                          include_magnitude: bool = False, magnitude_beta: float = 1.0,
                          include_ot: bool = False, min_teacher_keep: float | None = None,
                          eps: float = 1e-8):
    """Backward-compatible SGPD bundle with separately enabled extra terms.

    Patch and keep are always returned for older callers. Magnitude and OT
    return differentiable zero unless explicitly enabled.
    """
    patch = sgpd_patch_loss(
        student_aux["backbone_local_features"], teacher_aux["backbone_local_features"],
        teacher_aux["assignment_keep"], match_index, confidence,
        importance_mode=importance_mode, eps=eps)
    keep = keep_alignment_loss(
        student_aux["assignment_keep"], teacher_aux["assignment_keep"],
        match_index, confidence, eps=eps)
    magnitude = patch_magnitude_loss(
        student_aux["backbone_local_features"], teacher_aux["backbone_local_features"],
        teacher_aux["assignment_keep"], match_index, confidence,
        importance_mode=importance_mode, beta=magnitude_beta, eps=eps
    ) if include_magnitude else patch * 0.0
    if include_ot:
        with _fp32(student_aux["log_assignment"]):
            values, ot_weights, low_keep = _conditional_ot_values(
                student_aux["log_assignment"], teacher_aux["log_assignment"],
                match_index, confidence, min_teacher_keep)
            ot = _weighted(values, ot_weights, eps)
            ot_mass = ot_weights.sum().detach()
            low_keep_count = low_keep.sum().detach()
    else:
        ot = patch * 0.0
        ot_mass = confidence.detach().new_zeros(())
        low_keep_count = torch.zeros((), dtype=torch.long, device=confidence.device)
    with _fp32(student_aux["assignment_keep"]):
        _, target_keep, q = _keep_inputs(
            student_aux["assignment_keep"], teacher_aux["assignment_keep"],
            match_index, confidence)
        importance = _importance(q, target_keep, importance_mode)
    return {"patch": patch, "keep": keep, "magnitude": magnitude, "ot": ot,
            "positive_patches": (q > 0).sum().detach(),
            "positive_mass": q.sum().detach(),
            "teacher_keep_mass": (q * target_keep).sum().detach(),
            "patch_weight_mass": importance.sum().detach(),
            "ot_mass": ot_mass, "low_teacher_keep_patches": low_keep_count}


def descriptor_alignment(student_desc: torch.Tensor, teacher_desc: torch.Tensor):
    """Global alignment in the unchanged day SALAD descriptor space."""
    if (student_desc.ndim != 2 or teacher_desc.shape != student_desc.shape
            or student_desc.device != teacher_desc.device):
        raise ValueError("Global descriptors must have the same [B,D] shape/device")
    with _fp32(student_desc):
        student, teacher = student_desc.float(), teacher_desc.detach().float()
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise ValueError("Nonfinite global descriptors")
        if not len(student):
            return student.sum() * 0.0
        difference = F.normalize(student, dim=1, eps=1e-12) - F.normalize(teacher, dim=1, eps=1e-12)
        return 0.5 * difference.square().sum(dim=1).mean()


def cross_domain_infonce(student_desc, teacher_desc, place_ids, temperature=0.1):
    """Multi-positive query-to-day place discrimination; day targets detach."""
    if (student_desc.ndim != 2 or teacher_desc.shape != student_desc.shape
            or student_desc.device != teacher_desc.device):
        raise ValueError("Contrastive descriptors must have the same [B,D] shape/device")
    if len(place_ids) != len(student_desc):
        raise ValueError("place_ids must match batch length")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    with _fp32(student_desc):
        queries, references = student_desc.float(), teacher_desc.detach().float()
        if not torch.isfinite(queries).all() or not torch.isfinite(references).all():
            raise ValueError("Nonfinite contrastive descriptors")
        if not len(queries):
            return queries.sum() * 0.0
        logits = (F.normalize(queries, dim=1) @ F.normalize(references, dim=1).T) / temperature
        logits = logits - logits.amax(dim=1, keepdim=True).detach()
        positive = torch.tensor([[a == b for b in place_ids] for a in place_ids],
                                dtype=torch.bool, device=logits.device)
        losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(
            logits.masked_fill(~positive, -torch.inf), dim=1)
        return losses.mean()

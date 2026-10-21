"""SGPD and global descriptor losses. Teacher targets are always detached."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def weighted_patch_losses(student_aux: dict, teacher_aux: dict,
                          match_index: torch.Tensor, confidence: torch.Tensor, *,
                          importance_mode: str = "saliency"):
    """SGPD: confidence * teacher keep weighted cosine and optional keep MSE.

    match_index: [B,N_student] -> flattened teacher patch. -1 means unknown.
    confidence: [B,N_student] in [0,1], zero for all unverified matches.
    No gradients pass into teacher data or confidence.
    """
    if importance_mode not in ("saliency", "uniform"):
        raise ValueError("importance_mode must be saliency or uniform")
    student = student_aux["backbone_local_features"].float().flatten(2)
    teacher = teacher_aux["backbone_local_features"].detach().float().flatten(2)
    B, C, N = student.shape
    _, Ct, Nt = teacher.shape
    if Ct != C or match_index.shape != (B, N) or confidence.shape != (B, N):
        raise ValueError("Patch/feature/correspondence shapes mismatch")
    if match_index.dtype != torch.long:
        raise TypeError("match_index must be torch.long")
    if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("confidence must lie in [0,1]")
    valid = (match_index >= 0) & (match_index < Nt)
    if ((confidence > 0) & ~valid).any():
        raise ValueError("positive-confidence patch has an invalid match index")
    index = match_index.clamp(0, Nt - 1)
    target = teacher.gather(2, index.unsqueeze(1).expand(-1, C, -1))
    teacher_keep = teacher_aux["assignment_keep"].detach().float().gather(1, index)
    student_keep = student_aux["assignment_keep"].float()
    q = confidence.detach().float() * valid.float()
    importance = q * teacher_keep.detach() if importance_mode == "saliency" else q
    # Keep probability is an OT routing value, NOT a semantic landmark label.
    cosine = (F.normalize(student, dim=1, eps=1e-6) *
              F.normalize(target, dim=1, eps=1e-6)).sum(dim=1).clamp(-1, 1)
    feat_loss = ((1.0 - cosine) * importance).sum() / (importance.sum() + 1e-8)
    keep_loss = ((student_keep - teacher_keep).square() * q).sum() / (q.sum() + 1e-8)
    return {"patch": feat_loss, "keep": keep_loss,
            "positive_patches": (q > 0).sum().detach(),
            "positive_mass": q.sum().detach(),
            "teacher_keep_mass": importance.sum().detach()}


def descriptor_alignment(student_desc: torch.Tensor, teacher_desc: torch.Tensor):
    """0.5 * squared L2 distance of unit descriptors, avoiding tiny per-dim MSE."""
    return .5 * (F.normalize(student_desc.float(), dim=1) -
                 F.normalize(teacher_desc.detach().float(), dim=1)).square().sum(1).mean()


def cross_domain_infonce(student_desc, teacher_desc, place_ids, temperature=0.1):
    """Night queries versus detached day references; multi-positive place labels.

    This is a cross-domain retrieval objective, NOT the source paper's Triplet.
    For a batch with one unique place, loss is identically zero.
    """
    if len(place_ids) != len(student_desc) or len(place_ids) != len(teacher_desc):
        raise ValueError("place_ids must match batch length")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = F.normalize(student_desc.float(), dim=-1)
    k = F.normalize(teacher_desc.detach().float(), dim=-1)
    logits = (q @ k.T) / temperature
    mask = torch.tensor([[a == b for b in place_ids] for a in place_ids],
                        dtype=torch.bool, device=logits.device)
    return (torch.logsumexp(logits, dim=1) -
            torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)).mean()


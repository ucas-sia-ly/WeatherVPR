"""Exact additive decomposition of a fixed SALAD retrieval margin.

These are signed contributions to one positive-minus-negative cosine score at
the *observed* forward pass. They are not gradients, semantic segmentation,
attention probabilities, or causal effects of removing a patch. SALAD cluster
normalizers are held at their observed values for the additive decomposition.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F


def _float_tensor(value, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if min(value.shape) < 1:
        raise ValueError(f"{name} must be nonempty")
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")
    return value


def signed_patch_margin(descriptor, content_features, log_assignment,
                        gallery_positive, gallery_negative, *, token_dim=256,
                        eps=1e-12, atol=1e-5) -> dict[str, torch.Tensor]:
    """Decompose a fixed positive-minus-negative SALAD cosine margin.

    Arguments have shapes ``descriptor/positive/negative [B,D]``, projected
    ``content_features [B,C,N]``, and full ``log_assignment [B,K+1,N]`` with
    dustbin last. SALAD flattens the cluster block in **C,K** order, so
    ``D = token_dim + C*K``. Every batch row may use its own fixed Gallery
    positive and negative; choosing these images is the caller's responsibility.

    Cluster vectors are recomputed using SALAD's multiplication-and-reduction
    order. The final normalizer accounts for zero or sub-epsilon clusters,
    rather than assuming every cluster has unit norm. A nonzero, unit-normalized
    pre-concatenation CLS branch is required, as in the current trained SALAD.
    The reconstructed cluster slice and score must agree with the supplied
    descriptor within ``atol``; incompatible tensors fail rather than producing
    a plausible but incorrect heatmap. ``-inf`` log probabilities are accepted
    as exact zero assignments; NaN, +inf, and probabilities above one fail.

    Outputs are detached FP32 tensors on the input device. ``residual`` is
    ``total_reconstructed - direct_margin``. Gallery vectors and the supplied
    descriptor are normalized before cosine scoring. This is a read-only
    diagnostic and never contributes a training gradient.
    """
    if type(token_dim) is not int or token_dim < 1:
        raise ValueError("token_dim must be a positive integer")
    for name, value in (("eps", eps), ("atol", atol)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(descriptor, torch.Tensor):
        raise ValueError("descriptor must be a 2D Tensor")
    precision = (torch.autocast(device_type=descriptor.device.type, enabled=False)
                 if descriptor.device.type in ("cpu", "cuda") else nullcontext())
    with torch.no_grad(), precision:
        query = _float_tensor(descriptor, "descriptor", 2)
        content = _float_tensor(content_features, "content_features", 3)
        positive = _float_tensor(gallery_positive, "gallery_positive", 2)
        negative = _float_tensor(gallery_negative, "gallery_negative", 2)
        if (not isinstance(log_assignment, torch.Tensor)
                or log_assignment.ndim != 3 or min(log_assignment.shape) < 1):
            raise ValueError("log_assignment must be nonempty [B,K+1,N]")
        if not log_assignment.is_floating_point():
            raise TypeError("log_assignment must be floating point")
        assignment_log = log_assignment.detach().float()
        if (torch.isnan(assignment_log).any() or torch.isposinf(assignment_log).any()
                or (assignment_log > math.log1p(atol)).any()):
            raise ValueError("log_assignment must encode probabilities in [0,1]")
        if any(value.device != query.device
               for value in (content, assignment_log, positive, negative)):
            raise ValueError("All attribution tensors must share a device")
        batch, channels, patches = content.shape
        clusters = assignment_log.shape[1] - 1
        if (clusters < 1 or tuple(assignment_log.shape) != (batch, clusters + 1, patches)
                or tuple(query.shape) != (batch, token_dim + channels * clusters)
                or positive.shape != query.shape or negative.shape != query.shape):
            raise ValueError("Incompatible SALAD batch, descriptor, cluster, or patch shapes")
        for name, value in (("descriptor", query), ("gallery_positive", positive),
                            ("gallery_negative", negative)):
            norms = value.norm(dim=1, keepdim=True)
            if (norms <= eps).any() or not torch.isfinite(norms).all():
                raise ValueError(f"{name} must have finite nonzero norms")
        query = F.normalize(query, dim=1, eps=eps)
        delta = (F.normalize(positive, dim=1, eps=eps)
                 - F.normalize(negative, dim=1, eps=eps))
        cls = query[:, :token_dim]
        cls_energy = cls.square().sum(1)
        if (cls_energy <= eps * eps).any():
            raise ValueError("SALAD attribution requires a nonzero CLS branch")

        probabilities = assignment_log[:, :-1].exp()
        # Match SALAD's [B,C,K,N] reduction and subsequent C,K flatten order.
        aggregate = (content[:, :, None, :] * probabilities[:, None, :, :]).sum(-1)
        cluster_norm = aggregate.norm(dim=1).clamp_min(eps)
        unit_clusters = aggregate / cluster_norm[:, None, :]
        global_norm = (1 + unit_clusters.square().sum((1, 2))).sqrt()
        reconstructed_clusters = unit_clusters / global_norm[:, None, None]
        actual_clusters = query[:, token_dim:].reshape(batch, channels, clusters)
        if (not torch.isfinite(reconstructed_clusters).all()
                or not torch.allclose(reconstructed_clusters, actual_clusters,
                                      atol=atol, rtol=0)):
            raise ValueError("Descriptor cluster slice does not match SALAD auxiliaries")
        expected_cls_energy = global_norm.square().reciprocal()
        if not torch.allclose(cls_energy, expected_cls_energy, atol=atol, rtol=0):
            raise ValueError("Descriptor does not have SALAD's normalized CLS branch")

        cluster_delta = delta[:, token_dim:].reshape(batch, channels, clusters)
        # Score each observed patch before summing it into its cluster vectors.
        projected_scores = torch.einsum("bcn,bck->bkn", content, cluster_delta)
        patch_scores = (probabilities * projected_scores
                        / cluster_norm[:, :, None]
                        / global_norm[:, None, None]).sum(1)
        cls_score = (cls * delta[:, :token_dim]).sum(1)
        reconstructed = cls_score + patch_scores.sum(1)
        direct = (query * delta).sum(1)
        residual = reconstructed - direct
        if (not torch.isfinite(patch_scores).all()
                or not torch.isfinite(residual).all()
                or (residual.abs() > atol).any()):
            raise ValueError("Patch attribution failed SALAD margin additivity check")
        return {
            "patch_contributions": patch_scores.detach(),
            "cls_contribution": cls_score.detach(),
            "cluster_contribution": patch_scores.sum(1).detach(),
            "total_reconstructed": reconstructed.detach(),
            "direct_margin": direct.detach(),
            "residual": residual.detach(),
            "cls_energy_fraction": cls_energy.detach(),
            "cluster_energy_fraction": actual_clusters.square().sum((1, 2)).detach(),
        }

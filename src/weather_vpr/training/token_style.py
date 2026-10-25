"""Call-local partial Patch statistic mixing, inspired by TFS-ViT.

This VPR variant preserves CLS and token positions, uses detached per-image
channel statistics over Patch tokens only, and updates a subset of patches.
It is not the authors' original TFS implementation. It owns no Torch module,
parameters or global RNG state and is never installed on an inference model.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import torch


class TokenStyleAugment:
    """Mix one other batch image's style into independently selected patches.

    ``transform`` is deterministic for (seed, epoch, step). The caller alone
    decides whether a particular training branch receives the transform.
    """

    def __init__(self, seed: int, probability: float = 0.5, fraction: float = 0.3,
                 strength: float = 0.2, beta: float = 0.1, eps: float = 1e-6):
        if type(seed) is not int:
            raise ValueError("Token style seed must be an integer")
        for name, value in (("probability", probability), ("fraction", fraction),
                            ("strength", strength)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise ValueError(f"Token style {name} must be finite and in [0,1]")
        for name, value in (("beta", beta), ("eps", eps)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Token style {name} must be finite and positive")
        self.seed = seed
        self.probability = float(probability)
        self.fraction = float(fraction)
        self.strength = float(strength)
        self.beta = float(beta)
        self.eps = float(eps)
        self.last_record: dict = {}

    def configuration(self) -> dict:
        return {"method": "partial_patch_statistic_mix_v1", "seed": self.seed,
                "probability": self.probability, "fraction": self.fraction,
                "strength": self.strength, "beta": self.beta, "eps": self.eps,
                "statistics": "per-image per-channel; patch tokens only; FP32",
                "std": "sqrt(var(correction=0) + eps)",
                "statistics_detached": True, "cls_preserved": True,
                "donors": "batch derangement; no self donor", "gain_clipping": None}

    def _record(self, epoch, step, *, enabled, skip_reason=None, **extra):
        self.last_record = {"version": 1, "seed": self.seed, "epoch": epoch,
                            "step": step, "enabled": enabled, "applied_images": 0,
                            "applied_patches": 0, "raw_gain_min": 0.0,
                            "raw_gain_max": 0.0, "standardized_shift_abs_max": 0.0,
                            "relative_rms_delta": 0.0, "skip_reason": skip_reason, **extra}

    def transform(self, tokens: torch.Tensor, *, epoch: int, step: int) -> torch.Tensor:
        # True off switches do not inspect tensors or instantiate a local RNG.
        # This includes nonfinite tensors, so a disabled option is exact identity.
        if not self.probability or not self.fraction or not self.strength:
            self._record(epoch, step, enabled=False, skip_reason="disabled")
            return tokens
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
                or not tokens.is_floating_point() or tokens.shape[1] < 2
                or tokens.shape[2] < 1):
            raise ValueError("Token style requires floating [batch,1+patches,channels] tokens")
        batch, sequence, _ = tokens.shape
        count = math.floor((sequence - 1) * self.fraction)
        common = {"batch_size": batch, "patch_count": sequence - 1,
                  "patches_per_image": count}
        if batch < 2:
            self._record(epoch, step, enabled=True, skip_reason="batch_smaller_than_two", **common)
            return tokens
        if count == 0:
            self._record(epoch, step, enabled=True, skip_reason="no_selected_patches", **common)
            return tokens
        if type(epoch) is not int or type(step) is not int or epoch < 0 or step < 0:
            raise ValueError("Token style epoch and step must be nonnegative integers")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Token style inputs must be finite")
        payload = json.dumps(["token_style_v1", self.seed, epoch, step],
                             separators=(",", ":")).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
        selected = [index for index in range(batch) if rng.random() < self.probability]
        if not selected:
            self._record(epoch, step, enabled=True, skip_reason="no_selected_images", **common)
            return tokens
        donors = list(range(batch))
        # Rejection of fixed points samples a uniform random derangement. It
        # avoids both accidental self donors and dependence on global RNGs.
        while True:
            rng.shuffle(donors)
            if all(index != donor for index, donor in enumerate(donors)):
                break
        indices = [sorted(rng.sample(range(sequence - 1), count)) for _ in selected]
        lambdas = [self.strength * (1.0 - rng.betavariate(self.beta, self.beta))
                   for _ in selected]
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            patches = tokens[:, 1:].float()
            statistics = patches.detach()
            mean = statistics.mean(dim=1, keepdim=True)
            std = (statistics.var(dim=1, correction=0, keepdim=True) + self.eps).sqrt()
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                raise ValueError("Token style FP32 statistics must be finite")
            image_ids = torch.tensor(selected, dtype=torch.long, device=tokens.device)
            donor_ids = torch.tensor([donors[index] for index in selected],
                                     dtype=torch.long, device=tokens.device)
            patch_ids = torch.tensor(indices, dtype=torch.long, device=tokens.device)
            mix = torch.tensor(lambdas, dtype=torch.float32, device=tokens.device)[:, None, None]
            recipient_mean, recipient_std = mean[image_ids], std[image_ids]
            mixed_mean = recipient_mean + mix * (mean[donor_ids] - recipient_mean)
            mixed_std = recipient_std + mix * (std[donor_ids] - recipient_std)
            gain = mixed_std / recipient_std
            shift = (mixed_mean - recipient_mean) / recipient_std
            original = patches[image_ids[:, None], patch_ids]
            stylized = ((original - recipient_mean) / recipient_std) * mixed_std + mixed_mean
            if not bool(torch.isfinite(stylized).all()):
                raise ValueError("Token style transformed patches must be finite")
            # Only selected patches pass through FP32. Clone/CopySlices keeps
            # untouched BF16 samples, patches and CLS exactly as supplied.
            output = tokens.clone()
            output[image_ids[:, None], patch_ids + 1] = stylized.to(tokens.dtype)
            if not bool(torch.isfinite(output).all()):
                raise ValueError("Token style cast result must be finite")
            actual_delta = output[:, 1:].detach().float() - statistics
            # Common scaling avoids overflow in diagnostic sum-of-squares.
            scale = torch.maximum(statistics.abs().amax(), actual_delta.abs().amax()).clamp_min(self.eps)
            rms = (statistics / scale).square().mean().sqrt().clamp_min(self.eps / scale)
            relative_rms_delta = float((actual_delta / scale).square().mean().sqrt() / rms)
            per_image_scale = torch.maximum(statistics.abs().amax((1, 2)),
                                            actual_delta.abs().amax((1, 2))).clamp_min(self.eps)
            per_image_rms = (statistics / per_image_scale[:, None, None]).square().mean((1, 2)).sqrt()
            per_image_rms = torch.maximum(per_image_rms, self.eps / per_image_scale)
            per_image_delta = (actual_delta / per_image_scale[:, None, None]).square().mean((1, 2)).sqrt()
            per_image_relative_rms_delta = (per_image_delta / per_image_rms).tolist()
            centered = statistics - mean
            centered_scale = torch.maximum(centered.abs().amax((1, 2)),
                                             actual_delta.abs().amax((1, 2))).clamp_min(self.eps)
            centered_rms = (centered / centered_scale[:, None, None]).square().mean((1, 2)).sqrt()
            centered_rms = torch.maximum(centered_rms, self.eps / centered_scale)
            centered_delta = (actual_delta / centered_scale[:, None, None]).square().mean((1, 2)).sqrt()
            per_image_centered_relative_rms_delta = (centered_delta / centered_rms).tolist()
            gain_quantiles = torch.quantile(gain.flatten(), torch.tensor(
                [0.5, 0.9, 0.95, 0.99], dtype=torch.float32, device=tokens.device)).tolist()
        lambda_record = [0.0] * batch
        for index, value in zip(selected, lambdas):
            lambda_record[index] = value
        self._record(
            epoch, step, enabled=True, **common, applied_images=len(selected),
            applied_patches=len(selected) * count, selected_images=selected,
            donor_indices=donors, lambdas=lambda_record,
            patch_indices=indices if len(selected) * count <= 4096 else None,
            raw_gain_min=float(gain.min()), raw_gain_max=float(gain.max()),
            raw_gain_percentiles=dict(zip(("p50", "p90", "p95", "p99"), gain_quantiles)),
            raw_gain_fraction_gt2=float((gain > 2).float().mean()),
            raw_gain_fraction_gt4=float((gain > 4).float().mean()),
            raw_gain_fraction_gt8=float((gain > 8).float().mean()),
            standardized_shift_abs_max=float(shift.abs().max()),
            relative_rms_delta=relative_rms_delta,
            relative_rms_definition="RMS(actual_delta over all patches)/max(RMS(input patches),eps)",
            per_image_relative_rms_delta=per_image_relative_rms_delta,
            per_image_centered_relative_rms_delta=per_image_centered_relative_rms_delta)
        return output

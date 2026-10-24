"""Detached photometric reliability of the actual augmented Student input.

This only reduces local supervision where camera processing leaves pixels
nearly black or clips all RGB channels. It has no geometric or semantic role:
it never creates matches, classifies buildings, or reads Student features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


PATCH_VISIBILITY_VERSION = "srgb_nonclipped_fraction_v1"
PATCH_VISIBILITY_PARAMETERS = {
    "patch_size": 14,
    "dark_srgb_luma_threshold": 0.01,
    "saturated_all_channels_threshold": 0.99,
    "aggregation": "fraction_of_visible_pixels",
    "input_normalization": "imagenet_rgb",
}


def visibility_metadata():
    """JSON-compatible constants for binding training logs to the heuristic."""
    return {"mode": "photometric", "version": PATCH_VISIBILITY_VERSION,
            **PATCH_VISIBILITY_PARAMETERS}


def patch_visibility_from_image_tensor(normalized_student, *, patch_size=14):
    """Return detached FP32 ``[B,H/patch_size,W/patch_size]`` reliability.

    Inputs use the project's ImageNet RGB normalization after all Student
    photometric augmentations. A pixel is visible when sRGB luminance exceeds
    0.01 and at least one RGB channel is below 0.99. The score is the fraction
    of visible pixels per patch, with no hard patch exclusion threshold. Most
    ordinary low-light pixels remain eligible. This is a conservative proxy
    for clipping, not a calibrated estimate of recoverable scene information.
    """
    if (not isinstance(normalized_student, torch.Tensor)
            or normalized_student.ndim != 4 or normalized_student.shape[1] != 3
            or min(normalized_student.shape) < 1):
        raise ValueError("normalized_student must be nonempty [B,3,H,W]")
    if not normalized_student.is_floating_point():
        raise TypeError("normalized_student must be floating point")
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if (normalized_student.shape[2] % patch_size
            or normalized_student.shape[3] % patch_size):
        raise ValueError("Image dimensions must be divisible by patch_size")
    with torch.no_grad():
        image = normalized_student.detach().float()
        if not torch.isfinite(image).all():
            raise ValueError("Nonfinite normalized_student")
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = (image * std + mean).clamp(0, 1)
        luma = (rgb * image.new_tensor((0.2126, 0.7152, 0.0722))
                .view(1, 3, 1, 1)).sum(1, keepdim=True)
        dark = luma <= PATCH_VISIBILITY_PARAMETERS["dark_srgb_luma_threshold"]
        saturated = (rgb >= PATCH_VISIBILITY_PARAMETERS[
            "saturated_all_channels_threshold"]).all(1, keepdim=True)
        return F.avg_pool2d((~dark & ~saturated).float(), patch_size,
                            stride=patch_size).squeeze(1).detach()

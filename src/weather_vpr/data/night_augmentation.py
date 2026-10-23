"""Training-only camera degradations for synthetic and real night images.

The caller owns when to enable this transform. It never changes image geometry,
so clean images can still supply patch correspondence and Teacher targets. All
randomness comes from an explicit local seed; validation/test must leave it off.
Ranges describe augmentation hypotheses, not calibrated camera measurements.
Strengths at or below 1e-6 are treated as identity: changes are below useful
8-bit resolution, while inverse-square shot-noise scaling would be unstable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


_OP_PROBABILITIES = {
    "exposure": 0.9,
    "vignette": 0.4,
    "motion_blur": 0.35,
    "bloom": 0.4,
    "noise": 0.8,
    "desaturation": 0.45,
    "photometric": 0.6,
    "jpeg": 0.65,
}
_IDENTITY_STRENGTH_EPS = 1e-6


@dataclass(frozen=True)
class NightAugmentationConfig:
    profile: str = "off"
    strength: float = 1.0
    probability: float = 0.8
    operation_probabilities: dict[str, float] = field(default_factory=dict)
    gain_range: tuple[float, float] = (0.6, 2.0)
    exposure_target_range: tuple[float, float] = (0.06, 0.20)
    gamma_range: tuple[float, float] = (0.75, 1.30)
    bloom_threshold_range: tuple[float, float] = (0.50, 0.85)
    bloom_strength_range: tuple[float, float] = (0.15, 0.55)
    bloom_radius_fraction_range: tuple[float, float] = (0.004, 0.018)
    motion_length_range: tuple[int, int] = (3, 9)
    photon_count_range: tuple[float, float] = (500.0, 2400.0)
    read_noise_std_range: tuple[float, float] = (0.0005, 0.0035)
    jpeg_quality_range: tuple[int, int] = (40, 90)
    saturation_range: tuple[float, float] = (0.35, 0.85)
    vignette_strength_range: tuple[float, float] = (0.15, 0.55)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.75, 1.25)

    def __post_init__(self):
        if self.profile not in ("off", "camera"):
            raise ValueError("night augmentation profile must be 'off' or 'camera'")
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 2:
            raise ValueError("night augmentation strength must be finite and in [0, 2]")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("night augmentation probability must be in [0, 1]")
        if not isinstance(self.operation_probabilities, Mapping):
            raise ValueError("operation_probabilities must be an object")
        for name, probability in self.operation_probabilities.items():
            if name not in _OP_PROBABILITIES:
                raise ValueError(f"unknown camera operation: {name}")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"operation probability for {name} must be in [0, 1]")
        for name in self.__dataclass_fields__:
            if not name.endswith("_range"):
                continue
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(not math.isfinite(x) for x in value)
                    or value[0] > value[1] or value[0] < 0):
                raise ValueError(f"{name} must be an ordered finite nonnegative pair")
            if name in ("gain_range", "gamma_range", "photon_count_range") and value[0] <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("motion_length_range", "jpeg_quality_range"):
            if any(int(x) != x for x in getattr(self, name)):
                raise ValueError(f"{name} must contain integers")
        if not 1 <= self.motion_length_range[0] <= self.motion_length_range[1] <= 31:
            raise ValueError("motion_length_range must be within [1, 31]")
        if not 1 <= self.jpeg_quality_range[0] <= self.jpeg_quality_range[1] <= 100:
            raise ValueError("jpeg_quality_range must be within [1, 100]")
        for name in ("exposure_target_range", "bloom_threshold_range",
                     "saturation_range", "vignette_strength_range"):
            if getattr(self, name)[1] > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.bloom_radius_fraction_range[1] > 0.1:
            raise ValueError("bloom radius fraction must not exceed 0.1")
        if self.read_noise_std_range[1] > 0.25:
            raise ValueError("read noise standard deviation must not exceed 0.25")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NightAugmentationConfig":
        if not isinstance(value, Mapping):
            raise ValueError("night augmentation config must be a JSON object")
        return cls(**dict(value))


def night_augmentation_seed(global_seed: int, epoch: int, sample: str | int) -> int:
    """Stable across worker count, process restart, and Python hash randomization."""
    payload = f"{int(global_seed)}\0{int(epoch)}\0{sample}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def _rgb_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def _motion_blur(signal: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Line point spread function with replicated borders, never wrapped pixels."""
    half = length // 2
    offsets = {(int(round(t * math.sin(angle))), int(round(t * math.cos(angle))))
               for t in range(-half, half + 1)}
    radius = max(max(abs(y), abs(x)) for y, x in offsets)
    padded = np.pad(signal, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    height, width = signal.shape[:2]
    result = np.zeros_like(signal)
    for y, x in sorted(offsets):
        result += padded[radius+y:radius+y+height, radius+x:radius+x+width]
    return result / len(offsets)


class NightCameraAugmentation:
    def __init__(self, config: NightAugmentationConfig):
        if not isinstance(config, NightAugmentationConfig):
            raise TypeError("config must be NightAugmentationConfig")
        self.config = config
        self.probabilities = {**_OP_PROBABILITIES, **config.operation_probabilities}

    def __call__(self, image: Image.Image, *, seed: int) -> tuple[Image.Image, dict]:
        if not isinstance(image, Image.Image):
            raise TypeError("night augmentation expects a PIL image")
        if min(image.size) < 1:
            raise ValueError("night augmentation requires nonempty image dimensions")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        cfg = self.config
        record = {"profile": cfg.profile, "strength": float(cfg.strength),
                  "seed": int(seed), "applied": False, "operations": {}}
        original = image.convert("RGB").copy()
        if (cfg.profile == "off" or cfg.strength <= _IDENTITY_STRENGTH_EPS
                or cfg.probability == 0):
            return original, record
        rng = np.random.default_rng(seed)
        if rng.random() >= cfg.probability:
            return original, record
        strength = cfg.strength
        operations = record["operations"]
        choose = lambda name: rng.random() < self.probabilities[name]
        draw = lambda bounds: float(rng.uniform(*bounds))
        signal = _to_linear(np.asarray(original, dtype=np.float32) / 255.0)
        gamma = 1.0

        if choose("exposure"):
            luminance = signal @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            target = draw(cfg.exposure_target_range)
            gain = float(np.clip(target / max(float(luminance.mean()), 1e-4), *cfg.gain_range))
            gain = gain ** strength
            gamma = max(0.1, 1 + strength * (draw(cfg.gamma_range) - 1))
            signal *= gain
            operations["exposure"] = {"gain": gain, "gamma": gamma,
                                      "target_linear_luminance": target}

        if choose("vignette"):
            amount = min(0.95, draw(cfg.vignette_strength_range) * strength)
            height, width = signal.shape[:2]
            center_x, center_y = rng.uniform(-0.08, 0.08, 2)
            yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
            radius = (xx-center_x)**2 + (yy-center_y)**2
            radius /= max(float(radius.max()), 1e-6)
            signal *= (1 - amount * radius**0.8)[..., None]
            operations["vignette"] = {"strength": amount,
                                      "center_offset": [float(center_x), float(center_y)]}

        if choose("motion_blur"):
            sampled = int(rng.integers(cfg.motion_length_range[0], cfg.motion_length_range[1]+1))
            length = min(31, max(1, int(round(1 + strength*(sampled-1)))))
            length = min(31, length if length % 2 else length+1)
            angle = float(rng.uniform(0, math.pi))
            signal = _motion_blur(signal, length, angle)
            operations["motion_blur"] = {"length": length, "angle_degrees": math.degrees(angle)}

        if choose("bloom"):
            threshold = draw(cfg.bloom_threshold_range)
            amount = draw(cfg.bloom_strength_range) * strength
            radius = max(0.1, min(original.size) * draw(cfg.bloom_radius_fraction_range))
            highlights = np.clip((signal-threshold) / max(1-threshold, 1e-4), 0, 1)
            halo = np.asarray(_rgb_image(highlights).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)/255
            signal += amount * halo
            operations["bloom"] = {"threshold": threshold, "strength": amount, "radius_pixels": radius}

        if choose("noise"):
            photons = draw(cfg.photon_count_range) / strength**2
            read_std = draw(cfg.read_noise_std_range) * strength
            # Shot noise acts on expected nonnegative photoelectrons; Gaussian
            # read noise is then added before clipping and camera tone mapping.
            signal = rng.poisson(np.clip(signal, 0, 4) * photons).astype(np.float32) / photons
            signal += rng.normal(0, read_std, signal.shape).astype(np.float32)
            operations["noise"] = {"photon_count": photons, "read_noise_std": read_std}

        clipped_fraction = float(np.mean(np.max(signal, axis=2) >= 1))
        signal = np.clip(signal, 0, 1) ** gamma
        output = _rgb_image(_to_srgb(signal))
        if choose("desaturation"):
            saturation = max(0.0, 1 + strength*(draw(cfg.saturation_range)-1))
            output = ImageEnhance.Color(output).enhance(saturation)
            operations["desaturation"] = {"saturation": saturation}

        if choose("photometric"):
            # A sampled brightness/contrast policy supplies strong photometric
            # variation without the rotations/crops of generic RandAugment.
            sampled_ops = rng.choice(["brightness", "contrast"], size=int(rng.integers(1, 3)), replace=False)
            params = {}
            for op in sampled_ops:
                bounds = cfg.brightness_range if op == "brightness" else cfg.contrast_range
                factor = max(0.0, 1 + strength*(draw(bounds)-1))
                enhancer = ImageEnhance.Brightness if op == "brightness" else ImageEnhance.Contrast
                output = enhancer(output).enhance(factor)
                params[str(op)] = factor
            operations["photometric"] = params

        if choose("jpeg"):
            sampled = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1]+1))
            quality = int(np.clip(round(100 - strength*(100-sampled)), 1, 100))
            buffer = BytesIO()
            output.save(buffer, format="JPEG", quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                output = compressed.convert("RGB").copy()
            operations["jpeg"] = {"quality": quality, "subsampling": 2}

        record["applied"] = bool(operations)
        record["clipped_pixel_fraction"] = clipped_fraction
        return output, record

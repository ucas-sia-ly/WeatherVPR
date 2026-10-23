"""Explicit low-light views of real Day photographs for Night Student training.

This is a photometric augmentation hypothesis, not a physical night renderer:
it does not invent lamps, change scene content, or claim to reproduce real night
illumination. Every call applies sub-unit exposure in linear sRGB. Optional
smooth illumination and white balance precede camera degradations. The caller
must resize to the network resolution first and decide whether to construct
this transform at all; there is deliberately no probability of skipping the
mandatory low-light conversion and no ``off`` profile.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np
from PIL import Image

from weather_vpr.data.night_augmentation import (
    NightAugmentationConfig, NightCameraAugmentation, _rgb_image, _to_linear,
    _to_srgb,
)


_DEFAULT_CAMERA_OPERATIONS = {"exposure": 0.0, "bloom": 0.0, "photometric": 0.0}


def _default_camera_config() -> NightAugmentationConfig:
    return NightAugmentationConfig(
        profile="camera", probability=1.0,
        operation_probabilities=dict(_DEFAULT_CAMERA_OPERATIONS),
    )


def _finite_range(name: str, value: Any) -> tuple[float, float]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(x, Real) or isinstance(x, bool)
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must be an ordered pair of finite numbers")
    low, high = map(float, value)
    if low > high:
        raise ValueError(f"{name} must be an ordered pair of finite numbers")
    return low, high


@dataclass(frozen=True)
class SourceNightConfig:
    """Serializable low-light policy; camera exposure cannot be re-enabled.

    ``camera_probability`` and ``camera_strength`` own the camera stage gate
    and strength, replacing those fields in the nested camera config. A nested
    ``profile='off'`` or zero camera strength disables only camera degradations,
    never the mandatory source darkening.
    A zero source loss weight should make the caller skip this transform.
    """

    gain_range: tuple[float, float] = (0.03, 0.30)
    gamma_range: tuple[float, float] = (1.0, 1.30)
    illumination_probability: float = 0.7
    illumination_floor_range: tuple[float, float] = (0.35, 1.0)
    illumination_grid_size: tuple[int, int] = (3, 3)
    white_balance_probability: float = 0.75
    white_balance_range: tuple[float, float] = (0.8, 1.2)
    camera_probability: float = 0.8
    camera_strength: float = 1.0
    camera_config: NightAugmentationConfig = field(default_factory=_default_camera_config)

    def __post_init__(self):
        for name in ("gain_range", "gamma_range", "illumination_floor_range",
                     "white_balance_range"):
            object.__setattr__(self, name, _finite_range(name, getattr(self, name)))
        if not 0 < self.gain_range[0] <= self.gain_range[1] < 1:
            raise ValueError("gain_range must be positive and strictly below 1")
        if self.gamma_range[0] < 1:
            raise ValueError("gamma_range must be at least 1 in linear sRGB")
        if not 0 < self.illumination_floor_range[0] <= self.illumination_floor_range[1] <= 1:
            raise ValueError("illumination_floor_range must be in (0, 1]")
        if self.white_balance_range[0] <= 0:
            raise ValueError("white_balance_range must be positive")
        for name in ("illumination_probability", "white_balance_probability",
                     "camera_probability"):
            value = getattr(self, name)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, float(value))
        if (not isinstance(self.camera_strength, Real) or isinstance(self.camera_strength, bool)
                or not math.isfinite(self.camera_strength) or not 0 <= self.camera_strength <= 2):
            raise ValueError("camera_strength must be finite and in [0, 2]")
        object.__setattr__(self, "camera_strength", float(self.camera_strength))
        # Colour balance must not cancel the required per-channel darkening.
        if (self.white_balance_probability > 0
                and self.gain_range[1] * self.white_balance_range[1] >= 1):
            raise ValueError("gain_range times white_balance_range must remain below 1")
        grid = self.illumination_grid_size
        if (not isinstance(grid, (tuple, list)) or len(grid) != 2
                or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool)
                       or not 2 <= x <= 8 for x in grid)):
            raise ValueError("illumination_grid_size must contain two integers in [2, 8]")
        object.__setattr__(self, "illumination_grid_size", tuple(map(int, grid)))
        camera = self.camera_config
        if isinstance(camera, Mapping):
            camera = NightAugmentationConfig.from_dict(camera)
        if not isinstance(camera, NightAugmentationConfig):
            raise ValueError("camera_config must be a NightAugmentationConfig or JSON object")
        if camera.operation_probabilities.get("exposure", 0) != 0:
            raise ValueError("source-night camera exposure must remain disabled")
        camera = replace(
            camera, probability=self.camera_probability, strength=self.camera_strength,
            operation_probabilities={**_DEFAULT_CAMERA_OPERATIONS,
                                     **camera.operation_probabilities},
        )
        object.__setattr__(self, "camera_config", camera)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceNightConfig":
        if not isinstance(value, Mapping):
            raise ValueError("source-night config must be a JSON object")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"invalid source-night config: {exc}") from exc


def _domain_seed(seed: int, domain: str) -> int:
    payload = f"source-night-v1\0{seed}\0{domain}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _mean_luminance(signal: np.ndarray) -> float:
    return float((signal @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean())


class SourceNightAugmentation:
    """Darken a resized PIL image without changing geometry or global RNGs.

    Returns ``(RGB_image, record)``. ``record['operations']`` always contains
    ``exposure`` and ``gamma``; optional camera operations have ``camera_``
    prefixes so existing operation counters can count them independently.
    ``record['camera']`` retains the complete existing camera-stage record.
    All seeds, sampled gains, colour factors and illumination grid values are
    recorded; records can be serialized with ``allow_nan=False``.
    """

    def __init__(self, config: SourceNightConfig):
        if not isinstance(config, SourceNightConfig):
            raise TypeError("config must be SourceNightConfig")
        # Snapshot mutable nested dictionaries and revalidate before execution.
        self.config = SourceNightConfig.from_dict(config.to_dict())
        self.camera = NightCameraAugmentation(self.config.camera_config)

    def __call__(self, image: Image.Image, *, seed: int) -> tuple[Image.Image, dict]:
        if not isinstance(image, Image.Image):
            raise TypeError("source-night augmentation expects a PIL image")
        if min(image.size) < 1:
            raise ValueError("source-night augmentation requires nonempty image dimensions")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        cfg = self.config
        seeds = {name: _domain_seed(int(seed), name)
                 for name in ("exposure", "gamma", "illumination", "white_balance", "camera")}
        original = image.convert("RGB").copy()
        signal = _to_linear(np.asarray(original, dtype=np.float32) / 255.0)
        input_mean = _mean_luminance(signal)
        exposure_rng = np.random.default_rng(seeds["exposure"])
        gain = float(np.exp(exposure_rng.uniform(*np.log(cfg.gain_range))))
        gamma = float(np.random.default_rng(seeds["gamma"]).uniform(*cfg.gamma_range))
        signal *= gain
        operations: dict[str, Any] = {
            "exposure": {"gain": gain, "distribution": "log_uniform", "domain": "linear_srgb"},
            "gamma": {"gamma": gamma, "domain": "linear_srgb"},
        }

        illumination_rng = np.random.default_rng(seeds["illumination"])
        if illumination_rng.random() < cfg.illumination_probability:
            floor = float(illumination_rng.uniform(*cfg.illumination_floor_range))
            grid = illumination_rng.uniform(0, 1, size=cfg.illumination_grid_size)
            grid = (grid-grid.min()) / max(float(grid.max()-grid.min()), 1e-12)
            grid = (floor + (1-floor) * grid).astype(np.float32)
            # Bicubic interpolation produces a smooth low-frequency scalar
            # field. Clamp its interpolation overshoot to the sampled bounds.
            illumination = np.asarray(Image.fromarray(grid).resize(
                original.size, Image.Resampling.BICUBIC), dtype=np.float32)
            illumination = np.clip(illumination, floor, 1.0)
            signal *= illumination[..., None]
            operations["illumination"] = {
                "floor": floor, "grid_shape_hw": list(grid.shape),
                "grid_values": grid.tolist(), "interpolation": "bicubic_clipped",
                "actual_min": float(illumination.min()),
                "actual_max": float(illumination.max()),
            }

        balance_rng = np.random.default_rng(seeds["white_balance"])
        if balance_rng.random() < cfg.white_balance_probability:
            factors = balance_rng.uniform(*cfg.white_balance_range, size=3).astype(np.float32)
            signal *= factors[None, None, :]
            operations["white_balance"] = {"rgb_factors": factors.tolist()}
        signal = np.clip(signal, 0, 1) ** gamma
        pre_camera_mean = _mean_luminance(signal)
        darkened = _rgb_image(_to_srgb(signal))
        output, camera_record = self.camera(darkened, seed=seeds["camera"])
        for name, parameters in camera_record["operations"].items():
            operations[f"camera_{name}"] = parameters
        record = {
            "profile": "source_night", "version": 1, "seed": int(seed),
            "stage_seeds": seeds, "applied": True, "operations": operations,
            "resolution_hw": [original.height, original.width],
            "input_linear_luminance_mean": input_mean,
            "pre_camera_linear_luminance_mean": pre_camera_mean,
            "output_srgb_luminance_mean": _mean_luminance(
                np.asarray(output, dtype=np.float32)/255.0),
            "camera": camera_record,
        }
        return output, record

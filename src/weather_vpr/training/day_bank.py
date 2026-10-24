"""An immutable Teacher descriptor bank built only from training Day images.

The paired dataset itself is never indexed: doing so would also decode its
generated weather image and sample Student augmentation. Cache identities use
the actual Day image bytes, including their ordered places, rather than trusting
the image hashes recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import warnings

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from weather_vpr.data.place_sampler import normalize_place
from weather_vpr.utils.backend import enable_backend_imports


_FORMAT_VERSION = 1
_IDENTITY_KEYS = (
    "format_version", "checkpoint_sha256", "preprocessing", "manifest_sha256",
    "source_images", "places",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_tensors(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    """Hash and map a stable cache file without keeping all its bytes in RAM.

    Cache writers publish with atomic replacement. On Linux, the open-file
    path pins the same inode for hashing and mapping even during replacement.
    Other platforms reject a replacement observed while loading instead.
    """
    with path.open("rb") as stream:
        original = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("cache tensor SHA-256 mismatch")
        stable_path = Path("/proc/self/fd") / str(stream.fileno())
        pinned = stable_path.is_file()
        tensors = load_file(str(stable_path if pinned else path), device="cpu")
        current = os.fstat(stream.fileno()) if pinned else path.stat()
        # Unlinking the old inode during atomic publication changes ctime,
        # while its pinned bytes and mtime remain unchanged.
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size,
                                 stat.st_mtime_ns, None if pinned else stat.st_ctime_ns)
        if identity(current) != identity(original):
            raise ValueError("cache tensor changed while loading")
        return tensors


def _signature(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_size(preprocessing: dict, dataset) -> tuple[int, int]:
    if not isinstance(preprocessing, dict):
        raise ValueError("Day bank preprocessing must be the SALAD preprocessing dictionary")
    size = preprocessing.get("image_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)):
        raise ValueError("Day bank preprocessing.image_size must be two positive integers")
    expected = {
        "version": "rgb_bilinear_imagenet_v1", "image_size": list(size),
        "color_space": "RGB", "input_range": [0.0, 1.0], "resize": "PIL_bilinear",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    }
    if preprocessing != expected:
        raise ValueError("Day bank requires the original SALAD RGB/bilinear/ImageNet preprocessing")
    if hasattr(dataset, "size") and tuple(dataset.size) != tuple(size):
        raise ValueError("Training dataset image size differs from Day bank preprocessing")
    return tuple(size)


class _DayImages(Dataset):
    """Decode exactly the signed Day bytes, preserving dataset row order."""

    def __init__(self, sources: list[dict], image_size: tuple[int, int]):
        self.sources = sources
        self.image_size = image_size

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index]
        path = Path(source["path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise OSError(f"Cannot read training Day image {path}: {error}") from error
        if hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise RuntimeError(f"Training Day image changed while building its bank: {path}; rerun")
        enable_backend_imports()
        from workflow.reliability_cache import preprocess_teacher_image
        try:
            with Image.open(io.BytesIO(payload)) as image:
                return preprocess_teacher_image(image, self.image_size)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode training Day image {path}: {error}") from error


def _validate_descriptors(descriptors: torch.Tensor, rows: int,
                          shape=None, expected_dim: int | None = None) -> None:
    if (not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 2
            or descriptors.dtype != torch.float32 or descriptors.device.type != "cpu"
            or descriptors.shape[0] != rows or descriptors.shape[1] < 1):
        raise ValueError("Day bank descriptors must be CPU FP32 [training_rows, descriptor_dim]")
    if shape is not None and list(descriptors.shape) != shape:
        raise ValueError("Day bank tensor shape differs from cache metadata")
    if expected_dim is not None and descriptors.shape[1] != expected_dim:
        raise ValueError("Day bank descriptor dimension differs from frozen SALAD")
    for start in range(0, len(descriptors), 1024):
        part = descriptors[start:start + 1024]
        if not torch.isfinite(part).all():
            raise ValueError("Day bank contains nonfinite descriptors")
        norms = torch.linalg.vector_norm(part, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-5):
            raise ValueError("Day bank descriptors are not L2 normalized")


def _descriptor_dim(net) -> int | None:
    config = getattr(getattr(net, "base", None), "config", None)
    if isinstance(config, dict) and config.get("agg_arch") == "SALAD":
        agg = config.get("agg_config", {})
        values = [agg.get(key) for key in ("num_clusters", "cluster_dim", "token_dim")]
        if all(type(value) is int and value > 0 for value in values):
            return values[0] * values[1] + values[2]
    return None


def _load_cache(path: Path, expected: dict, expected_dim: int | None):
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        return None, "cache tensor or metadata is missing"
    try:
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("cache metadata is not a dictionary")
        for key in _IDENTITY_KEYS:
            if saved.get(key) != expected[key]:
                return None, f"cache identity changed: {key}"
        saved_identity = {key: saved[key] for key in _IDENTITY_KEYS}
        if (saved.get("cache_signature") != _signature(expected)
                or _signature(saved_identity) != _signature(expected)):
            raise ValueError("cache signature mismatch")
        tensors = _load_verified_tensors(path, saved.get("tensor_sha256"))
        if set(tensors) != {"descriptors"}:
            raise ValueError("cache must contain only descriptors")
        descriptors = tensors["descriptors"]
        shape = saved.get("descriptor_shape")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(value) is not int or value < 1 for value in shape)):
            raise ValueError("invalid cached descriptor_shape")
        _validate_descriptors(descriptors, len(expected["places"]), shape, expected_dim)
        return (descriptors, saved), None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError) as error:
        return None, f"invalid Day bank cache: {error}"


def _publish_cache(path: Path, metadata: dict, descriptors: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_tensor = temporary_json = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".day_bank_",
                                         suffix=".safetensors", delete=False) as stream:
            temporary_tensor = Path(stream.name)
        save_file({"descriptors": descriptors.contiguous()}, str(temporary_tensor))
        payload = dict(metadata, tensor_sha256=_sha256(temporary_tensor))
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".day_bank_", suffix=".json",
                                         mode="w", encoding="utf-8", delete=False) as stream:
            temporary_json = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # The JSON commits the tensor hash. Interrupted/concurrent publication
        # can only produce a cache miss, never accept a mismatched tensor pair.
        os.replace(temporary_tensor, path)
        temporary_tensor = None
        os.replace(temporary_json, sidecar)
        temporary_json = None
        return payload
    finally:
        for temporary in (temporary_tensor, temporary_json):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_or_build_day_bank(net, dataset, checkpoint_hash, preprocessing, cache_path,
                           device, batch_size=8, num_workers=0):
    """Return ``(CPU FP32 normalized descriptors, ordered places, metadata)``.

    ``dataset.rows`` must provide ``source_file`` and ``place``; ``manifest``
    identifies its original JSONL. Only these training Day files are read.
    ``cache_path=None`` builds in memory. A cache uses that safetensors path
    and a JSON sidecar with ``.json`` appended. Missing or invalid caches are
    rebuilt; ``cache_status`` and ``rebuild_reason`` record this decision.
    Teacher forwards run under ``no_grad`` without changing any train/eval mode.
    """
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
        raise ValueError("Day bank batch_size must be positive and num_workers nonnegative")
    if (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_hash)):
        raise ValueError("Day bank checkpoint_hash must be a lowercase SHA-256")
    size = _image_size(preprocessing, dataset)
    manifest = Path(dataset.manifest).expanduser().resolve(strict=True)
    rows = list(dataset.rows)
    if not rows:
        raise ValueError("Cannot build a Day bank from an empty training dataset")
    sources, places = [], []
    for index, row in enumerate(rows):
        try:
            source = Path(row["source_file"]).expanduser().resolve(strict=True)
            if not source.is_file():
                raise FileNotFoundError(f"not a file: {source}")
            place = normalize_place(row["place"])
            image_hash = _sha256(source)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid training Day row {index}: {error}") from error
        sources.append({"path": str(source), "sha256": image_hash})
        places.append(place)
    # Canonical JSON containers prevent tuple/list differences on a cache hit.
    identity = {
        "format_version": _FORMAT_VERSION, "checkpoint_sha256": checkpoint_hash,
        "preprocessing": json.loads(json.dumps(preprocessing, allow_nan=False)),
        "manifest_sha256": _sha256(manifest), "source_images": sources,
        "places": [list(place) for place in places],
    }
    expected_dim = _descriptor_dim(net)
    path = Path(cache_path).expanduser().resolve() if cache_path is not None else None
    existed, reason = False, None
    if path is not None:
        existed = path.exists() or path.with_suffix(path.suffix + ".json").exists()
        cached, reason = _load_cache(path, identity, expected_dim)
        if cached is not None:
            descriptors, saved = cached
            return descriptors, places, dict(saved, cache_status="hit", rebuild_reason=None)
        if existed:
            warnings.warn(f"Rebuilding Day bank {path}: {reason}", RuntimeWarning, stacklevel=2)
    loader = DataLoader(_DayImages(sources, size), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False,
                        pin_memory=torch.device(device).type == "cuda",
                        generator=torch.Generator().manual_seed(0))
    chunks = []
    with torch.no_grad():
        for images in loader:
            descriptors = net.teacher(images.to(device), return_aux=False)
            if (not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 2
                    or descriptors.shape[0] != len(images) or descriptors.shape[1] < 1):
                raise ValueError("Frozen Teacher must return [batch, descriptor_dim] Day descriptors")
            descriptors = descriptors.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(descriptors).all() or (descriptors.norm(dim=1) <= 1e-12).any():
                raise ValueError("Frozen Teacher returned nonfinite or zero Day descriptors")
            chunks.append(F.normalize(descriptors, dim=1))
    descriptors = torch.cat(chunks, dim=0).contiguous()
    _validate_descriptors(descriptors, len(rows), expected_dim=expected_dim)
    if _sha256(manifest) != identity["manifest_sha256"]:
        raise RuntimeError("Training manifest changed while building the Day bank; rerun")
    metadata = dict(identity, cache_signature=_signature(identity),
                    descriptor_shape=list(descriptors.shape),
                    cache_status=("memory" if path is None else "rebuilt" if existed else "built"),
                    rebuild_reason=reason)
    if path is not None:
        metadata = _publish_cache(path, metadata, descriptors)
    return descriptors, places, metadata

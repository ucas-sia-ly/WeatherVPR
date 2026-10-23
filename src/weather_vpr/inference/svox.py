"""SVOX test: fixed day gallery versus real night queries at 25 m UTM radius.

The day SALAD descriptor space is shared by the baseline and the night adapter.
The gallery cache is tied to the full checkpoint, ordered image bytes and exact
preprocessing contract; changing any of them forces a fresh extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.inference.loading import load_weather_model
from weather_vpr.utils.backend import enable_backend_imports, project_root


def _require_night_adapter_dir(value: Path) -> Path:
    """Check run artifacts before loading SVOX data or the backbone."""
    directory = Path(value).expanduser().resolve()
    missing = [name for name in ("adapter.json", "adapter.safetensors")
               if not (directory / name).is_file()]
    if directory.is_dir() and not missing:
        return directory
    if not directory.exists():
        problem = "directory does not exist"
        next_step = ("Create it by training: OUTPUT=runs/night_full_sgpd_01 "
                     "bash configs/night_full_sgpd.sh.")
    elif not directory.is_dir():
        problem = "path is not a directory"
        next_step = "Pass a completed training output directory via --adapter-dir."
    else:
        problem = "incomplete directory; missing " + ", ".join(missing)
        next_step = ("Training may still be running; wait until both adapter.json and "
                     "adapter.safetensors are saved. If it stopped, start a new run in an "
                     "unused directory, for example OUTPUT=runs/night_full_sgpd_02 "
                     "bash configs/night_full_sgpd.sh.")
    smoke = project_root() / "runs/night_sgpd_smoke_20261021"
    smoke_hint = (
        "For a quick pipeline check, use --adapter-dir runs/night_sgpd_smoke_20261021 "
        "(a short smoke run, not a trained result)."
        if (smoke / "adapter.json").is_file() and (smoke / "adapter.safetensors").is_file()
        else "For a quick pipeline check, create a smoke adapter with "
             "OUTPUT=runs/night_full_global_smoke bash configs/night_full_global.sh."
    )
    raise FileNotFoundError(
        f"Night adapter {problem}: {directory}.\n"
        f"{next_step}\n" + smoke_hint
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gallery_fingerprint(images) -> str:
    """Hash the ordered SVOX gallery IDs and image bytes, not file timestamps."""
    digest = hashlib.sha256()
    for image in images:
        name = image.id.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with image.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_night_dataset(svox_root: Path, *, split: str = "test"):
    """Use the native SVOX night protocol; train split is for development checks."""
    enable_backend_imports()
    from workflow.evaluation import load_evaluation_set

    if split not in {"train", "test"}:
        raise ValueError("SVOX night evaluation supports train or test split")
    root = Path(svox_root).expanduser().resolve()
    # The generic upstream loader would prefer this file over SVOX coordinates.
    if (root / "evaluation_manifest.json").exists():
        raise ValueError("SVOX night evaluation requires the native UTM protocol; remove evaluation_manifest.json")
    dataset = load_evaluation_set(
        "SVOX", root, split=split, query_subdirs=["queries_night"], positive_radius=25.0
    )
    expected = {"ground_truth": "utm_radius", "positive_radius_meters": 25.0,
                "split": split, "query_subdirs": ["queries_night"]}
    if dataset.protocol != expected:
        raise ValueError(f"Unexpected SVOX protocol: {dataset.protocol}")
    return dataset


def _cache_metadata(images, gallery_hash: str, checkpoint_hash: str, preprocessing: dict) -> dict:
    return {"format_version": 1, "checkpoint_sha256": checkpoint_hash,
            "gallery_sha256": gallery_hash, "image_ids": [image.id for image in images],
            "preprocessing": preprocessing}


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _load_gallery_cache(cache_path: Path, expected: dict, descriptor_dim: int):
    metadata_path = _metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in expected.items()):
            return None
        if saved.get("tensor_sha256") != sha256(cache_path):
            return None
        tensors = load_file(str(cache_path), device="cpu")
        if set(tensors) != {"descriptors"}:
            return None
        vectors = tensors["descriptors"]
        if (vectors.dtype != torch.float32
                or tuple(vectors.shape) != (len(expected["image_ids"]), descriptor_dim)
                or not torch.isfinite(vectors).all()):
            return None
        return vectors
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def _save_gallery_cache(cache_path: Path, metadata: dict, vectors: torch.Tensor) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(cache_path)
    temporary_cache = temporary_metadata = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=".svox_gallery_",
                                         suffix=".safetensors", delete=False) as stream:
            temporary_cache = Path(stream.name)
        save_file({"descriptors": vectors.detach().cpu().contiguous()}, str(temporary_cache))
        payload = dict(metadata, tensor_sha256=sha256(temporary_cache))
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=".svox_gallery_",
                                         suffix=".json", mode="w", encoding="utf-8",
                                         delete=False) as stream:
            temporary_metadata = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary_cache, cache_path)
        temporary_cache = None
        os.replace(temporary_metadata, metadata_path)
        temporary_metadata = None
    finally:
        for path in (temporary_cache, temporary_metadata):
            if path is not None:
                path.unlink(missing_ok=True)


class _NightStudent(torch.nn.Module):
    def __init__(self, net: WeatherAdapterSALAD):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.student(images, "night", return_aux=False)


def _validate_device(value: str) -> str:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU or CUDA evaluation devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA is unavailable for requested device {value}")
    return str(device)


def _validate_query_scales(scales, num_clusters: int | None = None):
    if scales is None:
        return None
    if (not isinstance(scales, (tuple, list)) or not scales
            or any(type(scale) is not int or scale <= 0 or scale % 14 for scale in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError('query-scales must be distinct positive integer multiples of 14')
    if num_clusters is not None and any((scale // 14) ** 2 <= num_clusters for scale in scales):
        raise ValueError('Each query scale must provide more patches than SALAD clusters')
    return tuple(scales)


def _parse_query_scales(value: str):
    try:
        return _validate_query_scales(tuple(int(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'query-scales must be a comma-separated list of distinct positive multiples of 14, e.g. 224,336'
        ) from exc


def _scaled_query_batch_size(batch_size: int, scale: int) -> int:
    return max(1, batch_size * 224 * 224 // (scale * scale))


def _unit_query_vectors(vectors, count: int, descriptor_dim: int):
    if not isinstance(vectors, torch.Tensor) or tuple(vectors.shape) != (count, descriptor_dim):
        raise ValueError('Query extraction must return [query_count, descriptor_dim] descriptors')
    vectors = vectors.detach().float().cpu()
    if not torch.isfinite(vectors).all():
        raise ValueError('Query descriptors must be finite in FP32')
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError('Query descriptors must have finite nonzero norms')
    return vectors / norms


def _extract_multiscale_queries(model, images, scales, device: str, batch_size: int,
                                num_workers: int, descriptor_dim: int):
    """Resize each original image independently; average unit FP32 descriptors."""
    from workflow.evaluation import extract_descriptors

    total = None
    for scale in scales:
        # The extractor reopens the original paths for every scale. Passing the
        # original image metadata prevents 224 -> 336 resampling cascades.
        vectors = extract_descriptors(model, images, (scale, scale), device,
                                      _scaled_query_batch_size(batch_size, scale), num_workers)
        vectors = _unit_query_vectors(vectors, len(images), descriptor_dim)
        if total is None:
            total = vectors
        else:
            total += vectors
    if total is None:
        raise ValueError('At least one query scale is required')
    return _unit_query_vectors(total / len(scales), len(images), descriptor_dim)


def _default_output_path(split: str, query_scales=None):
    name = 'svox_night_evaluation' if split == 'test' else 'svox_train_night_evaluation'
    if query_scales is not None:
        name += '_query' + '-'.join(map(str, query_scales))
    return project_root() / 'runs' / (name + '.json')


def evaluate_svox_night(*, svox_root: Path, checkpoint: Path, backbone_repo: Path,
                        adapter_dir: Path | None, output: Path, cache_path: Path | None,
                        device: str, retrieval_device: str, batch_size: int,
                        num_workers: int, query_chunk_size: int,
                        reference_chunk_size: int, reference_cache_mib: int,
                        save_hard_cases: bool = False, split: str = "test",
                        adapter_scale: float = 1.0,
                        query_scales: tuple[int, ...] | None = None) -> dict:
    """Write baseline and optional adapter R@1/5/10 against one unchanged gallery."""
    if batch_size < 1 or num_workers < 0 or query_chunk_size < 1 or reference_chunk_size < 1 or reference_cache_mib < 0:
        raise ValueError("Batch/chunk sizes must be positive; workers and cache MiB nonnegative")
    query_scales = _validate_query_scales(query_scales)
    if adapter_dir is not None:
        adapter_dir = _require_night_adapter_dir(adapter_dir)
    device = _validate_device(device)
    retrieval_device = _validate_device(retrieval_device)
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    dataset = load_night_dataset(svox_root, split=split)
    enable_backend_imports()
    from workflow.evaluation import build_results, exact_retrieval, extract_descriptors
    from workflow.model import input_preprocessing, load_checkpoint_model

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone_repo = Path(backbone_repo).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if cache_path is not None:
        cache_path = Path(cache_path).expanduser().resolve()
    checkpoint_hash = sha256(checkpoint)
    print(f"SVOX {split}: {len(dataset.references)} day gallery, {len(dataset.queries)} night queries; "
          f"{dataset.summary()['num_queries_without_positives']} queries without a 25 m positive", flush=True)
    print(f"Loading frozen DINOv2-SALAD: {checkpoint}", flush=True)
    base = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    if tuple(base.image_size) != (224, 224):
        raise ValueError(f"Native SVOX comparison requires 224x224 preprocessing, got {base.image_size}")
    preprocessing = input_preprocessing(base.image_size)
    descriptor_dim = (base.config["agg_config"]["num_clusters"] *
                      base.config["agg_config"]["cluster_dim"] +
                      base.config["agg_config"]["token_dim"])
    _validate_query_scales(query_scales, base.config['agg_config']['num_clusters'])
    query_preprocessing = {
        'mode': 'native_single_scale' if query_scales is None else 'query_multiscale',
        'scales': [224] if query_scales is None else list(query_scales),
        'per_scale': [preprocessing] if query_scales is None else [
            input_preprocessing((scale, scale)) for scale in query_scales],
        'aggregation': ('original_model_descriptor' if query_scales is None else
                        'FP32 L2 per scale -> equal-weight mean -> L2'),
        'resize_source': 'original RGB image independently at each scale',
        'batch_sizes': [batch_size] if query_scales is None else [
            _scaled_query_batch_size(batch_size, scale) for scale in query_scales],
        'scope': 'same query policy for baseline and Night; frozen 224 Day Gallery unchanged',
    }
    student = None
    metadata = None
    if adapter_dir is not None:
        metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
        if (metadata.get("weather") != "night" or
                metadata.get("checkpoint_sha256") != checkpoint_hash or
                metadata.get("model_config") != base.config or
                metadata.get("preprocessing") != preprocessing):
            raise ValueError("Night adapter is incompatible with checkpoint, model configuration or preprocessing")
        net = load_weather_model(base, "night", adapter_dir, metadata,
                                 device=device, adapter_scale=adapter_scale)
        student = _NightStudent(net)

    vectors = None
    gallery_hash = None
    if cache_path is not None:
        print("Hashing ordered gallery image bytes for cache validation...", flush=True)
        gallery_hash = gallery_fingerprint(dataset.references)
        expected_cache = _cache_metadata(dataset.references, gallery_hash,
                                         checkpoint_hash, preprocessing)
        vectors = _load_gallery_cache(cache_path, expected_cache, descriptor_dim)
        if vectors is not None:
            print(f"Loaded day gallery descriptors: {cache_path}", flush=True)
    if vectors is None:
        print("Extracting original day SALAD gallery descriptors...", flush=True)
        vectors = extract_descriptors(base, dataset.references, base.image_size,
                                      device, batch_size, num_workers)
        if cache_path is not None:
            _save_gallery_cache(cache_path, expected_cache, vectors)
            print(f"Saved day gallery descriptors: {cache_path}", flush=True)

    print("Extracting unadapted night query descriptors...", flush=True)
    if query_scales is None:
        baseline_vectors = extract_descriptors(base, dataset.queries, base.image_size,
                                               device, batch_size, num_workers)
    else:
        baseline_vectors = _extract_multiscale_queries(
            base, dataset.queries, query_scales, device, batch_size, num_workers, descriptor_dim)

    def score(queries, label):
        print(f"Ranking {label} queries against the original day gallery...", flush=True)
        rows = exact_retrieval(vectors, queries, dataset.positives, device=retrieval_device,
                               query_chunk_size=query_chunk_size,
                               reference_chunk_size=reference_chunk_size,
                               reference_cache_bytes=reference_cache_mib * 1024 * 1024)
        return build_results(dataset, rows, save_hard_cases=save_hard_cases)

    baseline = score(baseline_vectors, "baseline")
    result = {"dataset": "SVOX", "split": split, "query_group": "queries_night",
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "image_size": list(base.image_size), "preprocessing": preprocessing,
              "query_preprocessing": query_preprocessing,
              "gallery_cache": str(cache_path) if cache_path else None,
              "gallery_sha256": gallery_hash,
              "baseline": baseline, "adapter": None, "delta_recall": None}

    if student is not None:
        print("Extracting trained night query descriptors...", flush=True)
        if query_scales is None:
            adapted_vectors = extract_descriptors(student, dataset.queries,
                                                  base.image_size, device, batch_size, num_workers)
        else:
            adapted_vectors = _extract_multiscale_queries(
                student, dataset.queries, query_scales, device, batch_size, num_workers, descriptor_dim)
        adapted = score(adapted_vectors, "adapter")
        result["adapter"] = {"directory": str(adapter_dir),
                             "metadata": metadata, "scale": adapter_scale,
                             "evaluation": adapted}
        result["delta_recall"] = {key: adapted["recall"][key] - baseline["recall"][key]
                                  for key in ("R@1", "R@5", "R@10")}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print("Baseline:", baseline["recall"], flush=True)
    if result["adapter"] is not None:
        print("Night model:", result["adapter"]["evaluation"]["recall"], flush=True)
        print("Delta:", result["delta_recall"], flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate real SVOX night queries against the unchanged day gallery "
                    "using 25 m UTM positives. Baseline and optional night adapter share one day gallery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--svox-root", type=Path, default=project_root() / "data/svox",
                        help="Folder containing images/{train,test}/gallery and queries_night")
    parser.add_argument("--split", choices=("train", "test"), default="test",
                        help="Use train for real-weather development checks; test for final reporting")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=project_root() / "third_party/dinov2")
    parser.add_argument("--adapter-dir", type=Path,
                        help="Night training output with adapter.json and adapter.safetensors; omit for baseline only")
    parser.add_argument("--adapter-scale", type=float, default=1.0,
                        help="LoRA update multiplier; choose only on development data")
    parser.add_argument("--output", type=Path,
                        help="Result JSON; default runs/svox_night_evaluation.json (train has its own name)")
    parser.add_argument("--gallery-cache", type=Path,
                        help="Reusable day gallery cache; defaults to runs/cache/svox_<split>_gallery.safetensors")
    parser.add_argument("--no-gallery-cache", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--retrieval-device", help="Ranking device; defaults to --device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument('--query-scales', type=_parse_query_scales,
                        help='Optional query-only square sizes, e.g. 224,336; same policy for baseline and Night')
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--reference-cache-mib", type=int, default=0,
                        help="GPU/CPU reference cache for exact rank computation")
    parser.add_argument("--save-hard-cases", action="store_true",
                        help="Include every Recall@1 error and its exact positive rank in the JSON")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate image paths and 25 m positives without loading the model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check_data:
        print(json.dumps(load_night_dataset(args.svox_root, split=args.split).summary(), indent=2,
                         ensure_ascii=False))
        return
    if args.adapter_dir is not None:
        try:
            _require_night_adapter_dir(args.adapter_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None
    output = args.output or _default_output_path(args.split, args.query_scales)
    cache = args.gallery_cache or project_root() / 'runs/cache' / f'svox_{args.split}_gallery.safetensors'
    evaluate_svox_night(
        svox_root=args.svox_root, checkpoint=args.checkpoint,
        backbone_repo=args.backbone_repo, adapter_dir=args.adapter_dir,
        output=output, cache_path=None if args.no_gallery_cache else cache,
        device=args.device, retrieval_device=args.retrieval_device or args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        reference_cache_mib=args.reference_cache_mib,
        save_hard_cases=args.save_hard_cases, split=args.split,
        adapter_scale=args.adapter_scale,
        query_scales=args.query_scales,
    )


if __name__ == "__main__":
    main()

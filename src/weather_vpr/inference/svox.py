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
from weather_vpr.utils.backend import enable_backend_imports, project_root


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


def load_night_dataset(svox_root: Path):
    """Use the bundled SVOX loader while pinning its native test/night protocol."""
    enable_backend_imports()
    from workflow.evaluation import load_evaluation_set

    root = Path(svox_root).expanduser().resolve()
    # The generic upstream loader would prefer this file over SVOX coordinates.
    if (root / "evaluation_manifest.json").exists():
        raise ValueError("SVOX night evaluation requires the native UTM protocol; remove evaluation_manifest.json")
    dataset = load_evaluation_set(
        "SVOX", root, split="test", query_subdirs=["queries_night"], positive_radius=25.0
    )
    expected = {"ground_truth": "utm_radius", "positive_radius_meters": 25.0,
                "split": "test", "query_subdirs": ["queries_night"]}
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


def evaluate_svox_night(*, svox_root: Path, checkpoint: Path, backbone_repo: Path,
                        adapter_dir: Path | None, output: Path, cache_path: Path | None,
                        device: str, retrieval_device: str, batch_size: int,
                        num_workers: int, query_chunk_size: int,
                        reference_chunk_size: int, reference_cache_mib: int,
                        save_hard_cases: bool = False) -> dict:
    """Write baseline and optional adapter R@1/5/10 against one unchanged gallery."""
    if batch_size < 1 or num_workers < 0 or query_chunk_size < 1 or reference_chunk_size < 1 or reference_cache_mib < 0:
        raise ValueError("Batch/chunk sizes must be positive; workers and cache MiB nonnegative")
    device = _validate_device(device)
    retrieval_device = _validate_device(retrieval_device)
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    dataset = load_night_dataset(svox_root)
    enable_backend_imports()
    from workflow.evaluation import build_results, exact_retrieval, extract_descriptors
    from workflow.model import input_preprocessing, load_checkpoint_model

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone_repo = Path(backbone_repo).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if cache_path is not None:
        cache_path = Path(cache_path).expanduser().resolve()
    checkpoint_hash = sha256(checkpoint)
    print(f"SVOX test: {len(dataset.references)} day gallery, {len(dataset.queries)} night queries; "
          f"{dataset.summary()['num_queries_without_positives']} queries without a 25 m positive", flush=True)
    print(f"Loading frozen DINOv2-SALAD: {checkpoint}", flush=True)
    base = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    if tuple(base.image_size) != (224, 224):
        raise ValueError(f"Native SVOX comparison requires 224x224 preprocessing, got {base.image_size}")
    preprocessing = input_preprocessing(base.image_size)
    descriptor_dim = (base.config["agg_config"]["num_clusters"] *
                      base.config["agg_config"]["cluster_dim"] +
                      base.config["agg_config"]["token_dim"])
    student = None
    metadata = None
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir).expanduser().resolve(strict=True)
        metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
        if (metadata.get("weather") != "night" or
                metadata.get("checkpoint_sha256") != checkpoint_hash or
                metadata.get("model_config") != base.config or
                metadata.get("preprocessing") != preprocessing):
            raise ValueError("Night adapter is incompatible with checkpoint, model configuration or preprocessing")
        net = WeatherAdapterSALAD(base, ("night",), bottleneck=metadata["bottleneck"]).to(device).eval()
        weights = load_file(str(adapter_dir / "adapter.safetensors"), device="cpu")
        net.adapters["night"].load_state_dict(weights, strict=True)
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
    baseline_vectors = extract_descriptors(base, dataset.queries, base.image_size,
                                           device, batch_size, num_workers)

    def score(queries, label):
        print(f"Ranking {label} queries against the original day gallery...", flush=True)
        rows = exact_retrieval(vectors, queries, dataset.positives, device=retrieval_device,
                               query_chunk_size=query_chunk_size,
                               reference_chunk_size=reference_chunk_size,
                               reference_cache_bytes=reference_cache_mib * 1024 * 1024)
        return build_results(dataset, rows, save_hard_cases=save_hard_cases)

    baseline = score(baseline_vectors, "baseline")
    result = {"dataset": "SVOX", "split": "test", "query_group": "queries_night",
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "image_size": list(base.image_size), "preprocessing": preprocessing,
              "gallery_cache": str(cache_path) if cache_path else None,
              "gallery_sha256": gallery_hash,
              "baseline": baseline, "adapter": None, "delta_recall": None}

    if student is not None:
        print("Extracting adapted night query descriptors...", flush=True)
        adapted_vectors = extract_descriptors(student, dataset.queries,
                                              base.image_size, device, batch_size, num_workers)
        adapted = score(adapted_vectors, "adapter")
        result["adapter"] = {"directory": str(adapter_dir),
                             "metadata": metadata, "evaluation": adapted}
        result["delta_recall"] = {key: adapted["recall"][key] - baseline["recall"][key]
                                  for key in ("R@1", "R@5", "R@10")}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print("Baseline:", baseline["recall"], flush=True)
    if result["adapter"] is not None:
        print("Adapter:", result["adapter"]["evaluation"]["recall"], flush=True)
        print("Delta:", result["delta_recall"], flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate real SVOX test/queries_night against the unchanged test/gallery "
                    "using 25 m UTM positives. Baseline and optional night adapter share one day gallery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--svox-root", type=Path, default=project_root() / "data/svox",
                        help="Folder containing images/test/gallery and images/test/queries_night")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=project_root() / "third_party/dinov2")
    parser.add_argument("--adapter-dir", type=Path,
                        help="Night training output with adapter.json and adapter.safetensors; omit for baseline only")
    parser.add_argument("--output", type=Path, default=project_root() / "runs/svox_night_evaluation.json")
    parser.add_argument("--gallery-cache", type=Path,
                        default=project_root() / "runs/cache/svox_test_gallery.safetensors",
                        help="Reusable day gallery cache, validated against checkpoint, ordered images and preprocessing")
    parser.add_argument("--no-gallery-cache", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--retrieval-device", help="Ranking device; defaults to --device")
    parser.add_argument("--batch-size", type=int, default=16)
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
        print(json.dumps(load_night_dataset(args.svox_root).summary(), indent=2,
                         ensure_ascii=False))
        return
    evaluate_svox_night(
        svox_root=args.svox_root, checkpoint=args.checkpoint,
        backbone_repo=args.backbone_repo, adapter_dir=args.adapter_dir,
        output=args.output, cache_path=None if args.no_gallery_cache else args.gallery_cache,
        device=args.device, retrieval_device=args.retrieval_device or args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        reference_cache_mib=args.reference_cache_mib,
        save_hard_cases=args.save_hard_cases,
    )


if __name__ == "__main__":
    main()

"""Paired Night/Day evaluation against one original, frozen Day Gallery.

Dataset loaders supply verified query positives. This module never derives
ground truth from model predictions or from image appearance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from weather_vpr.inference.loading import load_weather_model
from weather_vpr.inference.svox import (
    _NightStudent, _cache_metadata, _load_gallery_cache, _require_night_adapter_dir,
    _query_fusion_metadata, _save_gallery_cache, _save_query_descriptors,
    _validate_device, _validate_query_fusion_alpha, fused_query_descriptors,
    gallery_fingerprint, sha256,
)
from weather_vpr.utils.backend import enable_backend_imports, project_root


class _Progress(torch.nn.Module):
    """Report extraction progress without changing the model's output."""

    def __init__(self, model, label: str, count: int):
        super().__init__()
        self.model = model
        self.label = label
        self.count = count
        self.seen = 0
        self.started = self.reported = time.monotonic()

    def forward(self, images):
        output = self.model(images)
        self.seen += len(images)
        now = time.monotonic()
        if now - self.reported >= 20 or self.seen == self.count:
            elapsed = now - self.started
            print(f"{self.label}: {self.seen}/{self.count} images, "
                  f"{self.seen / max(elapsed, .001):.1f} images/s", flush=True)
            self.reported = now
        return output


def evaluate_night_benchmark(*, dataset, checkpoint: Path, backbone_repo: Path,
                             adapter_dir: Path | None, output: Path,
                             cache_path: Path | None, device: str = "auto",
                             retrieval_device: str = "auto", batch_size: int = 32,
                             num_workers: int = 4, query_chunk_size: int = 32,
                             reference_chunk_size: int = 4096,
                             reference_cache_mib: int = 4096,
                             query_fusion_alpha: float | None = None,
                             query_descriptor_path: Path | None = None) -> dict:
    """Compare fixed weights at native 224 input, with no test-time augmentation."""
    if (batch_size < 1 or num_workers < 0 or query_chunk_size < 1
            or reference_chunk_size < 1 or reference_cache_mib < 0):
        raise ValueError("Batch/chunk sizes must be positive; workers/cache nonnegative")
    query_fusion_alpha = _validate_query_fusion_alpha(query_fusion_alpha)
    if query_fusion_alpha is not None and adapter_dir is None:
        raise ValueError('query-fusion-alpha requires a Night adapter')
    if query_descriptor_path is not None and adapter_dir is None:
        raise ValueError('Saving Teacher/Student query descriptors requires a Night adapter')
    if adapter_dir is not None:
        adapter_dir = _require_night_adapter_dir(adapter_dir)
    device = _validate_device(device)
    retrieval_device = _validate_device(retrieval_device)
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    enable_backend_imports()
    from workflow.evaluation import build_results, exact_retrieval, extract_descriptors
    from workflow.model import input_preprocessing, load_checkpoint_model

    dataset.validate(check_images=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone_repo = Path(backbone_repo).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve() if cache_path is not None else None
    checkpoint_hash = sha256(checkpoint)
    weights_hash = sha256(adapter_dir / "adapter.safetensors") if adapter_dir else None
    metadata_hash = sha256(adapter_dir / "adapter.json") if adapter_dir else None
    started = datetime.now(timezone.utc).isoformat()
    summary = dataset.summary()
    print(f"{dataset.name}: {summary['num_references']} Gallery, {summary['num_queries']} Queries, "
          f"{summary['num_queries_without_positives']} without positives; "
          f"protocol={dataset.protocol.get('ground_truth')}, split={dataset.protocol.get('split')}",
          flush=True)
    print(f"Loading original Day DINOv2-SALAD: {checkpoint}", flush=True)
    base = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    if tuple(base.image_size) != (224, 224):
        raise ValueError("This comparison requires the original 224x224 preprocessing")
    preprocessing = input_preprocessing(base.image_size)
    agg_config = base.config["agg_config"]
    descriptor_dim = agg_config["num_clusters"] * agg_config["cluster_dim"] + agg_config["token_dim"]
    metadata = student = None
    if adapter_dir is not None:
        metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
        if (metadata.get("weather") != "night"
                or metadata.get("checkpoint_sha256") != checkpoint_hash
                or metadata.get("model_config") != base.config
                or metadata.get("preprocessing") != preprocessing):
            raise ValueError("Night artifact is incompatible with the Day checkpoint or preprocessing")
        student = _NightStudent(load_weather_model(base, "night", adapter_dir, metadata, device=device))

    print("Hashing ordered Gallery and Query image bytes...", flush=True)
    gallery_hash = gallery_fingerprint(dataset.references)
    query_hash = gallery_fingerprint(dataset.queries)
    cache_metadata = _cache_metadata(dataset.references, gallery_hash, checkpoint_hash, preprocessing)
    gallery = _load_gallery_cache(cache_path, cache_metadata, descriptor_dim) if cache_path else None

    def extract(model, images, label):
        print(f"Extracting {label}...", flush=True)
        return extract_descriptors(_Progress(model, label, len(images)), images,
                                   base.image_size, device, batch_size, num_workers)

    if gallery is None:
        gallery = extract(base, dataset.references, "original Day Gallery")
        if cache_path:
            _save_gallery_cache(cache_path, cache_metadata, gallery)
            print(f"Saved Gallery cache: {cache_path}", flush=True)
    else:
        print(f"Loaded original Day Gallery cache: {cache_path}", flush=True)

    def score(queries, label):
        print(f"Exact retrieval: {label}", flush=True)
        rows = exact_retrieval(gallery, queries, dataset.positives, device=retrieval_device,
                               query_chunk_size=query_chunk_size,
                               reference_chunk_size=reference_chunk_size,
                               reference_cache_bytes=reference_cache_mib * 1024 * 1024)
        return build_results(dataset, rows, save_hard_cases=True), rows

    baseline_vectors = extract(base, dataset.queries, "Day baseline Night Queries")
    baseline, baseline_rows = score(baseline_vectors, "Day baseline Night Queries")
    result = {
        "dataset": dataset.name, "started_utc": started,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "image_size": list(base.image_size), "preprocessing": preprocessing,
        "query_preprocessing": {"mode": "native_single_scale", "test_time_augmentation": False},
        "gallery_cache": str(cache_path) if cache_path else None,
        "gallery_sha256": gallery_hash, "query_sha256": query_hash,
        "baseline": baseline, "adapter": None, "delta_recall": None,
        "per_query": [{"query_id": image.id, "baseline": row}
                      for image, row in zip(dataset.queries, baseline_rows)],
    }
    if student is not None:
        adapted_vectors = extract(student, dataset.queries, "Night specialist Queries")
        adapted, adapted_rows = score(adapted_vectors, "Night specialist Queries")
        # Reject a concurrent change to either artifact rather than mislabelling
        # the weights used during this evaluation.
        if (sha256(adapter_dir / "adapter.safetensors") != weights_hash
                or sha256(adapter_dir / "adapter.json") != metadata_hash):
            raise RuntimeError("Night artifact changed during evaluation; results were not saved")
        result["adapter"] = {"directory": str(adapter_dir), "metadata": metadata,
                             "weights_sha256": weights_hash, "metadata_sha256": metadata_hash,
                             "evaluation": adapted}
        result["delta_recall"] = {key: adapted["recall"][key] - baseline["recall"][key]
                                  for key in ("R@1", "R@5", "R@10")}
        for row, adapted_row in zip(result["per_query"], adapted_rows):
            row["night"] = adapted_row
        paired = [(day, night) for day, night in zip(baseline_rows, adapted_rows)
                  if day is not None and night is not None]
        result["paired_top1"] = {
            "fixed": sum(day["rank"] > 1 and night["rank"] == 1 for day, night in paired),
            "regressed": sum(day["rank"] == 1 and night["rank"] > 1 for day, night in paired),
            "both_correct": sum(day["rank"] == 1 and night["rank"] == 1 for day, night in paired),
            "both_wrong": sum(day["rank"] > 1 and night["rank"] > 1 for day, night in paired),
        }
        if query_descriptor_path is not None:
            result['query_descriptors'] = _save_query_descriptors(
                query_descriptor_path, teacher_vectors=baseline_vectors,
                student_vectors=adapted_vectors, images=dataset.queries,
                checkpoint_hash=checkpoint_hash, adapter_dir=adapter_dir,
                preprocessing=preprocessing, query_preprocessing=result['query_preprocessing'])
        if query_fusion_alpha is not None:
            fused_vectors = fused_query_descriptors(adapted_vectors, baseline_vectors,
                                                     query_fusion_alpha)
            fused, fused_rows = score(fused_vectors, "Teacher/Student fusion Queries")
            result['fusion'] = {
                **_query_fusion_metadata(query_fusion_alpha, result['query_preprocessing']),
                'evaluation': fused,
                'delta_recall': {key: fused['recall'][key] - baseline['recall'][key]
                                 for key in ('R@1', 'R@5', 'R@10')},
            }
            for row, fused_row in zip(result['per_query'], fused_rows):
                row['fusion'] = fused_row
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output)
    print(f"Saved {output}\nBaseline: {baseline['recall']}", flush=True)
    if student is not None:
        print(f"Night: {adapted['recall']}\nDelta: {result['delta_recall']}", flush=True)
    if 'fusion' in result:
        print(f"Query fusion: {result['fusion']['evaluation']['recall']}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("tokyo247", "robotcar-public", "manifest"), required=True)
    parser.add_argument("--root", type=Path, help="Image root (defaults to the local dataset folder)")
    parser.add_argument("--condition", choices=("night", "night-rain"), default="night",
                        help="RobotCar public pose subset; evaluated only, never used for training")
    parser.add_argument("--manifest", type=Path, help="Explicit, externally verified query-to-Gallery positives")
    parser.add_argument("--tokyo-mat", type=Path,
                        default=project_root() / "configs/evaluation_sources/tokyo247_netvlad_v100.mat")
    parser.add_argument("--tokyo-source", type=Path,
                        default=project_root() / "configs/evaluation_sources/tokyo247_netvlad_v100.source.json")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=project_root() / "third_party/dinov2")
    parser.add_argument("--adapter-dir", type=Path, help="Night model output; omit for Day baseline only")
    parser.add_argument('--query-fusion-alpha', type=float,
                        help='Optional Student weight in normalized Teacher/Student query fusion [0,1]; requires --adapter-dir')
    parser.add_argument('--save-query-descriptors', type=Path,
                        help='Export freshly extracted Teacher/Student query tensors and provenance JSON; requires --adapter-dir')
    parser.add_argument("--output", type=Path, help="Result JSON")
    parser.add_argument("--gallery-cache", type=Path)
    parser.add_argument("--no-gallery-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--reference-cache-mib", type=int, default=4096)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset == "manifest" and (args.root is None or args.manifest is None):
        parser.error("--dataset manifest requires --root and --manifest")
    if args.dataset != "manifest" and args.manifest is not None:
        parser.error("Native dataset protocols cannot be overridden by --manifest")
    try:
        _validate_query_fusion_alpha(args.query_fusion_alpha)
        if args.query_fusion_alpha is not None and args.adapter_dir is None:
            parser.error('--query-fusion-alpha requires --adapter-dir')
        if args.save_query_descriptors is not None and args.adapter_dir is None:
            parser.error('--save-query-descriptors requires --adapter-dir')
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    enable_backend_imports()
    if args.dataset == "tokyo247":
        from weather_vpr.inference.tokyo import load_tokyo_night
        dataset = load_tokyo_night(args.root or project_root() / "data/tokyo247",
                                   args.tokyo_mat, args.tokyo_source)
    elif args.dataset == "robotcar-public":
        from weather_vpr.inference.robotcar import load_robotcar_public_night
        dataset = load_robotcar_public_night(args.root or project_root() / "data/RobotCar-Seasons",
                                             args.condition)
    else:
        from workflow.evaluation import load_manifest
        dataset = load_manifest(args.manifest, args.root, args.manifest.stem)
        dataset.protocol["manifest_sha256"] = sha256(args.manifest)
    if args.check_data:
        dataset.validate(check_images=True)
        print(json.dumps(dataset.summary(), indent=2, ensure_ascii=False))
        return
    run_name = args.dataset if args.dataset != "robotcar-public" else f"robotcar_public_{args.condition}"
    output = args.output or project_root() / "runs" / f"{run_name}_night_evaluation.json"
    # Manifest stems distinguish independent galleries; the cache also binds
    # every ordered image byte and cannot be reused across different contents.
    cache_name = args.dataset if args.dataset != "manifest" else args.manifest.stem
    cache = args.gallery_cache or project_root() / "runs/cache" / f"{cache_name}_day_gallery.safetensors"
    evaluate_night_benchmark(
        dataset=dataset, checkpoint=args.checkpoint, backbone_repo=args.backbone_repo,
        adapter_dir=args.adapter_dir, output=output,
        cache_path=None if args.no_gallery_cache else cache,
        device=args.device, retrieval_device=args.retrieval_device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        query_chunk_size=args.query_chunk_size, reference_chunk_size=args.reference_chunk_size,
        reference_cache_mib=args.reference_cache_mib,
        query_fusion_alpha=args.query_fusion_alpha,
        query_descriptor_path=args.save_query_descriptors,
    )


if __name__ == "__main__":
    main()

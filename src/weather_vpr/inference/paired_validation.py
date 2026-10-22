"""Synthetic night validation on exact Qwen/day pairs, separate from real SVOX.

The frozen day SALAD gallery is shared by unadapted and adapted night queries.
Recall positives use (city, place_id); paired cosine uses each query's exact
source image, even when several day images belong to the same place.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from weather_vpr.data.paired_dataset import resolve_file, sha256
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.inference.loading import load_weather_model
from weather_vpr.utils.backend import enable_backend_imports, project_root


def _require_night_adapter_dir(value: Path) -> Path:
    """Check run artifacts before loading a model or reading validation images."""
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


def load_pair_validation(manifest: Path, *, limit_pairs: int | None = None,
                         verify_hashes: bool = True):
    """Build deterministic gallery/query metadata from an explicit night split."""
    enable_backend_imports()
    from workflow.evaluation import EvaluationImage, EvaluationSet

    manifest = Path(manifest).expanduser().resolve(strict=True)
    if limit_pairs is not None and limit_pairs < 1:
        raise ValueError("limit-pairs must be positive")
    rows = []
    seen_samples = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("condition") != "night" or row.get("passed") is not True or
                row.get("eligible_for_training") is not True or
                row.get("accepted", True) is not True):
            raise ValueError(f"Line {line_number}: validation split contains a nonaccepted night row")
        sample = row.get("sample_id")
        if not isinstance(sample, str) or not sample or sample in seen_samples:
            raise ValueError(f"Line {line_number}: missing or duplicate sample_id")
        seen_samples.add(sample)
        if row.get("city") in (None, "") or row.get("place_id") in (None, ""):
            raise ValueError(f"Line {line_number}: missing city/place_id")
        source = resolve_file(row["source_path"], manifest)
        generated = resolve_file(row["output_path"], manifest)
        if source == generated:
            raise ValueError(f"Line {line_number}: source and generated are identical")
        if verify_hashes:
            for key, image in (("source_sha256", source), ("output_sha256", generated)):
                if row.get(key) is None or sha256(image) != row[key]:
                    raise ValueError(f"Line {line_number}: missing or incorrect {key}: {image}")
        rows.append((sample, (str(row["city"]), str(row["place_id"])), source, generated))
    if not rows:
        raise ValueError(f"Empty night validation manifest: {manifest}")
    rows.sort(key=lambda value: value[0])
    if limit_pairs is not None:
        rows = rows[:limit_pairs]

    references = []
    queries = []
    source_index = {}
    source_place = {}
    query_paths = set()
    exact_source_indices = []
    places = []
    for sample, place, source, generated in rows:
        if source in source_place and source_place[source] != place:
            raise ValueError(f"Source image has inconsistent city/place_id: {source}")
        if source not in source_index:
            source_index[source] = len(references)
            source_place[source] = place
            references.append(EvaluationImage(f"day/{len(references):06d}", source))
        if generated in query_paths:
            raise ValueError(f"Generated image appears in two validation rows: {generated}")
        query_paths.add(generated)
        queries.append(EvaluationImage(sample, generated))
        exact_source_indices.append(source_index[source])
        places.append(place)
    place_to_indices = {}
    for source, index in source_index.items():
        place_to_indices.setdefault(source_place[source], []).append(index)
    positives = [place_to_indices[place] for place in places]
    dataset = EvaluationSet(
        "synthetic_night_pairs", references, queries, positives,
        {"ground_truth": "city_place_id", "pairing": "explicit_manifest_source_output",
         "manifest": str(manifest), "condition": "night", "selection": "sample_id_ascending",
         "limit_pairs": limit_pairs, "hashes_verified": verify_hashes},
    )
    dataset.validate()
    return dataset, exact_source_indices, places


def mean_paired_cosine(gallery: torch.Tensor, queries: torch.Tensor,
                       source_indices: list[int]) -> float:
    """Mean unit-descriptor cosine against each generated image's exact day source."""
    if (gallery.ndim != 2 or queries.ndim != 2 or gallery.shape[1] != queries.shape[1]
            or len(source_indices) != len(queries) or not source_indices
            or min(source_indices) < 0 or max(source_indices) >= len(gallery)):
        raise ValueError("Invalid paired descriptors or source indices")
    source = gallery[torch.as_tensor(source_indices, dtype=torch.long)]
    cosine = (F.normalize(queries.float(), dim=1) *
              F.normalize(source.float(), dim=1)).sum(1)
    if not torch.isfinite(cosine).all():
        raise ValueError("Nonfinite paired cosine")
    return float(cosine.mean())


class _NightStudent(torch.nn.Module):
    def __init__(self, net: WeatherAdapterSALAD):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.student(images, "night", return_aux=False)


def _device(value: str) -> str:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA evaluation devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA unavailable for {value}")
    return str(device)


def evaluate_pairs(*, manifest: Path, checkpoint: Path, backbone_repo: Path,
                   adapter_dir: Path, output: Path, device: str = "auto",
                   retrieval_device: str | None = None, batch_size: int = 16,
                   num_workers: int = 2, limit_pairs: int | None = None,
                   verify_hashes: bool = True, save_hard_cases: bool = False,
                   adapter_scale: float = 1.0) -> dict:
    """Evaluate one night adapter and its unadapted baseline on a held-out split."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    adapter_dir = _require_night_adapter_dir(adapter_dir)
    device = _device(device)
    retrieval_device = _device(retrieval_device or device)
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    dataset, exact_indices, places = load_pair_validation(
        manifest, limit_pairs=limit_pairs, verify_hashes=verify_hashes)
    from workflow.evaluation import build_results, exact_retrieval, extract_descriptors
    from workflow.model import input_preprocessing, load_checkpoint_model

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone_repo = Path(backbone_repo).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    checkpoint_hash = sha256(checkpoint)
    print(f"Synthetic night validation: {len(dataset.queries)} pairs, "
          f"{len(dataset.references)} unique day images, {len(set(places))} places", flush=True)
    base = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    preprocessing = input_preprocessing(base.image_size)
    metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
    if (metadata.get("weather") != "night" or metadata.get("checkpoint_sha256") != checkpoint_hash
            or metadata.get("model_config") != base.config
            or metadata.get("preprocessing") != preprocessing):
        raise ValueError("Night adapter is incompatible with checkpoint, model configuration or preprocessing")
    net = load_weather_model(base, "night", adapter_dir, metadata,
                             device=device, adapter_scale=adapter_scale)

    print("Extracting original day gallery descriptors...", flush=True)
    gallery = extract_descriptors(base, dataset.references, base.image_size,
                                  device, batch_size, num_workers)
    print("Extracting unadapted night query descriptors...", flush=True)
    baseline_queries = extract_descriptors(base, dataset.queries, base.image_size,
                                           device, batch_size, num_workers)
    print("Extracting adapted night query descriptors...", flush=True)
    adapted_queries = extract_descriptors(_NightStudent(net), dataset.queries,
                                          base.image_size, device, batch_size, num_workers)

    def score(queries):
        ranks = exact_retrieval(gallery, queries, dataset.positives, device=retrieval_device)
        result = build_results(dataset, ranks, save_hard_cases=save_hard_cases)
        result["mean_paired_cosine"] = mean_paired_cosine(gallery, queries, exact_indices)
        return result

    baseline = score(baseline_queries)
    adapted = score(adapted_queries)
    result = {
        "dataset": "synthetic_night_pairs", "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256(Path(manifest).resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "adapter_dir": str(adapter_dir),
        "adapter_metadata": metadata, "adapter_scale": adapter_scale,
        "preprocessing": preprocessing,
        "validation_pairs": len(dataset.queries), "gallery_images": len(dataset.references),
        "places": len(set(places)), "limited_subset": limit_pairs is not None,
        "baseline": baseline, "adapter": adapted,
        "delta_recall": {key: adapted["recall"][key] - baseline["recall"][key]
                         for key in ("R@1", "R@5", "R@10")},
        "delta_mean_paired_cosine": (adapted["mean_paired_cosine"] -
                                     baseline["mean_paired_cosine"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print("Baseline:", baseline["recall"], "pair cosine:", baseline["mean_paired_cosine"], flush=True)
    print("Adapter:", adapted["recall"], "pair cosine:", adapted["mean_paired_cosine"], flush=True)
    print("Delta:", result["delta_recall"], "pair cosine:", result["delta_mean_paired_cosine"], flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a held-out Qwen night/day split. The original day SALAD gallery "
                    "is shared by unadapted and adapted night queries; this is separate from real SVOX test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=project_root() / "data/splits/night_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=project_root() / "third_party/dinov2")
    parser.add_argument("--adapter-dir", type=Path,
                        help="Output directory of a trained night Adapter; required for evaluation")
    parser.add_argument("--adapter-scale", type=float, default=1.0,
                        help="LoRA update multiplier; choose only on development data")
    parser.add_argument("--output", type=Path,
                        help="Result JSON; defaults to runs/night_pairs_validation.json or a named smoke subset")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--retrieval-device", help="Defaults to --device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-pairs", type=int,
                        help="First N sample IDs for a smoke check; result is labelled as a subset")
    parser.add_argument("--skip-hash-verification", action="store_true",
                        help="Skip source/generated image SHA-256 checks")
    parser.add_argument("--save-hard-cases", action="store_true")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate the split and print counts without loading a model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check_data:
        dataset, _, places = load_pair_validation(
            args.manifest, limit_pairs=args.limit_pairs,
            verify_hashes=not args.skip_hash_verification)
        print(json.dumps({"pairs": len(dataset.queries),
                          "unique_day_images": len(dataset.references),
                          "places": len(set(places)), "protocol": dataset.protocol},
                         indent=2, ensure_ascii=False))
        return
    if args.adapter_dir is None:
        raise SystemExit("--adapter-dir is required unless --check-data is selected")
    try:
        _require_night_adapter_dir(args.adapter_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    output = args.output or (project_root() / "runs" /
                             (f"night_pairs_validation_first{args.limit_pairs}.json"
                              if args.limit_pairs else "night_pairs_validation.json"))
    evaluate_pairs(
        manifest=args.manifest, checkpoint=args.checkpoint,
        backbone_repo=args.backbone_repo, adapter_dir=args.adapter_dir,
        output=output, device=args.device, retrieval_device=args.retrieval_device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        limit_pairs=args.limit_pairs, verify_hashes=not args.skip_hash_verification,
        save_hard_cases=args.save_hard_cases, adapter_scale=args.adapter_scale,
    )


if __name__ == "__main__":
    main()

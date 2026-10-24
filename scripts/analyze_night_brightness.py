#!/usr/bin/env python3
"""Describe retrieval results by a fixed image mean-grayscale proxy.

Thresholds come only from the SVOX train diagnostic queries. They are applied
unchanged to each supplied evaluation; this script does not fit a model, choose
fusion weights, change positives, or interpret bins as physical illumination.
Only one image is opened at a time. No descriptors or model weights are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_NAMES = ("lower_mean_grayscale", "middle_mean_grayscale", "higher_mean_grayscale")
METHODS = {"baseline": "baseline", "pure_night": "night", "fusion": "fusion"}
KS = (1, 5, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation JSON must be an object: {path}")
    return value


def linear_quantile(values: list[float], probability: float) -> float:
    """Linear interpolation at (N - 1) * probability, including tied values."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("Quantile requires nonempty values and probability in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def mean_grayscale(path: Path) -> float:
    # Keep image imports out of module initialization so --help is lightweight.
    from PIL import Image, ImageStat

    with Image.open(path) as original:
        # Deliberately use the encoded image, without EXIF rotation or any
        # model-specific normalization, to keep this proxy reproducible.
        with original.convert("RGB") as rgb:
            with rgb.resize((64, 64), Image.Resampling.BILINEAR) as thumbnail:
                with thumbnail.convert("L") as gray:
                    value = float(ImageStat.Stat(gray).mean[0]) / 255.0
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid grayscale proxy for {path}")
    return value


def load_dataset(result: dict, args):
    """Load native metadata only, preserving exactly its ordered query IDs."""
    name = result.get("dataset")
    if name == "SVOX":
        from weather_vpr.inference.svox import load_night_dataset

        split = result.get("split")
        if split not in {"train", "test"}:
            raise ValueError("SVOX evaluation must explicitly identify train or test split")
        dataset = load_night_dataset(args.svox_root, split=split)
    elif name == "Tokyo247_night":
        from weather_vpr.inference.tokyo import load_tokyo_night

        dataset = load_tokyo_night(args.tokyo_root, args.tokyo_mat, args.tokyo_source)
    elif name in {"RobotCar-Seasons_v2_public_train_diagnostic_night",
                  "RobotCar-Seasons_v2_public_train_diagnostic_night-rain"}:
        from weather_vpr.inference.robotcar import load_robotcar_public_night

        condition = "night-rain" if name.endswith("_night-rain") else "night"
        dataset = load_robotcar_public_night(args.robotcar_root, condition)
    else:
        raise ValueError(f"Unsupported dataset for brightness diagnostics: {name!r}")
    baseline = result.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("protocol") != dataset.protocol:
        raise ValueError(f"Recorded protocol differs from the current native loader for {name}")
    if name != dataset.name or baseline.get("num_queries") != len(dataset.queries):
        raise ValueError(f"Recorded dataset/query count differs from the current native loader for {name}")
    return dataset


def rank_value(row, *, has_positives: bool, reference_count: int, context: str):
    if row is None:
        rank = None
    elif isinstance(row, dict) and "rank" in row:
        rank = row["rank"]
    else:
        raise ValueError(f"Expected a retrieval rank object or null at {context}")
    if not has_positives:
        if rank is not None:
            raise ValueError(f"Query without native positives has a non-null rank at {context}")
        return None
    if type(rank) is not int or not 1 <= rank <= reference_count:
        raise ValueError(f"Native evaluable query requires an integer rank in Gallery range at {context}")
    return rank


def summarize(result: dict, dataset, thresholds: tuple[float, float]) -> dict:
    rows = result.get("per_query")
    if not isinstance(rows, list) or len(rows) != len(dataset.queries):
        raise ValueError("Evaluation needs one per_query row per native query; rerun with rank export")
    if len(dataset.positives) != len(dataset.queries):
        raise ValueError("Native positives must match the ordered queries")
    # Require both pure models. Fusion may be absent in older evaluations, but
    # partial export is rejected rather than producing incomparable denominators.
    available = {"baseline": True, "pure_night": True,
                 "fusion": any(isinstance(row, dict) and "fusion" in row for row in rows)}
    buckets = []
    for name in BIN_NAMES:
        buckets.append({"bin": name, "query_count": 0, "evaluated_query_count": 0,
                        "queries_without_positives": 0, "mean_grayscale_values": [],
                        "hits": {method: {f"R@{k}": 0 for k in KS}
                                 for method, present in available.items() if present}})
    low, high = thresholds
    for image, positives, row in zip(dataset.queries, dataset.positives, rows):
        if not isinstance(row, dict) or row.get("query_id") != image.id:
            raise ValueError(f"per_query IDs/order differ from native queries at {image.id}")
        ranks = {}
        for method, key in METHODS.items():
            if available[method]:
                if key not in row:
                    raise ValueError(f"Missing {key} rank at {image.id}")
                ranks[method] = rank_value(row[key], has_positives=bool(positives),
                                          reference_count=len(dataset.references),
                                          context=f"{image.id}/{key}")
        value = mean_grayscale(image.path)
        index = 0 if value < low else 1 if value < high else 2
        bucket = buckets[index]
        bucket["query_count"] += 1
        bucket["mean_grayscale_values"].append(value)
        if not positives:
            bucket["queries_without_positives"] += 1
            continue
        bucket["evaluated_query_count"] += 1
        for method, rank in ranks.items():
            for k in KS:
                bucket["hits"][method][f"R@{k}"] += int(rank <= k)
    for bucket in buckets:
        values = bucket.pop("mean_grayscale_values")
        bucket["mean_grayscale"] = {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }
        count = bucket["evaluated_query_count"]
        hits = bucket.pop("hits")
        for method in METHODS:
            bucket[method] = ({"hits": hits[method],
                               "recall_percent": {key: 100.0 * n / count if count else None
                                                  for key, n in hits[method].items()}}
                              if available[method] else None)
    fusion_metadata = result.get("fusion")
    if fusion_metadata is not None and not isinstance(fusion_metadata, dict):
        raise ValueError("Fusion metadata must be an object or null")
    return {"dataset": dataset.name, "protocol": dataset.protocol,
            "query_count": len(dataset.queries), "available_methods": available,
            "fusion_alpha": (fusion_metadata or {}).get("alpha"), "bins": buckets}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True,
                        help="SVOX train evaluation JSON; supplies the fixed threshold source only")
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="Evaluation JSON files with ordered baseline/night/(optional fusion) per_query ranks")
    parser.add_argument("--output", type=Path, required=True,
                        help="New diagnostic JSON; existing files are never overwritten")
    parser.add_argument("--svox-root", type=Path, default=PROJECT_ROOT / "data/svox")
    parser.add_argument("--tokyo-root", type=Path, default=PROJECT_ROOT / "data/tokyo247")
    parser.add_argument("--robotcar-root", type=Path, default=PROJECT_ROOT / "data/RobotCar-Seasons")
    parser.add_argument("--tokyo-mat", type=Path,
                        default=PROJECT_ROOT / "configs/evaluation_sources/tokyo247_netvlad_v100.mat")
    parser.add_argument("--tokyo-source", type=Path,
                        default=PROJECT_ROOT / "configs/evaluation_sources/tokyo247_netvlad_v100.source.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    development = read_result(args.development)
    if development.get("dataset") != "SVOX" or development.get("split") != "train":
        raise ValueError("Threshold development must be the SVOX train diagnostic split")
    dataset = load_dataset(development, args)
    values = [mean_grayscale(image.path) for image in dataset.queries]
    if len(values) != 702:
        raise ValueError(f"Expected the fixed 702 SVOX train Night queries, found {len(values)}")
    thresholds = (linear_quantile(values, 1 / 3), linear_quantile(values, 2 / 3))
    payload = {
        "purpose": "evaluation_only_fixed_image_mean_grayscale_diagnostic",
        "proxy": {"definition": "PIL RGB -> bilinear 64x64 -> L mean / 255",
                  "range": [0, 1], "exif_transpose": False,
                  "limitations": ("Image brightness depends on exposure, processing, and scene contents. "
                                  "Bins are not semantic twilight/streetlight/darkness labels or measured lux.")},
        "threshold_source": {"evaluation": str(args.development.resolve()),
                             "evaluation_sha256": sha256(args.development),
                             "dataset": "SVOX", "split": "train", "query_count": len(values),
                             "probabilities": [1 / 3, 2 / 3],
                             "quantile": "linear interpolation at (N-1)*p",
                             "thresholds": list(thresholds)},
        "boundaries": ["value < lower_threshold", "lower_threshold <= value < upper_threshold",
                       "upper_threshold <= value"],
        "selection": "No training, fusion-alpha selection, test adaptation, or ground-truth changes",
        "results": [],
    }
    # Release the development metadata before loading another native dataset.
    del dataset, development, values
    for path in args.results:
        result = read_result(path)
        dataset = load_dataset(result, args)
        diagnostic = summarize(result, dataset, thresholds)
        diagnostic["evaluation"] = str(path.resolve())
        diagnostic["evaluation_sha256"] = sha256(path)
        payload["results"].append(diagnostic)
        print(f"Analyzed {dataset.name}: {len(dataset.queries)} queries", flush=True)
        del result, dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify and vendor a teacher-matched Stage 2B patch calibration.

The historical calibration's file paths are provenance, not runtime inputs.
This script checks the checkpoint and preprocessing contract, recovers every
calibration place from that provenance, and writes an exclusion list for the
training loader. No image or model inference is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE = (ROOT / "third_party/salad/outputs/ccr_followup_dino_am_fullreal"
                     / "calibration.json")
PACKAGED_SOURCE = ROOT / "configs/dino_salad_stage2b_calibration.json"
DEFAULT_SOURCE = HISTORICAL_SOURCE if HISTORICAL_SOURCE.is_file() else PACKAGED_SOURCE
PREPROCESSING = {
    "version": "rgb_bilinear_imagenet_v1",
    "image_size": [224, 224],
    "color_space": "RGB",
    "input_range": [0.0, 1.0],
    "resize": "PIL_bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
EXPECTED_MODEL = {
    "backbone_arch": "dinov2_vitb14",
    "agg_arch": "SALAD",
    "image_size": [224, 224],
    "agg_config": {
        "num_channels": 768, "num_clusters": 64,
        "cluster_dim": 128, "token_dim": 256,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def place_from_gsv_filename(value: str) -> tuple[str, str]:
    """Read the exact GSV city/place identity; ignore camera/date/view suffix."""
    name = Path(value).name
    fields = name.split("_")
    if len(fields) < 3 or not fields[0] or not fields[1].isdigit():
        raise ValueError(f"Cannot read GSV city/place from calibration source: {value}")
    return fields[0], str(int(fields[1]))


def load_calibration(path: Path, checkpoint_hash: str) -> tuple[dict, list[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected one calibration object, not an array")
    teacher = payload.get("provenance", {}).get("teacher", {})
    if teacher.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Calibration teacher SHA256 differs from the selected checkpoint")
    if teacher.get("image_preprocessing") != PREPROCESSING:
        raise ValueError("Calibration preprocessing differs from 224x224 RGB bilinear/ImageNet")
    model = teacher.get("model_config", {})
    for key, value in EXPECTED_MODEL.items():
        if key == "agg_config":
            if any(model.get(key, {}).get(k) != v for k, v in value.items()):
                raise ValueError("Calibration SALAD dimensions differ from the original checkpoint")
        elif model.get(key) != value:
            raise ValueError(f"Calibration model {key} differs from the original checkpoint")
    if payload.get("search_radius") != 1:
        raise ValueError("Stage 2B A-only correspondence requires radius=1 calibration")
    probability = payload.get("tail_probability")
    samples = payload.get("samples")
    if (not isinstance(probability, (int, float)) or not 0 < probability < 1
            or not isinstance(samples, int) or samples < math.ceil(1 / probability)):
        raise ValueError("Calibration sample count/tail probability is invalid")
    for key, low, high in (("cosine_min", -1, 1), ("margin_min", 0, 2),
                           ("structure_max", 0, 2)):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid calibration {key}")
    null_pairs = payload.get("provenance", {}).get("null_pairs")
    if (not isinstance(null_pairs, list) or len(null_pairs) < 2
            or any(not isinstance(row, list) or len(row) != 2
                   or not all(isinstance(value, str) and value for value in row)
                   for row in null_pairs)):
        raise ValueError("Calibration lacks explicit mismatched-place null pairs")
    sources = [row[0] for row in null_pairs]
    outputs = [row[1] for row in null_pairs]
    if len(set(sources)) != len(sources) or len(set(outputs)) != len(outputs):
        raise ValueError("Calibration null pairs reuse a source or generated image")
    places = sorted({place_from_gsv_filename(source) for source in sources})
    if len(places) != len(sources):
        raise ValueError("Calibration sources are not from distinct places")
    return payload, places


def verify_pair_inventory(path: Path, calibration: dict,
                          places: list[tuple[str, str]]) -> str | None:
    """When the historical selected-pair inventory exists, check null permutation."""
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(places):
        raise ValueError("Selected calibration pair inventory has the wrong size")
    seen = {(str(row.get("city")), str(row.get("place_id"))) for row in rows}
    if seen != set(places):
        raise ValueError("Selected calibration places differ from null-pair provenance")
    genuine = {(row.get("source_path"), row.get("output_path")) for row in rows}
    null = {tuple(row) for row in calibration["provenance"]["null_pairs"]}
    if (len(genuine) != len(rows) or len(null) != len(rows)
            or {a for a, _ in genuine} != {a for a, _ in null}
            or {b for _, b in genuine} != {b for _, b in null}
            or genuine & null):
        raise ValueError("Null pairs must permute selected outputs across different places")
    if any(row.get("passed") is not True or row.get("eligible_for_training") is not True
           for row in rows):
        raise ValueError("Calibration inventory includes a rejected generation")
    return sha256(path)


def inspect_training(day_folder: Path, manifest: Path | None,
                     reserved: set[tuple[str, str]]) -> dict:
    """Report overlap; the output exclusion file removes it at train time."""
    day_files = sorted(day_folder.glob("*.jpg")) if day_folder.is_dir() else []
    day_places = {place_from_gsv_filename(path.name) for path in day_files}
    report = {"day_images": len(day_files),
              "day_overlap": [list(p) for p in sorted(day_places & reserved)]}
    if manifest is not None and manifest.is_file():
        places = set()
        rows = 0
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "city" not in row or "place_id" not in row:
                raise ValueError(f"Training manifest line {line_number} lacks city/place_id")
            place = str(row["city"]), str(row["place_id"])
            if row.get("source_path") and place_from_gsv_filename(row["source_path"]) != place:
                raise ValueError(f"Training manifest line {line_number} source/place mismatch")
            places.add(place)
            rows += 1
        report.update(manifest_rows=rows,
                      manifest_overlap=[list(p) for p in sorted(places & reserved)])
    return report


def write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different content: {path}")
        return
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-calibration", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/dino_salad.ckpt")
    parser.add_argument("--calibration-pairs", type=Path,
                        help="Optional historical selected-pair inventory for null-pair checks")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/train/manifest.jsonl")
    parser.add_argument("--train-day-folder", type=Path, default=ROOT / "data/train/day/night")
    parser.add_argument("--output-calibration", type=Path,
                        default=ROOT / "configs/dino_salad_stage2b_calibration.json")
    parser.add_argument("--output-exclusions", type=Path,
                        default=ROOT / "configs/dino_salad_calibration_excluded_places.json")
    parser.add_argument("--output-provenance", type=Path,
                        default=ROOT / "configs/dino_salad_calibration_provenance.json")
    args = parser.parse_args()
    source = args.source_calibration.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    pair_inventory = (args.calibration_pairs if args.calibration_pairs is not None
                      else source.with_name("calibration_pairs.json"))
    checkpoint_hash = sha256(checkpoint)
    calibration, places = load_calibration(source, checkpoint_hash)
    inventory_hash = verify_pair_inventory(pair_inventory, calibration, places)
    calibration_hash = sha256(source)
    receipt = source.with_name("calibration.receipt.json")
    if receipt.is_file() and json.loads(receipt.read_text()).get("sha256") != calibration_hash:
        raise ValueError("Historical calibration receipt SHA256 does not match")
    reserved = set(places)
    overlap = inspect_training(args.train_day_folder, args.train_manifest, reserved)
    exclusions = [list(place) for place in places]
    provenance = {
        "format_version": 1,
        "calibration_sha256": calibration_hash,
        "checkpoint_sha256": checkpoint_hash,
        "image_preprocessing": PREPROCESSING,
        "search_radius": 1,
        "null_pair_count": len(calibration["provenance"]["null_pairs"]),
        "selected_pair_inventory_sha256": inventory_hash,
        "calibration_places": exclusions,
        "separation_rule": "Pass output_exclusions to --exclude-places when training; exclude every image from these city/place identities.",
        "historical_paths_are_provenance_only": True,
    }
    # A migrated project may retain only the vendored calibration. Preserve a
    # previously verified inventory digest if that old inventory is unavailable.
    if inventory_hash is None and args.output_provenance.is_file():
        existing = json.loads(args.output_provenance.read_text(encoding="utf-8"))
        stable_keys = ("format_version", "calibration_sha256", "checkpoint_sha256",
                       "image_preprocessing", "search_radius", "null_pair_count",
                       "calibration_places", "separation_rule",
                       "historical_paths_are_provenance_only")
        if any(existing.get(key) != provenance[key] for key in stable_keys):
            raise ValueError("Existing calibration provenance differs from vendored calibration")
        provenance["selected_pair_inventory_sha256"] = existing.get(
            "selected_pair_inventory_sha256")
    write_exact(args.output_calibration, source.read_bytes())
    write_exact(args.output_exclusions,
                (json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n").encode())
    write_exact(args.output_provenance,
                (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"calibration": str(args.output_calibration),
                      "exclusions": str(args.output_exclusions),
                      "provenance": str(args.output_provenance),
                      "reserved_places": len(places), "training_overlap_before_exclusion": overlap},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

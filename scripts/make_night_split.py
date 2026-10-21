#!/usr/bin/env python3
"""Make a reproducible place-disjoint train/validation split for night pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def place_key(value: object) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [city, place_id], got {value!r}")
    city, place_id = value
    if not isinstance(city, str) or not city or place_id is None or str(place_id) == "":
        raise ValueError(f"Invalid place: {value!r}")
    return city, str(place_id)


def load_exclusions(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, list):
        places = content
    elif isinstance(content, dict) and isinstance(content.get("groups"), dict):
        groups = content["groups"]
        unexpected = set(groups) - {"validation", "calibration"}
        if unexpected:
            raise ValueError(f"Unknown exclusion groups in {path}: {sorted(unexpected)}")
        places = [value for name in ("validation", "calibration")
                  for value in groups.get(name, [])]
    else:
        raise ValueError(f"Exclusions must be a place list or reserved groups JSON: {path}")
    return {place_key(value) for value in places}


def resolve_image(value: str, manifest: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = [manifest.parent / path, project_root / path]
    found = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(found) != 1:
        raise ValueError(f"Missing or ambiguous image path {value!r} in {manifest}")
    return found.pop()


def load_pairs(manifest: Path, project_root: Path) -> list[dict]:
    rows = []
    seen_ids = set()
    seen_sources = set()
    seen_outputs = set()
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") != "night":
                continue
            if row.get("passed") is not True or row.get("eligible_for_training") is not True:
                continue
            if "city" not in row or "place_id" not in row:
                raise ValueError(f"Missing city/place_id at {manifest}:{line_number}")
            key = place_key([row["city"], row["place_id"]])
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Missing or duplicate sample_id at {manifest}:{line_number}")
            source = resolve_image(row["source_path"], manifest, project_root)
            output = resolve_image(row["output_path"], manifest, project_root)
            if source == output or source in seen_sources or output in seen_outputs:
                raise ValueError(f"Repeated or identical image at {manifest}:{line_number}")
            rows.append({**row, "_place": key, "_source": source, "_output": output})
            seen_ids.add(sample_id)
            seen_sources.add(source)
            seen_outputs.add(output)
    if not rows:
        raise ValueError(f"No accepted night pairs in {manifest}")
    return rows


def rewrite_rows(rows: list[dict], target: Path) -> list[dict]:
    result = []
    for row in rows:
        item = {key: value for key, value in row.items() if not key.startswith("_")}
        item["source_path"] = Path(os.path.relpath(row["_source"], target.parent)).as_posix()
        item["output_path"] = Path(os.path.relpath(row["_output"], target.parent)).as_posix()
        result.append(item)
    return result


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def create_split(manifest: Path, output_dir: Path, exclusions_path: Path | None,
                 val_fraction: float, seed: int, project_root: Path) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    manifest = manifest.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    exclusions_path = exclusions_path.expanduser().resolve(strict=True) if exclusions_path else None
    train_path = output_dir / "night_train.jsonl"
    val_path = output_dir / "night_val.jsonl"
    metadata_path = output_dir / "night_split.json"
    heldout_path = output_dir / "heldout_places.json"
    output_paths = {train_path, val_path, metadata_path, heldout_path}
    if manifest in output_paths or exclusions_path in output_paths:
        raise ValueError("Input manifest/exclusions cannot be overwritten by split outputs")

    rows = load_pairs(manifest, project_root)
    externally_excluded = load_exclusions(exclusions_path)
    eligible = [row for row in rows if row["_place"] not in externally_excluded]
    if len(eligible) < 2:
        raise ValueError("Fewer than two eligible night pairs remain after exclusions")
    eligible_places = {row["_place"] for row in eligible}
    places_by_city: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in eligible:
        places_by_city[row["_place"][0]].add(row["_place"])

    validation_places: set[tuple[str, str]] = set()
    for city in sorted(places_by_city):
        places = sorted(places_by_city[city])
        if len(places) < 2:
            continue  # Keep singleton cities in training.
        count = min(len(places) - 1, max(1, round(len(places) * val_fraction)))
        rng = random.Random(f"{seed}:{city}")
        validation_places.update(rng.sample(places, count))
    train_rows = [row for row in eligible if row["_place"] not in validation_places]
    val_rows = [row for row in eligible if row["_place"] in validation_places]
    if not train_rows or not val_rows:
        raise ValueError("Split produced an empty train or validation set")
    train_places = {row["_place"] for row in train_rows}
    if train_places & validation_places or train_places & externally_excluded:
        raise AssertionError("Place isolation failed")

    heldout = sorted(externally_excluded | validation_places)
    input_places = {row["_place"] for row in rows}
    details = {
        "version": 1,
        "weather": "night",
        "seed": seed,
        "validation_fraction_requested": val_fraction,
        "input_manifest": Path(os.path.relpath(manifest, output_dir)).as_posix(),
        "input_manifest_sha256": sha256(manifest),
        "external_exclusions": (Path(os.path.relpath(exclusions_path, output_dir)).as_posix()
                                if exclusions_path else None),
        "external_exclusions_sha256": sha256(exclusions_path) if exclusions_path else None,
        "input_pairs": len(rows),
        "input_places": len(input_places),
        "external_exclusion_places": len(externally_excluded),
        "external_exclusion_places_in_input": len(input_places & externally_excluded),
        "excluded_pairs": len(rows) - len(eligible),
        "train_pairs": len(train_rows),
        "train_places": len(train_places),
        "validation_pairs": len(val_rows),
        "validation_places": len(validation_places),
        "validation_fraction_of_eligible_pairs": len(val_rows) / len(eligible),
        "validation_fraction_of_eligible_places": len(validation_places) / len(eligible_places),
        "heldout_places_file": heldout_path.name,
        "heldout_places_count": len(heldout),
    }
    # Resolve paths relative to each output manifest, then write all artifacts.
    train_content = jsonl(rewrite_rows(train_rows, train_path))
    val_content = jsonl(rewrite_rows(val_rows, val_path))
    heldout_content = json.dumps([list(place) for place in heldout], ensure_ascii=False, indent=2) + "\n"
    metadata_content = json.dumps(details, ensure_ascii=False, indent=2) + "\n"
    write_atomic(train_path, train_content)
    write_atomic(val_path, val_content)
    write_atomic(heldout_path, heldout_content)
    write_atomic(metadata_path, metadata_content)
    return details


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_exclusions = project_root / "third_party/salad/configs/splits/ccr_reserved_places.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=project_root / "data/train/manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data/splits")
    parser.add_argument("--exclude-places", type=Path,
                        help="JSON list of [city, place_id], or reserved validation/calibration groups")
    parser.add_argument("--no-default-exclusions", action="store_true",
                        help="Ignore the bundled reserved-place file when --exclude-places is omitted")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    exclusions = args.exclude_places
    if exclusions is None and not args.no_default_exclusions and default_exclusions.is_file():
        exclusions = default_exclusions
    result = create_split(args.manifest, args.output_dir, exclusions,
                          args.val_fraction, args.seed, project_root)
    print(f"Night split: {result['train_pairs']} train pairs / {result['validation_pairs']} validation pairs; "
          f"{result['excluded_pairs']} externally excluded pairs. Files: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

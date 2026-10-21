#!/usr/bin/env python3
"""Build a portable manifest from the reviewed Qwen generation metadata.

Only exact sample_id -> weather filename and source basename -> day filename
matches are accepted. The generated manifest contains no WORKSPACE paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise ValueError(f"No images in {directory}")
    by_name = {p.name: p for p in files}
    if len(by_name) != len(files):
        raise ValueError(f"Duplicate image filenames in {directory}")
    return by_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail_if_changed(path: Path, expected: str | None, line_number: int) -> None:
    if not expected:
        raise ValueError(f"Metadata line {line_number} has no SHA-256 for {path.name}")
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch on metadata line {line_number}: {path}")


def relative_path(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def build_manifest(metadata: Path, train_root: Path, output: Path, verify_hashes: bool) -> int:
    metadata = metadata.expanduser().resolve(strict=True)
    train_root = train_root.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output == metadata:
        raise ValueError("Output must differ from metadata input")
    day_files = image_files(train_root / "day" / "night")
    weather_files = image_files(train_root / "weather" / "night")

    rows: list[dict] = []
    used_day: set[str] = set()
    used_weather: set[str] = set()
    with metadata.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("condition") != "night":
                continue
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or Path(sample_id).name != sample_id:
                raise ValueError(f"Invalid sample_id on metadata line {line_number}")
            weather_name = f"{sample_id}.png"
            if weather_name not in weather_files:
                continue  # The local directory may contain a reviewed subset.
            if weather_name in used_weather:
                raise ValueError(f"Ambiguous sample_id on metadata line {line_number}: {sample_id}")
            if not all(record.get(key) is True for key in ("passed", "eligible_for_training", "accepted")):
                raise ValueError(f"Local weather image is not accepted on metadata line {line_number}: {weather_name}")
            source = record.get("source_path")
            if not isinstance(source, str) or not source:
                raise ValueError(f"Missing source_path on metadata line {line_number}")
            day_name = Path(source).name
            if day_name not in day_files:
                raise ValueError(f"Missing day image on metadata line {line_number}: {day_name}")
            if day_name in used_day:
                raise ValueError(f"Ambiguous day image on metadata line {line_number}: {day_name}")
            if not record.get("city") or record.get("place_id") is None:
                raise ValueError(f"Missing city/place_id on metadata line {line_number}")
            day_path = day_files[day_name]
            weather_path = weather_files[weather_name]
            if verify_hashes:
                fail_if_changed(day_path, record.get("source_sha256"), line_number)
                fail_if_changed(weather_path, record.get("output_sha256"), line_number)
            rows.append({
                "sample_id": sample_id,
                "city": record["city"],
                "place_id": record["place_id"],
                "condition": "night",
                "passed": True,
                "eligible_for_training": True,
                "accepted": True,
                "source_path": relative_path(day_path, output),
                "output_path": relative_path(weather_path, output),
                "source_sha256": record.get("source_sha256"),
                "output_sha256": record.get("output_sha256"),
            })
            used_day.add(day_name)
            used_weather.add(weather_name)

    missing_weather = sorted(weather_files.keys() - used_weather)
    missing_day = sorted(day_files.keys() - used_day)
    if missing_weather or missing_day:
        raise ValueError(
            "Incomplete pairing: "
            f"{len(missing_weather)} unmatched weather images {missing_weather[:3]}, "
            f"{len(missing_day)} unmatched day images {missing_day[:3]}"
        )
    rows.sort(key=lambda row: row["sample_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Finish validation before replacing an existing manifest.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    return len(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Original Qwen images/metadata.jsonl (read once; not needed for training)")
    parser.add_argument("--train-root", type=Path, default=project_root / "data" / "train")
    parser.add_argument("--output", type=Path,
                        help="Manifest path (default: TRAIN_ROOT/manifest.jsonl)")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="Check every local pair against SHA-256 values in the metadata")
    args = parser.parse_args()
    output = args.output or args.train_root / "manifest.jsonl"
    count = build_manifest(args.metadata, args.train_root, output, args.verify_hashes)
    print(f"Wrote {count} exact night pairs to {output.resolve()}")


if __name__ == "__main__":
    main()

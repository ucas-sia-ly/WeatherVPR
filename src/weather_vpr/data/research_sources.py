"""Train-only research inventories, without model or tensor dependencies.

Annotation membership establishes labels/splits. File names alone never establish
geographic places, and exact-content groups never certify route independence.
Builders write metadata only and never move, flatten, or alter source datasets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile

from weather_vpr.data.real_pairs import (
    _check_split_isolation, _load_review, _validate_review, file_sha256,
    load_review_source,
)


SOURCE_NAMES = frozenset({"qwen_night", "source_night", "darkdriving_train_night",
                          "bdd_train_night", "nightcity_train"})
GSV_FIELDS = ("place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def portable_path(path: Path | str, project_root: Path | str) -> str:
    path, root = Path(path).absolute(), Path(project_root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def checked_file(path: Path | str, *, within: Path | None = None) -> Path:
    """Require real files and forbid symlink traversal/escape from a dataset."""
    path = Path(path).expanduser().absolute()
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"Symlinked research source is forbidden: {current}")
        current = current.parent
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Not a source file: {path}")
    if within is not None and not resolved.is_relative_to(within.resolve()):
        raise ValueError(f"Source escapes dataset root: {path}")
    return resolved


def _resolve(value: str, manifest: Path, root: Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [manifest.parent / path, root / path]
    found = {checked_file(candidate) for candidate in candidates if candidate.is_file()}
    if len(found) != 1:
        raise ValueError(f"Missing/ambiguous source path {value!r} in {manifest}")
    return found.pop()


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{number}")
                rows.append(row)
    return rows


def load_source_manifest(path: Path | str, project_root: Path | str,
                         allowed_sources: Iterable[str] | None = None) -> list[dict]:
    """Read the common diagnostic schema, rejecting any non-training record.

    Returned paths are absolute. Optional SHA values describe expected bytes;
    consumers must verify them before extracting/cache reuse. ``source_night``
    points at clean Day and requires the existing Source Night augmentation.
    """
    root, path = Path(project_root).resolve(), checked_file(path)
    allowed = set(SOURCE_NAMES if allowed_sources is None else allowed_sources)
    result, seen = [], set()
    for row in _jsonl(path):
        if row.get("split") != "train":
            raise ValueError("Research source manifests must be train-only")
        if row.get("source") not in allowed:
            raise ValueError(f"Unknown/disallowed research source: {row.get('source')}")
        for key in ("sample_id", "path", "group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing/invalid {key} in research source")
        identity = row["source"], row["sample_id"]
        if identity in seen:
            raise ValueError(f"Duplicate source/sample_id: {identity}")
        seen.add(identity)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Research source metadata must be an object")
        digest = row.get("sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Invalid image SHA256")
        result.append({**row, "path": str(_resolve(row["path"], path, root))})
    if not result:
        raise ValueError("Empty research source manifest")
    return result


def _inspect_image(path: Path, expected_sha: str | None = None) -> str:
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("Research inventories require strict image decoding")
    digest = file_sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Image SHA256 changed: {path}")
    with Image.open(path) as image:
        image.load()
        image.convert("RGB")
    return digest


def _row(source: str, sample_id: str, path: Path, group: str, root: Path,
         metadata: dict, sha256: str | None = None) -> dict:
    result = {"schema_version": 1, "kind": "domain_diagnostic_train",
              "sample_id": sample_id, "source": source, "split": "train",
              "path": portable_path(path, root), "group_id": group,
              "metadata": metadata}
    if sha256:
        result["sha256"] = sha256
    return result


def collect_exclusions(project_root: Path | str, extra_files=()) -> tuple[set, dict]:
    """Union all local heldout/validation/calibration identities before sampling."""
    root = Path(project_root).resolve()
    paths = set(Path(path).resolve() for path in extra_files)
    for pattern in ("data/splits/*heldout*.json", "data/splits/*excluded*.json",
                    "data/splits/*val*.jsonl", "data/splits/*dev*.jsonl",
                    "data/splits/*test*.jsonl", "data/splits/*calibration*.json*",
                    "configs/*calibration*excluded*places*.json"):
        paths.update(root.glob(pattern))
    reserved = root / "third_party/salad/configs/splits/ccr_reserved_places.json"
    if reserved.is_file():
        paths.add(reserved)
    places, evidence = set(), {}
    for path in sorted(paths):
        path = checked_file(path)
        evidence[portable_path(path, root)] = file_sha256(path)
        if path.suffix == ".jsonl":
            content = [(row["city"], row["place_id"]) for row in _jsonl(path)
                       if "city" in row and "place_id" in row]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                content = value
            elif isinstance(value, dict) and isinstance(value.get("groups"), dict):
                content = [pair for group in value["groups"].values() for pair in group]
            elif isinstance(value, dict) and isinstance(value.get("calibration_places"), list):
                content = value["calibration_places"]
            else:
                raise ValueError(f"Unrecognized place exclusion schema: {path}")
        for place in content:
            if not isinstance(place, (list, tuple)) or len(place) != 2:
                raise ValueError(f"Invalid heldout place in {path}: {place!r}")
            places.add((str(place[0]), str(int(place[1]))))
    return places, evidence


def accepted_qwen_sources(project_root: Path | str, manifest: Path | str,
                          *, verify_images=False, excluded_places=()) -> tuple[list, dict]:
    root, manifest = Path(project_root).resolve(), checked_file(manifest)
    annotation_sha = file_sha256(manifest)
    excluded = set(excluded_places)
    rows, seen, places = [], set(), set()
    for original in _jsonl(manifest):
        if (original.get("condition", "").lower() != "night"
                or original.get("passed") is not True
                or original.get("eligible_for_training") is not True
                or original.get("accepted", True) is not True):
            continue
        name = original["sample_id"]
        place = str(original["city"]), str(int(original["place_id"]))
        if place in excluded:
            raise ValueError(f"Training Qwen manifest contains excluded place: {place}")
        if name in seen:
            raise ValueError(f"Duplicate accepted Qwen sample: {name}")
        original_sample_index = len(seen)
        seen.add(name)
        places.add(place)
        day = _resolve(original["source_path"], manifest, root)
        night = _resolve(original["output_path"], manifest, root)
        if day == night:
            raise ValueError("Qwen Day and Night paths are identical")
        base = {"city": place[0], "place_id": place[1],
                "grouping_kind": "verified_gsv_place",
                "place_label_verified": True,
                "geographic_independence_known": False,
                "annotation": portable_path(manifest, root),
                "annotation_sha256": annotation_sha,
                "original_sample_index": original_sample_index,
                "paired_day_path": portable_path(day, root)}
        for source, path, hash_key in (("qwen_night", night, "output_sha256"),
                                        ("source_night", day, "source_sha256")):
            digest = original.get(hash_key)
            if verify_images:
                digest = _inspect_image(path, digest)
            metadata = {**base, "transform": "source_night" if source == "source_night"
                        else "canonical_qwen_night", "image_hash_verified": bool(verify_images)}
            rows.append(_row(source, name, path, f"gsv:{place[0]}:{place[1]}",
                             root, metadata, digest))
    if not seen:
        raise ValueError("No accepted training Qwen rows")
    return rows, {"accepted_pairs": len(seen), "unique_places": len(places),
                  "annotation": portable_path(manifest, root),
                  "annotation_sha256": annotation_sha}


def darkdriving_train_sources(project_root: Path | str, dataset_root: Path | str,
                              review_manifest: Path | str,
                              *, verify_images=False) -> tuple[list, dict]:
    root = Path(project_root).resolve()
    data_root, source_rows, source_hash = load_review_source(dataset_root)
    review = checked_file(review_manifest)
    reviews, review_hash = _load_review(review)
    sources = {row["sample_id"]: row for row in source_rows}
    seen = set()
    for item in reviews:
        name = item["sample_id"]
        if name not in sources or name in seen:
            raise ValueError(f"Unknown/duplicate DarkDriving review sample: {name}")
        seen.add(name)
        _validate_review(item, sources[name], source_hash)
    _check_split_isolation(source_rows, reviews)
    rows, accepted, review_counts, split_counts = [], [], Counter(), Counter()
    for item in reviews:
        review_counts[item["review_status"]] += 1
        split_counts[item["split"] or "unassigned"] += 1
        if (item["review_status"] == "unreviewed" or item["pair_valid"] is not True
                or item["visibility_valid"] is not True):
            continue
        accepted.append(item)
        if item["split"] != "train":
            continue
        source = sources[item["sample_id"]]
        path = checked_file(data_root / source["night_path"], within=data_root)
        digest = source["night_sha256"]
        if verify_images:
            for side in ("day", "night"):
                _inspect_image(checked_file(data_root / source[f"{side}_path"], within=data_root),
                               source[f"{side}_sha256"])
        metadata = {"annotation": portable_path(data_root / "manifest.jsonl", root),
                    "annotation_sha256": source_hash,
                    "review_manifest": portable_path(review, root),
                    "review_manifest_sha256": review_hash,
                    "review_status": item["review_status"],
                    "scene_group": item["scene_group"],
                    "sequence_id": source["sequence_id"],
                    "grouping_kind": "exact_content_component",
                    "geographic_independence_known": False,
                    "route_independence_certified": False,
                    "image_hash_verified": bool(verify_images),
                    "paired_day_path": portable_path(data_root / source["day_path"], root)}
        rows.append(_row("darkdriving_train_night", source["sample_id"], path,
                         "darkdriving_content:" + source["content_group"], root, metadata, digest))
    report = {"source_pairs": len(source_rows), "reviewed_records": len(reviews),
              "unrecorded_source_pairs": len(source_rows) - len(reviews),
              "review_status_counts": dict(sorted(review_counts.items())),
              "split_counts": dict(sorted(split_counts.items())),
              "accepted_all_splits": len(accepted),
              "accepted_train": len(rows),
              "accepted_status_counts": dict(sorted(Counter(
                  item["review_status"] for item in accepted).items())),
              "exact_content_groups": len({row["content_group"] for row in source_rows}),
              "sequence_counts": dict(sorted(Counter(row["sequence_id"] for row in source_rows).items())),
              "train_scene_groups": dict(sorted(Counter(row["metadata"]["scene_group"] for row in rows).items())),
              "unique_day_file_hashes": len({row["day_sha256"] for row in source_rows}),
              "unique_night_file_hashes": len({row["night_sha256"] for row in source_rows}),
              "unique_day_pixel_hashes": len({row["day_pixel_sha256"] for row in source_rows}),
              "unique_night_pixel_hashes": len({row["night_pixel_sha256"] for row in source_rows}),
              "source_manifest_sha256": source_hash, "review_manifest_sha256": review_hash,
              "route_independence_certified": False, "geographic_places_known": False,
              "near_duplicate_groups_verified": False,
              "limitation": "Exact content and filename sequence are not geographic/route labels; automatic acceptance is not manual review."}
    if not rows:
        raise ValueError("No accepted, source-bound DarkDriving training rows")
    return rows, report


def iter_json_array(path: Path | str, chunk_size=1024 * 1024):
    """Stream a large top-level JSON array without loading annotation objects together."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer, ended, started, need_value, any_values = "", False, False, True, False
        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer and not ended:
                    block = stream.read(chunk_size)
                    buffer += block
                    ended = not block
                    continue
                if not buffer.startswith("["):
                    raise ValueError(f"Expected top-level JSON array: {path}")
                buffer, started = buffer[1:], True
                continue
            if not buffer and not ended:
                block = stream.read(chunk_size)
                buffer += block
                ended = not block
                continue
            if buffer.startswith("]"):
                if need_value and any_values:
                    raise ValueError("Trailing comma in JSON annotation array")
                remainder = buffer[1:] + stream.read()
                if remainder.strip():
                    raise ValueError("Trailing data after JSON array")
                return
            if not need_value:
                if not buffer.startswith(","):
                    raise ValueError("Expected comma in JSON annotation array")
                buffer, need_value = buffer[1:], True
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as error:
                if ended:
                    raise ValueError(f"Invalid/truncated JSON annotation array: {path}") from error
                block = stream.read(chunk_size)
                buffer += block
                ended = not block
                continue
            if end == len(buffer) and not ended:
                block = stream.read(chunk_size)
                buffer += block
                ended = not block
                continue
            yield value
            buffer, need_value, any_values = buffer[end:], False, True


def bdd_train_sources(project_root: Path | str, dataset_root: Path | str,
                      *, verify_images=False) -> tuple[list, dict]:
    """Intersect local train image names with ORIGINAL official train-night labels.

    Inner ``testA/testB`` are local packaging, not authoritative BDD splits.
    They may enter only when the ORIGINAL train annotation explicitly names them.
    No validation/test annotation or image root is used.
    """
    root, data_root = Path(project_root).resolve(), Path(dataset_root).resolve()
    annotation = checked_file(data_root / "labels/bdd100k_labels_images_train.json", within=data_root)
    annotation_hash = file_sha256(annotation)
    image_root = data_root / "images/train"
    image_index = defaultdict(list)
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[path.name].append(checked_file(path, within=image_root))
    rows, names, official_night, missing = [], set(), 0, 0
    for item in iter_json_array(annotation):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Invalid official BDD annotation record")
        name = item["name"]
        if Path(name).name != name or name in names:
            raise ValueError(f"Invalid/duplicate official BDD image name: {name}")
        names.add(name)
        attributes = item.get("attributes", {})
        if attributes.get("timeofday") != "night":
            continue
        official_night += 1
        candidates = image_index.get(name, [])
        if not candidates:
            missing += 1
            continue
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous local BDD train image: {name}")
        path = candidates[0]
        digest = _inspect_image(path) if verify_images else None
        rows.append(_row("bdd_train_night", name, path, "bdd_image:" + name, root,
                         {"annotation": portable_path(annotation, root),
                          "annotation_sha256": annotation_hash, "official_split": "train",
                          "timeofday": "night", "attributes": attributes,
                          "local_container": path.relative_to(image_root).parts[0],
                          "grouping_kind": "image_identity_geography_unknown",
                          "geographic_independence_known": False,
                          "image_hash_verified": bool(verify_images)}, digest))
    return sorted(rows, key=lambda row: row["sample_id"]), {
        "official_train_records": len(names), "official_train_night": official_night,
        "local_official_train_night": len(rows), "missing_train_night_images": missing,
        "local_image_count": sum(map(len, image_index.values())),
        "annotation": portable_path(annotation, root), "annotation_sha256": annotation_hash,
        "container_names_are_not_splits": True,
        "geographic_independence_known": False}


def nightcity_train_sources(project_root: Path | str, dataset_root: Path | str,
                            *, verify_images=False) -> tuple[list, dict]:
    root, data_root = Path(project_root).resolve(), Path(dataset_root).resolve()
    images = data_root / "raw/NightCity/NightCity-images/images/train"
    labels = data_root / "raw/NightCity/NightCity-label/label/train"
    metadata_files = [data_root / name for name in ("README.md", "dataset_infos.json", "metafile.yaml")]
    provenance = {portable_path(checked_file(path, within=data_root), root): file_sha256(path)
                  for path in metadata_files if path.is_file()}
    if not images.is_dir() or not labels.is_dir() or not provenance:
        raise ValueError("NightCity requires original train image/label directories and provenance")
    # Explicit train annotation membership is stronger than repository packaging metadata.
    annotation_names = {}
    for path in sorted(labels.glob("*_labelIds.png")):
        path = checked_file(path, within=labels)
        name = path.name.removesuffix("_labelIds.png")
        if name in annotation_names:
            raise ValueError(f"Duplicate NightCity training label: {name}")
        annotation_names[name] = path
    if not annotation_names:
        raise ValueError("No official NightCity training labels")
    # Bind split membership cheaply to all label identities; hashes of selected labels
    # are included so later changes invalidate the diagnostic cache provenance.
    label_inventory_hash = hashlib.sha256("\n".join(sorted(annotation_names)).encode()).hexdigest()
    rows, missing = [], 0
    for name, label in sorted(annotation_names.items()):
        path = images / f"{name}.png"
        if not path.is_file():
            missing += 1
            continue
        path = checked_file(path, within=images)
        digest = _inspect_image(path) if verify_images else None
        rows.append(_row("nightcity_train", name, path, "nightcity_image:" + name, root,
                         {"annotation": portable_path(label, root),
                          "annotation_sha256": file_sha256(label),
                          "train_label_identity_sha256": label_inventory_hash,
                          "official_split": "train", "dataset_metadata": provenance,
                          "grouping_kind": "image_identity_geography_unknown",
                          "geographic_independence_known": False,
                          "image_hash_verified": bool(verify_images)}, digest))
    return rows, {"official_train_labels": len(annotation_names), "train_images_with_label": len(rows),
                  "missing_train_images": missing, "train_label_identity_sha256": label_inventory_hash,
                  "dataset_metadata": provenance, "geographic_independence_known": False,
                  "val_test_images_used": 0}


def gsv_csv_filename(row: dict) -> str:
    """Official GSV-Cities CSV -> file binding, preserving raw lat/lon strings."""
    if any(not isinstance(row.get(field), str) or not row[field] for field in GSV_FIELDS):
        raise ValueError("Incomplete GSV CSV binding")
    for field in ("city_id", "panoid"):
        if Path(row[field]).name != row[field] or row[field] in {".", ".."}:
            raise ValueError(f"Invalid GSV CSV path field: {field}")
    return "_".join((row["city_id"], str(int(row["place_id"])).zfill(7),
                     str(int(row["year"])).zfill(4), str(int(row["month"])).zfill(2),
                     str(int(row["northdeg"])).zfill(3), row["lat"], row["lon"], row["panoid"])) + ".jpg"


def _gsv_csv_rows(dataset_root: Path):
    for annotation in sorted((dataset_root / "Dataframes").glob("*.csv")):
        annotation = checked_file(annotation, within=dataset_root)
        digest = file_sha256(annotation)
        with annotation.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(GSV_FIELDS) - set(reader.fieldnames):
                raise ValueError(f"Missing GSV CSV columns: {annotation}")
            for number, raw in enumerate(reader, 1):
                binding = {field: raw[field] for field in GSV_FIELDS}
                if binding["city_id"] != annotation.stem:
                    raise ValueError(f"GSV city differs from source CSV: {annotation}")
                filename = gsv_csv_filename(binding)
                yield annotation, digest, number, binding, dataset_root / "Images" / binding["city_id"] / filename


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"research_sources_v1:{seed}:{value}".encode()).hexdigest()


def build_gsv_scale_manifests(project_root: Path | str, dataset_root: Path | str,
                              qwen_manifest: Path | str, *, new_places=5000,
                              views_per_new_place=2, seed=42,
                              excluded_places=(), verify_images=False) -> tuple[dict, dict]:
    """Retain the complete current Day pool; add genuinely NEW CSV-labelled places."""
    if type(new_places) is not int or new_places < 1:
        raise ValueError("new_places must be a positive integer")
    if type(views_per_new_place) is not int or views_per_new_place < 2:
        raise ValueError("At least two verified views per added GSV place are required")
    root, data_root = Path(project_root).resolve(), Path(dataset_root).resolve()
    manifest = checked_file(qwen_manifest)
    qwen_hash = file_sha256(manifest)
    current = {}
    for row in _jsonl(manifest):
        if (row.get("condition", "").lower() != "night" or row.get("passed") is not True
                or row.get("eligible_for_training") is not True or row.get("accepted", True) is not True):
            continue
        path = _resolve(row["source_path"], manifest, root)
        place = str(row["city"]), str(int(row["place_id"]))
        if path.name in current:
            raise ValueError(f"Duplicate current GSV Day filename: {path.name}")
        current[path.name] = (place, path, row.get("source_sha256"))
    excluded = set(excluded_places)
    if set(place for place, _, _ in current.values()) & excluded:
        raise ValueError("Current Day training pool overlaps heldout places")
    place_views, city_counts, csv_hashes = Counter(), Counter(), {}
    all_places, absent_files, csv_count = set(), 0, 0
    matched_current = {}
    for annotation, digest, number, binding, path in _gsv_csv_rows(data_root):
        csv_count += 1
        csv_hashes[portable_path(annotation, root)] = digest
        place = binding["city_id"], str(int(binding["place_id"]))
        all_places.add(place)
        if not path.is_file():
            absent_files += 1
            continue
        checked_file(path, within=data_root)
        city_counts[place[0]] += 1
        if place not in excluded:
            place_views[place] += 1
        if path.name in current:
            expected, original, image_hash = current[path.name]
            if place != expected:
                raise ValueError(f"Current GSV label differs from CSV: {path.name}")
            if path.name in matched_current:
                raise ValueError(f"Ambiguous duplicate GSV CSV image: {path.name}")
            # Use the original CSV-bound path for the new stream while proving
            # it has exactly the bytes used by the current canonical G1 copy.
            original_hash = file_sha256(original)
            if image_hash is not None and original_hash != image_hash:
                raise ValueError(f"Current GSV Day SHA256 changed: {original}")
            if file_sha256(path) != original_hash:
                raise ValueError(f"GSV CSV image differs from canonical training Day: {path}")
            matched_current[path.name] = (annotation, digest, number, binding, path, original_hash)
    if len(matched_current) != len(current):
        raise ValueError(f"Current GSV Day files missing CSV/image binding: {len(current)-len(matched_current)}")
    current_places = {place for place, _, _ in current.values()}
    candidates = [place for place, count in place_views.items()
                  if place not in current_places and count >= views_per_new_place]
    chosen = set(sorted(candidates, key=lambda place: (_rank(seed, repr(place)), place))[:new_places])
    if len(chosen) < new_places:
        raise ValueError(f"Only {len(chosen)} eligible NEW GSV places; requested {new_places}")
    selected_views = defaultdict(list)
    for annotation, digest, number, binding, path in _gsv_csv_rows(data_root):
        place = binding["city_id"], str(int(binding["place_id"]))
        if place not in chosen or not path.is_file():
            continue
        rank = _rank(seed, portable_path(path, root))
        selected_views[place].append((rank, annotation, digest, number, binding, path, None))
        selected_views[place].sort(key=lambda item: item[0])
        del selected_views[place][views_per_new_place:]

    def make_record(item, *, current_pool):
        annotation, digest, number, binding, path, image_hash = item
        path = checked_file(path)
        if verify_images:
            image_hash = _inspect_image(path, image_hash)
        place = binding["city_id"], str(int(binding["place_id"]))
        return {"schema_version": 1, "kind": "gsv_day_train", "source": "gsv_day",
                "sample_id": "gsv_" + hashlib.sha256(path.name.encode()).hexdigest()[:24],
                "split": "train", "path": portable_path(path, root),
                "city": place[0], "place_id": place[1],
                "group_id": f"gsv:{place[0]}:{place[1]}",
                **({"sha256": image_hash} if image_hash else {}),
                "metadata": {"annotation_csv": portable_path(annotation, root),
                             "annotation_sha256": digest, "annotation_row": number,
                             "csv_binding": binding, "current_qwen_day_pool": current_pool,
                             "grouping_kind": "verified_gsv_place",
                             "place_label_verified": True,
                             "geographic_independence_known": False,
                             "image_hash_verified": bool(verify_images or current_pool)}}

    g1 = [make_record(item, current_pool=True) for _, item in sorted(matched_current.items())]
    new_rows = [make_record(item[1:], current_pool=False)
                for place in sorted(selected_views) for item in selected_views[place]]
    g2 = sorted(g1 + new_rows, key=lambda row: (row["city"], int(row["place_id"]), row["sample_id"]))
    report = {"csv_rows": csv_count, "csv_unique_places": len(all_places),
              "existing_csv_image_rows": sum(city_counts.values()), "missing_csv_image_rows": absent_files,
              "existing_csv_rows_per_city": dict(sorted(city_counts.items())),
              "excluded_places": len(excluded), "excluded_places_present_in_csv": len(all_places & excluded),
              "eligible_existing_places": len(place_views),
              "g1": _gsv_counts(g1), "g2": _gsv_counts(g2),
              "new_places_requested": new_places, "new_places_added": len(chosen),
              "views_per_new_place": views_per_new_place, "seed": seed,
              "selection": "SHA256 rank of explicit CSV city/place, retaining every G1 Day; minimum verified views required",
              "source_csvs": csv_hashes, "qwen_manifest": portable_path(manifest, root),
              "qwen_manifest_sha256": qwen_hash, "evaluation_gallery_changed": False,
              "note": "Known GSV place identities are annotation-backed; no new geographic independence certification is inferred."}
    return {"g1_day": g1, "g2_day": g2}, report


def _gsv_counts(rows):
    places = Counter((row["city"], row["place_id"]) for row in rows)
    return {"images": len(rows), "unique_places": len(places),
            "cities": dict(sorted(Counter(row["city"] for row in rows).items())),
            "views_per_place_histogram": dict(sorted(Counter(places.values()).items()))}


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> str:
    """Exclusive write: never silently replace a previous research manifest."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return file_sha256(path)


def write_json(path: Path | str, value) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_research_sources(project_root: Path | str, output: Path | str, *,
                             qwen_manifest: Path | str | None = None,
                             real_review: Path | str | None = None,
                             include_bdd=False, include_nightcity=False,
                             include_gsv=False, gsv_new_places=5000,
                             gsv_views_per_place=2, seed=42,
                             verify_images=False, extra_exclusions=()) -> dict:
    """Create a fresh, signed inventory directory; no dataset content is changed."""
    root = Path(project_root).resolve()
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Research output already exists: {output}")
    # Reserve output before doing expensive auditing, so competing calls cannot
    # target the same location. Failure keeps a partial, visibly incomplete audit.
    output.mkdir(parents=True, exist_ok=False)
    exclusions, evidence = collect_exclusions(root, extra_exclusions)
    merged = output / "excluded_gsv_places.json"
    write_json(merged, sorted(exclusions))
    exclusion_files = {portable_path(merged, root): file_sha256(merged)}
    for path, digest in evidence.items():
        actual = Path(path) if Path(path).is_absolute() else root / path
        if actual.suffix == ".json" and isinstance(json.loads(actual.read_text()), list):
            exclusion_files[path] = digest
    qwen_manifest = Path(qwen_manifest) if qwen_manifest else root / "data/splits/night_train.jsonl"
    combined, reports = [], {}
    rows, reports["qwen"] = accepted_qwen_sources(root, qwen_manifest,
        verify_images=verify_images, excluded_places=exclusions)
    combined.extend(rows)
    if real_review is not None:
        rows, reports["darkdriving"] = darkdriving_train_sources(root,
            root / "data/DarkDriving_Base", real_review, verify_images=verify_images)
        combined.extend(rows)
    if include_bdd:
        rows, reports["bdd"] = bdd_train_sources(root, root / "data/bdd100k", verify_images=verify_images)
        combined.extend(rows)
    if include_nightcity:
        rows, reports["nightcity"] = nightcity_train_sources(root, root / "data/NightCity", verify_images=verify_images)
        combined.extend(rows)
    counts, manifests = {}, {}
    for source in sorted({row["source"] for row in combined}):
        source_rows = sorted((row for row in combined if row["source"] == source), key=lambda row: row["sample_id"])
        path = output / f"{source}.jsonl"
        manifests[source] = {"path": portable_path(path, root), "sha256": write_jsonl(path, source_rows)}
        counts[source] = {"rows": len(source_rows), "unique_sample_ids": len({row["sample_id"] for row in source_rows}),
                          "groups": len({row["group_id"] for row in source_rows})}
    combined_path = output / "train_sources.jsonl"
    manifests["combined"] = {"path": portable_path(combined_path, root),
        "sha256": write_jsonl(combined_path, sorted(combined, key=lambda row: (row["source"], row["sample_id"])))}
    if include_gsv:
        pools, reports["gsv"] = build_gsv_scale_manifests(root, root / "data/gsv-cities", qwen_manifest,
            new_places=gsv_new_places, views_per_new_place=gsv_views_per_place,
            seed=seed, excluded_places=exclusions, verify_images=verify_images)
        for name, rows in pools.items():
            path = output / f"{name}.jsonl"
            digest = write_jsonl(path, rows)
            sidecar = {"schema_version": 1, "kind": "gsv_day_train", "manifest_sha256": digest,
                       "source_csvs": reports["gsv"]["source_csvs"], "exclusion_files": exclusion_files,
                       "exclusion_evidence": evidence, "counts": _gsv_counts(rows),
                       "qwen_manifest": portable_path(qwen_manifest, root),
                       "qwen_manifest_sha256": reports["gsv"]["qwen_manifest_sha256"],
                       "seed": seed, "evaluation_gallery_changed": False}
            write_json(Path(str(path) + ".metadata.json"), sidecar)
            manifests[name] = {"path": portable_path(path, root), "sha256": digest,
                               "metadata": portable_path(Path(str(path) + ".metadata.json"), root)}
    report = {"schema_version": 1, "kind": "train_only_research_inventory", "seed": seed,
              "source_counts": counts, "sources": reports, "manifests": manifests,
              "exclusion_evidence": evidence, "excluded_places": len(exclusions),
              "image_verification": "strict_decode_and_sha256" if verify_images else "path_and_annotation_binding_only",
              "training_started": False, "validation_test_images_used": False,
              "route_independence_certified": False}
    write_json(output / "inventory.json", report)
    return report

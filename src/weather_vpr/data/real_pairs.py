"""Reviewed real Day/Night supervision, without inferred geographic labels.

A review manifest is evidence supplied by its author. ``automatic`` records an
automatic decision, never an independent human verification. Neither a filename
prefix nor an exact-content component establishes a geographic place. Only
explicitly accepted, assigned records enter the requested split.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageFile

from weather_vpr.data.darkdriving import load_darkdriving_manifest


_REVIEW_KEYS = {
    "sample_id", "scene_group", "split", "pair_valid", "visibility_valid",
    "review_status", "reject_reason", "source_manifest_sha256", "day_sha256",
    "night_sha256",
}
_SPLITS = {"train", "dev", "test"}
_STATUSES = {"manual", "automatic", "unreviewed"}


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def load_review_source(root: Path | str) -> tuple[Path, list[dict], str]:
    """Read a cleaned source and bind it to its original preparation metadata.

    This checks the inventory and manifest, not image quality or route overlap.
    Actual reviewed image bytes are verified by :class:`ReviewedRealPairs`.
    """
    root = Path(root).expanduser().resolve(strict=True)
    manifest = root / "manifest.jsonl"
    original_hash = file_sha256(manifest)
    rows = load_darkdriving_manifest(root)
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    if (not isinstance(info, dict) or type(info.get("pairs")) is not int
            or info["pairs"] != len(rows)
            or info.get("manifest_sha256") != original_hash
            or file_sha256(manifest) != original_hash):
        raise ValueError("DarkDriving source manifest differs from dataset_info.json")
    for row in rows:
        for field in ("sample_id", "sequence_id", "content_group"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Invalid source {field}: {row.get(field)!r}")
        for side in ("day", "night"):
            for field in (f"{side}_sha256", f"{side}_pixel_sha256"):
                if not _is_hash(row.get(field)):
                    raise ValueError(f"Invalid source {field}: {row['sample_id']}")
    return root, rows, original_hash


def _json_record(pairs):
    # Duplicate JSON object keys must not silently overwrite review decisions.
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate JSON key: {key}")
        record[key] = value
    return record


def _load_review(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_json_record,
                                parse_constant=lambda value: (_ for _ in ()).throw(
                                    ValueError(f"Invalid JSON constant: {value}")))
        except ValueError as error:
            raise ValueError(f"Invalid review JSONL line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Review line {line_number} must be a JSON object")
        if missing := _REVIEW_KEYS - record.keys():
            raise ValueError(f"Review line {line_number} missing fields: {sorted(missing)}")
        rows.append(record)
    if not rows:
        raise ValueError("Empty real-pair review manifest")
    return rows, hashlib.sha256(payload).hexdigest()


def _validate_review(row: dict, source: dict, source_hash: str) -> None:
    sample_id = row["sample_id"]
    status, split, group = row["review_status"], row["split"], row["scene_group"]
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValueError(f"Invalid review_status for {sample_id}: {status!r}")
    if split is not None and (not isinstance(split, str) or split not in _SPLITS):
        raise ValueError(f"Invalid split for {sample_id}: {split!r}")
    if group is not None and (not isinstance(group, str) or not group.strip() or group != group.strip()):
        raise ValueError(f"scene_group must be a nonempty, trimmed string for {sample_id}")
    if split is not None and group is None:
        raise ValueError(f"Assigned split requires scene_group for {sample_id}")
    for field in ("pair_valid", "visibility_valid"):
        if row[field] is not None and type(row[field]) is not bool:
            raise ValueError(f"{field} must be bool or null for {sample_id}")
        if status != "unreviewed" and type(row[field]) is not bool:
            raise ValueError(f"Reviewed {field} must be bool for {sample_id}")
    reason = row["reject_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(f"reject_reason must be a nonempty string or null for {sample_id}")
    accepted = status != "unreviewed" and row["pair_valid"] and row["visibility_valid"]
    if accepted and (split is None or group is None):
        raise ValueError(f"Accepted review requires split and scene_group for {sample_id}")
    if accepted and reason is not None:
        raise ValueError(f"Accepted review cannot have reject_reason for {sample_id}")
    if status != "unreviewed" and not accepted and reason is None:
        raise ValueError(f"Rejected review requires reject_reason for {sample_id}")
    for field, expected in (("source_manifest_sha256", source_hash),
                            ("day_sha256", source["day_sha256"]),
                            ("night_sha256", source["night_sha256"])):
        if not _is_hash(row[field]) or row[field] != expected:
            raise ValueError(f"Review {field} differs from source for {sample_id}")


def _check_split_isolation(source_rows: list[dict], reviews: list[dict]) -> None:
    """Connect scenes, declared components and exact reuse before checking splits.

    Include unreviewed source rows in the graph: A--unreviewed--B still links A
    and B. Scene edges also include unassigned, unreviewed and rejected records:
    a scene edge followed by a content edge must never bridge two splits.
    """
    parent = {row["sample_id"]: row["sample_id"] for row in source_rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in source_rows:
        name = row["sample_id"]
        keys = [("content_group", row["content_group"])]
        # Reuse across either side also joins the component.
        keys.extend((kind, row[f"{side}_{kind}"])
                    for kind in ("sha256", "pixel_sha256") for side in ("day", "night"))
        for key in keys:
            if key in seen:
                parent[find(name)] = find(seen[key])
            else:
                seen[key] = name
    seen_scenes = {}
    for row in reviews:
        group = row["scene_group"]
        if group is None:
            continue
        name = row["sample_id"]
        if group in seen_scenes:
            parent[find(name)] = find(seen_scenes[group])
        else:
            seen_scenes[group] = name
    assigned = {}
    for row in reviews:
        if row["split"] is None:
            continue
        for kind, group in (("scene_group", row["scene_group"]),
                            ("content_group", find(row["sample_id"]))):
            key = kind, group
            previous = assigned.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"{kind} crosses splits: {group!r} ({previous}/{row['split']})")


class ReviewedRealPairs:
    """Accepted real pairs with strict provenance and SALAD image preprocessing.

    Only ``pair_valid=true``, ``visibility_valid=true`` and review status
    ``manual``/``automatic`` enter the selected split (default ``train``). These
    labels do not independently prove route isolation; ``scene_group`` must be
    supplied from a defensible review, never inferred as a place label.

    ``day_sources`` and ``sources_signature()`` expose ordered cache inputs.
    ``rows`` and returned ``day_path``/``night_path`` use absolute image paths.
    Hashes are verified on initialization for every reviewed pair and again at
    each read, preventing stale cached targets after a file changes.
    """

    def __init__(self, root: Path | str, review_manifest: Path | str,
                 image_size=(224, 224), *, split="train"):
        if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                or any(type(value) is not int or value < 1 for value in image_size)):
            raise ValueError("image_size must be two positive integers")
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError("split must be train, dev, or test")
        self.root, source_rows, self.source_manifest_sha256 = load_review_source(root)
        self.source_manifest = self.root / "manifest.jsonl"
        self.review_manifest = Path(review_manifest).expanduser().resolve(strict=True)
        self.manifest = self.review_manifest
        reviews, self.review_manifest_sha256 = _load_review(self.review_manifest)
        self.image_size = self.size = tuple(image_size)
        self.split = split
        self.preprocessing = {
            "version": "rgb_bilinear_imagenet_v1", "image_size": list(image_size),
            "color_space": "RGB", "input_range": [0.0, 1.0], "resize": "PIL_bilinear",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        }
        sources = {row["sample_id"]: row for row in source_rows}
        seen = set()
        for review in reviews:
            name = review["sample_id"]
            if not isinstance(name, str) or name not in sources:
                raise ValueError(f"Unknown reviewed sample_id: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate reviewed sample_id: {name}")
            seen.add(name)
            source = sources[name]
            _validate_review(review, source, self.source_manifest_sha256)
            for side in ("day", "night"):
                path = self.root / source[f"{side}_path"]
                if file_sha256(path) != source[f"{side}_sha256"]:
                    raise ValueError(f"Reviewed image bytes differ from source: {path}")
        _check_split_isolation(source_rows, reviews)
        self.review_counts = {
            status: sum(row["review_status"] == status for row in reviews)
            for status in sorted(_STATUSES)
        }
        self.rows = []
        for review in reviews:
            if (review["split"] != split or review["review_status"] == "unreviewed"
                    or review["pair_valid"] is not True or review["visibility_valid"] is not True):
                continue
            # Review notes cannot override source paths, grouping or hashes.
            row = {**sources[review["sample_id"]],
                   **{key: review[key] for key in _REVIEW_KEYS}}
            for side in ("day", "night"):
                row[f"{side}_path"] = str(self.root / row[f"{side}_path"])
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No accepted real pairs in split {split!r}; complete the review first")
        self._day_versions = [self._file_version(Path(row['day_path'])) for row in self.rows]
        if (file_sha256(self.source_manifest) != self.source_manifest_sha256
                or file_sha256(self.review_manifest) != self.review_manifest_sha256):
            raise RuntimeError("Real-pair source or review manifest changed during loading; rerun")

    @property
    def day_sources(self) -> list[dict]:
        return [{"path": row["day_path"], "sha256": row["day_sha256"]} for row in self.rows]

    def sources_signature(self) -> dict:
        """JSON-compatible provenance for an ordered paired-target cache identity."""
        if (file_sha256(self.source_manifest) != self.source_manifest_sha256
                or file_sha256(self.review_manifest) != self.review_manifest_sha256):
            raise RuntimeError("Real-pair source or review manifest changed after loading; rerun")
        keys = ("sample_id", "scene_group", "content_group", "day_path", "night_path",
                "day_sha256", "night_sha256", "review_status")
        return {
            "format_version": 1, "source_manifest_sha256": self.source_manifest_sha256,
            "review_manifest_sha256": self.review_manifest_sha256, "split": self.split,
            "preprocessing": {**self.preprocessing, "image_size": list(self.image_size)},
            "source_images": [{key: row[key] for key in keys} for row in self.rows],
        }

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _file_version(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _read_tensor(self, row, side):
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise RuntimeError("Reviewed real pairs require strict Pillow image decoding")
        import numpy as np
        import torch
        mean = torch.tensor(self.preprocessing["mean"])[:, None, None]
        std = torch.tensor(self.preprocessing["std"])[:, None, None]
        height, width = self.image_size
        path = Path(row[f"{side}_path"])
        if path.is_symlink():
            raise RuntimeError(f"Reviewed image became a symlink: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row[f"{side}_sha256"]:
            raise RuntimeError(f"Reviewed image changed after validation: {path}; rerun")
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        return (tensor.contiguous().float().div_(255) - mean) / std

    def read_night_tensor(self, index):
        """Read only Night when its signed Day target has already been cached.

        Day bytes were validated at construction/cache extraction; a cheap
        inode/size/timestamp check rejects later changes without decoding Day.
        Night continues to receive full byte-hash and strict decode checks.
        """
        row = self.rows[index]
        day = Path(row['day_path'])
        if day.is_symlink() or self._file_version(day) != self._day_versions[index]:
            raise RuntimeError(f'Cached-target Day image changed after validation: {day}; rerun')
        return self._read_tensor(row, 'night')

    def __getitem__(self, index):
        row = self.rows[index]
        result = {key: row[key] for key in (
            "sample_id", "sequence_id", "content_group", "scene_group", "day_path",
            "night_path", "day_sha256", "night_sha256", "review_status")}
        result["index"] = index
        for side in ("day", "night"):
            result[side] = self._read_tensor(row, side)
            result[f"{side}_tensor"] = result[side]
        return result

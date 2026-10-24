"""Tokyo247 night retrieval using the complete official NetVLAD test protocol.

The local query release also contains extra images. Only query IDs in the
checksum-verified official MAT participate; night is the CSV time-of-day label,
not a filename or brightness heuristic. Gallery geometry comes from that MAT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from weather_vpr.utils.backend import enable_backend_imports


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokyo_night(root: Path, mat_path: Path, source_path: Path):
    """Return a validated ``workflow.evaluation.EvaluationSet`` for night.

    ``source_path`` must identify the MAT's SHA256. Optional declared image
    counts and night IDs are checked as well. MATLAB database JPEG names map to
    the original local PNG files; IDs and order remain exactly as in the MAT.
    No image descriptor/model computation is performed here.
    """
    enable_backend_imports()
    from workflow.evaluation import EvaluationImage, EvaluationSet
    import numpy as np
    from scipy.io import loadmat
    from scipy.spatial import cKDTree

    root = Path(root).expanduser().resolve()
    mat_path = Path(mat_path).expanduser().resolve()
    source_path = Path(source_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Tokyo247 root does not exist: {root}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Tokyo247 source metadata must be a JSON object")
    expected_hash = source.get("mat_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError("Tokyo247 source metadata requires a valid mat_sha256")
    actual_hash = _sha256(mat_path)
    if actual_hash != expected_hash.lower():
        raise ValueError("Tokyo247 MAT SHA256 does not match source metadata")
    if "mat_file" in source and source["mat_file"] != mat_path.name:
        raise ValueError("Tokyo247 source mat_file does not match the selected MAT")
    if source.get("dataset", "Tokyo247") != "Tokyo247" or source.get("split", "test") != "test":
        raise ValueError("Tokyo247 source metadata must describe the test split")

    payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    struct = payload.get("dbStruct")
    required = ("whichSet", "dbImageFns", "qImageFns", "utmDb", "utmQ",
                "numImages", "numQueries", "posDistThr", "posDistSqThr")
    if struct is None or any(not hasattr(struct, key) for key in required):
        raise ValueError("Tokyo247 MAT requires a complete scalar dbStruct")
    if struct.whichSet != "test":
        raise ValueError("Tokyo247 MAT must define the official test split")

    def integer(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Tokyo247 {label} must be a positive integer")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Tokyo247 {label} must be a positive integer") from exc
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError(f"Tokyo247 {label} must be a positive integer")
        return int(number)

    ref_count = integer(struct.numImages, "numImages")
    query_count = integer(struct.numQueries, "numQueries")

    def names(value, count, label):
        values = np.atleast_1d(value)
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f"Tokyo247 {label} size does not match its declared count")
        result = []
        for name in values:
            if not isinstance(name, (str, np.str_)):
                raise ValueError(f"Tokyo247 {label} must contain string filenames")
            name = str(name)
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                    or path.as_posix() != name or path.suffix != ".jpg"
                    or (label == "qImageFns" and len(path.parts) != 1)):
                raise ValueError(f"Unsafe or invalid Tokyo247 {label} filename: {name!r}")
            result.append(name)
        if len(set(result)) != count:
            raise ValueError(f"Tokyo247 {label} contains duplicate IDs")
        return result

    ref_names = names(struct.dbImageFns, ref_count, "dbImageFns")
    query_names = names(struct.qImageFns, query_count, "qImageFns")

    def coordinates(value, count, label):
        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Tokyo247 {label} must contain numeric coordinates") from exc
        if count == 1 and values.shape == (2,):
            values = values.reshape(2, 1)
        if values.shape != (2, count) or not np.isfinite(values).all():
            raise ValueError(f"Tokyo247 {label} must be finite with shape (2, {count})")
        return values.T

    ref_xy = coordinates(struct.utmDb, ref_count, "utmDb")
    query_xy = coordinates(struct.utmQ, query_count, "utmQ")
    try:
        radius, radius_squared = float(struct.posDistThr), float(struct.posDistSqThr)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tokyo247 positive distance thresholds must be numeric") from exc
    if radius != 25.0 or radius_squared != 625.0:
        raise ValueError("Tokyo247 official positives require posDistThr=25 and posDistSqThr=625")

    references = [EvaluationImage(name, root / "database_gsv_vga" / Path(name).with_suffix(".png"))
                  for name in ref_names]
    all_queries = [EvaluationImage(name, root / "queries" / name) for name in query_names]
    missing = [str(image.path) for image in references + all_queries if not image.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Tokyo247 protocol images: {missing[:5]}")
    night_indices = []
    for index, image in enumerate(all_queries):
        csv_path = image.path.with_suffix(".csv")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if (len(rows) != 1 or len(rows[0]) < 9 or rows[0][0] != image.id
                or rows[0][5] not in {"D", "S", "N"}):
            raise ValueError(f"Invalid Tokyo247 query CSV metadata: {csv_path}")
        if rows[0][5] == "N":
            night_indices.append(index)
    if not night_indices:
        raise ValueError("Tokyo247 official query subset has no night labels")
    queries = [all_queries[index] for index in night_indices]
    for key, count in (("gallery_count", ref_count), ("official_full_query_count", query_count),
                       ("night_query_count", len(queries))):
        if key in source and integer(source[key], key) != count:
            raise ValueError(f"Tokyo247 {key} does not match source metadata")
    if "night_ids" in source:
        ids = source["night_ids"]
        if (not isinstance(ids, list) or any(not isinstance(value, str) for value in ids)
                or len(set(ids)) != len(ids) or set(ids) != {image.id for image in queries}):
            raise ValueError("Tokyo247 night IDs do not match source metadata")
    positives = [sorted(int(index) for index in row) for row in
                 cKDTree(ref_xy).query_ball_point(query_xy[night_indices], radius)]
    dataset = EvaluationSet("Tokyo247_night", references, queries, positives, {
        "ground_truth": "official_netvlad_mat_utm_radius",
        "split": "test", "positive_radius_meters": radius,
        "gallery": "complete_official_mat_gallery",
        "official_full_query_count": query_count,
        "query_subset": "official_mat_queries_with_csv_column_6_equal_N",
        "night_label_rule": "queries/<qImageFn stem>.csv column 6 (zero-based index 5) == N",
        "query_coordinates": "official_mat_utmQ_not_local_csv_coordinates",
        "gallery_extension_mapping": "MAT .jpg names map to local original .png images",
        "mat_path": str(mat_path), "mat_sha256": actual_hash,
        "source_path": str(source_path), "source_sha256": _sha256(source_path),
        "source_url": source.get("source_url"), "source_page": source.get("source_page"),
        "archive_member": source.get("archive_member"),
        "local_query_release_note": (
            "Local queries use the expanded v3 naming; only official MAT query IDs are used. "
            "Exact identity of local v3 images versus original v2 images has not been "
            "independently verified byte for byte."
        ),
    })
    dataset.validate()
    return dataset

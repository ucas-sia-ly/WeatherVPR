"""Read-only retrieval diagnostics on public RobotCar Seasons v2 train poses.

This is a custom 25 m position-retrieval protocol, not the official withheld
test/localization benchmark. No real-night image is used for optimization.
The official READMEs specify that all aligned reference models share a world
frame and that the public 4x4 transforms map camera coordinates to that frame.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re

from weather_vpr.utils.backend import enable_backend_imports


_SOURCE = "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/"
_CONDITIONS = frozenset({"overcast-reference", "dawn", "dusk", "night", "night-rain",
                         "overcast-summer", "overcast-winter", "rain", "snow", "sun"})
_CAMERAS = frozenset({"left", "rear", "right"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name(value: str) -> str:
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value or len(path.parts) != 3
            or path.parts[0] not in _CONDITIONS or path.parts[1] not in _CAMERAS
            or not re.fullmatch(r"[0-9]+\.(jpg|png)", path.name)):
        raise ValueError(f"Unsafe or invalid RobotCar image name: {value!r}")
    return path.with_suffix(".jpg").as_posix()


def _inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"RobotCar path escapes dataset root: {path}")
    return path


def _metadata_path(root: Path) -> Path:
    candidates = []
    for directory in (root / "metadata", root):
        candidates.extend(p for p in directory.glob("robotcar_v2_train.txt*")
                          if p.name == "robotcar_v2_train.txt" or p.name.startswith("robotcar_v2_train.txt?"))
    if len(candidates) != 1:
        raise ValueError("RobotCar requires exactly one public robotcar_v2_train.txt "
                         "in the root or metadata directory (URL query suffix is supported)")
    return _inside(candidates[0], root)


def load_robotcar_public_night(root: Path, condition: str = "night"):
    """Return an EvaluationSet from public poses, without reading the test list.

    Queries are the selected condition's public-train rear images. References
    are every image registered in all 49 aligned overcast-reference models,
    including all three cameras. Positives have 3D Euclidean center distance
    <=25 m; there is deliberately no orientation or camera-side filter.
    """
    if condition not in {"night", "night-rain"}:
        raise ValueError("RobotCar public night condition must be night or night-rain")
    enable_backend_imports()
    from workflow.evaluation import EvaluationImage, EvaluationSet
    import numpy as np
    from scipy.spatial import cKDTree

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RobotCar root does not exist: {root}")
    metadata = _metadata_path(root)
    train = {}
    with metadata.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 17:
                raise ValueError(f"RobotCar train pose must contain name and 16 values: line {number}")
            name = _name(parts[0])
            if name in train:
                raise ValueError(f"Duplicate RobotCar train image: {name}")
            try:
                matrix = np.asarray(parts[1:], dtype=np.float64).reshape(4, 4)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid RobotCar train pose: line {number}") from exc
            if (not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
                raise ValueError(f"RobotCar train pose must be finite homogeneous 4x4: {name}")
            rotation = matrix[:3, :3]
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                    or abs(np.linalg.det(rotation) - 1) > 1e-4):
                raise ValueError(f"RobotCar train pose has invalid rotation: {name}")
            # This is C, not the COLMAP world-to-camera translation t.
            train[name] = matrix[:3, 3].copy()
    if not train:
        raise ValueError("RobotCar public train metadata is empty")

    model_root = root / "3D-models/individual/colmap_reconstructions"
    expected = [model_root / f"{index:03d}_aligned" / f"{index:03d}_aligned" / "images.txt"
                for index in range(1, 50)]
    discovered = sorted(model_root.rglob("images.txt"))
    if set(discovered) != set(expected):
        raise ValueError("RobotCar requires all 49 aligned COLMAP images.txt models with no extra models")
    reference_centers = {}
    model_sources = []
    declared_counts = {}
    for model in expected:
        model = _inside(model, root)
        model_sources.append({"path": model.relative_to(root).as_posix(), "sha256": _sha256(model)})
        count = 0
        image_ids = set()
        declared = None
        with model.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.startswith("#"):
                    match = re.match(r"# Number of images:\s*([0-9]+)", line)
                    if match:
                        declared = int(match.group(1))
                    continue
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 10:
                    raise ValueError(f"Invalid RobotCar COLMAP pose row in {model}: {number}")
                try:
                    image_id, camera_id = int(parts[0]), int(parts[8])
                    values = np.asarray(parts[1:8], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid RobotCar COLMAP numeric pose in {model}") from exc
                if image_id <= 0 or camera_id <= 0 or image_id in image_ids:
                    raise ValueError(f"Duplicate or invalid RobotCar COLMAP image ID in {model}")
                image_ids.add(image_id)
                if not np.isfinite(values).all() or abs(np.linalg.norm(values[:4]) - 1) > 1e-3:
                    raise ValueError(f"RobotCar COLMAP pose must have finite values and unit quaternion: {model}")
                qw, qx, qy, qz = values[:4] / np.linalg.norm(values[:4])
                rotation = np.array([
                    [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                    [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
                    [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
                ])
                name = _name(parts[9])
                if not name.startswith("overcast-reference/"):
                    raise ValueError(f"RobotCar reference model contains nonreference image: {name}")
                if name in reference_centers:
                    raise ValueError(f"Duplicate RobotCar Gallery image across models: {name}")
                reference_centers[name] = -rotation.T @ values[4:7]
                count += 1
                # COLMAP stores a second, possibly empty, POINTS2D row.
                if next(stream, None) is None:
                    raise ValueError(f"Missing COLMAP POINTS2D row in {model}")
        if not count or (declared is not None and declared != count):
            raise ValueError(f"RobotCar COLMAP image count does not match model header: {model}")
        declared_counts[model.parent.name] = count

    reference_names = sorted(reference_centers)
    inventory = {p.relative_to(root / "images").as_posix()
                 for p in (root / "images/overcast-reference").rglob("*")
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if inventory != set(reference_names):
        raise ValueError("RobotCar Gallery image inventory must match the complete 49-model registered Gallery; "
                         f"missing={len(set(reference_names) - inventory)}, extra={len(inventory - set(reference_names))}")
    query_names = sorted(name for name in train if name.startswith(f"{condition}/rear/"))
    if not query_names:
        raise ValueError(f"RobotCar public train poses contain no {condition} rear queries")
    if set(query_names) & set(reference_names):
        raise ValueError("RobotCar reference/query identities overlap")

    references = [EvaluationImage(name, _inside(root / "images" / name, root)) for name in reference_names]
    queries = [EvaluationImage(name, _inside(root / "images" / name, root)) for name in query_names]
    ref_xyz = np.asarray([reference_centers[name] for name in reference_names])
    query_xyz = np.asarray([train[name] for name in query_names])
    overlap = sorted(set(train) & set(reference_centers))
    overlap_error = max((float(np.linalg.norm(train[name] - reference_centers[name])) for name in overlap),
                        default=None)
    if overlap_error is not None and overlap_error > .01:
        raise ValueError("RobotCar train/reference duplicate poses disagree by more than 1 cm")
    positives = [sorted(int(index) for index in row) for row in
                 cKDTree(ref_xyz).query_ball_point(query_xyz, 25.0)]
    if any(not row for row in positives):
        raise ValueError("Every RobotCar public diagnostic query must have a 25 m positive")
    bundle = hashlib.sha256()
    for source in model_sources:
        bundle.update((source["path"] + "\0" + source["sha256"] + "\n").encode("utf-8"))
    dataset = EvaluationSet(f"RobotCar-Seasons_v2_public_train_diagnostic_{condition}",
                            references, queries, positives, {
        "ground_truth": "public_camera_to_world_and_aligned_colmap_3d_radius",
        "split": "public_train_diagnostic", "usage": "evaluation_only_no_training",
        "official_test": False, "condition": condition, "query_camera": "rear",
        "gallery_cameras": ["left", "rear", "right"],
        "gallery": "complete_registered_overcast_reference_in_all_49_aligned_models",
        "positive_radius_meters": 25.0, "positive_boundary": "distance <= 25.0",
        "distance": "3D_Euclidean_world_camera_centers", "orientation_filter": None,
        "query_pose_convention": "camera_to_world_p_w=R*p_c+C; center=matrix[:3,3]",
        "reference_pose_convention": "COLMAP_world_to_camera_p_c=R*p_w+t; center=-R.T@t",
        "coordinate_frame": "official_README_all_49_aligned_models_share_global_world_frame",
        "reference_extension_mapping": "COLMAP .png names map to local original .jpg images",
        "metadata_path": str(metadata), "metadata_sha256": _sha256(metadata),
        "public_train_pose_count": len(train),
        "colmap_pose_models": model_sources, "colmap_pose_bundle_sha256": bundle.hexdigest(),
        "colmap_model_image_counts": declared_counts,
        "train_reference_pose_overlap_count": len(overlap),
        "train_reference_max_center_disagreement_meters": overlap_error,
        "primary_source_urls": [_SOURCE + "README_RobotCar_v2.md",
                                _SOURCE + "README_RobotCar-Seasons.md",
                                _SOURCE + "robotcar_v2_train.txt"],
        "limitations": ("Custom position-only retrieval diagnostic on public train poses, not official test. "
                        "All camera sides are eligible positives; no viewpoint/visibility/orientation filter. "
                        "Official test query poses are not public and are never inferred here."),
    })
    dataset.validate(check_images=True)
    return dataset

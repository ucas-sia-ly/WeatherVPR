#!/usr/bin/env python3
"""Export deterministic held-out Day/Qwen CLS-attention comparisons.

Run serially through ``scripts/run_bounded.py``. This is a qualitative
diagnostic of the last DINOv2 attention block; it is neither a segmentation
result nor a measure of verified patch correspondence or retrieval quality.
Only the first accepted validation pairs are used, without visual selection.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New PNG path; a JSON provenance sidecar is also written")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/night_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=ROOT / "third_party/dinov2")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pdf", action="store_true", help="Also export a new PDF")
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 16:
        parser.error("--samples must be in [1, 16]")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.output = args.output.expanduser().resolve()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end with .png")
    destinations = [args.output, args.output.with_suffix(".png.json")]
    if args.pdf:
        destinations.append(args.output.with_suffix(".pdf"))
    for path in destinations:
        if path.exists():
            parser.error(f"Refusing to overwrite existing output: {path}")
    return args


def load_training_configuration(directory: Path):
    """Reuse effective camera ranges without interpreting metadata as ranges."""
    from weather_vpr.data.night_augmentation import NightAugmentationConfig

    path = directory / "training_config.json"
    configuration = json.loads(path.read_text(encoding="utf-8"))
    effective = configuration.get("effective_night_augmentation")
    if not isinstance(effective, dict):
        raise ValueError("training_config.json must include effective_night_augmentation")
    field_names = {item.name for item in fields(NightAugmentationConfig)}
    config = NightAugmentationConfig.from_dict(
        {key: value for key, value in effective.items() if key in field_names})
    if config.profile != "camera" or config.probability != 1.0:
        raise ValueError("This comparison requires the all-Qwen camera profile (probability=1)")
    return configuration, config, path


def validate_held_out(dataset, training_config: dict) -> dict:
    """Check place separation against the actual source training manifest."""
    train_manifest = project_file(training_config["manifest"])
    if dataset.manifest == train_manifest:
        raise ValueError("Attention preview must use a held-out manifest")
    excluded = set()
    if training_config.get("exclude_places"):
        path = project_file(training_config["exclude_places"])
        excluded = {tuple(map(str, value)) for value in json.loads(path.read_text())}
    train_places = set()
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("condition", "").lower() == "night"
                and row.get("passed") is True
                and row.get("eligible_for_training") is True):
            place = (str(row["city"]), str(row["place_id"]))
            if place not in excluded:
                train_places.add(place)
    overlaps = train_places & {row["place"] for row in dataset.rows}
    if overlaps:
        raise ValueError(f"Validation manifest contains {len(overlaps)} training places")
    return {"training_manifest": str(train_manifest),
            "training_manifest_sha256": sha256(train_manifest),
            "training_places": len(train_places), "overlapping_places": 0,
            "policy": "all validation places disjoint from eligible, nonexcluded training places"}


def normalized_attention(backbone, image, grid_shape, device):
    import torch
    from weather_vpr.models.attention import capture_cls_attention

    with torch.no_grad(), capture_cls_attention(backbone) as capture:
        backbone(image.unsqueeze(0).to(device))
    raw = capture.attention.detach().float().cpu().squeeze(0)
    if raw.numel() != math.prod(grid_shape) or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Expected finite nonnegative last-block CLS-to-patch probabilities")
    mass = raw.sum().item()
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("CLS patch attention has zero mass")
    probabilities = (raw / mass).reshape(grid_shape).numpy()
    return probabilities, mass


def describe_attention(probabilities, patch_mass: float) -> dict:
    import numpy as np

    flat = probabilities.reshape(-1).astype(np.float64)
    positive = flat[flat > 0]
    return {"probabilities": probabilities.tolist(), "raw_patch_mass": patch_mass,
            "entropy_nats": float(-(positive * np.log(positive)).sum()),
            "min": float(flat.min()), "max": float(flat.max()),
            "p95": float(np.quantile(flat, 0.95)), "sum": float(flat.sum())}


def grid_js_divergence(first, second) -> float:
    """A descriptive grid statistic, not correspondence-aware supervision."""
    import numpy as np

    first = first.reshape(-1).astype(np.float64)
    second = second.reshape(-1).astype(np.float64)
    middle = (first + second) / 2

    def kl(value):
        mask = value > 0
        return np.sum(value[mask] * np.log(value[mask] / middle[mask]))

    return float((kl(first) + kl(second)) / 2)


def main() -> None:
    args = arguments()
    # Set this before importing/loading the local DINOv2 implementation.
    if args.device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    import torch

    from weather_vpr.data.paired_dataset import AcceptedWeatherPairs
    from weather_vpr.data.student_augmentation import sample_augmentation_seed
    from weather_vpr.inference.loading import load_weather_model
    from weather_vpr.utils.backend import enable_backend_imports

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    enable_backend_imports()
    from workflow.model import input_preprocessing, load_checkpoint_model

    directory = project_file(args.adapter_dir)
    checkpoint = project_file(args.checkpoint)
    backbone_repo = project_file(args.backbone_repo)
    manifest = project_file(args.manifest)
    configuration, camera, configuration_path = load_training_configuration(directory)
    metadata_path = directory / "adapter.json"
    weights_path = directory / "adapter.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("adapter_type") != "full" or metadata.get("weather") != "night":
        raise ValueError("The preview requires an independent full Night backbone artifact")
    checkpoint_hash = sha256(checkpoint)
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Night artifact does not match the supplied Day checkpoint")

    base = load_checkpoint_model(checkpoint, device=args.device, backbone_repo=backbone_repo)
    preprocessing = input_preprocessing(base.image_size)
    if metadata.get("model_config") != base.config or metadata.get("preprocessing") != preprocessing:
        raise ValueError("Night artifact does not match model configuration/preprocessing")
    if tuple(base.image_size) != (224, 224):
        raise ValueError("This preview requires the current 224 x 224 training resolution")
    net = load_weather_model(base, "night", directory, metadata, device=args.device)
    net.requires_grad_(False)
    net.eval()
    dataset = AcceptedWeatherPairs(manifest, "night", base.image_size,
                                   verify_hashes=True, night_augmentation=camera, seed=args.seed)
    # Epoch zero is fixed for a reproducible diagnostic, independent of when
    # the selected checkpoint was saved; this does not update the model.
    dataset.set_epoch(0)
    separation = validate_held_out(dataset, configuration)
    if len(dataset) < args.samples:
        raise ValueError(f"Manifest has only {len(dataset)} accepted pairs")
    grid_shape = tuple(int(size) // 14 for size in base.image_size)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    std = np.asarray(preprocessing["std"], dtype=np.float32)

    def pixels(tensor):
        return np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)

    figure, axes = plt.subplots(args.samples, 5, figsize=(17.5, 3.55 * args.samples),
                                squeeze=False, layout="constrained")
    titles = ("Clean Day", "Camera-degraded Qwen", "Day model on clean Day",
              "Day model on degraded Qwen", "Night model on degraded Qwen")
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=11)
    rows = []
    try:
        for index in range(args.samples):
            sample = dataset[index]
            day, degraded = sample["day_tensor"], sample["student_tensor"]
            day_pixels, degraded_pixels = pixels(day), pixels(degraded)
            teacher_day, mass_day = normalized_attention(net.base.backbone, day, grid_shape, args.device)
            teacher_night, mass_teacher_night = normalized_attention(
                net.base.backbone, degraded, grid_shape, args.device)
            student_night, mass_student_night = normalized_attention(
                net.adapters["night"], degraded, grid_shape, args.device)
            maps = (teacher_day, teacher_night, student_night)
            vmax = float(np.quantile(np.concatenate([value.reshape(-1) for value in maps]), 0.95))
            if not math.isfinite(vmax) or vmax <= 0:
                raise ValueError("Invalid attention display color limit")
            normalizer = Normalize(vmin=0, vmax=vmax, clip=True)
            axes[index, 0].imshow(day_pixels)
            axes[index, 1].imshow(degraded_pixels)
            for column, (image, probability) in enumerate(
                    zip((day_pixels, degraded_pixels, degraded_pixels), maps), 2):
                axes[index, column].imshow(image)
                heatmap = axes[index, column].imshow(
                    probability, cmap="magma", norm=normalizer, alpha=0.60,
                    interpolation="bilinear", extent=(0, image.shape[1], image.shape[0], 0))
            colorbar = figure.colorbar(heatmap, ax=list(axes[index, 2:]),
                                      shrink=0.75, fraction=0.018, pad=0.01)
            colorbar.set_label("Patch probability (shared row scale)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
            row = sample["row"]
            axes[index, 0].set_ylabel(f'{index + 1}. {row["city"]} / {row["place_id"]}', fontsize=9)
            for axis in axes[index]:
                axis.set_xticks([])
                axis.set_yticks([])
            rows.append({
                "validation_index": index, "sample_id": row.get("sample_id"),
                "city": str(row["city"]), "place_id": str(row["place_id"]),
                "source_path": str(row["source_file"]), "source_sha256": sha256(row["source_file"]),
                "qwen_path": str(row["target_file"]), "qwen_sha256": sha256(row["target_file"]),
                "augmentation_seed": sample_augmentation_seed(args.seed, 0, index),
                "augmentation_record": sample["augmentation_record"],
                "color_vmin": 0, "color_vmax": vmax,
                "teacher_clean_day": describe_attention(teacher_day, mass_day),
                "teacher_degraded_qwen": describe_attention(teacher_night, mass_teacher_night),
                "student_degraded_qwen": describe_attention(student_night, mass_student_night),
                "grid_js_divergence_nats": {
                    "teacher_day_vs_teacher_degraded": grid_js_divergence(teacher_day, teacher_night),
                    "teacher_day_vs_student_degraded": grid_js_divergence(teacher_day, student_night),
                },
            })
        figure.suptitle(
            "Last-block CLS-to-patch attention | first held-out pairs | no sample selection\n"
            "Each map sums to 1; shared scale per row clips at pooled 95th percentile. "
            "Attention is not segmentation or verified correspondence.", fontsize=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing artifacts even if another job
        # created a destination after initial argument validation.
        with args.output.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=180)
        if args.pdf:
            with args.output.with_suffix(".pdf").open("xb") as stream:
                figure.savefig(stream, format="pdf")
    finally:
        plt.close(figure)
    payload = {
        "format_version": 1, "scope": "qualitative held-out synthetic-night diagnostic; no training",
        "sampling": "first accepted validation rows in manifest order; no visual/result selection",
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "accepted_pair_count": len(dataset), "place_separation": separation,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "night_weights": str(weights_path), "night_weights_sha256": sha256(weights_path),
        "night_metadata_sha256": sha256(metadata_path),
        "training_configuration_sha256": sha256(configuration_path),
        "backbone_repo": str(backbone_repo), "preprocessing": preprocessing,
        "seed": args.seed, "zero_based_augmentation_epoch": 0,
        "camera_configuration": camera.to_dict(), "grid_shape_hw": grid_shape,
        "attention": "FP32 last-block CLS-to-patch softmax, averaged over heads, then patch-normalized",
        "color_scale": "per-row pooled 95th percentile over three patch-normalized maps; larger values clipped",
        "overlay_alpha": 0.60, "display_interpolation": "bilinear",
        "caveats": ["Attention is not building segmentation.",
                    "No Qwen/Day geometry assumption or verified patch correspondence is made.",
                    "Same-grid JS is descriptive only and is not retrieval accuracy or a correspondence score.",
                    "Camera-degraded validation views are a synthetic diagnostic, not a real-night benchmark."],
        "rows": rows, "output": str(args.output), "output_sha256": sha256(args.output),
    }
    with args.output.with_suffix(".png.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved {args.output} and provenance sidecar", flush=True)


if __name__ == "__main__":
    main()

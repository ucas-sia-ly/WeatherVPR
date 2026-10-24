#!/usr/bin/env python3
"""Measure conditional attention shape and absolute trusted-support mass.

This read-only diagnostic uses the first accepted held-out Day/Qwen pairs in
manifest order. Run serially through scripts/run_bounded.py. Attention is not
semantic segmentation, and conditional KL does not constrain attention outside
its verified correspondence support.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def shared_helpers():
    """Import provenance/held-out checks without invoking the preview main."""
    path = ROOT / "scripts/visualize_patch_attention.py"
    spec = importlib.util.spec_from_file_location("_attention_preview_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import shared attention diagnostic helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/night_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=ROOT / "third_party/dinov2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    args.output = args.output.expanduser().resolve()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end with .json")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    return args


def numeric_summary(rows):
    """Population standard deviation over all fixed rows, without selection."""
    values = {}

    def visit(node, prefix=""):
        for key, value in node.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                visit(value, name)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite diagnostic scalar: {name}")
                values.setdefault(name, []).append(float(value))

    for row in rows:
        visit(row["metrics"])
    return {name: {"count": len(v), "mean": statistics.mean(v),
                   "std_population": statistics.pstdev(v),
                   "min": min(v), "max": max(v)} for name, v in values.items()}


def invariance_example(torch, criterion):
    """An exact conditional invariance, with the production loss's smoothing."""
    teacher = torch.tensor([[0.6, 0.3, 0.05, 0.05]], dtype=torch.float32)
    logits = torch.tensor([[2.0, 1.0, -1.0, -2.0]], dtype=torch.float32)
    changed = logits + torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    before, after = logits.softmax(-1), changed.softmax(-1)
    confidence = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    matches = torch.arange(4, dtype=torch.long).unsqueeze(0)
    keep = torch.ones_like(teacher)
    before_loss = float(criterion(before, teacher, keep, matches, confidence)["loss"])
    after_loss = float(criterion(after, teacher, keep, matches, confidence)["loss"])
    target = (2 / 3, 1 / 3)
    conditional = (math.exp(1) / (math.exp(1) + 1), 1 / (math.exp(1) + 1))
    exact = sum(p * math.log(p / q) for p, q in zip(target, conditional))
    tolerance = 5e-6
    if abs(before_loss - after_loss) > tolerance:
        raise ValueError("Toy production KL changed beyond expected FP32/epsilon tolerance")
    mass_before = float(before[0, :2].sum())
    mass_after = float(after[0, :2].sum())
    if not mass_after < mass_before / 100:
        raise ValueError("Toy did not sufficiently decrease trusted attention mass")
    return {
        "teacher_raw_patch_probabilities": teacher[0].tolist(),
        "confidence": confidence[0].tolist(), "teacher_keep": keep[0].tolist(),
        "match_index": matches[0].tolist(),
        "logits_before": logits[0].tolist(), "logits_after": changed[0].tolist(),
        "student_raw_probabilities_before": before[0].tolist(),
        "student_raw_probabilities_after": after[0].tolist(),
        "teacher_conditional_probability": list(target),
        "student_exact_conditional_probability_before_and_after": list(conditional),
        "exact_unsmoothed_KL_before_and_after": exact,
        "current_loss_KL_before": before_loss, "current_loss_KL_after": after_loss,
        "current_loss_absolute_difference": abs(before_loss - after_loss),
        "current_loss_epsilon": 1e-8, "numerical_tolerance": tolerance,
        "raw_trusted_support_mass_before": mass_before,
        "raw_trusted_support_mass_after": mass_after,
        "trusted_mass_ratio_after_to_before": mass_after / mass_before,
        "conclusion": "Increasing only untrusted logits leaves the exact conditional KL "
                      "unchanged while removing attention mass from trusted patches. "
                      "The production loss differs only by additive smoothing/FP32 error.",
    }


def main():
    args = arguments()
    if args.device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    # Torch is deliberately imported only after CLI validation.
    import torch
    from weather_vpr.data.correspondence import correspondence, load_calibration
    from weather_vpr.data.paired_dataset import AcceptedWeatherPairs
    from weather_vpr.data.patch_visibility import (
        patch_visibility_from_image_tensor, visibility_metadata,
    )
    from weather_vpr.data.student_augmentation import sample_augmentation_seed
    from weather_vpr.inference.loading import load_weather_model
    from weather_vpr.losses.attention import trusted_attention_alignment
    from weather_vpr.models.attention import capture_cls_attention
    from weather_vpr.utils.backend import enable_backend_imports

    helper = shared_helpers()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    enable_backend_imports()
    from workflow.model import input_preprocessing, load_checkpoint_model

    directory = helper.project_file(args.adapter_dir)
    checkpoint = helper.project_file(args.checkpoint)
    backbone_repo = helper.project_file(args.backbone_repo)
    manifest = helper.project_file(args.manifest)
    config, camera, config_path = helper.load_training_configuration(directory)
    metadata_path = directory / "adapter.json"
    weights_path = directory / "adapter.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("adapter_type") != "full" or metadata.get("weather") != "night":
        raise ValueError("Expected an independent full Night DINOv2 artifact")
    checkpoint_hash = helper.sha256(checkpoint)
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Night artifact does not match the Day checkpoint")
    if config.get("patch_visibility") != "photometric":
        raise ValueError("This diagnostic requires the actual photometric visibility configuration")
    importance = config.get("importance_mode")
    if importance not in ("uniform", "saliency"):
        raise ValueError("Unknown training importance mode")
    calibration_path = helper.project_file(config["pair_calibration"])
    calibration_hash = helper.sha256(calibration_path)
    if metadata.get("pair_calibration_sha256") != calibration_hash:
        raise ValueError("Night artifact calibration hash differs from its training configuration")

    base = load_checkpoint_model(checkpoint, device=args.device, backbone_repo=backbone_repo)
    preprocessing = input_preprocessing(base.image_size)
    if metadata.get("model_config") != base.config or metadata.get("preprocessing") != preprocessing:
        raise ValueError("Night artifact model configuration/preprocessing differs")
    if tuple(base.image_size) != (224, 224):
        raise ValueError("Expected current Stage2B 224 x 224 model")
    net = load_weather_model(base, "night", directory, metadata, device=args.device)
    net.float().eval().requires_grad_(False)
    dataset = AcceptedWeatherPairs(manifest, "night", base.image_size,
                                  verify_hashes=True, night_augmentation=camera, seed=42)
    dataset.set_epoch(0)
    separation = helper.validate_held_out(dataset, config)
    if len(dataset) < args.pairs:
        raise ValueError(f"Only {len(dataset)} accepted held-out pairs; requested {args.pairs}")
    calibration = load_calibration(calibration_path, checkpoint, base.image_size,
                                   int(config.get("calibration_index", 0)))
    if calibration is None:
        raise ValueError("Stage2B calibration is required; no identity correspondence fallback")

    def forward(image, *, student=False, auxiliary=False):
        backbone = net.adapters["night"] if student else net.base.backbone
        with capture_cls_attention(backbone) as capture:
            output = (net.student(image, "night", return_aux=auxiliary) if student
                      else net.teacher(image, return_aux=auxiliary))
        attention = capture.attention.detach().float()
        if (attention.shape != (1, 256) or not torch.isfinite(attention).all()
                or (attention < 0).any() or attention.sum() <= 0):
            raise ValueError("Expected finite, nonzero genuine last-block CLS patch attention")
        return output, attention

    def attention_metrics(attention, support, weights):
        total = attention.sum()
        raw_mass = attention[support].sum()
        return {
            "raw_total_patch_mass": float(total),
            "raw_unweighted_trusted_support_mass": float(raw_mass),
            "patch_normalized_unweighted_trusted_support_mass": float(raw_mass / total),
            "raw_weighted_trusted_mass": float((attention * weights).sum()),
        }

    rows = []
    # Four one-image forwards per pair, FP32 without AMP or any backward graph.
    with torch.no_grad(), torch.autocast(device_type=args.device, enabled=False):
        toy = invariance_example(torch, trusted_attention_alignment)
        for index in range(args.pairs):
            sample = dataset[index]
            row = sample["row"]
            if not row.get("source_sha256") or not row.get("output_sha256"):
                raise ValueError(f"Held-out row {index} lacks mandatory image provenance hashes")
            day = sample["day_tensor"].unsqueeze(0).to(args.device, dtype=torch.float32)
            clean = sample["weather_tensor"].unsqueeze(0).to(args.device, dtype=torch.float32)
            degraded = sample["student_tensor"].unsqueeze(0).to(args.device, dtype=torch.float32)
            day_output, day_attention = forward(day, auxiliary=True)
            clean_output, clean_attention = forward(clean, auxiliary=True)
            _, day_aux = day_output
            _, clean_aux = clean_output
            match, original_q = correspondence([sample], day_aux, clean_aux,
                                               calibration, dataset.manifest)
            match = match.detach().to(args.device)
            original_q = original_q.detach().float().to(args.device)
            visibility = patch_visibility_from_image_tensor(degraded).flatten(1)
            q = (original_q * visibility).detach()
            _, before_attention = forward(degraded)
            _, after_attention = forward(degraded, student=True)
            keep = day_aux["assignment_keep"].flatten(1).detach().float()
            if match.shape != q.shape or q.shape != day_attention.shape or keep.shape != q.shape:
                raise ValueError("Correspondence, keep, visibility and attention grids differ")
            in_range = (match >= 0) & (match < keep.shape[1])
            safe = match.clamp(0, keep.shape[1] - 1)
            q = q * in_range
            trusted = q > 0
            weights = q * (keep.gather(1, safe) if importance == "saliency" else 1)
            weighted_support = weights > 0
            weight_sum = weights.sum()
            effective = float(weight_sum.square() / weights.square().sum()) if weight_sum > 0 else 0.0
            unique_day = safe[trusted].unique()
            teacher_unique_mass = day_attention[0, unique_day].sum()
            before = trusted_attention_alignment(before_attention, day_attention, keep,
                                                match, q, importance_mode=importance)
            after = trusted_attention_alignment(after_attention, day_attention, keep,
                                               match, q, importance_mode=importance)
            metrics = {
                "patch_count": int(q.numel()),
                "stage2b_a_patches_before_visibility": int(((original_q > 0) & in_range).sum()),
                "trusted_patches_after_visibility": int(trusted.sum()),
                "local_coverage": float(trusted.float().mean()),
                "positive_confidence_mass_before_visibility": float((original_q * in_range).sum()),
                "positive_confidence_mass_after_visibility": float(q.sum()),
                "mean_actual_input_visibility": float(visibility.mean()),
                "effective_weighted_support_patches": effective,
                "weighted_support_patch_count": int(weighted_support.sum()),
                "weight_sum": float(weight_sum),
                "unique_matched_day_patches": int(unique_day.numel()),
                "many_to_one_match_repeats": int(trusted.sum()) - int(unique_day.numel()),
                "teacher_day_raw_mass_on_unique_matched_day_patches": float(teacher_unique_mass),
                "teacher_day_patch_normalized_mass_on_unique_matched_day_patches":
                    float(teacher_unique_mass / day_attention.sum()),
                "teacher_day_repeated_mapped_weighted_mass":
                    float((day_attention.gather(1, safe) * weights).sum()),
                "teacher_clean_qwen": attention_metrics(clean_attention, trusted, weights),
                "original_day_model_on_degraded_qwen": attention_metrics(before_attention, trusted, weights),
                "trained_night_model_on_degraded_qwen": attention_metrics(after_attention, trusted, weights),
                "conditional_KL_teacher_day_to_original_degraded": float(before["loss"]),
                "conditional_KL_teacher_day_to_trained_degraded": float(after["loss"]),
                "conditional_KL_change_after_minus_before": float(after["loss"] - before["loss"]),
                "current_loss_valid_query": int(after["attention_valid_queries"]),
                "current_loss_coverage": float(after["attention_coverage"]),
                "raw_trusted_mass_change_after_minus_before":
                    float(after_attention[trusted].sum() - before_attention[trusted].sum()),
            }
            rows.append({
                "validation_index": index, "sample_id": row.get("sample_id"),
                "city": str(row["city"]), "place_id": str(row["place_id"]),
                "day_path": str(row["source_file"]), "day_sha256": helper.sha256(row["source_file"]),
                "qwen_path": str(row["target_file"]), "qwen_sha256": helper.sha256(row["target_file"]),
                "augmentation_seed": sample_augmentation_seed(42, 0, index),
                "augmentation_record": sample["augmentation_record"], "metrics": metrics,
                "stage2b_a_match_index": match[0].cpu().tolist(),
                "stage2b_a_confidence_before_visibility": original_q[0].cpu().tolist(),
                "actual_degraded_patch_visibility": visibility[0].cpu().tolist(),
                "effective_confidence_after_visibility": q[0].cpu().tolist(),
                "teacher_keep": keep[0].cpu().tolist(), "loss_weights": weights[0].cpu().tolist(),
                "raw_cls_to_patch_attention": {
                    "teacher_clean_day": day_attention[0].cpu().tolist(),
                    "teacher_clean_qwen": clean_attention[0].cpu().tolist(),
                    "original_day_model_on_degraded_qwen": before_attention[0].cpu().tolist(),
                    "trained_night_model_on_degraded_qwen": after_attention[0].cpu().tolist(),
                },
            })
            del day_output, clean_output, day_aux, clean_aux
            print(f"Held-out pair {index + 1}/{args.pairs}: "
                  f"trusted={metrics['trusted_patches_after_visibility']}/256, "
                  f"KL {metrics['conditional_KL_teacher_day_to_original_degraded']:.6f}"
                  f" -> {metrics['conditional_KL_teacher_day_to_trained_degraded']:.6f}", flush=True)

    valid_rows = [r for r in rows if r["metrics"]["current_loss_valid_query"]]
    payload = {
        "schema_version": 1, "scope": "Read-only held-out synthetic Day/Qwen diagnostic; no training",
        "sampling": "First accepted validation pairs in manifest order; no coverage or image selection",
        "pair_count": len(rows), "accepted_heldout_pairs": len(dataset),
        "manifest": str(manifest), "manifest_sha256": helper.sha256(manifest),
        "place_separation": separation, "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "night_weights": str(weights_path),
        "night_weights_sha256": helper.sha256(weights_path),
        "night_metadata_sha256": helper.sha256(metadata_path),
        "training_configuration_sha256": helper.sha256(config_path),
        "stage2b_calibration": str(calibration_path), "stage2b_calibration_sha256": calibration_hash,
        "stage2b_calibration_index": int(config.get("calibration_index", 0)),
        "script_sha256": helper.sha256(Path(__file__)), "backbone_repo": str(backbone_repo),
        "preprocessing": preprocessing, "precision": "FP32; no autocast; torch.no_grad",
        "seed": 42, "augmentation_epoch": 0, "camera_configuration": camera.to_dict(),
        "visibility": visibility_metadata(), "importance_mode": importance,
        "no_real_night_training": "No optimization is performed. The selected training artifact "
            "references its accepted Day/Qwen training manifest. This diagnostic reads only "
            "held-out Day/Qwen pairs; it does not read or adapt to real-night test images.",
        "definitions": {
            "trusted_support": "Valid Stage2B A match with positive_conf * actual degraded visibility > 0",
            "weight": "positive_conf * actual degraded visibility * matched Teacher keep "
                      "in saliency mode; keep omitted in uniform mode",
            "effective_weighted_support_patches": "(sum(weight))^2 / sum(weight^2); zero if no weight",
            "raw_unweighted_trusted_support_mass": "Sum of genuine CLS softmax probabilities on "
                "trusted Night patches; not confidence/keep weighted; includes original "
                "CLS/register-token normalization",
            "patch_normalized_unweighted_trusted_support_mass": "Raw trusted mass divided by "
                "total raw attention to all patch tokens",
            "raw_weighted_trusted_mass": "Sum(raw attention * fixed loss weight); this is "
                "the small weighted mass reported by training, not the unweighted attention mass",
            "teacher_unique_day_mass": "Teacher attention summed once per unique matched Day patch",
            "conditional_KL": "Current trusted_attention_alignment: mapped Day attention and "
                "Student attention are each weighted and normalized within trusted support",
            "std_population": "Population standard deviation across every fixed selected pair",
        },
        "conditional_support_invariance_example": toy,
        "summary_all_fixed_pairs": numeric_summary(rows),
        "conditional_KL_valid_pairs": len(valid_rows),
        "summary_valid_conditional_KL_pairs": numeric_summary(valid_rows) if valid_rows else {},
        "caveats": ["Attention is not building segmentation or a causal explanation of retrieval.",
                    "Matched support is verified on clean Day/Qwen geometry, never assumed by grid index.",
                    "Conditional KL measures shape within the selected support and leaves total "
                    "attention mass outside that support unconstrained.",
                    "A small confidence/keep-weighted mass does not imply small raw attention mass.",
                    "Held-out camera-degraded Qwen images are not real-night evaluation images."],
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}; conditional KL valid pairs {len(valid_rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()

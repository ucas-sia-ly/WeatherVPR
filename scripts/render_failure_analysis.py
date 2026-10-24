#!/usr/bin/env python3
"""Render saved Night-VPR diagnostics without loading descriptors or models.

Input is the analysis JSON produced by the failure-analysis workflow. The
renderer opens only its explicitly selected image files and saved patch maps.
It does not rerank retrievals, select checkpoints, or infer semantic labels.
Existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHOD_ORDER = (
    "day", "source_night", "late4", "global_control", "sgpd", "sgpd_attention",
)
METHOD_LABELS = {
    "day": "Original Day",
    "source_night": "Source Night",
    "late4": "Late 4",
    "global_control": "Global control",
    "sgpd": "+ SGPD",
    "sgpd_attention": "+ SGPD + attention",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def selected_cases(analysis: dict) -> list[dict]:
    cases = analysis.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("analysis.cases must be a nonempty list")
    by_id = {}
    query_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query_id"), str):
            raise ValueError("Every case must have a string query_id")
        if case["query_id"] in query_ids:
            raise ValueError(f"Duplicate query_id: {case['query_id']}")
        query_ids.add(case["query_id"])
        for identifier in (case["query_id"], case.get("case_id", case["query_id"])):
            if not isinstance(identifier, str):
                raise ValueError("Case identifiers must be strings")
            if identifier in by_id and by_id[identifier] is not case:
                raise ValueError(f"Ambiguous case identifier: {identifier}")
            by_id[identifier] = case
    identifiers = analysis.get("figure_case_ids")
    if identifiers is None:
        identifiers = [case.get("case_id", case["query_id"])
                       for case in cases if case.get("visualize") is True]
    if not isinstance(identifiers, list) or not identifiers:
        raise ValueError("Provide figure_case_ids or cases with visualize=true")
    if any(not isinstance(value, str) for value in identifiers):
        raise ValueError("figure_case_ids must contain strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("figure_case_ids must not contain duplicates")
    try:
        return [by_id[value] for value in identifiers]
    except KeyError as error:
        raise ValueError(f"Selected query_id is missing: {error.args[0]}") from error


def number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a finite number for {context}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite number for {context}")
    return value


def rank_text(method: dict) -> str:
    value = method.get("rank")
    if value is None:
        return "positive rank unavailable"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Positive ranks must be one-based positive integers or null")
    return f"positive rank {value}"


def patch_map(value, context: str, *, probability: bool = False, signed: bool = False):
    import numpy as np

    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{context} must be a finite two-dimensional array")
    if not signed and (result < 0).any():
        raise ValueError(f"{context} must be nonnegative")
    if probability:
        total = float(result.sum())
        if total <= 0:
            raise ValueError(f"{context} has zero attention mass")
        result = result / total
    return result


def cohorts_text(case: dict) -> str:
    cohorts = case.get("cohorts", case.get("cohort", []))
    if isinstance(cohorts, str):
        cohorts = [cohorts]
    if not isinstance(cohorts, list) or any(not isinstance(item, str) for item in cohorts):
        raise ValueError("Case cohorts must be a string or list of strings")
    return ", ".join(cohorts) or "selected diagnostic case"


def image_rgb(path: Path, image_size: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    with Image.open(path) as original:
        with original.convert("RGB") as rgb:
            if image_size is None:
                return np.asarray(rgb).copy()
            # Saved maps describe the model's stretched RGB input, not a crop
            # or an EXIF-rotated display of the encoded photograph.
            with rgb.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR) as resized:
                return np.asarray(resized).copy()


def short_identifier(value: str, maximum: int = 68) -> str:
    if len(value) <= maximum:
        return value
    return "..." + value[-(maximum - 3):]


def case_filename(index: int, query_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", query_id).strip("._-")[-64:]
    suffix = suffix or hashlib.sha256(query_id.encode()).hexdigest()[:12]
    return f"case_{index + 1:02d}_{suffix}"


def save_figure(figure, path: Path) -> dict:
    import matplotlib.pyplot as plt

    try:
        with path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=160, facecolor="white")
    finally:
        plt.close(figure)
    return {"path": str(path), "sha256": sha256(path)}


def render_retrieval(case: dict, methods: list[str], path: Path) -> dict:
    import matplotlib.pyplot as plt

    columns = 4
    cell_count = len(methods) + 2
    rows = math.ceil(cell_count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16.4, 3.8 * rows + 0.8),
                                squeeze=False, layout="constrained")
    cells = list(axes.flat)
    original = case["methods"]["day"]
    entries = [
        ("Night query", case["query_path"], None),
        ("Fixed Day-model positive\n(positive by evaluation protocol)",
         case.get("fixed_positive_ref_path", original["pos_ref_path"]), None),
    ]
    for name in methods:
        method = case["methods"][name]
        margin = number(method["margin"], f"{case['query_id']} / {name} / margin")
        title = (f"{METHOD_LABELS.get(name, name)}: top-1\n"
                 f"{rank_text(method)} | margin {margin:+.4f}")
        entries.append((title, method["top1_ref_path"], method.get("rank") == 1))
    for axis, (title, value, success) in zip(cells, entries):
        axis.imshow(image_rgb(project_file(value)))
        axis.set_title(title, fontsize=10, color="#15763b" if success is True
                       else "#b42a2a" if success is False else "#222222")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(short_identifier(Path(value).name, 52), fontsize=8)
        if success is not None:
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#15763b" if success else "#b42a2a")
                spine.set_linewidth(2.0)
    for axis in cells[len(entries):]:
        axis.set_axis_off()
    figure.suptitle(
        f"{short_identifier(case['query_id'], 100)}\n"
        f"{textwrap.fill(cohorts_text(case), width=145)}\n"
        "Positive/negative labels follow benchmark coordinates; they are not visual judgments. "
        "Margin = nearest negative distance - nearest positive distance (descriptor L2).", fontsize=11)
    return save_figure(figure, path)


def render_maps(case: dict, methods: list[str], image_size: tuple[int, int], path: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np

    picture_path = project_file(case.get("preprocessed_query_path", case["query_path"]))
    pixels = image_rgb(picture_path, image_size)
    collections = []
    for key in ("attention_map", "keep_map"):
        collections.append([patch_map(case["methods"][name][key],
                                      f"{case['query_id']} / {name} / {key}",
                                      probability=key == "attention_map")
                            for name in methods])
    signed_maps = ["signed_margin_map" in case["methods"][name] for name in methods]
    if any(signed_maps) and not all(signed_maps):
        raise ValueError("Signed margin maps must be supplied for every method or none")
    if all(signed_maps):
        collections.append([patch_map(case["methods"][name]["signed_margin_map"],
                                      f"{case['query_id']} / {name} / signed_margin_map",
                                      signed=True) for name in methods])
    scales = []
    for row_index, values in enumerate(collections):
        maximum = max(float(np.abs(value).max()) if row_index == 2 else float(value.max())
                      for value in values)
        scales.append(maximum if maximum > 0 else 1.0)
    figure, axes = plt.subplots(len(collections), len(methods),
                                figsize=(3.05 * len(methods) + 1.1, 3.25 * len(collections) + 1.2),
                                squeeze=False, layout="constrained")
    for row_index, (values, maximum) in enumerate(zip(collections, scales)):
        normalizer = (TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum) if row_index == 2
                      else Normalize(vmin=0, vmax=maximum, clip=False))
        for column, (name, value) in enumerate(zip(methods, values)):
            axis = axes[row_index, column]
            axis.imshow(pixels)
            heatmap = axis.imshow(value, cmap="RdBu_r" if row_index == 2 else "magma", norm=normalizer,
                                  alpha=0.58, interpolation="nearest",
                                  extent=(0, image_size[1], image_size[0], 0))
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"{METHOD_LABELS.get(name, name)}\n"
                               f"{rank_text(case['methods'][name])}", fontsize=10)
            if column == 0:
                axis.set_ylabel(("CLS-to-patch attention", "SALAD assignment keep",
                                 "Fixed-contrast patch contribution")[row_index], fontsize=10)
            if row_index == 2:
                method = case["methods"][name]
                caption = f"patch sum {float(value.sum()):+.4f}"
                if "cls_contribution" in method:
                    caption += f" | CLS {method['cls_contribution']:+.4f}"
                if "fixed_cosine_margin" in method:
                    caption += f"\nfixed cosine margin {method['fixed_cosine_margin']:+.4f}"
            elif row_index == 0:
                positive = value[value > 0]
                entropy = float(-(positive * np.log(positive)).sum())
                normalized_entropy = entropy / math.log(value.size) if value.size > 1 else 0.0
                caption = f"max {float(value.max()):.4f} | normalized entropy {normalized_entropy:.3f}"
            else:
                caption = (f"max {float(value.max()):.4f} | "
                           f"mean {float(np.mean(value)):.4f}")
            axis.set_xlabel(caption, fontsize=8)
        colorbar = figure.colorbar(heatmap, ax=list(axes[row_index]), shrink=0.75,
                                  fraction=0.016, pad=0.012)
        colorbar.set_label(("Patch probability", "Keep mass",
                            "Positive - negative cosine contribution")[row_index], fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"{short_identifier(case['query_id'], 110)}\n"
        "Same Night query, six encoders | one absolute color scale per row, no percentile clipping\n"
        "CLS attention sums to 1. Keep uses saved probabilities. "
        "Signed map: red supports fixed GT, blue supports fixed nonpositive.\n"
        "Patch + CLS contributions sum to the fixed positive-minus-negative cosine margin, "
        "conditional on current features and OT; no causal or semantic claim.", fontsize=11)
    artifact = save_figure(figure, path)
    artifact["scales"] = {
        "attention": {"vmin": 0, "vmax": scales[0], "patch_normalized": True},
        "keep": {"vmin": 0, "vmax": scales[1], "patch_normalized": False},
        "interpolation": "nearest", "overlay_alpha": 0.58,
    }
    if len(collections) == 3:
        artifact["scales"]["signed_margin"] = {
            "vmin": -scales[2], "vmax": scales[2], "vcenter": 0,
            "red": "supports fixed positive", "blue": "supports fixed nonpositive",
        }
    return artifact


def render_margins(cases: list[dict], methods: list[str], path: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    values = np.asarray([[number(case["methods"][name]["margin"], "margin")
                          for name in methods] for case in cases], dtype=np.float64)
    maximum = max(float(np.abs(values).max()), 1e-8)
    figure, axis = plt.subplots(figsize=(max(9, 1.85 * len(methods)), 0.7 * len(cases) + 2.5),
                                layout="constrained")
    heatmap = axis.imshow(values, cmap="RdYlGn", aspect="auto",
                          norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum))
    axis.set_xticks(range(len(methods)), [METHOD_LABELS.get(name, name) for name in methods],
                    rotation=15, ha="right", fontsize=10)
    axis.set_yticks(range(len(cases)), [f"{index + 1:02d}. {short_identifier(case['query_id'], 42)}"
                                      for index, case in enumerate(cases)], fontsize=9)
    for row, case in enumerate(cases):
        for column, name in enumerate(methods):
            method = case["methods"][name]
            rank = method.get("rank")
            axis.text(column, row, f"{values[row, column]:+.4f}\nrank {rank if rank else '?'}",
                      ha="center", va="center", fontsize=9, color="black")
    colorbar = figure.colorbar(heatmap, ax=axis, shrink=0.9)
    colorbar.set_label("Nearest negative L2 - nearest positive L2")
    axis.set_title("Selected contrasting cases: descriptor margins\n"
                   "These cases were selected for diagnosis; this is not a population-level result.",
                   fontsize=12)
    return save_figure(figure, path)


def validate(analysis: dict, cases: list[dict], methods: list[str]) -> tuple[int, int]:
    preprocessing = analysis.get("map_preprocessing", {})
    image_size = preprocessing.get("image_size", [224, 224])
    if (not isinstance(image_size, list) or len(image_size) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for size in image_size)):
        raise ValueError("map_preprocessing.image_size must contain two positive integers")
    if preprocessing.get("resize", "bilinear_stretch") != "bilinear_stretch":
        raise ValueError("Only bilinear_stretch map preprocessing is supported")
    if preprocessing.get("exif_transpose", False) is not False:
        raise ValueError("EXIF-transposed map inputs are not supported by this renderer")
    for case in cases:
        project_file(case["query_path"])
        cohorts_text(case)
        if "preprocessed_query_path" in case:
            project_file(case["preprocessed_query_path"])
        if "fixed_positive_ref_path" in case:
            project_file(case["fixed_positive_ref_path"])
        signed_maps = ["signed_margin_map" in case["methods"][name] for name in methods]
        if any(signed_maps) and not all(signed_maps):
            raise ValueError("Signed margin maps must be supplied for every method or none")
        for name in methods:
            method = case["methods"][name]
            rank_text(method)
            number(method["margin"], "margin")
            project_file(method["top1_ref_path"])
            project_file(method["pos_ref_path"])
            attention = patch_map(method["attention_map"], "attention_map", probability=True)
            keep = patch_map(method["keep_map"], "keep_map")
            if attention.shape != keep.shape:
                raise ValueError("Attention and Keep maps must use the same patch grid")
            if "patch_norm_map" in method:
                patch_norm = patch_map(method["patch_norm_map"], "patch_norm_map")
                if patch_norm.shape != attention.shape:
                    raise ValueError("Patch norm map must use the same patch grid")
            if "signed_margin_map" in method:
                contribution = patch_map(method["signed_margin_map"], "signed_margin_map", signed=True)
                if contribution.shape != attention.shape:
                    raise ValueError("Signed margin map must use the same patch grid")
                if "cls_contribution" in method and "fixed_cosine_margin" in method:
                    total = float(contribution.sum()) + number(method["cls_contribution"], "CLS contribution")
                    expected = number(method["fixed_cosine_margin"], "fixed cosine margin")
                    if not math.isclose(total, expected, rel_tol=1e-4, abs_tol=1e-5):
                        raise ValueError("Patch + CLS contributions do not match the fixed cosine margin")
    return tuple(image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    analysis_path = project_file(args.analysis)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output}")
    with analysis_path.open(encoding="utf-8") as stream:
        analysis = json.load(stream)
    if not isinstance(analysis, dict):
        parser.error("Analysis JSON must be an object")
    cases = selected_cases(analysis)
    methods = analysis.get("method_order", list(DEFAULT_METHOD_ORDER))
    if (not isinstance(methods, list) or not methods or "day" not in methods
            or any(not isinstance(name, str) or not name for name in methods)
            or len(methods) != len(set(methods))):
        parser.error("method_order must contain unique method names, including day")

    import matplotlib
    matplotlib.use("Agg")

    image_size = validate(analysis, cases, methods)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    image_files = {}
    for index, case in enumerate(cases):
        filename = case_filename(index, case["query_id"])
        artifacts.append({
            "case_id": case.get("case_id"), "query_id": case["query_id"],
            "retrieval": render_retrieval(case, methods, output / f"{filename}_retrieval.png"),
            "maps": render_maps(case, methods, image_size, output / f"{filename}_maps.png"),
        })
        for value in [case["query_path"], case.get("preprocessed_query_path"),
                      case.get("fixed_positive_ref_path")]:
            if value:
                path = project_file(value)
                image_files[str(path)] = sha256(path)
        for method in methods:
            for key in ("top1_ref_path", "pos_ref_path"):
                path = project_file(case["methods"][method][key])
                if str(path) not in image_files:
                    image_files[str(path)] = sha256(path)
        print(f"Rendered {index + 1}/{len(cases)}: {case['query_id']}", flush=True)
    margin_artifact = render_margins(cases, methods, output / "selected_case_margins.png")
    manifest = {
        "format_version": 1,
        "analysis_path": str(analysis_path), "analysis_sha256": sha256(analysis_path),
        "renderer_path": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "method_order": methods,
        "figure_case_ids": [case.get("case_id", case["query_id"]) for case in cases],
        "map_preprocessing": {"image_size": list(image_size), "resize": "bilinear_stretch",
                              "exif_transpose": False},
        "selection": "Input figure_case_ids, or input cases explicitly marked visualize=true",
        "caveats": [
            "The renderer performs no inference or retrieval selection.",
            "Positive labels follow the input evaluation protocol, not visual judgments.",
            "Attention and Keep are distinct, and neither is semantic segmentation.",
            "Maps are descriptive evidence; they do not by themselves establish causation.",
            "Selected contrasting cases are not an estimate of full-dataset behavior.",
        ],
        "cases": artifacts, "selected_case_margins": margin_artifact,
        "image_sha256": image_files,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Synthetic night validation on exact Qwen/day pairs, separate from real SVOX.

The frozen day SALAD gallery is shared by unadapted and adapted night queries.
Recall positives use (city, place_id); paired cosine uses each query's exact
source image, even when several day images belong to the same place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from weather_vpr.data.night_augmentation import NightCameraAugmentation, NightAugmentationConfig
from weather_vpr.data.paired_dataset import resolve_file, sha256
from weather_vpr.data.student_augmentation import student_training_tensor
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.inference.loading import load_weather_model
from weather_vpr.utils.backend import enable_backend_imports, project_root


def _require_night_adapter_dir(value: Path) -> Path:
    """Check run artifacts before loading a model or reading validation images."""
    directory = Path(value).expanduser().resolve()
    missing = [name for name in ("adapter.json", "adapter.safetensors")
               if not (directory / name).is_file()]
    if directory.is_dir() and not missing:
        return directory
    if not directory.exists():
        problem = "directory does not exist"
        next_step = ("Create it by training: OUTPUT=runs/night_full_sgpd_01 "
                     "bash configs/night_full_sgpd.sh.")
    elif not directory.is_dir():
        problem = "path is not a directory"
        next_step = "Pass a completed training output directory via --adapter-dir."
    else:
        problem = "incomplete directory; missing " + ", ".join(missing)
        next_step = ("Training may still be running; wait until both adapter.json and "
                     "adapter.safetensors are saved. If it stopped, start a new run in an "
                     "unused directory, for example OUTPUT=runs/night_full_sgpd_02 "
                     "bash configs/night_full_sgpd.sh.")
    smoke = project_root() / "runs/night_sgpd_smoke_20261021"
    smoke_hint = (
        "For a quick pipeline check, use --adapter-dir runs/night_sgpd_smoke_20261021 "
        "(a short smoke run, not a trained result)."
        if (smoke / "adapter.json").is_file() and (smoke / "adapter.safetensors").is_file()
        else "For a quick pipeline check, create a smoke adapter with "
             "OUTPUT=runs/night_full_global_smoke bash configs/night_full_global.sh."
    )
    raise FileNotFoundError(
        f"Night adapter {problem}: {directory}.\n"
        f"{next_step}\n" + smoke_hint
    )


def load_pair_validation(manifest: Path, *, limit_pairs: int | None = None,
                         verify_hashes: bool = True,
                         distractor_manifest: Path | None = None):
    """Build deterministic gallery/query metadata from an explicit night split."""
    enable_backend_imports()
    from workflow.evaluation import EvaluationImage, EvaluationSet

    manifest = Path(manifest).expanduser().resolve(strict=True)
    if limit_pairs is not None and limit_pairs < 1:
        raise ValueError("limit-pairs must be positive")
    rows = []
    seen_samples = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("condition") != "night" or row.get("passed") is not True or
                row.get("eligible_for_training") is not True or
                row.get("accepted", True) is not True):
            raise ValueError(f"Line {line_number}: validation split contains a nonaccepted night row")
        sample = row.get("sample_id")
        if not isinstance(sample, str) or not sample or sample in seen_samples:
            raise ValueError(f"Line {line_number}: missing or duplicate sample_id")
        seen_samples.add(sample)
        if row.get("city") in (None, "") or row.get("place_id") in (None, ""):
            raise ValueError(f"Line {line_number}: missing city/place_id")
        source = resolve_file(row["source_path"], manifest)
        generated = resolve_file(row["output_path"], manifest)
        if source == generated:
            raise ValueError(f"Line {line_number}: source and generated are identical")
        if verify_hashes:
            for key, image in (("source_sha256", source), ("output_sha256", generated)):
                if row.get(key) is None or sha256(image) != row[key]:
                    raise ValueError(f"Line {line_number}: missing or incorrect {key}: {image}")
        rows.append((sample, (str(row["city"]), str(row["place_id"])), source, generated))
    if not rows:
        raise ValueError(f"Empty night validation manifest: {manifest}")
    validation_places = {row[1] for row in rows}
    validation_sources = {row[2] for row in rows}
    rows.sort(key=lambda value: value[0])
    if limit_pairs is not None:
        rows = rows[:limit_pairs]

    references = []
    queries = []
    source_index = {}
    source_place = {}
    query_paths = set()
    exact_source_indices = []
    places = []
    for sample, place, source, generated in rows:
        if source in source_place and source_place[source] != place:
            raise ValueError(f"Source image has inconsistent city/place_id: {source}")
        if source not in source_index:
            source_index[source] = len(references)
            source_place[source] = place
            references.append(EvaluationImage(f"day/{len(references):06d}", source))
        if generated in query_paths:
            raise ValueError(f"Generated image appears in two validation rows: {generated}")
        query_paths.add(generated)
        queries.append(EvaluationImage(sample, generated))
        exact_source_indices.append(source_index[source])
        places.append(place)
    place_to_indices = {}
    for source, index in source_index.items():
        place_to_indices.setdefault(source_place[source], []).append(index)
    positives = [place_to_indices[place] for place in places]
    dataset = EvaluationSet(
        "synthetic_night_pairs", references, queries, positives,
        {"ground_truth": "city_place_id", "pairing": "explicit_manifest_source_output",
         "manifest": str(manifest), "condition": "night", "selection": "sample_id_ascending",
         "limit_pairs": limit_pairs, "hashes_verified": verify_hashes},
    )
    if distractor_manifest is not None:
        _add_day_distractors(dataset, validation_places, validation_sources,
                            distractor_manifest)
    dataset.validate()
    return dataset, exact_source_indices, places


def _add_day_distractors(dataset, validation_places, validation_sources, manifest: Path) -> None:
    """Append eligible, disjoint training Day sources; never inspect generated images."""
    from workflow.evaluation import EvaluationImage

    manifest = Path(manifest).expanduser().resolve(strict=True)
    existing = {image.path.resolve() for image in dataset.references}
    sources = {}
    eligible_rows = 0
    added = 0
    for line_number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get('condition') != 'night' or row.get('passed') is not True or
                row.get('eligible_for_training') is not True or row.get('accepted', True) is not True):
            continue
        eligible_rows += 1
        if row.get('city') in (None, '') or row.get('place_id') in (None, ''):
            raise ValueError(f'Distractor line {line_number}: missing city/place_id')
        place = (str(row['city']), str(row['place_id']))
        if place in validation_places:
            raise ValueError(f'Distractor line {line_number}: training/validation place overlap: {place}')
        source = resolve_file(row['source_path'], manifest)
        if source in validation_sources:
            raise ValueError(f'Distractor line {line_number}: training/validation source overlap: {source}')
        expected_hash = row.get('source_sha256')
        if expected_hash is None or sha256(source) != expected_hash:
            raise ValueError(f'Distractor line {line_number}: missing or incorrect source_sha256: {source}')
        if source in sources and sources[source] != place:
            raise ValueError(f'Distractor source has inconsistent city/place_id: {source}')
        sources[source] = place
        if source in existing:
            continue
        existing.add(source)
        dataset.references.append(EvaluationImage(f'distractor/{added:06d}', source))
        added += 1
    if not eligible_rows:
        raise ValueError(f'No eligible night training Day distractors: {manifest}')
    dataset.protocol['distractors'] = {
        'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
        'eligible_rows': eligible_rows, 'unique_source_images': len(sources),
        'added_images': added, 'duplicate_source_rows': eligible_rows - len(sources),
        'source_hashes_verified': True, 'generated_images_read': False,
        'place_overlap_policy': 'reject against entire validation manifest, including outside a limited subset',
        'validation_places_checked': len(validation_places),
        'validation_sources_checked': len(validation_sources),
    }


def _query_augmentation_config(profile: str, config_path: Path | None,
                               strength: float, probability: float) -> NightAugmentationConfig:
    values = {}
    if config_path is not None:
        values = json.loads(Path(config_path).expanduser().resolve(strict=True).read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError('augmentation-config must contain a JSON object')
    values.update(profile=profile, strength=strength, probability=probability)
    return NightAugmentationConfig(**values)


def _query_seed(seed: int, sample_id: str) -> int:
    value = f'{seed}:{sample_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


def _query_augmentation_metadata(config: NightAugmentationConfig, seed: int,
                                  image_size, config_path: Path | None = None) -> dict:
    return {
        **config.to_dict(),
        'effective_operation_probabilities': NightCameraAugmentation(config).probabilities,
        'seed': seed, 'seed_policy': 'sha256(seed, query sample_id), first 64 bits',
        'resolution_hw': list(image_size),
        'order': 'RGB -> original SALAD bilinear resize -> fixed camera degradation -> ImageNet normalization',
        'scope': 'auxiliary synthetic development queries only; same tensor for baseline and Night; clean Day Gallery',
        'config_path': str(Path(config_path).expanduser().resolve()) if config_path is not None else None,
        'config_sha256': sha256(Path(config_path).expanduser().resolve()) if config_path is not None else None,
    }


class _FixedCameraQueries(Dataset):
    """Sample-ID seeded photometric views, independent of loader order/workers."""

    def __init__(self, images, image_size, config: NightAugmentationConfig, seed: int):
        self.images = images
        self.image_size = tuple(image_size)
        self.augmentation = NightCameraAugmentation(config)
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        with Image.open(sample.path) as image:
            tensor, _ = student_training_tensor(
                image, self.image_size, self.augmentation, seed=_query_seed(self.seed, sample.id))
        return tensor


def _extract_fixed_queries(models, images, image_size, device, batch_size,
                            num_workers, config: NightAugmentationConfig, seed: int):
    """Generate each camera batch once and send identical inputs to both models."""
    dataset = _FixedCameraQueries(images, image_size, config, seed)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=torch.device(device).type == 'cuda')
    outputs = [[] for _ in models]
    for model in models:
        model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            for model, parts in zip(models, outputs):
                output = model(batch)
                if (not isinstance(output, torch.Tensor) or output.ndim != 2 or
                        output.shape[0] != batch.shape[0] or not torch.isfinite(output).all()):
                    raise ValueError('Model must return finite [batch, descriptor_dim] descriptors')
                parts.append(output.detach().float().cpu())
    if any(not parts for parts in outputs):
        raise ValueError('Fixed query extraction requires nonempty queries')
    return tuple(torch.cat(parts, dim=0) for parts in outputs)


def _default_output_path(limit_pairs=None, query_augmentation='off', distractor_manifest=None):
    suffix = '_camera' if query_augmentation == 'camera' else ''
    if distractor_manifest is not None:
        suffix += '_distractors'
    if limit_pairs is not None:
        suffix += f'_first{limit_pairs}'
    return project_root() / 'runs' / f'night_pairs_validation{suffix}.json'


def mean_paired_cosine(gallery: torch.Tensor, queries: torch.Tensor,
                       source_indices: list[int]) -> float:
    """Mean unit-descriptor cosine against each generated image's exact day source."""
    if (gallery.ndim != 2 or queries.ndim != 2 or gallery.shape[1] != queries.shape[1]
            or len(source_indices) != len(queries) or not source_indices
            or min(source_indices) < 0 or max(source_indices) >= len(gallery)):
        raise ValueError("Invalid paired descriptors or source indices")
    source = gallery[torch.as_tensor(source_indices, dtype=torch.long)]
    cosine = (F.normalize(queries.float(), dim=1) *
              F.normalize(source.float(), dim=1)).sum(1)
    if not torch.isfinite(cosine).all():
        raise ValueError("Nonfinite paired cosine")
    return float(cosine.mean())


def _cross_view_retrieval(gallery: torch.Tensor, queries: torch.Tensor, positives,
                          source_indices, *, device: str = 'cpu', query_chunk_size: int = 32,
                          reference_chunk_size: int = 4096):
    """Exact bounded-memory L2 ranks with each query's own source removed.

    All returned reference indices retain their original Gallery indexing.
    Distances tied in FP32 use the original reference order, as in the main
    retrieval evaluator. No network forward or descriptor recomputation occurs.
    """
    if query_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError('Cross-view retrieval chunk sizes must be positive')
    gallery = torch.as_tensor(gallery).detach().float().cpu()
    queries = torch.as_tensor(queries).detach().float().cpu()
    if (gallery.ndim != 2 or queries.ndim != 2 or gallery.shape[1] != queries.shape[1]
            or not len(gallery) or len(queries) != len(positives)
            or len(queries) != len(source_indices)
            or not torch.isfinite(gallery).all() or not torch.isfinite(queries).all()):
        raise ValueError('Invalid cross-view descriptors or query metadata')
    for source, targets in zip(source_indices, positives):
        if (source < 0 or source >= len(gallery) or not targets or source in targets
                or any(index < 0 or index >= len(gallery) for index in targets)):
            raise ValueError('Cross-view positives must be valid non-source Gallery indices')
    results = []
    with torch.inference_mode():
        for start in range(0, len(queries), query_chunk_size):
            query = queries[start:start + query_chunk_size].to(device)
            targets = positives[start:start + len(query)]
            sources = torch.tensor(source_indices[start:start + len(query)], device=device)
            query_norms = query.square().sum(1, keepdim=True)
            best_distance = torch.full((len(query),), float('inf'), device=device)
            best_index = torch.full((len(query),), len(gallery), dtype=torch.long, device=device)
            predicted_distance = best_distance.clone()
            predicted_index = best_index.clone()

            def distance_blocks():
                for offset in range(0, len(gallery), reference_chunk_size):
                    reference = gallery[offset:offset + reference_chunk_size].to(device)
                    indices = torch.arange(offset, offset + len(reference), device=device)
                    distances = (query_norms + reference.square().sum(1)[None, :] -
                                 2 * query @ reference.T).clamp_min_(0)
                    distances.masked_fill_(sources[:, None] == indices[None, :], float('inf'))
                    yield offset, indices, distances

            for offset, indices, distances in distance_blocks():
                values, local_indices = distances.min(1)
                better = values < predicted_distance
                predicted_distance = torch.where(better, values, predicted_distance)
                predicted_index = torch.where(better, local_indices + offset, predicted_index)
                positive_distances = torch.full_like(distances, float('inf'))
                rows, columns = [], []
                for row, target_indices in enumerate(targets):
                    for index in target_indices:
                        if offset <= index < offset + len(indices):
                            rows.append(row)
                            columns.append(index - offset)
                if rows:
                    row_indices = torch.tensor(rows, device=device, dtype=torch.long)
                    column_indices = torch.tensor(columns, device=device, dtype=torch.long)
                    positive_distances[row_indices, column_indices] = distances[row_indices, column_indices]
                    values, local_indices = positive_distances.min(1)
                    better = values < best_distance
                    best_distance = torch.where(better, values, best_distance)
                    best_index = torch.where(better, local_indices + offset, best_index)
            ranks = torch.ones(len(query), dtype=torch.long, device=device)
            for _, indices, distances in distance_blocks():
                precedes = ((distances < best_distance[:, None]) |
                            ((distances == best_distance[:, None]) &
                             (indices[None, :] < best_index[:, None])))
                ranks += precedes.sum(1)
            for rank, correct, predicted, correct_distance, predicted_dist in zip(
                    ranks.cpu().tolist(), best_index.cpu().tolist(), predicted_index.cpu().tolist(),
                    best_distance.sqrt().cpu().tolist(), predicted_distance.sqrt().cpu().tolist()):
                results.append({'rank': rank, 'ground_truth_index': correct,
                                'predicted_index': predicted, 'distance_gt': correct_distance,
                                'distance_pred': predicted_dist})
    return results


def cross_view_results(dataset, gallery: torch.Tensor, queries: torch.Tensor, source_indices,
                       *, device: str = 'cpu', save_hard_cases: bool = False) -> dict:
    """Auxiliary, selected multi-view validation subset; no exact-source candidate."""
    enable_backend_imports()
    from workflow.evaluation import EvaluationSet, build_results

    if (len(source_indices) != len(dataset.queries) or len(queries) != len(dataset.queries)
            or len(gallery) != len(dataset.references)):
        raise ValueError('Cross-view descriptors/source indices must cover the original validation queries')
    alternatives = []
    selected = []
    for index, (targets, source) in enumerate(zip(dataset.positives, source_indices)):
        if source not in targets:
            raise ValueError('Exact Day source must be one of the original place positives')
        other_views = sorted(set(targets) - {source})
        if other_views:
            selected.append(index)
            alternatives.append(other_views)
    protocol = {
        **dataset.protocol,
        'ground_truth': 'same_city_place_id_different_day_source',
        'query_subset': 'only queries with another distinct Day source at the same city/place in validation',
        'candidate_exclusion': 'each query exact Day source removed from all ranked candidates and negatives',
        'selection_bias': 'auxiliary existing multi-view validation subset; not representative of all queries or real night',
        'descriptor_reuse': 'same frozen Day Gallery and same query descriptors as main paired validation',
        'original_queries': len(dataset.queries),
        'excluded_single_view_queries': len(dataset.queries) - len(selected),
    }
    subset = EvaluationSet(dataset.name + '_cross_view_subset', dataset.references,
                           [dataset.queries[index] for index in selected], alternatives, protocol)
    if not selected:
        return {**subset.summary(), 'evaluated_queries': 0, 'recall': None,
                'distance_metric': 'euclidean', 'rank_base': 1, 'tie_break': 'reference_order',
                'error_queries': [], 'num_error_queries': 0, 'hard_cases_saved': save_hard_cases}
    ranks = _cross_view_retrieval(gallery, queries[selected], alternatives,
                                  [source_indices[index] for index in selected], device=device)
    return {**build_results(subset, ranks, save_hard_cases=save_hard_cases),
            'evaluated_queries': len(selected)}


class _NightStudent(torch.nn.Module):
    def __init__(self, net: WeatherAdapterSALAD):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.student(images, "night", return_aux=False)


def _device(value: str) -> str:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA evaluation devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA unavailable for {value}")
    return str(device)


def evaluate_pairs(*, manifest: Path, checkpoint: Path, backbone_repo: Path,
                   adapter_dir: Path, output: Path, device: str = "auto",
                   retrieval_device: str | None = None, batch_size: int = 16,
                   num_workers: int = 2, limit_pairs: int | None = None,
                   verify_hashes: bool = True, save_hard_cases: bool = False,
                   adapter_scale: float = 1.0, distractor_manifest: Path | None = None,
                   query_augmentation: str = 'off', augmentation_config: Path | None = None,
                   augmentation_strength: float = 1.0, augmentation_probability: float = 0.8,
                   augmentation_seed: int = 20261024,
                   gallery_cache: Path | None = None) -> dict:
    """Evaluate one night adapter and its unadapted baseline on a held-out split."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    adapter_dir = _require_night_adapter_dir(adapter_dir)
    if not isinstance(augmentation_seed, int) or isinstance(augmentation_seed, bool) or augmentation_seed < 0:
        raise ValueError('augmentation-seed must be a nonnegative integer')
    query_config = _query_augmentation_config(query_augmentation, augmentation_config,
                                             augmentation_strength, augmentation_probability)
    device = _device(device)
    retrieval_device = _device(retrieval_device or device)
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    dataset, exact_indices, places = load_pair_validation(
        manifest, limit_pairs=limit_pairs, verify_hashes=verify_hashes,
        distractor_manifest=distractor_manifest)
    from workflow.evaluation import build_results, exact_retrieval, extract_descriptors
    from workflow.model import input_preprocessing, load_checkpoint_model

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone_repo = Path(backbone_repo).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if (query_augmentation != 'off' or distractor_manifest is not None) and output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if (previous.get('query_augmentation', {}).get('profile', 'off') == 'off' and
                previous.get('distractor_manifest') is None):
            raise FileExistsError(f'Refusing to overwrite clean paired validation: {output}; choose another output')
    checkpoint_hash = sha256(checkpoint)
    print(f"Synthetic night validation: {len(dataset.queries)} pairs, "
          f"{len(dataset.references)} unique day images, {len(set(places))} places", flush=True)
    base = load_checkpoint_model(checkpoint, device=device, backbone_repo=backbone_repo)
    preprocessing = input_preprocessing(base.image_size)
    query_augmentation_info = _query_augmentation_metadata(
        query_config, augmentation_seed, base.image_size, augmentation_config)
    dataset.protocol['query_augmentation'] = query_augmentation_info
    metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
    if (metadata.get("weather") != "night" or metadata.get("checkpoint_sha256") != checkpoint_hash
            or metadata.get("model_config") != base.config
            or metadata.get("preprocessing") != preprocessing):
        raise ValueError("Night adapter is incompatible with checkpoint, model configuration or preprocessing")
    net = load_weather_model(base, "night", adapter_dir, metadata,
                             device=device, adapter_scale=adapter_scale)

    gallery = None
    gallery_hash = None
    gallery_cache_hit = False
    if gallery_cache is not None:
        from weather_vpr.inference.svox import (
            _cache_metadata, _load_gallery_cache, _save_gallery_cache, gallery_fingerprint,
        )
        gallery_cache = Path(gallery_cache).expanduser().resolve()
        aggregation = base.config['agg_config']
        descriptor_dim = (aggregation['num_clusters'] * aggregation['cluster_dim'] +
                          aggregation['token_dim'])
        print('Hashing ordered Day gallery image bytes for cache validation...', flush=True)
        gallery_hash = gallery_fingerprint(dataset.references)
        expected_cache = _cache_metadata(dataset.references, gallery_hash,
                                         checkpoint_hash, preprocessing)
        gallery = _load_gallery_cache(gallery_cache, expected_cache, descriptor_dim)
        if gallery is not None:
            # Shared SVOX loading already checks dtype, shape and finiteness.
            # Reject stale/non-SALAD tensors rather than silently normalizing
            # them and changing the original retrieval space.
            norms = torch.linalg.vector_norm(gallery, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-4):
                gallery = None
            else:
                gallery_cache_hit = True
                print(f'Loaded frozen Day gallery descriptors: {gallery_cache}', flush=True)
    if gallery is None:
        print("Extracting original day gallery descriptors...", flush=True)
        gallery = extract_descriptors(base, dataset.references, base.image_size,
                                      device, batch_size, num_workers)
        if gallery_cache is not None:
            _save_gallery_cache(gallery_cache, expected_cache, gallery)
            print(f'Saved frozen Day gallery descriptors: {gallery_cache}', flush=True)
    if query_augmentation == 'camera':
        print('Extracting baseline and Night descriptors from identical fixed camera queries...', flush=True)
        baseline_queries, adapted_queries = _extract_fixed_queries(
            (base, _NightStudent(net)), dataset.queries, base.image_size, device,
            batch_size, num_workers, query_config, augmentation_seed)
    else:
        print("Extracting unadapted night query descriptors...", flush=True)
        baseline_queries = extract_descriptors(base, dataset.queries, base.image_size,
                                               device, batch_size, num_workers)
        print("Extracting adapted night query descriptors...", flush=True)
        adapted_queries = extract_descriptors(_NightStudent(net), dataset.queries,
                                              base.image_size, device, batch_size, num_workers)

    def score(queries):
        ranks = exact_retrieval(gallery, queries, dataset.positives, device=retrieval_device)
        result = build_results(dataset, ranks, save_hard_cases=save_hard_cases)
        result["mean_paired_cosine"] = mean_paired_cosine(gallery, queries, exact_indices)
        return result

    baseline = score(baseline_queries)
    adapted = score(adapted_queries)
    cross_view_baseline = cross_view_results(dataset, gallery, baseline_queries, exact_indices,
                                             device=retrieval_device, save_hard_cases=save_hard_cases)
    cross_view_adapter = cross_view_results(dataset, gallery, adapted_queries, exact_indices,
                                            device=retrieval_device, save_hard_cases=save_hard_cases)
    result = {
        "dataset": "synthetic_night_pairs", "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256(Path(manifest).resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "adapter_dir": str(adapter_dir),
        "adapter_metadata": metadata, "adapter_scale": adapter_scale,
        "preprocessing": preprocessing,
        "gallery_cache": str(gallery_cache) if gallery_cache is not None else None,
        "gallery_cache_hit": gallery_cache_hit, "gallery_sha256": gallery_hash,
        "validation_pairs": len(dataset.queries), "gallery_images": len(dataset.references),
        "places": len(set(places)), "limited_subset": limit_pairs is not None,
        "distractor_manifest": dataset.protocol.get('distractors', {}).get('manifest'),
        "distractor_manifest_sha256": dataset.protocol.get('distractors', {}).get('manifest_sha256'),
        "distractor_images": dataset.protocol.get('distractors', {}).get('added_images', 0),
        "query_augmentation": query_augmentation_info,
        "baseline": baseline, "adapter": adapted,
        "cross_view_baseline": cross_view_baseline, "cross_view_adapter": cross_view_adapter,
        "delta_recall": {key: adapted["recall"][key] - baseline["recall"][key]
                         for key in ("R@1", "R@5", "R@10")},
        "delta_mean_paired_cosine": (adapted["mean_paired_cosine"] -
                                     baseline["mean_paired_cosine"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print("Baseline:", baseline["recall"], "pair cosine:", baseline["mean_paired_cosine"], flush=True)
    print("Adapter:", adapted["recall"], "pair cosine:", adapted["mean_paired_cosine"], flush=True)
    print("Delta:", result["delta_recall"], "pair cosine:", result["delta_mean_paired_cosine"], flush=True)
    print('Auxiliary cross-view subset:', cross_view_baseline['evaluated_queries'],
          'queries; baseline:', cross_view_baseline['recall'],
          'Night:', cross_view_adapter['recall'], flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a held-out Qwen night/day split. The original day SALAD gallery "
                    "is shared by unadapted and adapted night queries; this is separate from real SVOX test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=project_root() / "data/splits/night_val.jsonl")
    parser.add_argument('--distractor-manifest', type=Path,
                        help='Eligible training manifest: append verified, disjoint Day source images only')
    parser.add_argument('--query-augmentation', choices=('off', 'camera'), default='off',
                        help='Auxiliary synthetic development corruption; fixed same query view for both models')
    parser.add_argument('--augmentation-config', type=Path)
    parser.add_argument('--augmentation-strength', type=float, default=1.0)
    parser.add_argument('--augmentation-probability', type=float, default=0.8)
    parser.add_argument('--augmentation-seed', '--seed', type=int, default=20261024)
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "weights/dino_salad.ckpt")
    parser.add_argument("--backbone-repo", type=Path, default=project_root() / "third_party/dinov2")
    parser.add_argument("--adapter-dir", type=Path,
                        help="Output directory of a trained night Adapter; required for evaluation")
    parser.add_argument("--adapter-scale", type=float, default=1.0,
                        help="LoRA update multiplier; choose only on development data")
    parser.add_argument("--output", type=Path,
                        help="Result JSON; camera, distractor and limited runs have separate default filenames")
    parser.add_argument('--gallery-cache', type=Path,
                        help='Optional reusable frozen Day gallery cache; Night queries are always recomputed')
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--retrieval-device", help="Defaults to --device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-pairs", type=int,
                        help="First N sample IDs for a smoke check; result is labelled as a subset")
    parser.add_argument("--skip-hash-verification", action="store_true",
                        help="Skip source/generated image SHA-256 checks")
    parser.add_argument("--save-hard-cases", action="store_true")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate the split and print counts without loading a model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check_data:
        dataset, _, places = load_pair_validation(
            args.manifest, limit_pairs=args.limit_pairs,
            verify_hashes=not args.skip_hash_verification, distractor_manifest=args.distractor_manifest)
        query_config = _query_augmentation_config(args.query_augmentation, args.augmentation_config,
                                                 args.augmentation_strength, args.augmentation_probability)
        if args.augmentation_seed < 0:
            raise ValueError('augmentation-seed must be a nonnegative integer')
        dataset.protocol['query_augmentation'] = _query_augmentation_metadata(
            query_config, args.augmentation_seed, (224, 224), args.augmentation_config)
        print(json.dumps({"pairs": len(dataset.queries),
                          "unique_day_images": len(dataset.references),
                          "places": len(set(places)), "protocol": dataset.protocol},
                         indent=2, ensure_ascii=False))
        return
    if args.adapter_dir is None:
        raise SystemExit("--adapter-dir is required unless --check-data is selected")
    try:
        _require_night_adapter_dir(args.adapter_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    output = args.output or _default_output_path(
        args.limit_pairs, args.query_augmentation, args.distractor_manifest)
    evaluate_pairs(
        manifest=args.manifest, checkpoint=args.checkpoint,
        backbone_repo=args.backbone_repo, adapter_dir=args.adapter_dir,
        output=output, device=args.device, retrieval_device=args.retrieval_device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        limit_pairs=args.limit_pairs, verify_hashes=not args.skip_hash_verification,
        save_hard_cases=args.save_hard_cases, adapter_scale=args.adapter_scale,
        distractor_manifest=args.distractor_manifest, query_augmentation=args.query_augmentation,
        augmentation_config=args.augmentation_config, augmentation_strength=args.augmentation_strength,
        augmentation_probability=args.augmentation_probability, augmentation_seed=args.augmentation_seed,
        gallery_cache=args.gallery_cache,
    )


if __name__ == "__main__":
    main()

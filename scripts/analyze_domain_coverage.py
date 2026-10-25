#!/usr/bin/env python3
"""Frozen original-Day, training-only domain coverage and matched intervention.

Run serially through scripts/run_bounded.py. This never trains a model, computes
Recall, reads evaluation donors or changes an augmentation policy automatically.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torch.utils.data import DataLoader

from weather_vpr.training.domain_coverage import (
    DiagnosticImages, PREPROCESSING, STATISTICS, balanced_sample, coverage_report,
    extract_prefix_statistics, intervention_report, model_forward_provenance, model_state_digest,
    read_stats_cache, sha256, signed_image_rows, signature, verify_row_provenance, write_stats_cache,
)
from weather_vpr.utils.backend import enable_backend_imports, project_root


def main(argv=None):
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True,
                        help='Train-only inventory manifest(s), with admission metadata')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report file')
    parser.add_argument('--cache', type=Path, required=True, help='Immutable .safetensors statistics cache')
    parser.add_argument('--checkpoint', type=Path, default=root/'weights/dino_salad.ckpt')
    parser.add_argument('--backbone-repo', type=Path, default=root/'third_party/dinov2')
    parser.add_argument('--source-night-config', type=Path, default=root/'configs/source_night.json')
    parser.add_argument('--max-per-source', type=int, default=512)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pca-dimensions', type=int, default=32)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--intervention-recipients', type=int, default=64)
    parser.add_argument('--intervention-batch-size', type=int, default=16)
    parser.add_argument('--intervention-after-block', type=int, default=7)
    parser.add_argument('--skip-interventions', action='store_true',
                        help='Coverage only; report explicitly records missing intervention evidence')
    args = parser.parse_args(argv)
    output, cache = args.output.expanduser().resolve(), args.cache.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Result already exists: {output}')
    if (args.batch_size < 1 or args.num_workers < 0 or args.seed < 0 or args.max_per_source < 2
            or args.pca_dimensions < 1 or args.neighbors < 1 or args.intervention_recipients < 1
            or args.intervention_batch_size < 1):
        raise ValueError('Positive batches/counts/PCA/kNN and nonnegative workers/seed required')
    if (len(set(args.layers)) != len(args.layers) or not args.layers
            or any(not 0 <= layer <= 7 for layer in args.layers)):
        raise ValueError('Unique prefix layers in [0,7] are required')
    if not args.skip_interventions and args.intervention_after_block not in args.layers:
        raise ValueError('Intervention layer must be among extracted layers')
    device = torch.device(args.device)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires a supporting CUDA GPU')
    if device.type == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    # Data inventory owns source annotation/review and split admission checks.
    from weather_vpr.data.research_sources import load_source_manifest
    all_rows = []
    for manifest in args.manifest:
        all_rows.extend(load_source_manifest(manifest, project_root=root))
    selected, counts = balanced_sample(all_rows, args.max_per_source, args.seed)
    verified_provenance = verify_row_provenance(selected, root)
    selected = signed_image_rows(selected, args.seed)
    source_config = json.loads(args.source_night_config.read_text())
    enable_backend_imports()
    from workflow.model import load_checkpoint_model, REGISTERED_PRETRAINED_TEACHERS
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash not in REGISTERED_PRETRAINED_TEACHERS:
        raise ValueError('Coverage coordinates must use the registered ORIGINAL Day SALAD checkpoint')
    import weather_vpr.training.domain_coverage as implementation
    from weather_vpr.data import source_night, night_augmentation, student_augmentation
    identity = {'format_version': 1, 'checkpoint_sha256': checkpoint_hash,
                'preprocessing': PREPROCESSING, 'statistics': STATISTICS,
                'layers': args.layers, 'forward_precision': args.precision,
                'seed': args.seed, 'max_per_source': args.max_per_source,
                'sampling': 'deterministic group-balanced without replacement; Qwen/Source share canonical scenes',
                'source_night_config': source_config,
                'source_night_config_sha256': sha256(args.source_night_config),
                'source_night_augmentation_epoch': 0,
                'manifests': [{'path': str(path.resolve()), 'sha256': sha256(path)} for path in args.manifest],
                'verified_annotation_and_review_files': verified_provenance,
                'implementation_sha256': {Path(module.__file__).name: sha256(module.__file__)
                                           for module in (implementation, source_night, night_augmentation,
                                                          student_augmentation)},
                'selected_rows': selected}
    dataset = DiagnosticImages(selected, source_config)
    started = time.monotonic()
    base = load_checkpoint_model(args.checkpoint, device=args.device, backbone_repo=args.backbone_repo)
    base.requires_grad_(False).eval()
    before_hash = model_state_digest(base)
    identity['loaded_model_state_sha256'] = before_hash
    identity['forward_implementation'] = model_forward_provenance(base, root)
    identity['diagnostic_script_sha256'] = sha256(__file__)
    cache_hit = cache.exists() or cache.with_suffix(cache.suffix+'.json').exists()
    if cache_hit:
        tensors = read_stats_cache(cache, identity)
        print(f'Verified exact statistics cache {cache}', flush=True)
    else:
        generator = torch.Generator().manual_seed(args.seed)
        options = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                       pin_memory=device.type == 'cuda', generator=generator)
        if args.num_workers:
            options['prefetch_factor'] = 1
        loader = DataLoader(dataset, **options)
        parts = {f'block_{layer}': [] for layer in args.layers}
        parts['brightness'] = []
        seen = 0
        for images, brightness in loader:
            images = images.to(device, non_blocking=True)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
            with torch.no_grad(), amp:
                batch = extract_prefix_statistics(base, images, args.layers)
            for key, value in batch.items():
                parts[key].append(value.cpu())
            parts['brightness'].append(brightness.float())
            seen += len(images)
            print(f'Frozen prefix statistics {seen}/{len(dataset)}', flush=True)
            del images, batch
        tensors = {key: torch.cat(values, 0) for key, values in parts.items()}
        write_stats_cache(cache, tensors, identity)
    extraction_seconds = time.monotonic()-started
    print('Fitting group-disjoint synthetic-reference standardization/PCA/calibration', flush=True)
    coverage = coverage_report(tensors, selected, args.layers, args.seed, args.pca_dimensions, args.neighbors)
    interventions = ({'status': 'not_run', 'decision': 'insufficient_intervention_evidence'}
                     if args.skip_interventions else intervention_report(
                         base, dataset, tensors, selected, device=args.device, precision=args.precision,
                         batch_size=args.intervention_batch_size, recipients=args.intervention_recipients,
                         seed=args.seed, after_block=args.intervention_after_block))
    after_hash = model_state_digest(base)
    if before_hash != after_hash:
        raise AssertionError('Diagnostic changed original Day model weights/buffers')
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise AssertionError('Diagnostic unexpectedly accumulated gradients')
    report = {'purpose': 'train-only descriptive domain coverage and matched intervention; no model updates or Recall',
              'decision': 'requires_content_review', 'seed': args.seed, 'precision': args.precision,
              'checkpoint_sha256': checkpoint_hash, 'cache': str(cache),
              'cache_signature': signature(identity), 'cache_hit': cache_hit,
              'sample_counts': counts, 'layers': args.layers, 'statistics': STATISTICS,
              'preprocessing': PREPROCESSING, 'coverage': coverage, 'interventions': interventions,
              'model_state_sha256_before': before_hash, 'model_state_sha256_after': after_hash,
              'optimizer_updates': 0, 'recall_evaluations': 0,
              'statistics_extraction_or_cache_load_seconds': extraction_seconds,
              'elapsed_seconds': time.monotonic()-started,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'content_review_required': ['Compare camera/road/scene composition at similar brightness.',
                                          'DarkDriving technical grouping does not establish independent routes.',
                                          'Review sampled images and non-overlap outliers before donor experiments.',
                                          'Distance/brightness/descriptor drift does not prove transferable lighting.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Wrote {output}; decision=requires_content_review; optimizer_updates=0', flush=True)


if __name__ == '__main__':
    main()

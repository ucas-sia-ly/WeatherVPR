#!/usr/bin/env python3
"""Train-only diagnostics for the fixed phase-two Token style candidate.

No optimizer, validation queries, external images or automatic policy changes.
Run this script through scripts/run_bounded.py, serially with model workloads.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs, compact_training_batch, sha256
from weather_vpr.data.place_sampler import DistinctPlaceBatchSampler
from weather_vpr.data.source_night import SourceNightConfig
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.training.optimization import configure_full_backbone_scope
from weather_vpr.training.token_style import TokenStyleAugment
from weather_vpr.utils.backend import enable_backend_imports, project_root


def state_digest(model):
    """Stream tensors to CPU individually, rather than cloning both backbones."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def describe(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if not len(values):
        return {'count': 0}
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError('Non-finite diagnostic statistic')
    return {'count': len(values), 'mean': float(values.mean()), 'min': float(values.min()),
            'p50': float(values.quantile(.5)), 'p95': float(values.quantile(.95)),
            'p99': float(values.quantile(.99)), 'max': float(values.max())}


def main(argv=None):
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batches', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint', type=Path, default=root / 'weights/dino_salad.ckpt')
    parser.add_argument('--backbone-repo', type=Path, default=root / 'third_party/dinov2')
    parser.add_argument('--manifest', type=Path, default=root / 'data/splits/night_train.jsonl')
    parser.add_argument('--exclude-places', type=Path, default=root / 'data/splits/heldout_places.json')
    parser.add_argument('--augmentation-config', type=Path, default=root / 'configs/night_camera.json')
    parser.add_argument('--source-night-config', type=Path, default=root / 'configs/source_night.json')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Diagnostic output already exists: {output}')
    if args.batches < 1 or args.batch_size < 2 or args.num_workers < 0:
        raise ValueError('Need positive batches, batch-size >= 2 and nonnegative workers')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU diagnostic requires BF16 support, matching phase-two training')
    if device.type == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera = json.loads(args.augmentation_config.read_text())
    camera.update(profile='camera', strength=1., probability=.8)
    dataset = AcceptedWeatherPairs(args.manifest, 'night', (224, 224), args.exclude_places, True,
                                  night_augmentation=NightAugmentationConfig(**camera), seed=args.seed,
                                  source_night_config=SourceNightConfig.from_dict(
                                      json.loads(args.source_night_config.read_text())))
    dataset.set_epoch(0)
    sampler = DistinctPlaceBatchSampler([row['place'] for row in dataset.rows],
                                       args.batch_size, seed=args.seed)
    sampler.set_epoch(0)
    loader_options = dict(batch_sampler=sampler, num_workers=args.num_workers,
                          collate_fn=compact_training_batch, pin_memory=device.type == 'cuda')
    if args.num_workers:
        loader_options['prefetch_factor'] = 1
    loader = DataLoader(dataset, **loader_options)
    enable_backend_imports()
    from workflow.model import load_checkpoint_model
    base = load_checkpoint_model(args.checkpoint, device=args.device, backbone_repo=args.backbone_repo)
    net = WeatherAdapterSALAD(base, ('night',), adapter_type='full').to(device)
    scope = configure_full_backbone_scope(net.adapters['night'], 'late', 4)
    net.train()
    initial_keys, before_hash = set(net.state_dict()), state_digest(net)
    style = TokenStyleAugment(args.seed, probability=.5, fraction=.3, strength=.2, beta=.1)
    gains, centered_changes, descriptor_drifts, records = [], [], [], []
    keep_drifts, assignment_drifts = [], []
    first_images = None
    amp = lambda: (torch.autocast('cuda', dtype=torch.bfloat16)
                   if device.type == 'cuda' else nullcontext())
    for step, batch in enumerate(islice(loader, args.batches)):
        images = torch.stack([item['student_tensor'] for item in batch]).to(device)
        if first_images is None:
            first_images = images.clone()

        def transform(tokens):
            transformed = style.transform(tokens, epoch=0, step=step)
            if transformed.shape != tokens.shape or not torch.equal(transformed[:, 0], tokens[:, 0]):
                raise AssertionError('Token shape or CLS changed at the injection point')
            patches, changed = tokens[:, 1:].float(), transformed[:, 1:].float()
            mask = torch.zeros(tokens.shape[:2], device=device, dtype=torch.bool)[:, 1:]
            record = dict(style.last_record)
            for image, indices in zip(record.get('selected_images', []), record.get('patch_indices', [])):
                mask[image, indices] = True
            if not torch.equal(transformed[:, 1:][~mask], tokens[:, 1:][~mask]):
                raise AssertionError('An unselected patch changed')
            mean = patches.mean(1, keepdim=True)
            std = (patches.var(1, correction=0, keepdim=True) + style.eps).sqrt()
            for image in record.get('selected_images', []):
                donor, coefficient = record['donor_indices'][image], record['lambdas'][image]
                gains.extend(((std[image] + coefficient * (std[donor] - std[image]))
                              / std[image]).flatten().cpu().tolist())
            denominator = (patches - mean).square().mean((1, 2)).sqrt().clamp_min(style.eps)
            relative = (changed - patches).square().mean((1, 2)).sqrt() / denominator
            centered_changes.extend(relative.cpu().tolist())
            record.update(after_block=7, cls_unchanged=True, unselected_patches_unchanged=True,
                          relative_centered_rms_delta=relative.cpu().tolist())
            records.append(record)
            return transformed

        with torch.no_grad(), amp():
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            plain, plain_aux = net.student(images, 'night', return_aux=True)
            # Match any backbone stochastic operations, isolating style drift.
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            styled, styled_aux = net.student(images, 'night', return_aux=True,
                                            token_transform=transform, token_transform_after_block=7)
        drift = .5 * (F.normalize(plain.float(), dim=1) - F.normalize(styled.float(), dim=1)).square().sum(1)
        descriptor_drifts.extend(drift.cpu().tolist())
        keep_drifts.extend((plain_aux['assignment_keep'] - styled_aux['assignment_keep']).abs()
                           .flatten(1).mean(1).cpu().tolist())
        assignment_drifts.extend((plain_aux['log_assignment'] - styled_aux['log_assignment']).abs()
                                 .flatten(1).mean(1).cpu().tolist())
        records[-1]['main_indices'] = [item['index'] for item in batch]
        print(f'Diagnostic batch {step + 1}/{args.batches}: applied={records[-1]["applied_images"]}, '
              f'descriptor cosine drift={float(drift.mean()):.6f}', flush=True)
        del images, plain, plain_aux, styled, styled_aux, drift
    if len(records) != args.batches:
        raise RuntimeError('Training manifest has fewer batches than requested')
    net.eval()
    with torch.no_grad(), amp():
        inference1 = net.student(first_images, 'night', return_aux=False)
        inference2 = net.student(first_images, 'night', return_aux=False)
        original_day = net.teacher(first_images, return_aux=False)
    if not torch.equal(inference1, inference2) or not torch.equal(inference1, original_day):
        raise AssertionError('Plain inference is nondeterministic or differs from original Day initialization')
    after_hash = state_digest(net)
    if before_hash != after_hash or initial_keys != set(net.state_dict()):
        raise AssertionError('Diagnostic changed model weights, buffers or checkpoint keys')
    gain_tensor = torch.tensor(gains)
    report = {'purpose': 'train-only fixed-candidate diagnostic; no optimizer or Recall evaluation',
              'seed': args.seed, 'batches': len(records), 'batch_size': args.batch_size,
              'precision': 'bf16' if device.type == 'cuda' else 'fp32',
              'checkpoint_sha256': sha256(args.checkpoint), 'manifest_sha256': sha256(dataset.manifest),
              'exclude_places_sha256': sha256(args.exclude_places),
              'augmentation_config_sha256': sha256(args.augmentation_config),
              'scope': scope, 'token_style': {**style.configuration(), 'after_block': 7},
              'gain': {**describe(gains), **{f'fraction_gt{threshold}': (float((gain_tensor > threshold).float().mean())
                                                                    if len(gain_tensor) else None)
                                           for threshold in (2, 4, 8)}},
              'relative_centered_rms_delta': describe(centered_changes),
              'descriptor_cosine_drift': describe(descriptor_drifts),
              'assignment_keep_mean_absolute_drift': describe(keep_drifts),
              'log_assignment_mean_absolute_drift': describe(assignment_drifts),
              'model_state_sha256_before': before_hash, 'model_state_sha256_after': after_hash,
              'plain_inference_deterministic': True, 'plain_inference_equals_original_day': True,
              'nonfinite_count': 0, 'records': records,
              'policy': 'No clipping policy was selected or changed by this script'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('gain', 'relative_centered_rms_delta',
                     'descriptor_cosine_drift', 'assignment_keep_mean_absolute_drift',
                     'log_assignment_mean_absolute_drift')}, indent=2), flush=True)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()

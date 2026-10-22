"""Full Night DINOv2 fine-tuning on spatially separated real SVOX train data.

The frozen Day DINOv2-SALAD gallery is used as a fixed descriptor bank. Only
SVOX train/night query images are decoded; SVOX test filenames are read solely
to enforce a spatial exclusion buffer. SALAD parameters stay frozen while its
aggregation passes gradients back to the Night backbone.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from PIL import Image

from weather_vpr.data.svox_real_night import (
    SVOXNightQueryDataset, load_real_night_train, northing_holdout, utm_from_name,
)
from weather_vpr.inference.svox import (
    _cache_metadata, _load_gallery_cache, _save_gallery_cache,
    gallery_fingerprint, sha256,
)
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.utils.backend import enable_backend_imports, project_root

enable_backend_imports()
from workflow.evaluation import EvaluationSet, build_results, exact_retrieval, extract_descriptors  # noqa: E402
from workflow.model import input_preprocessing, load_checkpoint_model  # noqa: E402
from workflow.reliability_cache import preprocess_teacher_image  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--svox-root', type=Path, default=project_root() / 'data/svox')
    p.add_argument('--checkpoint', type=Path, default=project_root() / 'weights/dino_salad.ckpt')
    p.add_argument('--backbone-repo', type=Path, default=project_root() / 'third_party/dinov2')
    p.add_argument('--init-adapter-dir', type=Path,
                   help='Optional compatible full Night DINOv2 run; otherwise initialize from Day')
    p.add_argument('--output-dir', type=Path, default=project_root() / 'runs/night_full_real_01')
    p.add_argument('--gallery-cache', type=Path,
                   default=project_root() / 'runs/cache/svox_train_gallery.safetensors')
    p.add_argument('--test-buffer-m', type=float, default=250.0,
                   help='Exclude train queries and gallery keys near test geometry')
    p.add_argument('--negative-radius-m', type=float, default=250.0,
                   help='Keys beyond this distance are negatives; 25 m to this radius are ignored')
    p.add_argument('--validation-fraction', type=float, default=0.2,
                   help='Northern train-route tail reserved for internal development; 0 uses all retained queries')
    p.add_argument('--validation-gap-m', type=float, default=50.0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=0, help='0 runs all requested epochs')
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--layerwise-lr-decay', type=float, default=0.7,
                   help='0<value<=1; early DINOv2 layers use progressively smaller nonzero LR')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--anchor-weight', type=float, default=0.0,
                   help='Optional same-day-image student/teacher descriptor anchor')
    p.add_argument('--night-baseline-weight', type=float, default=100.0,
                   help='Optional same-night-image anchor to frozen Day teacher descriptor')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--check-data', action='store_true',
                   help='Report spatial exclusions and development split without loading model weights')
    return p.parse_args(argv)


def _validate(args):
    if min(args.batch_size, args.gradient_accumulation_steps, args.epochs) < 1:
        raise ValueError('Batch size, gradient accumulation, and epochs must be positive')
    if args.max_steps < 0 or args.num_workers < 0:
        raise ValueError('max-steps and num-workers must be nonnegative')
    if not math.isfinite(args.negative_radius_m) or args.negative_radius_m <= 25.0:
        raise ValueError('negative-radius-m must exceed the native 25 m positive radius')
    if not math.isfinite(args.test_buffer_m) or args.test_buffer_m <= 25.0:
        raise ValueError('test-buffer-m must exceed the native 25 m positive radius')
    for name in ('learning_rate', 'temperature'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(args.layerwise_lr_decay) or not 0 < args.layerwise_lr_decay <= 1:
        raise ValueError('layerwise-lr-decay must be in (0, 1]')
    for name in ('weight_decay', 'anchor_weight', 'night_baseline_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not math.isfinite(args.validation_fraction) or not 0 <= args.validation_fraction < 0.5:
        raise ValueError('validation-fraction must be in [0, 0.5)')
    if not math.isfinite(args.validation_gap_m) or args.validation_gap_m < 0:
        raise ValueError('validation-gap-m must be finite and nonnegative')
    if args.bf16 and (torch.device(args.device).type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('--bf16 requires a compatible CUDA GPU')


def full_gallery_geographic_infonce(
    query_descriptors: torch.Tensor, query_coordinates: torch.Tensor,
    day_bank: torch.Tensor, reference_coordinates: torch.Tensor, *,
    positive_radius_meters: float = 25.0, negative_radius_meters: float = 50.0,
    temperature: float = 0.1,
) -> dict:
    """Multi-positive InfoNCE with keys between the two radii omitted.

    Every train gallery key within 25 m contributes to the numerator. Only
    gallery keys at or beyond the negative radius can act as negatives. This
    also prevents geographically identical keys from becoming false negatives.
    All similarity and log-sum-exp arithmetic is FP32.
    """
    if query_descriptors.ndim != 2 or day_bank.ndim != 2 or query_descriptors.shape[1] != day_bank.shape[1]:
        raise ValueError('Query and day descriptor dimensions disagree')
    if (query_coordinates.shape != (len(query_descriptors), 2)
            or reference_coordinates.shape != (len(day_bank), 2)):
        raise ValueError('UTM coordinate shapes disagree with descriptor counts')
    if (not math.isfinite(positive_radius_meters) or positive_radius_meters <= 0
            or not math.isfinite(negative_radius_meters)
            or negative_radius_meters <= positive_radius_meters
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Invalid geographic radii or InfoNCE temperature')
    query = F.normalize(query_descriptors.float(), dim=1)
    # The gallery is normalized once before training; repeating normalization
    # here would scan roughly 700 MB of day vectors at every small query batch.
    bank = day_bank.detach().float().to(query.device)
    q_xy = query_coordinates.detach().double().to(query.device)
    r_xy = reference_coordinates.detach().double().to(query.device)
    squared_distance = ((q_xy[:, None, :] - r_xy[None, :, :]) ** 2).sum(-1)
    positives = squared_distance <= positive_radius_meters ** 2
    negatives = squared_distance >= negative_radius_meters ** 2
    if not positives.any(dim=1).all():
        raise ValueError('A real-night query has no 25 m gallery positive')
    valid = positives | negatives
    logits = (query @ bank.T) / temperature
    loss = (torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1) -
            torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)).mean()
    return {
        'loss': loss,
        'positive_counts': positives.sum(1).tolist(),
        'effective_negative_counts': negatives.sum(1).tolist(),
        'ambiguous_counts': (~valid).sum(1).tolist(),
        'nearest_positive_indices': squared_distance.masked_fill(~positives, torch.inf).argmin(1),
    }


def _read_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_teacher_image(image, size)


def night_baseline_descriptor_loss(net: WeatherAdapterSALAD,
                                   night_images: torch.Tensor,
                                   student_descriptors: torch.Tensor) -> torch.Tensor:
    """Anchor a Night query to the frozen Day model on the *same* image.

    The target forward and target tensor are detached from autograd, while the
    student descriptor remains differentiable through Night DINOv2 and SALAD.
    """
    with torch.no_grad():
        day_descriptors = net.teacher(night_images, return_aux=False)
    student = F.normalize(student_descriptors.float(), dim=1)
    target = F.normalize(day_descriptors.detach().float(), dim=1)
    return (0.5 * (student - target).square().sum(dim=1)).mean()


def night_layerwise_optimizer_groups(backbone: torch.nn.Module, *, learning_rate: float,
                                     weight_decay: float, decay: float) -> list[dict]:
    """Give every Night DINOv2 parameter a positive LR, with earlier layers slower."""
    if not 0 < decay <= 1:
        raise ValueError('Layerwise LR decay must be in (0, 1]')
    blocks = getattr(getattr(backbone, 'model', None), 'blocks', None)
    if blocks is None or len(blocks) == 0:
        raise ValueError('A full DINOv2 backbone with Transformer blocks is required')
    buckets: dict[int, list[torch.nn.Parameter]] = {}
    count = len(blocks)
    for name, parameter in backbone.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f'Frozen Night DINOv2 parameter: {name}')
        if name.startswith('model.blocks.'):
            fields = name.split('.')
            try:
                index = int(fields[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(f'Invalid DINOv2 block parameter: {name}') from exc
            if not 0 <= index < count:
                raise ValueError(f'Invalid DINOv2 block index: {name}')
            depth = count - 1 - index
        elif name.startswith('model.norm.'):
            depth = 0
        else:
            # Patch embedding, CLS and positional tokens are the earliest stage.
            depth = count
        buckets.setdefault(depth, []).append(parameter)
    return [{'params': buckets[depth], 'lr': learning_rate * decay ** depth,
             'weight_decay': weight_decay, 'layer_depth': depth}
            for depth in sorted(buckets)]


class _NightStudent(torch.nn.Module):
    def __init__(self, net: WeatherAdapterSALAD):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.student(images, 'night', return_aux=False)


def _dev_recall(model, images, positives, references, gallery_descriptors,
                image_size, device, batch_size, num_workers,
                label='SVOX train internal development'):
    if not images:
        return None
    query_descriptors = extract_descriptors(model, images, image_size, device,
                                             batch_size, num_workers)
    rows = exact_retrieval(gallery_descriptors, query_descriptors, positives,
                           device=device, query_chunk_size=16,
                           reference_chunk_size=4096,
                           reference_cache_bytes=gallery_descriptors.numel() * 4 + len(references) * 4)
    evaluation = EvaluationSet(label, references, images,
                                positives, {'split': 'train', 'ground_truth': 'utm_radius',
                                            'positive_radius_meters': 25.0})
    return build_results(evaluation, rows)['recall']


def _save_model(directory: Path, net: WeatherAdapterSALAD, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights = {key: tensor.detach().cpu().contiguous()
               for key, tensor in net.adapters['night'].state_dict().items()}
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.night_full_',
                                     suffix='.safetensors', delete=False) as stream:
        temporary_weights = Path(stream.name)
    try:
        save_file(weights, str(temporary_weights))
        os.replace(temporary_weights, directory / 'adapter.safetensors')
    finally:
        temporary_weights.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.night_full_',
                                     suffix='.json', mode='w', encoding='utf-8',
                                     delete=False) as stream:
        temporary_metadata = Path(stream.name)
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    try:
        os.replace(temporary_metadata, directory / 'adapter.json')
    finally:
        temporary_metadata.unlink(missing_ok=True)


def _init_full_night(net: WeatherAdapterSALAD, directory: Path | None,
                     checkpoint_hash: str, model_config: dict, preprocessing: dict):
    if directory is None:
        return None
    run = Path(directory).expanduser().resolve(strict=True)
    metadata = json.loads((run / 'adapter.json').read_text(encoding='utf-8'))
    if (metadata.get('weather') != 'night' or
            metadata.get('checkpoint_sha256') != checkpoint_hash or
            metadata.get('model_config') != model_config or
            metadata.get('preprocessing') != preprocessing or
            adapter_model_kwargs(metadata)['adapter_type'] != 'full'):
        raise ValueError('Initial Night backbone is incompatible with the Day SALAD checkpoint')
    weights = run / 'adapter.safetensors'
    net.adapters['night'].load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    return {'directory': str(run), 'adapter_sha256': sha256(weights)}


def main(argv=None):
    args = parse_args(argv)
    _validate(args)
    out = args.output_dir.expanduser().resolve()
    if not args.check_data and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; use a new run')
    real = load_real_night_train(args.svox_root,
                                 test_buffer_meters=args.test_buffer_m)
    images = real.filtered.queries
    train_indices, dev_indices, buffer_indices = northing_holdout(
        images, fraction=args.validation_fraction, gap_meters=args.validation_gap_m)
    counts = {
        'source_train_gallery': len(real.full.references),
        'source_train_night_queries': len(real.full.queries),
        'retained_train_gallery': len(real.filtered.references),
        'retained_train_night_queries': len(images),
        'optimization_queries': len(train_indices),
        'development_queries': len(dev_indices),
        'internal_buffer_queries': len(buffer_indices),
        'min_positives_per_retained_query': min(map(len, real.filtered.positives)),
        'max_positives_per_retained_query': max(map(len, real.filtered.positives)),
        'test_geometry_filenames': real.test_geometry_count,
        'test_geometry_sha256': real.test_geometry_sha256,
        'test_buffer_meters': args.test_buffer_m,
        'development_gap_meters': args.validation_gap_m,
    }
    print(json.dumps(counts, indent=2), flush=True)
    if args.check_data:
        return counts

    print(json.dumps({'effective_configuration': {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }}, indent=2), flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_hash = sha256(checkpoint)
    base = load_checkpoint_model(checkpoint, device=str(device),
                                 backbone_repo=args.backbone_repo)
    image_size = tuple(base.image_size)
    if image_size != (224, 224):
        raise ValueError(f'Native SVOX training needs 224x224, got {image_size}')
    preprocessing = input_preprocessing(image_size)
    descriptor_dim = (base.config['agg_config']['num_clusters'] *
                      base.config['agg_config']['cluster_dim'] +
                      base.config['agg_config']['token_dim'])
    cache = args.gallery_cache.expanduser().resolve()
    gallery_hash = gallery_fingerprint(real.full.references)
    expected_cache = _cache_metadata(real.full.references, gallery_hash,
                                     checkpoint_hash, preprocessing)
    full_gallery = _load_gallery_cache(cache, expected_cache, descriptor_dim)
    if full_gallery is None:
        print('Extracting original Day SALAD SVOX train gallery...', flush=True)
        full_gallery = extract_descriptors(base, real.full.references, image_size,
                                           str(device), max(args.batch_size, 16),
                                           args.num_workers)
        _save_gallery_cache(cache, expected_cache, full_gallery)
    else:
        print(f'Loaded verified Day SALAD gallery: {cache}', flush=True)
    gallery_cpu = full_gallery[list(real.original_reference_indices)].contiguous()
    day_bank = F.normalize(gallery_cpu.to(device).float(), dim=1).detach()
    reference_xy = torch.tensor([utm_from_name(image.path) for image in real.filtered.references],
                                dtype=torch.float64, device=device)
    query_xy = torch.tensor([utm_from_name(image.path) for image in images],
                            dtype=torch.float64, device=device)
    del full_gallery

    net = WeatherAdapterSALAD(base, ('night',), adapter_type='full').to(device)
    initial = _init_full_night(net, args.init_adapter_dir, checkpoint_hash,
                               base.config, preprocessing)
    params = tuple(net.adapter_parameters('night'))
    if not params or any(not p.requires_grad for p in params):
        raise RuntimeError('Full Night DINOv2 has no trainable parameters')
    optimizer = torch.optim.AdamW(night_layerwise_optimizer_groups(
        net.adapters['night'], learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, decay=args.layerwise_lr_decay))
    loader = DataLoader(SVOXNightQueryDataset(images, train_indices, image_size),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda')
    dev_images = [images[index] for index in dev_indices]
    dev_positives = [real.filtered.positives[index] for index in dev_indices]
    train_images = [images[index] for index in train_indices]
    train_positives = [real.filtered.positives[index] for index in train_indices]
    baseline_dev = _dev_recall(base, dev_images, dev_positives,
                               real.filtered.references, gallery_cpu, image_size,
                               str(device), max(args.batch_size, 16), args.num_workers)
    baseline_train = _dev_recall(base, train_images, train_positives,
                                 real.filtered.references, gallery_cpu, image_size,
                                 str(device), max(args.batch_size, 16), args.num_workers,
                                 label='SVOX train optimization queries')
    initial_dev = _dev_recall(_NightStudent(net), dev_images, dev_positives,
                              real.filtered.references, gallery_cpu, image_size,
                              str(device), max(args.batch_size, 16), args.num_workers)
    initial_train = _dev_recall(_NightStudent(net), train_images, train_positives,
                                real.filtered.references, gallery_cpu, image_size,
                                str(device), max(args.batch_size, 16), args.num_workers,
                                label='SVOX train optimization queries')
    print(f'Frozen Day baseline on internal SVOX-train development: {baseline_dev}', flush=True)
    print(f'Initial Night model on internal development: {initial_dev}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / 'training_config.json').write_text(json.dumps({
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }, indent=2), encoding='utf-8')
    split_record = {
        'policy': 'test geometry exclusion then northern train-route development tail',
        'counts': counts,
        'validation_fraction': args.validation_fraction,
        'train_query_ids': [images[i].id for i in train_indices],
        'development_query_ids': [images[i].id for i in dev_indices],
        'internal_buffer_query_ids': [images[i].id for i in buffer_indices],
    }
    (out / 'split.json').write_text(json.dumps(split_record, indent=2), encoding='utf-8')
    metrics_file = out / 'training_log.jsonl'
    step = 0
    optimizer_steps = 0
    best_score = None
    best_epoch = 0
    stopped = False
    accumulation_count = 0
    optimizer.zero_grad(set_to_none=True)

    def update(count):
        nonlocal optimizer_steps, accumulation_count
        if not any(param.grad is not None for param in params):
            raise RuntimeError('No gradients reached the Night DINOv2')
        if count != args.gradient_accumulation_steps:
            correction = args.gradient_accumulation_steps / count
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulation_count = 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_losses = []
        epoch_night_baseline_losses = []
        positive_counts = negative_counts = ambiguous_counts = query_count = 0
        for night_batch, indices in loader:
            night_batch = night_batch.to(device, non_blocking=True)
            indices = indices.to(device)
            autocast = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                        if args.bf16 else nullcontext())
            with autocast:
                night_descriptors = net.student(night_batch, 'night', return_aux=False)
            result = full_gallery_geographic_infonce(
                night_descriptors, query_xy[indices], day_bank, reference_xy,
                negative_radius_meters=args.negative_radius_m,
                temperature=args.temperature)
            contrast = result['loss']
            if not torch.isfinite(contrast):
                raise FloatingPointError('Nonfinite real-night geographic InfoNCE')
            total_loss = contrast
            anchor_value = 0.0
            if args.anchor_weight > 0:
                nearest = result['nearest_positive_indices'].detach().cpu().tolist()
                day_batch = torch.stack([
                    _read_image(real.filtered.references[index].path, image_size)
                    for index in nearest
                ]).to(device)
                with autocast:
                    day_student = net.student(day_batch, 'night', return_aux=False)
                same_image_target = day_bank[nearest]
                anchor = (1 - F.cosine_similarity(day_student.float(),
                                                  same_image_target.float(), dim=1)).mean()
                total_loss = total_loss + args.anchor_weight * anchor
                anchor_value = float(anchor.detach())
            night_baseline_value = 0.0
            if args.night_baseline_weight > 0:
                with autocast:
                    night_baseline = night_baseline_descriptor_loss(
                        net, night_batch, night_descriptors)
                if not torch.isfinite(night_baseline):
                    raise FloatingPointError('Nonfinite real-night baseline descriptor loss')
                total_loss = total_loss + args.night_baseline_weight * night_baseline
                night_baseline_value = float(night_baseline.detach())
            (total_loss / args.gradient_accumulation_steps).backward()
            accumulation_count += 1
            if accumulation_count == args.gradient_accumulation_steps:
                update(accumulation_count)
            step += 1
            epoch_losses.append(float(contrast.detach()))
            epoch_night_baseline_losses.append(night_baseline_value)
            positive_counts += sum(result['positive_counts'])
            negative_counts += sum(result['effective_negative_counts'])
            ambiguous_counts += sum(result['ambiguous_counts'])
            query_count += len(indices)
            row = {
                'epoch': epoch, 'step': step, 'optimizer_step': optimizer_steps,
                'geographic_contrast': float(contrast.detach()),
                'same_day_anchor': anchor_value,
                'night_baseline_alignment': night_baseline_value,
                'weighted_total': float(total_loss.detach()),
                'positive_keys': sum(result['positive_counts']),
                'effective_negative_keys': sum(result['effective_negative_counts']),
                'masked_ambiguous_keys': sum(result['ambiguous_counts']),
                'query_count': len(indices),
            }
            with metrics_file.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            if step == 1 or step % 20 == 0:
                print(row, flush=True)
            if args.max_steps and step >= args.max_steps:
                stopped = True
                break
        if accumulation_count:
            update(accumulation_count)
        net.eval()
        dev_recall = _dev_recall(_NightStudent(net), dev_images, dev_positives,
                                 real.filtered.references, gallery_cpu, image_size,
                                 str(device), max(args.batch_size, 16), args.num_workers)
        train_recall = _dev_recall(_NightStudent(net), train_images, train_positives,
                                   real.filtered.references, gallery_cpu, image_size,
                                   str(device), max(args.batch_size, 16), args.num_workers,
                                   label='SVOX train optimization queries')
        epoch_summary = {
            'epoch': epoch, 'steps': step, 'optimizer_steps': optimizer_steps,
            'mean_geographic_contrast': sum(epoch_losses) / len(epoch_losses),
            'mean_night_baseline_alignment': (
                sum(epoch_night_baseline_losses) / len(epoch_night_baseline_losses)),
            'mean_positive_keys_per_query': positive_counts / query_count,
            'mean_effective_negative_keys_per_query': negative_counts / query_count,
            'mean_masked_ambiguous_keys_per_query': ambiguous_counts / query_count,
            'effective_negative_keys': negative_counts,
            'masked_ambiguous_keys': ambiguous_counts,
            'baseline_dev_recall': baseline_dev,
            'baseline_train_recall': baseline_train,
            'initial_dev_recall': initial_dev,
            'initial_train_recall': initial_train,
            'dev_recall': dev_recall,
            'train_recall': train_recall,
        }
        # Current epoch may contain a shorter final batch; log exact query count.
        epoch_summary['optimized_queries_this_epoch'] = query_count
        print(json.dumps(epoch_summary), flush=True)
        with metrics_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': 'epoch_end', **epoch_summary}) + '\n')
        score = tuple(dev_recall[f'R@{rank}'] for rank in (1, 5, 10)) if dev_recall else None
        if best_score is None or (score is not None and score > best_score) or dev_recall is None:
            best_score = score
            best_epoch = epoch
            metadata = {
                'weather': 'night', 'adapter_type': 'full',
                'bottleneck': 64, 'lora_rank': 8, 'lora_alpha': 16.0,
                'lora_dropout': 0.0, 'lora_blocks': 4,
                'checkpoint_sha256': checkpoint_hash,
                'model_config': base.config, 'preprocessing': preprocessing,
                'training_stage': 'svox_real_night_geographic',
                'source_split': 'train', 'test_images_decoded': False,
                'test_geometry_exclusion': {
                    'radius_meters': args.test_buffer_m,
                    'filename_count': real.test_geometry_count,
                    'filenames_sha256': real.test_geometry_sha256,
                },
                'source_counts': counts,
                'gallery_sha256': gallery_hash,
                'initial_night_backbone': initial,
                'optimization_queries': len(train_indices),
                'development_queries': len(dev_indices),
                'validation_fraction': args.validation_fraction,
                'validation_gap_meters': args.validation_gap_m,
                'positive_radius_meters': 25.0,
                'negative_radius_meters': args.negative_radius_m,
                'temperature': args.temperature,
                'anchor_weight': args.anchor_weight,
                'night_baseline_weight': args.night_baseline_weight,
                'learning_rate': args.learning_rate,
                'layerwise_lr_decay': args.layerwise_lr_decay,
                'weight_decay': args.weight_decay,
                'precision': 'bf16' if args.bf16 else 'fp32',
                'batch_size': args.batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'seed': args.seed,
                'steps': step, 'optimizer_steps': optimizer_steps,
                'epochs_completed': epoch, 'selected_epoch': epoch,
                'selection_metric': 'SVOX train internal development Recall@1,5,10' if dev_recall else 'last epoch',
                'baseline_dev_recall': baseline_dev,
                'baseline_train_recall': baseline_train,
                'initial_dev_recall': initial_dev,
                'initial_train_recall': initial_train,
                'selected_dev_recall': dev_recall,
                'selected_train_recall': train_recall,
                'code_version': 'weather_vpr_full_real_night_v2',
            }
            _save_model(out, net, metadata)
        if stopped:
            break
    print(f'Saved best Night DINOv2 from epoch {best_epoch}: {out / "adapter.safetensors"}', flush=True)


if __name__ == '__main__':
    main()

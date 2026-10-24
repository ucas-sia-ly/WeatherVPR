"""Train a weather DINOv2, LoRA, or output adapter against frozen day SALAD."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import random
import tempfile
import time
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from weather_vpr.utils.backend import enable_backend_imports, project_root
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs, sha256, compact_training_batch
from weather_vpr.data.correspondence import correspondence, load_calibration
from weather_vpr.data.place_sampler import DistinctPlaceBatchSampler
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.losses.distillation import weighted_patch_losses, descriptor_alignment
from weather_vpr.data.student_augmentation import (
    augmentation_metadata, sampled_augmentation_counts, merge_augmentation_counts,
)
from weather_vpr.training.augmentation import (
    add_augmentation_arguments, load_augmentation_config, training_configuration,
)
from weather_vpr.training.auxiliary import (
    build_auxiliary_branch, prepare_auxiliary_dataset,
    paired_global_alignment, AuxiliaryGradientProbe,
)

# Import the vendored SALAD backend from this project.
enable_backend_imports()
from workflow.model import input_preprocessing, load_checkpoint_model

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=project_root() / "weights/dino_salad.ckpt", help='FULL SALAD checkpoint')
    p.add_argument('--backbone-repo', type=Path, default=project_root() / "third_party/dinov2",
                   help='Locally cloned facebookresearch/dinov2 with hubconf.py')
    p.add_argument('--manifest', type=Path, default=project_root() / "data/splits/night_train.jsonl", help='Reviewed Qwen JSONL')
    p.add_argument('--weather', choices=('night','rain','fog','snow','sun'), required=True)
    p.add_argument('--output-dir', type=Path, default=project_root() / "runs/night_full_sgpd_01")
    p.add_argument('--init-adapter-dir', type=Path,
                   help='Initialize from a completed matching run (including a full Night backbone)')
    p.add_argument('--exclude-places', type=Path, help='JSON [[city,place_id], ...] for held-out places')
    p.add_argument('--pair-calibration', type=Path,
                   default=project_root() / 'configs/dino_salad_stage2b_calibration.json',
                   help='Stage2B JSON calibration for the exact teacher checkpoint')
    p.add_argument('--calibration-index', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=1,
                   help='Optimizer updates after this many batches; bounds activation memory')
    p.add_argument('--batch-sampling', choices=('distinct','random'), default='distinct')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=0, help='0 = all')
    p.add_argument('--save-every-steps', type=int, default=200,
                   help='Checkpoint after this many batches; 0 saves only at epoch/end')
    p.add_argument('--adapter-type', choices=('full','output','lora'), default='full')
    p.add_argument('--full-train-scope', choices=('all', 'early', 'late'), default=None,
                   help='Independent Night backbone defaults to the last N blocks; all is an explicit ablation; '
                        'LoRA/output use all unless a scope is explicitly supplied')
    p.add_argument('--full-train-blocks', type=int, default=4,
                   help='Trainable first/last blocks for partial full-weight training; embeddings/final norm frozen')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=16)
    p.add_argument('--lora-dropout', type=float, default=0)
    p.add_argument('--lora-blocks', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--global-weight', type=float, default=1.0)
    p.add_argument('--contrast-weight', type=float, default=1.0)
    p.add_argument('--patch-weight', type=float, default=0.1)
    p.add_argument('--keep-weight', type=float, default=0.0)
    p.add_argument('--ot-weight', type=float, default=0.0)
    p.add_argument('--magnitude-weight', type=float, default=0.0)
    p.add_argument('--attention-weight', type=float, default=0.0,
                   help='Trusted-match KL of final DINOv2 CLS-to-patch attention; independent full Night only')
    p.add_argument('--patch-visibility', choices=('off', 'photometric'), default='off',
                   help='Optionally downweight clipped/near-black pixels in the augmented Student patch support')
    p.add_argument('--importance-mode', choices=['saliency', 'uniform'], default='saliency')
    p.add_argument('--min-teacher-keep', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--contrast-queue-size', type=int, default=256)
    p.add_argument('--fixed-day-bank', type=Path,
                   help='Cache all training Day descriptors; use a fixed bank instead of a FIFO queue')
    p.add_argument('--day-bank-device', choices=('cpu', 'cuda', 'auto'), default='cpu',
                   help='Keep negative keys on GPU to avoid transferring the entire bank for every loss')
    p.add_argument('--contrast-hard-negatives', type=int, default=0,
                   help='Keep all same-place positives and K hardest negative keys; 0 uses every key')
    p.add_argument('--relation-weight', type=float, default=0.0,
                   help='Distill Day retrieval neighborhoods over different-place Day keys')
    p.add_argument('--relation-temperature', type=float, default=0.1)
    p.add_argument('--relation-teacher-temperature', type=float, default=0.1)
    p.add_argument('--relation-hard-negatives', type=int, default=64)
    p.add_argument('--synthetic-guidance-weight', type=float, default=0.0,
                   help='Optional anchor to the base model response on clean Qwen Night; validate its utility using Night Recall')
    p.add_argument('--source-night-weight', type=float, default=0.0,
                   help='Retrieval supervision on explicitly darkened training Day photos, targeting frozen clean Day keys')
    p.add_argument('--source-night-samples', type=int, default=2,
                   help='Source-derived night views per microbatch; never clean Day Student inputs')
    p.add_argument('--source-night-config', type=Path,
                   help='JSON configuration for mandatory darkening and optional camera corruption')
    p.add_argument('--auxiliary-mode', choices=('none', 'real', 'synthetic'), default='none',
                   help='Stage-one additional paired global supervision; never replaces the main batch')
    p.add_argument('--auxiliary-weight', type=float, default=0.1,
                   help='Weight of the separately averaged paired cosine; zero bypasses all auxiliary work')
    p.add_argument('--auxiliary-samples', type=int, default=2,
                   help='Extra Student views per main microbatch, separate from Source Night')
    p.add_argument('--real-pairs-root', type=Path,
                   help='Cleaned DarkDriving root; required for active real auxiliary supervision')
    p.add_argument('--real-pairs-review', type=Path,
                   help='Explicit source-bound review JSONL with scene/split assignments')
    p.add_argument('--auxiliary-day-cache', type=Path,
                   help='Independent real Day targets; synthetic reuses the fixed GSV Day bank when available')
    p.add_argument('--auxiliary-gradient-every', type=int, default=0,
                   help='Probe first microbatch of each Nth update before clipping; 0 disables diagnostics')
    p.add_argument('--auxiliary-cache-batch-size', type=int, default=0,
                   help='Independent Teacher extraction batch; 0 uses auxiliary-samples')
    p.add_argument('--auxiliary-cache-workers', type=int, default=0,
                   help='Parallel Day decoding during one-time real Teacher target extraction')
    p.add_argument('--layer-decay', type=float, default=1.0,
                   help='Layer-wise LR decay over selected full-backbone parameters')
    p.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    p.add_argument('--warmup-steps', type=int, default=0, help='Optimizer updates, not microbatches')
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--gradient-checkpointing', action='store_true',
                   help='Recompute full DINOv2 blocks during backward to reduce activation memory')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--prefetch-factor', type=int, default=2,
                   help='Queued tensor batches per worker; ignored with zero workers')
    p.add_argument('--compact-batches', action='store_true',
                   help='Global-only training: omit full-resolution PIL images from DataLoader output')
    p.add_argument('--no-correspondence-cache', action='store_true',
                   help='Recompute trusted patch matches every epoch instead of caching them in RAM')
    p.add_argument('--seed', type=int, default=42)
    add_augmentation_arguments(p)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    if args.full_train_scope is None:
        args.full_train_scope = 'late' if args.adapter_type == 'full' else 'all'
    return args


def _validate(args):
    load_augmentation_config(args)
    if (args.prefetch_factor < 1 or args.auxiliary_cache_batch_size < 0
            or args.auxiliary_cache_workers < 0):
        raise ValueError('Prefetch must be positive; cache batch/worker counts nonnegative')
    if args.day_bank_device == 'cuda' and torch.device(args.device).type != 'cuda':
        raise ValueError('CUDA Day bank requires CUDA training')
    if args.compact_batches and any(getattr(args, name) > 0 for name in (
            'patch_weight', 'keep_weight', 'ot_weight', 'magnitude_weight', 'attention_weight')):
        raise ValueError('Compact tensor batches require global-only training (local weights zero)')
    if (not math.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0
            or args.auxiliary_samples < 1 or args.auxiliary_gradient_every < 0):
        raise ValueError('Auxiliary weight must be finite/nonnegative, samples positive, probe interval nonnegative')
    auxiliary_enabled = args.auxiliary_mode != 'none' and args.auxiliary_weight > 0
    if auxiliary_enabled and (args.weather != 'night' or args.adapter_type != 'full'):
        raise ValueError('Paired auxiliary supervision requires the independent full Night backbone')
    if auxiliary_enabled and args.auxiliary_mode == 'real' and (
            args.real_pairs_root is None or args.real_pairs_review is None):
        raise ValueError('Real auxiliary requires --real-pairs-root and --real-pairs-review')
    if auxiliary_enabled and args.auxiliary_day_cache is not None and args.fixed_day_bank is not None:
        if args.auxiliary_day_cache.expanduser().resolve() == args.fixed_day_bank.expanduser().resolve():
            raise ValueError('Paired Day cache must not overwrite the GSV Day bank')
    if args.auxiliary_gradient_every and not auxiliary_enabled:
        raise ValueError('Auxiliary gradient diagnostics require an active auxiliary branch')
    if min(args.max_steps, args.num_workers, args.contrast_queue_size,
           args.contrast_hard_negatives, args.relation_hard_negatives, args.warmup_steps) < 0:
        raise ValueError('Step counts, worker counts, queue size and hard-negative counts must be nonnegative')
    if min(args.batch_size, args.epochs, args.hidden, args.lora_rank,
           args.lora_blocks, args.gradient_accumulation_steps) < 1:
        raise ValueError('batch-size, epochs, hidden, LoRA rank/blocks and accumulation must be positive')
    if args.save_every_steps < 0:
        raise ValueError('save-every-steps must be nonnegative')
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise ValueError('lora-alpha must be finite and positive')
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        raise ValueError('lora-dropout must be finite in [0,1)')
    if not math.isfinite(args.min_teacher_keep) or not 0 <= args.min_teacher_keep < 1:
        raise ValueError('min-teacher-keep must be finite in [0,1)')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.relation_temperature, args.relation_teacher_temperature)):
        raise ValueError('Relation temperatures must be finite and positive')
    if not math.isfinite(args.layer_decay) or not 0 < args.layer_decay <= 1:
        raise ValueError('layer-decay must be finite in (0,1]')
    if args.adapter_type != 'full' and (args.layer_decay != 1 or args.gradient_checkpointing):
        raise ValueError('Layer decay and gradient checkpointing require adapter-type full')
    if args.full_train_blocks < 1:
        raise ValueError('full-train-blocks must be positive')
    if args.adapter_type != 'full' and args.full_train_scope != 'all':
        raise ValueError('Block training scope requires adapter-type full')
    if not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError('min-lr-ratio must be finite in [0,1]')
    if args.source_night_samples < 1:
        raise ValueError('source-night-samples must be positive')
    if args.lr_schedule == 'constant' and args.warmup_steps:
        raise ValueError('warmup-steps requires lr-schedule cosine')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight-decay must be finite and nonnegative')
    weights = ('global_weight','contrast_weight','patch_weight','keep_weight',
               'ot_weight','magnitude_weight','attention_weight','relation_weight',
               'synthetic_guidance_weight','source_night_weight')
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in weights):
        raise ValueError('Loss weights must be finite and nonnegative')
    if not any(getattr(args, name) > 0 for name in weights):
        raise ValueError('At least one loss weight must be positive')
    if args.source_night_weight > 0 and args.global_weight == args.contrast_weight == 0:
        raise ValueError('Source night retrieval needs global-weight or contrast-weight > 0')
    if args.source_night_weight > 0 and args.weather != 'night':
        raise ValueError('Source night views are currently specified only for weather night')
    if args.attention_weight > 0 and args.adapter_type != 'full':
        raise ValueError('Attention distillation requires the independent full Night backbone')
    if (args.contrast_weight > 0 and args.batch_size < 2 and args.contrast_queue_size == 0
            and args.fixed_day_bank is None
            and not any(getattr(args, name) > 0 for name in weights if name != 'contrast_weight')):
        raise ValueError('Contrast-only training needs batch-size >= 2 or a queue')
    if any(getattr(args, name) > 0 for name in ('patch_weight','keep_weight','ot_weight',
                                                'magnitude_weight','attention_weight')) and args.pair_calibration is None:
        raise ValueError('Local losses require verified Stage2B calibration; identity patches are invalid')
    if args.bf16 and (torch.device(args.device).type != 'cuda'
                      or not torch.cuda.is_bf16_supported()):
        raise ValueError('--bf16 requires a compatible CUDA GPU')


def _load_initial_adapter(net, args, checkpoint_hash, base, preprocessing):
    if args.init_adapter_dir is None:
        return
    directory = args.init_adapter_dir.expanduser().resolve(strict=True)
    metadata = json.loads((directory / 'adapter.json').read_text(encoding='utf-8'))
    expected = dict(adapter_type=args.adapter_type, bottleneck=args.hidden,
                    lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout, lora_blocks=args.lora_blocks)
    if (metadata.get('weather') != args.weather or
            metadata.get('checkpoint_sha256') != checkpoint_hash or
            metadata.get('model_config') != base.config or
            metadata.get('preprocessing') != preprocessing or
            adapter_model_kwargs(metadata) != expected):
        raise ValueError('Initial adapter is incompatible with weather, checkpoint or adapter configuration')
    net.adapters[args.weather].load_state_dict(
        load_file(str(directory / 'adapter.safetensors'), device='cpu'), strict=True)


def _save_run(out, net, weather, metadata):
    """Publish lightweight weights and metadata via atomic replacements."""
    out.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous()
             for key, value in net.adapters[weather].state_dict().items()}
    with tempfile.NamedTemporaryFile(dir=out, prefix='.adapter_', suffix='.safetensors',
                                     delete=False) as handle:
        weights_tmp = Path(handle.name)
    try:
        save_file(state, str(weights_tmp))
        os.replace(weights_tmp, out / 'adapter.safetensors')
    finally:
        weights_tmp.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=out, prefix='.adapter_', suffix='.json',
                                     mode='w', encoding='utf-8', delete=False) as handle:
        metadata_tmp = Path(handle.name)
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    try:
        os.replace(metadata_tmp, out / 'adapter.json')
    finally:
        metadata_tmp.unlink(missing_ok=True)


def main(argv=None):
    run_started = time.monotonic()
    args = parse_args(argv)
    _validate(args)
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; use a new experiment directory')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_size = (224, 224)  # match Stage2B calibration and checkpoint preprocessing
    augmentation_config = load_augmentation_config(args)
    augmentation_info = augmentation_metadata(augmentation_config, args.seed, model_size,
                                              args.augmentation_epoch_offset)
    source_night_config = None
    source_night_info = {'enabled': False}
    if args.source_night_weight > 0:
        from weather_vpr.data.source_night import SourceNightConfig
        from weather_vpr.data.student_augmentation import SOURCE_NIGHT_SEED_POLICY
        values = (json.loads(args.source_night_config.expanduser().resolve(strict=True)
                             .read_text(encoding='utf-8'))
                  if args.source_night_config is not None else {})
        source_night_config = SourceNightConfig.from_dict(values)
        source_night_info = {
            'enabled': True, 'config': source_night_config.to_dict(),
            'seed_policy': SOURCE_NIGHT_SEED_POLICY,
            'scope': 'Student receives only a simulated low-light view of training Day; frozen target is clean Day',
            'loss': 'source_night_weight * (global_weight * global + contrast_weight * cross_domain_infonce)',
            'geometry': 'unchanged; no source-view patch supervision is introduced',
        }
    dataset = AcceptedWeatherPairs(args.manifest, args.weather, model_size,
                                  args.exclude_places, args.verify_hashes,
                                  night_augmentation=augmentation_config, seed=args.seed,
                                  source_night_config=source_night_config)
    reviewed_pairs = prepare_auxiliary_dataset(args, model_size)
    loader_options = dict(num_workers=args.num_workers,
                          collate_fn=compact_training_batch if args.compact_batches else list,
                          pin_memory=torch.device(args.device).type == 'cuda')
    if args.num_workers:
        loader_options['prefetch_factor'] = args.prefetch_factor
    if args.batch_sampling == 'distinct':
        sampler = DistinctPlaceBatchSampler([row['place'] for row in dataset.rows],
                                            args.batch_size, seed=args.seed)
        dl = DataLoader(dataset, batch_sampler=sampler, **loader_options)
    else:
        sampler = None
        dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        drop_last=False, **loader_options)
    if torch.device(args.device).type == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    base = load_checkpoint_model(args.checkpoint, device=args.device,
                                 backbone_repo=args.backbone_repo)
    if tuple(base.image_size) != model_size:
        raise ValueError(f'Expected trained model image size {model_size}; got {base.image_size}')
    if args.adapter_type == 'lora' and base.config['backbone_arch'] != 'dinov2_vitb14':
        raise ValueError('This LoRA experiment is specified for dinov2_vitb14')
    net = WeatherAdapterSALAD(base, (args.weather,), bottleneck=args.hidden,
                              adapter_type=args.adapter_type, lora_rank=args.lora_rank,
                              lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                              lora_blocks=args.lora_blocks).to(args.device)
    if args.gradient_checkpointing:
        net.adapters[args.weather].gradient_checkpointing = True
    teacher_hash = sha256(args.checkpoint)
    preprocessing = input_preprocessing(model_size)
    _load_initial_adapter(net, args, teacher_hash, base, preprocessing)
    from weather_vpr.training.optimization import (
        configure_full_backbone_scope, full_backbone_parameter_groups, warmup_cosine_multiplier,
    )
    scope_info = (configure_full_backbone_scope(
        net.adapters[args.weather], args.full_train_scope, args.full_train_blocks)
        if args.adapter_type == 'full' else None)
    params = tuple(param for param in net.adapter_parameters(args.weather) if param.requires_grad)
    if not params:
        raise RuntimeError('Selected weather adapter has no trainable parameters')
    if scope_info is not None:
        print(f'Night full-weight training scope: {scope_info}', flush=True)
    groups = (full_backbone_parameter_groups(
        net.adapters[args.weather], learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, layer_decay=args.layer_decay)
        if args.adapter_type == 'full' else params)
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    remaining_batches = args.max_steps or len(dl) * args.epochs
    planned_updates = 0
    for _ in range(args.epochs):
        batches = min(len(dl), remaining_batches)
        planned_updates += math.ceil(batches / args.gradient_accumulation_steps)
        remaining_batches -= batches
    if args.lr_schedule == 'cosine':
        warmup_cosine_multiplier(0, planned_updates, warmup_steps=args.warmup_steps,
                                 min_multiplier=args.min_lr_ratio)
    bank_descriptors = None
    bank_info = None
    if args.fixed_day_bank is not None:
        from weather_vpr.training.day_bank import load_or_build_day_bank
        print(f'Preparing frozen Day bank from {len(dataset)} training pairs...', flush=True)
        bank_descriptors, bank_places, bank_info = load_or_build_day_bank(
            net, dataset, teacher_hash, preprocessing, args.fixed_day_bank,
            args.device, batch_size=args.batch_size, num_workers=args.num_workers)
        queue_device = (args.device if args.day_bank_device in ('cuda', 'auto')
                        and torch.device(args.device).type == 'cuda' else 'cpu')
        queue = DayDescriptorQueue(len(bank_places), storage_device=queue_device,
                                   include_current_keys=False)
        queue.enqueue(bank_descriptors, bank_places)
        print(f'Fixed training Day bank: {len(queue)} keys; {bank_info["cache_status"]}', flush=True)
    else:
        queue = DayDescriptorQueue(args.contrast_queue_size, storage_device='cpu')
    auxiliary = build_auxiliary_branch(
        args, dataset, net, teacher_hash, preprocessing,
        day_descriptors=bank_descriptors, day_bank_info=bank_info, reviewed_pairs=reviewed_pairs)
    need_local = any(getattr(args, name) > 0 for name in
                     ('patch_weight','keep_weight','ot_weight','magnitude_weight','attention_weight'))
    need_attention = args.attention_weight > 0
    if need_attention:
        from weather_vpr.models.attention import capture_cls_attention
        from weather_vpr.losses.attention import trusted_attention_alignment
    if args.patch_visibility == 'photometric':
        from weather_vpr.data.patch_visibility import patch_visibility_from_image_tensor, visibility_metadata
        visibility_info = visibility_metadata()
    else:
        visibility_info = {'mode': 'off'}
    effective_calibration = args.pair_calibration if need_local else None
    cal = load_calibration(effective_calibration, args.checkpoint, model_size, args.calibration_index)
    out.mkdir(parents=True, exist_ok=True)
    resolved_config = training_configuration(args, augmentation_info)
    resolved_config['effective_source_night'] = source_night_info
    resolved_config['effective_full_train_scope'] = scope_info
    resolved_config['effective_patch_visibility'] = visibility_info
    resolved_config['effective_auxiliary'] = (auxiliary.metadata() if auxiliary is not None
                                             else {'enabled': False, 'mode': args.auxiliary_mode})
    (out / 'training_config.json').write_text(
        json.dumps(resolved_config, indent=2), encoding='utf-8')
    # Persist the input/optimization contract before the first update. This is
    # a run plan, not a declaration that an exploratory run is a final test.
    (out / 'experiment_plan.json').write_text(json.dumps({
        'checkpoint_sha256': teacher_hash, 'initial_adapter':
        str(args.init_adapter_dir) if args.init_adapter_dir else None,
        'manifest_sha256': sha256(dataset.manifest), 'seed': args.seed,
        'planned_optimizer_updates': planned_updates,
        'main_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'source_night_samples': args.source_night_samples if args.source_night_weight else 0,
        'full_train_scope': scope_info, 'preprocessing': preprocessing,
        'auxiliary': resolved_config['effective_auxiliary'],
        'train_code_sha256': sha256(Path(__file__)),
        'auxiliary_code_sha256': sha256(Path(__file__).with_name('auxiliary.py')),
        'statement': 'Fixed training run budget; no test-driven selection is performed by this script',
    }, indent=2), encoding='utf-8')
    log_path = out / 'training_log.jsonl'
    match_cache = {} if need_local and not args.no_correspondence_cache else None
    total_positive_patches = 0
    manifest_hash = sha256(dataset.manifest)
    calibration_hash = sha256(effective_calibration) if effective_calibration else None
    step = 0
    optimizer_steps = 0
    accumulation_count = 0
    augmentation_counts = {'samples': 0, 'applied_samples': 0, 'operations': {}}
    source_night_counts = {'samples': 0, 'applied_samples': 0, 'operations': {}}
    last_gradient_norm = 0.0
    student_views = {'qwen': 0, 'source_night': 0, 'real_auxiliary': 0, 'synthetic_auxiliary': 0}
    main_stream_digest = hashlib.sha256()
    auxiliary_skipped_batches = auxiliary_empty_streak = 0
    optimizer.zero_grad(set_to_none=True)

    def optimizer_update(count):
        """Finish one accumulation window, including a short final window."""
        nonlocal optimizer_steps, accumulation_count, last_gradient_norm
        if not any(param.grad is not None for param in params):
            raise RuntimeError('No weather-backbone/adapter gradients: check DINOv2 no_grad boundary')
        if count != args.gradient_accumulation_steps:
            correction = args.gradient_accumulation_steps / count
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(correction)
        multiplier = (warmup_cosine_multiplier(
            optimizer_steps, planned_updates, warmup_steps=args.warmup_steps,
            min_multiplier=args.min_lr_ratio) if args.lr_schedule == 'cosine' else 1.0)
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group['lr'] = initial_lr * multiplier
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
        last_gradient_norm = float(norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulation_count = 0

    def metadata(epochs_completed):
        return dict(
            weather=args.weather, adapter_type=args.adapter_type,
            elapsed_seconds=time.monotonic() - run_started,
            peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(args.device)
                                       if torch.device(args.device).type == 'cuda' else 0),
            bottleneck=args.hidden, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, lora_blocks=args.lora_blocks,
            lora_targets=['attn.qkv', 'attn.proj'] if args.adapter_type == 'lora' else [],
            steps=step, optimizer_steps=optimizer_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_size=args.batch_size,
            effective_batch_size=args.batch_size * args.gradient_accumulation_steps,
            save_every_steps=args.save_every_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            layer_decay=args.layer_decay, lr_schedule=args.lr_schedule,
            full_train_scope=scope_info,
            warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
            gradient_checkpointing=args.gradient_checkpointing,
            precision='bf16' if args.bf16 else 'fp32',
            epochs_completed=epochs_completed, training_pairs=len(dataset),
            loss_weights={'global': args.global_weight, 'contrast': args.contrast_weight,
                          'patch': args.patch_weight, 'keep': args.keep_weight,
                          'ot': args.ot_weight, 'magnitude': args.magnitude_weight,
                          'attention': args.attention_weight,
                          'relation': args.relation_weight,
                          'synthetic_guidance': args.synthetic_guidance_weight,
                          'source_night': args.source_night_weight},
            positive_patches_seen=total_positive_patches,
            checkpoint_sha256=teacher_hash, model_config=base.config,
            preprocessing=preprocessing, manifest=str(dataset.manifest),
            manifest_sha256=manifest_hash,
            pair_calibration=str(effective_calibration) if effective_calibration else None,
            pair_calibration_sha256=calibration_hash,
            exclude_places=str(args.exclude_places) if args.exclude_places else None,
            init_adapter_dir=str(args.init_adapter_dir) if args.init_adapter_dir else None,
            batch_sampling=args.batch_sampling, contrast_queue_size=args.contrast_queue_size,
            fixed_day_bank=bank_info, contrast_hard_negatives=args.contrast_hard_negatives,
            relation_temperature=args.relation_temperature,
            relation_teacher_temperature=args.relation_teacher_temperature,
            relation_hard_negatives=args.relation_hard_negatives,
            source_night=source_night_info,
            auxiliary=(auxiliary.metadata() if auxiliary is not None else {
                'enabled': False, 'mode': args.auxiliary_mode, 'weight': args.auxiliary_weight}),
            auxiliary_gradient_every=args.auxiliary_gradient_every,
            auxiliary_skipped_batches=auxiliary_skipped_batches,
            student_views=dict(student_views),
            main_stream_sha256=main_stream_digest.hexdigest(),
            source_night_samples=args.source_night_samples,
            sampled_source_night_counts=source_night_counts,
            min_teacher_keep=args.min_teacher_keep, seed=args.seed,
            importance_mode=args.importance_mode,
            patch_visibility=visibility_info,
            attention_distillation={'enabled': need_attention, 'block': 'last',
                'map': 'head-mean CLS-to-patch self-attention',
                'support': 'Stage2B A-only correspondences weighted by teacher keep and optional observed visibility',
                'teacher': 'frozen clean Day; no identity-coordinate fallback'},
            night_augmentation=augmentation_info,
            sampled_augmentation_counts=augmentation_counts,
            code_version=('weather_vpr_stage1_paired_auxiliary_v8' if auxiliary is not None else
                          'weather_vpr_full_night_patch_attention_v7' if args.adapter_type == 'full'
                          else 'weather_vpr_lora_v1'),
            statement='Experimental model; real-weather Recall must be measured separately')

    for epoch in range(args.epochs):
        dataset.set_epoch(args.augmentation_epoch_offset + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        for batch_index, batch in enumerate(dl):
            main_indices = [item['index'] for item in batch]
            main_stream_digest.update(json.dumps(
                [epoch, main_indices], separators=(',', ':')).encode('utf-8'))
            probe = (AuxiliaryGradientProbe(params, args.gradient_accumulation_steps)
                     if auxiliary is not None and args.auxiliary_gradient_every
                     and (optimizer_steps + 1) % args.auxiliary_gradient_every == 0
                     and accumulation_count == 0 else None)
            day = (torch.stack([v['day_tensor'] for v in batch]).to(args.device, non_blocking=True)
                   if bank_descriptors is None or need_local else None)
            weather = (torch.stack([v['weather_tensor'] for v in batch]).to(args.device, non_blocking=True)
                       if need_local or args.synthetic_guidance_weight > 0 else None)
            student_weather = torch.stack([v['student_tensor'] for v in batch]).to(args.device)
            sampled_counts = sampled_augmentation_counts([v['augmentation_record'] for v in batch])
            merge_augmentation_counts(augmentation_counts, sampled_counts)
            places = [item['place'] for item in batch]
            teacher_attention = None
            # Pass 1: the original frozen Day teacher.
            if bank_descriptors is not None and not need_local:
                day_desc = bank_descriptors[[item['index'] for item in batch]].to(args.device)
                day_aux = None
            else:
                if need_attention:
                    with capture_cls_attention(net.base.backbone) as capture:
                        teacher_output = net.teacher(day, return_aux=need_local)
                        teacher_attention = capture.attention.detach()
                else:
                    teacher_output = net.teacher(day, return_aux=need_local)
                day_desc, day_aux = teacher_output if need_local else (teacher_output, None)
            generated_day_desc = None
            if need_local:
                indices = [item['index'] for item in batch]
                if match_cache is None or any(index not in match_cache for index in indices):
                    # Frozen generated-view features verify correspondence only;
                    # student targets still come exclusively from the day branch.
                    generated_day_desc, generated_teacher_aux = net.teacher(weather, return_aux=True)
                    fresh_match, fresh_q = correspondence(batch, day_aux, generated_teacher_aux,
                                                          cal, dataset.manifest)
                    if match_cache is not None:
                        for index, m, conf in zip(indices, fresh_match, fresh_q):
                            match_cache[index] = (m.cpu(), conf.cpu())
                    del generated_teacher_aux
                if match_cache is None:
                    match, q = fresh_match, fresh_q
                else:
                    match = torch.stack([match_cache[index][0] for index in indices])
                    q = torch.stack([match_cache[index][1] for index in indices])
                match, q = match.to(args.device), q.to(args.device)
                verified_count = int((q > 0).sum())
                verified_mass = float(q.sum())
                if args.patch_visibility == 'photometric':
                    visibility = patch_visibility_from_image_tensor(student_weather).flatten(1)
                    if visibility.shape != q.shape:
                        raise ValueError('Augmented visibility and trusted match patch grids differ')
                    q = (q.detach() * visibility).detach()
                    mean_visibility = float(visibility.mean())
                    del visibility
                else:
                    mean_visibility = 1.0
            else:
                match = q = None
                verified_count = 0
                verified_mass = mean_visibility = 0.0
            if args.synthetic_guidance_weight > 0 and generated_day_desc is None:
                generated_day_desc = net.teacher(weather, return_aux=False)
            # Pass 2: only the selected weather parameters receive gradients.
            ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                   if args.bf16 else nullcontext())
            student_attention = None
            with ctx:
                if need_attention:
                    with capture_cls_attention(net.adapters[args.weather]) as capture:
                        student_output = net.student(student_weather, args.weather, return_aux=need_local)
                        student_attention = capture.attention
                else:
                    student_output = net.student(student_weather, args.weather, return_aux=need_local)
            desc, aux = student_output if need_local else (student_output, None)
            zero = desc.float().sum() * 0.0
            g_loss = descriptor_alignment(desc, day_desc) if args.global_weight > 0 else zero
            if args.contrast_weight > 0:
                contrast = queue.contrastive_loss(
                    desc, day_desc, places, args.temperature,
                    hard_negatives=args.contrast_hard_negatives)
                c_loss = contrast['loss']
            else:
                contrast = {'effective_negative_pairs': 0,
                            'effective_negatives_per_query': [],
                            'mean_effective_negatives': 0.0,
                            'positive_pairs': 0}
                c_loss = zero
            if args.relation_weight > 0:
                relation = queue.relational_distillation_loss(
                    desc, day_desc, places,
                    student_temperature=args.relation_temperature,
                    teacher_temperature=args.relation_teacher_temperature,
                    hard_negatives=args.relation_hard_negatives)
                r_loss = relation['loss']
            else:
                relation = {'valid_queries': 0, 'mean_support': 0., 'teacher_entropy': 0.}
                r_loss = zero
            s_loss = (descriptor_alignment(desc, generated_day_desc)
                      if args.synthetic_guidance_weight > 0 else zero)
            if need_attention:
                attention = trusted_attention_alignment(
                    student_attention, teacher_attention, day_aux['assignment_keep'].flatten(1),
                    match, q, importance_mode=args.importance_mode)
                a_loss = attention['loss']
            else:
                attention = {'attention_valid_queries': 0, 'attention_trusted_patches': 0,
                             'attention_coverage': 0.0, 'attention_weight_mass': 0.0,
                             'attention_teacher_mass': 0.0, 'attention_student_mass': 0.0}
                a_loss = zero
            if match is not None:
                local = weighted_patch_losses(
                    aux, day_aux, match, q, importance_mode=args.importance_mode,
                    include_magnitude=args.magnitude_weight > 0,
                    include_ot=args.ot_weight > 0,
                    min_teacher_keep=args.min_teacher_keep)
                p_loss, k_loss = local['patch'], local['keep']
                o_loss, m_loss = local['ot'], local['magnitude']
                coverage = int(local['positive_patches'])
                patch_mass = float(local['positive_mass'])
                saliency_mass = float(local['teacher_keep_mass'])
                patch_weight_mass = float(local['patch_weight_mass'])
                ot_mass = float(local['ot_mass'])
                low_teacher_keep = int(local['low_teacher_keep_patches'])
            else:
                p_loss, k_loss, o_loss, m_loss, coverage = zero, zero, zero, zero, 0
                patch_mass = saliency_mass = patch_weight_mass = ot_mass = 0.0
                low_teacher_keep = 0
            total = (args.global_weight*g_loss + args.contrast_weight*c_loss +
                     args.patch_weight*p_loss + args.keep_weight*k_loss +
                     args.ot_weight*o_loss + args.magnitude_weight*m_loss +
                     args.attention_weight*a_loss + args.relation_weight*r_loss +
                     args.synthetic_guidance_weight*s_loss)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite training loss')
            (total / args.gradient_accumulation_steps).backward()
            student_views['qwen'] += len(batch)
            source_value = source_global_value = source_contrast_value = 0.0
            source_sampled_counts = {'samples': 0, 'applied_samples': 0, 'operations': {}}
            source_positive_pairs = source_negative_pairs = 0
            if args.source_night_weight > 0:
                count = min(len(batch), args.source_night_samples)
                source_tensor = torch.stack([item['source_night_tensor']
                                             for item in batch[:count]]).to(args.device)
                source_sampled_counts = sampled_augmentation_counts(
                    [item['source_night_record'] for item in batch[:count]])
                merge_augmentation_counts(source_night_counts, source_sampled_counts)
                with (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                      if args.bf16 else nullcontext()):
                    source_desc = net.student(source_tensor, args.weather, return_aux=False)
                source_zero = source_desc.float().sum() * 0.0
                source_global = (descriptor_alignment(source_desc, day_desc[:count])
                                 if args.global_weight > 0 else source_zero)
                if args.contrast_weight > 0:
                    source_contrast = queue.contrastive_loss(
                        source_desc, day_desc[:count], places[:count], args.temperature,
                        hard_negatives=args.contrast_hard_negatives)
                    source_c_loss = source_contrast['loss']
                    source_positive_pairs = int(source_contrast['positive_pairs'])
                    source_negative_pairs = int(source_contrast['effective_negative_pairs'])
                else:
                    source_c_loss = source_zero
                source_loss = args.global_weight * source_global + args.contrast_weight * source_c_loss
                if not torch.isfinite(source_loss):
                    raise FloatingPointError('Non-finite source night retrieval loss')
                (args.source_night_weight * source_loss / args.gradient_accumulation_steps).backward()
                source_value = float(source_loss.detach())
                source_global_value = float(source_global.detach())
                source_contrast_value = float(source_c_loss.detach())
                student_views['source_night'] += count
                del source_tensor, source_desc, source_loss, source_global, source_c_loss, source_zero
            if probe is not None:
                probe.after_main()
            auxiliary_value = auxiliary_weighted_value = 0.0
            auxiliary_count = 0
            auxiliary_ids, auxiliary_groups = [], []
            auxiliary_sampled_counts = {'samples': 0, 'applied_samples': 0, 'operations': {}}
            auxiliary_gradient = None
            if auxiliary is not None:
                # Keep data decoding, stochastic model operations AND backward
                # recomputation out of the historical main RNG streams.
                with auxiliary.rng_context(args.augmentation_epoch_offset + epoch, step):
                    extra = auxiliary.next_batch(args.augmentation_epoch_offset + epoch, step)
                    auxiliary_images = extra['images'].to(args.device)
                    auxiliary_targets = extra['targets'].to(args.device)
                    auxiliary_weights = extra['weights'].to(args.device)
                    with (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                          if args.bf16 else nullcontext()):
                        auxiliary_desc = net.student(auxiliary_images, args.weather, return_aux=False)
                    alignment = paired_global_alignment(auxiliary_desc, auxiliary_targets, auxiliary_weights)
                    auxiliary_loss = alignment['loss']
                    auxiliary_count = alignment['valid_samples']
                    if not auxiliary_count:
                        auxiliary_skipped_batches += 1
                        auxiliary_empty_streak += 1
                        if auxiliary_empty_streak >= 3:
                            raise RuntimeError('Three consecutive auxiliary batches have no accepted pairs; review pipeline')
                    else:
                        auxiliary_empty_streak = 0
                    if not torch.isfinite(auxiliary_loss):
                        raise FloatingPointError('Non-finite paired auxiliary loss')
                    if auxiliary_count:
                        (args.auxiliary_weight * auxiliary_loss / args.gradient_accumulation_steps).backward()
                    auxiliary_value = float(auxiliary_loss.detach())
                    auxiliary_weighted_value = args.auxiliary_weight * auxiliary_value
                    auxiliary_ids, auxiliary_groups = extra['sample_ids'], extra['groups']
                    auxiliary_sampled_counts = extra['augmentation']
                    student_views[args.auxiliary_mode + '_auxiliary'] += len(auxiliary_images)
                    del auxiliary_images, auxiliary_targets, auxiliary_weights, auxiliary_desc
                    del auxiliary_loss, alignment, extra
                if probe is not None:
                    auxiliary_gradient = probe.after_auxiliary()
            del probe
            if not any(param.grad is not None for param in params):
                raise RuntimeError('No weather-backbone/adapter gradients: check DINOv2 no_grad boundary')
            accumulation_count += 1
            final_microbatch = (batch_index + 1 == len(dl)
                                or (args.max_steps and step + 1 >= args.max_steps))
            did_update = accumulation_count == args.gradient_accumulation_steps or final_microbatch
            if did_update:
                optimizer_update(accumulation_count)
            if bank_descriptors is None and (args.contrast_weight > 0 or args.relation_weight > 0):
                queue.enqueue(day_desc, places)
            step += 1
            total_positive_patches += coverage
            metrics = dict(step=step, optimizer_step=optimizer_steps,
                           elapsed_seconds=time.monotonic() - run_started,
                           accumulation_count=accumulation_count,
                           epoch=epoch+1, weather=args.weather,
                           adapter_type=args.adapter_type,
                           main_sample_indices=main_indices,
                           source_sample_indices=main_indices[:min(len(batch), args.source_night_samples)]
                           if args.source_night_weight > 0 else [],
                           augmentation=sampled_counts,
                           loss=(float(total.detach()) +
                                 args.source_night_weight * source_value + auxiliary_weighted_value),
                           global_loss=float(g_loss.detach()),
                           global_retrieval_loss=float(c_loss.detach()),
                           contrast=float(c_loss.detach()), patch=float(p_loss.detach()),
                           keep=float(k_loss.detach()), ot=float(o_loss.detach()),
                           magnitude=float(m_loss.detach()), positive_patches=coverage,
                           attention_loss=float(a_loss.detach()),
                           attention_valid_queries=int(attention['attention_valid_queries']),
                           attention_trusted_patches=int(attention['attention_trusted_patches']),
                           attention_coverage=float(attention['attention_coverage']),
                           attention_weight_mass=float(attention['attention_weight_mass']),
                           attention_teacher_mass=float(attention['attention_teacher_mass']),
                           attention_student_mass=float(attention['attention_student_mass']),
                           verified_positive_patches=verified_count,
                           verified_positive_mass=verified_mass,
                           patch_visibility_mean=mean_visibility,
                           relation_loss=float(r_loss.detach()),
                           relation_valid_queries=int(relation['valid_queries']),
                           relation_mean_support=float(relation['mean_support']),
                           relation_teacher_entropy=float(relation['teacher_entropy']),
                           synthetic_guidance_loss=float(s_loss.detach()),
                           source_night_loss=source_value,
                           source_night_global_loss=source_global_value,
                           source_night_contrast_loss=source_contrast_value,
                           source_night_augmentation=source_sampled_counts,
                           source_night_positive_pairs=source_positive_pairs,
                           source_night_negative_pairs=source_negative_pairs,
                           auxiliary_loss=auxiliary_value,
                           auxiliary_weighted_loss=auxiliary_weighted_value,
                           auxiliary_samples=auxiliary_count,
                           auxiliary_sample_ids=auxiliary_ids,
                           auxiliary_groups=auxiliary_groups,
                           auxiliary_augmentation=auxiliary_sampled_counts,
                           auxiliary_gradient=auxiliary_gradient,
                           auxiliary_skipped_batches=auxiliary_skipped_batches,
                           cumulative_student_views=dict(student_views),
                           gradient_norm_pre_clip=last_gradient_norm if did_update else None,
                           max_learning_rate=max(group['lr'] for group in optimizer.param_groups),
                           min_learning_rate=min(group['lr'] for group in optimizer.param_groups),
                           candidate_negative_pairs=int(contrast.get('candidate_negative_pairs', 0)),
                           patch_coverage=coverage / (q.numel() if q is not None else len(batch) * 256),
                           positive_mass=patch_mass, teacher_keep_mass=saliency_mass,
                           patch_weight_mass=patch_weight_mass,
                           ot_mass=ot_mass, low_teacher_keep_patches=low_teacher_keep,
                           effective_negative_pairs=int(contrast['effective_negative_pairs']),
                           effective_negatives_per_query=contrast['effective_negatives_per_query'],
                           mean_effective_negatives=float(contrast['mean_effective_negatives']),
                           positive_pairs=int(contrast['positive_pairs']),
                           mean_positive_keys_per_query=int(contrast['positive_pairs']) / len(batch),
                           queue_size=len(queue), fixed_day_bank=bank_descriptors is not None)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
            if ((step == 1 and args.adapter_type != 'full' and did_update) or
                    (args.save_every_steps and step % args.save_every_steps == 0 and did_update)):
                _save_run(out, net, args.weather, metadata(epoch + 1))
            if step == 1 or step % 20 == 0:
                print(metrics, flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if accumulation_count:
            optimizer_update(accumulation_count)
        _save_run(out, net, args.weather, metadata(epoch + 1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f'Saved {args.weather} {args.adapter_type} parameters: {out / "adapter.safetensors"}')
    if need_local and total_positive_patches == 0:
        print('WARNING: trusted patch coverage was zero; local losses gave no training signal.')


if __name__ == '__main__':
    main()

"""Train one weather LoRA or output adapter using a shared frozen DINOv2-SALAD."""
from __future__ import annotations
import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import random
import tempfile
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from weather_vpr.utils.backend import enable_backend_imports, project_root
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs, sha256
from weather_vpr.data.correspondence import correspondence, load_calibration
from weather_vpr.data.place_sampler import DistinctPlaceBatchSampler
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.losses.distillation import weighted_patch_losses, descriptor_alignment

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
    p.add_argument('--output-dir', type=Path, default=project_root() / "runs/night_lora_sgpd_01")
    p.add_argument('--init-adapter-dir', type=Path, help='Initialize from a completed matching adapter run')
    p.add_argument('--exclude-places', type=Path, help='JSON [[city,place_id], ...] for held-out places')
    p.add_argument('--pair-calibration', type=Path,
                   default=project_root() / 'configs/dino_salad_stage2b_calibration.json',
                   help='Stage2B JSON calibration for the exact teacher checkpoint')
    p.add_argument('--calibration-index', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--batch-sampling', choices=('distinct','random'), default='distinct')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=0, help='0 = all')
    p.add_argument('--save-every-steps', type=int, default=200)
    p.add_argument('--adapter-type', choices=('output','lora'), default='lora')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=16)
    p.add_argument('--lora-dropout', type=float, default=0)
    p.add_argument('--lora-blocks', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--global-weight', type=float, default=1.0)
    p.add_argument('--contrast-weight', type=float, default=1.0)
    p.add_argument('--patch-weight', type=float, default=0.1)
    p.add_argument('--keep-weight', type=float, default=0.0)
    p.add_argument('--ot-weight', type=float, default=0.0)
    p.add_argument('--magnitude-weight', type=float, default=0.0)
    p.add_argument('--importance-mode', choices=['saliency', 'uniform'], default='saliency')
    p.add_argument('--min-teacher-keep', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--contrast-queue-size', type=int, default=256)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--no-correspondence-cache', action='store_true',
                   help='Recompute trusted patch matches every epoch instead of caching them in RAM')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def _validate(args):
    if args.max_steps < 0 or args.num_workers < 0 or args.contrast_queue_size < 0:
        raise ValueError('max-steps, num-workers and contrast-queue-size must be nonnegative')
    if min(args.batch_size, args.epochs, args.hidden, args.lora_rank,
           args.lora_blocks, args.save_every_steps) < 1:
        raise ValueError('batch-size, epochs, hidden, LoRA rank/blocks and save interval must be positive')
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise ValueError('lora-alpha must be finite and positive')
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        raise ValueError('lora-dropout must be finite in [0,1)')
    if not math.isfinite(args.min_teacher_keep) or not 0 <= args.min_teacher_keep < 1:
        raise ValueError('min-teacher-keep must be finite in [0,1)')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be finite and positive')
    weights = ('global_weight','contrast_weight','patch_weight','keep_weight',
               'ot_weight','magnitude_weight')
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in weights):
        raise ValueError('Loss weights must be finite and nonnegative')
    if not any(getattr(args, name) > 0 for name in weights):
        raise ValueError('At least one loss weight must be positive')
    if (args.contrast_weight > 0 and args.batch_size < 2 and args.contrast_queue_size == 0
            and not any(getattr(args, name) > 0 for name in weights if name != 'contrast_weight')):
        raise ValueError('Contrast-only training needs batch-size >= 2 or a queue')
    if any(getattr(args, name) > 0 for name in ('patch_weight','keep_weight','ot_weight',
                                                'magnitude_weight')) and args.pair_calibration is None:
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
    args = parse_args(argv)
    _validate(args)
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; use a new experiment directory')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_size = (224, 224)  # match Stage2B calibration and checkpoint preprocessing
    dataset = AcceptedWeatherPairs(args.manifest, args.weather, model_size,
                                  args.exclude_places, args.verify_hashes)
    if args.batch_sampling == 'distinct':
        sampler = DistinctPlaceBatchSampler([row['place'] for row in dataset.rows],
                                            args.batch_size, seed=args.seed)
        dl = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                        collate_fn=list, pin_memory=torch.device(args.device).type == 'cuda')
    else:
        sampler = None
        dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        num_workers=args.num_workers, collate_fn=list, drop_last=False,
                        pin_memory=torch.device(args.device).type == 'cuda')
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
    teacher_hash = sha256(args.checkpoint)
    preprocessing = input_preprocessing(model_size)
    _load_initial_adapter(net, args, teacher_hash, base, preprocessing)
    params = tuple(net.adapter_parameters(args.weather))
    if not params or any(not param.requires_grad for param in params):
        raise RuntimeError('Selected weather adapter has no trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
    queue = DayDescriptorQueue(args.contrast_queue_size, storage_device='cpu')
    need_local = any(getattr(args, name) > 0 for name in
                     ('patch_weight','keep_weight','ot_weight','magnitude_weight'))
    effective_calibration = args.pair_calibration if need_local else None
    cal = load_calibration(effective_calibration, args.checkpoint, model_size, args.calibration_index)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training_log.jsonl'
    match_cache = {} if need_local and not args.no_correspondence_cache else None
    total_positive_patches = 0
    manifest_hash = sha256(dataset.manifest)
    calibration_hash = sha256(effective_calibration) if effective_calibration else None
    step = 0

    def metadata(epochs_completed):
        return dict(
            weather=args.weather, adapter_type=args.adapter_type,
            bottleneck=args.hidden, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, lora_blocks=args.lora_blocks,
            lora_targets=['attn.qkv', 'attn.proj'] if args.adapter_type == 'lora' else [],
            steps=step, epochs_completed=epochs_completed, training_pairs=len(dataset),
            loss_weights={'global': args.global_weight, 'contrast': args.contrast_weight,
                          'patch': args.patch_weight, 'keep': args.keep_weight,
                          'ot': args.ot_weight, 'magnitude': args.magnitude_weight},
            positive_patches_seen=total_positive_patches,
            checkpoint_sha256=teacher_hash, model_config=base.config,
            preprocessing=preprocessing, manifest=str(dataset.manifest),
            manifest_sha256=manifest_hash,
            pair_calibration=str(effective_calibration) if effective_calibration else None,
            pair_calibration_sha256=calibration_hash,
            exclude_places=str(args.exclude_places) if args.exclude_places else None,
            init_adapter_dir=str(args.init_adapter_dir) if args.init_adapter_dir else None,
            batch_sampling=args.batch_sampling, contrast_queue_size=args.contrast_queue_size,
            min_teacher_keep=args.min_teacher_keep, seed=args.seed,
            importance_mode=args.importance_mode, code_version='weather_vpr_lora_v1',
            statement='Experimental model; real-weather Recall must be measured separately')

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        for batch in dl:
            day = torch.stack([v['day_tensor'] for v in batch]).to(args.device)
            weather = torch.stack([v['weather_tensor'] for v in batch]).to(args.device)
            places = [item['place'] for item in batch]
            # Pass 1: Day teacher. No adapter, no gradients, reusable base model.
            teacher_output = net.teacher(day, return_aux=need_local)
            day_desc, day_aux = teacher_output if need_local else (teacher_output, None)
            if need_local:
                indices = [item['index'] for item in batch]
                if match_cache is None or any(index not in match_cache for index in indices):
                    # Frozen generated-view features verify correspondence only;
                    # student targets still come exclusively from the day branch.
                    _, generated_teacher_aux = net.teacher(weather, return_aux=True)
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
            else:
                match = q = None
            # Pass 2: only the selected weather adapter receives gradients.
            optimizer.zero_grad(set_to_none=True)
            ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                   if args.bf16 else nullcontext())
            with ctx:
                student_output = net.student(weather, args.weather, return_aux=need_local)
            desc, aux = student_output if need_local else (student_output, None)
            zero = desc.float().sum() * 0.0
            g_loss = descriptor_alignment(desc, day_desc) if args.global_weight > 0 else zero
            if args.contrast_weight > 0:
                contrast = queue.contrastive_loss(desc, day_desc, places, args.temperature)
                c_loss = contrast['loss']
            else:
                contrast = {'effective_negative_pairs': 0,
                            'effective_negatives_per_query': [],
                            'mean_effective_negatives': 0.0,
                            'positive_pairs': 0}
                c_loss = zero
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
                     args.ot_weight*o_loss + args.magnitude_weight*m_loss)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite training loss')
            total.backward()
            if not any(param.grad is not None for param in params):
                raise RuntimeError('No adapter gradients: check DINOv2 no_grad boundary')
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            if args.contrast_weight > 0:
                queue.enqueue(day_desc, places)
            step += 1
            total_positive_patches += coverage
            metrics = dict(step=step, epoch=epoch+1, weather=args.weather,
                           adapter_type=args.adapter_type,
                           loss=float(total.detach()), global_loss=float(g_loss.detach()),
                           global_retrieval_loss=float(c_loss.detach()),
                           contrast=float(c_loss.detach()), patch=float(p_loss.detach()),
                           keep=float(k_loss.detach()), ot=float(o_loss.detach()),
                           magnitude=float(m_loss.detach()), positive_patches=coverage,
                           patch_coverage=coverage / (q.numel() if q is not None else len(batch) * 256),
                           positive_mass=patch_mass, teacher_keep_mass=saliency_mass,
                           patch_weight_mass=patch_weight_mass,
                           ot_mass=ot_mass, low_teacher_keep_patches=low_teacher_keep,
                           effective_negative_pairs=int(contrast['effective_negative_pairs']),
                           effective_negatives_per_query=contrast['effective_negatives_per_query'],
                           mean_effective_negatives=float(contrast['mean_effective_negatives']),
                           positive_pairs=int(contrast['positive_pairs']), queue_size=len(queue))
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
            if step == 1 or step % args.save_every_steps == 0:
                _save_run(out, net, args.weather, metadata(epoch + 1))
            if step == 1 or step % 20 == 0:
                print(metrics, flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        _save_run(out, net, args.weather, metadata(epoch + 1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f'Saved only {args.weather} {args.adapter_type} parameters: {out / "adapter.safetensors"}')
    if need_local and total_positive_patches == 0:
        print('WARNING: trusted patch coverage was zero; local losses gave no training signal.')


if __name__ == '__main__':
    main()

"""Train exactly one weather output adapter. Run from scripts/train.py."""
from __future__ import annotations
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from weather_vpr.utils.backend import enable_backend_imports, project_root
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs, sha256
from weather_vpr.data.correspondence import correspondence, load_calibration
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.losses.distillation import weighted_patch_losses, descriptor_alignment, cross_domain_infonce

# WORKSPACE sources are reused rather than bundled or copied.
enable_backend_imports()
from workflow.model import input_preprocessing, load_checkpoint_model

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=project_root() / "weights/dino_salad.ckpt", help='FULL SALAD checkpoint')
    p.add_argument('--backbone-repo', type=Path, default=project_root() / "third_party/dinov2",
                   help='Locally cloned facebookresearch/dinov2 with hubconf.py')
    p.add_argument('--manifest', type=Path, default=project_root() / "data/train/manifest.jsonl", help='Reviewed Qwen JSONL')
    p.add_argument('--weather', required=True, help='night, rain, fog, snow, ...')
    p.add_argument('--output-dir', type=Path, default=project_root() / "runs/night_experiment")
    p.add_argument('--exclude-places', type=Path, help='JSON [[city,place_id], ...] for held-out places')
    p.add_argument('--pair-calibration', type=Path,
                   help='Stage2B JSON calibration for the exact teacher checkpoint')
    p.add_argument('--calibration-index', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=0, help='0 = all')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--global-weight', type=float, default=1.0)
    p.add_argument('--contrast-weight', type=float, default=1.0)
    p.add_argument('--patch-weight', type=float, default=0.1)
    p.add_argument('--keep-weight', type=float, default=0.0)
    p.add_argument('--importance-mode', choices=['saliency', 'uniform'], default='saliency')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--no-correspondence-cache', action='store_true',
                   help='Recompute trusted patch matches every epoch instead of caching them in RAM')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError('max-steps cannot be negative')
    if min(args.batch_size, args.epochs, args.hidden) < 1:
        raise ValueError('batch-size, epochs, hidden must be positive')
    if args.num_workers < 0:
        raise ValueError('num-workers cannot be negative')
    if any(getattr(args, name) < 0 for name in ('global_weight','contrast_weight','patch_weight','keep_weight')):
        raise ValueError('Loss weights cannot be negative')
    if not any(getattr(args, name) > 0 for name in ('global_weight','contrast_weight','patch_weight','keep_weight')):
        raise ValueError('At least one loss weight must be positive')
    if args.contrast_weight > 0 and args.batch_size < 2 and not any(
            getattr(args, name) > 0 for name in ('global_weight','patch_weight','keep_weight')):
        raise ValueError('Contrast-only training needs batch-size >= 2')
    if (args.patch_weight > 0 or args.keep_weight > 0) and args.pair_calibration is None:
        raise ValueError('Local losses require verified Stage2B calibration; do not assume identity patches')
    if args.bf16 and (torch.device(args.device).type != 'cuda'
                      or not torch.cuda.is_bf16_supported()):
        raise ValueError('--bf16 requires a compatible CUDA GPU')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; use a new experiment directory')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_size = (224, 224)  # match Stage2B calibration and checkpoint preprocessing
    dataset = AcceptedWeatherPairs(args.manifest, args.weather, model_size,
                                  args.exclude_places, args.verify_hashes)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=list, drop_last=False,
                    pin_memory=torch.device(args.device).type == 'cuda')
    if args.device == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    base = load_checkpoint_model(args.checkpoint, device=args.device,
                                 backbone_repo=args.backbone_repo)
    if tuple(base.image_size) != model_size:
        raise ValueError(f'Expected trained model image size {model_size}; got {base.image_size}')
    net = WeatherAdapterSALAD(base, (args.weather,), bottleneck=args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(net.adapter_parameters(args.weather),
                                   lr=args.learning_rate, weight_decay=1e-4)
    need_local = args.patch_weight > 0 or args.keep_weight > 0
    cal = load_calibration(args.pair_calibration, args.checkpoint, model_size, args.calibration_index)
    teacher_hash = sha256(args.checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training_log.jsonl'
    match_cache = {} if need_local and not args.no_correspondence_cache else None
    total_positive_patches = 0
    step = 0
    for epoch in range(args.epochs):
        net.train()
        for batch in dl:
            day = torch.stack([v['day_tensor'] for v in batch]).to(args.device)
            weather = torch.stack([v['weather_tensor'] for v in batch]).to(args.device)
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
            # Pass 2: Weather student. Gradients flow Adapter <- frozen SALAD,
            # but no gradients/optimizer states are needed for DINOv2/SALAD.
            optimizer.zero_grad(set_to_none=True)
            ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                   if args.bf16 else nullcontext())
            with ctx:
                student_output = net.student(weather, args.weather, return_aux=need_local)
            desc, aux = student_output if need_local else (student_output, None)
            zero = desc.float().sum() * 0.0
            g_loss = descriptor_alignment(desc, day_desc) if args.global_weight > 0 else zero
            c_loss = (cross_domain_infonce(desc, day_desc,
                            [item['place'] for item in batch], args.temperature)
                      if args.contrast_weight > 0 else zero)
            if match is not None:
                local = weighted_patch_losses(aux, day_aux, match, q, importance_mode=args.importance_mode)
                p_loss, k_loss = local['patch'], local['keep']
                coverage = int(local['positive_patches'])
            else:
                p_loss, k_loss, coverage = zero, zero, 0
            total = (args.global_weight*g_loss + args.contrast_weight*c_loss +
                     args.patch_weight*p_loss + args.keep_weight*k_loss)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite training loss')
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.adapter_parameters(args.weather), max_norm=1.0)
            optimizer.step()
            step += 1
            total_positive_patches += coverage
            metrics = dict(step=step, epoch=epoch+1, weather=args.weather,
                           loss=float(total.detach()), global_loss=float(g_loss.detach()),
                           contrast=float(c_loss.detach()), patch=float(p_loss.detach()),
                           keep=float(k_loss.detach()), positive_patches=coverage)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
            if step == 1 or step % 20 == 0:
                print(metrics, flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    save_file({key: val.detach().cpu().contiguous() for key, val in
               net.adapters[args.weather].state_dict().items()},
              str(out / 'adapter.safetensors'))
    meta = dict(weather=args.weather, bottleneck=args.hidden, steps=step,
                epochs_completed=epoch + 1, training_pairs=len(dataset),
                loss_weights={'global': args.global_weight, 'contrast': args.contrast_weight,
                              'patch': args.patch_weight, 'keep': args.keep_weight},
                positive_patches_seen=total_positive_patches,
                checkpoint_sha256=teacher_hash, model_config=base.config,
                preprocessing=input_preprocessing(model_size),
                manifest=str(dataset.manifest), manifest_sha256=sha256(dataset.manifest),
                pair_calibration=str(args.pair_calibration) if args.pair_calibration else None,
                pair_calibration_sha256=sha256(args.pair_calibration) if args.pair_calibration else None,
                exclude_places=str(args.exclude_places) if args.exclude_places else None,
                seed=args.seed,
                code_version='weather_vpr_standalone_v4', importance_mode=args.importance_mode,
                statement='Experimental model; no real-weather recall measured by this script')
    (out / 'adapter.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved ONLY adapter parameters: {out / "adapter.safetensors"}')
    if need_local and total_positive_patches == 0:
        print('WARNING: trusted patch coverage was zero; local losses gave no training signal.')


if __name__ == '__main__':
    main()

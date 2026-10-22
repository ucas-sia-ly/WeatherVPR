"""Load a weather model, or the original day model, and emit one .npy descriptor."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from weather_vpr.utils.backend import enable_backend_imports, project_root
from weather_vpr.inference.loading import load_weather_model
from weather_vpr.models.io import adapter_model_kwargs

enable_backend_imports()
from workflow.model import load_checkpoint_model, input_preprocessing
from workflow.reliability_cache import preprocess_teacher_image

def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''):
            dig.update(x)
    return dig.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=project_root() / 'weights/dino_salad.ckpt')
    p.add_argument('--backbone-repo', type=Path, default=project_root() / 'third_party/dinov2')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--weather', required=True, help='day for original model, or an adapter name')
    p.add_argument('--adapter', action='append', default=[],
                   help='weather=/path/to/training/output ; repeat for each weather')
    p.add_argument('--adapter-scale', type=float, default=1.0,
                   help='Scale the selected LoRA update; use a development-selected value')
    p.add_argument('--output', type=Path, required=True, help='Output numpy .npy descriptor')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    models = {}
    for arg in args.adapter:
        if '=' not in arg:
            p.error('--adapter must be weather=/path/to/adapter_directory')
        name, location = arg.split('=', 1)
        if not name or name == 'day' or name in models:
            p.error('Adapter names must be unique and not day')
        models[name] = Path(location).expanduser().resolve()
    if args.weather != 'day' and args.weather not in models:
        p.error(f'--weather {args.weather} has no loaded adapter')
    base = load_checkpoint_model(args.checkpoint, args.device, backbone_repo=args.backbone_repo)
    ckpt_hash = sha256(args.checkpoint)
    cfg = {}
    for name, directory in models.items():
        metadata = json.loads((directory / 'adapter.json').read_text())
        if (metadata['weather'] != name or metadata['checkpoint_sha256'] != ckpt_hash
                or metadata['model_config'] != base.config
                or metadata['preprocessing'] != input_preprocessing(base.image_size)):
            raise ValueError(f'Adapter {name} incompatible with base checkpoint/config')
        adapter_model_kwargs(metadata)
        cfg[name] = metadata
    # Only the requested weather model needs to be resident.  This also lets
    # callers register different artifact types without duplicating DINOv2s.
    net = (load_weather_model(base, args.weather, models[args.weather],
                              cfg[args.weather], device=args.device,
                              adapter_scale=args.adapter_scale)
           if args.weather != 'day' else None)
    with Image.open(args.image) as im:
        img = preprocess_teacher_image(im, base.image_size).unsqueeze(0).to(args.device)
    with torch.no_grad():
        if args.weather == 'day':
            descriptor = base(img, return_aux=False)
        else:
            descriptor = net.student(img, args.weather, return_aux=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, descriptor.float().cpu().numpy())
    print(f'Saved {descriptor.shape[-1]}-D descriptor to {args.output}')


if __name__ == '__main__':
    main()

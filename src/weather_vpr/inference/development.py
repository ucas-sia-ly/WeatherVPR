"""Evaluate frozen configurations on the same spatial SVOX-train holdout.

Only train images are decoded. Test filenames provide the pre-existing spatial
exclusion buffer, using exactly the real-night training split policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from weather_vpr.data.svox_real_night import load_real_night_train, northing_holdout
from weather_vpr.inference.loading import load_weather_model
from weather_vpr.inference.svox import (
    _cache_metadata, _load_gallery_cache, _save_gallery_cache,
    _require_night_adapter_dir, gallery_fingerprint, sha256,
)
from weather_vpr.utils.backend import enable_backend_imports, project_root

enable_backend_imports()
from workflow.evaluation import EvaluationSet, build_results, exact_retrieval, extract_descriptors
from workflow.model import input_preprocessing, load_checkpoint_model


class _NightQuery(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        return self.net.student(images, 'night', return_aux=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--svox-root', type=Path, default=project_root() / 'data/svox')
    parser.add_argument('--checkpoint', type=Path, default=project_root() / 'weights/dino_salad.ckpt')
    parser.add_argument('--backbone-repo', type=Path, default=project_root() / 'third_party/dinov2')
    parser.add_argument('--gallery-cache', type=Path,
                        default=project_root() / 'runs/cache/svox_train_gallery.safetensors')
    parser.add_argument('--test-buffer-m', type=float, default=250.0)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--validation-gap-m', type=float, default=50.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or not 0 < args.validation_fraction < 0.5:
        raise ValueError('Invalid development fraction, batch size or workers')
    if args.adapter_dir is not None:
        args.adapter_dir = _require_night_adapter_dir(args.adapter_dir)
    if torch.device(args.device).type == 'cpu':
        os.environ['XFORMERS_DISABLED'] = '1'
    real = load_real_night_train(args.svox_root, test_buffer_meters=args.test_buffer_m)
    train, dev, gap = northing_holdout(real.filtered.queries,
                                      fraction=args.validation_fraction,
                                      gap_meters=args.validation_gap_m)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_hash = sha256(checkpoint)
    base = load_checkpoint_model(checkpoint, device=args.device,
                                 backbone_repo=args.backbone_repo)
    preprocessing = input_preprocessing(base.image_size)
    gallery_hash = gallery_fingerprint(real.full.references)
    expected = _cache_metadata(real.full.references, gallery_hash, checkpoint_hash, preprocessing)
    dimension = (base.config['agg_config']['num_clusters'] * base.config['agg_config']['cluster_dim']
                 + base.config['agg_config']['token_dim'])
    gallery = _load_gallery_cache(args.gallery_cache, expected, dimension)
    if gallery is None:
        gallery = extract_descriptors(base, real.full.references, base.image_size,
                                      args.device, args.batch_size, args.num_workers)
        _save_gallery_cache(args.gallery_cache, expected, gallery)
    gallery = gallery[list(real.original_reference_indices)].contiguous()
    images = [real.filtered.queries[i] for i in dev]
    positives = [real.filtered.positives[i] for i in dev]
    dataset = EvaluationSet('SVOX train spatial development', real.filtered.references,
                             images, positives, {
                                 'split': 'train', 'ground_truth': 'utm_radius',
                                 'positive_radius_meters': 25.0,
                                 'test_geometry_buffer_meters': args.test_buffer_m,
                                 'validation_fraction': args.validation_fraction,
                                 'validation_gap_meters': args.validation_gap_m,
                             })

    def evaluate(model):
        vectors = extract_descriptors(model, images, base.image_size, args.device,
                                       args.batch_size, args.num_workers)
        ranks = exact_retrieval(gallery, vectors, positives, device=args.device,
                                query_chunk_size=16, reference_chunk_size=4096,
                                reference_cache_bytes=gallery.numel() * 4 + len(gallery) * 4)
        return build_results(dataset, ranks, save_hard_cases=True)

    result = {
        'protocol': 'same SVOX train spatial holdout as train_real_night.py',
        'checkpoint_sha256': checkpoint_hash, 'gallery_sha256': gallery_hash,
        'test_images_decoded': False, 'test_geometry_sha256': real.test_geometry_sha256,
        'counts': {'optimization': len(train), 'development': len(dev), 'gap': len(gap),
                   'gallery': len(real.filtered.references)},
        'development_query_ids': [image.id for image in images],
        'baseline': evaluate(base), 'adapter': None,
    }
    if args.adapter_dir is not None:
        metadata = json.loads((args.adapter_dir / 'adapter.json').read_text())
        if (metadata.get('weather') != 'night'
                or metadata.get('checkpoint_sha256') != checkpoint_hash
                or metadata.get('model_config') != base.config
                or metadata.get('preprocessing') != preprocessing):
            raise ValueError('Night checkpoint is incompatible with original Day SALAD')
        net = load_weather_model(base, 'night', args.adapter_dir, metadata, device=args.device)
        result['adapter'] = {
            'directory': str(args.adapter_dir),
            'adapter_sha256': sha256(args.adapter_dir / 'adapter.safetensors'),
            'metadata': metadata, 'evaluation': evaluate(_NightQuery(net)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('Development baseline:', result['baseline']['recall'], flush=True)
    if result['adapter']:
        print('Development Night:', result['adapter']['evaluation']['recall'], flush=True)
    print(f'Saved {args.output}', flush=True)
    return result

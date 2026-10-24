#!/usr/bin/env python3
"""Compare a fixed fusion grid on SVOX train diagnostics using exported queries.

No model is trained or loaded. Test splits are deliberately not accepted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from safetensors.torch import load_file
from weather_vpr.inference.svox import (
    _cache_metadata, _load_gallery_cache, fused_query_descriptors,
    gallery_fingerprint, load_night_dataset, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True,
                        help='Fresh SVOX train evaluation with --save-query-descriptors')
    parser.add_argument('--svox-root', type=Path, default=Path('data/svox'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = json.loads(args.evaluation.read_text())
    if result.get('dataset') != 'SVOX' or result.get('split') != 'train':
        parser.error('Fusion selection is restricted to the SVOX train diagnostic split')
    if not result.get('adapter') or not result.get('query_descriptors') or not result.get('gallery_cache'):
        parser.error('Evaluation must include Night, a Day Gallery cache, and exported query descriptors')
    dataset = load_night_dataset(args.svox_root, split='train')
    if result['baseline']['protocol'] != dataset.protocol:
        raise ValueError('Evaluation ground-truth protocol does not match current native SVOX train')
    query_path = Path(result['query_descriptors']['path'])
    meta = json.loads(Path(result['query_descriptors']['metadata_path']).read_text())
    adapter = Path(result['adapter']['directory'])
    if (meta['query_ids'] != [image.id for image in dataset.queries]
            or meta['query_sha256'] != gallery_fingerprint(dataset.queries)
            or meta['checkpoint_sha256'] != result['checkpoint_sha256']
            or meta['night_weights_sha256'] != sha256(adapter / 'adapter.safetensors')
            or meta['night_metadata_sha256'] != sha256(adapter / 'adapter.json')
            or meta['preprocessing'] != result['preprocessing']
            or meta['query_preprocessing'] != result['query_preprocessing']
            or meta['tensor_sha256'] != sha256(query_path)):
        raise ValueError('Query export provenance no longer matches evaluation/images/weights')
    queries = load_file(str(query_path), device='cpu')
    if set(queries) != {'teacher', 'student'} or queries['teacher'].ndim != 2:
        raise ValueError('Expected ordered Teacher and Student descriptor matrices')
    dim = queries['teacher'].shape[1]
    expected = _cache_metadata(dataset.references, result['gallery_sha256'],
                               result['checkpoint_sha256'], result['preprocessing'])
    references = _load_gallery_cache(Path(result['gallery_cache']), expected, dim)
    if references is None:
        raise ValueError('Day Gallery cache does not match the recorded extraction')
    from workflow.evaluation import build_results, exact_retrieval
    scores = []
    for alpha in (0., .25, .5, .75, 1.):
        vectors = fused_query_descriptors(queries['student'], queries['teacher'], alpha)
        if len(vectors) != len(dataset.queries):
            raise ValueError('Exported query count differs from native protocol')
        rows = exact_retrieval(references, vectors, dataset.positives, device=args.device,
                               query_chunk_size=32, reference_chunk_size=4096,
                               reference_cache_bytes=1024 ** 3)
        metrics = build_results(dataset, rows, save_hard_cases=True)
        if alpha in (0., 1.):
            expected_recall = (result['baseline'] if alpha == 0 else result['adapter']['evaluation'])['recall']
            if metrics['recall'] != expected_recall:
                raise ValueError('Normalized endpoint Recall differs from the original evaluation; inspect before selection')
        scores.append({'alpha': alpha, 'evaluation': metrics,
                       'per_query': [{'query_id': image.id, **row} if row else
                                     {'query_id': image.id, 'rank': None}
                                     for image, row in zip(dataset.queries, rows)]})
        print(f"alpha={alpha}: {metrics['recall']}", flush=True)
    payload = {'dataset': 'SVOX', 'split': 'train', 'purpose': 'development_only_fusion_grid',
               'source_evaluation': str(args.evaluation.resolve()),
               'query_export_sha256': meta['tensor_sha256'],
               'night_weights_sha256': meta['night_weights_sha256'],
               'gallery_sha256': result['gallery_sha256'], 'query_sha256': meta['query_sha256'],
               'protocol': dataset.protocol, 'scores': scores,
               'cost': 'Mixed alpha requires Teacher and Student Query encoders at deployment'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

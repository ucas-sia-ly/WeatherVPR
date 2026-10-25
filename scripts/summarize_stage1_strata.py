#!/usr/bin/env python3
"""Describe fixed P0 per-query changes by brightness and 200 m UTM grid cell.

Reads completed development results and image pixels only. No model is loaded,
and grid cells are descriptive spatial groups, never certified routes or trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
BRIGHTNESS_EDGES = (0., .1, .25, .5, 1.)
SPATIAL_METERS = 200
TOPK = (1, 5, 10)
COMPARISONS = (('A', 'B'), ('C', 'B'))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def brightness_bin(value: float) -> str:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Brightness outside [0,1]: {value}')
    for index, (low, high) in enumerate(zip(BRIGHTNESS_EDGES, BRIGHTNESS_EDGES[1:])):
        if value < high or index == len(BRIGHTNESS_EDGES) - 2:
            return f'[{low:g},{high:g}' + (']' if high == 1 else ')')
    raise AssertionError('Unreachable brightness interval')


def spatial_bin(filename: str) -> str:
    fields = Path(filename).name.split('@')
    try:
        east, north = float(fields[1]), float(fields[2])
    except (ValueError, IndexError) as error:
        raise ValueError(f'Expected canonical SVOX UTM filename: {filename}') from error
    if not math.isfinite(east) or not math.isfinite(north):
        raise ValueError(f'Nonfinite UTM coordinates: {filename}')
    return f'utm200m_{math.floor(east / SPATIAL_METERS)}_{math.floor(north / SPATIAL_METERS)}'


def query_metadata(queries_root: Path) -> list[dict]:
    rows = []
    for path in sorted(queries_root.glob('*.jpg')):
        with Image.open(path) as image:
            resized = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            means = ImageStat.Stat(resized).mean
        luminance = sum(weight * mean for weight, mean in zip((.2126, .7152, .0722), means)) / 255
        rows.append({'query_id': f'images/train/queries_night/{path.name}',
                     'path': str(path.resolve()), 'sha256': digest(path),
                     'brightness': luminance, 'brightness_bin': brightness_bin(luminance),
                     'spatial_bin': spatial_bin(path.name)})
    if not rows:
        raise ValueError(f'No canonical SVOX development query images: {queries_root}')
    return rows


def validate_evaluation(evaluation: dict, query_ids: set[str]) -> None:
    if evaluation['num_queries'] != len(query_ids):
        raise ValueError('Evaluation/query image counts differ')
    errors = evaluation['error_ranks']
    if not set(errors).issubset(query_ids):
        raise ValueError('Evaluation contains unknown query IDs')
    if any(not isinstance(rank, int) or rank < 2 for rank in errors.values()):
        raise ValueError('Error lists must contain one-based non-Top1 ranks')
    for k in TOPK:
        hits = len(query_ids) - sum(rank > k for rank in errors.values())
        if evaluation['hits'][f'R@{k}'] != hits:
            raise ValueError('Evaluation error ranks cannot reconstruct published TopK hits')


def grouped_changes(before: dict, after: dict, queries: list[dict], field: str, k: int) -> dict:
    groups = {}
    if field == 'brightness_bin':
        groups = {brightness_bin(edge): [] for edge in BRIGHTNESS_EDGES[:-1]}
    for row in queries:
        groups.setdefault(row[field], []).append(row['query_id'])
    output = {}
    for name, ids in sorted(groups.items()):
        old_hits = {qid for qid in ids if before['error_ranks'].get(qid, 1) <= k}
        new_hits = {qid for qid in ids if after['error_ranks'].get(qid, 1) <= k}
        repaired, regressed = sorted(new_hits - old_hits), sorted(old_hits - new_hits)
        output[name] = {'queries': len(ids), 'before_hits': len(old_hits), 'after_hits': len(new_hits),
                        'repaired_count': len(repaired), 'regressed_count': len(regressed),
                        'repaired': repaired, 'regressed': regressed,
                        'recall_delta': (len(new_hits) - len(old_hits)) / len(ids) if ids else None}
    return output


def build_report(summary: dict, queries: list[dict]) -> dict:
    if summary.get('state') != 'complete' or summary.get('smoke'):
        raise ValueError('Require a completed full-epoch multi-seed summary')
    ids = {row['query_id'] for row in queries}
    if len(ids) != len(queries):
        raise ValueError('Duplicate canonical query IDs')
    records = summary['records']
    seeds = sorted({int(key.split('_')[0][4:]) for key in records})
    result = {}
    for seed in seeds:
        for before_arm, after_arm in COMPARISONS:
            before_record = records.get(f'seed{seed}_{before_arm}', {})
            after_record = records.get(f'seed{seed}_{after_arm}', {})
            if 'evaluation' not in before_record or 'evaluation' not in after_record:
                continue
            before_full, after_full = before_record['evaluation'], after_record['evaluation']
            if (before_full['gallery_sha256'] != after_full['gallery_sha256']
                    or before_full['checkpoint_sha256'] != after_full['checkpoint_sha256']
                    or before_full['baseline'] != after_full['baseline']):
                raise ValueError('Compared evaluations have different gallery/checkpoint/baseline provenance')
            before, after = before_full['night'], after_full['night']
            validate_evaluation(before, ids)
            validate_evaluation(after, ids)
            result[f'seed{seed}_{after_arm}_vs_{before_arm}'] = {
                f'R@{k}': {field: grouped_changes(before, after, queries, field, k)
                           for field in ('brightness_bin', 'spatial_bin')} for k in TOPK}
    if not result:
        raise ValueError('No completed B-A or B-C paired evaluations')
    return {'format_version': 1, 'evaluation_split': 'SVOX train (development) night',
            'query_count': len(queries), 'queries': queries,
            'predeclared_protocol': {'brightness_edges': BRIGHTNESS_EDGES,
                'brightness_definition': 'Rec.709 weighted mean of 224x224 PIL bilinear RGB bytes / 255; encoded pixels, not linear radiance',
                'spatial_cell_meters': SPATIAL_METERS, 'spatial_origin_utm_meters': [0, 0],
                'comparisons': ['B-A', 'B-C'], 'topk': TOPK},
            'paired_changes_by_stratum': result,
            'limitations': ['Brightness can correlate with content, camera and place.',
                           'UTM grid cells are descriptive groups; adjacent cells and correlated frames are not independent routes.',
                           'No bootstrap confidence intervals, p-values, model selection or new retrieval evaluation.']}


def markdown(report: dict) -> str:
    lines = ['# P0 descriptive query strata', '',
             'Fixed brightness edges: 0, 0.1, 0.25, 0.5, 1. Spatial groups: floor(UTM/200 m).', '',
             'Groups describe correlated development queries; they do not establish independent routes or significance.', '',
             '| Comparison | TopK | Brightness bin | Queries | Repaired | Regressed | Net hits |',
             '|---|---|---|---:|---:|---:|---:|']
    for comparison, metrics in report['paired_changes_by_stratum'].items():
        for metric, fields in metrics.items():
            for name, group in fields['brightness_bin'].items():
                lines.append(f"| {comparison} | {metric} | {name} | {group['queries']} | "
                             f"{group['repaired_count']} | {group['regressed_count']} | "
                             f"{group['after_hits']-group['before_hits']:+d} |")
    lines += ['', 'Full per-query changes in each 200 m spatial cell and brightness bin are in the adjacent JSON artifact.', '',
              *report['limitations'], '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh .json artifact; .md companion is also written')
    parser.add_argument('--queries-root', type=Path, default=ROOT / 'data/svox/images/train/queries_night')
    args = parser.parse_args(argv)
    if args.output.suffix != '.json':
        parser.error('--output must end with .json')
    if args.output.exists() or args.output.with_suffix('.md').exists():
        parser.error('Refusing existing output artifacts')
    report = build_report(json.loads(args.summary.read_text()), query_metadata(args.queries_root))
    report['summary_path'] = str(args.summary.resolve())
    report['summary_sha256'] = digest(args.summary)
    report['script_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    args.output.with_suffix('.md').write_text(markdown(report))
    print(f'Wrote {args.output}: {report["query_count"]} development queries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

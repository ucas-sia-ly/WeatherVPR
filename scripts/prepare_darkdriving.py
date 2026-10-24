#!/usr/bin/env python3
"""Clean an audited DarkDriving copy, or strictly verify its portable manifest."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weather_vpr.data.darkdriving import load_darkdriving_manifest


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plan_removals(records):
    """Remove both sides of bad pairs; retain the first complete content pair."""
    sides = {side: {r['name']: r for r in records if r['split'] == side}
             for side in ('Day', 'Night')}
    if set(sides['Day']) != set(sides['Night']):
        raise ValueError('Audit has unpaired files; reconcile them before cleaning')
    removed, signatures = {}, {}
    for name in sorted(sides['Day']):
        day, night = sides['Day'][name], sides['Night'][name]
        if day['errors'] or night['errors']:
            removed[name] = {'reason': 'corrupt_pair', 'kept_sample': '',
                             'day_error': '; '.join(day['errors']),
                             'night_error': '; '.join(night['errors'])}
            continue
        if day.get('warnings') or night.get('warnings'):
            raise ValueError(f'Audit warnings need review before cleaning: {name}')
        signature = (day['pixel_sha256'], night['pixel_sha256'])
        if signature in signatures:
            removed[name] = {'reason': 'duplicate_complete_pair',
                             'kept_sample': signatures[signature],
                             'day_error': '', 'night_error': ''}
        else:
            signatures[signature] = name
    return sides, removed


def content_groups(rows):
    """Connect pairs sharing either exact image, to expose potential split leaks."""
    parent = {r['sample_id']: r['sample_id'] for r in rows}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    seen = {}
    for row in rows:
        name = row['sample_id']
        for side in ('day', 'night'):
            key = (side, row[f'{side}_pixel_sha256'])
            if key in seen:
                a, b = sorted((find(name), find(seen[key])))
                parent[b] = a
            else:
                seen[key] = name
    for row in rows:
        row['content_group'] = find(row['sample_id'])


def prepare(root, audit, backup, workers):
    if (root / 'manifest.jsonl').exists():
        raise ValueError('Already prepared; use --check-only to validate it')
    if backup == root or root in backup.parents:
        raise ValueError('Backup must be outside the active dataset')
    if backup.exists():
        raise ValueError(f'Backup already exists: {backup}')
    records = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    sides, removed = plan_removals(records)
    expected = {f'{side}/{name}' for side in sides for name in sides[side]}
    if len(expected) != len(records) or {r['path'] for r in records} != expected:
        raise ValueError('Audit contains duplicate or unexpected paths')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if expected != actual:
        raise ValueError('Dataset inventory changed since the audit')
    # A cached decode result is usable only if every original byte still matches.
    def verify_audit(row):
        path = root / row['path']
        if path.is_symlink() or file_hash(path) != row['sha256']:
            raise ValueError(f'Image changed since the audit: {path}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(verify_audit, records), 1):
            if i % 3000 == 0 or i == len(records):
                print(f'Preflight SHA-256: {i}/{len(records)}', flush=True)
    rows = []
    for name in sorted(sides['Day']):
        if name in removed:
            continue
        row = {'sample_id': Path(name).stem,
               'sequence_id': Path(name).stem.rsplit('_', 1)[0]}
        for side, folder in (('day', 'Day'), ('night', 'Night')):
            original = sides[folder][name]
            row.update({f'{side}_path': f'{folder}/{name}',
                        f'{side}_sha256': original['sha256'],
                        f'{side}_pixel_sha256': original['pixel_sha256'],
                        f'{side}_bytes': original['bytes'],
                        f'{side}_width': original['width'],
                        f'{side}_height': original['height']})
        rows.append(row)
    content_groups(rows)
    backup.mkdir(parents=True)
    journal = []
    for name, reason in sorted(removed.items()):
        for side in ('Day', 'Night'):
            row = sides[side][name]
            journal.append({'original_path': str(root / side / name),
                            'backup_path': str(backup / 'quarantine' / side / name),
                            'sha256': row['sha256'], 'bytes': row['bytes'], **reason})
    # Write the complete recovery plan before moving any image.
    (backup / 'moves.json').write_text(json.dumps(journal, indent=2) + '\n')
    moved = []
    try:
        for item in journal:
            source, target = Path(item['original_path']), Path(item['backup_path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(item)
        manifest = ''.join(json.dumps(row) + '\n' for row in rows)
        (root / 'manifest.jsonl').write_text(manifest)
        with (root / 'pairs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = defaultdict(int)
        for item in removed.values():
            counts[item['reason']] += 1
        metadata = {
            'schema_version': 1, 'dataset': 'DarkDriving_Base',
            'prepared_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'original_pairs': len(sides['Day']), 'pairs': len(rows),
            'removed_pairs': dict(counts), 'backup_directory': str(backup),
            'source_audit_sha256': file_hash(audit),
            'manifest_sha256': file_hash(root / 'manifest.jsonl'),
            'ground_truth': 'Same-basename Day/Night pairs from the local source archive; '
                            'no independently verified geographic or pixel-level labels',
            'content_group': 'Connected component of exact image reuse on either side; '
                             'not a place ID and not protection against nearby video frames',
            'split': 'Unsplit; real Night images do not enter the existing Qwen training manifests',
        }
        (root / 'dataset_info.json').write_text(json.dumps(metadata, indent=2) + '\n')
        load_darkdriving_manifest(root)
    except BaseException:
        for item in reversed(moved):
            shutil.move(item['backup_path'], item['original_path'])
        for filename in ('manifest.jsonl', 'pairs.csv', 'dataset_info.json'):
            (root / filename).unlink(missing_ok=True)
        raise
    print(json.dumps(metadata, indent=2), flush=True)


def validate(root, workers):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    rows = load_darkdriving_manifest(root)
    metadata = json.loads((root / 'dataset_info.json').read_text())
    if metadata['pairs'] != len(rows) or file_hash(root / 'manifest.jsonl') != metadata['manifest_sha256']:
        raise ValueError('Dataset metadata or manifest checksum differs')
    with (root / 'pairs.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    if csv_rows != [{k: str(v) for k, v in row.items()} for row in rows]:
        raise ValueError('pairs.csv and manifest.jsonl differ')
    tasks = [(row, side) for row in rows for side in ('day', 'night')]

    def check(task):
        row, side = task
        path = root / row[f'{side}_path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row[f'{side}_sha256'] or len(raw) != row[f'{side}_bytes']:
            raise ValueError(f'Image bytes differ: {path}')
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if (image.format, image.mode, image.size) != (
                    'JPEG', 'RGB', (row[f'{side}_width'], row[f'{side}_height'])):
                raise ValueError(f'Unexpected image format/dimensions: {path}')
            if hashlib.sha256(image.tobytes()).hexdigest() != row[f'{side}_pixel_sha256']:
                raise ValueError(f'Decoded pixels differ: {path}')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(check, tasks), 1):
            if i % 1000 == 0 or i == len(tasks):
                print(f'Strict full decode + SHA-256: {i}/{len(tasks)}', flush=True)
    result = {'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'pairs': len(rows), 'images': len(tasks), 'errors': 0,
              'strict_full_resolution_decode': True, 'file_and_pixel_sha256': True,
              'manifest_sha256': metadata['manifest_sha256']}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/DarkDriving_Base'))
    parser.add_argument('--check-only', action='store_true', help='Read and verify all images; no cleanup')
    parser.add_argument('--audit-jsonl', type=Path, help='Previously completed strict audit files.jsonl')
    parser.add_argument('--backup-dir', type=Path, help='New backup directory outside the dataset')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.expanduser().resolve(strict=True)
    if not args.check_only:
        if not args.audit_jsonl or not args.backup_dir:
            parser.error('Cleanup requires --audit-jsonl and --backup-dir')
        prepare(root, args.audit_jsonl.expanduser().resolve(strict=True),
                args.backup_dir.expanduser().resolve(), args.workers)
    validate(root, args.workers)


if __name__ == '__main__':
    main()

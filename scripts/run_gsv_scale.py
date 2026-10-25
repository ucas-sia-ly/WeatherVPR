#!/usr/bin/env python3
"""Fixed-budget G1/G2 x A/B independent-Source exploration, development only.

Every cell starts from the original Day checkpoint. No historical A/B result is
reused: independent G1 Source sampling differs from the old same-batch Source.
Full runs require prepared Source caches before plan freeze. Smoke runs may
materialize missing Source caches, then freeze their bytes before later phases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_stage1_multiseed as p0

common = p0.common
CELLS = ('G1_A', 'G1_B', 'G2_A', 'G2_B')
COMPARISONS = (('G1_A', 'G1_B'), ('G2_A', 'G2_B'), ('G1_A', 'G2_A'), ('G1_B', 'G2_B'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh experiment directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed42', action='store_const', const=42, dest='seed')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke microbatch limit; 0 is a full epoch')
    parser.add_argument('--skip-evaluation', action='store_true', help='Smoke only')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/dino_salad.ckpt'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/night_train.jsonl'))
    parser.add_argument('--day-bank', type=Path, default=Path('runs/cache/qwen_training_day.safetensors'))
    parser.add_argument('--real-pairs-root', type=Path, default=Path('data/DarkDriving_Base'))
    parser.add_argument('--real-pairs-review', type=Path, default=common.default_review())
    parser.add_argument('--real-day-cache', type=Path, default=common.default_real_day_cache())
    parser.add_argument('--gallery-cache', type=Path, default=Path('runs/cache/svox_train_eval_day.safetensors'))
    for scale in ('g1', 'g2'):
        parser.add_argument(f'--{scale}-manifest', type=Path,
                            default=Path(f'outputs/revised_plan_20261025/data_inventory/{scale}_day.jsonl'))
        parser.add_argument(f'--{scale}-cache', type=Path,
                            default=Path(f'runs/cache/revised_{scale.upper()}_source_day.safetensors'))
    parser.add_argument('--max-gib', type=float, default=7.)
    parser.add_argument('--high-gib', type=float, default=6.)
    parser.add_argument('--reserve-gib', type=float, default=2.)
    args = parser.parse_args(argv)
    if (args.seed is None or args.seed <= 0 or args.max_steps < 0
            or not 0 < args.high_gib <= args.max_gib <= 8 or args.reserve_gib <= 0):
        parser.error('Require positive seed, max-steps>=0 and 0<high-gib<=max-gib<=8, reserve-gib>0')
    if args.skip_evaluation and not args.max_steps:
        parser.error('--skip-evaluation requires --max-steps smoke limit')
    if args.real_pairs_review is None:
        parser.error('A source-bound --real-pairs-review is required for B cells')
    for key in ('output', 'checkpoint', 'manifest', 'day_bank', 'real_pairs_root', 'real_pairs_review',
                'real_day_cache', 'gallery_cache', 'g1_manifest', 'g2_manifest', 'g1_cache', 'g2_cache'):
        setattr(args, key, common.resolve(getattr(args, key)))
    if len({args.day_bank, args.real_day_cache, args.g1_cache, args.g2_cache}) != 4:
        parser.error('Main, real and each scale must use separate fixed Day cache paths')
    return args


def source_manifest(args, cell: str) -> Path:
    return getattr(args, cell[:2].lower() + '_manifest')


def source_cache(args, cell: str) -> Path:
    return getattr(args, cell[:2].lower() + '_cache')


def arm_environment(args, cell: str) -> dict[str, str]:
    if cell not in CELLS:
        raise ValueError(f'Unknown cell: {cell}')
    scale, arm = cell.split('_')
    result = p0.arm_environment(args, args.seed, arm)
    result.update(OUTPUT=str(args.output / cell), GSV_SCALE=scale, GSV_SCALE_ARM=arm,
                  GSV_G1_MANIFEST=str(args.g1_manifest), GSV_G2_MANIFEST=str(args.g2_manifest),
                  GSV_G1_CACHE=str(args.g1_cache), GSV_G2_CACHE=str(args.g2_cache),
                  SOURCE_DAY_MANIFEST=str(source_manifest(args, cell)),
                  SOURCE_DAY_CACHE=str(source_cache(args, cell)), SOURCE_DAY_BANK_DEVICE='cuda',
                  BATCH_SAMPLING='distinct', TEMPERATURE='0.1', CONTRAST_QUEUE_SIZE='256')
    return result


def bounded_command(args, cell: str, phase: str) -> list[str]:
    prefix = [args.python, str(ROOT / 'scripts/run_bounded.py'), '--max-gib', str(args.max_gib),
              '--high-gib', str(args.high_gib), '--reserve-gib', str(args.reserve_gib),
              '--log', str(args.output / f'{cell}.{phase}.memory.jsonl'), '--']
    if phase == 'train':
        return prefix + ['bash', str(ROOT / 'configs/night_gsv_scale.sh')]
    if phase != 'eval':
        raise ValueError(f'Unknown phase: {phase}')
    return prefix + [args.python, str(ROOT / 'scripts/evaluate_svox.py'), '--split', 'train',
                     '--checkpoint', str(args.checkpoint), '--adapter-dir', str(args.output / cell),
                     '--device', 'cuda', '--retrieval-device', 'cuda', '--batch-size', '64',
                     '--num-workers', '4', '--reference-cache-mib', '1024',
                     '--gallery-cache', str(args.gallery_cache), '--save-hard-cases',
                     '--output', str(args.output / cell / 'svox_train.json')]


def manifest_info(path: Path, qwen_manifest: Path) -> tuple[dict, list[Path], set[tuple[str, str]]]:
    sidecar = Path(str(path) + '.metadata.json')
    metadata = json.loads(sidecar.read_text())
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != 'gsv_day_train'
            or metadata.get('manifest_sha256') != common.digest(path)
            or metadata.get('qwen_manifest_sha256') != common.digest(qwen_manifest)
            or metadata.get('evaluation_gallery_changed') is not False):
        raise ValueError(f'Invalid signed Source manifest metadata: {sidecar}')
    paths = [path, sidecar]
    for field in ('source_csvs', 'exclusion_files', 'exclusion_evidence'):
        for name, expected in metadata[field].items():
            source = common.resolve(name)
            if common.digest(source) != expected:
                raise ValueError(f'Source manifest provenance changed: {source}')
            paths.append(source)
    places, sample_ids = set(), set()
    images = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            place = (str(row['city']), str(int(row['place_id'])))
            if row['group_id'] != f'gsv:{place[0]}:{place[1]}' or row['sample_id'] in sample_ids:
                raise ValueError(f'Invalid or duplicate Source identity in {path}')
            if row['metadata'].get('place_label_verified') is not True:
                raise ValueError(f'Unverified Source place in {path}')
            places.add(place)
            sample_ids.add(row['sample_id'])
            images += 1
    if metadata['counts']['images'] != images or metadata['counts']['unique_places'] != len(places):
        raise ValueError(f'Source manifest count mismatch: {path}')
    info = {'manifest': str(path), 'images': images, 'known_training_places': len(places),
            'manifest_sha256': metadata['manifest_sha256']}
    return info, paths, places


def make_plan(args) -> dict:
    paths = [args.checkpoint, args.manifest, args.day_bank, args.gallery_cache, args.real_day_cache,
             args.real_pairs_review, args.real_pairs_root / 'manifest.jsonl',
             ROOT / 'data/splits/heldout_places.json', ROOT / 'configs/night_camera.json',
             ROOT / 'configs/source_night.json', ROOT / 'configs/dino_salad_stage2b_calibration.json']
    pools, pool_places = {}, {}
    for scale in ('G1', 'G2'):
        manifest = getattr(args, scale.lower() + '_manifest')
        pools[scale], provenance, pool_places[scale] = manifest_info(manifest, args.manifest)
        paths.extend(provenance)
    if not pool_places['G1'] < pool_places['G2']:
        raise ValueError('G2 must strictly expand G1 known places')
    pools['G2']['new_places_over_G1'] = len(pool_places['G2'] - pool_places['G1'])
    pools['G1']['new_places_over_G1'] = 0
    for cache in (args.day_bank, args.gallery_cache, args.real_day_cache):
        paths.extend((cache, Path(str(cache) + '.json')))
    missing_source_caches = []
    for cache in (args.g1_cache, args.g2_cache):
        sidecar = Path(str(cache) + '.json')
        if cache.is_file() and sidecar.is_file():
            paths.extend((cache, sidecar))
        elif args.max_steps and not cache.exists() and not sidecar.exists():
            missing_source_caches.append(str(cache))
        else:
            raise FileNotFoundError(f'Full runs require prepared signed Source caches; partial caches are refused: {cache}')
    queries = sorted((ROOT / 'data/svox/images/train/queries_night').glob('*.jpg'))
    if not args.skip_evaluation and not queries:
        raise FileNotFoundError('SVOX development query images are absent')
    paths.extend(queries)
    code = sorted(set((ROOT / 'src/weather_vpr').rglob('*.py')) |
                  set((ROOT / 'scripts').glob('*.py')) | set((ROOT / 'configs').glob('*.sh')) |
                  set((ROOT / 'third_party/salad/workflow').glob('*.py')) |
                  set((ROOT / 'third_party/salad/models/aggregators').glob('*.py')) |
                  set((ROOT / 'third_party/dinov2/dinov2').rglob('*.py')))
    return {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'output': str(args.output), 'seed': args.seed, 'cells': CELLS,
            'smoke_microbatch_limit': args.max_steps, 'source_pools': pools,
            'initialization': 'Each cell freshly initializes the original Day model; no historical arm reuse',
            'evaluation_split': 'SVOX train/development only',
            'budget': {'epochs': 1, 'microbatch': 4, 'accumulation': 2, 'effective_main': 8,
                       'source_per_microbatch': 1, 'real_per_B_microbatch': 1,
                       'full_optimizer_updates': 908, 'full_main_views': 7263, 'full_source_views': 1816,
                       'full_B_real_views': 1816, 'learning_rate': 1e-6,
                       'warmup_updates': 0 if args.max_steps else 32, 'precision': 'bf16'},
            'selection_policy': 'Fixed final checkpoints; one exploratory seed, no test or hyperparameter selection',
            'source_policy': 'Independent private-RNG Source stream; separate fixed Source-only Day bank; unchanged Qwen samples/targets/keys',
            'cache_materialization_policy': 'Only smoke may build initially absent Source caches; freeze tensor/sidecar hashes immediately after first training consumer and before later phases',
            'initially_absent_source_caches': missing_source_caches,
            'materialized_source_caches': {},
            'scale_interpretation': 'Scale changes available Source places and Source-only candidate keys; sampled Source-view and update counts stay fixed',
            'comparisons': [f'{after}_vs_{before}' for before, after in COMPARISONS],
            'required_files_sha256': {str(path): common.digest(path) for path in dict.fromkeys(paths)},
            'code_sha256': {str(path.relative_to(ROOT)): common.digest(path) for path in code},
            'commands': {cell: {'environment': arm_environment(args, cell),
                        'train': bounded_command(args, cell, 'train'),
                        'eval': bounded_command(args, cell, 'eval')} for cell in CELLS}}


def training_summary(args, cell: str, plan: dict) -> dict:
    directory = args.output / cell
    config = json.loads((directory / 'training_config.json').read_text())
    metadata = json.loads((directory / 'adapter.json').read_text())
    normalized = dict(config, source_day_manifest=None)
    p0.validate_protocol(normalized, args.seed, cell[-1], smoke=bool(args.max_steps))
    for key, expected in (('source_day_manifest', source_manifest(args, cell)),
                          ('source_day_cache', source_cache(args, cell)), ('fixed_day_bank', args.day_bank)):
        if common.resolve(config.get(key, '')) != expected:
            raise RuntimeError(f'{cell}: wrong {key}')
    if config.get('source_day_bank_device') != 'cuda' or config.get('day_bank_device') != 'cuda':
        raise RuntimeError(f'{cell}: expected CUDA main/Source-only banks')
    source = metadata['source_stream']
    pool = plan['source_pools'][cell[:2]]
    if (not source.get('enabled') or source['training_images'] != pool['images']
            or source['training_places'] != pool['known_training_places']
            or common.resolve(source['manifest']) != source_manifest(args, cell)):
        raise RuntimeError(f'{cell}: actual Source pool differs from frozen plan')
    streams = {name: hashlib.sha256() for name in ('main', 'source', 'real_auxiliary')}
    exposed_places, exposed_samples = set(), set()
    rows = 0
    with (directory / 'training_log.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            token = row.get('token_style')
            if token is not None and token.get('enabled') is not False:
                raise RuntimeError(f'{cell}: token stylization is forbidden')
            if row.get('source_bank_policy') != 'separate_source_only':
                raise RuntimeError(f'{cell}: main/Source banks were not separated')
            common.update_hash(streams['main'], [row['main_sample_indices'], row['augmentation']])
            common.update_hash(streams['source'], [row['source_sample_indices'], row['source_day_sample_ids'],
                               row['source_day_places'], row['source_night_augmentation']])
            common.update_hash(streams['real_auxiliary'], [row['auxiliary_sample_ids'], row['auxiliary_augmentation']])
            exposed_places.update((str(city), str(int(place))) for city, place in row['source_day_places'])
            exposed_samples.update(row['source_day_sample_ids'])
    _, _, g1_places = manifest_info(args.g1_manifest, args.manifest)
    _, _, allowed_places = manifest_info(source_manifest(args, cell), args.manifest)
    if not exposed_places.issubset(allowed_places) or source['exposed_unique_places'] != len(exposed_places):
        raise RuntimeError(f'{cell}: exposed Source place accounting differs')
    if not args.max_steps:
        expected_views = {'qwen': 7263, 'source_night': 1816,
                          'real_auxiliary': 1816 if cell.endswith('_B') else 0, 'synthetic_auxiliary': 0}
        if metadata['optimizer_steps'] != 908 or rows != 1816 or metadata['student_views'] != expected_views:
            raise RuntimeError(f'{cell}: full fixed update/view budget mismatch')
    return {'directory': str(directory), 'seed': args.seed, 'cell': cell,
            'optimizer_steps': metadata['optimizer_steps'], 'microbatches': rows,
            'student_views': metadata['student_views'], 'elapsed_seconds': metadata.get('elapsed_seconds'),
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'manifest_sha256': metadata['manifest_sha256'],
            'main_stream_sha256': metadata['main_stream_sha256'],
            'adapter_sha256': common.digest(directory / 'adapter.safetensors'),
            'stream_audit_sha256': {key: value.hexdigest() for key, value in streams.items()},
            'known_source_training_places': source['training_places'], 'source_training_images': source['training_images'],
            'exposed_source_places': len(exposed_places), 'exposed_source_samples': len(exposed_samples),
            'exposed_new_places_over_G1': len(exposed_places - g1_places),
            'source_manifest_sha256': pool['manifest_sha256'],
            'source_day_bank_signature_sha256': hashlib.sha256(
                json.dumps(source['day_bank'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'peak_cuda_allocated_bytes': metadata.get('peak_cuda_allocated_bytes'),
            'memory_guard': common.memory_summary(args.output / f'{cell}.train.memory.jsonl')}


def freeze_materialized_source_cache(args, cell: str, plan: dict) -> dict:
    cache = source_cache(args, cell)
    hashes = {str(path): common.digest(path) for path in (cache, Path(str(cache) + '.json'))}
    for path, value in hashes.items():
        expected = plan['required_files_sha256'].get(path)
        if expected is not None and expected != value:
            raise RuntimeError(f'{cell}: a frozen Source cache changed during training: {path}')
        if expected is None and (not args.max_steps or str(cache) not in plan['initially_absent_source_caches']):
            raise RuntimeError(f'{cell}: unplanned Source cache materialization: {path}')
    plan['required_files_sha256'].update(hashes)
    record = {'tensor_path': str(cache), 'sha256': hashes,
              'built_after_initial_smoke_plan': str(cache) in plan['initially_absent_source_caches']}
    plan['materialized_source_caches'][cell[:2]] = record
    common.atomic_json(args.output / 'experiment_plan.json', plan)
    return record


def summarize_records(records: dict) -> dict:
    trained = {cell: record['training'] for cell, record in records.items() if 'training' in record}
    evaluated = {cell: record['evaluation'] for cell, record in records.items() if 'evaluation' in record}
    integrity, paired = {}, {}
    if len(trained) > 1:
        for key in ('main_stream_sha256', 'checkpoint_sha256', 'manifest_sha256'):
            integrity[key + '_equal_across_cells'] = len({record[key] for record in trained.values()}) == 1
        integrity['main_log_stream_equal_across_cells'] = len({record['stream_audit_sha256']['main'] for record in trained.values()}) == 1
    for scale in ('G1', 'G2'):
        if all(f'{scale}_{arm}' in trained for arm in ('A', 'B')):
            a, b = (trained[f'{scale}_{arm}'] for arm in ('A', 'B'))
            integrity[scale + '_source_stream_equal_AB'] = a['stream_audit_sha256']['source'] == b['stream_audit_sha256']['source']
            integrity[scale + '_source_manifest_equal_AB'] = a['source_manifest_sha256'] == b['source_manifest_sha256']
    if 'G1_B' in trained and 'G2_B' in trained:
        integrity['real_auxiliary_stream_equal_across_scales'] = (
            trained['G1_B']['stream_audit_sha256']['real_auxiliary'] == trained['G2_B']['stream_audit_sha256']['real_auxiliary'])
    if integrity and not all(integrity.values()):
        raise RuntimeError(f'Experiment stream/provenance comparability failed: {integrity}')
    if evaluated:
        first = next(iter(evaluated.values()))
        for evaluation in evaluated.values():
            if any(evaluation[key] != first[key] for key in ('gallery_sha256', 'checkpoint_sha256', 'baseline')):
                raise RuntimeError('Frozen development gallery, Day baseline or checkpoint differs')
    for before, after in COMPARISONS:
        if before in evaluated and after in evaluated:
            change = common.paired_changes(evaluated[before]['night'], evaluated[after]['night'])
            for key, details in change.items():
                details['recall_delta'] = (evaluated[after]['night']['recall'][key] - evaluated[before]['night']['recall'][key])
            paired[f'{after}_vs_{before}'] = change
    return {'integrity': integrity, 'paired_changes': paired}


def write_summary(args, state: dict) -> None:
    summary = {'state': state['state'], 'seed': args.seed, 'smoke': bool(args.max_steps),
               'cells': state['cells'], **summarize_records(state['cells']),
               'limitation': 'Single exploratory seed and development only; independent Source G1 differs from historical same-batch Source. '
                             'Available places and actual exposed places are distinct. No significance or generalization claim.'}
    common.atomic_json(args.output / 'summary.json', summary)
    lines = ['# Independent GSV Source scale: G1/G2 x A/B', '',
             f'Seed {args.seed}; four fresh original-Day initializations. Fixed main/Source/real-view budgets.', '',
             '| Cell | Updates | Available Source places | Exposed Source places | Exposed new places vs G1 | Dev R@1 | R@5 | R@10 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell in CELLS:
        record = state['cells'][cell]
        train, evaluation = record.get('training', {}), record.get('evaluation', {}).get('night')
        values = [f"{evaluation['hits'][key]}/{evaluation['num_queries']} ({100*evaluation['recall'][key]:.2f}%)"
                  for key in p0.METRICS] if evaluation and not args.max_steps else ['smoke/pending'] * 3
        lines.append(f"| {cell} | {train.get('optimizer_steps','pending')} | {train.get('known_source_training_places','pending')} | "
                     f"{train.get('exposed_source_places','pending')} | {train.get('exposed_new_places_over_G1','pending')} | " + ' | '.join(values) + ' |')
    lines += ['', 'B-A within each scale and G2-G1 within each arm have per-query TopK repaired/regressed IDs in [summary.json](summary.json).', '',
              'Main and same-scale Source comparability: ' + json.dumps(summary['integrity'], sort_keys=True), '',
              summary['limitation'], '',
              'Source scaling changes its available positive samples and fixed Source-only candidate keys. Qwen samples and fixed bank remain unchanged. '
              'Stream checks compare IDs, places and augmentation operation counters, not independent augmented-image byte hashes.', '',
              '[Frozen plan](experiment_plan.json) · [Status](status.json)', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'Refusing nonempty output: {args.output}')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    common.atomic_json(args.output / 'experiment_plan.json', plan)
    state = {'state': 'planned', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'cells': {cell: {'state': 'pending', 'origin': 'fresh_original_day'} for cell in CELLS}}
    common.atomic_json(args.output / 'status.json', state)
    write_summary(args, state)
    if args.plan_only:
        print(f'Plan only: {args.output}', flush=True)
        return 0
    common.atomic_json(ROOT / 'runs/gsv_scale_training_latest.json',
                       {'run_root': str(args.output), 'runner_pid': os.getpid(), 'seed': args.seed,
                        'cells': CELLS, 'smoke_microbatch_limit': args.max_steps})
    started = time.monotonic()
    try:
        state['state'] = 'running'
        for cell in CELLS:
            entry, commands = state['cells'][cell], plan['commands'][cell]
            environment = dict(os.environ)
            environment.update(commands['environment'])
            for phase in ('train', 'eval'):
                if phase == 'eval' and args.skip_evaluation:
                    continue
                entry['state'] = phase + '_running'
                state.update(active_cell=cell, active_phase=phase)
                common.atomic_json(args.output / 'status.json', state)
                common.verify_frozen_plan(plan)
                directory = args.output / cell
                if phase == 'eval':
                    common.verify_adapter_artifact(directory, entry['training']['adapter_sha256'])
                print(f'{datetime.now().astimezone().isoformat()} Starting {cell} {phase}', flush=True)
                with (args.output / f'{cell}.{phase}.console.log').open('w') as console:
                    result = subprocess.run(commands[phase], cwd=ROOT, env=environment,
                                            stdout=console, stderr=subprocess.STDOUT)
                entry[phase + '_returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'{cell} {phase} failed ({result.returncode}); inspect console/memory logs')
                if phase == 'train':
                    entry['training'] = training_summary(args, cell, plan)
                    entry['source_cache_artifact'] = freeze_materialized_source_cache(args, cell, plan)
                else:
                    common.verify_adapter_artifact(directory, entry['training']['adapter_sha256'])
                    entry['eval_memory_guard'] = common.memory_summary(args.output / f'{cell}.eval.memory.jsonl')
                    entry['evaluation'] = p0.evaluation_summary(directory / 'svox_train.json')
                write_summary(args, state)
                print(f'{datetime.now().astimezone().isoformat()} Finished {cell} {phase}', flush=True)
            entry['state'] = 'smoke_training_complete' if args.skip_evaluation else 'complete'
            common.atomic_json(args.output / 'status.json', state)
        state['state'] = 'complete'
    except BaseException as error:
        state.update(state='failed', error=str(error))
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic() - started,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        common.atomic_json(args.output / 'status.json', state)
        write_summary(args, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())

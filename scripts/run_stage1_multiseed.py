#!/usr/bin/env python3
"""Run the fixed P0 A/B/C replication serially; evaluate SVOX development only.

Seeds 43/44 are new runs. Seed 42 is an explicitly labelled historical reference
(stage-two T0/T1 and stage-one C_fast), never counted as a new replication.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_token_style_stage2 as common

ROOT = common.ROOT
ARMS = ('A', 'B', 'C')
METRICS = ('R@1', 'R@5', 'R@10')


def default_references() -> dict[str, Path]:
    first = ROOT / 'runs/night_real_stage1_20261025_053154_seed42'
    second = ROOT / 'runs/night_token_stage2_20261025_075002_seed42'
    return {'A': second / 'T0', 'B': second / 'T1', 'C': first / 'C_fast'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh directory; nonempty paths refused')
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke microbatch limit; 0 means one epoch')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true', help='Smoke runs only')
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/dino_salad.ckpt'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/night_train.jsonl'))
    parser.add_argument('--day-bank', type=Path, default=Path('runs/cache/qwen_training_day.safetensors'))
    parser.add_argument('--real-pairs-root', type=Path, default=Path('data/DarkDriving_Base'))
    parser.add_argument('--real-pairs-review', type=Path, default=common.default_review())
    parser.add_argument('--real-day-cache', type=Path, default=common.default_real_day_cache())
    parser.add_argument('--gallery-cache', type=Path, default=Path('runs/cache/svox_train_eval_day.safetensors'))
    for arm, path in default_references().items():
        parser.add_argument(f'--seed42-{arm.lower()}', type=Path, default=path)
    parser.add_argument('--max-gib', type=float, default=7.)
    parser.add_argument('--high-gib', type=float, default=6.)
    parser.add_argument('--reserve-gib', type=float, default=2.)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s <= 0 or s == 42 for s in args.seeds):
        parser.error('New seeds must be unique positive integers excluding historical seed 42')
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms must be unique')
    if args.max_steps < 0 or not 0 < args.high_gib <= args.max_gib <= 8 or args.reserve_gib <= 0:
        parser.error('Require max-steps>=0, 0<high-gib<=max-gib<=8, reserve-gib>0')
    if args.skip_evaluation and not args.max_steps:
        parser.error('--skip-evaluation requires a smoke --max-steps limit')
    if 'B' in args.arms and args.real_pairs_review is None:
        parser.error('B requires a completed --real-pairs-review')
    for name in ('output', 'checkpoint', 'manifest', 'day_bank', 'real_pairs_root',
                 'real_pairs_review', 'real_day_cache', 'gallery_cache', 'seed42_a', 'seed42_b', 'seed42_c'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, common.resolve(value))
    return args


def job_name(seed: int, arm: str) -> str:
    return f'seed{seed}_{arm}'


def job_args(args, seed: int):
    result = copy.copy(args)
    result.run_root = args.output / f'seed{seed}'
    result.seed = seed
    return result


def arm_environment(args, seed: int, arm: str) -> dict[str, str]:
    result = common.arm_environment(job_args(args, seed), arm)
    result.pop('STAGE2_ARM')
    result.update(STAGE1_ARM=arm, COMPACT_BATCHES='1', TOKEN_STYLE_PROBABILITY='0',
                  TOKEN_STYLE_FRACTION='0.3', TOKEN_STYLE_STRENGTH='0.2',
                  TOKEN_STYLE_BETA='0.1', TOKEN_STYLE_AFTER_BLOCK='7',
                  SOURCE_DAY_MANIFEST='', SOURCE_DAY_CACHE='',
                  SOURCE_DAY_CACHE_BATCH_SIZE='64', SOURCE_DAY_CACHE_WORKERS='4',
                  SOURCE_DAY_BANK_DEVICE='cpu', CALIBRATION='configs/dino_salad_stage2b_calibration.json')
    return result


def bounded_command(args, seed: int, arm: str, phase: str) -> list[str]:
    prefix = [args.python, str(ROOT / 'scripts/run_bounded.py'), '--max-gib', str(args.max_gib),
              '--high-gib', str(args.high_gib), '--reserve-gib', str(args.reserve_gib),
              '--log', str(args.output / f'{job_name(seed, arm)}.{phase}.memory.jsonl'), '--']
    if phase == 'train':
        return prefix + ['bash', str(ROOT / 'configs/night_real_stage1_fast.sh')]
    if phase != 'eval':
        raise ValueError(f'Unknown phase: {phase}')
    directory = args.output / f'seed{seed}' / arm
    return prefix + [args.python, str(ROOT / 'scripts/evaluate_svox.py'), '--split', 'train',
                     '--checkpoint', str(args.checkpoint), '--adapter-dir', str(directory),
                     '--device', 'cuda', '--retrieval-device', 'cuda', '--batch-size', '64',
                     '--num-workers', '4', '--reference-cache-mib', '1024',
                     '--gallery-cache', str(args.gallery_cache), '--save-hard-cases',
                     '--output', str(directory / 'svox_train.json')]


def validate_protocol(config: dict, seed: int, arm: str, *, smoke: bool = False) -> None:
    expected = dict(seed=seed, batch_size=4, gradient_accumulation_steps=2, epochs=1,
                    learning_rate=1e-6, weight_decay=.01, layer_decay=1., lr_schedule='cosine',
                    warmup_steps=0 if smoke else 32, min_lr_ratio=.1, source_night_samples=1,
                    source_night_weight=.5, auxiliary_samples=1, gradient_checkpointing=False,
                    bf16=True, compact_batches=True, batch_sampling='distinct', adapter_type='full',
                    full_train_scope='late', full_train_blocks=4, global_weight=.1,
                    contrast_weight=.1, relation_weight=.1, synthetic_guidance_weight=0.,
                    contrast_queue_size=256, temperature=.1, num_workers=4, prefetch_factor=1,
                    night_augmentation='camera', augmentation_strength=1., augmentation_probability=.8,
                    augmentation_epoch_offset=0, contrast_hard_negatives=64, relation_hard_negatives=64,
                    relation_temperature=.1, relation_teacher_temperature=.1,
                    auxiliary_mode={'A': 'none', 'B': 'real', 'C': 'synthetic'}[arm],
                    auxiliary_gradient_every=0 if arm == 'A' else 100,
                    auxiliary_weight=0. if arm == 'A' else .1)
    for key in ('patch', 'keep', 'ot', 'magnitude', 'attention'):
        expected[key + '_weight'] = 0.
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f'{seed}/{arm}: protocol mismatch {key}: {config.get(key)!r} != {value!r}')
    if (config.get('init_adapter_dir') or config.get('source_day_manifest')
            or config.get('token_style_probability', 0) != 0):
        raise RuntimeError(f'{seed}/{arm}: continuation, independent Source or token style is forbidden')


def training_summary(directory: Path, seed: int, arm: str, *, memory_log: Path,
                     max_steps: int = 0) -> dict:
    config = json.loads((directory / 'training_config.json').read_text())
    metadata = json.loads((directory / 'adapter.json').read_text())
    validate_protocol(config, seed, arm, smoke=bool(max_steps))
    if not max_steps and metadata['optimizer_steps'] != 908:
        raise RuntimeError(f'{seed}/{arm}: incomplete full optimizer budget')
    streams = {name: hashlib.sha256() for name in ('main', 'source', 'auxiliary')}
    rows = 0
    with (directory / 'training_log.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            token_style = row.get('token_style')
            if token_style is not None and token_style.get('enabled') is not False:
                raise RuntimeError(f'{seed}/{arm}: unexpected token stylization')
            common.update_hash(streams['main'], [row['main_sample_indices'], row['augmentation']])
            common.update_hash(streams['source'], [row['source_sample_indices'], row['source_night_augmentation']])
            common.update_hash(streams['auxiliary'], [row['auxiliary_sample_ids'], row['auxiliary_augmentation']])
    views = metadata['student_views']
    if not max_steps:
        expected = {'qwen': 7263, 'source_night': 1816,
                    'real_auxiliary': 1816 if arm == 'B' else 0,
                    'synthetic_auxiliary': 1816 if arm == 'C' else 0}
        if rows != 1816 or views != expected:
            raise RuntimeError(f'{seed}/{arm}: full view budget differs: {views}, rows={rows}')
    return {'directory': str(directory), 'seed': seed, 'arm': arm,
            'optimizer_steps': metadata['optimizer_steps'], 'microbatches': rows,
            'student_views': views, 'elapsed_seconds': metadata.get('elapsed_seconds'),
            'checkpoint_sha256': metadata['checkpoint_sha256'],
            'manifest_sha256': metadata['manifest_sha256'],
            'adapter_sha256': common.digest(directory / 'adapter.safetensors'),
            'main_stream_sha256': metadata['main_stream_sha256'],
            'stream_audit_sha256': {key: value.hexdigest() for key, value in streams.items()},
            'peak_cuda_allocated_bytes': metadata.get('peak_cuda_allocated_bytes'),
            'memory_guard': common.memory_summary(memory_log)}


def reference_memory_log(directory: Path) -> Path:
    for suffix in ('.train.memory.jsonl', '.memory.jsonl'):
        path = directory.parent / (directory.name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f'No successful training memory log for historical reference {directory}')


def evaluation_summary(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw.get('dataset') != 'SVOX' or raw.get('split') != 'train' or raw.get('query_group') != 'queries_night':
        raise RuntimeError(f'Only SVOX train/development night evaluation is allowed: {path}')
    result = common.evaluation_summary(path)
    for key in ('baseline', 'night'):
        evaluation = result[key]
        count = evaluation['num_queries']
        for k in (1, 5, 10):
            misses = sum(rank > k for rank in evaluation['error_ranks'].values())
            if count - misses != evaluation['hits'][f'R@{k}']:
                raise RuntimeError(f'Incomplete per-query TopK errors in {path}: {key}/R@{k}')
    return result


def make_plan(args) -> dict:
    paths = [args.checkpoint, args.manifest, args.day_bank, args.gallery_cache,
             ROOT / 'data/splits/heldout_places.json', ROOT / 'configs/night_camera.json',
             ROOT / 'configs/source_night.json', ROOT / 'configs/dino_salad_stage2b_calibration.json']
    if 'B' in args.arms:
        paths += [args.real_pairs_review, args.real_pairs_root / 'manifest.jsonl', args.real_day_cache]
    for cache in (args.day_bank, args.gallery_cache, args.real_day_cache if 'B' in args.arms else None):
        if cache is not None and Path(str(cache) + '.json').is_file():
            paths.append(Path(str(cache) + '.json'))
    references = {}
    if not args.max_steps:
        for arm in args.arms:
            directory = getattr(args, f'seed42_{arm.lower()}')
            memory = reference_memory_log(directory)
            references[arm] = {
                'origin': 'historical_seed42_reference_not_new_replication',
                'training': training_summary(directory, 42, arm, memory_log=memory),
                'evaluation': evaluation_summary(directory / 'svox_train.json')}
            paths.extend(directory / name for name in ('adapter.json', 'adapter.safetensors',
                         'training_config.json', 'training_log.jsonl', 'svox_train.json'))
            paths.append(memory)
    # Freeze actual development-query bytes as well as the signed frozen Day gallery.
    queries = sorted((ROOT / 'data/svox/images/train/queries_night').glob('*.jpg'))
    if not args.skip_evaluation and not queries:
        raise FileNotFoundError('SVOX development night queries are absent')
    paths.extend(queries)
    code = sorted(set((ROOT / 'src/weather_vpr').rglob('*.py')) |
                  set((ROOT / 'scripts').glob('*.py')) | set((ROOT / 'configs').glob('*.sh')) |
                  set((ROOT / 'third_party/salad/workflow').glob('*.py')) |
                  set((ROOT / 'third_party/salad/models/aggregators').glob('*.py')) |
                  set((ROOT / 'third_party/dinov2/dinov2').rglob('*.py')))
    return {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'output': str(args.output), 'seeds_new': args.seeds, 'arms': args.arms,
            'historical_seed42': references, 'smoke_microbatch_limit': args.max_steps,
            'evaluation_split': 'SVOX train (development) only',
            'initialization': 'Original Day checkpoint independently for each arm; no Night continuation',
            'budget': {'epochs': 1, 'microbatch': 4, 'accumulation': 2, 'effective_main': 8,
                       'source_per_microbatch': 1, 'auxiliary_per_enabled_microbatch': 1,
                       'learning_rate': 1e-6, 'warmup_updates': 0 if args.max_steps else 32,
                       'full_optimizer_updates': 908, 'precision': 'bf16'},
            'selection_policy': 'Fixed final checkpoint; no new external-test evaluation or hyperparameter selection',
            'smoke_policy': 'Smoke runs exclude historical full-epoch references and aggregate accuracy claims',
            'statistics_policy': 'Recall mean and sample std across seeds; paired seed deltas; no independence/significance claim for correlated queries',
            'stream_policy': 'Main/Source equality checked within each seed; IDs and operation counters are not bytewise augmented-image hashes',
            'required_files_sha256': {str(p): common.digest(p) for p in dict.fromkeys(paths)},
            'code_sha256': {str(p.relative_to(ROOT)): common.digest(p) for p in code},
            'commands': {job_name(seed, arm): {'environment': arm_environment(args, seed, arm),
                         'train': bounded_command(args, seed, arm, 'train'),
                         'eval': bounded_command(args, seed, arm, 'eval')}
                         for seed in args.seeds for arm in args.arms}}


def mean_std(values: list[float]) -> dict:
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_std': statistics.stdev(values) if len(values) > 1 else None,
            'values': values}


def summarize_records(records: dict[str, dict]) -> dict:
    integrity, paired = {}, {}
    evaluations = [r['evaluation'] for r in records.values() if 'evaluation' in r]
    if evaluations:
        first = evaluations[0]
        for evaluation in evaluations[1:]:
            if (evaluation['gallery_sha256'] != first['gallery_sha256']
                    or evaluation['checkpoint_sha256'] != first['checkpoint_sha256']
                    or evaluation['baseline'] != first['baseline']):
                raise RuntimeError('Development gallery, query/baseline results or checkpoint differ')
    seeds = sorted({int(name.split('_')[0][4:]) for name in records})
    for seed in seeds:
        trained = [records[job_name(seed, arm)]['training'] for arm in ARMS
                   if 'training' in records.get(job_name(seed, arm), {})]
        if len(trained) < 2:
            continue
        integrity[str(seed)] = {key + '_stream_equal': len({t['stream_audit_sha256'][key] for t in trained}) <= 1
                                for key in ('main', 'source')}
        integrity[str(seed)]['checkpoint_equal'] = len({t['checkpoint_sha256'] for t in trained}) <= 1
        integrity[str(seed)]['manifest_equal'] = len({t['manifest_sha256'] for t in trained}) <= 1
        for before, after in (('A', 'B'), ('A', 'C'), ('C', 'B')):
            a = records.get(job_name(seed, before), {}).get('evaluation')
            b = records.get(job_name(seed, after), {}).get('evaluation')
            if a is not None and b is not None:
                paired[f'seed{seed}_{after}_vs_{before}'] = common.paired_changes(a['night'], b['night'])
    if any(not all(checks.values()) for checks in integrity.values()):
        raise RuntimeError(f'Main/Source/provenance comparability failed: {integrity}')
    aggregates, deltas = {}, {}
    for arm in ARMS:
        eligible = [seed for seed in seeds if 'evaluation' in records.get(job_name(seed, arm), {})]
        aggregates[arm] = {'seeds': eligible, 'recall': {
            key: mean_std([records[job_name(seed, arm)]['evaluation']['night']['recall'][key]
                           for seed in eligible]) for key in METRICS}}
    for before, after in (('A', 'B'), ('A', 'C'), ('C', 'B')):
        eligible = [seed for seed in seeds if all('evaluation' in records.get(job_name(seed, arm), {})
                                                 for arm in (before, after))]
        deltas[f'{after}_minus_{before}'] = {'seeds': eligible, 'recall_delta': {
            key: mean_std([records[job_name(seed, after)]['evaluation']['night']['recall'][key] -
                           records[job_name(seed, before)]['evaluation']['night']['recall'][key]
                           for seed in eligible]) for key in METRICS}}
    return {'integrity': integrity, 'paired_changes': paired, 'aggregate': aggregates,
            'paired_seed_deltas': deltas}


def write_summary(args, state: dict) -> None:
    records = state['jobs']
    data = summarize_records(records)
    if args.max_steps:
        # Smoke outputs remain useful for correctness, never multi-seed accuracy evidence.
        data['aggregate'] = {}
        data['paired_seed_deltas'] = {}
    summary = {'state': state['state'], 'smoke': bool(args.max_steps), 'records': records, **data,
               'limitation': 'Development evidence only. Three seeds do not establish domain generalization; '
                             'queries are geographically correlated. Historical seed42 is reused, not rerun.'}
    common.atomic_json(args.output / 'summary.json', summary)
    lines = ['# P0 A/B/C multi-seed replication', '',
             'A: no extra cosine. B: real DarkDriving paired cosine. C: equally weighted extra Qwen paired cosine.', '',
             'Seed 42 reuses historical T0/T1/C_fast; new seeds start independently from the original Day model.', '',
             '| Seed | Arm | Origin | Updates | Dev R@1 | R@5 | R@10 |',
             '|---:|---|---|---:|---:|---:|---:|']
    for name, record in records.items():
        train, evaluation = record.get('training', {}), record.get('evaluation', {}).get('night')
        cells = [f"{evaluation['hits'][key]}/{evaluation['num_queries']} ({100*evaluation['recall'][key]:.2f}%)"
                 for key in METRICS] if evaluation else ['pending'] * 3
        seed, arm = name.split('_')
        lines.append(f"| {seed[4:]} | {arm} | {record['origin']} | {train.get('optimizer_steps', 'pending')} | " + ' | '.join(cells) + ' |')
    if not args.max_steps:
        lines += ['', '| Arm | Evaluated seeds | R@1 mean ± sample std | R@5 | R@10 |',
                  '|---|---|---:|---:|---:|']
        for arm in ARMS:
            aggregate = data['aggregate'][arm]
            cells = []
            for key in METRICS:
                m = aggregate['recall'][key]
                cells.append('pending' if m['mean'] is None else
                             f"{100*m['mean']:.2f}% ± " + ('n/a' if m['sample_std'] is None else f"{100*m['sample_std']:.2f} pp"))
            lines.append(f"| {arm} | {aggregate['seeds']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Per-query TopK repairs/regressions and paired seed deltas are in [summary.json](summary.json).', '',
              'This queue evaluates development only. It does not select a test winner, establish route independence, '
              'or interpret correlated queries as independent trials. Stream audits compare IDs and operation counters.', '',
              '[Frozen plan](experiment_plan.json) · [Status](status.json)', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'Refusing nonempty output directory: {args.output}')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    common.atomic_json(args.output / 'experiment_plan.json', plan)
    jobs = {job_name(42, arm): {**record, 'state': 'historical_reference_complete'}
            for arm, record in plan['historical_seed42'].items()}
    jobs.update({job_name(seed, arm): {'origin': 'new_replication', 'state': 'pending'}
                 for seed in args.seeds for arm in args.arms})
    state = {'state': 'planned', 'jobs': jobs, 'started_utc': datetime.now(timezone.utc).isoformat()}
    common.atomic_json(args.output / 'status.json', state)
    write_summary(args, state)
    if args.plan_only:
        print(f'Plan only: {args.output}', flush=True)
        return 0
    common.atomic_json(ROOT / 'runs/stage1_multiseed_latest.json',
                       {'run_root': str(args.output), 'runner_pid': os.getpid(), 'new_seeds': args.seeds,
                        'arms': args.arms, 'smoke_microbatch_limit': args.max_steps})
    started = time.monotonic()
    try:
        state['state'] = 'running'
        for seed in args.seeds:
            for arm in args.arms:
                name = job_name(seed, arm)
                entry, commands = jobs[name], plan['commands'][name]
                directory = args.output / f'seed{seed}' / arm
                environment = dict(os.environ)
                environment.update(commands['environment'])
                for phase in ('train', 'eval'):
                    if phase == 'eval' and args.skip_evaluation:
                        continue
                    state.update(active_job=name, active_phase=phase)
                    entry['state'] = phase + '_running'
                    common.atomic_json(args.output / 'status.json', state)
                    common.verify_frozen_plan(plan)
                    if phase == 'eval':
                        common.verify_adapter_artifact(directory, entry['training']['adapter_sha256'])
                    print(f'{datetime.now().astimezone().isoformat()} Starting {name} {phase}', flush=True)
                    with (args.output / f'{name}.{phase}.console.log').open('w') as console:
                        result = subprocess.run(commands[phase], cwd=ROOT, env=environment,
                                                stdout=console, stderr=subprocess.STDOUT)
                    entry[phase + '_returncode'] = result.returncode
                    if result.returncode:
                        raise RuntimeError(f'{name} {phase} failed ({result.returncode}); inspect console/memory logs')
                    memory = args.output / f'{name}.{phase}.memory.jsonl'
                    if phase == 'train':
                        entry['training'] = training_summary(directory, seed, arm, memory_log=memory,
                                                             max_steps=args.max_steps)
                    else:
                        common.verify_adapter_artifact(directory, entry['training']['adapter_sha256'])
                        entry['eval_memory_guard'] = common.memory_summary(memory)
                        entry['evaluation'] = evaluation_summary(directory / 'svox_train.json')
                    write_summary(args, state)
                    print(f'{datetime.now().astimezone().isoformat()} Finished {name} {phase}', flush=True)
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

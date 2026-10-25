#!/usr/bin/env python3
"""Evaluate all frozen revised-plan adapters on four existing night protocols.

This is an exploratory external comparison, never a training or model-selection
queue. SF-XL stays reserved. Every result is freshly computed from bound weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
METRICS = ('R@1', 'R@5', 'R@10')
BENCHMARKS = ('svox_test', 'tokyo247_night', 'robotcar_night', 'robotcar_night_rain')
EXPECTED = {
    'svox_test': (823, (763, 810, 816)),
    'tokyo247_night': (105, (91, 95, 98)),
    'robotcar_night': (197, (179, 187, 190)),
    'robotcar_night_rain': (224, (214, 216, 218)),
}
CACHE_NAMES = {'svox_test': 'svox_test_gallery', 'tokyo247_night': 'tokyo247_day_gallery',
               'robotcar_night': 'robotcar-public_day_gallery',
               'robotcar_night_rain': 'robotcar-public_day_gallery'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh output directory')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--models', nargs='+', help='Optional exact model IDs from the two completed summaries')
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/dino_salad.ckpt'))
    parser.add_argument('--stage1-summary', type=Path,
                        default=Path('runs/revised_stage1_multiseed_20261025/summary.json'))
    parser.add_argument('--scale-summary', type=Path,
                        default=Path('runs/revised_gsv_scale_20261025/summary.json'))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    for key in ('output', 'checkpoint', 'stage1_summary', 'scale_summary'):
        setattr(args, key, common.resolve(getattr(args, key)))
    for key in ('models', 'benchmarks'):
        values = getattr(args, key)
        if values is not None and len(values) != len(set(values)):
            parser.error(f'--{key} must contain unique IDs')
    return args


def load_models(args):
    """Validate successful full-budget artifacts against their training records."""
    models = {}
    checkpoint_hash = common.digest(args.checkpoint)
    for summary_path, container, family in ((args.stage1_summary, 'records', 'P0'),
                                            (args.scale_summary, 'cells', 'Source_scale')):
        summary = json.loads(summary_path.read_text())
        if summary.get('state') != 'complete' or summary.get('smoke') is not False:
            raise ValueError(f'Require a completed non-smoke source experiment: {summary_path}')
        expected_names = ({f'seed{seed}_{arm}' for seed in (42, 43, 44) for arm in 'ABC'}
                          if family == 'P0' else {'G1_A', 'G1_B', 'G2_A', 'G2_B'})
        if set(summary[container]) != expected_names:
            raise ValueError(f'Unexpected completed model set: {summary_path}')
        for name, record in summary[container].items():
            training = record['training']
            directory = common.resolve(training['directory'])
            metadata_path = directory / 'adapter.json'
            weights_path = directory / 'adapter.safetensors'
            metadata = json.loads(metadata_path.read_text())
            if (training['optimizer_steps'] != 908 or training['microbatches'] != 1816
                    or metadata['optimizer_steps'] != 908
                    or metadata['student_views'] != training['student_views']):
                raise ValueError(f'{name}: incomplete/mismatched full training budget')
            if (training['checkpoint_sha256'] != checkpoint_hash
                    or metadata['checkpoint_sha256'] != checkpoint_hash
                    or metadata.get('init_adapter_dir') is not None
                    or metadata.get('weather') != 'night'):
                raise ValueError(f'{name}: incompatible checkpoint/initialization/weather')
            actual = common.digest(weights_path)
            if actual != training['adapter_sha256']:
                raise ValueError(f'{name}: weights differ from completed training summary')
            models[name] = {
                'family': family, 'seed': training['seed'],
                'arm': training.get('arm', name.rsplit('_', 1)[-1]),
                'directory': str(directory), 'adapter_sha256': actual,
                'metadata_sha256': common.digest(metadata_path),
                'source_summary': str(summary_path),
                'origin': record.get('origin', 'fresh_original_day'),
                'optimizer_steps': 908, 'microbatches': 1816,
                'checkpoint_sha256': checkpoint_hash,
            }
    names = args.models or list(models)
    missing = set(names) - set(models)
    if missing:
        raise ValueError(f'Unknown model IDs: {sorted(missing)}')
    return {name: models[name] for name in names}


def benchmark_inputs(name):
    """Bind native truth and ordered query identities without importing a model."""
    cache = ROOT / 'runs/cache' / (CACHE_NAMES[name] + '.safetensors')
    truth = []
    if name == 'svox_test':
        root = ROOT / 'data/svox'
        paths = sorted((root / 'images/test/queries_night').glob('*.jpg'))
        query_ids = [path.relative_to(root).as_posix() for path in paths]
        description = 'SVOX native test night, 25 m UTM positives encoded in original filenames'
    elif name == 'tokyo247_night':
        root = ROOT / 'data/tokyo247'
        source = ROOT / 'configs/evaluation_sources/tokyo247_netvlad_v100.source.json'
        truth = [source, ROOT / 'configs/evaluation_sources/tokyo247_netvlad_v100.mat']
        # Bind all CSV labels, including daytime labels, so night membership cannot drift.
        truth += sorted((root / 'queries').glob('*.csv'))
        query_ids = list(json.loads(source.read_text())['night_ids'])
        paths = [root / 'queries' / query_id for query_id in query_ids]
        description = 'Tokyo247 official NetVLAD MAT + CSV night labels; local v3 query release caveat retained'
    else:
        root = ROOT / 'data/RobotCar-Seasons'
        condition = 'night-rain' if name == 'robotcar_night_rain' else 'night'
        metadata = [path for directory in (root / 'metadata', root)
                    for path in directory.glob('robotcar_v2_train.txt*')
                    if path.name == 'robotcar_v2_train.txt' or path.name.startswith('robotcar_v2_train.txt?')]
        if len(metadata) != 1:
            raise ValueError('Require exactly one public RobotCar train pose file')
        truth = metadata + sorted((root / '3D-models/individual/colmap_reconstructions').rglob('images.txt'))
        if len(truth) != 50:
            raise ValueError('Require RobotCar public poses plus all 49 aligned COLMAP pose models')
        query_ids = sorted(Path(line.split()[0]).with_suffix('.jpg').as_posix()
                           for line in metadata[0].read_text().splitlines()
                           if line.strip() and not line.startswith('#')
                           and line.split()[0].startswith(condition + '/rear/'))
        paths = [root / 'images' / query_id for query_id in query_ids]
        description = ('RobotCar public-train rear-camera 25 m 3D position diagnostic; '
                       'complete aligned three-camera overcast gallery; not official test')
    if len(query_ids) != EXPECTED[name][0] or len(query_ids) != len(set(query_ids)):
        raise ValueError(f'{name}: native query count/identities differ from declared protocol')
    sidecar = Path(str(cache) + '.json')
    cache_meta = json.loads(sidecar.read_text())
    return {'name': name, 'root': str(root), 'description': description,
            'cache': str(cache), 'gallery_sha256': cache_meta['gallery_sha256'],
            'query_ids': query_ids, 'query_paths': [str(path) for path in paths],
            'files': [str(path) for path in [cache, sidecar, *truth, *paths]],
            'expected_num_queries': EXPECTED[name][0],
            'historical_baseline_hits_for_audit_only': dict(zip(METRICS, EXPECTED[name][1]))}


def job_name(model, benchmark):
    return f'{model}.{benchmark}'


def bounded_command(args, model, benchmark, models, benchmarks):
    name = job_name(model, benchmark)
    prefix = [args.python, str(ROOT / 'scripts/run_bounded.py'), '--max-gib', '7', '--high-gib', '6',
              '--reserve-gib', '2', '--log', str(args.output / f'{name}.memory.jsonl'), '--', args.python]
    if benchmark == 'svox_test':
        command = [str(ROOT / 'scripts/evaluate_svox.py'), '--split', 'test', '--save-hard-cases',
                   '--svox-root', benchmarks[benchmark]['root']]
    else:
        command = [str(ROOT / 'scripts/evaluate_night_benchmarks.py')]
        if benchmark == 'tokyo247_night':
            command += ['--dataset', 'tokyo247']
        else:
            command += ['--dataset', 'robotcar-public', '--condition',
                        'night-rain' if benchmark == 'robotcar_night_rain' else 'night']
        command += ['--root', benchmarks[benchmark]['root']]
    return prefix + command + [
        '--checkpoint', str(args.checkpoint), '--adapter-dir', models[model]['directory'],
        '--device', 'cuda', '--retrieval-device', 'cuda', '--batch-size', '64', '--num-workers', '4',
        '--query-chunk-size', '32', '--reference-chunk-size', '4096', '--reference-cache-mib', '4096',
        '--gallery-cache', benchmarks[benchmark]['cache'], '--output', str(args.output / (name + '.json'))]


def make_plan(args):
    models = load_models(args)
    benchmarks = {name: benchmark_inputs(name) for name in args.benchmarks}
    shared = [str(args.checkpoint), str(args.stage1_summary), str(args.scale_summary)]
    preflight_root = ROOT / 'runs/revised_external_night_preflight_20261025'
    preflight_names = {'svox_test': 'svox_test', 'tokyo247_night': 'tokyo247',
                       'robotcar_night': 'robotcar_night', 'robotcar_night_rain': 'robotcar_night_rain'}
    for benchmark_name, benchmark in benchmarks.items():
        preflight = preflight_root / (preflight_names[benchmark_name] + '.json')
        if preflight.is_file():
            benchmark['files'].append(str(preflight))
            benchmark['bound_preflight'] = str(preflight)
    model_files = {name: [str(Path(model['directory']) / filename)
                          for filename in ('adapter.json', 'adapter.safetensors', 'training_config.json')]
                   for name, model in models.items()}
    files = set(shared)
    files.update(path for paths in model_files.values() for path in paths)
    files.update(path for benchmark in benchmarks.values() for path in benchmark['files'])
    code = sorted(set((ROOT / 'src/weather_vpr').rglob('*.py')) |
                  set((ROOT / 'scripts').glob('*.py')) | set((ROOT / 'configs').glob('*.sh')) |
                  set((ROOT / 'third_party/salad/workflow').glob('*.py')) |
                  set((ROOT / 'third_party/salad/models/aggregators').glob('*.py')) |
                  set((ROOT / 'third_party/dinov2/dinov2').rglob('*.py')))
    hashes = {path: common.digest(Path(path)) for path in sorted(files)}
    checkpoint_hash = hashes[str(args.checkpoint)]
    for name, benchmark in benchmarks.items():
        cache_metadata = json.loads(Path(benchmark['cache'] + '.json').read_text())
        if cache_metadata['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError(f'{name}: gallery cache uses a different checkpoint')
    return {
        'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'intent': 'Exploratory external comparison of every completed final model; no tuning or winner selection',
        'output': str(args.output), 'models': models, 'benchmarks': benchmarks,
        'protocol': {'image_size': [224, 224], 'precision': 'fp32', 'query_fusion': None,
                     'test_time_augmentation': False, 'gallery': 'original frozen Day checkpoint',
                     'batch_size': 64, 'num_workers': 4, 'reference_cache_mib': 4096,
                     'training': False, 'expected_default_jobs': 52},
        'reserved': 'SF-XL remains untouched; Aachen/official RobotCar tests lack local public retrieval ground truth',
        'limitations': 'SVOX/Tokyo/RobotCar protocols have been inspected historically; these are not newly blind tests. '
                       'RobotCar uses public train poses as evaluation only, not the official benchmark. '
                       'Tokyo local v3 query images have not been independently byte-verified against v2.',
        'shared_files': shared, 'model_files': model_files, 'required_files_sha256': hashes,
        'code_sha256': {str(path.relative_to(ROOT)): common.digest(path) for path in code},
        'commands': {job_name(model, benchmark): bounded_command(args, model, benchmark, models, benchmarks)
                     for benchmark in benchmarks for model in models},
    }


def verify_plan(plan, model, benchmark):
    """Rehash this phase's inputs; unrelated models are checked before their phases."""
    paths = set(plan['shared_files'] + plan['model_files'][model] + plan['benchmarks'][benchmark]['files'])
    for path in paths:
        if common.digest(Path(path)) != plan['required_files_sha256'][path]:
            raise RuntimeError(f'Frozen evaluation input changed: {path}')
    for path, expected in plan['code_sha256'].items():
        if common.digest(ROOT / path) != expected:
            raise RuntimeError(f'Frozen evaluation code changed: {path}')


def verify_all_inputs(plan):
    for path, expected in plan['required_files_sha256'].items():
        if common.digest(Path(path)) != expected:
            raise RuntimeError(f'Frozen evaluation input changed: {path}')
    for path, expected in plan['code_sha256'].items():
        if common.digest(ROOT / path) != expected:
            raise RuntimeError(f'Frozen evaluation code changed: {path}')


def compact_evaluation(evaluation, expected_ids):
    count = evaluation['num_evaluated_queries']
    if count != len(expected_ids) or evaluation.get('num_queries_without_positives', 0):
        raise RuntimeError('Every predeclared query must have public positives and be evaluated')
    errors = evaluation['error_queries']
    ranks = {row['query_id']: row['rank'] for row in errors}
    if (not evaluation.get('hard_cases_saved') or len(ranks) != len(errors)
            or set(ranks) - set(expected_ids) or any(rank <= 1 for rank in ranks.values())
            or len(ranks) != evaluation['num_error_queries']):
        raise RuntimeError('Incomplete/invalid per-query errors')
    hits = {metric: round(evaluation['recall'][metric] * count) for metric in METRICS}
    for k in (1, 5, 10):
        metric = f'R@{k}'
        if count - sum(rank > k for rank in ranks.values()) != hits[metric]:
            raise RuntimeError(f'Per-query errors do not reproduce {metric}')
    return {'num_queries': count, 'recall': evaluation['recall'], 'hits': hits, 'error_ranks': ranks}


def read_evaluation(path, plan, model_name, benchmark_name):
    raw = json.loads(path.read_text())
    model = plan['models'][model_name]
    benchmark = plan['benchmarks'][benchmark_name]
    if (raw.get('checkpoint_sha256') != model['checkpoint_sha256']
            or raw.get('gallery_sha256') != benchmark['gallery_sha256']
            or raw.get('image_size') != [224, 224]
            or raw.get('query_preprocessing', {}).get('mode') != 'native_single_scale'
            or 'fusion' in raw or raw.get('query_preprocessing', {}).get('test_time_augmentation', False)):
        raise RuntimeError('Evaluation checkpoint/gallery/preprocessing differs from frozen protocol')
    adapter = raw['adapter']
    if Path(adapter['directory']).resolve() != Path(model['directory']).resolve():
        raise RuntimeError('Evaluation used a different adapter directory')
    if adapter.get('scale', 1) != 1:
        raise RuntimeError('Adapter scaling is forbidden')
    for field, expected in (('weights_sha256', model['adapter_sha256']),
                            ('metadata_sha256', model['metadata_sha256'])):
        if field in adapter and adapter[field] != expected:
            raise RuntimeError(f'Bound adapter {field} differs')
    baseline = compact_evaluation(raw['baseline'], benchmark['query_ids'])
    night = compact_evaluation(adapter['evaluation'], benchmark['query_ids'])
    if raw['baseline'].get('protocol') != adapter['evaluation'].get('protocol'):
        raise RuntimeError('Baseline and adapter truth/protocol differ')
    if benchmark_name == 'svox_test':
        if raw.get('split') != 'test' or raw.get('query_group') != 'queries_night':
            raise RuntimeError('SVOX test night protocol required')
    else:
        rows = raw.get('per_query', [])
        if len(rows) != len(benchmark['query_ids']) or {row['query_id'] for row in rows} != set(benchmark['query_ids']):
            raise RuntimeError('External per-query identities differ from frozen query set')
        for label, compact in (('baseline', baseline), ('night', night)):
            ranks = {row['query_id']: row[label]['rank'] for row in rows}
            if {key: rank for key, rank in ranks.items() if rank > 1} != compact['error_ranks']:
                raise RuntimeError('Per-query ranks and hard cases disagree')
    return {'path': str(path), 'result_sha256': common.digest(path),
            'checkpoint_sha256': raw['checkpoint_sha256'], 'gallery_sha256': raw['gallery_sha256'],
            'query_sha256': raw.get('query_sha256'), 'preprocessing': raw['preprocessing'],
            'query_preprocessing': raw['query_preprocessing'], 'protocol': raw['baseline']['protocol'],
            'baseline': baseline, 'night': night,
            'historical_baseline_matches': baseline['hits'] == benchmark['historical_baseline_hits_for_audit_only']}


def aggregate(plan, state):
    evaluated = {name: entry['evaluation'] for name, entry in state['jobs'].items() if 'evaluation' in entry}
    datasets = {}
    for benchmark_name, benchmark in plan['benchmarks'].items():
        entries = {model: evaluated[job_name(model, benchmark_name)] for model in plan['models']
                   if job_name(model, benchmark_name) in evaluated}
        if not entries:
            continue
        reference = next(iter(entries.values()))
        for evaluation in entries.values():
            for key in ('checkpoint_sha256', 'gallery_sha256', 'query_sha256', 'preprocessing',
                        'query_preprocessing', 'protocol', 'baseline'):
                if evaluation[key] != reference[key]:
                    raise RuntimeError(f'{benchmark_name}: models differ in {key}')
        comparisons = {}
        pairs = [(f'seed{seed}_{before}', f'seed{seed}_{after}')
                 for seed in (42, 43, 44) for before, after in (('A', 'B'), ('A', 'C'), ('C', 'B'))]
        pairs += [('G1_A', 'G1_B'), ('G2_A', 'G2_B'), ('G1_A', 'G2_A'), ('G1_B', 'G2_B')]
        for before, after in pairs:
            if before in entries and after in entries:
                comparisons[f'{after}_vs_{before}'] = common.paired_changes(entries[before]['night'], entries[after]['night'])
        means = {}
        for arm in 'ABC':
            names = [f'seed{seed}_{arm}' for seed in (42, 43, 44)]
            if all(name in entries for name in names):
                means[arm] = {'seeds': [42, 43, 44], 'n': 3, 'metrics': {}}
                for metric in METRICS:
                    values = [entries[name]['night']['recall'][metric] for name in names]
                    means[arm]['metrics'][metric] = {'mean': statistics.mean(values),
                                                    'sample_std': statistics.stdev(values),
                                                    'hits': [entries[name]['night']['hits'][metric] for name in names]}
        datasets[benchmark_name] = {'num_queries': benchmark['expected_num_queries'],
                                   'baseline': reference['baseline'], 'models': entries,
                                   'paired_changes': comparisons, 'P0_three_seed': means,
                                   'historical_baseline_matches': reference['historical_baseline_matches']}
    return {'state': state['state'], 'intent': plan['intent'], 'completed_jobs': len(evaluated),
            'planned_jobs': len(state['jobs']), 'benchmarks': datasets, 'limitations': plan['limitations'],
            'reserved': plan['reserved'],
            'baseline_audit_warnings': [name for name, value in datasets.items() if not value['historical_baseline_matches']]}


def write_summary(args, plan, state):
    summary = aggregate(plan, state)
    common.atomic_json(args.output / 'summary.json', summary)
    lines = ['# Frozen external night evaluation', '', plan['intent'] + '.', '',
             f"State: {state['state']}; {summary['completed_jobs']}/{summary['planned_jobs']} fresh evaluations.", '',
             '| Model | ' + ' | '.join(plan['benchmarks']) + ' |',
             '|---|' + '|'.join('---:' for _ in plan['benchmarks']) + '|']
    for model in ['Original Day', *plan['models']]:
        cells = []
        for name in plan['benchmarks']:
            dataset = summary['benchmarks'].get(name)
            evaluation = (dataset['baseline'] if model == 'Original Day' else
                          dataset['models'].get(model, {}).get('night')) if dataset else None
            cells.append('/'.join(str(evaluation['hits'][metric]) for metric in METRICS) +
                         f" / {evaluation['num_queries']}" if evaluation else 'pending')
        lines.append('| ' + model + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Cells show correct queries at R@1/R@5/R@10, followed by query count.', '',
              'Three-seed means/sample standard deviations and paired repairs/regressions at every TopK are in summary.json.', '',
              plan['limitations'], '', plan['reserved'], '',
              'No training, fusion, TTA, external-score hyperparameter search, or historical result substitution.', '']
    if summary['baseline_audit_warnings']:
        lines += ['Historical baseline audit mismatch: ' + ', '.join(summary['baseline_audit_warnings']), '']
    (args.output / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'Refusing nonempty output directory: {args.output}')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    common.atomic_json(args.output / 'experiment_plan.json', plan)
    state = {'state': 'planned', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'jobs': {name: {'state': 'pending'} for name in plan['commands']}}
    common.atomic_json(args.output / 'status.json', state)
    write_summary(args, plan, state)
    if args.plan_only:
        print(f'Plan only: {args.output}', flush=True)
        return 0
    common.atomic_json(ROOT / 'runs/revised_night_evaluation_latest.json',
                       {'output': str(args.output), 'runner_pid': os.getpid(), 'models': list(plan['models']),
                        'benchmarks': list(plan['benchmarks']), 'planned_jobs': len(state['jobs'])})
    started = time.monotonic()
    environment = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    try:
        state['state'] = 'running'
        for benchmark in plan['benchmarks']:
            for model in plan['models']:
                name = job_name(model, benchmark)
                entry = state['jobs'][name]
                state['active_job'] = name
                entry['state'] = 'running'
                common.atomic_json(args.output / 'status.json', state)
                print(f'{datetime.now().astimezone().isoformat()} Starting {name}', flush=True)
                verify_plan(plan, model, benchmark)
                with (args.output / (name + '.console.log')).open('w') as console:
                    result = subprocess.run(plan['commands'][name], cwd=ROOT, env=environment,
                                            stdout=console, stderr=subprocess.STDOUT)
                entry['returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'{name} failed with exit {result.returncode}; inspect console/memory logs')
                verify_plan(plan, model, benchmark)
                entry['memory_guard'] = common.memory_summary(args.output / (name + '.memory.jsonl'))
                entry['evaluation'] = read_evaluation(args.output / (name + '.json'), plan, model, benchmark)
                entry['state'] = 'complete'
                common.atomic_json(args.output / 'status.json', state)
                write_summary(args, plan, state)
                print(f'{datetime.now().astimezone().isoformat()} Finished {name}', flush=True)
        verify_all_inputs(plan)
        state['state'] = 'complete'
    except BaseException as error:
        state['state'] = 'failed'
        state['error'] = str(error)
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds'] = time.monotonic() - started
        common.atomic_json(args.output / 'status.json', state)
        write_summary(args, plan, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())

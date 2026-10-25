#!/usr/bin/env python3
"""Run the fixed T0/T1/T2/T3 experiment serially with bounded RAM and dev evaluation.

Every arm starts at the original Day checkpoint. This queue never selects an
external-test winner or searches hyperparameters. --max-steps is for smoke runs.
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
ARMS = ('T0', 'T1', 'T2', 'T3')
STYLE = dict(after_block=7, fraction=.3, strength=.2, beta=.1)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def resolve(path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def default_review() -> Path | None:
    latest = ROOT / 'runs/stage1_training_latest.json'
    if latest.exists():
        return Path(json.loads(latest.read_text())['run_root']) / 'real_review.jsonl'
    return None


def default_real_day_cache() -> Path:
    review = default_review()
    if review is not None and (review.parent / 'real_day.safetensors').is_file():
        return review.parent / 'real_day.safetensors'
    return ROOT / 'runs/cache/stage1_real_day.safetensors'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, help='Fresh directory; nonempty paths are refused')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=0, help='Microbatch limit, 0 is one complete epoch')
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/dino_salad.ckpt'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/night_train.jsonl'))
    parser.add_argument('--real-pairs-root', type=Path, default=Path('data/DarkDriving_Base'))
    parser.add_argument('--real-pairs-review', type=Path, default=default_review())
    parser.add_argument('--real-day-cache', type=Path, default=default_real_day_cache())
    parser.add_argument('--day-bank', type=Path, default=Path('runs/cache/qwen_training_day.safetensors'))
    parser.add_argument('--gallery-cache', type=Path, default=Path('runs/cache/svox_train_eval_day.safetensors'))
    parser.add_argument('--training-diagnostic', type=Path,
                        help='Completed train-only Token diagnostic to bind before any development evaluation')
    parser.add_argument('--max-gib', type=float, default=7.)
    parser.add_argument('--high-gib', type=float, default=6.)
    parser.add_argument('--reserve-gib', type=float, default=2.)
    parser.add_argument('--plan-only', action='store_true', help='Freeze plan and commands without executing them')
    parser.add_argument('--skip-evaluation', action='store_true', help='Smoke runs only; no Recall claim')
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms must be unique')
    if args.max_steps < 0 or not 0 < args.high_gib <= args.max_gib <= 8 or args.reserve_gib <= 0:
        parser.error('Require max-steps>=0, 0<high-gib<=max-gib<=8, reserve-gib>0')
    if args.skip_evaluation and not args.max_steps and not args.plan_only:
        parser.error('--skip-evaluation is restricted to smoke runs (--max-steps)')
    if any(arm in ('T1', 'T3') for arm in args.arms) and args.real_pairs_review is None:
        parser.error('--real-pairs-review is required when stage-one latest metadata is absent')
    if args.run_root is None:
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        args.run_root = ROOT / f'runs/night_token_stage2_{stamp}_seed{args.seed}'
    for key in ('run_root', 'checkpoint', 'manifest', 'real_pairs_root', 'real_day_cache',
                'day_bank', 'gallery_cache', 'real_pairs_review', 'training_diagnostic'):
        if getattr(args, key) is not None:
            setattr(args, key, resolve(getattr(args, key)))
    return args


def arm_environment(args, arm: str) -> dict[str, str]:
    """Set all protocol knobs explicitly so a parent shell cannot change the run."""
    result = dict(
        STAGE2_ARM=arm, PYTHON=args.python, CKPT=str(args.checkpoint),
        MANIFEST=str(args.manifest), DAY_BANK=str(args.day_bank),
        REAL_PAIRS_ROOT=str(args.real_pairs_root),
        REAL_PAIRS_REVIEW=str(args.real_pairs_review or ''),
        AUXILIARY_DAY_CACHE=str(args.real_day_cache),
        OUTPUT=str(args.run_root / arm), SEED=str(args.seed),
        BATCH_SIZE='4', GRADIENT_ACCUMULATION_STEPS='2', SOURCE_NIGHT_SAMPLES='1',
        AUXILIARY_SAMPLES='1', EPOCHS='1', MAX_STEPS=str(args.max_steps),
        BF16='1', LEARNING_RATE='0.000001', WEIGHT_DECAY='0.01', LAYER_DECAY='1',
        LR_SCHEDULE='cosine', WARMUP_STEPS='0' if args.max_steps else '32', MIN_LR_RATIO='0.1',
        SAVE_EVERY_STEPS='0', VERIFY_HASHES='1', DEVICE='cuda',
        DAY_BANK_DEVICE='cuda', GRADIENT_CHECKPOINTING='0', NUM_WORKERS='4',
        PREFETCH_FACTOR='1', AUXILIARY_CACHE_BATCH_SIZE='64', AUXILIARY_CACHE_WORKERS='4',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', AUXILIARY_WEIGHT='0.1',
        AUXILIARY_GRADIENT_EVERY='100', SOURCE_NIGHT_WEIGHT='0.5',
        GLOBAL_WEIGHT='0.1', CONTRAST_WEIGHT='0.1', RELATION_WEIGHT='0.1',
        CONTRAST_HARD_NEGATIVES='64', RELATION_HARD_NEGATIVES='64',
        RELATION_TEMPERATURE='0.1', RELATION_TEACHER_TEMPERATURE='0.1',
        NIGHT_AUGMENTATION='camera', AUGMENTATION_CONFIG='configs/night_camera.json',
        AUGMENTATION_STRENGTH='1', AUGMENTATION_PROBABILITY='0.8', AUGMENTATION_EPOCH_OFFSET='0',
        SOURCE_NIGHT_CONFIG='configs/source_night.json', EXCLUDE_PLACES='data/splits/heldout_places.json',
        SOURCE_DAY_MANIFEST='', SOURCE_DAY_CACHE='',
        INIT_ADAPTER_DIR='', DINO_DIR='third_party/dinov2', PATCH_VISIBILITY='off',
        IMPORTANCE_MODE='saliency', MIN_TEACHER_KEEP='0.01',
    )
    return result


def bounded_command(args, arm: str, phase: str) -> list[str]:
    prefix = [args.python, str(ROOT / 'scripts/run_bounded.py'),
              '--max-gib', str(args.max_gib), '--high-gib', str(args.high_gib),
              '--reserve-gib', str(args.reserve_gib), '--log',
              str(args.run_root / f'{arm}.{phase}.memory.jsonl'), '--']
    if phase == 'train':
        return prefix + ['bash', str(ROOT / 'configs/night_token_stage2.sh')]
    if phase != 'eval':
        raise ValueError(f'Unknown phase {phase}')
    return prefix + [args.python, str(ROOT / 'scripts/evaluate_svox.py'),
                     '--split', 'train', '--checkpoint', str(args.checkpoint),
                     '--adapter-dir', str(args.run_root / arm), '--device', 'cuda',
                     '--retrieval-device', 'cuda', '--batch-size', '64', '--num-workers', '4',
                     '--reference-cache-mib', '1024', '--gallery-cache', str(args.gallery_cache),
                     '--save-hard-cases', '--output', str(args.run_root / arm / 'svox_train.json')]


def make_plan(args) -> dict:
    paths = [args.checkpoint, args.manifest, args.day_bank,
             ROOT / 'data/splits/heldout_places.json', ROOT / 'configs/night_camera.json',
             ROOT / 'configs/source_night.json']
    if any(arm in ('T1', 'T3') for arm in args.arms):
        paths.extend([args.real_pairs_review, args.real_pairs_root / 'manifest.jsonl'])
    diagnostic = None
    if args.training_diagnostic is not None:
        paths.append(args.training_diagnostic)
        data = json.loads(args.training_diagnostic.read_text())
        if (data['nonfinite_count'] != 0 or not data['plain_inference_deterministic']
                or data['model_state_sha256_before'] != data['model_state_sha256_after']):
            raise ValueError('Train-only diagnostic failed correctness checks')
        diagnostic = {
            'path': str(args.training_diagnostic), 'batches': data['batches'],
            'gain': data['gain'], 'relative_centered_rms_delta': data['relative_centered_rms_delta'],
            'decision': 'Use the predeclared candidate without gain clipping; diagnostic evidence is bound '
                        'before development evaluation. This script does not select a clipping policy from Recall.',
        }
    required = {str(path): digest(path) for path in paths}
    code = sorted(set((ROOT / 'src/weather_vpr').rglob('*.py')) |
                  set((ROOT / 'scripts').glob('*.py')) |
                  set((ROOT / 'configs').glob('*.sh')) |
                  set((ROOT / 'third_party/salad/workflow').glob('*.py')) |
                  set((ROOT / 'third_party/salad/models/aggregators').glob('*.py')) |
                  set((ROOT / 'third_party/dinov2/dinov2').rglob('*.py')))
    return {
        'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_root': str(args.run_root), 'arms': list(args.arms), 'seed': args.seed,
        'smoke_microbatch_limit': args.max_steps, 'evaluation_split': 'SVOX train (development)',
        'initialization': 'Every arm freshly initializes from the original Day checkpoint; no Night continuation',
        'budget': {'epochs': 1, 'microbatch': 4, 'accumulation': 2, 'effective_main': 8,
                   'source_per_microbatch': 1, 'real_per_microbatch_when_enabled': 1,
                   'precision': 'bf16', 'learning_rate': 1e-6, 'expected_full_updates': 908,
                   'warmup_updates': 0 if args.max_steps else 32,
                   'smoke_schedule_note': 'Smoke limit disables warmup only; full protocol remains warmup32'},
        'token_style': {**STYLE, 'probability': {'T0': 0, 'T1': 0, 'T2': .5, 'T3': .5},
                        'scope': 'Qwen Student patches only; CLS, Teacher, Source, real branch, inference clean'},
        'train_only_preflight': diagnostic,
        'required_files_sha256': required,
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in code},
        'stage1_reference': str(ROOT / 'outputs/stage1_results_20261025/report.md'),
        'selection_policy': 'Fixed final checkpoints; no external-test evaluation or hyperparameter selection',
        'cache_policy': 'Signed existing frozen-Day caches only; each consumer validates cache provenance',
        'memory_guard': {'max_gib': args.max_gib, 'high_gib': args.high_gib, 'reserve_gib': args.reserve_gib},
        'commands': {arm: {'environment': arm_environment(args, arm),
                           'train': bounded_command(args, arm, 'train'),
                           'eval': bounded_command(args, arm, 'eval')} for arm in args.arms},
    }


def verify_frozen_plan(plan: dict) -> None:
    for name, expected in plan['required_files_sha256'].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f'Required experiment input changed after plan freeze: {name}')
    for name, expected in plan['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'Experiment code changed after plan freeze: {name}')


def memory_summary(path: Path) -> dict:
    last = None
    with path.open() as handle:
        for line in handle:
            last = json.loads(line)
    if (not last or last.get('event') != 'finish' or last.get('returncode') != 0
            or last.get('termination_reason') is not None
            or int(last.get('memory_events', {}).get('oom_kill', 0)) != 0):
        raise RuntimeError(f'Missing successful memory-guard completion: {path}')
    return last


def verify_adapter_artifact(directory: Path, expected_sha256: str) -> None:
    if digest(directory / 'adapter.safetensors') != expected_sha256:
        raise RuntimeError(f'Final adapter changed after training: {directory}')


def token_style_choices(record: dict) -> dict:
    return {key: record.get(key) for key in (
        'selected_images', 'donor_indices', 'lambdas', 'patch_indices', 'epoch', 'step')}


def update_hash(hasher, value) -> None:
    hasher.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode() + b'\n')


def training_summary(directory: Path, expected_updates: int | None = None) -> dict:
    metadata = json.loads((directory / 'adapter.json').read_text())
    configuration = json.loads((directory / 'training_config.json').read_text())
    if metadata['init_adapter_dir'] is not None:
        raise RuntimeError(f'{directory.name}: unexpected Night continuation')
    if expected_updates is not None and metadata['optimizer_steps'] != expected_updates:
        raise RuntimeError(f'{directory.name}: incomplete optimizer budget')
    streams = {key: hashlib.sha256() for key in ('main', 'source', 'real_auxiliary', 'token_style_choices')}
    diagnostics = []
    rows = 0
    with (directory / 'training_log.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            update_hash(streams['main'], [row['main_sample_indices'], row['augmentation']])
            update_hash(streams['source'], [row['source_sample_indices'], row['source_night_augmentation']])
            update_hash(streams['real_auxiliary'], [row['auxiliary_sample_ids'], row['auxiliary_augmentation']])
            if row.get('token_style') is not None:
                diagnostics.append(row['token_style'])
                update_hash(streams['token_style_choices'], token_style_choices(row['token_style']))
            elif metadata.get('token_style', {}).get('enabled'):
                raise RuntimeError(f'{directory.name}: enabled token style lacks per-step choices')
    return {
        'optimizer_steps': metadata['optimizer_steps'], 'microbatches': rows,
        'elapsed_seconds': metadata.get('elapsed_seconds'), 'student_views': metadata['student_views'],
        'checkpoint_sha256': metadata['checkpoint_sha256'],
        'adapter_sha256': digest(directory / 'adapter.safetensors'),
        'main_stream_sha256': metadata['main_stream_sha256'],
        'stream_audit_sha256': {key: hasher.hexdigest() for key, hasher in streams.items()},
        'token_style': metadata.get('token_style', configuration.get('effective_token_style')),
        'memory_guard': memory_summary(directory.parent / f'{directory.name}.train.memory.jsonl'),
        'diagnostic_steps': len(diagnostics), 'last_diagnostic': diagnostics[-1] if diagnostics else None,
        'peak_cuda_allocated_bytes': metadata.get('peak_cuda_allocated_bytes'),
    }


def evaluation_summary(path: Path) -> dict:
    result = json.loads(path.read_text())
    def compact(evaluation):
        count = evaluation['num_evaluated_queries']
        return {'num_queries': count, 'recall': evaluation['recall'],
                'hits': {k: round(v * count) for k, v in evaluation['recall'].items()},
                'error_ranks': {r['query_id']: r['rank'] for r in evaluation['error_queries']}}
    return {'gallery_sha256': result['gallery_sha256'], 'checkpoint_sha256': result['checkpoint_sha256'],
            'baseline': compact(result['baseline']), 'night': compact(result['adapter']['evaluation'])}


def paired_changes(before: dict, after: dict) -> dict:
    if before['num_queries'] != after['num_queries']:
        raise RuntimeError('Evaluation query counts differ')
    result = {}
    for k in (1, 5, 10):
        old = {qid for qid, rank in before['error_ranks'].items() if rank > k}
        new = {qid for qid, rank in after['error_ranks'].items() if rank > k}
        result[f'R@{k}'] = {'repaired': sorted(old - new), 'regressed': sorted(new - old),
                            'repaired_count': len(old - new), 'regressed_count': len(new - old)}
    return result


def write_summary(args, state):
    trained = {arm: entry['training'] for arm, entry in state['arms'].items() if 'training' in entry}
    evaluated = {arm: entry['evaluation'] for arm, entry in state['arms'].items() if 'evaluation' in entry}
    integrity = {}
    for key in ('main', 'source'):
        values = {entry['stream_audit_sha256'][key] for entry in trained.values()}
        integrity[key + '_stream_equal'] = len(values) <= 1
    if 'T1' in trained and 'T3' in trained:
        integrity['real_auxiliary_stream_equal'] = (
            trained['T1']['stream_audit_sha256']['real_auxiliary'] ==
            trained['T3']['stream_audit_sha256']['real_auxiliary'])
    if 'T2' in trained and 'T3' in trained:
        integrity['token_style_choices_equal'] = (
            trained['T2']['stream_audit_sha256']['token_style_choices'] ==
            trained['T3']['stream_audit_sha256']['token_style_choices'])
    integrity['original_checkpoint_equal'] = len({t['checkpoint_sha256'] for t in trained.values()}) <= 1
    comparisons = {}
    for before, after in (('T0', 'T1'), ('T0', 'T2'), ('T1', 'T3'), ('T2', 'T3')):
        if before in evaluated and after in evaluated:
            if evaluated[before]['gallery_sha256'] != evaluated[after]['gallery_sha256']:
                raise RuntimeError('Frozen Day gallery provenance differs across arms')
            comparisons[f'{after}_vs_{before}'] = paired_changes(evaluated[before]['night'], evaluated[after]['night'])
    summary = {'run_root': str(args.run_root), 'state': state['state'], 'seed': args.seed,
               'training': trained, 'integrity': integrity, 'evaluations': evaluated,
               'paired_changes': comparisons,
               'limitation': 'Single seed, development split only; no claim of generalization or significance'}
    atomic_json(args.run_root / 'summary.json', summary)
    lines = ['# Stage-two partial-token style experiment', '',
             'Fixed T0/T1/T2/T3, original Day initialization, one epoch, seed ' + str(args.seed) + '.', '',
             '| Arm | Real paired cosine | Partial-token style | Updates | SVOX dev R@1 | R@5 | R@10 |',
             '|---|---|---|---:|---:|---:|---:|']
    for arm in args.arms:
        train = trained.get(arm, {})
        e = evaluated.get(arm, {}).get('night')
        cells = [f"{e['hits'][f'R@{k}']}/{e['num_queries']} ({100*e['recall'][f'R@{k}']:.2f}%)" for k in (1,5,10)] if e else ['pending'] * 3
        lines.append(f"| {arm} | {'yes' if arm in ('T1','T3') else 'no'} | "
                     f"{'yes' if arm in ('T2','T3') else 'no'} | {train.get('optimizer_steps','pending')} | " + ' | '.join(cells) + ' |')
    if evaluated:
        baseline = next(iter(evaluated.values()))['baseline']
        lines.extend(['', 'Original Day baseline hits (R@1/5/10): ' +
                      '/'.join(str(baseline['hits'][f'R@{k}']) for k in (1,5,10)) +
                      f" out of {baseline['num_queries']} queries."])
    for label, change in comparisons.items():
        values = change['R@1']
        lines.extend(['', f"{label}, R@1: {values['repaired_count']} repaired, {values['regressed_count']} regressed."])
    lines.extend(['', 'Integrity: ' + json.dumps(integrity, sort_keys=True), '',
                  'Exact view streams, final adapter hashes, diagnostics, and per-query repairs/regressions are in [summary.json](summary.json).', '',
                  'Only SVOX development was evaluated; external tests remain untouched by this queue. '
                  'One seed cannot establish a stable gain. Main/Source augmentation signatures compare '
                  'sample IDs and per-step operation counters, not independently rehashed image tensors.', '',
                  '[Frozen plan](experiment_plan.json) · [Status](status.json) · '
                  '[Previous stage-one report](../../outputs/stage1_results_20261025/report.md)', ''])
    (args.run_root / 'report.md').write_text('\n'.join(lines))
    if trained and not all(integrity.values()):
        raise RuntimeError('Training stream/provenance integrity comparison failed')


def main(argv=None):
    args = parse_args(argv)
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise SystemExit(f'Refusing nonempty run directory: {args.run_root}')
    # Resolve/hash every required input before creating queue state or running a GPU job.
    plan = make_plan(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.run_root / 'experiment_plan.json', plan)
    state = {'state': 'planned', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'arms': {arm: {'state': 'pending'} for arm in args.arms}}
    atomic_json(args.run_root / 'status.json', state)
    latest = {'run_root': str(args.run_root), 'runner_pid': os.getpid(), 'arms': list(args.arms),
              'smoke_microbatch_limit': args.max_steps,
              'arm_output_dirs': {arm: str(args.run_root / arm) for arm in args.arms}}
    if args.plan_only:
        print(f'Plan only: {args.run_root}', flush=True)
        return 0
    atomic_json(ROOT / 'runs/stage2_training_latest.json', latest)
    started = time.monotonic()
    try:
        state['state'] = 'running'
        for arm in args.arms:
            entry = state['arms'][arm]
            environment = dict(os.environ)
            environment.update(plan['commands'][arm]['environment'])
            for phase in ('train', 'eval'):
                if phase == 'eval' and args.skip_evaluation:
                    continue
                entry['state'] = phase + '_running'
                state['active_arm'], state['active_phase'] = arm, phase
                atomic_json(args.run_root / 'status.json', state)
                print(f'{datetime.now().astimezone().isoformat()} Starting {arm} {phase}', flush=True)
                verify_frozen_plan(plan)
                if phase == 'eval':
                    verify_adapter_artifact(args.run_root / arm, entry['training']['adapter_sha256'])
                with (args.run_root / f'{arm}.{phase}.console.log').open('w') as console:
                    result = subprocess.run(plan['commands'][arm][phase], cwd=ROOT, env=environment,
                                            stdout=console, stderr=subprocess.STDOUT)
                entry[phase + '_returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'{arm} {phase} failed with exit {result.returncode}; inspect console/memory logs')
                if phase == 'train':
                    entry['training'] = training_summary(args.run_root / arm, None if args.max_steps else 908)
                else:
                    verify_adapter_artifact(args.run_root / arm, entry['training']['adapter_sha256'])
                    entry['eval_memory_guard'] = memory_summary(args.run_root / f'{arm}.eval.memory.jsonl')
                    entry['evaluation'] = evaluation_summary(args.run_root / arm / 'svox_train.json')
                write_summary(args, state)
                print(f'{datetime.now().astimezone().isoformat()} Finished {arm} {phase}', flush=True)
            entry['state'] = 'complete' if not args.skip_evaluation else 'smoke_training_complete'
            atomic_json(args.run_root / 'status.json', state)
        state['state'] = 'complete'
    except BaseException as error:
        state['state'] = 'failed'
        state['error'] = str(error)
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.run_root / 'status.json', state)
        write_summary(args, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())

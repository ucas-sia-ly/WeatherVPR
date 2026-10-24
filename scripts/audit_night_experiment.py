#!/usr/bin/env python3
"""Audit the four fixed Night experiments without constructing a model.

Run through ``scripts/run_bounded.py`` with its default 8 GiB RAM limit. Only
one Night state is loaded at a time, on CPU; the Day checkpoint stays on CPU.
An existing report is never overwritten. A failed audit writes a partial report
with its error and then raises, so it cannot be mistaken for a successful run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
STEPS = 908
PROFILES = {
    'full_control': ('all', tuple(range(12))),
    'full_day_anchor': ('all', tuple(range(12))),
    'early6': ('early', tuple(range(6))),
    'late6': ('late', tuple(range(6, 12))),
}
BLOCK_KEY = re.compile(r'^model\.blocks\.(\d+)\.')
BANK_IDENTITY_KEYS = (
    'format_version', 'checkpoint_sha256', 'preprocessing',
    'manifest_sha256', 'source_images', 'places',
)
# These differences are the intended independent variables. Everything else
# in training_config.json, including data, augmentation and optimizer options,
# must match exactly.
VARIABLE_CONFIG_KEYS = {
    'output_dir', 'full_train_scope', 'effective_full_train_scope',
    'day_preservation_weight', 'day_preservation_samples',
    'day_preservation_schedule',
}
COMMON_METADATA_KEYS = (
    'weather', 'adapter_type', 'steps', 'optimizer_steps', 'seed',
    'gradient_accumulation_steps', 'batch_size', 'effective_batch_size',
    'learning_rate', 'weight_decay', 'layer_decay', 'lr_schedule',
    'warmup_steps', 'min_lr_ratio', 'gradient_checkpointing', 'precision',
    'epochs_completed', 'training_pairs', 'checkpoint_sha256', 'model_config',
    'preprocessing', 'manifest', 'manifest_sha256', 'pair_calibration',
    'pair_calibration_sha256', 'exclude_places', 'init_adapter_dir',
    'batch_sampling', 'contrast_queue_size', 'contrast_hard_negatives',
    'relation_temperature', 'relation_teacher_temperature',
    'relation_hard_negatives', 'day_preservation_final_ratio',
    'day_view_augmentation', 'source_night', 'source_night_samples',
    'min_teacher_keep', 'importance_mode', 'night_augmentation', 'code_version',
    'sampled_augmentation_counts', 'sampled_source_night_counts',
)


class AuditError(RuntimeError):
    """A required experimental control failed."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def resolve_project_path(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def json_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise AuditError(f'{path}: nonfinite JSON constant {value}')
    with Path(path).open(encoding='utf-8') as stream:
        value = json.load(stream, parse_constant=reject_constant)
    require(isinstance(value, dict), f'{path}: expected a JSON object')
    return value


def memory_scope():
    """Fail before loading tensors unless the requested hard limit is active."""
    membership = Path('/proc/self/cgroup').read_text().splitlines()
    relative = next((line[3:] for line in membership if line.startswith('0::')), None)
    require(relative is not None, 'A Linux cgroup v2 RAM guard is required')
    directory = Path('/sys/fs/cgroup') / relative.lstrip('/')
    limit = (directory / 'memory.max').read_text().strip()
    swap = (directory / 'memory.swap.max').read_text().strip()
    require(limit != 'max' and 0 < int(limit) <= 8 * GIB,
            'Run this audit through scripts/run_bounded.py with --max-gib <= 8')
    require(swap == '0', 'The audit requires MemorySwapMax=0')
    return {'cgroup': str(directory), 'max_bytes': int(limit),
            'swap_max_bytes': 0}


def audit_memory_log(path):
    records = 0
    start = None
    last = None
    sampled_peak = 0
    sampled_minimum = None
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            require(line.strip(), f'{path}:{line_number}: blank memory record')
            value = json.loads(line)
            require(isinstance(value, dict), f'{path}:{line_number}: invalid memory record')
            records += 1
            if records == 1:
                start = value
            if value.get('event') == 'sample':
                sampled_peak = max(sampled_peak, value['cgroup_bytes'])
                available = value['system_available_bytes']
                sampled_minimum = (available if sampled_minimum is None else
                                   min(sampled_minimum, available))
            last = value
    require(start and start.get('event') == 'start', f'{path}: missing initial start')
    require(last and last.get('event') == 'finish', f'{path}: missing final finish')
    require(last.get('returncode') == 0, f'{path}: training did not return zero')
    require(last.get('termination_reason') is None, f'{path}: training was terminated')
    events = last.get('memory_events', {})
    require('oom_kill' in events and int(events['oom_kill']) == 0,
            f'{path}: missing or nonzero oom_kill counter')
    require(0 < start['max_bytes'] <= 8 * GIB, f'{path}: hard RAM cap exceeds 8 GiB')
    require(start['reserve_bytes'] >= 4 * GIB, f'{path}: RAM reserve below 4 GiB')
    peak = last.get('peak_cgroup_bytes_kernel')
    minimum = last.get('minimum_system_available_bytes')
    require(type(peak) is int and 0 < peak <= 8 * GIB,
            f'{path}: invalid kernel peak or peak exceeds 8 GiB: {peak}')
    require(type(minimum) is int and minimum >= 4 * GIB,
            f'{path}: system available RAM fell below 4 GiB: {minimum}')
    require(sampled_minimum is not None and sampled_minimum >= 4 * GIB,
            f'{path}: missing samples or a sample breached the RAM reserve')
    require(sampled_peak <= peak, f'{path}: sampled peak exceeds kernel peak')
    require(minimum <= sampled_minimum, f'{path}: inconsistent minimum available RAM')
    return {'path': str(path), 'sha256': sha256_file(path), 'records': records,
            'finish': last, 'status': 'passed'}


def audit_training_log(path, profile):
    main_hash = hashlib.sha256()
    source_hash = hashlib.sha256()
    weights = []
    main_samples = source_samples = 0
    with path.open(encoding='utf-8') as stream:
        for expected_step, line in enumerate(stream, 1):
            require(line.strip(), f'{path}:{expected_step}: blank training record')
            record = json.loads(line)
            require(isinstance(record, dict), f'{path}:{expected_step}: invalid record')
            require(record.get('step') == expected_step,
                    f'{path}: missing, duplicated or out-of-order step {expected_step}')
            require(record.get('optimizer_step') == expected_step,
                    f'{path}:{expected_step}: optimizer update mismatch')
            require(record.get('accumulation_count') == 0,
                    f'{path}:{expected_step}: unfinished accumulation')
            require(record.get('epoch') == 1 and record.get('weather') == 'night'
                    and record.get('adapter_type') == 'full',
                    f'{path}:{expected_step}: wrong training domain/epoch/model')
            require(record.get('fixed_day_bank') is True,
                    f'{path}:{expected_step}: frozen Day bank is not active')
            for key in ('loss', 'global_loss', 'global_retrieval_loss', 'relation_loss',
                        'source_night_loss', 'day_preservation_effective_weight'):
                value = record.get(key)
                require(type(value) in (int, float) and math.isfinite(value),
                        f'{path}:{expected_step}: missing/nonfinite {key}')
            weight = record['day_preservation_effective_weight']
            weights.append(weight)
            for key, digest in (('augmentation', main_hash),
                                ('source_night_augmentation', source_hash)):
                require(isinstance(record.get(key), dict),
                        f'{path}:{expected_step}: missing {key}')
                # Include the index and a record separator: step-level order is
                # part of the signature, not merely the aggregate counts.
                digest.update(canonical_bytes({'step': expected_step, key: record[key]}) + b'\n')
            main_samples += record['augmentation']['samples']
            source_samples += record['source_night_augmentation']['samples']
    require(len(weights) == STEPS, f'{path}: expected {STEPS} steps, got {len(weights)}')
    if profile == 'full_day_anchor':
        require(math.isclose(weights[0], .03, rel_tol=0, abs_tol=1e-12),
                f'{path}: anchor must start at .03')
        require(math.isclose(weights[-1], 0., rel_tol=0, abs_tol=1e-12),
                f'{path}: anchor must end at zero')
        require(all(0 <= later <= earlier for earlier, later in zip(weights, weights[1:])),
                f'{path}: anchor weights are negative or not monotonically decreasing')
        require(all(math.isclose(weight, .03 * (1 - index / (STEPS - 1)),
                                 rel_tol=0, abs_tol=1e-12)
                    for index, weight in enumerate(weights)),
                f'{path}: anchor weights differ from the fixed linear schedule')
    else:
        require(all(weight == 0 for weight in weights), f'{path}: unexpected Day anchor')
    return {'path': str(path), 'sha256': sha256_file(path), 'steps': len(weights),
            'augmentation_per_step_sha256': main_hash.hexdigest(),
            'source_night_augmentation_per_step_sha256': source_hash.hexdigest(),
            'main_samples': main_samples, 'source_night_samples': source_samples,
            'day_anchor': {'first': weights[0], 'last': weights[-1],
                           'minimum': min(weights), 'maximum': max(weights)},
            'status': 'passed'}


def audit_controls(run_root, checkpoint_hash, report):
    metadata = {}
    configs = {}
    reference_config = reference_metadata = reference_augmentations = None
    file_hashes = {}
    for profile, (scope_name, block_indices) in PROFILES.items():
        directory = run_root / profile
        meta = read_json(directory / 'adapter.json')
        config = read_json(directory / 'training_config.json')
        metadata[profile], configs[profile] = meta, config
        result = report['profiles'][profile] = {}
        require(meta.get('steps') == STEPS and meta.get('optimizer_steps') == STEPS,
                f'{profile}: incomplete training counters')
        require(meta.get('seed') == 42 and config.get('seed') == 42,
                f'{profile}: seed must be 42')
        require(meta.get('checkpoint_sha256') == checkpoint_hash,
                f'{profile}: wrong initialization checkpoint')
        require(meta.get('weather') == 'night' and meta.get('adapter_type') == 'full',
                f'{profile}: expected independent full Night backbone')
        require(meta.get('init_adapter_dir') is None and config.get('init_adapter_dir') is None,
                f'{profile}: unexpected warm-start/resume')
        require(config.get('verify_hashes') is True and config.get('num_workers') == 0,
                f'{profile}: training data verification or bounded worker setting changed')
        require(meta.get('epochs_completed') == 1 and meta.get('training_pairs') == 7263,
                f'{profile}: unexpected epochs or accepted training-pair count')
        require(meta.get('batch_size') == 8 and meta.get('gradient_accumulation_steps') == 1,
                f'{profile}: expected batch 8 with one optimizer step per batch')
        scope = meta.get('full_train_scope')
        require(isinstance(scope, dict) and scope.get('scope') == scope_name
                and scope.get('block_indices') == list(block_indices)
                and scope.get('block_count') == len(block_indices)
                and scope.get('total_blocks') == 12,
                f'{profile}: wrong full_train_scope: {scope}')
        require(scope == config.get('effective_full_train_scope')
                and config.get('full_train_scope') == scope_name,
                f'{profile}: saved configuration and metadata scopes disagree')
        require(meta.get('day_view_augmentation', {}).get('profile') == 'off',
                f'{profile}: unexpected Day Student view augmentation')
        expected_anchor = .03 if profile == 'full_day_anchor' else 0.
        require(meta.get('loss_weights', {}).get('day_preservation') == expected_anchor
                and config.get('day_preservation_weight') == expected_anchor,
                f'{profile}: wrong configured Day anchor')
        require(meta.get('day_preservation_final_ratio') == 0,
                f'{profile}: anchor final ratio must be zero')
        if profile == 'full_day_anchor':
            require(meta.get('day_preservation_samples') == 2
                    and meta.get('day_preservation_schedule') == 'linear',
                    f'{profile}: wrong Day anchor sample count/schedule')
        manifest = resolve_project_path(meta['manifest'])
        for path in (manifest, resolve_project_path(config['exclude_places']),
                     resolve_project_path(config['fixed_day_bank'])):
            if path not in file_hashes:
                file_hashes[path] = sha256_file(path)
        require(file_hashes[manifest] == meta.get('manifest_sha256'),
                f'{profile}: manifest bytes differ from recorded training manifest')
        bank = meta.get('fixed_day_bank')
        require(isinstance(bank, dict), f'{profile}: missing fixed Day bank metadata')
        bank_identity = {key: bank[key] for key in BANK_IDENTITY_KEYS}
        require(json_sha256(bank_identity) == bank.get('cache_signature'),
                f'{profile}: fixed Day bank signature is invalid')
        require(bank['checkpoint_sha256'] == checkpoint_hash
                and bank['manifest_sha256'] == meta['manifest_sha256']
                and bank['preprocessing'] == meta.get('preprocessing'),
                f'{profile}: fixed Day bank targets differ from Teacher/data/preprocessing')
        require(len(bank['source_images']) == 7263 and len(bank['places']) == 7263,
                f'{profile}: wrong Day bank row count')
        bank_path = resolve_project_path(config['fixed_day_bank'])
        require(file_hashes[bank_path] == bank.get('tensor_sha256'),
                f'{profile}: fixed Day bank bytes differ from training signature')
        common_config = {key: value for key, value in config.items()
                         if key not in VARIABLE_CONFIG_KEYS}
        common_meta = {key: meta[key] for key in COMMON_METADATA_KEYS}
        common_meta['loss_weights'] = {key: value for key, value in meta['loss_weights'].items()
                                       if key != 'day_preservation'}
        common_meta['fixed_day_bank_signature'] = bank['cache_signature']
        common_meta['fixed_day_bank_tensor_sha256'] = bank['tensor_sha256']
        config_hash = json_sha256(common_config)
        metadata_hash = json_sha256(common_meta)
        result.update({'metadata_sha256': sha256_file(directory / 'adapter.json'),
                       'training_config_sha256': sha256_file(directory / 'training_config.json'),
                       'common_config_sha256': config_hash,
                       'common_metadata_sha256': metadata_hash,
                       'full_train_scope': scope,
                       'fixed_day_bank_signature': bank['cache_signature'],
                       'fixed_day_bank_tensor_sha256': bank['tensor_sha256']})
        result['training_log'] = audit_training_log(directory / 'training_log.jsonl', profile)
        result['training_memory'] = audit_memory_log(run_root / f'{profile}.memory.jsonl')
        augmentations = tuple(result['training_log'][key] for key in (
            'augmentation_per_step_sha256', 'source_night_augmentation_per_step_sha256'))
        require(result['training_log']['main_samples'] == 7263
                and result['training_log']['source_night_samples'] == STEPS * 2,
                f'{profile}: wrong sampled main/source-night training view counts')
        if reference_config is None:
            reference_config, reference_metadata = common_config, common_meta
            reference_augmentations = augmentations
        else:
            require(common_config == reference_config,
                    f'{profile}: uncontrolled training_config difference: '
                    f'{sorted(key for key in set(common_config) | set(reference_config) if common_config.get(key) != reference_config.get(key))}')
            require(common_meta == reference_metadata,
                    f'{profile}: uncontrolled metadata difference: '
                    f'{sorted(key for key in set(common_meta) | set(reference_metadata) if common_meta.get(key) != reference_metadata.get(key))}')
            require(augmentations == reference_augmentations,
                    f'{profile}: per-step main or source-night augmentation sequence changed')
        print(f'Controls passed: {profile}', flush=True)
    report['common_controls'] = {
        'config_sha256': json_sha256(reference_config),
        'metadata_sha256': json_sha256(reference_metadata),
        'augmentation_per_step_sha256': reference_augmentations[0],
        'source_night_augmentation_per_step_sha256': reference_augmentations[1],
        'signed_files': {str(path): digest for path, digest in file_hashes.items()},
        'allowed_config_differences': sorted(VARIABLE_CONFIG_KEYS),
        'status': 'passed',
    }
    return metadata


def audit_states(run_root, checkpoint, metadata, report):
    # CUDA is deliberately unavailable to this process. No model, hub loader,
    # optimizer, Gallery tensor or training image is constructed/loaded.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    sys.path.insert(0, str(ROOT / 'src'))
    from weather_vpr.utils.backend import enable_backend_imports
    enable_backend_imports()
    import torch
    from safetensors.torch import load_file
    from workflow.model import checkpoint_state_and_config, read_checkpoint
    torch.set_num_threads(4)
    saved = read_checkpoint(checkpoint)
    state, model_config = checkpoint_state_and_config(saved)
    day = {key.removeprefix('backbone.'): value for key, value in state.items()
           if key.startswith('backbone.')}
    require(day, 'Teacher checkpoint contains no backbone.* tensors')
    require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                for value in day.values()), 'Teacher backbone contains non-CPU/non-tensor state')
    del state, saved
    gc.collect()
    day_keys = set(day)
    require({int(match.group(1)) for key in day
             if (match := BLOCK_KEY.match(key))} == set(range(12)),
            'Expected exactly twelve DINOv2 Transformer blocks in the Teacher')
    require(not any('aggregator' in key for key in day),
            'Teacher backbone extraction unexpectedly contains SALAD parameters')
    report['checkpoint'].update({'backbone_tensor_count': len(day),
                                 'backbone_numel': sum(value.numel() for value in day.values()),
                                 'model_config': model_config})
    for profile, (scope_name, block_indices) in PROFILES.items():
        require(metadata[profile]['model_config'] == model_config,
                f'{profile}: model configuration differs from original Day checkpoint')
        path = run_root / profile / 'adapter.safetensors'
        result = report['profiles'][profile]['state_validation'] = {
            'path': str(path), 'sha256': sha256_file(path), 'status': 'checking',
            'blocked_keys': [],
        }
        night = load_file(str(path), device='cpu')
        try:
            night_keys = set(night)
            result['missing_keys'] = sorted(day_keys - night_keys)
            result['unexpected_keys'] = sorted(night_keys - day_keys)
            require(night_keys == day_keys, f'{profile}: Night backbone state keys differ from Teacher')
            require(not any('aggregator' in key for key in night),
                    f'{profile}: Night state must not contain SALAD parameters')
            changed_keys = []
            changed_elements = 0
            per_block = {str(index): 0 for index in range(12)}
            protected_equal = 0
            shape_or_dtype = []
            for key in sorted(day):
                target, candidate = day[key], night[key]
                if target.shape != candidate.shape or target.dtype != candidate.dtype:
                    shape_or_dtype.append(key)
                    continue
                require(candidate.device.type == 'cpu', f'{profile}:{key}: tensor is not on CPU')
                require(bool(torch.isfinite(candidate).all()), f'{profile}:{key}: nonfinite Night state')
                equal = torch.equal(candidate, target)
                match = BLOCK_KEY.match(key)
                allowed = scope_name == 'all' or (match is not None and int(match.group(1)) in block_indices)
                if not equal:
                    changed_keys.append(key)
                    changed_elements += int(torch.count_nonzero(candidate != target))
                    if match:
                        per_block[match.group(1)] += 1
                    if not allowed:
                        result['blocked_keys'].append(key)
                elif not allowed:
                    protected_equal += 1
            result.update({'tensor_count': len(night), 'changed_tensor_count': len(changed_keys),
                           'changed_element_count': changed_elements, 'changed_keys': changed_keys,
                           'changed_tensor_count_by_block': per_block,
                           'protected_equal_tensor_count': protected_equal,
                           'shape_or_dtype_mismatch_keys': shape_or_dtype})
            require(not shape_or_dtype, f'{profile}: state shape/dtype mismatch: {shape_or_dtype}')
            require(not result['blocked_keys'],
                    f'{profile}: frozen state changed: {result["blocked_keys"]}')
            require(changed_keys, f'{profile}: no Night backbone tensor changed')
            if scope_name != 'all':
                missing_changes = [index for index in block_indices if per_block[str(index)] == 0]
                result['selected_blocks_without_changes'] = missing_changes
                require(not missing_changes,
                        f'{profile}: selected blocks have no changed tensor: {missing_changes}')
                require(protected_equal > 0, f'{profile}: no frozen state was verified')
            result['status'] = 'passed'
            print(f'State passed: {profile}; {len(changed_keys)} changed tensors', flush=True)
        finally:
            # Drop every reference to this mapped Night file before the next
            # profile, including the final loop-local tensor reference.
            candidate = None
            night.clear()
            del night
            gc.collect()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation retains an earlier audit verbatim, including when two
    # accidental invocations race to use the same output path.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('runs/night_regularization_20261024'))
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/dino_salad.ckpt'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    run_root = resolve_project_path(args.run_root)
    checkpoint = resolve_project_path(args.checkpoint)
    output = resolve_project_path(args.output) if args.output else run_root / 'control_checks.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing audit report: {output}; choose --output')
    report = {'format_version': 1, 'status': 'checking',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'run_root': str(run_root), 'checkpoint': {'path': str(checkpoint)},
              'profiles': {}}
    try:
        report['audit_memory_guard'] = memory_scope()
        checkpoint_hash = sha256_file(checkpoint)
        report['checkpoint']['sha256'] = checkpoint_hash
        metadata = audit_controls(run_root, checkpoint_hash, report)
        audit_states(run_root, checkpoint, metadata, report)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        # Linux ru_maxrss is KiB. The parent RAM guard records the stronger
        # whole-cgroup kernel peak and reserve in its own audit memory log.
        report['audit_process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_report(output, report)
    print(f'Audit passed: {output}', flush=True)


if __name__ == '__main__':
    main()

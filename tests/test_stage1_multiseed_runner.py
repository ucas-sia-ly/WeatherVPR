"""CPU-only checks for replication bookkeeping and protocol isolation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_stage1_multiseed.py'
SPEC = importlib.util.spec_from_file_location('stage1_multiseed_runner', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def evaluation(errors, count=10):
    hits = {f'R@{k}': count - sum(rank > k for rank in errors.values()) for k in (1, 5, 10)}
    return {'num_queries': count, 'hits': hits,
            'recall': {key: value / count for key, value in hits.items()}, 'error_ranks': errors}


def record(seed, arm, errors):
    baseline = evaluation({'q0': 20})
    return {'origin': 'new_replication', 'training': {
        'checkpoint_sha256': 'day', 'manifest_sha256': 'manifest',
        'stream_audit_sha256': {'main': f'main{seed}', 'source': f'source{seed}',
                                'auxiliary': f'{seed}{arm}'}},
        'evaluation': {'gallery_sha256': 'gallery', 'checkpoint_sha256': 'day',
                       'baseline': baseline, 'night': evaluation(errors)}}


class MultiSeedRunnerTests(unittest.TestCase):
    def test_new_seeds_exclude_duplicate_reference_and_repeated_seeds(self):
        for seeds in (['42'], ['43', '43'], ['-1']):
            with self.subTest(seeds=seeds), self.assertRaises(SystemExit):
                runner.parse_args(['--output', '/tmp/multiseed', '--seeds', *seeds])

    def test_defaults_match_requested_fixed_protocol(self):
        args = runner.parse_args(['--output', '/tmp/multiseed'])
        self.assertEqual(args.seeds, [43, 44])
        self.assertEqual(args.arms, ['A', 'B', 'C'])
        for arm in args.arms:
            env = runner.arm_environment(args, 43, arm)
            self.assertEqual((env['BATCH_SIZE'], env['GRADIENT_ACCUMULATION_STEPS'],
                              env['SOURCE_NIGHT_SAMPLES'], env['AUXILIARY_SAMPLES']), ('4', '2', '1', '1'))
            self.assertEqual(env['STAGE1_ARM'], arm)
            self.assertEqual(env['SEED'], '43')
            self.assertEqual(env['SOURCE_DAY_MANIFEST'], '')
            self.assertEqual(env['SOURCE_DAY_CACHE'], '')
            self.assertEqual(env['TOKEN_STYLE_PROBABILITY'], '0')
            self.assertEqual(env['INIT_ADAPTER_DIR'], '')
            self.assertEqual(env['WARMUP_STEPS'], '32')

    def test_smoke_does_not_silently_keep_full_warmup(self):
        args = runner.parse_args(['--output', '/tmp/multiseed', '--max-steps', '3', '--skip-evaluation'])
        self.assertEqual(runner.arm_environment(args, 44, 'B')['WARMUP_STEPS'], '0')
        with self.assertRaises(SystemExit):
            runner.parse_args(['--output', '/tmp/multiseed', '--skip-evaluation'])

    def test_only_development_evaluation_commands(self):
        args = runner.parse_args(['--output', '/tmp/multiseed'])
        command = runner.bounded_command(args, 44, 'C', 'eval')
        self.assertEqual(command[command.index('--split') + 1], 'train')
        self.assertIn('/tmp/multiseed/seed44/C', command)
        self.assertIn('64', command)
        self.assertIn('--max-gib', command)
        train = runner.bounded_command(args, 44, 'C', 'train')
        self.assertEqual(Path(train[-1]).name, 'night_real_stage1_fast.sh')
        with self.assertRaises(ValueError):
            runner.bounded_command(args, 44, 'C', 'test')

    def test_paired_topk_changes_and_sample_std_across_seeds(self):
        records = {
            'seed42_A': record(42, 'A', {'q1': 3, 'q2': 11}),
            'seed42_B': record(42, 'B', {'q2': 11, 'q3': 6}),
            'seed43_A': record(43, 'A', {'q1': 3, 'q2': 11}),
            'seed43_B': record(43, 'B', {'q2': 11}),
        }
        result = runner.summarize_records(records)
        change = result['paired_changes']['seed42_B_vs_A']
        self.assertEqual(change['R@1']['repaired'], ['q1'])
        self.assertEqual(change['R@1']['regressed'], ['q3'])
        self.assertEqual(change['R@5']['regressed'], ['q3'])
        self.assertEqual(change['R@10']['regressed_count'], 0)
        metric = result['aggregate']['B']['recall']['R@1']
        self.assertEqual(metric['n'], 2)
        self.assertAlmostEqual(metric['mean'], .85)
        self.assertAlmostEqual(metric['sample_std'], .1 / 2 ** .5)
        delta = result['paired_seed_deltas']['B_minus_A']['recall_delta']['R@1']
        self.assertAlmostEqual(delta['mean'], .05)
        self.assertTrue(all(result['integrity']['42'].values()))

    def test_different_seeds_may_differ_but_arms_within_seed_must_match(self):
        records = {'seed43_A': record(43, 'A', {}), 'seed44_A': record(44, 'A', {})}
        runner.summarize_records(records)
        records['seed43_B'] = record(43, 'B', {})
        records['seed43_B']['training']['stream_audit_sha256']['source'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'comparability'):
            runner.summarize_records(records)

    def test_gallery_or_baseline_mismatch_cannot_be_aggregated(self):
        records = {'seed43_A': record(43, 'A', {}), 'seed43_B': record(43, 'B', {})}
        records['seed43_B']['evaluation']['gallery_sha256'] = 'different'
        with self.assertRaisesRegex(RuntimeError, 'gallery'):
            runner.summarize_records(records)

    def test_development_eval_must_have_complete_topk_error_lists(self):
        def raw_result():
            eval_part = {'num_evaluated_queries': 2, 'recall': {'R@1': .5, 'R@5': 1., 'R@10': 1.},
                         'error_queries': [{'query_id': 'q1', 'rank': 3}]}
            return {'dataset': 'SVOX', 'split': 'train', 'query_group': 'queries_night',
                    'gallery_sha256': 'gallery', 'checkpoint_sha256': 'day',
                    'baseline': eval_part, 'adapter': {'evaluation': eval_part}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evaluation.json'
            raw = raw_result()
            path.write_text(json.dumps(raw))
            self.assertEqual(runner.evaluation_summary(path)['night']['hits']['R@1'], 1)
            raw['baseline']['error_queries'] = []
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                runner.evaluation_summary(path)
            raw = raw_result()
            raw['split'] = 'test'
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RuntimeError, 'development'):
                runner.evaluation_summary(path)

    def test_single_seed_does_not_fabricate_variance(self):
        result = runner.mean_std([.8])
        self.assertEqual(result['mean'], .8)
        self.assertIsNone(result['sample_std'])
        self.assertIsNone(runner.mean_std([])['mean'])

    def test_historical_disabled_token_record_and_legacy_none_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'training_config.json').write_text('{}')
            (directory / 'adapter.json').write_text(json.dumps({
                'optimizer_steps': 1, 'student_views': {'qwen': 2},
                'checkpoint_sha256': 'day', 'manifest_sha256': 'manifest',
                'main_stream_sha256': 'main'}))
            (directory / 'adapter.safetensors').write_bytes(b'fixture adapter')
            memory = directory / 'memory.jsonl'
            memory.write_text(json.dumps({'event': 'finish', 'returncode': 0,
                                          'termination_reason': None, 'memory_events': {}}) + '\n')
            row = {'main_sample_indices': [1], 'source_sample_indices': [1],
                   'augmentation': {}, 'source_night_augmentation': {},
                   'auxiliary_sample_ids': [], 'auxiliary_augmentation': {}}
            log = directory / 'training_log.jsonl'
            log.write_text('\n'.join(json.dumps({**row, 'token_style': value})
                                     for value in ({'enabled': False}, None)) + '\n')
            with mock.patch.object(runner, 'validate_protocol'):
                result = runner.training_summary(directory, 43, 'A', memory_log=memory, max_steps=2)
                self.assertEqual(result['microbatches'], 2)
                log.write_text(json.dumps({**row, 'token_style': {'enabled': True}}) + '\n')
                with self.assertRaisesRegex(RuntimeError, 'token stylization'):
                    runner.training_summary(directory, 43, 'A', memory_log=memory, max_steps=2)


if __name__ == '__main__':
    unittest.main()

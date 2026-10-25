"""Move to tests/test_gsv_scale_runner.py with the runner after P0 freeze ends."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_gsv_scale.py'
SPEC = importlib.util.spec_from_file_location('gsv_scale_runner', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def record(cell, hits):
    scale, arm = cell.split('_')
    count = 10
    errors = {f'q{i}': 2 for i in range(count - hits)}
    evaluation = {'num_queries': count, 'error_ranks': errors,
                  'recall': {'R@1': hits / count, 'R@5': 1., 'R@10': 1.},
                  'hits': {'R@1': hits, 'R@5': 10, 'R@10': 10}}
    return {'training': {'main_stream_sha256': 'main', 'checkpoint_sha256': 'day',
                        'manifest_sha256': 'qwen', 'source_manifest_sha256': scale,
                        'stream_audit_sha256': {'main': 'main', 'source': scale, 'real_auxiliary': arm}},
            'evaluation': {'gallery_sha256': 'gallery', 'checkpoint_sha256': 'day',
                           'baseline': {}, 'night': evaluation}}


class GSVScaleRunnerTests(unittest.TestCase):
    def test_fixed_budget_and_independent_source_are_explicit(self):
        args = runner.parse_args(['--output', '/tmp/gsv-scale'])
        self.assertEqual(args.seed, 42)
        environments = {cell: runner.arm_environment(args, cell) for cell in runner.CELLS}
        for cell, env in environments.items():
            self.assertEqual(env['BATCH_SIZE'], '4')
            self.assertEqual(env['GRADIENT_ACCUMULATION_STEPS'], '2')
            self.assertEqual(env['SOURCE_NIGHT_SAMPLES'], '1')
            self.assertEqual(env['AUXILIARY_SAMPLES'], '1')
            self.assertEqual(env['TOKEN_STYLE_PROBABILITY'], '0')
            self.assertEqual(env['INIT_ADAPTER_DIR'], '')
            self.assertEqual(env['GSV_SCALE_ARM'], cell[-1])
            self.assertEqual(env['DAY_BANK'], str(args.day_bank))
            self.assertEqual(env['BATCH_SAMPLING'], 'distinct')
            self.assertEqual(env['TEMPERATURE'], '0.1')
            self.assertEqual(env['CONTRAST_QUEUE_SIZE'], '256')
        self.assertEqual(environments['G1_A']['SOURCE_DAY_MANIFEST'], environments['G1_B']['SOURCE_DAY_MANIFEST'])
        self.assertNotEqual(environments['G1_A']['SOURCE_DAY_MANIFEST'], environments['G2_A']['SOURCE_DAY_MANIFEST'])
        self.assertNotEqual(environments['G1_A']['SOURCE_DAY_CACHE'], environments['G2_A']['SOURCE_DAY_CACHE'])

    def test_development_only_and_smoke_warmup(self):
        args = runner.parse_args(['--output', '/tmp/gsv-scale', '--max-steps', '3', '--skip-evaluation'])
        self.assertEqual(runner.arm_environment(args, 'G2_B')['WARMUP_STEPS'], '0')
        command = runner.bounded_command(args, 'G1_B', 'eval')
        self.assertEqual(command[command.index('--split') + 1], 'train')
        self.assertIn('/tmp/gsv-scale/G1_B', command)
        with self.assertRaises(SystemExit):
            runner.parse_args(['--output', '/tmp/gsv-scale', '--skip-evaluation'])

    def test_all_four_paired_comparisons(self):
        records = {cell: record(cell, hits) for cell, hits in
                   zip(runner.CELLS, (6, 7, 8, 9))}
        result = runner.summarize_records(records)
        self.assertTrue(all(result['integrity'].values()))
        self.assertEqual(set(result['paired_changes']),
                         {'G1_B_vs_G1_A', 'G2_B_vs_G2_A', 'G2_A_vs_G1_A', 'G2_B_vs_G1_B'})
        self.assertEqual(result['paired_changes']['G2_A_vs_G1_A']['R@1']['repaired_count'], 2)
        self.assertAlmostEqual(result['paired_changes']['G2_B_vs_G1_B']['R@1']['recall_delta'], .2)

    def test_same_scale_source_or_cross_scale_main_drift_is_rejected(self):
        records = {cell: record(cell, 7) for cell in runner.CELLS}
        records['G1_B']['training']['stream_audit_sha256']['source'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'comparability'):
            runner.summarize_records(records)
        records = {cell: record(cell, 7) for cell in runner.CELLS}
        records['G2_A']['training']['main_stream_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'comparability'):
            runner.summarize_records(records)

    def test_smoke_cache_materialization_is_frozen_before_later_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cache = output / 'source.safetensors'
            args = runner.parse_args(['--output', str(output), '--max-steps', '3', '--g1-cache', str(cache)])
            cache.write_bytes(b'new Source bank')
            Path(str(cache) + '.json').write_text('{"signed": true}')
            plan = {'required_files_sha256': {}, 'materialized_source_caches': {},
                    'initially_absent_source_caches': [str(cache)]}
            artifact = runner.freeze_materialized_source_cache(args, 'G1_A', plan)
            self.assertTrue(artifact['built_after_initial_smoke_plan'])
            self.assertIn(str(cache), plan['required_files_sha256'])
            cache.write_bytes(b'changed Source bank')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                runner.freeze_materialized_source_cache(args, 'G1_B', plan)


if __name__ == '__main__':
    unittest.main()

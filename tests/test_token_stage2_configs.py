"""Lightweight checks of fixed stage-two commands, hashing, and result comparison."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stage2_runner', ROOT / 'scripts/run_token_style_stage2.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class StageTwoConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.capture = self.directory / 'capture.json'
        self.stub = self.directory / 'python-stub'
        self.stub.write_text(
            '#!/usr/bin/env python3\n'
            'import json,os,sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["CAPTURE"]).write_text(json.dumps({"args":sys.argv[1:],"env":dict(os.environ)}))\n')
        self.stub.chmod(0o755)
        texts = ''.join(p.read_text() for p in (ROOT / 'configs').glob('*.sh'))
        names = set(re.findall(r'\$\{([A-Z][A-Z0-9_]*)', texts))
        self.environment = {k: v for k, v in os.environ.items() if k not in names}
        self.environment.update(PYTHON=str(self.stub), CAPTURE=str(self.capture), REAL_PAIRS_REVIEW='review.jsonl')

    def invoke(self, arm, **overrides):
        self.capture.unlink(missing_ok=True)
        environment = dict(self.environment, STAGE2_ARM=arm, **overrides)
        result = subprocess.run(['bash', str(ROOT / 'configs/night_token_stage2.sh')], cwd=ROOT,
                                env=environment, text=True, capture_output=True)
        capture = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return result, capture

    def test_fixed_factorial_arms_and_actual_training_budget(self):
        for arm in RUNNER.ARMS:
            with self.subTest(arm=arm):
                process, capture = self.invoke(arm)
                self.assertEqual(process.returncode, 0, process.stderr)
                env, args = capture['env'], capture['args']
                self.assertEqual(env['STAGE1_ARM'], 'B' if arm in ('T1', 'T3') else 'A')
                self.assertEqual(env['TOKEN_STYLE_PROBABILITY'], '0.5' if arm in ('T2', 'T3') else '0')
                for key, value in dict(TOKEN_STYLE_AFTER_BLOCK='7', TOKEN_STYLE_FRACTION='0.3',
                                       TOKEN_STYLE_STRENGTH='0.2', TOKEN_STYLE_BETA='0.1').items():
                    self.assertEqual(env[key], value)
                expected = {'--batch-size': '4', '--gradient-accumulation-steps': '2',
                            '--source-night-samples': '1', '--auxiliary-samples': '1',
                            '--num-workers': '4', '--day-bank-device': 'cuda', '--epochs': '1',
                            '--learning-rate': '0.000001', '--full-train-blocks': '4',
                            '--token-style-after-block': '7', '--token-style-fraction': '0.3',
                            '--token-style-strength': '0.2', '--token-style-beta': '0.1',
                            '--token-style-probability': '0.5' if arm in ('T2','T3') else '0'}
                for flag, value in expected.items():
                    self.assertEqual(args[args.index(flag)+1], value)
                self.assertIn('--bf16', args)
                self.assertNotIn('--gradient-checkpointing', args)
                self.assertNotIn('--init-adapter-dir', args)

    def test_protocol_style_values_cannot_be_overridden_accidentally(self):
        process, capture = self.invoke('T2', TOKEN_STYLE_PROBABILITY='1', TOKEN_STYLE_FRACTION='.9',
                                       TOKEN_STYLE_STRENGTH='5', TOKEN_STYLE_BETA='3', TOKEN_STYLE_AFTER_BLOCK='0')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(capture['env']['TOKEN_STYLE_PROBABILITY'], '0.5')
        self.assertEqual(capture['env']['TOKEN_STYLE_AFTER_BLOCK'], '7')
        self.assertEqual(capture['env']['TOKEN_STYLE_FRACTION'], '0.3')

    def test_invalid_arm_and_night_continuation_refused(self):
        process, capture = self.invoke('bad')
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(capture)
        process, capture = self.invoke('T2', INIT_ADAPTER_DIR='old_weights')
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(capture)


class StageTwoRunnerTests(unittest.TestCase):
    def arguments(self, *extra):
        return RUNNER.parse_args(['--run-root', '/tmp/weather-stage2-test', *extra])

    def test_train_and_dev_eval_both_use_sequential_memory_guard(self):
        args = self.arguments()
        for phase in ('train', 'eval'):
            command = RUNNER.bounded_command(args, 'T3', phase)
            self.assertEqual(command[1], str(ROOT / 'scripts/run_bounded.py'))
            for key, value in (('--max-gib', '7.0'), ('--high-gib', '6.0'), ('--reserve-gib', '2.0')):
                self.assertEqual(command[command.index(key)+1], value)
        evaluate = RUNNER.bounded_command(args, 'T3', 'eval')
        self.assertEqual(evaluate[evaluate.index('--split')+1], 'train')
        self.assertEqual(evaluate[evaluate.index('--batch-size')+1], '64')
        self.assertEqual(evaluate[evaluate.index('--reference-cache-mib')+1], '1024')
        self.assertIn('--save-hard-cases', evaluate)
        self.assertNotIn('--query-scales', evaluate)
        self.assertNotIn('--query-fusion-alpha', evaluate)

    def test_explicit_environment_preserves_fresh_control_budget(self):
        env = RUNNER.arm_environment(self.arguments(), 'T0')
        self.assertEqual(env['INIT_ADAPTER_DIR'], '')
        self.assertEqual(env['BATCH_SIZE'], '4')
        self.assertEqual(env['GRADIENT_ACCUMULATION_STEPS'], '2')
        self.assertEqual(env['SOURCE_NIGHT_SAMPLES'], '1')
        self.assertEqual(env['MAX_STEPS'], '0')
        self.assertEqual(env['GRADIENT_CHECKPOINTING'], '0')
        self.assertEqual(env['DAY_BANK_DEVICE'], 'cuda')

    def test_smoke_disables_only_the_short_schedule_warmup(self):
        full = RUNNER.arm_environment(self.arguments(), 'T0')
        smoke = RUNNER.arm_environment(self.arguments('--max-steps', '3', '--skip-evaluation'), 'T0')
        self.assertEqual(full['WARMUP_STEPS'], '32')
        self.assertEqual(smoke['WARMUP_STEPS'], '0')
        self.assertEqual(smoke['MAX_STEPS'], '3')
        self.assertEqual(smoke['LEARNING_RATE'], full['LEARNING_RATE'])

    def test_plan_hashes_inputs_before_any_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments('--arms', 'T0')
            root = Path(temporary)
            for key in ('checkpoint', 'manifest', 'day_bank'):
                path = root / key
                path.write_text(key)
                setattr(args, key, path)
            plan = RUNNER.make_plan(args)
            self.assertEqual(plan['arms'], ['T0'])
            self.assertEqual(plan['required_files_sha256'][str(args.checkpoint)], RUNNER.digest(args.checkpoint))
            self.assertIn('src/weather_vpr/training/train.py', plan['code_sha256'])
            self.assertEqual(plan['budget']['expected_full_updates'], 908)
            self.assertEqual(plan['token_style']['after_block'], 7)
            args.manifest.unlink()
            with self.assertRaises(FileNotFoundError):
                RUNNER.make_plan(args)

    def test_style_choices_include_random_draws_and_step_but_not_feature_amplitudes(self):
        record = {'selected_images': [0], 'donor_indices': [1, 0], 'lambdas': [.1, 0],
                  'patch_indices': [[2, 3]], 'epoch': 0, 'step': 9, 'raw_gain_max': 2}
        actual = RUNNER.token_style_choices(record)
        self.assertEqual(set(actual), {'selected_images', 'donor_indices', 'lambdas',
                                       'patch_indices', 'epoch', 'step'})
        record['raw_gain_max'] = 5
        self.assertEqual(actual, RUNNER.token_style_choices(record))
        record['donor_indices'] = [0, 1]
        self.assertNotEqual(actual, RUNNER.token_style_choices(record))

    def test_memory_guard_rejects_oom_and_reserve_termination_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'memory.jsonl'
            clean = {'event': 'finish', 'returncode': 0, 'termination_reason': None,
                     'memory_events': {'oom_kill': '0'}}
            path.write_text(json.dumps(clean))
            self.assertEqual(RUNNER.memory_summary(path), clean)
            for overrides in ({'returncode': 1}, {'termination_reason': 'reserve'},
                              {'memory_events': {'oom_kill': '1'}}, {'event': 'sample'}):
                path.write_text(json.dumps(dict(clean, **overrides)))
                with self.assertRaises(RuntimeError):
                    RUNNER.memory_summary(path)

    def test_changed_final_adapter_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'adapter.safetensors'
            path.write_bytes(b'final checkpoint')
            signature = RUNNER.digest(path)
            RUNNER.verify_adapter_artifact(directory, signature)
            path.write_bytes(b'changed checkpoint')
            with self.assertRaises(RuntimeError):
                RUNNER.verify_adapter_artifact(directory, signature)

    def test_pairwise_repair_and_regression_counts(self):
        before = {'num_queries': 4, 'error_ranks': {'a': 2, 'b': 6, 'c': 12}}
        after = {'num_queries': 4, 'error_ranks': {'b': 2, 'd': 11}}
        result = RUNNER.paired_changes(before, after)
        self.assertEqual(result['R@1']['repaired'], ['a', 'c'])
        self.assertEqual(result['R@1']['regressed'], ['d'])
        self.assertEqual(result['R@5']['repaired'], ['b', 'c'])
        self.assertEqual(result['R@10']['repaired'], ['c'])
        self.assertEqual(result['R@10']['regressed'], ['d'])

    def test_invalid_budget_duplicates_and_unbounded_skip_refused(self):
        for extra in (['--arms', 'T0', 'T0'], ['--max-gib', '9'], ['--max-steps', '-1'],
                      ['--high-gib', '8'], ['--reserve-gib', '0'], ['--skip-evaluation']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.arguments(*extra)
        args = self.arguments('--max-steps', '3', '--skip-evaluation')
        self.assertTrue(args.skip_evaluation)

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'existing'
            marker.write_text('preserve')
            with self.assertRaises(SystemExit):
                RUNNER.main(['--run-root', temporary, '--plan-only'])
            self.assertEqual(marker.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()

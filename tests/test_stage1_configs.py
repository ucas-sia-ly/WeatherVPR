"""Check stage-one shell commands and validation without constructing a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from weather_vpr.training.train import parse_args, _validate


class StageOneConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = Path(__file__).resolve().parents[1]
        self.capture = self.directory / 'arguments.json'
        self.stub = self.directory / 'python-stub'
        self.stub.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["ARGUMENT_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
        self.stub.chmod(0o755)
        scripts = ''.join((self.root / 'configs' / name).read_text() for name in (
            'night_real_stage1.sh', 'night_specialist.sh'))
        variables = set(re.findall(r'\$\{([A-Z][A-Z0-9_]*)', scripts))
        self.environment = {key: value for key, value in os.environ.items() if key not in variables}
        self.environment.update(PYTHON=str(self.stub), ARGUMENT_CAPTURE=str(self.capture))

    def invoke(self, arm=None, **overrides):
        self.capture.unlink(missing_ok=True)
        environment = dict(self.environment, **overrides)
        if arm is not None:
            environment['STAGE1_ARM'] = arm
        result = subprocess.run(['bash', str(self.root / 'configs/night_real_stage1.sh')],
                                cwd=self.root, env=environment, capture_output=True, text=True)
        arguments = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return result, arguments

    def value(self, arguments, flag):
        self.assertIn(flag, arguments)
        return arguments[arguments.index(flag) + 1]

    def test_all_arms_preserve_main_defaults_and_select_only_the_paired_branch(self):
        for arm, mode, weight in (('A', 'none', '0'), ('B', 'real', '0.1'),
                                  ('C', 'synthetic', '0.1'), ('D', 'real', '0.03')):
            with self.subTest(arm=arm):
                overrides = {'REAL_PAIRS_REVIEW': 'reviewed/train.jsonl'} if mode == 'real' else {}
                process, arguments = self.invoke(arm, **overrides)
                self.assertEqual(process.returncode, 0, process.stderr)
                expected = {
                    '--checkpoint': 'weights/dino_salad.ckpt', '--weather': 'night',
                    '--adapter-type': 'full', '--full-train-scope': 'late',
                    '--full-train-blocks': '4', '--batch-size': '8',
                    '--gradient-accumulation-steps': '1', '--source-night-samples': '2',
                    '--source-night-weight': '0.5', '--source-night-config': 'configs/source_night.json',
                    '--global-weight': '0.1', '--contrast-weight': '0.1', '--relation-weight': '0.1',
                    '--night-augmentation': 'camera', '--augmentation-config': 'configs/night_camera.json',
                    '--fixed-day-bank': 'runs/cache/qwen_training_day.safetensors',
                    '--layer-decay': '1', '--num-workers': '0', '--seed': '42',
                    '--auxiliary-mode': mode, '--auxiliary-weight': weight,
                    '--auxiliary-samples': '2', '--auxiliary-gradient-every': '0' if arm == 'A' else '100',
                    '--output-dir': f'runs/night_real_stage1_{arm}_seed42',
                }
                for flag, value in expected.items():
                    self.assertEqual(self.value(arguments, flag), value, flag)
                for flag in ('--patch-weight', '--keep-weight', '--ot-weight', '--magnitude-weight',
                             '--attention-weight', '--synthetic-guidance-weight'):
                    self.assertEqual(self.value(arguments, flag), '0', flag)
                self.assertNotIn('--init-adapter-dir', arguments)
                for flag in ('--bf16', '--verify-hashes', '--gradient-checkpointing'):
                    self.assertIn(flag, arguments)
                if mode == 'real':
                    self.assertEqual(self.value(arguments, '--real-pairs-root'), 'data/DarkDriving_Base')
                    self.assertEqual(self.value(arguments, '--real-pairs-review'), 'reviewed/train.jsonl')
                    self.assertEqual(self.value(arguments, '--auxiliary-day-cache'),
                                     'runs/cache/stage1_real_day.safetensors')
                else:
                    self.assertNotIn('--real-pairs-root', arguments)
                    self.assertNotIn('--real-pairs-review', arguments)
                    self.assertNotIn('--auxiliary-day-cache', arguments)

    def test_default_arm_is_b_and_explicit_budget_and_paths_are_forwarded(self):
        process, arguments = self.invoke(
            REAL_PAIRS_REVIEW='review folder/train.jsonl', REAL_PAIRS_ROOT='real data',
            AUXILIARY_DAY_CACHE='cache folder/real.safetensors', AUXILIARY_WEIGHT='.07',
            AUXILIARY_SAMPLES='3', AUXILIARY_GRADIENT_EVERY='25', SEED='44', MAX_STEPS='3',
            GRADIENT_ACCUMULATION_STEPS='2', OUTPUT='run folder/stage1')
        self.assertEqual(process.returncode, 0, process.stderr)
        for flag, value in (
                ('--auxiliary-mode', 'real'), ('--auxiliary-weight', '.07'),
                ('--auxiliary-samples', '3'), ('--auxiliary-gradient-every', '25'),
                ('--real-pairs-review', 'review folder/train.jsonl'),
                ('--real-pairs-root', 'real data'),
                ('--auxiliary-day-cache', 'cache folder/real.safetensors'),
                ('--seed', '44'), ('--max-steps', '3'),
                ('--gradient-accumulation-steps', '2'), ('--output-dir', 'run folder/stage1')):
            self.assertEqual(self.value(arguments, flag), value, flag)

    def test_stage_one_forces_late_four_and_disables_unplanned_losses(self):
        process, arguments = self.invoke(
            'C', FULL_TRAIN_SCOPE='all', FULL_TRAIN_BLOCKS='12', PATCH_WEIGHT='1',
            KEEP_WEIGHT='1', OT_WEIGHT='1', MAGNITUDE_WEIGHT='1', ATTENTION_WEIGHT='1',
            SYNTHETIC_GUIDANCE_WEIGHT='1', AUXILIARY_MODE='real',
            AUXILIARY_DAY_CACHE='unused_real_cache.safetensors')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(self.value(arguments, '--full-train-scope'), 'late')
        self.assertEqual(self.value(arguments, '--full-train-blocks'), '4')
        self.assertEqual(self.value(arguments, '--auxiliary-mode'), 'synthetic')
        for flag in ('--patch-weight', '--keep-weight', '--ot-weight', '--magnitude-weight',
                     '--attention-weight', '--synthetic-guidance-weight'):
            self.assertEqual(self.value(arguments, flag), '0')
        self.assertNotIn('--auxiliary-day-cache', arguments)

    def test_continued_night_initialization_is_rejected_for_every_arm(self):
        for arm in ('A', 'B', 'C', 'D'):
            with self.subTest(arm=arm):
                result, arguments = self.invoke(arm, INIT_ADAPTER_DIR='historical_night',
                                                REAL_PAIRS_REVIEW='review.jsonl')
                self.assertEqual(result.returncode, 2)
                self.assertIn('original Day checkpoint', result.stderr)
                self.assertIsNone(arguments)

    def test_real_arms_without_review_fail_before_starting_python(self):
        for arm in ('B', 'D'):
            with self.subTest(arm=arm):
                result, arguments = self.invoke(arm)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('REAL_PAIRS_REVIEW', result.stderr)
                self.assertIsNone(arguments)
        result, arguments = self.invoke('unknown')
        self.assertEqual(result.returncode, 2)
        self.assertIn('STAGE1_ARM', result.stderr)
        self.assertIsNone(arguments)


class StageOneArgumentTests(unittest.TestCase):
    def arguments(self, *extra):
        return parse_args(['--weather', 'night', '--device', 'cpu', '--patch-weight', '0', *extra])

    def test_default_is_true_bypass_and_active_modes_keep_full_late_four(self):
        default = self.arguments()
        _validate(default)
        self.assertEqual((default.auxiliary_mode, default.auxiliary_weight, default.auxiliary_samples),
                         ('none', .1, 2))
        self.assertEqual(default.auxiliary_gradient_every, 0)
        for extra in (['--auxiliary-mode', 'synthetic'],
                      ['--auxiliary-mode', 'real', '--real-pairs-root', 'real',
                       '--real-pairs-review', 'review.jsonl']):
            args = self.arguments(*extra)
            _validate(args)
            self.assertEqual((args.adapter_type, args.full_train_scope, args.full_train_blocks),
                             ('full', 'late', 4))

    def test_active_auxiliary_requires_full_night_and_complete_review_arguments(self):
        invalid = (
            ['--auxiliary-mode', 'real'],
            ['--auxiliary-mode', 'real', '--real-pairs-root', 'real'],
            ['--auxiliary-mode', 'real', '--real-pairs-review', 'review.jsonl'],
            ['--auxiliary-mode', 'synthetic', '--adapter-type', 'output'],
            ['--auxiliary-mode', 'synthetic', '--adapter-type', 'lora'],
            ['--auxiliary-mode', 'synthetic', '--weather', 'rain'],
            ['--auxiliary-mode', 'synthetic', '--fixed-day-bank', 'same.safetensors',
             '--auxiliary-day-cache', './same.safetensors'],
        )
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _validate(self.arguments(*extra))

    def test_weights_counts_and_diagnostics_validate_without_reading_files(self):
        for flag, value in (('--auxiliary-weight', '-1'), ('--auxiliary-weight', 'nan'),
                            ('--auxiliary-weight', 'inf'), ('--auxiliary-samples', '0'),
                            ('--auxiliary-samples', '-2'), ('--auxiliary-gradient-every', '-1'),
                            ('--auxiliary-gradient-every', '1')):
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                _validate(self.arguments(flag, value))
        # These paths intentionally do not exist: a disabled mode never needs
        # source data, and preflight (rather than CLI parsing) reads active data.
        args = self.arguments('--auxiliary-mode', 'real', '--auxiliary-weight', '0',
                              '--real-pairs-root', '/MISSING_REAL_ROOT',
                              '--real-pairs-review', '/MISSING_REVIEW.jsonl')
        _validate(args)
        active = self.arguments('--auxiliary-mode', 'synthetic', '--auxiliary-gradient-every', '10')
        _validate(active)


if __name__ == '__main__':
    unittest.main()

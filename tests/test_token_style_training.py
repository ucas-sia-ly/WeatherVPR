"""Exercise call-local Token style integration without loading model weights.

The ordinary serial-training fixtures keep Teacher, Source and auxiliary paths
visible. The small wrapper supplies Patch tokens only when the real training
loop explicitly passes a callback, so these checks detect branch leakage.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

import test_auxiliary_training as fixtures


_ORIGINAL_STUDENT = fixtures._TinyNet.student


def _style_aware_student(self, images, weather, return_aux=False, **kwargs):
    kind = int(images[0, 0])
    if not hasattr(self, 'raw_student_calls'):
        self.raw_student_calls = []
        self.token_callback_calls = []
    self.raw_student_calls.append((kind, images.detach().clone(), dict(kwargs)))
    if kwargs:
        if set(kwargs) != {'token_transform', 'token_transform_after_block'}:
            raise AssertionError(f'Unexpected Student keywords: {set(kwargs)}')
        if kind != 1:
            raise AssertionError('Token style leaked outside the main Qwen branch')
        if kwargs['token_transform_after_block'] != 7:
            raise AssertionError('Token style was inserted at a different layer')
        # Mean-zero spatial patterns retain the mock image descriptor before
        # stylization, while different image statistics give meaningful donors.
        feature = images[:, 2:]
        pattern = images.new_tensor([-1., -.5, .5, 1.])[None, :, None]
        scale = .3 + feature.abs()[:, None, :]
        patches = feature[:, None, :] + pattern * scale
        tokens = torch.cat((feature[:, None, :], patches), dim=1)
        transformed = kwargs['token_transform'](tokens)
        self.token_callback_calls.append((kind, tokens.detach().clone(),
                                          transformed.detach().clone()))
        # Keep the descriptor differentiable in the same trainable projection.
        # The mock images themselves are not trainable.
        images = torch.cat((images[:, :2], transformed[:, 1:].mean(dim=1)), dim=1)
    return _ORIGINAL_STUDENT(self, images, weather, return_aux=return_aux)


class TokenStyleTrainingTests(unittest.TestCase):
    # Reuse fixtures without inheriting/re-running their unrelated test suite.
    setUp = fixtures.AuxiliaryTrainingTests.setUp
    tearDown = fixtures.AuxiliaryTrainingTests.tearDown
    _run_serial_fixture = fixtures.AuxiliaryTrainingTests.run_tiny

    def run_tiny(self, mode='none', *, accumulation=2, extra=()):
        with patch.object(fixtures._TinyNet, 'student', new=_style_aware_student):
            return self._run_serial_fixture(mode, accumulation=accumulation, extra=extra)

    @staticmethod
    def enabled_options():
        # Force application for deterministic branch coverage, rather than
        # expecting a particular probability draw in a three-batch fixture.
        return ('--token-style-probability', '1', '--token-style-fraction', '1',
                '--token-style-strength', '.5', '--token-style-beta', '1')

    def assert_same_weights(self, first, second):
        expected = load_file(str(first['output'] / 'adapter.safetensors'))
        actual = load_file(str(second['output'] / 'adapter.safetensors'))
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def assert_same_raw_calls(self, first, second, *, kinds=None):
        def calls(result):
            return [(kind, image) for kind, image, _ in result['net'].raw_student_calls
                    if kinds is None or kind in kinds]
        expected, actual = calls(first), calls(second)
        self.assertEqual(len(actual), len(expected))
        for (kind, image), (target_kind, target) in zip(actual, expected):
            self.assertEqual(kind, target_kind)
            torch.testing.assert_close(image, target, rtol=0, atol=0)

    def test_off_switches_are_exact_training_identity_and_never_pass_callback(self):
        baseline = self.run_tiny('real')
        for switch in ('probability', 'fraction', 'strength'):
            with self.subTest(switch=switch):
                options = self.enabled_options() + (f'--token-style-{switch}', '0')
                result = self.run_tiny('real', extra=options)
                self.assert_same_weights(baseline, result)
                self.assert_same_raw_calls(baseline, result)
                self.assertEqual(result['net'].draws, baseline['net'].draws)
                self.assertFalse(result['net'].token_callback_calls)
                self.assertTrue(all(not kwargs for _, _, kwargs in result['net'].raw_student_calls))
                self.assertEqual(result['metadata']['main_stream_sha256'],
                                 baseline['metadata']['main_stream_sha256'])

    def test_active_transform_is_qwen_only_and_preserves_source_and_real_streams(self):
        baseline = self.run_tiny('real')
        result = self.run_tiny('real', extra=self.enabled_options())
        self.assert_same_raw_calls(baseline, result)
        self.assertEqual(result['net'].draws, baseline['net'].draws)
        self.assertEqual(result['metadata']['main_stream_sha256'],
                         baseline['metadata']['main_stream_sha256'])
        self.assertEqual(result['metadata']['student_views'], baseline['metadata']['student_views'])
        self.assertEqual(result['metadata']['optimizer_steps'], 2)
        self.assertEqual([row['auxiliary_sample_ids'] for row in result['log']],
                         [row['auxiliary_sample_ids'] for row in baseline['log']])
        self.assertEqual(len(result['net'].token_callback_calls), 3)
        self.assertTrue(any(not torch.equal(before, after)
                            for _, before, after in result['net'].token_callback_calls))
        for kind, before, after in result['net'].token_callback_calls:
            self.assertEqual(kind, 1)
            torch.testing.assert_close(before[:, 0], after[:, 0], rtol=0, atol=0)
        for kind, _, kwargs in result['net'].raw_student_calls:
            self.assertEqual(bool(kwargs), kind == 1)
        saved = load_file(str(result['output'] / 'adapter.safetensors'))
        baseline_saved = load_file(str(baseline['output'] / 'adapter.safetensors'))
        self.assertTrue(any(not torch.equal(saved[name], baseline_saved[name]) for name in saved))
        for name, value in result['net'].base.state_dict().items():
            torch.testing.assert_close(value, result['net'].initial_base[name], rtol=0, atol=0)
        self.assertFalse(any(parameter.grad is not None for parameter in result['net'].base.parameters()))

    def test_style_randomness_is_identical_with_and_without_extra_real_branch(self):
        synthetic_only = self.run_tiny('none', extra=self.enabled_options())
        with_real = self.run_tiny('real', extra=self.enabled_options())
        # Extra passes cannot change the Token mixing choices or main draws.
        expected = synthetic_only['net'].token_callback_calls
        actual = with_real['net'].token_callback_calls
        for (_, before, after), (_, target_before, target_after) in zip(actual, expected):
            torch.testing.assert_close(before, target_before, rtol=0, atol=0)
            torch.testing.assert_close(after, target_after, rtol=0, atol=0)
        self.assert_same_raw_calls(synthetic_only, with_real, kinds={1, 2})
        self.assertEqual([draw for draw in with_real['net'].draws if draw[0] != 3],
                         synthetic_only['net'].draws)

    def test_configuration_per_step_diagnostics_and_plain_checkpoint_are_persisted(self):
        result = self.run_tiny('real', extra=self.enabled_options()
                               + ('--augmentation-epoch-offset', '3'))
        output = result['output']
        metadata = result['metadata']
        self.assertIn('token_style', metadata)
        self.assertIn('token_style', json.loads((output / 'experiment_plan.json').read_text()))
        self.assertTrue(all('token_style' in row for row in result['log']))
        for step, row in enumerate(result['log']):
            record = row['token_style']
            self.assertEqual(record['after_block'], 7)
            self.assertEqual(record['epoch'], 3)
            self.assertEqual(record['step'], step)
            self.assertEqual(record['applied_images'], 2)
            self.assertEqual(record['applied_patches'], 8)
        saved = load_file(str(output / 'adapter.safetensors'))
        self.assertEqual(set(saved), set(result['net'].adapters['night'].state_dict()))
        # The standard plain model restore needs no stylization architecture.
        restored = fixtures._TinyNet(fixtures._Base(), ('night',))
        args = fixtures.train.parse_args(['--weather', 'night', '--device', 'cpu',
                                         '--init-adapter-dir', str(output)])
        fixtures.train._load_initial_adapter(restored, args, metadata['checkpoint_sha256'],
                                             restored.base, metadata['preprocessing'])
        for name, value in restored.adapters['night'].state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

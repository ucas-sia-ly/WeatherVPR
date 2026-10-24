"""Auxiliary anchor weights are scheduled by optimizer updates, not batches."""
import unittest

from weather_vpr.training.loss_schedule import auxiliary_weight_multiplier
from weather_vpr.training.train import parse_args, _validate


class DayAnchorTests(unittest.TestCase):
    def test_linear_exact_endpoints_and_constant_legacy_behavior(self):
        values = [auxiliary_weight_multiplier('linear', step, 5, .2) for step in range(5)]
        self.assertEqual(values[0], 1.)
        self.assertAlmostEqual(values[-1], .2)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual([auxiliary_weight_multiplier('constant', i, 5) for i in range(5)], [1.] * 5)
        self.assertEqual(auxiliary_weight_multiplier('linear', 0, 1), 1.)
        self.assertEqual(auxiliary_weight_multiplier('linear', 4, 5), 0.)

    def test_accumulation_window_reuses_same_update_weight(self):
        # Two microbatches per optimizer update must not decay twice as fast.
        values = [auxiliary_weight_multiplier('linear', microbatch // 2, 3)
                  for microbatch in range(6)]
        self.assertEqual(values, [1., 1., .5, .5, 0., 0.])

    def test_invalid_weights_and_incompatible_adapter_scope(self):
        for schedule, step, total, ratio in (('bad', 0, 3, 0), ('linear', -1, 3, 0),
                ('linear', 3, 3, 0), ('linear', 0, 0, 0), ('linear', True, 3, 0),
                ('linear', 0, 3, float('nan')), ('linear', 0, 3, -1)):
            with self.subTest(schedule=schedule, step=step, total=total, ratio=ratio):
                with self.assertRaises(ValueError):
                    auxiliary_weight_multiplier(schedule, step, total, ratio)
        for arguments in (['--adapter-type', 'lora', '--full-train-scope', 'early'],
                          ['--full-train-blocks', '0'], ['--day-preservation-final-ratio', '1.1']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _validate(parse_args(['--weather', 'night', '--device', 'cpu'] + arguments))


if __name__ == '__main__':
    unittest.main()

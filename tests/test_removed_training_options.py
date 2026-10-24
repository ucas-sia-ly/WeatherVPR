"""Retired Day Student regularizers cannot silently re-enter training."""
from contextlib import redirect_stderr
import io
import unittest

from weather_vpr.training.train import parse_args


class RemovedTrainingOptionsTests(unittest.TestCase):
    def test_day_preservation_and_restoration_options_are_rejected(self):
        for option, value in (
                ('--day-preservation-weight', '.03'),
                ('--day-preservation-samples', '2'),
                ('--day-preservation-schedule', 'linear'),
                ('--day-preservation-final-ratio', '0'),
                ('--day-view-augmentation', 'camera')):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(['--weather', 'night', option, value])
                self.assertEqual(error.exception.code, 2)

    def test_parsed_configuration_has_no_retired_day_options(self):
        args = vars(parse_args(['--weather', 'night', '--device', 'cpu']))
        self.assertFalse(any(name.startswith('day_preservation') for name in args))
        self.assertNotIn('day_view_augmentation', args)


if __name__ == '__main__':
    unittest.main()

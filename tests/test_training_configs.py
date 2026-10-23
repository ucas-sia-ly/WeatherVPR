"""Check that shell training entry points forward optional model/precision flags."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest


class TrainingConfigTests(unittest.TestCase):
    def test_camera_options_forwarded_in_all_full_training_stages(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            capture = temp / 'arguments.json'
            stub = temp / 'python-stub'
            stub.write_text('#!/usr/bin/env python3\n'
                            'import json, os, sys\n'
                            'from pathlib import Path\n'
                            'Path(os.environ["ARGUMENT_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=str(stub), ARGUMENT_CAPTURE=str(capture),
                       NIGHT_AUGMENTATION='camera', AUGMENTATION_STRENGTH='0.7',
                       AUGMENTATION_PROBABILITY='0.9', AUGMENTATION_CONFIG='ranges.json',
                       AUGMENTATION_EPOCH_OFFSET='3')
            for stage in ('global', 'sgpd'):
                subprocess.run(['bash', str(root / f'configs/night_full_{stage}.sh')],
                               cwd=root, env=env, check=True, capture_output=True)
                arguments = json.loads(capture.read_text())
                for flag, value in (('--night-augmentation', 'camera'),
                                    ('--augmentation-strength', '0.7'),
                                    ('--augmentation-probability', '0.9'),
                                    ('--augmentation-config', 'ranges.json'),
                                    ('--augmentation-epoch-offset', '3')):
                    self.assertEqual(arguments[arguments.index(flag) + 1], value)


if __name__ == '__main__':
    unittest.main()

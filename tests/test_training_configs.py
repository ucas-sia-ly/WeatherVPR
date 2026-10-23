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
            for stage in ('global', 'sgpd', 'real'):
                subprocess.run(['bash', str(root / f'configs/night_full_{stage}.sh')],
                               cwd=root, env=env, check=True, capture_output=True)
                arguments = json.loads(capture.read_text())
                for flag, value in (('--night-augmentation', 'camera'),
                                    ('--augmentation-strength', '0.7'),
                                    ('--augmentation-probability', '0.9'),
                                    ('--augmentation-config', 'ranges.json'),
                                    ('--augmentation-epoch-offset', '3')):
                    self.assertEqual(arguments[arguments.index(flag) + 1], value)

    def test_real_night_config_forwards_initial_checkpoint_and_bf16(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            capture = temp / 'arguments.json'
            stub = temp / 'python-stub'
            stub.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'Path(os.environ["ARGUMENT_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=str(stub), ARGUMENT_CAPTURE=str(capture),
                       INIT_ADAPTER_DIR='runs/test-initial-night', BF16='1')
            subprocess.run(['bash', str(root / 'configs/night_full_real.sh')],
                           cwd=root, env=env, check=True, capture_output=True)
            arguments = json.loads(capture.read_text())
            self.assertIn('--bf16', arguments)
            position = arguments.index('--init-adapter-dir')
            self.assertEqual(arguments[position + 1], 'runs/test-initial-night')
            env.update(BF16='0', INIT_ADAPTER_DIR='')
            subprocess.run(['bash', str(root / 'configs/night_full_real.sh')],
                           cwd=root, env=env, check=True, capture_output=True)
            arguments = json.loads(capture.read_text())
            self.assertNotIn('--bf16', arguments)
            self.assertNotIn('--init-adapter-dir', arguments)


if __name__ == '__main__':
    unittest.main()

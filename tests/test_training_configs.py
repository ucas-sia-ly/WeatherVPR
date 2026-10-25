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
            env = {key: value for key, value in os.environ.items()
                   if key not in {'FULL_TRAIN_SCOPE', 'FULL_TRAIN_BLOCKS', 'NUM_WORKERS'}}
            env.update(PYTHON=str(stub), ARGUMENT_CAPTURE=str(capture),
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
                self.assertEqual(arguments[arguments.index('--full-train-scope') + 1], 'all')
                self.assertEqual(arguments[arguments.index('--num-workers') + 1], '0')

    def test_specialist_uses_late_four_blocks_and_supports_explicit_all_ablation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / 'arguments.json'
            stub = Path(directory) / 'python-stub'
            stub.write_text('#!/usr/bin/env python3\n'
                            'import json, os, sys\n'
                            'from pathlib import Path\n'
                            'Path(os.environ["ARGUMENT_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if key not in {'FULL_TRAIN_SCOPE', 'FULL_TRAIN_BLOCKS', 'LAYER_DECAY',
                                  'NUM_WORKERS', 'OUTPUT'}}
            env.update(PYTHON=str(stub), ARGUMENT_CAPTURE=str(capture))
            for overrides, scope, count in (({}, 'late', '4'),
                                           ({'FULL_TRAIN_SCOPE': 'all', 'FULL_TRAIN_BLOCKS': '12'}, 'all', '12')):
                subprocess.run(['bash', str(root / 'configs/night_specialist.sh')],
                               cwd=root, env=dict(env, **overrides), check=True, capture_output=True)
                arguments = json.loads(capture.read_text())
                for flag, value in (('--full-train-scope', scope), ('--full-train-blocks', count),
                                    ('--layer-decay', '1'), ('--num-workers', '0')):
                    self.assertEqual(arguments[arguments.index(flag) + 1], value)
                self.assertFalse(any(flag.startswith('--day-preservation') or flag == '--day-view-augmentation'
                                     for flag in arguments))

    def test_patch_camera_policy_cannot_skip_all_operations(self):
        root = Path(__file__).resolve().parents[1]
        policy = json.loads((root / 'configs/night_patch_camera.json').read_text())
        self.assertEqual(policy['probability'], 1)
        self.assertEqual(policy['operation_probabilities']['exposure'], 1)


if __name__ == '__main__':
    unittest.main()

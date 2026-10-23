"""Student-only degradation, clean correspondence input, and worker-independent seeds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from weather_vpr.utils.backend import enable_backend_imports
enable_backend_imports()
from workflow.reliability_cache import preprocess_teacher_image
from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs
from weather_vpr.data.student_augmentation import sample_augmentation_seed
from weather_vpr.training.augmentation import load_augmentation_config
from weather_vpr.training.train import parse_args as parse_pair_args


def _camera_config():
    # Probability one ensures a changed view for this consistency test.
    return NightAugmentationConfig(profile='camera', probability=1.0,
                                   operation_probabilities={'exposure': 1.0})


class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        pixels = np.random.default_rng(7).integers(20, 235, size=(270, 340, 3), dtype=np.uint8)
        self.day = self.root / 'day.png'
        self.night = self.root / 'night.png'
        Image.fromarray(pixels).save(self.day)
        Image.fromarray((pixels * 0.45).astype(np.uint8)).save(self.night)
        self.manifest = self.root / 'pairs.jsonl'
        self.manifest.write_text('\n'.join(json.dumps({
            'condition': 'night', 'passed': True, 'eligible_for_training': True,
            'city': 'city', 'place_id': str(index),
            'source_path': str(self.day), 'output_path': str(self.night),
        }) for index in range(3)), encoding='utf-8')

    def tearDown(self):
        self.directory.cleanup()

    def test_paired_teacher_and_correspondence_views_remain_clean(self):
        dataset = AcceptedWeatherPairs(self.manifest, 'night',
                                        night_augmentation=_camera_config(), seed=24)
        sample = dataset[0]
        with Image.open(self.day) as image:
            expected_day = preprocess_teacher_image(image, (224, 224))
        with Image.open(self.night) as image:
            expected_night = preprocess_teacher_image(image, (224, 224))
            original_pixels = np.asarray(image).copy()
        torch.testing.assert_close(sample['day_tensor'], expected_day, rtol=0, atol=0)
        torch.testing.assert_close(sample['weather_tensor'], expected_night, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(sample['weather_image']), original_pixels)
        self.assertFalse(torch.equal(sample['student_tensor'], sample['weather_tensor']))
        self.assertEqual(sample['student_tensor'].shape, sample['weather_tensor'].shape)
        self.assertTrue(sample['augmentation_record']['applied'])

    def test_epoch_seed_is_reproducible_across_worker_counts(self):
        dataset = AcceptedWeatherPairs(self.manifest, 'night',
                                        night_augmentation=_camera_config(), seed=24,
                                        day_augmentation=_camera_config())
        def read(workers):
            loader = DataLoader(dataset, batch_size=2, num_workers=workers,
                                collate_fn=list, shuffle=False)
            return [sample for batch in loader for sample in batch]
        dataset.set_epoch(3)
        serial = read(0)
        parallel = read(2)
        for one, two in zip(serial, parallel):
            self.assertEqual(one['augmentation_record'], two['augmentation_record'])
            torch.testing.assert_close(one['student_tensor'], two['student_tensor'], rtol=0, atol=0)
            self.assertEqual(one['day_augmentation_record'], two['day_augmentation_record'])
            torch.testing.assert_close(one['day_student_tensor'], two['day_student_tensor'],
                                       rtol=0, atol=0)
            self.assertNotEqual(one['augmentation_record']['seed'],
                                one['day_augmentation_record']['seed'])
        dataset.set_epoch(4)
        next_epoch = read(2)
        self.assertTrue(all(one['augmentation_record']['seed'] != two['augmentation_record']['seed']
                            for one, two in zip(serial, next_epoch)))
        self.assertTrue(all(one['day_augmentation_record']['seed'] != two['day_augmentation_record']['seed']
                            for one, two in zip(serial, next_epoch)))
        self.assertNotEqual(sample_augmentation_seed(24, 3, 0), sample_augmentation_seed(24, 3, 1))

    def test_off_profile_preserves_old_student_preprocessing(self):
        dataset = AcceptedWeatherPairs(self.manifest, 'night')
        sample = dataset[0]
        torch.testing.assert_close(sample['student_tensor'], sample['weather_tensor'], rtol=0, atol=0)
        self.assertFalse(sample['augmentation_record']['applied'])
        self.assertIs(sample['day_student_tensor'], sample['day_tensor'])
        self.assertEqual(sample['day_augmentation_record']['profile'], 'off')
        self.assertFalse(sample['day_augmentation_record']['applied'])

    def test_day_student_augmentation_leaves_all_clean_and_weather_views_unchanged(self):
        reference = AcceptedWeatherPairs(self.manifest, 'night',
                                         night_augmentation=_camera_config(), seed=24)[0]
        augmented = AcceptedWeatherPairs(self.manifest, 'night',
                                         night_augmentation=_camera_config(), seed=24,
                                         day_augmentation=_camera_config())[0]
        for key in ('day_tensor', 'weather_tensor', 'student_tensor'):
            torch.testing.assert_close(augmented[key], reference[key], rtol=0, atol=0)
        for key in ('day_image', 'weather_image'):
            np.testing.assert_array_equal(np.asarray(augmented[key]), np.asarray(reference[key]))
        self.assertEqual(augmented['augmentation_record'], reference['augmentation_record'])
        self.assertEqual(augmented['day_student_tensor'].shape, augmented['day_tensor'].shape)
        self.assertFalse(torch.equal(augmented['day_student_tensor'], augmented['day_tensor']))
        self.assertTrue(augmented['day_augmentation_record']['applied'])
        self.assertIsNot(augmented['day_image'], augmented['weather_image'])
        with Image.open(self.day) as image:
            np.testing.assert_array_equal(np.asarray(augmented['day_image']), np.asarray(image))
            torch.testing.assert_close(augmented['day_tensor'],
                                       preprocess_teacher_image(image, (224, 224)), rtol=0, atol=0)

    def test_explicit_day_off_preserves_exact_clean_tensor(self):
        sample = AcceptedWeatherPairs(self.manifest, 'night',
                                       night_augmentation=_camera_config(),
                                       day_augmentation=NightAugmentationConfig(profile='off'))[0]
        self.assertIs(sample['day_student_tensor'], sample['day_tensor'])
        self.assertEqual(sample['day_augmentation_record']['profile'], 'off')
        self.assertEqual(sample['day_augmentation_record']['operations'], {})
        self.assertFalse(sample['day_augmentation_record']['applied'])

    def test_weather_seed_history_and_day_domain_separation(self):
        expected_weather = int.from_bytes(hashlib.sha256(b'24:3:0').digest()[:8], 'big')
        expected_day = int.from_bytes(hashlib.sha256(b'24:3:0:day').digest()[:8], 'big')
        self.assertEqual(sample_augmentation_seed(24, 3, 0), expected_weather)
        self.assertEqual(sample_augmentation_seed(24, 3, 0, view='weather'), expected_weather)
        self.assertEqual(sample_augmentation_seed(24, 3, 0, view='day'), expected_day)
        self.assertNotEqual(expected_day, expected_weather)
        for view in ('night', '', None):
            with self.subTest(view=view), self.assertRaisesRegex(ValueError, 'view'):
                sample_augmentation_seed(24, 3, 0, view=view)

    def test_paired_cli_config_and_epoch_offset(self):
        config = self.root / 'augmentation.json'
        config.write_text(json.dumps({'profile': 'off', 'strength': 0.1,
                                      'operation_probabilities': {'jpeg': 0.25}}))
        flags = ['--night-augmentation', 'camera', '--augmentation-strength', '0.7',
                 '--augmentation-probability', '0.9', '--augmentation-epoch-offset', '2',
                 '--augmentation-config', str(config), '--device', 'cpu']
        args = parse_pair_args(['--weather', 'night', *flags])
        resolved = load_augmentation_config(args)
        self.assertEqual(resolved.profile, 'camera')
        self.assertEqual(resolved.strength, 0.7)
        self.assertEqual(resolved.probability, 0.9)
        self.assertEqual(resolved.operation_probabilities['jpeg'], 0.25)
        self.assertEqual(args.augmentation_epoch_offset, 2)


if __name__ == '__main__':
    unittest.main()

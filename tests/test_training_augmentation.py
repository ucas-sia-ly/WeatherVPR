"""Student-only degradation, clean correspondence input, and worker-independent seeds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from weather_vpr.utils.backend import enable_backend_imports
enable_backend_imports()
from workflow.reliability_cache import preprocess_teacher_image
from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs
from weather_vpr.data.source_night import SourceNightConfig
from weather_vpr.data.student_augmentation import (
    SOURCE_NIGHT_SEED_POLICY, sample_augmentation_seed, student_training_tensor,
)
from weather_vpr.training.augmentation import load_augmentation_config
from weather_vpr.training.train import parse_args as parse_pair_args


def _camera_config():
    # Probability one ensures a changed view for this consistency test.
    return NightAugmentationConfig(profile='camera', probability=1.0,
                                   operation_probabilities={'exposure': 1.0})


def _source_night_config():
    # Camera corruptions can be disabled without disabling explicit darkening.
    return SourceNightConfig(camera_probability=0.0, illumination_probability=0.0,
                             white_balance_probability=0.0,
                             gain_range=(0.1, 0.1), gamma_range=(1.0, 1.0))


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
                                        source_night_config=_source_night_config())
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
            self.assertEqual(one['source_night_record'], two['source_night_record'])
            torch.testing.assert_close(one['source_night_tensor'], two['source_night_tensor'],
                                       rtol=0, atol=0)
            self.assertEqual(one['source_night_record']['seed_domain'], 'source_night')
            self.assertEqual(one['source_night_record']['seed_policy'], SOURCE_NIGHT_SEED_POLICY)
            self.assertNotEqual(one['source_night_record']['seed'],
                                one['augmentation_record']['seed'])
        dataset.set_epoch(4)
        next_epoch = read(2)
        self.assertTrue(all(one['augmentation_record']['seed'] != two['augmentation_record']['seed']
                            for one, two in zip(serial, next_epoch)))
        self.assertTrue(all(one['source_night_record']['seed'] != two['source_night_record']['seed']
                            for one, two in zip(serial, next_epoch)))
        self.assertNotEqual(sample_augmentation_seed(24, 3, 0), sample_augmentation_seed(24, 3, 1))

    def test_off_profile_preserves_old_student_preprocessing(self):
        dataset = AcceptedWeatherPairs(self.manifest, 'night')
        sample = dataset[0]
        torch.testing.assert_close(sample['student_tensor'], sample['weather_tensor'], rtol=0, atol=0)
        self.assertFalse(sample['augmentation_record']['applied'])
        self.assertNotIn('day_student_tensor', sample)
        self.assertNotIn('day_augmentation_record', sample)
        self.assertIsNone(sample['source_night_tensor'])
        self.assertIsNone(sample['source_night_record'])

    def test_disabled_source_night_has_no_transform_or_random_stream(self):
        with patch('weather_vpr.data.source_night.SourceNightAugmentation') as transform, \
                patch('weather_vpr.data.paired_dataset.sample_augmentation_seed',
                      wraps=sample_augmentation_seed) as seed, \
                patch('weather_vpr.data.paired_dataset.student_training_tensor',
                      wraps=student_training_tensor) as tensor:
            sample = AcceptedWeatherPairs(self.manifest, 'night', seed=24,
                                            source_night_config=None)[0]
        transform.assert_not_called()
        self.assertEqual(tensor.call_count, 1)  # Historical Qwen Student view only.
        self.assertEqual(seed.call_count, 1)  # Qwen Student weather view only.
        self.assertFalse(any(call.kwargs.get('view') == 'source_night'
                             for call in seed.call_args_list))
        self.assertIsNone(sample['source_night_tensor'])
        self.assertIsNone(sample['source_night_record'])

    def test_source_night_is_dark_without_camera_and_preserves_existing_views(self):
        options = dict(night_augmentation=_camera_config(), seed=24)
        reference = AcceptedWeatherPairs(self.manifest, 'night', **options)[0]
        dataset = AcceptedWeatherPairs(self.manifest, 'night', **options,
                                        source_night_config=_source_night_config())
        original_transform = dataset.source_night_augmentation
        with patch.object(dataset, 'source_night_augmentation',
                          wraps=original_transform) as transform:
            sample = dataset[0]
        self.assertEqual(transform.call_count, 1)
        self.assertEqual(transform.call_args.args[0].size, (224, 224))
        self.assertEqual(sample['source_night_tensor'].shape, sample['day_tensor'].shape)
        self.assertTrue(torch.isfinite(sample['source_night_tensor']).all())
        self.assertTrue((sample['source_night_tensor'] < sample['day_tensor']).all())
        self.assertTrue(sample['source_night_record']['applied'])
        self.assertIn('exposure', sample['source_night_record']['operations'])
        self.assertFalse(any(name.startswith('camera_')
                             for name in sample['source_night_record']['operations']))
        self.assertEqual(sample['source_night_record']['seed'],
                         sample_augmentation_seed(24, 0, 0, view='source_night'))
        for key in ('day_tensor', 'weather_tensor', 'student_tensor'):
            torch.testing.assert_close(sample[key], reference[key], rtol=0, atol=0)
        for key in ('day_image', 'weather_image'):
            np.testing.assert_array_equal(np.asarray(sample[key]), np.asarray(reference[key]))
        self.assertEqual(sample['augmentation_record'], reference['augmentation_record'])
        for key, path in (('day_image', self.day), ('weather_image', self.night)):
            with Image.open(path) as image:
                np.testing.assert_array_equal(np.asarray(sample[key]), np.asarray(image))

    def test_weather_seed_history_and_domain_separation(self):
        expected_weather = int.from_bytes(hashlib.sha256(b'24:3:0').digest()[:8], 'big')
        expected_source_night = int.from_bytes(
            hashlib.sha256(b'24:3:0:source_night').digest()[:8], 'big')
        self.assertEqual(sample_augmentation_seed(24, 3, 0), expected_weather)
        self.assertEqual(sample_augmentation_seed(24, 3, 0, view='weather'), expected_weather)
        self.assertEqual(sample_augmentation_seed(24, 3, 0, view='source_night'),
                         expected_source_night)
        self.assertNotEqual(expected_weather, expected_source_night)
        for view in ('day', 'night', '', None):
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

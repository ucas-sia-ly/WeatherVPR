"""Independent Source pool labels, cached-target ordering and RNG isolation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from weather_vpr.data.gsv_day import GSVDayImages, gsv_image_name
from weather_vpr.data.paired_dataset import sha256
from weather_vpr.data.source_night import SourceNightConfig
from weather_vpr.training.source_stream import build_source_stream, prepare_source_dataset


class _Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def teacher(self, images, return_aux=False):
        self.calls += 1
        random.random()
        np.random.random()
        torch.rand(3)
        means = images.mean(dim=(-1, -2))
        return torch.cat((means, torch.ones_like(means[:, :1])), dim=1) * self.scale


class SourceStreamTests(unittest.TestCase):
    def test_original_signed_heading_is_preserved_without_wrapping(self):
        row = dict(city_id='Example', place_id='7', year='2019', month='3',
                   northdeg='-1', lat='1.0', lon='2.0', panoid='pano')
        self.assertEqual(gsv_image_name(row), 'Example_0000007_2019_03_-01_1.0_2.0_pano.jpg')
        for heading in ('-11', '-3', '731'):
            self.assertIn('_' + heading.zfill(3) + '_', gsv_image_name({**row, 'northdeg': heading}))
        for key, value in (('northdeg', '-03'), ('place_id', '-1'), ('year', '02019')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                gsv_image_name({**row, key: value})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / 'gsv' / 'Dataframes' / 'Example.csv'
        self.csv_path.parent.mkdir(parents=True)
        self.bindings = []
        for index in range(8):
            self.bindings.append({
                'place_id': str(index // 2 + 1), 'year': '2020',
                'month': str(index % 2 + 1), 'northdeg': str(index * 10),
                'city_id': 'Example', 'lat': '1.2345', 'lon': '-2.3456',
                'panoid': f'panorama_{index}',
            })
        with self.csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.bindings[0]))
            writer.writeheader()
            writer.writerows(self.bindings)
        self.excluded = self.root / 'excluded.json'
        self.excluded.write_text('[["Example", "99"]]')
        self.rows = []
        for index, binding in enumerate(self.bindings):
            path = self.root / 'gsv' / 'Images' / 'Example' / gsv_image_name(binding)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (23, 19), (30 + index * 20, 120, 200 - index * 15)).save(path)
            place = binding['place_id']
            self.rows.append({
                'schema_version': 1, 'kind': 'gsv_day_train', 'source': 'gsv_day',
                'sample_id': f'gsv_{index}', 'split': 'train', 'path': str(path),
                'city': 'Example', 'place_id': place, 'group_id': f'gsv:Example:{place}',
                'sha256': sha256(path),
                'metadata': {'annotation_csv': str(self.csv_path),
                             'annotation_sha256': sha256(self.csv_path),
                             'annotation_row': index + 1, 'csv_binding': binding,
                             'grouping_kind': 'verified_gsv_place',
                             'place_label_verified': True,
                             'geographic_independence_known': False},
            })
        self.manifest = self.root / 'source.jsonl'
        self.publish(self.rows)
        self.net = _Teacher().train()
        self.preprocessing = {
            'version': 'rgb_bilinear_imagenet_v1', 'image_size': [14, 14],
            'color_space': 'RGB', 'input_range': [0.0, 1.0], 'resize': 'PIL_bilinear',
            'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
        }

    def publish(self, rows, manifest=None):
        manifest = manifest or self.manifest
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        Path(str(manifest) + '.metadata.json').write_text(json.dumps({
            'schema_version': 1, 'kind': 'gsv_day_train',
            'manifest_sha256': sha256(manifest),
            'source_csvs': {str(self.csv_path): sha256(self.csv_path)},
            'exclusion_files': {str(self.excluded): sha256(self.excluded)},
        }))
        return manifest

    def args(self, **kwargs):
        values = {'source_day_manifest': self.manifest, 'source_day_cache': self.root / 'day.safetensors',
                  'source_day_cache_batch_size': 2, 'source_day_cache_workers': 0,
                  'source_day_bank_device': 'cpu', 'source_night_weight': .5,
                  'source_night_samples': 2, 'source_night_config': None,
                  'exclude_places': self.excluded, 'seed': 42, 'device': 'cpu'}
        values.update(kwargs)
        return SimpleNamespace(**values)

    def build(self, **kwargs):
        return build_source_stream(self.args(**kwargs), self.net, 'a' * 64, self.preprocessing)

    def snapshot(self):
        return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()

    def assert_rng(self, snapshot):
        python, numpy, torch_rng = snapshot
        self.assertEqual(random.getstate(), python)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy[0])
        np.testing.assert_array_equal(current[1], numpy[1])
        self.assertEqual(current[2:], numpy[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_disabled_stream_never_opens_manifest_or_config(self):
        missing = self.root / 'missing.jsonl'
        with patch.object(Path, 'read_text', side_effect=AssertionError('disabled I/O')):
            self.assertIsNone(build_source_stream(
                self.args(source_day_manifest=missing, source_night_weight=0,
                          source_night_config=missing), None, 'a' * 64, self.preprocessing))
            self.assertIsNone(prepare_source_dataset(self.args(source_day_manifest=None), (14, 14)))

    def test_day_preprocessing_and_source_policy_match_original_implementation(self):
        dataset = prepare_source_dataset(self.args(), (14, 14))
        from workflow.reliability_cache import preprocess_teacher_image
        from weather_vpr.data.student_augmentation import student_training_tensor
        sample = dataset[3]
        with Image.open(self.rows[3]['path']) as image:
            expected_day = preprocess_teacher_image(image, (14, 14))
            expected_source, _ = student_training_tensor(
                image, (14, 14), dataset.source_night_augmentation,
                seed=sample['source_night_record']['seed'])
        torch.testing.assert_close(sample['day_tensor'], expected_day, rtol=0, atol=0)
        torch.testing.assert_close(sample['source_night_tensor'], expected_source, rtol=0, atol=0)
        self.assertEqual(sample['place'], ('Example', '2'))

    def test_cold_hit_and_batch_leave_main_rng_unchanged_and_repeat_inputs(self):
        before = self.snapshot()
        cold = self.build()
        self.assert_rng(before)
        self.assertEqual(cold.bank_info['cache_status'], 'built')
        calls = self.net.calls
        first = cold.next_batch(0, 0)
        self.assert_rng(before)
        warm = self.build()
        self.assert_rng(before)
        self.assertEqual(warm.bank_info['cache_status'], 'hit')
        self.assertEqual(calls, self.net.calls)
        second = warm.next_batch(0, 0)
        self.assertEqual(first['indices'], second['indices'])
        self.assertEqual(first['augmentation_records'], second['augmentation_records'])
        torch.testing.assert_close(first['images'], second['images'], rtol=0, atol=0)
        with warm.rng_context(0, 0):
            random.random()
            np.random.random()
            torch.rand(10)
        self.assert_rng(before)

    def test_targets_keep_csv_labels_and_index_alignment_and_separate_queue(self):
        stream = self.build()
        batch = stream.next_batch(0, 0)
        self.assertEqual(len(stream.queue), len(self.rows))
        self.assertFalse(stream.queue.include_current_keys)
        torch.testing.assert_close(batch['targets'], stream.targets[batch['indices']], rtol=0, atol=0)
        self.assertEqual(batch['places'], [stream.dataset.rows[i]['place'] for i in batch['indices']])
        self.assertEqual(batch['sample_ids'], [self.rows[i]['sample_id'] for i in batch['indices']])
        result = stream.queue.contrastive_loss(batch['targets'], batch['targets'], batch['places'])
        self.assertEqual(result['positive_pairs'], len(batch['indices']) * 2)
        self.assertEqual(batch['augmentation']['applied_samples'], 2)
        self.assertIsNone(self.net.scale.grad)
        self.assertTrue(self.net.training)

    def test_enlarging_pool_increases_distinct_source_places_at_equal_exposure(self):
        small_path = self.publish(self.rows[:4], self.root / 'small.jsonl')
        small = self.build(source_day_manifest=small_path, source_day_cache=None)
        large = self.build(source_day_cache=None)
        seen_small, seen_large = set(), set()
        for step in range(2):
            seen_small.update(small.next_batch(0, step)['places'])
            seen_large.update(large.next_batch(0, step)['places'])
        self.assertEqual(len(seen_small), 2)
        self.assertEqual(len(seen_large), 4)
        self.assertEqual(small.sampler.draws, large.sampler.draws)

    def test_csv_provenance_labels_split_and_filename_are_enforced(self):
        for field, changed, pattern in (
            ('place_id', '999', 'place label'), ('split', 'test', 'gsv_day/train'),
            ('group_id', 'filename_inferred', 'group_id'),
            ('path', self.rows[1]['path'], 'CSV-backed image'),
        ):
            rows = json.loads(json.dumps(self.rows))
            rows[0][field] = changed
            self.publish(rows)
            with self.assertRaisesRegex(ValueError, pattern):
                prepare_source_dataset(self.args(), (14, 14))
        rows = json.loads(json.dumps(self.rows))
        rows[0]['metadata']['csv_binding']['panoid'] = 'fake'
        self.publish(rows)
        with self.assertRaisesRegex(ValueError, 'csv_binding'):
            prepare_source_dataset(self.args(), (14, 14))

    def test_heldout_or_changed_exclusion_never_silently_enters_source_bank(self):
        self.excluded.write_text('[["Example", "1"]]')
        self.publish(self.rows)
        with self.assertRaisesRegex(ValueError, 'held-out place'):
            self.build()
        self.excluded.write_text('[["Example", "99"]]')
        self.publish(self.rows)
        dataset = prepare_source_dataset(self.args(), (14, 14))
        self.excluded.write_text('[["Example", "1"]]')
        with self.assertRaisesRegex(RuntimeError, 'provenance changed'):
            dataset.source_view(0, seed=42)

    def test_changed_source_or_csv_stops_instead_of_using_stale_targets(self):
        stream = self.build()
        Path(self.rows[0]['path']).write_bytes(b'changed after extraction')
        with self.assertRaisesRegex(RuntimeError, 'image changed'):
            stream.dataset.source_view(0, seed=42)
        self.csv_path.write_text(self.csv_path.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'provenance changed'):
            stream.next_batch(0, 0)

    def test_unsigned_runtime_exclusion_is_rejected(self):
        other = self.root / 'other.json'
        other.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'not signed'):
            prepare_source_dataset(self.args(exclude_places=other), (14, 14))


if __name__ == '__main__':
    unittest.main()

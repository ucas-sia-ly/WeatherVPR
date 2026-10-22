"""Geometry, positive/negative masking, and held-out protocol tests."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from weather_vpr.data.svox_real_night import load_real_night_train, northing_holdout
from weather_vpr.training.real_night import (
    _validate, full_gallery_geographic_infonce,
    night_baseline_descriptor_loss, night_layerwise_optimizer_groups, parse_args,
)


def _image(root: Path, split: str, folder: str, x: float, y: float, tag: str):
    directory = root / 'images' / split / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'@{x:09.2f}@{y:010.2f}@{tag}@.jpg'
    path.write_bytes(b'filename-only geometry fixture')
    return path


class RealNightTests(unittest.TestCase):
    def test_test_geometry_exclusion_and_native_train_positives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i, (x, y) in enumerate(((0, 0), (500, 0), (1000, 1000),
                                        (1500, 1500), (2000, 2000), (2500, 2500))):
                _image(root, 'train', 'gallery', x, y, f'day{i}')
                _image(root, 'train', 'queries_night', x + 5, y, f'night{i}')
            test_gallery = _image(root, 'test', 'gallery', 0, 0, 'testday')
            test_night = _image(root, 'test', 'queries_night', 500, 0, 'testnight')
            data = load_real_night_train(root, test_buffer_meters=250)
            self.assertEqual(len(data.full.queries), 6)
            self.assertEqual(len(data.filtered.queries), 4)
            self.assertEqual(len(data.filtered.references), 4)
            self.assertEqual([len(group) for group in data.filtered.positives], [1, 1, 1, 1])
            self.assertTrue(all('/train/' in str(image.path) for image in
                                data.filtered.references + data.filtered.queries))
            train, dev, buffer = northing_holdout(data.filtered.queries,
                                                  fraction=0.25, gap_meters=50)
            self.assertEqual((len(train), len(dev), len(buffer)), (3, 1, 0))
            # Exclusion is based on names; test image bytes are never decoded.
            test_gallery.write_bytes(b'changed day bytes')
            test_night.write_bytes(b'changed night bytes')
            again = load_real_night_train(root, test_buffer_meters=250)
            self.assertEqual(data.test_geometry_sha256, again.test_geometry_sha256)

    def test_northing_gap_prevents_nearby_query_leakage(self):
        from workflow.evaluation import EvaluationImage
        images = [EvaluationImage(str(i), Path(f'@000000.00@{y:010.2f}@q@.jpg'))
                  for i, y in enumerate((0, 100, 200, 300, 340, 350, 400, 500))]
        train, dev, buffer = northing_holdout(images, fraction=0.25, gap_meters=75)
        self.assertEqual(dev, [6, 7])
        self.assertEqual(train, [0, 1, 2, 3])
        self.assertEqual(buffer, [4, 5])
        self.assertEqual(northing_holdout(images, fraction=0)[0], list(range(8)))

    def test_geographic_multi_positive_and_ambiguous_mask(self):
        query = torch.tensor([[0.8, 0.6]], requires_grad=True)
        bank = torch.tensor([[1., 0.], [0.8, 0.6], [0., 1.], [-1., 0.]],
                            requires_grad=True)
        q_xy = torch.tensor([[0., 0.]], dtype=torch.float64)
        r_xy = torch.tensor([[0., 0.], [20., 0.], [30., 0.], [100., 0.]],
                            dtype=torch.float64)
        actual = full_gallery_geographic_infonce(query, q_xy, bank, r_xy)
        self.assertEqual(actual['positive_counts'], [2])
        self.assertEqual(actual['effective_negative_counts'], [1])
        self.assertEqual(actual['ambiguous_counts'], [1])
        logits = (query @ bank.T / 0.1).flatten()
        expected = torch.logsumexp(logits[[0, 1, 3]], 0) - torch.logsumexp(logits[[0, 1]], 0)
        torch.testing.assert_close(actual['loss'], expected)
        actual['loss'].backward()
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().sum()), 0)
        self.assertIsNone(bank.grad)

    def test_missing_positive_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no 25 m gallery positive'):
            full_gallery_geographic_infonce(
                torch.tensor([[1., 0.]]), torch.tensor([[0., 0.]]),
                torch.tensor([[1., 0.]]), torch.tensor([[26., 0.]]))

    def test_layerwise_lr_decay_keeps_every_night_parameter_trainable(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.patch_embed = torch.nn.Linear(2, 2)
                self.model.cls_token = torch.nn.Parameter(torch.zeros(1, 2))
                self.model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)
                                                         for _ in range(3)])
                self.model.norm = torch.nn.LayerNorm(2)

        backbone = TinyBackbone()
        groups = night_layerwise_optimizer_groups(backbone, learning_rate=1e-4,
                                                   weight_decay=0.01, decay=0.5)
        self.assertEqual({group['layer_depth'] for group in groups}, {0, 1, 2, 3})
        self.assertEqual({group['layer_depth']: group['lr'] for group in groups},
                         {0: 1e-4, 1: 5e-5, 2: 2.5e-5, 3: 1.25e-5})
        self.assertEqual({id(p) for group in groups for p in group['params']},
                         {id(p) for p in backbone.parameters()})
        self.assertEqual(sum(len(group['params']) for group in groups),
                         len(list(backbone.parameters())))

    def test_night_baseline_anchor_keeps_teacher_frozen_and_student_differentiable(self):
        class TinyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.day = torch.nn.Linear(2, 2, bias=False)
                self.seen_images = None

            def teacher(self, images, return_aux=True):
                self.seen_images = images
                self.assert_no_aux = return_aux is False
                return self.day(images)

        teacher = TinyTeacher()
        night = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            teacher.day.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
            night.weight.copy_(torch.tensor([[0.8, 0.3], [-0.2, 1.1]]))
        images = torch.tensor([[1., 2.], [2., -1.]])
        student = night(images)
        loss = night_baseline_descriptor_loss(teacher, images, student)
        self.assertIs(teacher.seen_images, images)
        self.assertTrue(teacher.assert_no_aux)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(night.weight.grad)
        self.assertGreater(float(night.weight.grad.abs().sum()), 0)
        self.assertIsNone(teacher.day.weight.grad)

    def test_independent_baseline_weight_and_geometry_radii(self):
        args = parse_args(['--device', 'cpu', '--test-buffer-m', '30',
                           '--negative-radius-m', '50'])
        self.assertEqual(args.night_baseline_weight, 100.0)
        _validate(args)
        args.night_baseline_weight = 1.0
        _validate(args)
        args.night_baseline_weight = -1.0
        with self.assertRaisesRegex(ValueError, 'night_baseline_weight'):
            _validate(args)


if __name__ == '__main__':
    unittest.main()

"""Paired retrieval checks without a GPU, checkpoint download, or real images."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from weather_vpr.inference.benchmarks import evaluate_night_benchmark, parse_args
from weather_vpr.inference.svox import sha256
from weather_vpr.utils.backend import enable_backend_imports


class NightBenchmarkTests(unittest.TestCase):
    def test_two_query_models_share_original_gallery_and_report_paired_errors(self):
        enable_backend_imports()
        from workflow.evaluation import EvaluationImage, EvaluationSet
        from workflow.model import input_preprocessing

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def image(name):
                path = root / name
                path.write_bytes(name.encode())
                return EvaluationImage(name, path)

            dataset = EvaluationSet('night_fixture', [image('r0'), image('r1')],
                                    [image('q0'), image('q1'), image('no_positive')],
                                    [[0], [1], []], {'ground_truth': 'verified_fixture'})

            class Base(torch.nn.Module):
                image_size = (224, 224)
                config = {'agg_config': {'num_clusters': 1, 'cluster_dim': 1, 'token_dim': 1}}

            base = Base()
            checkpoint = root / 'day.ckpt'
            checkpoint.write_bytes(b'day weights')
            adapter = root / 'night'
            adapter.mkdir()
            (adapter / 'adapter.safetensors').write_bytes(b'night weights')
            (adapter / 'adapter.json').write_text(json.dumps({
                'weather': 'night', 'checkpoint_sha256': sha256(checkpoint),
                'model_config': base.config, 'preprocessing': input_preprocessing(base.image_size),
            }))
            calls = []

            def extract(wrapper, images, size, device, batch_size, num_workers):
                calls.append((wrapper.model, images, size))
                if images is dataset.references:
                    self.assertIs(wrapper.model, base)
                    return torch.eye(2)
                if wrapper.model is base:
                    return torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
                return torch.tensor([[0., 1.], [0., 1.], [0., 1.]])

            kwargs = dict(dataset=dataset, checkpoint=checkpoint, backbone_repo=root,
                          adapter_dir=adapter, output=root / 'results.json',
                          cache_path=root / 'gallery.safetensors', device='cpu',
                          retrieval_device='cpu', num_workers=0, reference_cache_mib=0)
            with patch('workflow.model.load_checkpoint_model', return_value=base), \
                    patch('weather_vpr.inference.benchmarks.load_weather_model', return_value=Base()), \
                    patch('workflow.evaluation.extract_descriptors', side_effect=extract):
                result = evaluate_night_benchmark(**kwargs)
                self.assertEqual(result['baseline']['num_evaluated_queries'], 2)
                self.assertEqual(result['baseline']['recall']['R@1'], .5)
                self.assertEqual(result['adapter']['evaluation']['recall']['R@1'], .5)
                self.assertEqual(result['paired_top1'],
                                 {'fixed': 1, 'regressed': 1, 'both_correct': 0, 'both_wrong': 0})
                self.assertIsNone(result['per_query'][2]['baseline'])
                self.assertIsNone(result['per_query'][2]['night'])
                self.assertEqual(result['per_query'][0]['baseline']['rank'], 1)
                self.assertEqual(result['per_query'][0]['night']['rank'], 2)
                self.assertEqual([row[2] for row in calls], [(224, 224)] * 3)
                saved = json.loads(kwargs['output'].read_text())
                self.assertEqual(saved['adapter']['weights_sha256'], sha256(adapter / 'adapter.safetensors'))
                cache_bytes = kwargs['cache_path'].read_bytes()
                calls.clear()
                evaluate_night_benchmark(**kwargs)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(row[1] is dataset.queries for row in calls))
                self.assertEqual(kwargs['cache_path'].read_bytes(), cache_bytes)

    def test_cli_requires_explicit_ground_truth_for_other_datasets(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (['--dataset', 'manifest'],
                         ['--dataset', 'manifest', '--root', 'images'],
                         ['--dataset', 'tokyo247', '--manifest', 'other.json']):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    parse_args(args)


if __name__ == '__main__':
    unittest.main()

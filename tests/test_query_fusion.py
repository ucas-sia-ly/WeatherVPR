"""Query-only model fusion and reproducible exports; tiny CPU fixtures only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from weather_vpr.inference import benchmarks, svox
from weather_vpr.inference.svox import fused_query_descriptors
from weather_vpr.utils.backend import enable_backend_imports


class QueryFusionTests(unittest.TestCase):
    def test_normalizes_each_model_and_detaches_both_inputs(self):
        student = torch.tensor([[0., 10.]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[3., 0.]], dtype=torch.float16, requires_grad=True)
        fused = fused_query_descriptors(student, teacher, .25)
        expected = torch.tensor([[.75, .25]])
        expected /= expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(fused, expected)
        self.assertEqual(fused.dtype, torch.float32)
        self.assertFalse(fused.requires_grad)
        self.assertEqual(fused.device.type, 'cpu')
        torch.testing.assert_close(fused_query_descriptors(student, teacher, 0), torch.tensor([[1., 0.]]))
        torch.testing.assert_close(fused_query_descriptors(student, teacher, 1), torch.tensor([[0., 1.]]))

    def test_rejects_invalid_alpha_shape_values_and_cancellation(self):
        good = torch.tensor([[1., 0.]])
        for alpha in (None, True, -.01, 1.01, float('nan'), float('inf'), '.5'):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                fused_query_descriptors(good, good, alpha)
        invalid = [torch.ones(2), torch.ones(2, 2), torch.ones(1, 3),
                   torch.ones(0, 2), torch.ones(1, 0), torch.zeros(1, 2),
                   torch.tensor([[float('nan'), 0.]]), torch.tensor([[float('inf'), 0.]]),
                   [[1., 0.]]]
        for vectors in invalid:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                fused_query_descriptors(vectors, good, .5)
            with self.subTest(teacher=vectors), self.assertRaises(ValueError):
                fused_query_descriptors(good, vectors, .5)
        with self.assertRaisesRegex(ValueError, 'nonzero norms'):
            fused_query_descriptors(good, -good, .5)

    def test_cli_requires_adapter_and_checks_weight_range(self):
        for module, base_args in ((svox, []), (benchmarks, ['--dataset', 'tokyo247'])):
            self.assertIsNone(module.parse_args(base_args).query_fusion_alpha)
            self.assertIsNone(module.parse_args(base_args).save_query_descriptors)
            valid = module.parse_args(base_args + ['--adapter-dir', 'night', '--query-fusion-alpha', '.25',
                                                  '--save-query-descriptors', 'query.safetensors'])
            self.assertEqual(valid.query_fusion_alpha, .25)
            self.assertEqual(valid.save_query_descriptors, Path('query.safetensors'))
            for suffix in (['--query-fusion-alpha', '.5'], ['--save-query-descriptors', 'q.safetensors'],
                           ['--adapter-dir', 'night', '--query-fusion-alpha', 'nan'],
                           ['--adapter-dir', 'night', '--query-fusion-alpha', '1.1']):
                with self.subTest(module=module.__name__, args=suffix), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.parse_args(base_args + suffix)

    def _fixture(self, root):
        enable_backend_imports()
        from workflow.evaluation import EvaluationImage, EvaluationSet
        from workflow.model import input_preprocessing

        images = []
        for index in range(5):
            path = root / f'{index}.jpg'
            path.write_bytes(bytes([index]))
            images.append(EvaluationImage(str(index), path))
        dataset = EvaluationSet('fixture', images[:3], images[3:], [[0], []], {'split': 'fixture'})

        class Base(torch.nn.Module):
            image_size = (224, 224)
            config = {'agg_config': {'num_clusters': 1, 'cluster_dim': 1, 'token_dim': 1}}

        base = Base()
        checkpoint = root / 'day.ckpt'
        checkpoint.write_bytes(b'day')
        adapter = root / 'night'
        adapter.mkdir()
        (adapter / 'adapter.safetensors').write_bytes(b'night')
        (adapter / 'adapter.json').write_text(json.dumps({
            'weather': 'night', 'checkpoint_sha256': svox.sha256(checkpoint),
            'model_config': base.config, 'preprocessing': input_preprocessing(base.image_size)}))
        return dataset, base, dict(checkpoint=checkpoint, backbone_repo=root, adapter_dir=adapter,
                                  output=root / 'result.json', cache_path=root / 'gallery.safetensors',
                                  device='cpu', retrieval_device='cpu', batch_size=2, num_workers=0,
                                  query_chunk_size=2, reference_chunk_size=3, reference_cache_mib=0)

    def test_evaluators_reuse_extractions_preserve_gallery_and_export_ordered_queries(self):
        for module in (svox, benchmarks):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, base, kwargs = self._fixture(root)
                from workflow.evaluation import exact_retrieval
                extra = {'svox_root': root} if module is svox else {'dataset': dataset}
                query_calls = []

                def extract(model, images, size, device, batch_size, num_workers):
                    if images is dataset.references:
                        return torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
                    query_calls.append(size)
                    value = [3., 0.] if len(query_calls) % 2 else [0., 10.]
                    return torch.tensor([value, value])

                loader_patch = patch.object(svox, 'load_night_dataset', return_value=dataset)
                with loader_patch, patch('workflow.model.load_checkpoint_model', return_value=base), \
                        patch.object(module, 'load_weather_model', return_value=torch.nn.Identity()), \
                        patch('workflow.evaluation.extract_descriptors', side_effect=extract), \
                        patch('workflow.evaluation.exact_retrieval', wraps=exact_retrieval) as rank:
                    evaluator = svox.evaluate_svox_night if module is svox else benchmarks.evaluate_night_benchmark
                    pure = evaluator(**kwargs, **extra)
                    self.assertNotIn('fusion', pure)
                    cache_before = kwargs['cache_path'].read_bytes()
                    metadata_before = svox._metadata_path(kwargs['cache_path']).read_bytes()
                    result = evaluator(**kwargs, **extra, query_fusion_alpha=.25,
                                       query_descriptor_path=root / 'queries.safetensors')

                self.assertEqual(len(query_calls), 4)  # Two models once per evaluation.
                self.assertEqual(rank.call_count, 5)  # Pure baseline + student; then + fusion.
                self.assertEqual(result['baseline']['recall']['R@1'], 1)
                self.assertEqual(result['adapter']['evaluation']['recall']['R@1'], 0)
                self.assertEqual(result['fusion']['evaluation']['recall']['R@1'], 1)
                self.assertEqual(result['fusion']['alpha'], .25)
                self.assertTrue(result['fusion']['teacher_queries_reused_from_baseline'])
                self.assertEqual(result['fusion']['additional_teacher_query_forwards_after_baseline'], 0)
                self.assertEqual(result['fusion']['query_preprocessing'], result['query_preprocessing'])
                self.assertEqual(result['per_query'][0]['night']['rank'], 2)
                self.assertEqual(result['per_query'][0]['fusion']['rank'], 1)
                self.assertIsNone(result['per_query'][1]['fusion'])
                self.assertEqual(kwargs['cache_path'].read_bytes(), cache_before)
                self.assertEqual(svox._metadata_path(kwargs['cache_path']).read_bytes(), metadata_before)
                tensors = load_file(str(root / 'queries.safetensors'))
                self.assertEqual(set(tensors), {'teacher', 'student'})
                torch.testing.assert_close(tensors['teacher'], torch.tensor([[3., 0.], [3., 0.]]))
                torch.testing.assert_close(tensors['student'], torch.tensor([[0., 10.], [0., 10.]]))
                metadata = json.loads((root / 'queries.safetensors.json').read_text())
                self.assertEqual(metadata['query_ids'], ['3', '4'])
                self.assertEqual(metadata['query_sha256'], svox.gallery_fingerprint(dataset.queries))
                self.assertEqual(metadata['checkpoint_sha256'], svox.sha256(kwargs['checkpoint']))
                self.assertEqual(metadata['night_weights_sha256'], svox.sha256(kwargs['adapter_dir'] / 'adapter.safetensors'))
                self.assertEqual(metadata['tensor_sha256'], svox.sha256(root / 'queries.safetensors'))
                self.assertEqual(list(root.glob('.query_descriptors_*')), [])

    def test_multiscale_aggregation_precedes_fusion_for_both_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, base, kwargs = self._fixture(root)
            from workflow.evaluation import exact_retrieval
            calls = []

            def extract(model, images, size, device, batch_size, num_workers):
                if images is dataset.references:
                    return torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
                calls.append(size)
                value = ([3., 0.], [0., 100.], [0., 10.], [-100., 0.])[len(calls) - 1]
                return torch.tensor([value, value])

            with patch.object(svox, 'load_night_dataset', return_value=dataset), \
                    patch('workflow.model.load_checkpoint_model', return_value=base), \
                    patch.object(svox, 'load_weather_model', return_value=torch.nn.Identity()), \
                    patch('workflow.evaluation.extract_descriptors', side_effect=extract), \
                    patch('workflow.evaluation.exact_retrieval', wraps=exact_retrieval) as rank:
                result = svox.evaluate_svox_night(**kwargs, svox_root=root,
                                                 query_scales=(224, 336), query_fusion_alpha=.25)
            expected = torch.tensor([[1., 2.], [1., 2.]]) / 5 ** .5
            torch.testing.assert_close(rank.call_args_list[-1].args[1], expected)
            self.assertEqual(calls, [(224, 224), (336, 336), (224, 224), (336, 336)])
            self.assertEqual(result['fusion']['query_preprocessing']['scales'], [224, 336])
            self.assertEqual(result['preprocessing']['image_size'], [224, 224])

    def test_failed_export_leaves_no_partial_query_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, base, kwargs = self._fixture(root)
            with patch.object(svox, 'save_file', side_effect=RuntimeError('interrupted write')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted write'):
                    svox._save_query_descriptors(
                        root / 'queries.safetensors', teacher_vectors=torch.ones(2, 2),
                        student_vectors=torch.ones(2, 2), images=dataset.queries,
                        checkpoint_hash=svox.sha256(kwargs['checkpoint']),
                        adapter_dir=kwargs['adapter_dir'], preprocessing={}, query_preprocessing={})
            self.assertFalse((root / 'queries.safetensors').exists())
            self.assertFalse((root / 'queries.safetensors.json').exists())
            self.assertEqual(list(root.glob('.query_descriptors_*')), [])


if __name__ == '__main__':
    unittest.main()

"""Small SVOX protocol and gallery-cache checks; no model download required."""
import tempfile
import unittest
import shutil
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from weather_vpr.inference.svox import (
    _cache_metadata, _load_gallery_cache, _save_gallery_cache,
    _default_output_path, _extract_multiscale_queries,
    evaluate_svox_night, gallery_fingerprint, load_night_dataset,
    parse_args, sha256,
)


class SVOXNightTests(unittest.TestCase):
    def _dataset(self, root):
        gallery = root / "images/test/gallery"
        queries = root / "images/test/queries_night"
        gallery.mkdir(parents=True)
        queries.mkdir(parents=True)
        for x in (50, 0, 20):
            (gallery / f"@{x:09.2f}@000000.00@201205@id@F@.jpg").write_bytes(bytes([x + 1]))
        for x in (1000, 1):
            (queries / f"@{x:09.2f}@000000.00@20141216@.jpg").write_bytes(bytes([x % 256]))
        return load_night_dataset(root)

    def test_native_utm_protocol_and_empty_positive_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            self.assertEqual(dataset.protocol["positive_radius_meters"], 25.0)
            self.assertEqual([p.name for p in (row.path for row in dataset.references)],
                             sorted(row.path.name for row in dataset.references))
            self.assertEqual([len(row) for row in dataset.positives], [2, 0])
            self.assertEqual(dataset.summary()["num_evaluated_queries"], 1)
            self.assertEqual(dataset.summary()["num_queries_without_positives"], 1)

            from workflow.evaluation import build_results, exact_retrieval
            references = torch.eye(3)
            queries = torch.stack((references[0], references[0]))
            ranked = exact_retrieval(references, queries, dataset.positives,
                                     device="cpu", query_chunk_size=1,
                                     reference_chunk_size=2)
            self.assertIsNone(ranked[1])
            result = build_results(dataset, ranked)
            self.assertEqual(result["recall"]["R@1"], 1.0)
            self.assertEqual(result["num_queries_without_positives"], 1)

    def test_train_split_can_be_checked_without_touching_test_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test = self._dataset(root)
            shutil.copytree(root / 'images/test', root / 'images/train')
            train = load_night_dataset(root, split='train')
            self.assertEqual(train.protocol['split'], 'train')
            self.assertEqual(test.protocol['split'], 'test')
            self.assertEqual(train.summary()['num_queries'], test.summary()['num_queries'])
            with self.assertRaises(ValueError):
                load_night_dataset(root, split='val')

    def test_retrieval_finite_validation_is_bounded_and_checks_late_chunks(self):
        from weather_vpr.utils.backend import enable_backend_imports
        enable_backend_imports()
        from workflow.evaluation import exact_retrieval

        references = torch.arange(17, dtype=torch.float32)[:, None]
        queries = torch.tensor([[0.0], [10.0], [16.0]])
        finite = torch.isfinite
        with patch("torch.isfinite", wraps=finite) as checks:
            result = exact_retrieval(references, queries, [[0], [10], [16]],
                                     query_chunk_size=2, reference_chunk_size=4)
        self.assertEqual([row["rank"] for row in result], [1, 1, 1])
        self.assertTrue(all(call.args[0].shape[0] <= 4 for call in checks.call_args_list))
        for invalid_group in ("references", "queries"):
            bad_references, bad_queries = references.clone(), queries.clone()
            (bad_references if invalid_group == "references" else bad_queries)[-1, 0] = float("nan")
            with self.subTest(invalid_group=invalid_group), self.assertRaisesRegex(ValueError, "finite"):
                exact_retrieval(bad_references, bad_queries, [[0], [10], [16]],
                                query_chunk_size=2, reference_chunk_size=4)

    def test_cache_invalidated_by_images_weights_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            images = dataset.references
            cache = root / "cache/gallery.safetensors"
            fingerprint = gallery_fingerprint(images)
            metadata = _cache_metadata(images, fingerprint, "checkpoint-a", {"size": [224, 224]})
            vectors = torch.arange(6, dtype=torch.float32).reshape(3, 2)
            _save_gallery_cache(cache, metadata, vectors)
            torch.testing.assert_close(_load_gallery_cache(cache, metadata, 2), vectors)
            self.assertIsNone(_load_gallery_cache(cache, metadata, 3))
            self.assertIsNone(_load_gallery_cache(cache, {**metadata, "checkpoint_sha256": "checkpoint-b"}, 2))
            self.assertIsNone(_load_gallery_cache(cache, {**metadata, "preprocessing": {"size": [448, 448]}}, 2))
            images[0].path.write_bytes(b"changed image")
            changed = _cache_metadata(images, gallery_fingerprint(images),
                                      "checkpoint-a", {"size": [224, 224]})
            self.assertNotEqual(changed["gallery_sha256"], fingerprint)
            self.assertIsNone(_load_gallery_cache(cache, changed, 2))

    def test_query_scales_cli_validation_and_separate_default_output(self):
        self.assertIsNone(parse_args([]).query_scales)
        self.assertEqual(parse_args(['--query-scales', '224,336']).query_scales, (224, 336))
        for invalid in ('', '224,', ',224', '224,,336', '224,224', '225,336',
                        '0,336', '-14,336', '224.0,336', 'nan,336'):
            with self.subTest(value=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['--query-scales', invalid])
        self.assertEqual(_default_output_path('test').name, 'svox_night_evaluation.json')
        self.assertEqual(_default_output_path('train').name, 'svox_train_night_evaluation.json')
        self.assertEqual(_default_output_path('test', (224, 336)).name,
                         'svox_night_evaluation_query224-336.json')
        self.assertNotEqual(_default_output_path('train', (224, 336)), _default_output_path('train'))

    def test_multiscale_queries_independently_reopen_original_and_average_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)  # Enable the vendored backend import path.
            from workflow.evaluation import EvaluationImage
            image_path = root / 'query.png'
            Image.new('RGB', (31, 17), (120, 90, 45)).save(image_path)
            images = [EvaluationImage('query/0', image_path)]
            model = torch.nn.Identity()
            calls = []

            def extract(actual_model, actual_images, size, device, batch_size, num_workers):
                self.assertIs(actual_model, model)
                self.assertIs(actual_images, images)
                with Image.open(actual_images[0].path) as original:
                    calls.append((original.size, size, batch_size))
                return torch.tensor([[3., 0.]]) if size == (224, 224) else torch.tensor([[0., 100.]])

            with patch('workflow.evaluation.extract_descriptors', side_effect=extract):
                vectors = _extract_multiscale_queries(model, images, (224, 336), 'cpu', 16, 0, 2)
            torch.testing.assert_close(vectors, torch.tensor([[2 ** -0.5, 2 ** -0.5]]))
            self.assertEqual(vectors.dtype, torch.float32)
            self.assertEqual(calls, [((31, 17), (224, 224), 16), ((31, 17), (336, 336), 7)])

    def test_multiscale_query_descriptor_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._dataset(Path(temporary))
            for invalid in (torch.zeros(1, 2), torch.ones(1, 3), torch.ones(2, 2), torch.ones(2),
                            torch.tensor([[float('nan'), 1.]]), torch.tensor([[float('inf'), 1.]])):
                with self.subTest(shape=invalid.shape), \
                        patch('workflow.evaluation.extract_descriptors', return_value=invalid):
                    with self.assertRaises(ValueError):
                        _extract_multiscale_queries(torch.nn.Identity(), [object()], (224,), 'cpu', 2, 0, 2)
            with patch('workflow.evaluation.extract_descriptors', side_effect=[
                    torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.]])]):
                with self.assertRaisesRegex(ValueError, 'nonzero norms'):
                    _extract_multiscale_queries(torch.nn.Identity(), [object()], (224, 336), 'cpu', 2, 0, 2)

    def _fake_evaluation(self, root):
        dataset = self._dataset(root)
        from workflow.model import input_preprocessing

        class Base(torch.nn.Module):
            image_size = (224, 224)
            config = {'agg_config': {'num_clusters': 1, 'cluster_dim': 1, 'token_dim': 1}}

        base = Base()
        checkpoint = root / 'day.ckpt'
        checkpoint.write_bytes(b'fixture')
        adapter = root / 'night'
        adapter.mkdir()
        (adapter / 'adapter.safetensors').touch()
        (adapter / 'adapter.json').write_text(json.dumps({
            'weather': 'night', 'checkpoint_sha256': sha256(checkpoint),
            'model_config': base.config, 'preprocessing': input_preprocessing(base.image_size)}))
        kwargs = dict(svox_root=root, checkpoint=checkpoint, backbone_repo=root, adapter_dir=adapter,
                      output=root / 'result.json', cache_path=root / 'gallery.safetensors', device='cpu',
                      retrieval_device='cpu', batch_size=16, num_workers=0, query_chunk_size=32,
                      reference_chunk_size=4096, reference_cache_mib=0)
        return dataset, base, kwargs

    def test_baseline_and_night_share_query_policy_without_changing_gallery_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, base, kwargs = self._fake_evaluation(root)
            from workflow.evaluation import exact_retrieval
            calls = []

            def extract(model, images, size, device, batch_size, num_workers):
                calls.append((model, images, size, batch_size))
                if images is dataset.references:
                    return torch.tensor([[1., 0.], [0., 1.], [.8, .6]])
                value = [3., 0.] if size == (224, 224) else [0., 100.]
                return torch.tensor([value for _ in images])

            with patch('weather_vpr.inference.svox.load_night_dataset', return_value=dataset), \
                    patch('workflow.model.load_checkpoint_model', return_value=base), \
                    patch('weather_vpr.inference.svox.load_weather_model', return_value=torch.nn.Identity()), \
                    patch('workflow.evaluation.extract_descriptors', side_effect=extract), \
                    patch('workflow.evaluation.exact_retrieval', wraps=exact_retrieval) as rank, \
                    patch('weather_vpr.inference.svox._cache_metadata', wraps=_cache_metadata) as cache_metadata:
                native = evaluate_svox_night(**kwargs)
                # Native/default query tensors are used directly, preserving
                # the previous extraction path without extra normalization.
                torch.testing.assert_close(rank.call_args_list[0].args[1], torch.tensor([[3., 0.], [3., 0.]]))
                self.assertEqual([call[2] for call in calls], [(224, 224)] * 3)
                original_cache = kwargs['cache_path'].read_bytes()
                original_metadata = kwargs['cache_path'].with_suffix('.safetensors.json').read_bytes()
                calls.clear()
                multiscale = evaluate_svox_night(**kwargs, query_scales=(224, 336))
                self.assertEqual([call[2:] for call in calls], [
                    ((224, 224), 16), ((336, 336), 7), ((224, 224), 16), ((336, 336), 7)])
                self.assertTrue(all(call[1] is dataset.queries for call in calls))
                self.assertIs(calls[0][0], base)
                self.assertIs(calls[1][0], base)
                self.assertIs(calls[2][0], calls[3][0])
                self.assertIsNot(calls[2][0], base)
                expected = torch.full((2, 2), 2 ** -0.5)
                torch.testing.assert_close(rank.call_args_list[2].args[1], expected)
                torch.testing.assert_close(rank.call_args_list[3].args[1], expected)
                self.assertEqual(kwargs['cache_path'].read_bytes(), original_cache)
                self.assertEqual(kwargs['cache_path'].with_suffix('.safetensors.json').read_bytes(), original_metadata)
                self.assertEqual(cache_metadata.call_args_list[0], cache_metadata.call_args_list[1])
                self.assertEqual(native['preprocessing'], multiscale['preprocessing'])
                self.assertEqual(multiscale['preprocessing']['image_size'], [224, 224])
                self.assertEqual(multiscale['query_preprocessing']['scales'], [224, 336])
                self.assertEqual(multiscale['query_preprocessing']['per_scale'][1]['image_size'], [336, 336])
                self.assertEqual(multiscale['query_preprocessing']['batch_sizes'], [16, 7])
                self.assertEqual(base.image_size, (224, 224))

    def test_query_scale_must_have_more_patches_than_clusters(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, base, kwargs = self._fake_evaluation(Path(temporary))
            with patch('weather_vpr.inference.svox.load_night_dataset', return_value=dataset), \
                    patch('workflow.model.load_checkpoint_model', return_value=base), \
                    patch('workflow.evaluation.extract_descriptors') as extract:
                with self.assertRaisesRegex(ValueError, 'more patches than SALAD clusters'):
                    evaluate_svox_night(**kwargs, query_scales=(14,))
                extract.assert_not_called()

    def test_manifest_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            (root / "evaluation_manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native UTM"):
                load_night_dataset(root)

    def test_missing_adapter_fails_before_svox_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "runs/night_sgpd_01"
            with self.assertRaises(FileNotFoundError) as caught:
                evaluate_svox_night(
                    svox_root=root / "missing_svox", checkpoint=root / "missing.ckpt",
                    backbone_repo=root / "missing_dinov2", adapter_dir=missing,
                    output=root / "result.json", cache_path=None, device="cpu",
                    retrieval_device="cpu", batch_size=1, num_workers=0,
                    query_chunk_size=1, reference_chunk_size=1, reference_cache_mib=0,
                )
            self.assertIn(str(missing), str(caught.exception))
            self.assertIn("OUTPUT=runs/night_full_sgpd_01 bash configs/night_full_sgpd.sh", str(caught.exception))
            self.assertIn("smoke", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

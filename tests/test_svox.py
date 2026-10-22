"""Small SVOX protocol and gallery-cache checks; no model download required."""
import tempfile
import unittest
import shutil
from pathlib import Path

import torch

from weather_vpr.inference.svox import (
    _cache_metadata, _load_gallery_cache, _save_gallery_cache,
    evaluate_svox_night, gallery_fingerprint, load_night_dataset,
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

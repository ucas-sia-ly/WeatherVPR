"""Explicit pair and city/place validation semantics without a real model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.paired_dataset import sha256
from weather_vpr.inference.paired_validation import (
    _FixedCameraQueries, _default_output_path, _extract_fixed_queries,
    _query_augmentation_config, _query_augmentation_metadata, _cross_view_retrieval,
    cross_view_results,
    evaluate_pairs, load_pair_validation, mean_paired_cosine,
)


class PairedValidationTests(unittest.TestCase):
    def _manifest(self, root):
        (root / "day").mkdir()
        (root / "night").mkdir()
        rows = []
        # The same source can produce multiple Qwen images; another image from
        # the same place is a Recall positive but not the exact cosine target.
        for sample, source, place in (("b", "b", "1"), ("a", "a", "1"),
                                      ("c", "c", "2"), ("d", "a", "1")):
            day = root / "day" / f"{source}.jpg"
            if not day.exists():
                day.write_bytes(source.encode())
            night = root / "night" / f"{sample}.png"
            night.write_bytes(sample.encode())
            rows.append({"sample_id": sample, "city": "A", "place_id": place,
                         "condition": "night", "passed": True,
                         "eligible_for_training": True,
                         "source_path": f"day/{source}.jpg",
                         "output_path": f"night/{sample}.png",
                         "source_sha256": sha256(day), "output_sha256": sha256(night)})
        manifest = root / "night_val.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return manifest

    def _distractors(self, root, rows=None):
        source = root / "day/train.jpg"
        source.write_bytes(b"training day image")
        row = {"sample_id": "train", "city": "B", "place_id": "1",
               "condition": "night", "passed": True, "eligible_for_training": True,
               "source_path": "day/train.jpg", "source_sha256": sha256(source),
               # A missing generated image is intentional: the gallery only
               # consumes and hashes the original Day sources.
               "output_path": "does-not-exist.png", "output_sha256": "unused"}
        rows = rows if rows is not None else [row, {**row, "sample_id": "another-generation"}]
        manifest = root / "night_train.jsonl"
        manifest.write_text("\n".join(json.dumps(value) for value in rows) + "\n", encoding="utf-8")
        return manifest

    def test_exact_source_and_same_place_recall_positives(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary))
            dataset, exact, places = load_pair_validation(manifest)
            self.assertEqual([query.id for query in dataset.queries], ["a", "b", "c", "d"])
            self.assertEqual(exact, [0, 1, 2, 0])
            self.assertEqual(dataset.positives, [[0, 1], [0, 1], [2], [0, 1]])
            self.assertEqual(len(set(places)), 2)
            limited, _, _ = load_pair_validation(manifest, limit_pairs=2)
            self.assertEqual(limited.summary()["num_queries"], 2)
            self.assertEqual(limited.summary()["num_references"], 2)

    def test_paired_cosine_uses_exact_source(self):
        gallery = torch.eye(3)
        queries = torch.stack((gallery[0], gallery[1], gallery[2], gallery[0]))
        self.assertAlmostEqual(mean_paired_cosine(gallery, queries, [0, 1, 2, 0]), 1.0)
        self.assertAlmostEqual(mean_paired_cosine(gallery, queries, [1, 0, 2, 1]), 0.25)

    def _cross_view_metadata(self, positives, reference_count):
        from weather_vpr.utils.backend import enable_backend_imports
        enable_backend_imports()
        from workflow.evaluation import EvaluationImage, EvaluationSet
        references = [EvaluationImage(f"day/{index}", Path(f"day/{index}.jpg"))
                      for index in range(reference_count)]
        queries = [EvaluationImage(f"night/{index}", Path(f"night/{index}.jpg"))
                   for index in range(len(positives))]
        return EvaluationSet("fixture", references, queries, positives)

    def test_cross_view_exposes_failure_hidden_by_exact_pair_recall(self):
        gallery = torch.nn.functional.normalize(torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [.9, 0., .435], [0., .9, .435]]), dim=1)
        dataset = self._cross_view_metadata([[0, 1], [0, 1], [2, 3], [2, 3]], 4)
        from workflow.evaluation import build_results, exact_retrieval
        ordinary = build_results(dataset, exact_retrieval(gallery, gallery, dataset.positives))
        cross = cross_view_results(dataset, gallery, gallery, [0, 1, 2, 3], save_hard_cases=True)
        self.assertEqual(ordinary["recall"]["R@1"], 1.0)
        self.assertEqual(cross["recall"]["R@1"], 0.0)
        self.assertEqual(cross["evaluated_queries"], 4)
        self.assertEqual(cross["error_queries"][0]["ground_truth_ids"], ["day/1"])
        self.assertIn("selection_bias", cross["protocol"])

    def test_cross_view_removes_own_source_from_candidates_and_excludes_single_view(self):
        gallery = torch.eye(3)
        dataset = self._cross_view_metadata([[0, 1], [0, 1], [2]], 3)
        result = cross_view_results(dataset, gallery, gallery, [0, 1, 2])
        # Each orthogonal alternative ties the distractor, but comes first by
        # Gallery order. The perfect exact match must not count as a negative.
        self.assertEqual(result["recall"]["R@1"], 1.0)
        self.assertEqual(result["evaluated_queries"], 2)
        self.assertEqual(result["protocol"]["excluded_single_view_queries"], 1)
        self.assertEqual(result["num_evaluated_queries"], 2)

    def test_cross_view_no_alternative_returns_no_recall(self):
        gallery = torch.eye(2)
        dataset = self._cross_view_metadata([[0], [1]], 2)
        result = cross_view_results(dataset, gallery, gallery, [0, 1])
        self.assertEqual(result["evaluated_queries"], 0)
        self.assertIsNone(result["recall"])
        self.assertEqual(result["num_evaluated_queries"], 0)

    def test_cross_view_ties_keep_original_gallery_indices_across_chunks(self):
        self._cross_view_metadata([[0, 2]], 4)  # Ensure the backend is importable.
        from workflow.evaluation import exact_retrieval
        gallery = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
        queries = torch.tensor([[1., 0.], [1., 0.]])
        targets, sources = [[2, 3], [2, 3]], [0, 1]
        ranks = _cross_view_retrieval(gallery, queries, targets, sources,
                                      query_chunk_size=1, reference_chunk_size=1)
        self.assertEqual([row["rank"] for row in ranks], [2, 2])
        self.assertEqual([row["predicted_index"] for row in ranks], [1, 0])
        self.assertEqual([row["ground_truth_index"] for row in ranks], [2, 2])
        # Compare to the existing exact evaluator after explicitly removing the
        # source candidate and mapping every index back to the original Gallery.
        for index, source in enumerate(sources):
            kept = [reference for reference in range(len(gallery)) if reference != source]
            expected = exact_retrieval(gallery[kept], queries[index:index + 1],
                                       [[kept.index(target) for target in targets[index]]])[0]
            expected["ground_truth_index"] = kept[expected["ground_truth_index"]]
            expected["predicted_index"] = kept[expected["predicted_index"]]
            self.assertEqual(ranks[index], expected)
        self.assertEqual(ranks, _cross_view_retrieval(gallery, queries, targets, sources,
                                                     query_chunk_size=2, reference_chunk_size=3))

    def test_changed_pair_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            (root / "night/a.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "incorrect output_sha256"):
                load_pair_validation(manifest)

    def test_day_distractors_deduplicate_sources_without_reading_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            distractors = self._distractors(root)
            dataset, exact, _ = load_pair_validation(manifest, distractor_manifest=distractors)
            self.assertEqual(exact, [0, 1, 2, 0])
            self.assertEqual(dataset.positives, [[0, 1], [0, 1], [2], [0, 1]])
            self.assertEqual(len(dataset.references), 4)
            info = dataset.protocol["distractors"]
            self.assertEqual(info["added_images"], 1)
            self.assertEqual(info["duplicate_source_rows"], 1)
            self.assertEqual(info["manifest_sha256"], sha256(distractors))
            self.assertTrue(info["source_hashes_verified"])
            self.assertFalse(info["generated_images_read"])

    def test_distractors_reject_overlap_with_places_outside_limited_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            distractors = self._distractors(root)
            row = json.loads(distractors.read_text().splitlines()[0])
            # Place 2 is absent from the first two sorted validation pairs.
            row.update(city="A", place_id="2")
            distractors.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training/validation place overlap"):
                load_pair_validation(manifest, limit_pairs=2, distractor_manifest=distractors)

    def test_distractors_cannot_relabel_validation_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            distractors = self._distractors(root)
            row = json.loads(distractors.read_text().splitlines()[0])
            row.update(source_path="day/c.jpg", source_sha256=sha256(root / "day/c.jpg"))
            distractors.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training/validation source overlap"):
                load_pair_validation(manifest, limit_pairs=2, distractor_manifest=distractors)

    def test_distractor_source_hash_is_always_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            distractors = self._distractors(root)
            (root / "day/train.jpg").write_bytes(b"changed after manifest")
            with self.assertRaisesRegex(ValueError, "incorrect source_sha256"):
                load_pair_validation(manifest, verify_hashes=False, distractor_manifest=distractors)

    def test_camera_views_are_seeded_by_sample_id_and_shared_between_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            dataset, _, _ = load_pair_validation(manifest)
            for index, sample in enumerate(dataset.queries):
                Image.new("RGB", (19, 13), (100 + index * 20, 90, 60)).save(sample.path)
            originals = {sample.path: sample.path.read_bytes() for sample in dataset.queries}
            config = NightAugmentationConfig(
                profile="camera", probability=1.0,
                operation_probabilities={name: float(name == "noise") for name in (
                    "exposure", "vignette", "motion_blur", "bloom", "noise",
                    "desaturation", "photometric", "jpeg")})
            views = _FixedCameraQueries(dataset.queries, (14, 28), config, seed=17)
            reordered = _FixedCameraQueries(list(reversed(dataset.queries)), (14, 28), config, seed=17)
            self.assertTrue(torch.equal(views[0], views[0]))
            self.assertTrue(torch.equal(views[0], reordered[-1]))
            different_seed = _FixedCameraQueries(dataset.queries, (14, 28), config, seed=18)
            self.assertFalse(torch.equal(views[0], different_seed[0]))

            class Recorder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inputs = []

                def forward(self, images):
                    self.inputs.append(images.detach().clone())
                    self.assert_inference = not torch.is_grad_enabled()
                    return images.mean(dim=(2, 3))

            models = (Recorder(), Recorder())
            outputs = _extract_fixed_queries(models, dataset.queries, (14, 28), "cpu", 3,
                                             0, config, 17)
            self.assertEqual(tuple(outputs[0].shape), (4, 3))
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertTrue(torch.equal(torch.cat(models[0].inputs), torch.cat(models[1].inputs)))
            self.assertTrue(torch.equal(torch.cat(models[0].inputs)[0], views[0]))
            self.assertTrue(all(not model.training and model.assert_inference for model in models))
            self.assertEqual(originals, {sample.path: sample.path.read_bytes() for sample in dataset.queries})

    def test_camera_config_metadata_and_output_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "camera.json"
            config_path.write_text(json.dumps({"gamma_range": [0.8, 1.2],
                                              "operation_probabilities": {"noise": 1}}))
            config = _query_augmentation_config("camera", config_path, 0.7, 1.0)
            metadata = _query_augmentation_metadata(config, 17, (14, 28), config_path)
            self.assertEqual(metadata["profile"], "camera")
            self.assertEqual(metadata["resolution_hw"], [14, 28])
            self.assertEqual(metadata["strength"], 0.7)
            self.assertEqual(metadata["effective_operation_probabilities"]["noise"], 1)
            self.assertEqual(metadata["config_sha256"], sha256(config_path))
            json.dumps(metadata)
        paths = {_default_output_path(limit, profile, distractor)
                 for limit in (None, 2) for profile in ("off", "camera")
                 for distractor in (None, Path("train.jsonl"))}
        self.assertEqual(len(paths), 8)
        self.assertEqual(_default_output_path().name, "night_pairs_validation.json")

    def _evaluation_fixture(self, root):
        manifest = self._manifest(root)
        dataset, _, _ = load_pair_validation(manifest)
        for index, sample in enumerate(dataset.references + dataset.queries):
            Image.new("RGB", (18, 20), (70 + index * 10, 90, 120)).save(sample.path)
        # Refresh image hashes after replacing fixture bytes by valid images.
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        for row in rows:
            row["source_sha256"] = sha256(root / row["source_path"])
            row["output_sha256"] = sha256(root / row["output_path"])
        manifest.write_text("\n".join(map(json.dumps, rows)) + "\n")
        from workflow.model import input_preprocessing

        class Base(torch.nn.Module):
            image_size = (14, 28)
            config = {"agg_config": {"num_clusters": 1, "cluster_dim": 2, "token_dim": 1}}

            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, images):
                self.inputs.append(images.detach().clone())
                return torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=1)

        class Night(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def student(self, images, weather, return_aux=False):
                self.inputs.append(images.detach().clone())
                return torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=1)

        base, night = Base(), Night()
        checkpoint = root / "weights.ckpt"
        checkpoint.write_bytes(b"fixture")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter.safetensors").touch()
        (adapter / "adapter.json").write_text(json.dumps({
            "weather": "night", "checkpoint_sha256": sha256(checkpoint),
            "model_config": base.config, "preprocessing": input_preprocessing(base.image_size)}))
        kwargs = dict(manifest=manifest, checkpoint=checkpoint, backbone_repo=root, adapter_dir=adapter,
                      output=root / "camera.json", device="cpu", num_workers=0,
                      batch_size=16, query_augmentation="camera", augmentation_probability=1,
                      augmentation_seed=17)
        return dataset, base, night, kwargs

    def test_evaluation_gallery_stays_clean_with_camera_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, base, night, kwargs = self._evaluation_fixture(Path(temporary))
            from workflow.reliability_cache import preprocess_teacher_image

            with patch("workflow.model.load_checkpoint_model", return_value=base), \
                    patch("weather_vpr.inference.paired_validation.load_weather_model", return_value=night):
                result = evaluate_pairs(**kwargs)
            with Image.open(dataset.references[0].path) as image:
                clean_reference = preprocess_teacher_image(image, base.image_size)
            self.assertTrue(torch.equal(base.inputs[0][0], clean_reference))
            self.assertTrue(torch.equal(base.inputs[1], night.inputs[0]))
            self.assertEqual(result["query_augmentation"]["profile"], "camera")
            self.assertEqual(result["baseline"]["recall"], result["adapter"]["recall"])

    def test_gallery_cache_reused_and_invalidated_without_caching_night_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, base, night, kwargs = self._evaluation_fixture(root)
            from workflow.evaluation import extract_descriptors
            from workflow.model import input_preprocessing

            kwargs.update(gallery_cache=root / "gallery.safetensors", verify_hashes=False)
            with patch("workflow.model.load_checkpoint_model", return_value=base), \
                    patch("weather_vpr.inference.paired_validation.load_weather_model", return_value=night), \
                    patch("workflow.evaluation.extract_descriptors", wraps=extract_descriptors) as extract:
                first = evaluate_pairs(**kwargs)
                second = evaluate_pairs(**kwargs)
                self.assertFalse(first["gallery_cache_hit"])
                self.assertTrue(second["gallery_cache_hit"])
                self.assertEqual(first["gallery_cache"], str(kwargs["gallery_cache"]))
                self.assertEqual(extract.call_count, 1)
                self.assertEqual(len(night.inputs), 2)
                self.assertEqual(first["baseline"], second["baseline"])

                # Actual image bytes, rather than timestamps, bind the cache.
                Image.new("RGB", (18, 20), (95, 65, 140)).save(dataset.references[0].path)
                changed_image = evaluate_pairs(**kwargs)
                self.assertFalse(changed_image["gallery_cache_hit"])
                self.assertNotEqual(first["gallery_sha256"], changed_image["gallery_sha256"])
                self.assertEqual(extract.call_count, 2)

                # Update the adapter contract together with legitimate changes
                # to the frozen checkpoint/preprocessing being evaluated.
                base.image_size = (28, 28)
                metadata_path = kwargs["adapter_dir"] / "adapter.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["preprocessing"] = input_preprocessing(base.image_size)
                metadata_path.write_text(json.dumps(metadata))
                changed_preprocessing = evaluate_pairs(**kwargs)
                self.assertFalse(changed_preprocessing["gallery_cache_hit"])
                self.assertEqual(extract.call_count, 3)

                kwargs["checkpoint"].write_bytes(b"changed weights")
                metadata["checkpoint_sha256"] = sha256(kwargs["checkpoint"])
                metadata_path.write_text(json.dumps(metadata))
                changed_checkpoint = evaluate_pairs(**kwargs)
                self.assertFalse(changed_checkpoint["gallery_cache_hit"])
                self.assertEqual(extract.call_count, 4)
                self.assertEqual(len(night.inputs), 5)

    def test_gallery_cache_rejects_nonunit_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, base, night, kwargs = self._evaluation_fixture(root)
            from workflow.evaluation import extract_descriptors
            from workflow.model import input_preprocessing
            from weather_vpr.inference.svox import (
                _cache_metadata, _save_gallery_cache, gallery_fingerprint,
            )

            cache = root / "gallery.safetensors"
            expected = _cache_metadata(dataset.references, gallery_fingerprint(dataset.references),
                                       sha256(kwargs["checkpoint"]), input_preprocessing(base.image_size))
            _save_gallery_cache(cache, expected, torch.zeros(len(dataset.references), 3))
            with patch("workflow.model.load_checkpoint_model", return_value=base), \
                    patch("weather_vpr.inference.paired_validation.load_weather_model", return_value=night), \
                    patch("workflow.evaluation.extract_descriptors", wraps=extract_descriptors) as extract:
                result = evaluate_pairs(**kwargs, gallery_cache=cache)
                self.assertFalse(result["gallery_cache_hit"])
                self.assertEqual(extract.call_count, 1)

    def test_missing_adapter_fails_before_model_or_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "runs/night_sgpd_01"
            with self.assertRaises(FileNotFoundError) as caught:
                evaluate_pairs(manifest=root / "missing.jsonl", checkpoint=root / "missing.ckpt",
                               backbone_repo=root / "missing_dinov2", adapter_dir=missing,
                               output=root / "result.json", device="cpu", num_workers=0)
            self.assertIn(str(missing), str(caught.exception))
            self.assertIn("OUTPUT=runs/night_full_sgpd_01 bash configs/night_full_sgpd.sh", str(caught.exception))
            self.assertIn("smoke", str(caught.exception))

            missing.mkdir(parents=True)
            (missing / "adapter.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "missing adapter.safetensors"):
                evaluate_pairs(manifest=root / "missing.jsonl", checkpoint=root / "missing.ckpt",
                               backbone_repo=root / "missing_dinov2", adapter_dir=missing,
                               output=root / "result.json", device="cpu", num_workers=0)
            (missing / "training_log.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Training may still be running"):
                evaluate_pairs(manifest=root / "missing.jsonl", checkpoint=root / "missing.ckpt",
                               backbone_repo=root / "missing_dinov2", adapter_dir=missing,
                               output=root / "result.json", device="cpu", num_workers=0)


if __name__ == "__main__":
    unittest.main()

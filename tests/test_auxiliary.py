"""Exercise paired target caches and real/synthetic auxiliary data paths on CPU."""
from __future__ import annotations

from collections import Counter
import hashlib
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
import torch.nn.functional as F

from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.paired_dataset import AcceptedWeatherPairs
from weather_vpr.data.real_pairs import ReviewedRealPairs, file_sha256
from weather_vpr.data.student_augmentation import student_training_tensor
from weather_vpr.training.auxiliary import (
    AuxiliaryBranch, GroupBalancedCycle, auxiliary_seed,
    build_auxiliary_branch, load_or_build_pair_targets,
)
from weather_vpr.utils.backend import enable_backend_imports


class _TinyTeacher(torch.nn.Module):
    def __init__(self, *, consume_rng=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []
        self.consume_rng = consume_rng

    def teacher(self, images, return_aux=True):
        self.calls.append((torch.is_grad_enabled(), self.training, return_aux))
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(3)
        means = images.mean(dim=(-1, -2))
        return torch.cat((means, torch.ones_like(means[:, :1])), dim=1) * self.scale


class AuxiliaryDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "real"
        self.source_rows = []
        for index, name in enumerate(("route_a", "route_b", "route_c")):
            row = {"sample_id": name, "sequence_id": "route", "content_group": name}
            for side, folder in (("day", "Day"), ("night", "Night")):
                path = self.root / folder / f"{name}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                color = (20 + index * 37, 140 - index * 25, 220 - index * 31)
                if side == "night":
                    color = tuple(value // 2 for value in color)
                Image.new("RGB", (23, 19), color).save(path)
                row[f"{side}_path"] = f"{folder}/{name}.jpg"
                self._record_image_hash(row, side)
            self.source_rows.append(row)
        self.review = self.base / "review.jsonl"
        self.reviews = []
        for index, row in enumerate(reversed(self.source_rows)):
            self.reviews.append({
                "sample_id": row["sample_id"], "scene_group": f"scene_{index}",
                "split": "train", "pair_valid": True, "visibility_valid": True,
                "review_status": "manual", "reject_reason": None,
            })
        self._publish_source_and_review()
        self.dataset = self._load_dataset()
        self.net = _TinyTeacher().train()
        self.cache = self.base / "cache" / "real_day.safetensors"
        self.checkpoint_hash = "a" * 64
        self.preprocessing = self.dataset.preprocessing

    def _record_image_hash(self, row, side):
        path = self.root / row[f"{side}_path"]
        row[f"{side}_sha256"] = file_sha256(path)
        with Image.open(path) as image:
            image.load()
            row[f"{side}_pixel_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()

    def _publish_source_and_review(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in self.source_rows))
        digest = file_sha256(manifest)
        (self.root / "dataset_info.json").write_text(json.dumps({
            "pairs": len(self.source_rows), "manifest_sha256": digest}))
        sources = {row["sample_id"]: row for row in self.source_rows}
        for review in self.reviews:
            source = sources[review["sample_id"]]
            review.update(source_manifest_sha256=digest, day_sha256=source["day_sha256"],
                          night_sha256=source["night_sha256"])
        self._write_review()

    def _write_review(self):
        self.review.write_text("".join(json.dumps(row) + "\n" for row in self.reviews))

    def _load_dataset(self):
        return ReviewedRealPairs(self.root, self.review, image_size=(14, 14))

    def _build(self, **kwargs):
        return load_or_build_pair_targets(
            self.net, kwargs.pop("dataset", self.dataset),
            kwargs.pop("checkpoint_hash", self.checkpoint_hash),
            self.preprocessing, self.cache, "cpu", batch_size=kwargs.pop("batch_size", 2),
            **kwargs)

    def _args(self, **kwargs):
        return SimpleNamespace(**{
            "auxiliary_mode": "real", "auxiliary_weight": .1,
            "auxiliary_samples": 2, "auxiliary_day_cache": self.cache,
            "real_pairs_root": self.root, "real_pairs_review": self.review,
            "device": "cpu", "seed": 42, **kwargs,
        })

    def _rng_snapshot(self):
        return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()

    def _assert_rng_equal(self, expected):
        python, numpy, cpu = expected
        self.assertEqual(random.getstate(), python)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy[0])
        np.testing.assert_array_equal(current[1], numpy[1])
        self.assertEqual(current[2:], numpy[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), cpu, rtol=0, atol=0)

    def test_real_cache_is_day_only_ordered_normalized_frozen_and_has_no_place_labels(self):
        with patch.object(ReviewedRealPairs, "__getitem__",
                          side_effect=AssertionError("paired/night decoding during cache build")):
            result, metadata = self._build(batch_size=1)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(result.norm(dim=1), torch.ones(3))
        self.assertEqual(self.net.calls, [(False, True, False)] * 3)
        self.assertTrue(self.net.training)
        self.assertTrue(self.net.scale.requires_grad)
        self.assertIsNone(self.net.scale.grad)
        self.assertEqual(metadata["cache_status"], "built")
        self.assertEqual(metadata["source_images"], self.dataset.day_sources)
        self.assertNotIn("places", metadata)
        self.assertNotIn("place", metadata["dataset_signature"]["source_images"][0])
        self.assertNotIn("place_id", metadata["dataset_signature"]["source_images"][0])
        expected = []
        for index in range(len(self.dataset)):
            means = self.dataset[index]["day_tensor"].mean(dim=(-1, -2))
            expected.append(F.normalize(torch.cat((means, torch.ones(1))), dim=0))
        torch.testing.assert_close(result, torch.stack(expected))

    def test_unchanged_cache_hit_does_not_run_teacher(self):
        original, before = self._build()
        count = len(self.net.calls)
        cached, after = self._build()
        torch.testing.assert_close(original, cached, rtol=0, atol=0)
        self.assertEqual(len(self.net.calls), count)
        self.assertEqual(before["cache_signature"], after["cache_signature"])
        self.assertEqual(after["cache_status"], "hit")
        self.assertIsNone(after["rebuild_reason"])

    def test_checkpoint_review_and_row_order_changes_invalidate_cache(self):
        original, first = self._build()
        with self.assertWarnsRegex(RuntimeWarning, "identity changed"):
            _, changed_hash = self._build(checkpoint_hash="b" * 64)
        self.assertNotEqual(first["cache_signature"], changed_hash["cache_signature"])
        self.reviews[0]["review_status"] = "automatic"
        self._write_review()
        self.dataset = self._load_dataset()
        with self.assertWarnsRegex(RuntimeWarning, "identity changed"):
            reviewed, changed_review = self._build()
        self.assertEqual(changed_review["cache_status"], "rebuilt")
        self.assertNotEqual(first["cache_signature"], changed_review["cache_signature"])
        torch.testing.assert_close(reviewed, original)
        self.reviews.reverse()
        self._write_review()
        self.dataset = self._load_dataset()
        with self.assertWarnsRegex(RuntimeWarning, "identity changed"):
            reordered, changed_order = self._build()
        torch.testing.assert_close(reordered, original.flip(0))
        self.assertNotEqual(changed_review["cache_signature"], changed_order["cache_signature"])

    def test_updated_audit_for_changed_image_rebuilds_targets(self):
        original, first = self._build()
        row = self.source_rows[0]
        Image.new("RGB", (23, 19), (210, 10, 70)).save(self.root / row["day_path"])
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self._load_dataset()
        self._record_image_hash(row, "day")
        self._publish_source_and_review()
        self.dataset = self._load_dataset()
        with self.assertWarnsRegex(RuntimeWarning, "identity changed"):
            changed, info = self._build()
        self.assertNotEqual(first["cache_signature"], info["cache_signature"])
        self.assertFalse(torch.equal(changed[-1], original[-1]))
        torch.testing.assert_close(changed[:-1], original[:-1], rtol=0, atol=0)

    def test_tensor_hash_mismatch_and_invalid_safetensors_rebuild(self):
        expected, _ = self._build()
        self.cache.write_bytes(b"damaged tensor")
        with self.assertWarnsRegex(RuntimeWarning, "SHA-256 mismatch"):
            rebuilt, info = self._build()
        torch.testing.assert_close(rebuilt, expected, rtol=0, atol=0)
        self.assertEqual(info["cache_status"], "rebuilt")
        # Match the sidecar hash so the loader must reject the actual format,
        # rather than only detecting a mismatched file digest.
        self.cache.write_bytes(b"not a safetensors payload")
        sidecar = self.cache.with_suffix(self.cache.suffix + ".json")
        saved = json.loads(sidecar.read_text())
        saved["tensor_sha256"] = file_sha256(self.cache)
        sidecar.write_text(json.dumps(saved))
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding paired Day targets"):
            rebuilt, info = self._build()
        torch.testing.assert_close(rebuilt, expected, rtol=0, atol=0)
        self.assertEqual(info["cache_status"], "rebuilt")

    def test_real_branch_build_and_sampling_preserve_rng_on_cold_and_hit_cache(self):
        self.net.consume_rng = True
        state = self._rng_snapshot()
        first = build_auxiliary_branch(self._args(), object(), self.net,
                                       self.checkpoint_hash, self.preprocessing)
        self._assert_rng_equal(state)
        calls = len(self.net.calls)
        second = build_auxiliary_branch(self._args(), object(), self.net,
                                        self.checkpoint_hash, self.preprocessing)
        self._assert_rng_equal(state)
        self.assertEqual(len(self.net.calls), calls)
        self.assertEqual(first.target_info["cache_status"], "built")
        self.assertEqual(second.target_info["cache_status"], "hit")
        with first.rng_context(0, 0):
            batch = first.next_batch(0, 0)
        self._assert_rng_equal(state)
        with second.rng_context(0, 0):
            repeat = second.next_batch(0, 0)
        torch.testing.assert_close(batch["images"], repeat["images"], rtol=0, atol=0)
        self.assertEqual(batch["indices"], repeat["indices"])
        self.assertEqual(batch["augmentation"]["applied_samples"], 0)
        self.assertEqual(first.source_info["review_manifest"], str(self.review))
        for offset, index in enumerate(batch["indices"]):
            torch.testing.assert_close(batch["images"][offset], self.dataset[index]["night_tensor"])
            torch.testing.assert_close(batch["targets"][offset], first.targets[index])
            self.assertEqual(batch["sample_ids"][offset], self.dataset.rows[index]["sample_id"])

    def test_real_branch_rejects_cross_split_scene_before_teacher_runs(self):
        self.reviews[1].update(scene_group=self.reviews[0]["scene_group"], split="dev")
        self._write_review()
        with self.assertRaisesRegex(ValueError, "scene_group crosses splits"):
            build_auxiliary_branch(self._args(), object(), self.net,
                                   self.checkpoint_hash, self.preprocessing)
        self.assertEqual(self.net.calls, [])
        self.assertFalse(self.cache.exists())

    def test_none_and_zero_weight_branches_do_not_read_files(self):
        for mode, weight in (("none", .1), ("real", 0), ("synthetic", 0)):
            with self.subTest(mode=mode, weight=weight), \
                    patch("weather_vpr.training.auxiliary.enable_backend_imports",
                          side_effect=AssertionError("backend read")), \
                    patch.object(Path, "open", side_effect=AssertionError("file read")):
                args = SimpleNamespace(auxiliary_mode=mode, auxiliary_weight=weight)
                self.assertIsNone(build_auxiliary_branch(args, object(), object(), "bad", {}))

    def test_group_cycle_is_balanced_reproducible_and_uses_private_randomness(self):
        groups = ["long"] * 5 + ["short"] + ["medium"] * 2
        state = self._rng_snapshot()
        sampler = GroupBalancedCycle(groups, seed=123)
        repeated = GroupBalancedCycle(groups, seed=123)
        drawn = sampler.draw(30)
        self.assertEqual(drawn, repeated.draw(7) + repeated.draw(23))
        self._assert_rng_equal(state)
        self.assertEqual(Counter(groups[index] for index in drawn),
                         {"long": 10, "short": 10, "medium": 10})
        long = [index for index in drawn if groups[index] == "long"]
        self.assertEqual(set(long[:5]), set(range(5)))
        self.assertEqual(set(long[5:]), set(range(5)))
        metadata = sampler.metadata()
        self.assertEqual(metadata["unique_samples"], 8)
        self.assertEqual(metadata["repeated_exposures"], 22)
        self.assertEqual(metadata["samples_drawn"], 30)

    def test_synthetic_branch_reads_only_qwen_and_uses_expected_camera_pipeline(self):
        enable_backend_imports()
        synthetic_manifest = self.base / "synthetic.jsonl"
        entries = [{
            "sample_id": row["sample_id"], "city": "city", "place_id": str(index),
            "condition": "night", "passed": True, "eligible_for_training": True,
            "source_path": str(self.root / row["day_path"]),
            "output_path": str(self.root / row["night_path"]),
            "output_sha256": row["night_sha256"],
        } for index, row in enumerate(self.source_rows)]
        synthetic_manifest.write_text("".join(json.dumps(row) + "\n" for row in entries))
        config = NightAugmentationConfig(
            profile="camera", probability=1.0,
            operation_probabilities={name: float(name == "exposure") for name in (
                "exposure", "vignette", "motion_blur", "bloom", "noise",
                "desaturation", "photometric", "jpeg")},
            gain_range=(.5, .5), gamma_range=(1.0, 1.0))
        dataset = AcceptedWeatherPairs(synthetic_manifest, "night", image_size=(14, 14),
                                       night_augmentation=config, seed=42)
        targets = F.normalize(torch.arange(1, 13, dtype=torch.float32).view(3, 4), dim=1)
        args = self._args(auxiliary_mode="synthetic", auxiliary_day_cache=None)
        branch = build_auxiliary_branch(args, dataset, self.net, self.checkpoint_hash,
                                        self.preprocessing, day_descriptors=targets,
                                        day_bank_info={"cache_status": "hit"})
        self.assertEqual(self.net.calls, [])
        branch_again = build_auxiliary_branch(args, dataset, self.net, self.checkpoint_hash,
                                              self.preprocessing, day_descriptors=targets,
                                              day_bank_info={"cache_status": "hit"})
        # The synthetic branch must not construct unused Source Night/Day views.
        with patch.object(AcceptedWeatherPairs, "__getitem__",
                          side_effect=AssertionError("indexed full paired dataset")):
            with branch.rng_context(2, 7):
                batch = branch.next_batch(2, 7)
            with branch_again.rng_context(2, 7):
                repeat = branch_again.next_batch(2, 7)
        self.assertEqual(batch["augmentation"]["applied_samples"], 2)
        torch.testing.assert_close(batch["images"], repeat["images"], rtol=0, atol=0)
        from workflow.reliability_cache import preprocess_teacher_image
        for position, index in enumerate(batch["indices"]):
            row = dataset.rows[index]
            with Image.open(row["target_file"]) as image:
                expected, record = student_training_tensor(
                    image, dataset.size, dataset.augmentation,
                    seed=auxiliary_seed(42, "synthetic_view", 2, 7, position, index))
                clean = preprocess_teacher_image(image, dataset.size)
            self.assertTrue(record["applied"])
            self.assertFalse(torch.equal(expected, clean))
            torch.testing.assert_close(batch["images"][position], expected, rtol=0, atol=0)
            torch.testing.assert_close(batch["targets"][position], targets[index], rtol=0, atol=0)
        sampled_path = Path(dataset.rows[batch["indices"][0]]["target_file"])
        sampled_path.write_bytes(b"changed Qwen source bytes")
        # A fresh branch repeats exactly the same first sample stream.
        branch = AuxiliaryBranch(args, dataset, targets, {}, branch.groups, {})
        with self.assertRaisesRegex(RuntimeError, "Auxiliary Qwen image changed"):
            branch.next_batch(2, 7)


if __name__ == "__main__":
    unittest.main()

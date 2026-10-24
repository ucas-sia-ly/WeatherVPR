"""Training Day bank provenance, mode/gradient isolation, and cache integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from weather_vpr.training.day_bank import _validate_descriptors, load_or_build_day_bank


class _DayOnlyTrainingSet:
    def __init__(self, manifest, rows):
        self.manifest = manifest
        self.rows = rows
        self.size = (14, 14)

    def __getitem__(self, index):
        raise AssertionError("The paired dataset must never be indexed: it decodes night images")


class _Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def teacher(self, images, return_aux=True):
        self.calls.append((torch.is_grad_enabled(), self.training, return_aux))
        means = images.mean(dim=(-1, -2))
        return torch.cat([means, torch.ones_like(means[:, :1])], dim=1) * self.scale


class DayBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.first = self.root / "day_a.png"
        self.second = self.root / "day_b.png"
        Image.new("RGB", (23, 19), (30, 120, 220)).save(self.first)
        Image.new("RGB", (19, 23), (220, 80, 40)).save(self.second)
        self.manifest = self.root / "pairs.jsonl"
        self.manifest.write_text('{"fake": "only manifest bytes are signed"}\n', encoding="utf-8")
        self.dataset = _DayOnlyTrainingSet(self.manifest, [
            {"source_file": self.second, "place": ("city-b", "same-id"),
             "target_file": self.root / "MISSING_NIGHT.png"},
            {"source_file": self.first, "place": ("city-a", "same-id"),
             "target_file": self.root / "MISSING_NIGHT.png"},
        ])
        self.preprocessing = {
            "version": "rgb_bilinear_imagenet_v1", "image_size": [14, 14],
            "color_space": "RGB", "input_range": [0.0, 1.0], "resize": "PIL_bilinear",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        }
        self.net = _Teacher().train()
        self.path = self.root / "cache" / "training_day.safetensors"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, path=True, **kwargs):
        return load_or_build_day_bank(
            self.net, self.dataset, kwargs.pop("checkpoint_hash", "a" * 64),
            kwargs.pop("preprocessing", self.preprocessing),
            self.path if path else None, "cpu", **kwargs)

    def test_build_is_day_only_ordered_normalized_and_no_grad(self):
        descriptors, places, metadata = self.build(batch_size=1)
        self.assertEqual(places, [("city-b", "same-id"), ("city-a", "same-id")])
        self.assertEqual(descriptors.dtype, torch.float32)
        self.assertEqual(descriptors.device.type, "cpu")
        self.assertEqual(tuple(descriptors.shape), (2, 4))
        torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2))
        self.assertFalse(descriptors.requires_grad)
        self.assertTrue(self.net.training)
        self.assertTrue(self.net.scale.requires_grad)
        self.assertIsNone(self.net.scale.grad)
        self.assertEqual(self.net.calls, [(False, True, False), (False, True, False)])
        self.assertEqual(metadata["cache_status"], "built")
        self.assertEqual(metadata["places"], [list(place) for place in places])
        self.assertEqual([row["path"] for row in metadata["source_images"]],
                         [str(self.second), str(self.first)])
        self.assertEqual(metadata["source_images"][0]["sha256"],
                         hashlib.sha256(self.second.read_bytes()).hexdigest())
        self.assertTrue(self.path.is_file())
        self.assertTrue(self.path.with_suffix(".safetensors.json").is_file())
        # Expected values also verify that source order was not sorted/shuffled.
        from workflow.reliability_cache import preprocess_teacher_image
        expected = []
        for image_path in (self.second, self.first):
            with Image.open(image_path) as image:
                means = preprocess_teacher_image(image, (14, 14)).mean(dim=(-1, -2))
                expected.append(torch.nn.functional.normalize(torch.cat([means, torch.ones(1)]), dim=0))
        torch.testing.assert_close(descriptors, torch.stack(expected))

    def test_unchanged_cache_reuses_vectors_without_teacher_forward(self):
        original, places, metadata = self.build()
        count = len(self.net.calls)
        cached, cached_places, cached_metadata = self.build()
        torch.testing.assert_close(original, cached, rtol=0, atol=0)
        self.assertEqual(len(self.net.calls), count)
        self.assertEqual(places, cached_places)
        self.assertEqual(cached_metadata["cache_status"], "hit")
        self.assertIsNone(cached_metadata["rebuild_reason"])
        self.assertEqual(metadata["cache_signature"], cached_metadata["cache_signature"])

    def test_cache_hit_does_not_read_the_whole_tensor_into_python_bytes(self):
        original, _, _ = self.build()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
            cached, _, metadata = self.build()
        torch.testing.assert_close(original, cached, rtol=0, atol=0)
        self.assertEqual(metadata["cache_status"], "hit")

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "Linux stable file handles")
    def test_atomic_tensor_replacement_does_not_change_the_hashed_snapshot(self):
        original, _, _ = self.build()
        replacement = self.root / "replacement.safetensors"
        save_file({"descriptors": -original}, str(replacement))

        def replace_then_load(filename, **kwargs):
            replacement.replace(self.path)
            return load_file(filename, **kwargs)

        with patch("weather_vpr.training.day_bank.load_file", side_effect=replace_then_load):
            cached, _, metadata = self.build()
        torch.testing.assert_close(original, cached, rtol=0, atol=0)
        self.assertEqual(metadata["cache_status"], "hit")

    def test_descriptor_validation_checks_late_chunks(self):
        descriptors = torch.nn.functional.normalize(torch.ones(1025, 4), dim=1)
        descriptors[-1, -1] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _validate_descriptors(descriptors, len(descriptors))

    def test_actual_source_byte_change_invalidates_cache(self):
        original, _, metadata = self.build()
        # The manifest and filename stay the same; actual bytes must win.
        Image.new("RGB", (19, 23), (80, 230, 100)).save(self.second)
        with self.assertWarnsRegex(RuntimeWarning, "source_images"):
            rebuilt, _, changed = self.build()
        self.assertEqual(changed["cache_status"], "rebuilt")
        self.assertNotEqual(metadata["cache_signature"], changed["cache_signature"])
        self.assertFalse(torch.equal(original, rebuilt))

    def test_manifest_checkpoint_and_place_order_each_invalidate(self):
        self.build()
        self.manifest.write_text("changed manifest\n", encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "manifest_sha256"):
            self.build()
        with self.assertWarnsRegex(RuntimeWarning, "checkpoint_sha256"):
            self.build(checkpoint_hash="b" * 64)
        self.dataset.rows[0]["place"] = ("changed-city", "same-id")
        with self.assertWarnsRegex(RuntimeWarning, "places"):
            _, places, _ = self.build(checkpoint_hash="b" * 64)
        self.assertEqual(places[0], ("changed-city", "same-id"))

    def test_corrupt_tensor_or_nonunit_descriptors_are_rebuilt(self):
        self.build()
        self.path.write_bytes(b"broken safetensors")
        with self.assertWarnsRegex(RuntimeWarning, "SHA-256 mismatch"):
            self.build()
        tensors = load_file(str(self.path))
        save_file({"descriptors": tensors["descriptors"] * 2}, str(self.path))
        sidecar = self.path.with_suffix(".safetensors.json")
        metadata = json.loads(sidecar.read_text())
        metadata["tensor_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "L2 normalized"):
            descriptors, _, _ = self.build()
        torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2))

    def test_bad_metadata_rebuilds_and_missing_source_fails_clearly(self):
        self.build()
        self.path.with_suffix(".safetensors.json").write_text("not json", encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "invalid Day bank cache"):
            self.build()
        self.first.unlink()
        with self.assertRaisesRegex(ValueError, "Invalid training Day row 1"):
            self.build()

    def test_malformed_safetensors_with_matching_hash_is_rebuilt(self):
        self.build()
        self.path.write_bytes(b"malformed but signed tensor bytes")
        sidecar = self.path.with_suffix(".safetensors.json")
        metadata = json.loads(sidecar.read_text())
        metadata["tensor_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "invalid Day bank cache"):
            _, _, rebuilt = self.build()
        self.assertEqual(rebuilt["cache_status"], "rebuilt")

    def test_cold_and_warm_builds_preserve_the_main_torch_rng(self):
        before = torch.random.get_rng_state().clone()
        cold, _, cold_info = self.build(batch_size=1)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        warm, _, warm_info = self.build(batch_size=1)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(cold, warm, rtol=0, atol=0)
        self.assertEqual((cold_info['cache_status'], warm_info['cache_status']), ('built', 'hit'))

    def test_memory_bank_keeps_mode_and_never_publishes(self):
        self.net.eval()
        descriptors, _, metadata = self.build(path=False, num_workers=1)
        self.assertEqual(metadata["cache_status"], "memory")
        self.assertFalse(self.net.training)
        self.assertFalse(descriptors.requires_grad)
        self.assertFalse(self.path.exists())

    def test_refuses_incompatible_preprocessing_or_zero_teacher(self):
        incompatible = {**self.preprocessing, "resize": "PIL_nearest"}
        with self.assertRaisesRegex(ValueError, "original SALAD"):
            self.build(preprocessing=incompatible)
        self.net.scale.data.zero_()
        with self.assertRaisesRegex(ValueError, "nonfinite or zero"):
            self.build()


if __name__ == "__main__":
    unittest.main()

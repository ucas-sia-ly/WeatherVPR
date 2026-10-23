"""Camera augmentation invariants; no evaluation or global RNG side effects."""
import json
import unittest

import numpy as np
from PIL import Image

from weather_vpr.data.night_augmentation import (
    NightAugmentationConfig, NightCameraAugmentation, night_augmentation_seed,
)


OPS = ("exposure", "bloom", "motion_blur", "noise", "jpeg",
       "desaturation", "vignette", "photometric")


def patterned_image():
    y, x = np.mgrid[:48, :64]
    rgb = np.stack([(x*7+y*3) % 256, (x*3+y*9) % 256, (x*11+y*5) % 256], axis=2)
    rgb[10:15, 30:35] = 255
    return Image.fromarray(rgb.astype(np.uint8))


class NightAugmentationTests(unittest.TestCase):
    def test_disabled_paths_preserve_exact_pixels_and_input(self):
        image = patterned_image()
        before = np.asarray(image).copy()
        for config in (NightAugmentationConfig(),
                       NightAugmentationConfig(profile="camera", strength=0),
                       NightAugmentationConfig(profile="camera", probability=0)):
            output, record = NightCameraAugmentation(config)(image, seed=37)
            np.testing.assert_array_equal(np.asarray(output), before)
            self.assertFalse(record["applied"])
            self.assertEqual(record["operations"], {})
            self.assertIsNot(output, image)
        np.testing.assert_array_equal(np.asarray(image), before)

    def test_reproducible_local_rng_and_json_records(self):
        transform = NightCameraAugmentation(NightAugmentationConfig(profile="camera", probability=1))
        np.random.seed(912)
        state = np.random.get_state()
        first, record = transform(patterned_image(), seed=123)
        second, repeated = transform(patterned_image(), seed=123)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(record, repeated)
        self.assertEqual(json.loads(json.dumps(record)), record)
        after = np.random.get_state()
        for before_item, after_item in zip(state, after):
            np.testing.assert_array_equal(before_item, after_item)
        other, _ = transform(patterned_image(), seed=124)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(other)))

    def test_each_operation_can_be_isolated_without_geometry_change(self):
        image = patterned_image()
        original = np.asarray(image).copy()
        for op in OPS:
            with self.subTest(operation=op):
                probabilities = dict.fromkeys(OPS, 0.0)
                probabilities[op] = 1.0
                config = NightAugmentationConfig(profile="camera", probability=1,
                                                 operation_probabilities=probabilities)
                output, record = NightCameraAugmentation(config)(image, seed=18)
                array = np.asarray(output)
                self.assertEqual(output.size, image.size)
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(array.dtype, np.uint8)
                self.assertTrue(np.isfinite(array).all())
                self.assertGreaterEqual(array.min(), 0)
                self.assertLessEqual(array.max(), 255)
                self.assertEqual(set(record["operations"]), {op})
                self.assertFalse(np.array_equal(original, array))
                json.dumps(record)
        np.testing.assert_array_equal(np.asarray(image), original)

    def test_all_operations_disabled_is_pixel_identity(self):
        image = patterned_image()
        output, record = NightCameraAugmentation(NightAugmentationConfig(
            profile="camera", probability=1, operation_probabilities=dict.fromkeys(OPS, 0.0),
        ))(image, seed=11)
        np.testing.assert_array_equal(np.asarray(image), np.asarray(output))
        self.assertFalse(record["applied"])

    def test_tiny_strength_is_finite_identity_without_rng_side_effects(self):
        image = patterned_image()
        original = np.asarray(image).copy()
        state = np.random.get_state()
        for strength in (1e-6, 1e-10, 1e-300):
            with self.subTest(strength=strength):
                output, record = NightCameraAugmentation(NightAugmentationConfig(
                    profile="camera", probability=1, strength=strength,
                    operation_probabilities=dict.fromkeys(OPS, 1.0),
                ))(image, seed=11)
                np.testing.assert_array_equal(np.asarray(output), original)
                self.assertFalse(record["applied"])
                self.assertEqual(record["operations"], {})
                json.dumps(record, allow_nan=False)
        for before_item, after_item in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(before_item, after_item)

    def test_noise_acts_on_dark_pixels_and_bloom_spreads_bright_sources(self):
        probabilities = dict.fromkeys(OPS, 0.0)
        probabilities["noise"] = 1.0
        dark = Image.fromarray(np.full((64, 64, 3), 20, dtype=np.uint8))
        output, record = NightCameraAugmentation(NightAugmentationConfig(
            profile="camera", probability=1, operation_probabilities=probabilities,
            photon_count_range=(250, 250), read_noise_std_range=(0.01, 0.01),
        ))(dark, seed=1)
        self.assertGreater(np.asarray(output).std(), 1)
        self.assertEqual(record["operations"]["noise"]["photon_count"], 250)
        probabilities = dict.fromkeys(OPS, 0.0)
        probabilities["bloom"] = 1.0
        point = np.zeros((64, 64, 3), dtype=np.uint8)
        point[28:36, 28:36] = 255
        halo, _ = NightCameraAugmentation(NightAugmentationConfig(
            profile="camera", probability=1, operation_probabilities=probabilities,
            bloom_strength_range=(0.5, 0.5), bloom_radius_fraction_range=(0.05, 0.05),
        ))(Image.fromarray(point), seed=1)
        self.assertGreater(np.asarray(halo)[27, 30, 0], 0)

    def test_config_json_roundtrip_and_stable_epoch_seed(self):
        config = NightAugmentationConfig.from_dict({"profile": "camera", "gamma_range": [0.8, 1.2],
                                                  "operation_probabilities": {"jpeg": 0}})
        copied = NightAugmentationConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        first, record = NightCameraAugmentation(config)(patterned_image(), seed=5)
        second, other = NightCameraAugmentation(copied)(patterned_image(), seed=5)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(record, other)
        seed = night_augmentation_seed(7, 2, "city/place/image.jpg")
        self.assertEqual(seed, night_augmentation_seed(7, 2, "city/place/image.jpg"))
        self.assertNotEqual(seed, night_augmentation_seed(7, 3, "city/place/image.jpg"))
        self.assertNotEqual(seed, night_augmentation_seed(7, 2, "another-image.jpg"))

    def test_validation_and_extreme_valid_images(self):
        for kwargs in ({"profile": "unknown"}, {"strength": -1}, {"strength": 3},
                       {"strength": float("nan")}, {"probability": 2},
                       {"operation_probabilities": {"crop": 1}},
                       {"operation_probabilities": {"noise": -0.1}},
                       {"gamma_range": [0, 1]}, {"jpeg_quality_range": [1, 101]},
                       {"motion_length_range": [3.5, 9]}, {"photon_count_range": [10, 1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NightAugmentationConfig(**kwargs)
        transform = NightCameraAugmentation(NightAugmentationConfig(profile="camera", probability=1, strength=2))
        for value in (0, 255):
            result, record = transform(Image.fromarray(np.full((1, 1, 3), value, dtype=np.uint8)), seed=11)
            self.assertEqual(result.size, (1, 1))
            self.assertEqual(result.mode, "RGB")
            json.dumps(record, allow_nan=False)
        with self.assertRaises(ValueError):
            transform(patterned_image(), seed=-1)
        with self.assertRaises(TypeError):
            transform(np.zeros((2, 2, 3)), seed=1)


if __name__ == "__main__":
    unittest.main()

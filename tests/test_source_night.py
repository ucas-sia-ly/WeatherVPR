"""Source-photo low-light views must always darken before camera degradation."""
import json
import random
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
from PIL import Image

from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.source_night import SourceNightAugmentation, SourceNightConfig


def patterned_image():
    y, x = np.mgrid[:31, :47]
    rgb = np.stack([(x*5+y*3) % 256, (x*3+y*7) % 256, (x*11+y*5) % 256], axis=2)
    return Image.fromarray(rgb.astype(np.uint8))


class SourceNightTests(unittest.TestCase):
    def test_zero_camera_probability_still_applies_log_uniform_exposure(self):
        transform = SourceNightAugmentation(SourceNightConfig(camera_probability=0))
        image = patterned_image()
        for seed in range(20):
            output, record = transform(image, seed=seed)
            self.assertTrue(record["applied"])
            self.assertIn("exposure", record["operations"])
            self.assertEqual(record["operations"]["exposure"]["distribution"], "log_uniform")
            gain = record["operations"]["exposure"]["gain"]
            self.assertTrue(0.03 <= gain <= 0.30)
            self.assertLess(record["pre_camera_linear_luminance_mean"],
                            record["input_linear_luminance_mean"])
            self.assertLess(np.asarray(output).mean(), np.asarray(image).mean())
            self.assertFalse(record["camera"]["applied"])
            self.assertFalse(any(op.startswith("camera_") for op in record["operations"]))

    def test_fixed_linear_gain_and_gamma_are_not_srgb_brightness(self):
        config = SourceNightConfig(
            gain_range=(0.1, 0.1), gamma_range=(1, 1), camera_probability=0,
            illumination_probability=0, white_balance_probability=0,
        )
        output, record = SourceNightAugmentation(config)(
            Image.fromarray(np.full((4, 5, 3), 255, dtype=np.uint8)), seed=17)
        # Linear intensity .1 is approximately sRGB value 89, not 255*.1.
        np.testing.assert_array_equal(np.asarray(output), 89)
        self.assertAlmostEqual(record["operations"]["exposure"]["gain"], 0.1)
        self.assertEqual(record["operations"]["gamma"]["gamma"], 1)
        self.assertEqual(set(record["operations"]), {"exposure", "gamma"})

    def test_reproducible_local_rng_and_separated_operation_streams(self):
        image = patterned_image()
        transform = SourceNightAugmentation(SourceNightConfig(camera_probability=1))
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        first, record = transform(image, seed=123)
        second, repeated = transform(image, seed=123)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(record, repeated)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(python_state, random.getstate())
        different, other = transform(image, seed=124)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(different)))
        self.assertNotEqual(record["stage_seeds"], other["stage_seeds"])
        self.assertEqual(len(set(record["stage_seeds"].values())), 5)
        no_field = SourceNightAugmentation(SourceNightConfig(
            camera_probability=1, illumination_probability=0,
        ))
        _, no_field_record = no_field(image, seed=123)
        for name in ("exposure", "gamma", "white_balance"):
            self.assertEqual(record["operations"].get(name), no_field_record["operations"].get(name))

    def test_smooth_field_record_and_geometry_without_input_mutation(self):
        image = patterned_image()
        original = np.asarray(image).copy()
        config = SourceNightConfig(
            camera_probability=0, illumination_probability=1, white_balance_probability=1,
        )
        output, record = SourceNightAugmentation(config)(image, seed=9)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)
        self.assertIsNot(output, image)
        np.testing.assert_array_equal(np.asarray(image), original)
        field = record["operations"]["illumination"]
        grid = np.asarray(field["grid_values"])
        self.assertEqual(grid.shape, (3, 3))
        self.assertAlmostEqual(grid.min(), field["floor"], places=6)
        self.assertEqual(grid.max(), 1)
        self.assertGreaterEqual(field["actual_min"], field["floor"]-1e-7)
        self.assertLessEqual(field["actual_max"], 1)
        balance = record["operations"]["white_balance"]["rgb_factors"]
        self.assertEqual(len(balance), 3)
        self.assertTrue(all(0.8 <= factor <= 1.2 for factor in balance))

    def test_disabled_camera_profiles_and_strength_do_not_disable_darkening(self):
        image = Image.fromarray(np.full((6, 8, 3), 200, dtype=np.uint8))
        configs = (SourceNightConfig(camera_probability=1, camera_config=NightAugmentationConfig()),
                   SourceNightConfig(camera_probability=1, camera_strength=0))
        for config in configs:
            output, record = SourceNightAugmentation(config)(image, seed=3)
            self.assertTrue(record["applied"])
            self.assertFalse(record["camera"]["applied"])
            self.assertLess(np.asarray(output).mean(), np.asarray(image).mean())
            self.assertIn("exposure", record["operations"])

    def test_json_roundtrip_and_frozen_config(self):
        config = SourceNightConfig.from_dict({
            "gain_range": [0.04, 0.2], "camera_probability": 1,
            "camera_config": {"profile": "camera", "operation_probabilities": {"noise": 1}},
        })
        copied = SourceNightConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        first, record = SourceNightAugmentation(config)(patterned_image(), seed=2)
        second, repeated = SourceNightAugmentation(copied)(patterned_image(), seed=2)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(record, repeated)
        self.assertEqual(copied.camera_config.operation_probabilities["exposure"], 0)
        self.assertNotIn("exposure", record["camera"]["operations"])
        self.assertEqual(copied.camera_config.probability, 1)
        with self.assertRaises(FrozenInstanceError):
            config.camera_probability = 0

    def test_tiny_black_and_non_rgb_images_are_safe(self):
        transform = SourceNightAugmentation(SourceNightConfig(
            camera_probability=1, illumination_probability=1, white_balance_probability=1))
        for size in ((1, 1), (1, 5), (5, 1), (2, 2)):
            for mode in ("RGB", "L", "RGBA"):
                for value in (0, 255):
                    with self.subTest(size=size, mode=mode, value=value):
                        image = Image.new(mode, size, value)
                        output, record = transform(image, seed=19)
                        array = np.asarray(output)
                        self.assertEqual(output.size, size)
                        self.assertEqual(output.mode, "RGB")
                        self.assertEqual(array.dtype, np.uint8)
                        self.assertTrue(np.isfinite(array).all())
                        json.dumps(record, allow_nan=False)
        black, record = SourceNightAugmentation(SourceNightConfig(camera_probability=0))(
            Image.new("RGB", (1, 1)), seed=1)
        np.testing.assert_array_equal(np.asarray(black), 0)
        self.assertEqual(record["pre_camera_linear_luminance_mean"], 0)

    def test_validation_rejects_brightening_nonfinite_and_exposure_reenable(self):
        invalid = (
            {"gain_range": [0.1, 1]}, {"gain_range": [1, 2]}, {"gain_range": [0, 0.2]},
            {"gain_range": [0.2, 0.1]}, {"gain_range": [float("nan"), 0.3]},
            {"gain_range": [0.1, float("inf")]}, {"gamma_range": [0.9, 1.3]},
            {"gamma_range": [1, float("inf")]}, {"illumination_probability": float("nan")},
            {"illumination_probability": -0.1}, {"camera_probability": 2},
            {"camera_strength": float("nan")}, {"camera_strength": 3},
            {"white_balance_probability": float("inf")}, {"white_balance_range": [0, 1]},
            {"illumination_floor_range": [0, 1]}, {"illumination_floor_range": [0.3, 1.1]},
            {"illumination_grid_size": [1, 3]}, {"illumination_grid_size": [3.5, 3]},
            {"gain_range": [0.8, 0.9]},
            {"camera_config": {"profile": "camera", "operation_probabilities": {"exposure": 0.1}}},
            {"camera_config": NightAugmentationConfig(
                profile="off", operation_probabilities={"exposure": 1})},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SourceNightConfig(**kwargs)
        with self.assertRaises(ValueError):
            SourceNightConfig.from_dict([])
        with self.assertRaises(ValueError):
            SourceNightConfig.from_dict({"off": True})
        transform = SourceNightAugmentation(SourceNightConfig())
        for seed in (-1, True, 2.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                transform(patterned_image(), seed=seed)
        with self.assertRaises(TypeError):
            transform(np.zeros((3, 3, 3)), seed=1)
        with self.assertRaises(TypeError):
            SourceNightAugmentation(None)


if __name__ == "__main__":
    unittest.main()

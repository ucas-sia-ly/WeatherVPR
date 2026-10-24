"""Photometric visibility weights are detached and never create matches."""
import unittest

import torch

from weather_vpr.data.patch_visibility import (
    patch_visibility_from_image_tensor,
    visibility_metadata,
)


def _normalized(rgb):
    mean = rgb.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


class PatchVisibilityTests(unittest.TestCase):
    def test_visible_low_light_remains_reliable(self):
        image = _normalized(torch.full((2, 3, 14, 28), .03)).requires_grad_()
        result = patch_visibility_from_image_tensor(image)
        torch.testing.assert_close(result, torch.ones(2, 1, 2))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.dtype, torch.float32)

    def test_black_white_and_partial_pixels(self):
        rgb = torch.zeros(1, 3, 14, 42)
        rgb[:, :, :, 14:28] = 1.
        rgb[:, :, :7, 28:] = .2
        result = patch_visibility_from_image_tensor(_normalized(rgb))
        torch.testing.assert_close(result, torch.tensor([[[0., 0., .5]]]))

    def test_one_clipped_channel_does_not_mean_all_information_lost(self):
        rgb = torch.zeros(1, 3, 14, 14)
        rgb[:, 0] = 1.
        rgb[:, 1] = .2
        result = patch_visibility_from_image_tensor(_normalized(rgb))
        self.assertEqual(result.item(), 1.)

    def test_pixels_below_conservative_dark_threshold_are_downweighted(self):
        rgb = torch.full((1, 3, 14, 14), .005)
        self.assertEqual(patch_visibility_from_image_tensor(_normalized(rgb)).item(), 0.)

    def test_fp32_and_no_autocast_gradient(self):
        image = _normalized(torch.full((1, 3, 14, 14), .1)).bfloat16().requires_grad_()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = patch_visibility_from_image_tensor(image)
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.item(), 1.)

    def test_metadata_is_an_independent_json_compatible_copy(self):
        metadata = visibility_metadata()
        self.assertEqual(metadata["mode"], "photometric")
        self.assertEqual(metadata["patch_size"], 14)
        metadata["patch_size"] = 7
        self.assertEqual(visibility_metadata()["patch_size"], 14)

    def test_validation(self):
        for image, kwargs, exception in [
            (torch.ones(1, 3, 15, 14), {}, ValueError),
            (torch.ones(1, 1, 14, 14), {}, ValueError),
            (torch.ones(1, 3, 14, 14, dtype=torch.long), {}, TypeError),
            (torch.full((1, 3, 14, 14), float("nan")), {}, ValueError),
            (torch.ones(1, 3, 14, 14), {"patch_size": 0}, ValueError),
            (torch.ones(1, 3, 14, 14), {"patch_size": True}, ValueError),
        ]:
            with self.subTest(shape=image.shape, kwargs=kwargs):
                with self.assertRaises(exception):
                    patch_visibility_from_image_tensor(image, **kwargs)


if __name__ == "__main__":
    unittest.main()

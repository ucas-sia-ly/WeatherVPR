"""Exact fixed-pair SALAD margin decomposition; no model or images needed."""
import math
import unittest

import torch
import torch.nn.functional as F

from weather_vpr.inference.patch_attribution import signed_patch_margin


def fixture(content=None, probabilities=None):
    if content is None:
        content = torch.tensor([[[1., -2., 3., 1.], [.5, 2., -1., 4.]],
                                [[2., 1., -2., 3.], [1., -1., 3., 2.]]])
    if probabilities is None:
        probabilities = torch.tensor([[[.5, .1, .7, .3], [.2, .7, .1, .4]],
                                      [[.2, .6, .2, .3], [.5, .1, .4, .2]]])
    batch = content.shape[0]
    token = torch.tensor([[1., 2., -1.], [-2., 1., 2.]])[:batch]
    aggregate = (content[:, :, None] * probabilities[:, None]).sum(-1)
    descriptor = F.normalize(torch.cat([F.normalize(token, dim=1),
                                        F.normalize(aggregate, dim=1).flatten(1)], dim=1), dim=1)
    dustbin = 1 - probabilities.sum(1, keepdim=True)
    logs = torch.cat([probabilities, dustbin], dim=1).log()
    positive = F.normalize(torch.arange(1, descriptor.numel() + 1, dtype=torch.float32)
                           .reshape_as(descriptor), dim=1)
    negative = positive.flip(1)
    return descriptor, content, logs, positive, negative


class PatchAttributionTests(unittest.TestCase):
    def test_additive_margin_batch_and_salad_flatten_order(self):
        args = fixture()
        result = signed_patch_margin(*args, token_dim=3)
        descriptor, content, logs, positive, negative = args
        self.assertEqual(result["patch_contributions"].shape, (2, 4))
        direct = (descriptor * (positive - negative)).sum(1)
        torch.testing.assert_close(result["direct_margin"], direct)
        torch.testing.assert_close(result["total_reconstructed"], direct, atol=1e-6, rtol=0)
        torch.testing.assert_close(result["cls_energy_fraction"], torch.full((2,), 1 / 3))
        torch.testing.assert_close(result["cls_energy_fraction"] + result["cluster_energy_fraction"],
                                   torch.ones(2))
        # An explicit scalar C,K formula guards against silently flattening K,C.
        delta = positive - negative
        p = logs[:, :-1].exp()
        aggregate = (content[:, :, None] * p[:, None]).sum(-1)
        for row in range(2):
            for patch in range(4):
                expected = 0.
                for cluster in range(2):
                    norm = aggregate[row, :, cluster].norm()
                    for channel in range(2):
                        expected += (p[row, cluster, patch] * content[row, channel, patch]
                                     * delta[row, 3 + channel * 2 + cluster] / norm / math.sqrt(3))
                torch.testing.assert_close(result["patch_contributions"][row, patch], expected)

    def test_direction_and_identical_gallery_pairs(self):
        args = fixture()
        result = signed_patch_margin(*args, token_dim=3)
        reverse = signed_patch_margin(*args[:3], args[4], args[3], token_dim=3)
        torch.testing.assert_close(reverse["patch_contributions"], -result["patch_contributions"])
        torch.testing.assert_close(reverse["direct_margin"], -result["direct_margin"])
        same = signed_patch_margin(*args[:3], args[3], args[3], token_dim=3)
        self.assertEqual(same["patch_contributions"].abs().sum().item(), 0)
        self.assertEqual(same["direct_margin"].abs().sum().item(), 0)

    def test_zero_and_tiny_clusters_use_actual_normalization(self):
        for tiny in (0., 1e-15):
            with self.subTest(cluster=tiny):
                content = torch.tensor([[[1., 0.], [0., tiny]]])
                probabilities = torch.tensor([[[.8, 0.], [0., .8]]])
                args = fixture(content, probabilities)
                result = signed_patch_margin(*args, token_dim=3)
                torch.testing.assert_close(result["total_reconstructed"], result["direct_margin"],
                                           atol=1e-6, rtol=0)
                self.assertTrue(torch.isfinite(result["patch_contributions"]).all())
                self.assertAlmostEqual(result["cls_energy_fraction"].item(), .5, places=5)

    def test_normalizes_cosines_detaches_and_disables_autocast(self):
        args = tuple(value.clone().requires_grad_() for value in fixture())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = signed_patch_margin(args[0] * 3, args[1], args[2],
                                         args[3] * 4, args[4] * 7, token_dim=3)
        expected = signed_patch_margin(*args, token_dim=3)
        for name, tensor in result.items():
            self.assertEqual(tensor.dtype, torch.float32, name)
            self.assertFalse(tensor.requires_grad, name)
            torch.testing.assert_close(tensor, expected[name], atol=1e-6, rtol=1e-5)
        self.assertTrue(all(value.grad is None for value in args))

    def test_rejects_auxiliaries_from_another_forward(self):
        args = list(fixture())
        args[1] = args[1].flip(-1)
        with self.assertRaisesRegex(ValueError, "does not match SALAD"):
            signed_patch_margin(*args, token_dim=3)
        args = list(fixture())
        args[0] = args[0].roll(1, dims=1)
        with self.assertRaises(ValueError):
            signed_patch_margin(*args, token_dim=3)

    def test_invalid_inputs_fail_explicitly(self):
        good = fixture()
        for name, options in (("token_dim", {"token_dim": 0}),
                              ("token_dim", {"token_dim": True}),
                              ("eps", {"eps": 0}), ("eps", {"eps": math.nan}),
                              ("atol", {"atol": -1})):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, name):
                signed_patch_margin(*good, **({"token_dim": 3} | options))
        changes = [(0, torch.ones(2, 3)), (1, torch.ones(2, 3, 4)),
                   (2, torch.zeros(2, 1, 4)), (3, torch.ones(1, 7)),
                   (4, torch.zeros_like(good[4])),
                   (0, torch.zeros_like(good[0])),
                   (1, good[1].long()), (2, good[2].long())]
        for index, value in changes:
            args = list(good)
            args[index] = value
            with self.subTest(index=index, shape=value.shape), self.assertRaises((ValueError, TypeError)):
                signed_patch_margin(*args, token_dim=3)
        for index in range(5):
            args = [value.clone() for value in good]
            args[index].flatten()[0] = math.nan
            with self.subTest(nonfinite=index), self.assertRaises(ValueError):
                signed_patch_margin(*args, token_dim=3)
        for value in (math.inf, .1):
            args = [item.clone() for item in good]
            args[2].flatten()[0] = value
            with self.subTest(log_probability=value), self.assertRaises(ValueError):
                signed_patch_margin(*args, token_dim=3)


if __name__ == "__main__":
    unittest.main()

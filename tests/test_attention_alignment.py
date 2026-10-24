"""Trusted attention supervision does not infer or learn its own support."""
import math
import unittest

import torch

from weather_vpr.losses.attention import trusted_attention_alignment


class TrustedAttentionTests(unittest.TestCase):
    def test_matches_reorder_teacher_and_kl_direction(self):
        teacher = torch.tensor([[.2, .5, .3]])
        student = torch.tensor([[.4, .6]])
        result = trusted_attention_alignment(
            student, teacher, torch.ones_like(teacher),
            torch.tensor([[2, 0]]), torch.ones_like(student))
        expected = .6 * math.log(.6 / .4) + .4 * math.log(.4 / .6)
        self.assertAlmostEqual(result["loss"].item(), expected, places=6)
        self.assertEqual(result["attention_valid_queries"].item(), 1)
        self.assertEqual(result["attention_trusted_patches"].item(), 2)
        self.assertEqual(result["attention_coverage"].item(), 1.)

    def test_detached_teacher_weights_and_invalid_support(self):
        student = torch.tensor([[.1, .2, .3, .4]], requires_grad=True)
        teacher = torch.tensor([[.8, .2]], requires_grad=True)
        keep = torch.tensor([[.9, .5]], requires_grad=True)
        confidence = torch.tensor([[1., .7, 1., 1.]], requires_grad=True)
        result = trusted_attention_alignment(
            student, teacher, keep, torch.tensor([[0, 1, -1, 200]]), confidence)
        result["loss"].backward()
        self.assertGreater(student.grad[:, :2].abs().sum().item(), 0.)
        self.assertEqual(student.grad[:, 2:].abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(keep.grad)
        self.assertIsNone(confidence.grad)
        self.assertEqual(result["attention_coverage"].item(), .5)
        for key, value in result.items():
            if key != "loss":
                self.assertFalse(value.requires_grad, key)

    def test_saliency_and_uniform_are_independent_modes(self):
        student = torch.tensor([[.5, .5, .1]])
        teacher = torch.tensor([[.8, .2, .9]])
        keep = torch.tensor([[1., .1, 0.]])
        index = torch.tensor([[0, 1, 2]])
        q = torch.ones_like(student)
        saliency = trusted_attention_alignment(student, teacher, keep, index, q)
        uniform = trusted_attention_alignment(
            student, teacher, keep, index, q, importance_mode="uniform")
        self.assertEqual(saliency["attention_trusted_patches"].item(), 2)
        self.assertEqual(uniform["attention_trusted_patches"].item(), 3)
        self.assertNotAlmostEqual(saliency["loss"].item(), uniform["loss"].item())

    def test_empty_singleton_and_zero_teacher_rows_are_safe(self):
        student = torch.full((3, 2), .5, requires_grad=True)
        teacher = torch.tensor([[.5, .5], [.5, .5], [0., 0.]])
        index = torch.tensor([[-1, -1], [0, -1], [0, 1]])
        q = torch.tensor([[0., 0.], [1., 0.], [1., 1.]])
        result = trusted_attention_alignment(student, teacher, torch.ones_like(teacher), index, q)
        self.assertEqual(result["loss"].item(), 0.)
        self.assertEqual(result["attention_valid_queries"].item(), 0)
        result["loss"].backward()
        self.assertEqual(student.grad.abs().sum().item(), 0.)

    def test_only_eligible_images_enter_batch_mean(self):
        student = torch.tensor([[.2, .8], [.7, .3]], requires_grad=True)
        teacher = torch.tensor([[.8, .2], [.7, .3]])
        index = torch.tensor([[0, 1], [-1, -1]])
        q = torch.tensor([[1., 1.], [0., 0.]])
        batched = trusted_attention_alignment(student, teacher, torch.ones_like(teacher), index, q)
        single = trusted_attention_alignment(student[:1], teacher[:1], torch.ones(1, 2),
                                             index[:1], q[:1])
        torch.testing.assert_close(batched["loss"], single["loss"])
        batched["loss"].backward()
        self.assertEqual(student.grad[1].abs().sum().item(), 0.)

    def test_student_zero_mass_cannot_escape_loss(self):
        student = torch.zeros(1, 2, requires_grad=True)
        teacher = torch.tensor([[.9, .1]])
        result = trusted_attention_alignment(
            student, teacher, torch.ones_like(teacher), torch.tensor([[0, 1]]),
            torch.tensor([[.1, .1]]))
        expected = .9 * math.log(.9 / .5) + .1 * math.log(.1 / .5)
        self.assertAlmostEqual(result["loss"].item(), expected, places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0.)

    def test_fp32_under_autocast_and_scale_invariant_normal_support(self):
        student = torch.tensor([[.2, .3, .5]], dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.tensor([[.4, .1, .5]], dtype=torch.bfloat16)
        index = torch.tensor([[0, 1, 2]])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = trusted_attention_alignment(
                student, teacher, torch.ones(1, 3), index, torch.ones(1, 3))
            scaled = trusted_attention_alignment(
                student, teacher, torch.ones(1, 3), index, torch.full((1, 3), .01))
        self.assertEqual(result["loss"].dtype, torch.float32)
        torch.testing.assert_close(result["loss"], scaled["loss"], atol=1e-6, rtol=1e-6)

    def test_many_to_one_correspondence_keeps_repeated_teacher_mass(self):
        teacher = torch.tensor([[.2, .8]])
        student = torch.tensor([[.2, .2, .8]])
        result = trusted_attention_alignment(
            student, teacher, torch.ones_like(teacher), torch.tensor([[0, 0, 1]]),
            torch.ones_like(student))
        self.assertAlmostEqual(result["loss"].item(), 0., places=6)

    def test_conditional_kl_does_not_constrain_total_trusted_attention_mass(self):
        # The same relative probabilities on verified patches can coexist
        # with most attention moving to an unsupervised patch. This is the
        # intended conditional objective's limit, not a retrieval guarantee.
        teacher = torch.tensor([[.6, .3, .1]])
        full_mass = torch.tensor([[.6, .3, .1]])
        little_mass = torch.tensor([[.06, .03, .91]])
        keep = torch.ones_like(teacher)
        index = torch.tensor([[0, 1, -1]])
        q = torch.tensor([[1., 1., 0.]])
        first = trusted_attention_alignment(full_mass, teacher, keep, index, q)
        second = trusted_attention_alignment(little_mass, teacher, keep, index, q)
        torch.testing.assert_close(first["loss"], second["loss"], atol=1e-6, rtol=0)
        self.assertLess(second["loss"].item(), 1e-6)
        self.assertAlmostEqual(first["attention_student_mass"].item(), .9, places=6)
        self.assertAlmostEqual(second["attention_student_mass"].item(), .09, places=6)

    def test_validation(self):
        student, teacher = torch.ones(1, 2) / 2, torch.ones(1, 2) / 2
        keep, index, q = torch.ones(1, 2), torch.tensor([[0, 1]]), torch.ones(1, 2)
        cases = [
            ((student, teacher, keep, index, q), {"importance_mode": "semantic"}, ValueError),
            ((student, teacher, keep, index, q), {"eps": math.nan}, ValueError),
            ((student, teacher, keep, index, q), {"eps": 0}, ValueError),
            ((student, teacher, keep, index.int(), q), {}, TypeError),
            ((student, teacher, keep, index, q.long()), {}, TypeError),
            ((student, teacher, keep, index, torch.tensor([[2., 0.]])), {}, ValueError),
            ((student, teacher, keep, index, torch.tensor([[math.nan, 0.]])), {}, ValueError),
            ((torch.tensor([[-.1, .5]]), teacher, keep, index, q), {}, ValueError),
            ((student, teacher, torch.ones(1, 3), index, q), {}, ValueError),
            ((student, teacher, keep, torch.tensor([[0]]), q), {}, ValueError),
            ((student[:, None], teacher, keep, index, q), {}, ValueError),
        ]
        for args, kwargs, exception in cases:
            with self.subTest(kwargs=kwargs, args=[tuple(x.shape) for x in args]):
                with self.assertRaises(exception):
                    trusted_attention_alignment(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()

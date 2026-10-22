"""Loss contracts use tiny tensors and require no checkpoint or GPU."""
import math
import unittest

import torch
import torch.nn.functional as F

from weather_vpr.losses.distillation import (
    conditional_ot_distillation_loss,
    cross_domain_infonce,
    descriptor_alignment,
    keep_alignment_loss,
    patch_magnitude_loss,
    sgpd_patch_loss,
    weighted_patch_losses,
)
from weather_vpr.utils.backend import enable_backend_imports


class LossModeTests(unittest.TestCase):
    def test_saliency_and_uniform_share_support_but_change_weights(self):
        teacher = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
        student = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
        day = {"backbone_local_features": teacher,
               "assignment_keep": torch.tensor([[1., 0.]])}
        weather = {"backbone_local_features": student,
                   "assignment_keep": torch.tensor([[1., 0.]])}
        index = torch.tensor([[0, 1]])
        q = torch.ones(1, 2)
        saliency = weighted_patch_losses(weather, day, index, q, importance_mode="saliency")
        uniform = weighted_patch_losses(weather, day, index, q, importance_mode="uniform")
        self.assertAlmostEqual(saliency["patch"].item(), 0., places=6)
        self.assertAlmostEqual(uniform["patch"].item(), 0.5, places=6)
        self.assertEqual(saliency["positive_patches"].item(), 2)
        self.assertAlmostEqual(uniform["teacher_keep_mass"].item(), 1.0)
        self.assertAlmostEqual(uniform["patch_weight_mass"].item(), 2.0)
        self.assertEqual(saliency["ot"].item(), 0.)
        self.assertEqual(saliency["magnitude"].item(), 0.)

    def test_patch_uses_only_trusted_matches_and_detaches_teacher(self):
        student = torch.tensor([[[0., 1.], [1., 0.]]], requires_grad=True)
        teacher = torch.tensor([[[1., 1.], [0., 0.]]], requires_grad=True)
        keep = torch.tensor([[1., 1.]], requires_grad=True)
        confidence = torch.tensor([[1., 0.]], requires_grad=True)
        index = torch.tensor([[1, -1]])
        loss = sgpd_patch_loss(student, teacher, keep, index, confidence)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertGreater(student.grad[:, :, 0].abs().sum().item(), 0.)
        self.assertEqual(student.grad[:, :, 1].abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(keep.grad)
        self.assertIsNone(confidence.grad)
        with self.assertRaises(ValueError):
            sgpd_patch_loss(student, teacher, keep, index, torch.ones(1, 2))

    def test_keep_is_independent_and_zero_coverage_is_safe(self):
        student = torch.tensor([[0.2, 0.9]], requires_grad=True)
        teacher = torch.tensor([[0.8, 0.1]], requires_grad=True)
        loss = keep_alignment_loss(student, teacher, torch.tensor([[0, -1]]),
                                   torch.tensor([[1., 0.]]))
        self.assertAlmostEqual(loss.item(), 0.36, places=6)
        loss.backward()
        self.assertNotEqual(student.grad[0, 0].item(), 0.)
        self.assertEqual(student.grad[0, 1].item(), 0.)
        self.assertIsNone(teacher.grad)
        empty = keep_alignment_loss(student, teacher, torch.tensor([[-1, -1]]),
                                    torch.zeros(1, 2))
        self.assertEqual(empty.item(), 0.)
        self.assertTrue(torch.isfinite(empty))

    def test_magnitude_smooth_l1_complements_cosine(self):
        student = torch.tensor([[[2.], [0.]]], requires_grad=True)
        teacher = torch.tensor([[[1.], [0.]]], requires_grad=True)
        keep = torch.ones(1, 1)
        index = torch.zeros(1, 1, dtype=torch.long)
        q = torch.ones(1, 1)
        angle = sgpd_patch_loss(student, teacher, keep, index, q)
        magnitude = patch_magnitude_loss(student, teacher, keep, index, q, beta=1.)
        self.assertAlmostEqual(angle.item(), 0., places=6)
        self.assertAlmostEqual(magnitude.item(), 0.25, places=6)
        magnitude.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            patch_magnitude_loss(student, teacher, keep, index, q, beta=-1.)

    def test_conditional_ot_fp32_masks_low_teacher_keep(self):
        # Patch 1 has nearly all of its teacher mass in the dustbin.
        teacher = torch.log(torch.tensor([[[.36, 1e-7], [.54, 1e-7], [.10, 1-2e-7]]]))
        teacher.requires_grad_()
        student = torch.log(torch.tensor([[[.016, .45], [.004, .45], [.98, .10]]]))
        student.requires_grad_()
        index = torch.tensor([[0, 1]])
        q = torch.ones(1, 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = conditional_ot_distillation_loss(student, teacher, index, q,
                                                    min_teacher_keep=1e-4)
        expected = F.kl_div(torch.log(torch.tensor([.8, .2])),
                            torch.log(torch.tensor([.4, .6])),
                            log_target=True, reduction="sum")
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        # Raising student dustbin mass alone leaves the conditional mismatch.
        less_dustbin = student.detach().clone()
        less_dustbin[:, :, 0] = torch.log(torch.tensor([.72, .18, .10]))
        same_routing = conditional_ot_distillation_loss(
            less_dustbin, teacher, index, q, min_teacher_keep=1e-4)
        self.assertAlmostEqual(loss.item(), same_routing.item(), places=5)
        loss.backward()
        self.assertGreater(student.grad[:, :2, 0].abs().sum().item(), 0.)
        self.assertEqual(student.grad[:, :, 1].abs().sum().item(), 0.)
        self.assertEqual(student.grad[:, 2, 0].abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)

    def test_conditional_ot_empty_support_and_wrapper_opt_in(self):
        teacher = torch.zeros(1, 3, 2)
        student = torch.randn(1, 3, 2, requires_grad=True)
        zero = conditional_ot_distillation_loss(
            student, teacher, torch.tensor([[-1, -1]]), torch.zeros(1, 2))
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertEqual(student.grad.abs().sum().item(), 0.)
        aux_s = {"backbone_local_features": torch.ones(1, 2, 1, 2),
                 "assignment_keep": torch.tensor([[.4, .8]]),
                 "log_assignment": student}
        aux_t = {"backbone_local_features": torch.ones(1, 2, 1, 2),
                 "assignment_keep": torch.tensor([[.4, .8]]),
                 "log_assignment": teacher}
        result = weighted_patch_losses(
            aux_s, aux_t, torch.tensor([[0, 1]]), torch.ones(1, 2),
            include_magnitude=True, include_ot=True)
        self.assertEqual(result["magnitude"].item(), 0.)
        self.assertTrue(torch.isfinite(result["ot"]))
        self.assertGreater(result["ot_mass"].item(), 0.)

    def test_conditional_ot_agrees_with_salad_reference(self):
        enable_backend_imports()
        from workflow.synthetic_losses import conditional_ot_distillation_loss as reference
        teacher = torch.log(torch.tensor([[[.3, .1], [.6, .2], [.1, .7]]]))
        student = torch.log(torch.tensor([[[.2, .4], [.7, .4], [.1, .2]]]))
        match = torch.tensor([[1, 0]])
        confidence = torch.tensor([[.8, .5]])
        actual = conditional_ot_distillation_loss(
            student, teacher, match, confidence, min_teacher_keep=.01)
        expected = reference(student, teacher, match, confidence,
                             min_teacher_keep=.01)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    def test_global_alignment_and_discrimination_detach_day(self):
        student = torch.eye(3, requires_grad=True)
        teacher = torch.eye(3, requires_grad=True)
        aligned = descriptor_alignment(student, teacher)
        self.assertAlmostEqual(aligned.item(), 0., places=6)
        same = cross_domain_infonce(student, teacher, ["a", "b", "c"])
        worse = cross_domain_infonce(student, teacher.flip(0), ["a", "b", "c"])
        self.assertLess(same.item(), worse.item())
        (worse + descriptor_alignment(student, teacher.flip(0))).backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)
        self.assertEqual(cross_domain_infonce(student, teacher, ["a"] * 3).item(), 0.)
        with self.assertRaises(ValueError):
            cross_domain_infonce(student, teacher, ["a", "b", "c"], temperature=math.nan)

    def test_invalid_mode(self):
        x = {"backbone_local_features": torch.ones(1, 2, 1, 1),
             "assignment_keep": torch.ones(1, 1)}
        with self.assertRaises(ValueError):
            weighted_patch_losses(x, x, torch.zeros(1, 1, dtype=torch.long),
                                  torch.ones(1, 1), importance_mode="bad")


if __name__ == "__main__":
    unittest.main()

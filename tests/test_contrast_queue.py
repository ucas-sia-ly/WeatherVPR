"""Cross-batch day queue must preserve positives and student-only gradients."""

import unittest

import torch

from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.losses.distillation import cross_domain_infonce


class DayDescriptorQueueTests(unittest.TestCase):
    def test_fixed_bank_does_not_duplicate_current_keys_and_keeps_other_view_positive(self):
        queue = DayDescriptorQueue(3, include_current_keys=False)
        day = torch.tensor([[1., 0.]], requires_grad=True)
        student = torch.tensor([[.8, .2]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'populated'):
            queue.contrastive_loss(student, day, [('A', 1)])
        queue.enqueue(torch.tensor([[1., 0.], [.9, .1], [0., 1.]]),
                      [('A', 1), ('A', 1), ('B', 1)])
        result = queue.contrastive_loss(student, day, [('A', 1)])
        self.assertEqual(result['positive_pairs'], 2)
        self.assertEqual(result['candidate_negative_pairs'], 1)
        result['loss'].backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(day.grad)
        with self.assertRaisesRegex(ValueError, 'positive'):
            queue.contrastive_loss(student, day, [('C', 1)])

    def test_capacity_zero_matches_batch_only_infonce(self):
        student = torch.tensor([[1., 0.], [.9, .1], [0., 1.]], requires_grad=True)
        day = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        places = [("A", 1), ("A", 1), ("B", 1)]
        queue = DayDescriptorQueue(capacity=0)
        result = queue.contrastive_loss(student, day, places, temperature=.2)
        self.assertTrue(torch.allclose(result["loss"], cross_domain_infonce(student, day, places, .2), atol=1e-6))
        self.assertEqual(result["effective_negatives_per_query"], [1, 1, 2])
        self.assertEqual(result["effective_negative_pairs"], 4)
        self.assertEqual(result["positive_pairs"], 5)
        queue.enqueue(day, places)
        self.assertEqual(len(queue), 0)

    def test_queued_same_place_is_positive_not_false_negative(self):
        queue = DayDescriptorQueue(capacity=4)
        queue.enqueue(torch.tensor([[0., 1.], [-1., 0.]]), [("A", 1), ("B", 1)])
        student = torch.tensor([[1., 0.]], requires_grad=True)
        day = torch.tensor([[1., 0.]], requires_grad=True)
        result = queue.contrastive_loss(student, day, [("A", 1)], temperature=1.)
        logits = torch.tensor([1., 0., -1.])
        expected = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[:2], dim=0)
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertEqual(result["positive_pairs"], 2)
        self.assertEqual(result["effective_negative_pairs"], 1)
        self.assertEqual(result["effective_negatives_per_query"], [1])
        result["loss"].backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.)
        self.assertIsNone(day.grad)

    def test_enqueue_detaches_and_evicts_oldest(self):
        queue = DayDescriptorQueue(capacity=2)
        day = torch.tensor([[1., 0.]], requires_grad=True)
        queue.enqueue(day, [("A", 1)])
        queue.enqueue(torch.tensor([[0., 1.]]), [("B", 1)])
        queue.enqueue(torch.tensor([[-1., 0.]]), [("C", 1)])
        self.assertEqual(len(queue), 2)
        self.assertFalse(queue._descriptors.requires_grad)
        # A was evicted; both queued B and C are negatives for a current A.
        result = queue.contrastive_loss(torch.tensor([[1., 0.]]),
                                        torch.tensor([[1., 0.]]), [("A", 1)])
        self.assertEqual(result["effective_negative_pairs"], 2)
        self.assertEqual(result["positive_pairs"], 1)
        queue.clear()
        self.assertEqual(len(queue), 0)

    def test_same_numeric_place_in_different_cities_is_negative(self):
        queue = DayDescriptorQueue(capacity=1)
        queue.enqueue(torch.tensor([[0., 1.]]), [("Rome", 1)])
        result = queue.contrastive_loss(torch.tensor([[1., 0.]]),
                                        torch.tensor([[1., 0.]]), [("Bangkok", 1)])
        self.assertEqual(result["effective_negative_pairs"], 1)

    def test_invalid_dimension_and_temperature(self):
        queue = DayDescriptorQueue(capacity=2)
        with self.assertRaises(ValueError):
            queue.contrastive_loss(torch.ones(1, 2), torch.ones(1, 3), [("A", 1)])
        with self.assertRaises(ValueError):
            queue.contrastive_loss(torch.ones(1, 2), torch.ones(1, 2), [("A", 1)], 0)
        with self.assertRaises(ValueError):
            queue.enqueue(torch.ones(1, 2), [("A", 1), ("B", 1)])

    def test_hard_negatives_keep_all_positives_and_select_student_similarity(self):
        queue = DayDescriptorQueue(capacity=4)
        queued = torch.tensor([[0., 1.], [.8, .6], [-1., 0.], [0., -1.]], requires_grad=True)
        queue.enqueue(queued, [("A", 1), ("B", 1), ("C", 1), ("D", 1)])
        student = torch.tensor([[1., 0.]], requires_grad=True)
        day = torch.tensor([[1., 0.]], requires_grad=True)
        result = queue.contrastive_loss(student, day, [("A", 1)],
                                        temperature=1., hard_negatives=1)
        expected = torch.logsumexp(torch.tensor([1., 0., .8]), 0) - torch.logsumexp(
            torch.tensor([1., 0.]), 0)
        torch.testing.assert_close(result['loss'], expected)
        self.assertEqual(result['positive_pairs'], 2)
        self.assertEqual(result['candidate_negatives_per_query'], [3])
        self.assertEqual(result['selected_negatives_per_query'], [1])
        self.assertEqual(result['effective_negative_pairs'], 1)
        result['loss'].backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertIsNone(day.grad)
        self.assertIsNone(queued.grad)

    def test_hard_negative_limit_above_candidate_count_preserves_old_loss(self):
        queue = DayDescriptorQueue(0)
        student = torch.tensor([[1., .2], [.1, 1.], [-1., .3]], requires_grad=True)
        day = torch.eye(3, 2)
        places = [('A', 1), ('B', 1), ('C', 1)]
        full = queue.contrastive_loss(student, day, places)
        limited = queue.contrastive_loss(student, day, places, hard_negatives=50)
        torch.testing.assert_close(full['loss'], limited['loss'])
        self.assertEqual(full['effective_negatives_per_query'], limited['selected_negatives_per_query'])

    def test_relation_identity_with_matching_temperatures_is_zero(self):
        queue = DayDescriptorQueue(0)
        day = torch.tensor([[1., .1], [-.1, 1.], [-1., -.2]], requires_grad=True)
        student = day.detach().clone().requires_grad_(True)
        result = queue.relational_distillation_loss(
            student, day, [('A', 1), ('B', 1), ('C', 1)], hard_negatives=0)
        self.assertEqual(result['valid_queries'], 3)
        self.assertEqual(result['support_per_query'], [2, 2, 2])
        self.assertAlmostEqual(float(result['loss']), 0., places=6)
        result['loss'].backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(day.grad)

    def test_relation_uses_teacher_topk_and_masks_current_and_queued_same_place(self):
        queue = DayDescriptorQueue(4)
        queued = torch.tensor([[-.5, .5], [.9, .1], [.8, -.6], [0., 1.]], requires_grad=True)
        queue.enqueue(queued, [('A', 1), ('B', 1), ('C', 1), ('D', 1)])
        day = torch.tensor([[1., 0.]], requires_grad=True)
        student = torch.tensor([[0., 1.]], requires_grad=True)
        result = queue.relational_distillation_loss(
            student, day, [('A', 1)], student_temperature=1., teacher_temperature=1.,
            hard_negatives=2)
        # Teacher selects B,C. Student would prefer D, which must not choose support.
        keys = torch.nn.functional.normalize(queued.detach()[1:3], dim=1)
        tlog = torch.log_softmax(day.detach() @ keys.T, dim=1)
        slog = torch.log_softmax(student.detach() @ keys.T, dim=1)
        expected = (tlog.exp() * (tlog - slog)).sum()
        torch.testing.assert_close(result['loss'], expected)
        self.assertEqual(result['candidate_negatives_per_query'], [3])
        self.assertEqual(result['support_per_query'], [2])
        self.assertEqual(result['support_places_per_query'], [2])
        self.assertEqual(result['valid_queries'], 1)
        self.assertGreater(result['teacher_entropy'], 0.)
        result['loss'].backward()
        self.assertGreater(student.grad.abs().sum().item(), 0.)
        self.assertIsNone(day.grad)
        self.assertIsNone(queued.grad)

    def test_relation_requires_two_distinct_negative_places_and_two_support_keys(self):
        for queued_places, hard_negatives in [([('A', 1), ('A', 1)], 0),
                                             ([('B', 1), ('B', 1)], 0),
                                             ([('B', 1), ('C', 1)], 1)]:
            queue = DayDescriptorQueue(2)
            queue.enqueue(torch.tensor([[1., 0.], [0., 1.]]), queued_places)
            student = torch.tensor([[.8, .2]], requires_grad=True)
            day = torch.tensor([[1., 0.]], requires_grad=True)
            result = queue.relational_distillation_loss(
                student, day, [('A', 1)], hard_negatives=hard_negatives)
            self.assertEqual(result['valid_queries'], 0)
            self.assertEqual(result['mean_support'], 0.)
            self.assertEqual(result['teacher_entropy'], 0.)
            self.assertEqual(float(result['loss']), 0.)
            result['loss'].backward()
            torch.testing.assert_close(student.grad, torch.zeros_like(student))
            self.assertIsNone(day.grad)

    def test_relation_excludes_invalid_queries_from_mean(self):
        queue = DayDescriptorQueue(0)
        day = torch.tensor([[1., 0.], [1., 0.], [.8, .6], [-.8, .6]])
        student = torch.tensor([[.5, .5], [.9, .1], [.2, .8], [-.8, .2]], requires_grad=True)
        # A queries see B,C; B and C see A,A plus another place: all valid.
        places = [('A', 1), ('A', 1), ('B', 1), ('C', 1)]
        full = queue.relational_distillation_loss(student, day, places, hard_negatives=0)
        self.assertEqual(full['valid_queries'], 4)
        self.assertEqual(full['support_per_query'], [2, 2, 3, 3])
        limited = queue.relational_distillation_loss(
            student, day, places, student_temperature=1., teacher_temperature=1., hard_negatives=2)
        # B selects A,A, so its support has one place and is excluded. A queries
        # select B,C; C selects B and one A. The loss averages these three only.
        self.assertEqual(limited['valid_queries'], 3)
        self.assertEqual(limited['support_places_per_query'], [2, 2, 1, 2])
        keys = torch.nn.functional.normalize(day, dim=1)
        queries = torch.nn.functional.normalize(student, dim=1)
        expected = []
        for row, columns in ((0, [2, 3]), (1, [2, 3]), (3, [0, 2])):
            tlog = torch.log_softmax(keys[row] @ keys[columns].T, dim=0)
            slog = torch.log_softmax(queries[row] @ keys[columns].T, dim=0)
            expected.append((tlog.exp() * (tlog - slog)).sum())
        torch.testing.assert_close(limited['loss'], torch.stack(expected).mean())
        # A,A,B has only one distinct negative place for each A query.
        partial = queue.relational_distillation_loss(student[:3], day[:3], places[:3], hard_negatives=0)
        self.assertEqual(partial['valid_queries'], 0)
        self.assertTrue(torch.isfinite(full['loss']))

    def test_fp32_losses_under_autocast_and_argument_validation(self):
        queue = DayDescriptorQueue(0)
        student = torch.tensor([[1., .1], [.2, 1.], [-1., .1]], requires_grad=True)
        day = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        places = [('A', 1), ('B', 1), ('C', 1)]
        reference = queue.relational_distillation_loss(student, day, places, hard_negatives=0)['loss']
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            relation = queue.relational_distillation_loss(student, day, places, hard_negatives=0)['loss']
            contrast = queue.contrastive_loss(student, day, places, hard_negatives=1)['loss']
        self.assertEqual(relation.dtype, torch.float32)
        self.assertEqual(contrast.dtype, torch.float32)
        torch.testing.assert_close(relation, reference, rtol=0, atol=0)
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                queue.contrastive_loss(student, day, places, hard_negatives=bad)
            with self.assertRaises(ValueError):
                queue.relational_distillation_loss(student, day, places, hard_negatives=bad)
        with self.assertRaises(ValueError):
            queue.relational_distillation_loss(student, day, places, teacher_temperature=0.)
        with self.assertRaises(ValueError):
            queue.relational_distillation_loss(student, day, places, student_temperature=float('nan'))


if __name__ == "__main__":
    unittest.main()

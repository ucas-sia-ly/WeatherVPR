"""Cross-batch day queue must preserve positives and student-only gradients."""

import unittest

import torch

from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.losses.distillation import cross_domain_infonce


class DayDescriptorQueueTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

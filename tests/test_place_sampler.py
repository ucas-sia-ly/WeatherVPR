"""Place-disjoint batching behavior on small, unbalanced datasets."""

import unittest

from weather_vpr.data.place_sampler import DistinctPlaceBatchSampler


class DistinctPlaceBatchSamplerTests(unittest.TestCase):
    def test_uses_composite_city_and_place_key(self):
        places = [("Bangkok", 1), ("Rome", 1), ("Bangkok", 1), ("Osaka", 2)]
        sampler = DistinctPlaceBatchSampler(places, batch_size=3, seed=7)
        batches = list(sampler)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(4)))
        self.assertEqual(len(batches), len(sampler))
        first_places = {(str(places[i][0]), str(places[i][1])) for i in batches[0]}
        self.assertEqual(len(first_places), 3)

    def test_repeats_place_only_when_distinct_places_exhausted(self):
        places = [("A", 1)] * 3 + [("B", 1)] * 3
        batches = list(DistinctPlaceBatchSampler(places, batch_size=4, seed=1))
        self.assertEqual([len(batch) for batch in batches], [4, 2])
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(6)))
        self.assertNotEqual(places[batches[0][0]], places[batches[0][1]])
        self.assertEqual({places[i] for i in batches[0]}, {("A", 1), ("B", 1)})

    def test_epoch_order_reproducible_and_drop_last(self):
        places = [("City", i) for i in range(10)]
        sampler = DistinctPlaceBatchSampler(places, batch_size=4, seed=42, drop_last=True)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), 2)
        self.assertEqual(sum(map(len, first)), 8)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(list(sampler), list(DistinctPlaceBatchSampler(places, 4, seed=43, drop_last=True)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            DistinctPlaceBatchSampler([], batch_size=2)
        with self.assertRaises(ValueError):
            DistinctPlaceBatchSampler([("A", 1)], batch_size=0)
        with self.assertRaises(ValueError):
            DistinctPlaceBatchSampler([("A",)], batch_size=1)


if __name__ == "__main__":
    unittest.main()

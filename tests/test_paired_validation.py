"""Explicit pair and city/place validation semantics without a real model."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from weather_vpr.data.paired_dataset import sha256
from weather_vpr.inference.paired_validation import load_pair_validation, mean_paired_cosine


class PairedValidationTests(unittest.TestCase):
    def _manifest(self, root):
        (root / "day").mkdir()
        (root / "night").mkdir()
        rows = []
        # The same source can produce multiple Qwen images; another image from
        # the same place is a Recall positive but not the exact cosine target.
        for sample, source, place in (("b", "b", "1"), ("a", "a", "1"),
                                      ("c", "c", "2"), ("d", "a", "1")):
            day = root / "day" / f"{source}.jpg"
            if not day.exists():
                day.write_bytes(source.encode())
            night = root / "night" / f"{sample}.png"
            night.write_bytes(sample.encode())
            rows.append({"sample_id": sample, "city": "A", "place_id": place,
                         "condition": "night", "passed": True,
                         "eligible_for_training": True,
                         "source_path": f"day/{source}.jpg",
                         "output_path": f"night/{sample}.png",
                         "source_sha256": sha256(day), "output_sha256": sha256(night)})
        manifest = root / "night_val.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return manifest

    def test_exact_source_and_same_place_recall_positives(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary))
            dataset, exact, places = load_pair_validation(manifest)
            self.assertEqual([query.id for query in dataset.queries], ["a", "b", "c", "d"])
            self.assertEqual(exact, [0, 1, 2, 0])
            self.assertEqual(dataset.positives, [[0, 1], [0, 1], [2], [0, 1]])
            self.assertEqual(len(set(places)), 2)
            limited, _, _ = load_pair_validation(manifest, limit_pairs=2)
            self.assertEqual(limited.summary()["num_queries"], 2)
            self.assertEqual(limited.summary()["num_references"], 2)

    def test_paired_cosine_uses_exact_source(self):
        gallery = torch.eye(3)
        queries = torch.stack((gallery[0], gallery[1], gallery[2], gallery[0]))
        self.assertAlmostEqual(mean_paired_cosine(gallery, queries, [0, 1, 2, 0]), 1.0)
        self.assertAlmostEqual(mean_paired_cosine(gallery, queries, [1, 0, 2, 1]), 0.25)

    def test_changed_pair_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            (root / "night/a.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "incorrect output_sha256"):
                load_pair_validation(manifest)


if __name__ == "__main__":
    unittest.main()

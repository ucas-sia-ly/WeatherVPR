"""Tiny Tokyo247 protocol fixtures; no real data, model or GPU required."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from weather_vpr.inference.tokyo import load_tokyo_night


class TokyoNightTests(unittest.TestCase):
    def fixture(self, directory, **overrides):
        root = Path(directory)
        # Intentionally unsorted MAT order; threshold has inside/on/outside cases.
        names = ["a/view_b.jpg", "a/view_a.jpg", "a/view_c.jpg"]
        struct = dict(whichSet="test", dbImageFns=np.array(names, dtype=object),
                      qImageFns=np.array(["00001.jpg", "00002.jpg", "00003.jpg"], dtype=object),
                      utmDb=np.array([[25., 0., 25.0001], [0., 0., 0.]]),
                      utmQ=np.array([[0., 0., 0.], [0., 0., 0.]]),
                      numImages=3, numQueries=3, posDistThr=25., posDistSqThr=625.)
        struct.update(overrides)
        mat = root / "tokyo.mat"
        savemat(mat, {"dbStruct": struct})
        for name in names:
            path = root / "database_gsv_vga" / Path(name).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image fixture")
        (root / "queries").mkdir()
        # Column 5 (unused) is N for every image; column 6 is the time label.
        for i, label in enumerate(("D", "N", "S", "N"), 1):
            name = f"{i:05d}.jpg"
            (root / "queries" / name).write_bytes(b"image fixture")
            with (root / "queries" / Path(name).with_suffix(".csv")).open("w", newline="") as stream:
                # Deliberately inconsistent local CSV geometry must not enter GT.
                csv.writer(stream).writerow([name, 1., 2., "F", "N", label, "url", 999., 999.])
        source = root / "source.json"
        source.write_text(json.dumps(dict(
            dataset="Tokyo247", split="test", mat_file=mat.name,
            mat_sha256=hashlib.sha256(mat.read_bytes()).hexdigest(),
            gallery_count=3, official_full_query_count=3, night_query_count=1,
            night_ids=["00002.jpg"])))
        return root, mat, source

    def test_official_subset_time_label_and_complete_gallery_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(temporary)
            dataset = load_tokyo_night(*args)
            self.assertEqual([x.id for x in dataset.queries], ["00002.jpg"])
            self.assertEqual([x.id for x in dataset.references],
                             ["a/view_b.jpg", "a/view_a.jpg", "a/view_c.jpg"])
            self.assertEqual([x.path.suffix for x in dataset.references], [".png"] * 3)
            # Inclusive 25m boundary, canonical MAT coordinates, no reference reorder.
            self.assertEqual(dataset.positives, [[0, 1]])
            self.assertEqual(dataset.summary()["num_evaluated_queries"], 1)
            self.assertEqual(dataset.protocol["positive_radius_meters"], 25.)
            self.assertIn("byte for byte", dataset.protocol["local_query_release_note"])
            self.assertEqual(dataset.protocol["mat_sha256"],
                             hashlib.sha256(args[1].read_bytes()).hexdigest())

    def test_unverified_or_inconsistent_source_is_rejected(self):
        variants = ({"mat_sha256": "0" * 64}, {"mat_sha256": "bad"},
                    {"mat_file": "other.mat"}, {"gallery_count": 4},
                    {"night_ids": ["00004.jpg"]}, {"night_ids": ["00002.jpg"] * 2},
                    {"night_query_count": 2}, {"official_full_query_count": 4})
        for update in variants:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as temporary:
                root, mat, source = self.fixture(temporary)
                metadata = json.loads(source.read_text())
                metadata.update(update)
                source.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    load_tokyo_night(root, mat, source)

    def test_bad_mat_dimensions_ids_and_geometry_are_rejected(self):
        variants = ({"numImages": 4}, {"numQueries": 2},
                    {"utmDb": np.zeros((3, 2))}, {"utmQ": np.full((2, 3), np.nan)},
                    {"posDistThr": 50.}, {"posDistSqThr": 25.}, {"whichSet": "train"},
                    {"dbImageFns": np.array(["a/a.jpg", "a/a.jpg", "a/c.jpg"], dtype=object)},
                    {"qImageFns": np.array(["../q.jpg", "b.jpg", "c.jpg"], dtype=object)})
        for update in variants:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(temporary, **update)
                with self.assertRaises(ValueError):
                    load_tokyo_night(*args)

    def test_missing_images_or_csv_are_not_silently_skipped(self):
        for missing in ("database_gsv_vga/a/view_b.png", "queries/00002.jpg",
                        "queries/00001.jpg", "queries/00002.csv"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(temporary)
                (args[0] / missing).unlink()
                with self.assertRaises(FileNotFoundError):
                    load_tokyo_night(*args)

    def test_invalid_csv_label_or_identity_is_rejected(self):
        for row in (["00002.jpg", 1, 2, "F", "N", "?", "url", 0, 0],
                    ["wrong.jpg", 1, 2, "F", "N", "N", "url", 0, 0],
                    ["00002.jpg", 1, 2, "F", "N"]):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(temporary)
                with (args[0] / "queries/00002.csv").open("w", newline="") as stream:
                    csv.writer(stream).writerow(row)
                with self.assertRaises(ValueError):
                    load_tokyo_night(*args)


if __name__ == "__main__":
    unittest.main()

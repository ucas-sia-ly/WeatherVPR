"""Real supervision requires explicit review, split isolation and exact bytes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFile

from weather_vpr.data.real_pairs import ReviewedRealPairs, file_sha256


SPEC = importlib.util.spec_from_file_location(
    "prepare_real_pair_review",
    Path(__file__).resolve().parents[1] / "scripts/prepare_real_pair_review.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class RealPairTests(unittest.TestCase):
    def test_cached_day_night_only_path_matches_and_does_not_decode_day(self):
        import torch
        self.accept(0)
        self._write_reviews()
        pairs = ReviewedRealPairs(self.root, self.review, image_size=(14, 14))
        expected = pairs[0]['night_tensor']
        original = Path.read_bytes

        def read(path):
            if path.parent.name == 'Day':
                raise AssertionError('Cached Day must not be decoded again')
            return original(path)

        with patch.object(Path, 'read_bytes', read):
            actual = pairs.read_night_tensor(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        day = Path(pairs.rows[0]['day_path'])
        day.write_bytes(day.read_bytes() + b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Cached-target Day image changed'):
            pairs.read_night_tensor(0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        self.rows = []
        for index, name in enumerate(("drive_a", "drive_b", "drive_c")):
            row = {"sample_id": name, "sequence_id": "drive", "content_group": name}
            for side, folder in (("day", "Day"), ("night", "Night")):
                path = self.root / folder / f"{name}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (13, 9))
                image.putdata([(index * 33 + x * 7, (y * 21 + index * 5) % 256,
                                40 if side == "day" else 17)
                               for y in range(9) for x in range(13)])
                image.save(path)
                with Image.open(path) as decoded:
                    decoded.load()
                    pixel_hash = hashlib.sha256(decoded.tobytes()).hexdigest()
                row.update({f"{side}_path": f"{folder}/{name}.jpg",
                            f"{side}_sha256": file_sha256(path),
                            f"{side}_pixel_sha256": pixel_hash})
            self.rows.append(row)
        self._write_source()
        self.review = self.base / "review.jsonl"
        prepare.prepare_review(self.root, self.review)
        self.reviews = [json.loads(line) for line in self.review.read_text().splitlines()]

    def _write_source(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        (self.root / "dataset_info.json").write_text(json.dumps({
            "pairs": len(self.rows), "manifest_sha256": file_sha256(manifest)}))

    def _write_reviews(self):
        self.review.write_text("".join(json.dumps(row) + "\n" for row in self.reviews))

    def accept(self, index=0, split="train", group=None, status="manual"):
        self.reviews[index].update(
            scene_group=group or f"route_{index}", split=split,
            pair_valid=True, visibility_valid=True, review_status=status)

    def dataset(self, **kwargs):
        self._write_reviews()
        return ReviewedRealPairs(self.root, self.review, **kwargs)

    def test_template_preserves_unknown_evidence_and_refuses_overwrite(self):
        for row in self.reviews:
            self.assertIsNone(row["scene_group"])
            self.assertIsNone(row["split"])
            self.assertIsNone(row["pair_valid"])
            self.assertIsNone(row["visibility_valid"])
            self.assertEqual(row["review_status"], "unreviewed")
        with self.assertRaisesRegex(ValueError, "No accepted"):
            self.dataset()
        with self.assertRaises(FileExistsError):
            prepare.prepare_review(self.root, self.review)
        with self.assertRaisesRegex(ValueError, "Unknown sample"):
            prepare.prepare_review(self.root, self.base / "unknown.jsonl", ["missing"])
        result = prepare.prepare_review(self.root, self.base / "subset.jsonl", ["drive_b"])
        self.assertEqual((result["pairs"], result["accepted_pairs"]), (1, 0))

    def test_acceptance_filter_normalization_and_ordered_cache_sources(self):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        self.accept(0, status="automatic")
        self.accept(1, split="dev")
        self.reviews[2].update(pair_valid=False, visibility_valid=True,
                               review_status="manual", reject_reason="background mismatch")
        data = self.dataset(image_size=(5, 7))
        self.assertEqual(len(data), 1)
        item = data[0]
        self.assertEqual(item["review_status"], "automatic")
        self.assertEqual(item["sample_id"], "drive_a")
        self.assertNotIn("place", item)
        self.assertNotIn("place_id", item)
        self.assertEqual(tuple(item["day"].shape), (3, 5, 7))
        self.assertEqual(item["day"].dtype, torch.float32)
        with Image.open(item["day_path"]) as image:
            resized = image.convert("RGB").resize((7, 5), Image.Resampling.BILINEAR)
            raw = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        torch.testing.assert_close(item["day"], (raw - mean) / std, rtol=0, atol=0)
        batch = next(iter(DataLoader(data, batch_size=2)))
        self.assertEqual(tuple(batch["night"].shape), (1, 3, 5, 7))
        self.assertEqual(data.day_sources,
                         [{"path": item["day_path"], "sha256": item["day_sha256"]}])
        identity = data.sources_signature()
        self.assertEqual(identity["review_manifest_sha256"], file_sha256(self.review))
        self.assertEqual(identity["source_manifest_sha256"], file_sha256(data.source_manifest))
        self.assertEqual(data.manifest, self.review)
        self.assertIs(item["day"], item["day_tensor"])
        self.assertIs(item["night"], item["night_tensor"])
        self.assertEqual(identity["source_images"][0]["scene_group"], "route_0")
        self.assertEqual(len(ReviewedRealPairs(self.root, self.review, split="dev")), 1)

    def test_isolation_checks_all_splits_and_rejected_records(self):
        self.accept(0, group="same_route")
        self.accept(1, split="test", group="same_route")
        self.reviews[1].update(pair_valid=False, reject_reason="rejected but assigned")
        with self.assertRaisesRegex(ValueError, "scene_group crosses splits"):
            self.dataset()
        self.reviews[1]["scene_group"] = "other_route"
        self.rows[1]["content_group"] = self.rows[0]["content_group"]
        self._write_source()
        for row in self.reviews:
            row["source_manifest_sha256"] = file_sha256(self.root / "manifest.jsonl")
        with self.assertRaisesRegex(ValueError, "content_group crosses splits"):
            self.dataset()

    def test_exact_reuse_through_unreviewed_source_cannot_cross_splits(self):
        # A.day == C.day and C.night == B.night. Even with separate declared
        # components, the unreviewed C connects A and B by exact content.
        for target, origin, side in ((2, 0, "day"), (2, 1, "night")):
            source, duplicate = self.rows[origin], self.rows[target]
            payload = (self.root / source[f"{side}_path"]).read_bytes()
            (self.root / duplicate[f"{side}_path"]).write_bytes(payload)
            for field in ("sha256", "pixel_sha256"):
                duplicate[f"{side}_{field}"] = source[f"{side}_{field}"]
        self._write_source()
        self.review.unlink()
        prepare.prepare_review(self.root, self.review, ["drive_a", "drive_b"])
        self.reviews = [json.loads(line) for line in self.review.read_text().splitlines()]
        self.accept(0)
        self.accept(1, split="dev")
        with self.assertRaisesRegex(ValueError, "content_group crosses splits"):
            self.dataset()

    def test_scene_and_content_bridge_through_unassigned_review_cannot_cross_splits(self):
        # A(train)--scene--U(unassigned)--exact night--B(dev). Checking scene
        # and content components separately misses this mixed transitive path.
        bridge, dev = self.rows[1], self.rows[2]
        (self.root / bridge["night_path"]).write_bytes(
            (self.root / dev["night_path"]).read_bytes())
        for field in ("sha256", "pixel_sha256"):
            bridge[f"night_{field}"] = dev[f"night_{field}"]
        self._write_source()
        for review, source in zip(self.reviews, self.rows):
            review.update(source_manifest_sha256=file_sha256(self.root / "manifest.jsonl"),
                          night_sha256=source["night_sha256"])
        self.accept(0, group="route_a")
        self.reviews[1]["scene_group"] = "route_a"
        self.accept(2, split="dev", group="route_b")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.dataset()
        # A rejected bridge still records shared route evidence and cannot be
        # removed from the isolation graph merely by setting pair_valid=false.
        self.reviews[1].update(review_status="manual", pair_valid=False,
                               visibility_valid=True, reject_reason="pair mismatch")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            self.dataset()

    def test_review_types_unknown_samples_and_duplicates_are_rejected(self):
        self.accept()
        clean = copy.deepcopy(self.reviews)
        cases = (("pair_valid", 1, "pair_valid must be bool"),
                 ("visibility_valid", "true", "visibility_valid must be bool"),
                 ("scene_group", " ", "scene_group"),
                 ("scene_group", None, "scene_group"),
                 ("split", True, "Invalid split"),
                 ("split", None, "requires split"),
                 ("review_status", "human", "Invalid review_status"),
                 ("reject_reason", "rejected", "cannot have reject_reason"),
                 ("day_sha256", "0" * 64, "day_sha256 differs"),
                 ("source_manifest_sha256", 0, "source_manifest_sha256 differs"),
                 ("sample_id", [], "Unknown reviewed sample_id"))
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                self.reviews = copy.deepcopy(clean)
                self.reviews[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset()
        self.reviews = copy.deepcopy(clean) + [copy.deepcopy(clean[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate reviewed sample_id"):
            self.dataset()
        self.reviews = copy.deepcopy(clean)
        self._write_reviews()
        content = self.review.read_text().replace('"pair_valid": true',
                                                  '"pair_valid": false, "pair_valid": true', 1)
        self.review.write_text(content)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            ReviewedRealPairs(self.root, self.review)

    def test_changed_images_rejected_initially_and_at_read(self):
        self.accept()
        data = self.dataset()
        path = self.root / self.rows[0]["night_path"]
        Image.new("RGB", (13, 9), "white").save(path)
        with self.assertRaisesRegex(RuntimeError, "changed after validation"):
            data[0]
        with self.assertRaisesRegex(ValueError, "bytes differ from source"):
            self.dataset()

    def test_source_manifest_metadata_and_review_binding(self):
        self.accept()
        with (self.root / "manifest.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "differs from dataset_info"):
            self.dataset()
        self._write_source()
        self.reviews[0]["source_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_manifest_sha256 differs"):
            self.dataset()

    def test_source_metadata_cannot_be_overridden_by_review_notes(self):
        self.accept()
        self.reviews[0].update(content_group="invented", day_path="/wrong/path")
        item = self.dataset()[0]
        self.assertEqual(item["content_group"], "drive_a")
        self.assertEqual(item["day_path"], str(self.root / "Day/drive_a.jpg"))

    def test_cache_signature_detects_manifest_changes_during_extraction(self):
        self.accept()
        data = self.dataset()
        data.sources_signature()
        with self.review.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(RuntimeError, "manifest changed after loading"):
            data.sources_signature()

    def test_strict_decode_and_image_size_contract(self):
        self.accept()
        for size in (None, (True, 14), (0, 14), (14,), "14"):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "image_size"):
                    self.dataset(image_size=size)
        data = self.dataset()
        original = ImageFile.LOAD_TRUNCATED_IMAGES
        self.addCleanup(setattr, ImageFile, "LOAD_TRUNCATED_IMAGES", original)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with self.assertRaisesRegex(RuntimeError, "strict Pillow"):
            data[0]


if __name__ == "__main__":
    unittest.main()

"""Train-only provenance and true annotation-backed place expansion."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from weather_vpr.data.research_sources import (
    accepted_qwen_sources, bdd_train_sources, build_gsv_scale_manifests,
    collect_exclusions, darkdriving_train_sources, gsv_csv_filename,
    iter_json_array, load_source_manifest, nightcity_train_sources,
    prepare_research_sources, write_jsonl,
)
from weather_vpr.data.real_pairs import file_sha256


class ResearchSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def image(self, path, color=30):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (11, 9), color=(color, color + 1, color + 2)).save(path)
        return path

    def json(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def fixture_qwen_gsv(self):
        bindings = []
        for place, views in ((1, 1), (2, 3), (3, 2), (4, 2)):
            for view in range(views):
                row = {"place_id": str(place), "year": "2020", "month": "1",
                       "northdeg": str(view), "city_id": "City", "lat": "1.20",
                       "lon": "2.30", "panoid": f"p_{place}_{view}"}
                bindings.append(row)
                self.image(Path("data/gsv-cities/Images/City") / gsv_csv_filename(row), color=place * 20 + view)
        csv_path = self.root / "data/gsv-cities/Dataframes/City.csv"
        csv_path.parent.mkdir(parents=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=bindings[0].keys())
            writer.writeheader()
            writer.writerows(bindings)
        original = self.root / "data/gsv-cities/Images/City" / gsv_csv_filename(bindings[0])
        day = self.root / "data/train/day/night" / original.name
        day.parent.mkdir(parents=True)
        day.write_bytes(original.read_bytes())
        night = self.image("data/train/weather/night/qwen.png", color=3)
        manifest = self.root / "data/splits/night_train.jsonl"
        manifest.parent.mkdir(parents=True)
        write_jsonl(manifest, [{"sample_id": "qwen_0", "condition": "night", "passed": True,
                               "eligible_for_training": True, "accepted": True,
                               "city": "City", "place_id": 1,
                               "source_path": str(day), "output_path": str(night),
                               "source_sha256": file_sha256(day), "output_sha256": file_sha256(night)}])
        self.json("data/splits/heldout_places.json", [["City", "4"]])
        return manifest, csv_path, bindings

    def test_streaming_original_annotation_handles_boundaries_and_rejects_truncation(self):
        value = [{"name": "one", "attributes": {"timeofday": "night"}}, {"name": "two"}]
        path = self.json("array.json", value)
        self.assertEqual(list(iter_json_array(path, chunk_size=3)), value)
        for content in ('[{"name":"bad"}', '[1,]', '[1] null'):
            path.write_text(content)
            with self.assertRaises(ValueError):
                list(iter_json_array(path, chunk_size=2))

    def test_manifest_rejects_non_train_duplicate_and_symlink(self):
        image = self.image("image.jpg")
        path = self.root / "manifest.jsonl"
        row = {"sample_id": "id", "source": "qwen_night", "split": "train",
               "path": str(image), "group_id": "place"}
        write_jsonl(path, [row])
        self.assertEqual(load_source_manifest(path, self.root)[0]["path"], str(image))
        for rows in ([{**row, "split": "val"}], [row, row]):
            path.unlink()
            write_jsonl(path, rows)
            with self.assertRaises(ValueError):
                load_source_manifest(path, self.root)
        link = self.root / "link.jpg"
        link.symlink_to(image)
        path.unlink()
        write_jsonl(path, [{**row, "path": str(link)}])
        with self.assertRaisesRegex(ValueError, "Symlink"):
            load_source_manifest(path, self.root)

    def test_qwen_source_uses_exact_day_and_original_index_not_a_new_label(self):
        manifest, _, _ = self.fixture_qwen_gsv()
        rows, report = accepted_qwen_sources(self.root, manifest, verify_images=True)
        self.assertEqual(report["accepted_pairs"], 1)
        source = next(row for row in rows if row["source"] == "source_night")
        self.assertEqual(source["metadata"]["transform"], "source_night")
        self.assertEqual(source["metadata"]["original_sample_index"], 0)
        self.assertIn("data/train/day/night", source["path"])
        with self.assertRaisesRegex(ValueError, "excluded"):
            accepted_qwen_sources(self.root, manifest, excluded_places={("City", "1")})

    def test_exclusions_union_heldout_validation_calibration_and_reserved(self):
        self.json("data/splits/heldout_places.json", [["City", 1]])
        self.json("configs/dino_salad_calibration_excluded_places.json", [["City", "2"]])
        self.json("third_party/salad/configs/splits/ccr_reserved_places.json",
                  {"groups": {"validation": [["City", 3]], "calibration": [["Else", 1]]}})
        val = self.root / "data/splits/night_val.jsonl"
        write_jsonl(val, [{"city": "City", "place_id": 4}])
        places, evidence = collect_exclusions(self.root)
        self.assertEqual(places, {("City", "1"), ("City", "2"), ("City", "3"),
                                  ("City", "4"), ("Else", "1")})
        self.assertEqual(len(evidence), 4)

    def test_gsv_expansion_is_strict_superset_new_places_multiview_and_deterministic(self):
        manifest, _, _ = self.fixture_qwen_gsv()
        pools, report = build_gsv_scale_manifests(self.root, self.root / "data/gsv-cities", manifest,
            new_places=1, views_per_new_place=2, excluded_places={("City", "4")})
        again, _ = build_gsv_scale_manifests(self.root, self.root / "data/gsv-cities", manifest,
            new_places=1, views_per_new_place=2, excluded_places={("City", "4")})
        self.assertEqual(pools, again)
        self.assertEqual(report["csv_unique_places"], 4)
        self.assertEqual(report["g1"]["unique_places"], 1)
        self.assertEqual(report["g2"]["unique_places"], 2)
        self.assertEqual(len(pools["g2_day"]), 3)
        self.assertTrue({row["sample_id"] for row in pools["g1_day"]}
                        < {row["sample_id"] for row in pools["g2_day"]})
        for row in pools["g2_day"]:
            self.assertNotEqual(row["place_id"], "4")
            self.assertEqual(Path(row["path"]).name, gsv_csv_filename(row["metadata"]["csv_binding"]))
            self.assertIn("gsv-cities/Images/City", row["path"])

    def test_gsv_rejects_changed_canonical_day_and_invented_csv_label(self):
        manifest, csv_path, bindings = self.fixture_qwen_gsv()
        row = json.loads(manifest.read_text())
        Path(row["source_path"]).write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            build_gsv_scale_manifests(self.root, self.root / "data/gsv-cities", manifest, new_places=1)
        original = self.root / "data/gsv-cities/Images/City" / gsv_csv_filename(bindings[0])
        Path(row["source_path"]).write_bytes(original.read_bytes())
        row["place_id"] = 9
        manifest.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "label differs"):
            build_gsv_scale_manifests(self.root, self.root / "data/gsv-cities", manifest, new_places=1)

    def test_bdd_uses_original_train_annotations_even_for_inner_test_folder(self):
        self.image("data/bdd100k/images/train/testB/official_train.jpg")
        self.image("data/bdd100k/images/train/trainB/local_but_unlisted.jpg")
        self.image("data/bdd100k/images/val/official_val.jpg")
        annotation = self.json("data/bdd100k/labels/bdd100k_labels_images_train.json", [
            {"name": "official_train.jpg", "attributes": {"timeofday": "night"}},
            {"name": "missing.jpg", "attributes": {"timeofday": "night"}},
            {"name": "day.jpg", "attributes": {"timeofday": "daytime"}}])
        self.json("data/bdd100k/labels/bdd100k_labels_images_val.json", [
            {"name": "official_val.jpg", "attributes": {"timeofday": "night"}}])
        rows, report = bdd_train_sources(self.root, self.root / "data/bdd100k")
        self.assertEqual([row["sample_id"] for row in rows], ["official_train.jpg"])
        self.assertEqual(rows[0]["metadata"]["local_container"], "testB")
        self.assertEqual(rows[0]["metadata"]["annotation_sha256"], file_sha256(annotation))
        self.assertEqual(report["missing_train_night_images"], 1)

    def test_nightcity_requires_original_train_label_intersection(self):
        base = "data/NightCity/raw/NightCity"
        self.image(f"{base}/NightCity-images/images/train/yes.png")
        self.image(f"{base}/NightCity-images/images/train/unlabelled.png")
        self.image(f"{base}/NightCity-images/images/val/validation.png")
        self.image(f"{base}/NightCity-label/label/train/yes_labelIds.png")
        self.image(f"{base}/NightCity-label/label/val/validation_labelIds.png")
        self.json("data/NightCity/dataset_infos.json", {"splits": {"train": {}}})
        rows, report = nightcity_train_sources(self.root, self.root / "data/NightCity")
        self.assertEqual([row["sample_id"] for row in rows], ["yes"])
        self.assertEqual(report["val_test_images_used"], 0)
        self.assertFalse(rows[0]["metadata"]["geographic_independence_known"])

    def test_darkdriving_accepts_train_only_with_honest_audit_and_source_binding(self):
        data_root = self.root / "data/DarkDriving_Base"
        source, reviews = [], []
        for index, (status, split) in enumerate((("automatic", "train"), ("manual", "dev"),
                                                ("unreviewed", None))):
            name = f"seq_{index}"
            row = {"sample_id": name, "sequence_id": "seq", "content_group": name}
            for side, folder in (("day", "Day"), ("night", "Night")):
                path = self.image(f"data/DarkDriving_Base/{folder}/{name}.jpg",
                                  color=20 + index * 40 + (3 if side == "night" else 0))
                with Image.open(path) as image:
                    image.load()
                    pixel = hashlib.sha256(image.tobytes()).hexdigest()
                row.update({f"{side}_path": f"{folder}/{name}.jpg", f"{side}_sha256": file_sha256(path),
                            f"{side}_pixel_sha256": pixel})
            source.append(row)
            reviews.append({"sample_id": name, "scene_group": name if split else None, "split": split,
                            "pair_valid": True if split else None, "visibility_valid": True if split else None,
                            "review_status": status, "reject_reason": None,
                            "day_sha256": row["day_sha256"], "night_sha256": row["night_sha256"]})
        source_hash = write_jsonl(data_root / "manifest.jsonl", source)
        self.json("data/DarkDriving_Base/dataset_info.json", {"pairs": 3, "manifest_sha256": source_hash})
        for row in reviews:
            row["source_manifest_sha256"] = source_hash
        review = self.root / "review.jsonl"
        write_jsonl(review, reviews)
        rows, report = darkdriving_train_sources(self.root, data_root, review, verify_images=True)
        self.assertEqual([row["sample_id"] for row in rows], ["seq_0"])
        self.assertEqual(report["accepted_all_splits"], 2)
        self.assertEqual(report["review_status_counts"], {"automatic": 1, "manual": 1, "unreviewed": 1})
        self.assertFalse(report["route_independence_certified"])
        self.assertEqual(rows[0]["metadata"]["grouping_kind"], "exact_content_component")
        reviews[0]["night_sha256"] = "0" * 64
        review.write_text("".join(json.dumps(row) + "\n" for row in reviews))
        with self.assertRaisesRegex(ValueError, "differs from source"):
            darkdriving_train_sources(self.root, data_root, review)

    def test_preparation_fresh_outputs_signed_sidecars_and_never_overwrites(self):
        manifest, _, _ = self.fixture_qwen_gsv()
        output = self.root / "audit"
        report = prepare_research_sources(self.root, output, qwen_manifest=manifest,
            include_gsv=True, gsv_new_places=1)
        self.assertFalse(report["training_started"])
        sidecar = json.loads((output / "g2_day.jsonl.metadata.json").read_text())
        self.assertEqual(sidecar["manifest_sha256"], file_sha256(output / "g2_day.jsonl"))
        self.assertIn("audit/excluded_gsv_places.json", sidecar["exclusion_files"])
        self.assertEqual(len(load_source_manifest(output / "train_sources.jsonl", self.root)), 2)
        with self.assertRaises(FileExistsError):
            prepare_research_sources(self.root, output)


if __name__ == "__main__":
    unittest.main()

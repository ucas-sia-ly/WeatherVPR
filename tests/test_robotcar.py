"""RobotCar public-pose diagnostics: geometry and integrity, no model/GPU."""
import tempfile
import unittest
from pathlib import Path

from weather_vpr.inference.robotcar import load_robotcar_public_night


class RobotCarPublicNightTests(unittest.TestCase):
    def _fixture(self, root):
        metadata = root / "metadata/robotcar_v2_train.txt"
        metadata.parent.mkdir(parents=True)
        # camera-to-world: a 90-degree rotation must not rotate C=[5,0,0].
        metadata.write_text(
            "night/rear/100.jpg 0 -1 0 5 1 0 0 0 0 0 1 0 0 0 0 1\n"
            "night-rain/rear/200.jpg 1 0 0 30 0 1 0 0 0 0 1 0 0 0 0 1\n"
            "night/left/300.jpg 1 0 0 5 0 1 0 0 0 0 1 0 0 0 0 1\n")
        for name in ("night/rear/100.jpg", "night-rain/rear/200.jpg", "night/left/300.jpg"):
            image = root / "images" / name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
        models = []
        for index in range(1, 50):
            segment = f"{index:03d}_aligned"
            path = root / "3D-models/individual/colmap_reconstructions" / segment / segment / "images.txt"
            path.parent.mkdir(parents=True)
            camera = ("left", "right", "rear")[(index - 1) % 3]
            name = f"overcast-reference/{camera}/{index:04d}.png"
            image = root / "images" / Path(name).with_suffix(".jpg")
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            if index == 1:
                # COLMAP R=diag(-1,-1,1), t=(30,0,0), C=-R.T@t=(30,0,0).
                pose = "0 0 0 1 30 0 0"
            elif index == 2:
                pose = "1 0 0 0 -30.000001 0 0"
            elif index == 3:
                # Same horizontal position but >25m above: exclude in 3D.
                pose = "1 0 0 0 -5 0 -25.001"
            else:
                pose = f"1 0 0 0 {-1000-index} 0 0"
            path.write_text(f"# Number of images: 1, mean observations per image: 0\n"
                            f"1 {pose} 1 {name}\n\n")
            models.append(path)
        return metadata, models

    def test_pose_directions_inclusive_boundary_and_cross_camera_gallery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            dataset = load_robotcar_public_night(root)
            self.assertEqual(len(dataset.references), 49)
            self.assertEqual([image.id for image in dataset.queries], ["night/rear/100.jpg"])
            positives = [dataset.references[i].id for i in dataset.positives[0]]
            self.assertEqual(positives, ["overcast-reference/left/0001.jpg"])
            self.assertEqual(dataset.protocol["split"], "public_train_diagnostic")
            self.assertFalse(dataset.protocol["official_test"])
            self.assertEqual(dataset.protocol["usage"], "evaluation_only_no_training")
            self.assertEqual(dataset.protocol["positive_radius_meters"], 25)
            self.assertIsNone(dataset.protocol["orientation_filter"])
            self.assertEqual(len(dataset.protocol["colmap_pose_models"]), 49)
            self.assertEqual(len(dataset.protocol["colmap_pose_bundle_sha256"]), 64)

    def test_night_rain_is_a_separate_public_query_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, _ = self._fixture(root)
            # Browser-downloaded metadata may keep its URL query suffix.
            metadata.rename(metadata.with_name(metadata.name + "?utm_source=chatgpt.com"))
            dataset = load_robotcar_public_night(root, "night-rain")
            self.assertEqual([q.id for q in dataset.queries], ["night-rain/rear/200.jpg"])
            self.assertIn("night-rain", dataset.name)
            self.assertEqual(len(dataset.positives[0]), 2)
            with self.assertRaisesRegex(ValueError, "condition"):
                load_robotcar_public_night(root, "dusk")

    def test_rejects_duplicate_and_unsafe_query_names(self):
        for replacement in ("duplicate", "../../100.jpg"):
            with self.subTest(value=replacement), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                metadata, _ = self._fixture(root)
                text = metadata.read_text()
                if replacement == "duplicate":
                    metadata.write_text(text + text.splitlines()[0] + "\n")
                else:
                    metadata.write_text(text.replace("night/rear/100.jpg", replacement))
                with self.assertRaises(ValueError):
                    load_robotcar_public_night(root)

    def test_rejects_nonfinite_nonrigid_or_invalid_homogeneous_query_pose(self):
        for before, after in (("0 -1 0 5", "0 -1 0 nan"),
                              ("0 -1 0 5", "0 -2 0 5"),
                              ("0 0 0 1\n", "0 0 0 2\n")):
            with self.subTest(value=after), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                metadata, _ = self._fixture(root)
                metadata.write_text(metadata.read_text().replace(before, after, 1))
                with self.assertRaises(ValueError):
                    load_robotcar_public_night(root)

    def test_rejects_duplicate_gallery_name_and_bad_colmap_quaternion(self):
        for malformed in ("duplicate", "quaternion"):
            with self.subTest(value=malformed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, models = self._fixture(root)
                if malformed == "duplicate":
                    text = models[1].read_text().replace("overcast-reference/right/0002.png",
                                                        "overcast-reference/left/0001.png")
                    models[1].write_text(text)
                else:
                    models[0].write_text(models[0].read_text().replace("1 0 0 0 1 30", "1 0 0 0 0 30"))
                with self.assertRaises(ValueError):
                    load_robotcar_public_night(root)

    def test_rejects_missing_model_or_gallery_and_queries_without_positives(self):
        for malformed in ("model", "gallery", "query", "positives"):
            with self.subTest(value=malformed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                metadata, models = self._fixture(root)
                if malformed == "model":
                    models[-1].unlink()
                elif malformed == "gallery":
                    (root / "images/overcast-reference/left/0001.jpg").unlink()
                elif malformed == "query":
                    (root / "images/night/rear/100.jpg").unlink()
                else:
                    metadata.write_text(metadata.read_text().replace("0 -1 0 5", "0 -1 0 -50000"))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    load_robotcar_public_night(root)

    def test_rejects_disagreeing_public_reference_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, _ = self._fixture(root)
            metadata.write_text(metadata.read_text() +
                                "overcast-reference/left/0001.jpg 1 0 0 999 0 1 0 0 0 0 1 0 0 0 0 1\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_robotcar_public_night(root)


if __name__ == "__main__":
    unittest.main()

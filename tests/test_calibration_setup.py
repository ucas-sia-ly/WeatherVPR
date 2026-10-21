"""Small, CPU-free checks for the packaged Stage 2B calibration contract."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_calibration", ROOT / "scripts/prepare_calibration.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CalibrationSetupTests(unittest.TestCase):
    def setUp(self):
        self.packaged = ROOT / "configs/dino_salad_stage2b_calibration.json"
        self.payload = json.loads(self.packaged.read_text(encoding="utf-8"))
        self.teacher_hash = self.payload["provenance"]["teacher"]["checkpoint_sha256"]

    def test_packaged_calibration_and_exclusions(self):
        _, places = prepare.load_calibration(self.packaged, self.teacher_hash)
        exclusions = json.loads((ROOT / "configs/dino_salad_calibration_excluded_places.json")
                                .read_text(encoding="utf-8"))
        self.assertEqual([list(p) for p in places], exclusions)
        self.assertEqual(len(places), 12)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare.load_calibration(self.packaged, "0" * 64)

    def test_vendored_copy_prepares_without_historical_workspace(self):
        """Neither old image paths nor the historical pair inventory are read."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = directory / "dino_salad.ckpt"
            checkpoint.write_bytes(b"test-only placeholder")
            output = directory / "calibration.json"
            exclusions = directory / "excluded.json"
            provenance = directory / "provenance.json"
            argv = ["prepare_calibration.py", "--checkpoint", str(checkpoint),
                    "--train-manifest", str(directory / "missing_manifest.jsonl"),
                    "--train-day-folder", str(directory / "missing_images"),
                    "--output-calibration", str(output),
                    "--output-exclusions", str(exclusions),
                    "--output-provenance", str(provenance)]
            original_sha256 = prepare.sha256

            def checkpoint_stub(path):
                return self.teacher_hash if path == checkpoint else original_sha256(path)

            with patch.object(sys, "argv", argv), patch.object(
                    prepare, "DEFAULT_SOURCE", self.packaged), patch.object(
                    prepare, "sha256", side_effect=checkpoint_stub), redirect_stdout(io.StringIO()):
                prepare.main()
                prepare.main()  # Idempotent after moving away from old WORKSPACE.
            self.assertEqual(output.read_bytes(), self.packaged.read_bytes())
            self.assertEqual(len(json.loads(exclusions.read_text())), 12)
            self.assertTrue(json.loads(provenance.read_text())["historical_paths_are_provenance_only"])

    def test_training_overlap_is_reported_by_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            day = directory / "day"
            day.mkdir()
            (day / "Bangkok_0004068_view.jpg").write_bytes(b"x")
            (day / "Bangkok_0009999_view.jpg").write_bytes(b"x")
            manifest = directory / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "city": "Bangkok", "place_id": 4068,
                "source_path": "Bangkok_0004068_view.jpg",
            }) + "\n", encoding="utf-8")
            report = prepare.inspect_training(day, manifest, {("Bangkok", "4068")})
            self.assertEqual(report["day_overlap"], [["Bangkok", "4068"]])
            self.assertEqual(report["manifest_overlap"], [["Bangkok", "4068"]])


if __name__ == "__main__":
    unittest.main()

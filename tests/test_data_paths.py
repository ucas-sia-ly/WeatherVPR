"""Ensure paths resolve within independent project, without WORKSPACE."""
import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from weather_vpr.data.paired_dataset import resolve_file
from weather_vpr.utils.backend import project_root
class DataPathTests(unittest.TestCase):
    def test_manifest_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/'train/manifest.jsonl'
            manifest.parent.mkdir()
            image=manifest.parent/'day/a.jpg'
            image.parent.mkdir()
            image.write_bytes(b'not-an-image')
            self.assertEqual(resolve_file('day/a.jpg',manifest),image.resolve())
    def test_invalid_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'manifest.jsonl'
            with self.assertRaises(ValueError):resolve_file('missing/a.jpg',manifest)
    def test_own_root(self):
        self.assertTrue((project_root()/'README.md').is_file())
        self.assertTrue((project_root()/'scripts/train.py').is_file())
if __name__=='__main__':unittest.main()

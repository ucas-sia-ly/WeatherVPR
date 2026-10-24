"""Cleaning must preserve pairing, recoverability, and strict image validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from weather_vpr.data.darkdriving import DarkDrivingPairs, load_darkdriving_manifest

SPEC = importlib.util.spec_from_file_location(
    'prepare_darkdriving', Path(__file__).resolve().parents[1] / 'scripts/prepare_darkdriving.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class DarkDrivingTests(unittest.TestCase):
    def make_audit(self, root, base):
        records = []
        # a/b: full duplicate; a/c: only Day duplicate; d: broken Day image.
        for name, day, night in [('seq_a', 100, 30), ('seq_b', 100, 30),
                                 ('seq_c', 100, 50), ('seq_d', None, 70)]:
            for side, color in [('Day', day), ('Night', night)]:
                path = root / side / f'{name}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                if color is None:
                    path.write_bytes(b'broken JPEG')
                else:
                    Image.new('RGB', (12, 8), (color, color, color)).save(path)
                row = {'name': path.name, 'path': str(path.relative_to(root)),
                       'split': side, 'errors': [], 'warnings': [],
                       'sha256': prepare.file_hash(path), 'bytes': path.stat().st_size}
                try:
                    with Image.open(path) as image:
                        image.load()
                        row.update(width=image.width, height=image.height,
                                   pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                except OSError as exc:
                    row['errors'] = [str(exc)]
                records.append(row)
        audit = base / 'audit.jsonl'
        audit.write_text(''.join(json.dumps(row) + '\n' for row in records))
        return audit

    def clean(self, base):
        root = base / 'dataset'
        audit = self.make_audit(root, base)
        backup = base / 'backup'
        prepare.prepare(root, audit, backup, workers=2)
        return root, backup

    def test_cleanup_retains_single_side_reuse_and_backs_up_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, backup = self.clean(Path(tmp))
            self.assertEqual(sorted(p.stem for p in (root / 'Day').iterdir()), ['seq_a', 'seq_c'])
            self.assertEqual({p.name for p in (root / 'Day').iterdir()},
                             {p.name for p in (root / 'Night').iterdir()})
            journal = json.loads((backup / 'moves.json').read_text())
            self.assertEqual(len(journal), 4)
            for item in journal:
                self.assertEqual(prepare.file_hash(item['backup_path']), item['sha256'])
                self.assertFalse(Path(item['original_path']).exists())
            result = prepare.validate(root, 2)
            self.assertEqual(result['images'], 4)
            self.assertEqual(result['errors'], 0)
            rows = load_darkdriving_manifest(root)
            self.assertEqual(rows[0]['content_group'], rows[1]['content_group'])

    def test_stale_audit_refuses_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'dataset'
            audit = self.make_audit(root, base)
            (root / 'Day/seq_a.jpg').write_bytes(b'changed after audit')
            with self.assertRaisesRegex(ValueError, 'changed since the audit'):
                prepare.prepare(root, audit, base / 'backup', 2)
            self.assertFalse((base / 'backup').exists())
            self.assertEqual(len(list((root / 'Day').iterdir())), 4)

    def test_mutated_or_extra_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.clean(Path(tmp))
            (root / 'Day/extra.jpg').write_bytes(b'unexpected')
            with self.assertRaisesRegex(ValueError, 'extra entries'):
                DarkDrivingPairs(root)
            (root / 'Day/extra.jpg').unlink()
            (root / 'Day/seq_a.jpg').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Image bytes differ'):
                prepare.validate(root, 2)

    def test_loader_relocation_and_default_tensor_batches(self):
        import torch
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, _ = self.clean(base)
            relocated = base / 'relocated'
            root.rename(relocated)
            dataset = DarkDrivingPairs(relocated, image_size=(16, 24))
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch['day'].shape), (2, 3, 16, 24))
            self.assertEqual(batch['night'].dtype, torch.float32)
            self.assertTrue(torch.isfinite(batch['day']).all())
            self.assertGreaterEqual(batch['night'].min().item(), 0)
            self.assertLessEqual(batch['night'].max().item(), 1)
            self.assertIsInstance(DarkDrivingPairs(relocated, image_size=None)[0]['day'], Image.Image)


if __name__ == '__main__':
    unittest.main()

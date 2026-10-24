"""Portable, cleaned DarkDriving pairs; no inferred place labels or train split."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageFile


def load_darkdriving_manifest(root: Path) -> list[dict]:
    """Require exact agreement between the manifest and both image directories."""
    root = Path(root).expanduser().resolve(strict=True)
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError('Empty DarkDriving manifest')
    seen = set()
    content = set()
    for row in rows:
        name = row['sample_id']
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f'Invalid sample_id: {name!r}')
        if name in seen:
            raise ValueError(f'Duplicate sample_id: {name}')
        seen.add(name)
        signature = (row['day_sha256'], row['night_sha256'])
        if signature in content:
            raise ValueError(f'Duplicate complete Day/Night content: {name}')
        content.add(signature)
        for side, folder in (('day', 'Day'), ('night', 'Night')):
            expected = f'{folder}/{name}.jpg'
            if row[f'{side}_path'] != expected:
                raise ValueError(f'Unexpected {side} path for {name}')
            path = root / expected
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Missing or symlinked image: {path}')
        if not row.get('content_group') or not row.get('sequence_id'):
            raise ValueError(f'Missing grouping metadata: {name}')
    expected_names = {f'{name}.jpg' for name in seen}
    for folder in ('Day', 'Night'):
        actual = {p.name for p in (root / folder).iterdir()}
        if actual != expected_names:
            raise ValueError(f'{folder} differs from manifest: '
                             f'{len(expected_names - actual)} missing, '
                             f'{len(actual - expected_names)} extra entries')
    return rows


class DarkDrivingPairs:
    """Map-style dataset usable by PyTorch DataLoader without model dependencies.

    Default images are RGB float32 tensors [3, 224, 224] in [0, 1], without
    model normalization. ``image_size`` is (height, width); None returns PIL
    images. A paired ``transform(day, night)`` overrides this default, allowing
    callers to share random geometric augmentation between both views.
    ``content_group`` identifies exact-image reuse, not a geographic place.
    """

    def __init__(self, root: Path | str, image_size=(224, 224), *, transform=None):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.rows = load_darkdriving_manifest(self.root)
        if image_size is not None and (
                len(image_size) != 2 or any(
                    not isinstance(x, int) or isinstance(x, bool) or x < 1
                    for x in image_size)):
            raise ValueError('image_size must be two positive integers or None')
        self.image_size = tuple(image_size) if image_size is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise RuntimeError('DarkDriving requires strict Pillow image decoding')
        row = self.rows[index]
        images = []
        for side in ('day', 'night'):
            with Image.open(self.root / row[f'{side}_path']) as image:
                image.load()
                images.append(image.convert('RGB'))
        if self.transform is not None:
            day, night = self.transform(*images)
        elif self.image_size is None:
            day, night = images
        else:
            import numpy as np
            import torch

            height, width = self.image_size
            day, night = [torch.from_numpy(np.array(
                image.resize((width, height), Image.Resampling.BICUBIC), copy=True
            )).permute(2, 0, 1).contiguous().float().div_(255) for image in images]
        return {'sample_id': row['sample_id'], 'sequence_id': row['sequence_id'],
                'content_group': row['content_group'], 'day': day, 'night': night}

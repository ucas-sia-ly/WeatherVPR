"""Exact source/accepted Qwen pairs; never pair by place name alone."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from weather_vpr.utils.backend import project_root
from weather_vpr.data.night_augmentation import NightCameraAugmentation, NightAugmentationConfig
from weather_vpr.data.student_augmentation import sample_augmentation_seed, student_training_tensor

def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_file(value, manifest):
    """Accept exactly one existing path, relative to manifest or project root.

    Never guess pairs from image names/place IDs. Absolute paths are accepted
    only when they exist; moved manifests should use project-relative paths.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    candidates = [manifest.parent / path, project_root() / path]
    found = {c.resolve() for c in candidates if c.is_file()}
    if len(found) != 1:
        raise ValueError(f'Missing/ambiguous path {value} in {manifest}; '
                         'rewrite source_path/output_path for the new project')
    return found.pop()


class AcceptedWeatherPairs(Dataset):
    def __init__(self, manifest: Path, weather: str, image_size=(224, 224),
                 exclude_places: Path | None = None, verify_hashes=False,
                 night_augmentation: NightAugmentationConfig | None = None, seed: int = 42):
        self.manifest = manifest.expanduser().resolve(strict=True)
        self.size = tuple(image_size)
        self.augmentation = NightCameraAugmentation(night_augmentation or NightAugmentationConfig())
        self.seed = int(seed)
        self.epoch = 0
        excluded = set()
        if exclude_places is not None:
            content = json.loads(exclude_places.read_text())
            excluded = {tuple(map(str, p)) for p in content}
        self.rows = []
        for line_idx, line in enumerate(self.manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get('condition', '').lower() != weather.lower()
                or row.get('passed') is not True
                or row.get('eligible_for_training') is not True):
                continue
            if 'city' not in row or 'place_id' not in row:
                raise ValueError(f'Line {line_idx}: missing city/place_id')
            place = (str(row['city']), str(row['place_id']))
            if place in excluded:
                continue
            source = resolve_file(row['source_path'], self.manifest)
            target = resolve_file(row['output_path'], self.manifest)
            if source == target:
                raise ValueError(f'Line {line_idx}: source and generated are identical')
            if verify_hashes:
                for key, path in (('source_sha256', source), ('output_sha256', target)):
                    if row.get(key) and sha256(path) != row[key]:
                        raise ValueError(f'Line {line_idx}: changed file: {path}')
            self.rows.append({**row, 'source_file': source, 'target_file': target,
                              'place': place})
        if not self.rows:
            raise ValueError(f'No accepted {weather} pairs in {self.manifest}')

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        # DataLoader workers are re-created for every epoch (persistent_workers=False).
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row['source_file']) as im:
            day = im.convert('RGB')
        with Image.open(row['target_file']) as im:
            weather = im.convert('RGB')
        from workflow.reliability_cache import preprocess_teacher_image
        student_tensor, augmentation_record = student_training_tensor(
            weather, self.size, self.augmentation,
            seed=sample_augmentation_seed(self.seed, self.epoch, idx))
        return dict(index=idx, row=row, day_image=day, weather_image=weather,
                    day_tensor=preprocess_teacher_image(day, self.size),
                    weather_tensor=preprocess_teacher_image(weather, self.size),
                    student_tensor=student_tensor, augmentation_record=augmentation_record,
                    place=row['place'])

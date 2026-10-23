"""Real SVOX night training data with a geometry-only test exclusion zone.

Only filenames from the test directories are inspected. Test images and test
descriptors are never opened here. All positives are recomputed from retained
SVOX train gallery images with the native 25 metre UTM protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

from PIL import Image
import torch
from torch.utils.data import Dataset
from weather_vpr.data.night_augmentation import NightCameraAugmentation, NightAugmentationConfig
from weather_vpr.data.student_augmentation import sample_augmentation_seed, student_training_tensor

from weather_vpr.inference.svox import load_night_dataset
from weather_vpr.utils.backend import enable_backend_imports

enable_backend_imports()
from workflow.evaluation import EvaluationImage, EvaluationSet, coordinate_positives  # noqa: E402
from workflow.reliability_cache import preprocess_teacher_image  # noqa: E402


def utm_from_name(path: Path) -> tuple[float, float]:
    fields = path.name.split('@')
    try:
        point = float(fields[1]), float(fields[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'Expected @UTM_Easting@UTM_Northing@ filename: {path}') from exc
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f'Nonfinite UTM coordinate: {path}')
    return point


@dataclass(frozen=True)
class RealNightTrainSet:
    full: EvaluationSet
    filtered: EvaluationSet
    original_reference_indices: tuple[int, ...]
    original_query_indices: tuple[int, ...]
    test_geometry_sha256: str
    test_geometry_count: int
    test_buffer_meters: float


def load_real_night_train(svox_root: Path, *, test_buffer_meters: float = 250.0,
                          positive_radius_meters: float = 25.0) -> RealNightTrainSet:
    """Exclude train images near any test query or gallery *coordinate*.

    The test root is used only to list filenames. This protects against the
    near-duplicate RobotCar frames at the native train/test route boundary.
    """
    if not math.isfinite(test_buffer_meters) or test_buffer_meters <= positive_radius_meters:
        raise ValueError('test_buffer_meters must exceed the positive radius')
    if positive_radius_meters != 25.0:
        raise ValueError('SVOX real-night training uses the native 25 m positive radius')
    root = Path(svox_root).expanduser().resolve(strict=True)
    full = load_night_dataset(root, split='train')
    test_dir = root / 'images' / 'test'
    names = sorted(list((test_dir / 'gallery').glob('*.jpg')) +
                   list((test_dir / 'queries_night').glob('*.jpg')))
    if not names:
        raise FileNotFoundError(f'Test geometry directories are missing or empty: {test_dir}')
    digest = hashlib.sha256()
    test_geometry = []
    for path in names:
        coordinate = utm_from_name(path)
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(b'\n')
        test_geometry.append(EvaluationImage(relative, path))

    # The upstream grid routine parses coordinates from filenames and never
    # opens image bytes. It handles 39k test coordinates without scipy.
    near_reference = coordinate_positives(test_geometry, full.references, test_buffer_meters)
    near_query = coordinate_positives(test_geometry, full.queries, test_buffer_meters)
    reference_indices = tuple(i for i, matches in enumerate(near_reference) if not matches)
    query_indices = tuple(i for i, matches in enumerate(near_query) if not matches)
    if not reference_indices or not query_indices:
        raise ValueError('Test exclusion buffer removed all SVOX train images')
    references = [full.references[i] for i in reference_indices]
    queries = [full.queries[i] for i in query_indices]
    positives = coordinate_positives(references, queries, positive_radius_meters)
    missing = sum(not group for group in positives)
    if missing:
        raise ValueError(f'{missing} retained train queries lack a 25 m positive after test exclusion')
    filtered = EvaluationSet('SVOX', references, queries, positives, {
        'ground_truth': 'utm_radius', 'positive_radius_meters': 25.0,
        'split': 'train', 'query_subdirs': ['queries_night'],
        'test_geometry_buffer_meters': float(test_buffer_meters),
    })
    filtered.validate(check_images=False)
    return RealNightTrainSet(full, filtered, reference_indices, query_indices,
                             digest.hexdigest(), len(test_geometry), float(test_buffer_meters))


def northing_holdout(images: list[EvaluationImage], *, fraction: float = 0.2,
                     gap_meters: float = 50.0) -> tuple[list[int], list[int], list[int]]:
    """Reserve the northern route tail for development, with a spatial gap.

    A fraction of zero trains on every retained query. The split is spatial and
    deterministic; nearby frames cannot straddle the training/dev boundary.
    """
    if not math.isfinite(fraction) or not 0 <= fraction < 0.5:
        raise ValueError('validation fraction must be in [0, 0.5)')
    if not math.isfinite(gap_meters) or gap_meters < 0:
        raise ValueError('validation gap must be finite and nonnegative')
    count = len(images)
    if count < 1:
        raise ValueError('No real-night queries to split')
    if fraction == 0:
        return list(range(count)), [], []
    if count < 3:
        raise ValueError('Spatial development split needs at least three queries')
    northings = [utm_from_name(image.path)[1] for image in images]
    holdout_count = max(1, round(count * fraction))
    threshold = sorted(northings)[count - holdout_count]
    train = [i for i, y in enumerate(northings) if y < threshold - gap_meters]
    dev = [i for i, y in enumerate(northings) if y >= threshold]
    buffer = [i for i, y in enumerate(northings) if threshold - gap_meters <= y < threshold]
    if not train or not dev:
        raise ValueError('Spatial development split has no train or development queries')
    return train, dev, buffer


class SVOXNightQueryDataset(Dataset):
    """Read only selected SVOX train/night query images."""

    def __init__(self, images: list[EvaluationImage], selected: list[int],
                 image_size: tuple[int, int] = (224, 224),
                 night_augmentation: NightAugmentationConfig | None = None, seed: int = 42):
        self.images = images
        self.selected = list(selected)
        self.image_size = image_size
        self.augmentation = NightCameraAugmentation(night_augmentation or NightAugmentationConfig())
        self.seed = int(seed)
        self.epoch = 0
        if not self.selected or any(i < 0 or i >= len(images) for i in self.selected):
            raise ValueError('Selected query indices are empty or out of range')

    def __len__(self):
        return len(self.selected)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        query_index = self.selected[index]
        path = self.images[query_index].path
        with Image.open(path) as image:
            clean = image.convert('RGB')
        tensor = preprocess_teacher_image(clean, self.image_size)
        student_tensor, record = student_training_tensor(
            clean, self.image_size, self.augmentation,
            seed=sample_augmentation_seed(self.seed, self.epoch, query_index))
        return {'clean_tensor': tensor, 'student_tensor': student_tensor,
                'query_index': query_index, 'augmentation_record': record}

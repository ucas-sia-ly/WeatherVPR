"""Training input augmentation while retaining clean Teacher/correspondence views."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from PIL import Image

from weather_vpr.data.night_augmentation import NightCameraAugmentation, NightAugmentationConfig

if TYPE_CHECKING:
    from weather_vpr.data.source_night import SourceNightAugmentation


SEED_POLICY = 'sha256(training_seed, zero_based_epoch, original_sample_index), first 64 bits'
SOURCE_NIGHT_SEED_POLICY = (
    'sha256(training_seed:zero_based_epoch:original_sample_index:source_night), first 64 bits'
)


def sample_augmentation_seed(training_seed: int, epoch: int, sample_index: int,
                             *, view: str = 'weather') -> int:
    """Worker-independent seeds with separate weather and source-night streams.

    The default weather payload is unchanged, preserving historical runs.
    """
    if view not in ('weather', 'source_night'):
        raise ValueError("augmentation view must be 'weather' or 'source_night'")
    payload = f'{int(training_seed)}:{int(epoch)}:{int(sample_index)}'
    if view != 'weather':
        payload += f':{view}'
    value = payload.encode('ascii')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


def student_training_tensor(image: Image.Image, image_size: tuple[int, int],
                            augmentation: NightCameraAugmentation | SourceNightAugmentation,
                            *, seed: int):
    """Degrade network-resolution pixels, then use original SALAD normalization."""
    from workflow.reliability_cache import preprocess_teacher_image
    height, width = image_size
    # This new PIL object ensures the clean original image cannot be mutated.
    resized = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    degraded, record = augmentation(resized, seed=seed)
    return preprocess_teacher_image(degraded, image_size), record


def augmentation_metadata(config: NightAugmentationConfig, training_seed: int,
                          image_size: tuple[int, int], epoch_offset: int = 0) -> dict:
    return {
        **config.to_dict(),
        'effective_operation_probabilities': NightCameraAugmentation(config).probabilities,
        'training_seed': int(training_seed),
        'epoch_offset': int(epoch_offset),
        'seed_policy': SEED_POLICY,
        'resolution_hw': list(image_size),
        'order': 'RGB -> original SALAD bilinear resize -> camera augmentation -> ImageNet normalization',
        'scope': 'training Student weather/query only; Teacher, correspondence verification and evaluation stay clean',
    }


def sampled_augmentation_counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        for name in record.get('operations', {}):
            counts[name] = counts.get(name, 0) + 1
    return {'samples': len(records),
            'applied_samples': sum(bool(record.get('applied')) for record in records),
            'operations': counts}


def merge_augmentation_counts(total: dict, batch: dict) -> None:
    for key in ('samples', 'applied_samples'):
        total[key] = total.get(key, 0) + batch[key]
    operations = total.setdefault('operations', {})
    for name, count in batch['operations'].items():
        operations[name] = operations.get(name, 0) + count

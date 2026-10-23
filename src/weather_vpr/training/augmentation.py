"""Explicit Student augmentation configuration for paired training."""
from __future__ import annotations

import json
from pathlib import Path

from weather_vpr.data.night_augmentation import NightAugmentationConfig


def add_augmentation_arguments(parser) -> None:
    parser.add_argument('--night-augmentation', choices=('off', 'camera'), default='off',
                        help='Training Student camera degradation profile')
    parser.add_argument('--augmentation-strength', type=float, default=1.0)
    parser.add_argument('--augmentation-probability', type=float, default=0.8)
    parser.add_argument('--augmentation-epoch-offset', type=int, default=0,
                        help='Add to zero-based epoch for reproducible staged continuation draws')
    parser.add_argument('--augmentation-config', type=Path,
                        help='JSON overrides for camera operation probabilities and ranges; CLI profile/strength/probability win')


def load_augmentation_config(args) -> NightAugmentationConfig:
    if args.augmentation_epoch_offset < 0:
        raise ValueError('augmentation-epoch-offset must be nonnegative')
    values = {}
    if args.augmentation_config is not None:
        path = args.augmentation_config.expanduser().resolve(strict=True)
        values = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError('augmentation-config must contain a JSON object')
    values.update(profile=args.night_augmentation,
                  strength=args.augmentation_strength,
                  probability=args.augmentation_probability)
    return NightAugmentationConfig(**values)


def training_configuration(args, augmentation: dict) -> dict:
    return {**{key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()},
            'effective_night_augmentation': augmentation}

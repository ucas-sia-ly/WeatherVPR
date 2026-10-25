"""Independent, CSV-labelled GSV Day/Source Night coverage experiments.

The Qwen stream, its fixed Day bank, iteration budget and RNG are unchanged.
Only this branch uses the separate signed Source Day bank. No real-night
sample is assigned a fabricated geographical place label.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from weather_vpr.data.gsv_day import GSVDayImages
from weather_vpr.data.source_night import SourceNightConfig
from weather_vpr.data.student_augmentation import sampled_augmentation_counts
from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.training.auxiliary import GroupBalancedCycle, auxiliary_rng_context, auxiliary_seed
from weather_vpr.training.day_bank import load_or_build_day_bank


def source_stream_enabled(args):
    return (getattr(args, 'source_day_manifest', None) is not None
            and getattr(args, 'source_night_weight', 0) > 0)


def source_seed(seed, *parts):
    return auxiliary_seed(seed, 'independent_gsv_source_v1', *parts)


def prepare_source_dataset(args, image_size, source_night_config=None):
    """Validate labels/exclusions before model allocation; disabled means no I/O."""
    if not source_stream_enabled(args):
        return None
    if source_night_config is None:
        config_path = getattr(args, 'source_night_config', None)
        values = (json.loads(Path(config_path).expanduser().resolve(strict=True).read_text())
                  if config_path is not None else {})
        source_night_config = SourceNightConfig.from_dict(values)
    with auxiliary_rng_context(args.device, source_seed(args.seed, 'prepare')):
        return GSVDayImages(args.source_day_manifest, image_size,
                            exclude_places=getattr(args, 'exclude_places', None),
                            source_night_config=source_night_config, seed=args.seed)


class SourceStream:
    def __init__(self, args, dataset, targets, places, bank_info):
        self.args, self.dataset, self.bank_info = args, dataset, bank_info
        self.places = tuple(places)
        if self.places != tuple(row['place'] for row in dataset.rows):
            raise ValueError('Source Day bank place order does not match the verified dataset')
        if len(targets) != len(dataset):
            raise ValueError('Source Day bank descriptor rows do not match the verified dataset')
        storage = getattr(args, 'source_day_bank_device', 'cpu')
        if storage == 'auto':
            storage = args.device
        if torch.device(storage).type == 'cuda' and torch.device(args.device).type != 'cuda':
            raise ValueError('CUDA Source Day bank requires a CUDA training device')
        if storage == 'cuda':
            storage = args.device
        # Descriptor targets follow the bank device. This avoids retaining a
        # large additional CPU tensor during GPU-resident coverage experiments.
        self.targets = targets.to(storage)
        self.queue = DayDescriptorQueue(len(dataset), storage_device=storage,
                                        include_current_keys=False)
        self.queue.enqueue(self.targets, self.places)
        groups = [json.dumps(list(place), ensure_ascii=False) for place in self.places]
        self.sampler = GroupBalancedCycle(groups, source_seed(args.seed, 'sampler'))
        self.batches = 0
        self.dataset_signature = dataset.sources_signature()

    def rng_context(self, epoch, step):
        return auxiliary_rng_context(self.args.device, source_seed(self.args.seed, 'branch', epoch, step))

    def next_batch(self, epoch, step):
        count = self.args.source_night_samples
        # Sampling and input construction cannot advance the Qwen/auxiliary
        # RNG even if a caller constructs a batch outside the branch context.
        with auxiliary_rng_context(self.args.device, source_seed(self.args.seed, 'inputs', epoch, step)):
            indices = self.sampler.draw(count)
            images, records = [], []
            for position, index in enumerate(indices):
                image, record = self.dataset.source_view(
                    index, seed=source_seed(self.args.seed, 'view', epoch, step, position, index))
                images.append(image)
                records.append(record)
        self.batches += 1
        return {'images': torch.stack(images), 'targets': self.targets[indices],
                'places': [self.places[index] for index in indices],
                'sample_ids': [self.dataset.rows[index]['sample_id'] for index in indices],
                'indices': indices, 'augmentation': sampled_augmentation_counts(records),
                'augmentation_records': records}

    def metadata(self):
        self.dataset.assert_current()
        return {'enabled': True, 'manifest': str(self.dataset.manifest),
                'dataset_signature': self.dataset_signature,
                'training_images': len(self.dataset), 'training_places': len(set(self.places)),
                'cities': sorted({place[0] for place in self.places}),
                'samples_per_microbatch': self.args.source_night_samples,
                'source_night_config': self.dataset.source_night_config.to_dict(),
                'source_night_weight': self.args.source_night_weight,
                'sampling': self.sampler.metadata(), 'batches': self.batches,
                'exposed_unique_places': len(self.sampler.group_counts),
                'day_bank': self.bank_info, 'bank_storage_device': str(self.queue.storage_device),
                'bank_policy': 'separate Source-only fixed bank; Qwen bank unchanged across scales',
                'label_policy': 'original CSV-verified (city,place_id); held-out union excluded',
                'rng_policy': 'private Source sampler and SHA256 Python/NumPy/Torch branch context',
                'loss': 'source_night_weight * (global_weight * cosine + contrast_weight * InfoNCE); no Relation'}


def build_source_stream(args, net, checkpoint_hash, preprocessing, *, dataset=None,
                        source_night_config=None):
    """Build a separate frozen Day bank without touching the main data stream."""
    if not source_stream_enabled(args):
        return None
    if type(args.source_night_samples) is not int or args.source_night_samples < 1:
        raise ValueError('Independent Source samples must be positive')
    with auxiliary_rng_context(args.device, source_seed(args.seed, 'build')):
        if dataset is None:
            dataset = prepare_source_dataset(args, preprocessing['image_size'], source_night_config)
        before = dataset.sources_signature()
        targets, places, info = load_or_build_day_bank(
            net, dataset, checkpoint_hash, preprocessing,
            getattr(args, 'source_day_cache', None), args.device,
            batch_size=getattr(args, 'source_day_cache_batch_size', 64),
            num_workers=getattr(args, 'source_day_cache_workers', 0))
        if dataset.sources_signature() != before:
            raise RuntimeError('Source Day pool changed during frozen target extraction')
        info = {**info, 'source_dataset_signature': before,
                'purpose': 'independent_source_only_day_bank'}
        return SourceStream(args, dataset, targets, places, info)

"""Independent stage-one paired supervision, never a place-label bank.

The main GSV stream controls the iteration budget. An auxiliary stream cycles
over reviewed scene groups (real) or existing training places (synthetic), and
its data loading, model forward and backward use a private random context.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors import SafetensorError

from weather_vpr.data.paired_dataset import sha256
from weather_vpr.data.student_augmentation import (
    student_training_tensor, sampled_augmentation_counts,
)
from weather_vpr.training.day_bank import (
    _DayImages, _descriptor_dim, _image_size, _load_verified_tensors,
    _publish_cache, _signature, _validate_descriptors, load_or_build_day_bank,
)
from weather_vpr.utils.backend import enable_backend_imports


def auxiliary_seed(seed: int, *parts) -> int:
    payload = json.dumps([int(seed), 'paired_auxiliary_v1', *parts],
                         ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


@contextmanager
def auxiliary_rng_context(device, seed: int):
    """Restore Python/NumPy/CPU and selected CUDA RNG, including recompute.

    Use this around cache construction AND the complete auxiliary branch,
    not just its forward. The disabled branch never enters this context.
    """
    device = torch.device(device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class GroupBalancedCycle:
    """Round-robin shuffled groups, independently shuffled cycles per group.

    Groups are sampling strata, not supervision labels. Small groups repeat
    more often by design; exposure counts remain explicit and auditable.
    """

    def __init__(self, groups, seed):
        self.groups = tuple(groups)
        if not self.groups or any(not isinstance(g, str) or not g for g in self.groups):
            raise ValueError('Auxiliary groups must be nonempty strings')
        self.rng = random.Random(seed)
        self.members = {}
        for index, group in enumerate(self.groups):
            self.members.setdefault(group, []).append(index)
        self.group_order, self.pools = [], {}
        self.counts = Counter()
        self.group_counts = Counter()
        self.draws = 0

    def draw(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('Auxiliary sample count must be positive')
        indices = []
        for _ in range(count):
            if not self.group_order:
                self.group_order = list(self.members)
                self.rng.shuffle(self.group_order)
            group = self.group_order.pop()
            pool = self.pools.setdefault(group, [])
            if not pool:
                pool.extend(self.members[group])
                self.rng.shuffle(pool)
            index = pool.pop()
            indices.append(index)
            self.counts[index] += 1
            self.group_counts[group] += 1
            self.draws += 1
        return indices

    def metadata(self):
        return {'samples_drawn': self.draws, 'unique_samples': len(self.counts),
                'repeated_exposures': self.draws - len(self.counts),
                'scene_groups': len(self.members), 'group_exposures': dict(self.group_counts),
                'sampling': 'shuffled group round-robin; shuffled independent cycles within group'}


def paired_global_alignment(student, teacher, weights):
    """Weighted FP32 cosine alignment; targets never acquire gradients."""
    if (student.ndim != 2 or teacher.shape != student.shape
            or student.device != teacher.device):
        raise ValueError('Paired descriptors must share [batch,dim] shape/device')
    if weights.shape != (len(student),) or weights.device != student.device:
        raise ValueError('Pair weights must have shape [batch] on the descriptor device')
    with torch.autocast(device_type=student.device.type, enabled=False):
        student, teacher = student.float(), teacher.detach().float()
        weights = weights.detach().float()
        if (not torch.isfinite(student).all() or not torch.isfinite(teacher).all()
                or not torch.isfinite(weights).all() or ((weights != 0) & (weights != 1)).any()):
            raise ValueError('Pair descriptors must be finite; weights must be binary')
        valid = weights > 0
        count = int(valid.sum())
        if not count:
            return {'loss': student.sum() * 0, 'valid_samples': 0}
        if ((student[valid].norm(dim=1) <= 1e-12).any()
                or (teacher[valid].norm(dim=1) <= 1e-12).any()):
            raise ValueError('Valid pair descriptors cannot have zero norm')
        # Same stable form as descriptor_alignment, equal to 1-cosine for
        # normalized nonzero vectors, without negative roundoff near equality.
        delta = F.normalize(student[valid], dim=1) - F.normalize(teacher[valid], dim=1)
        loss = (0.5 * delta.square().sum(dim=1)).mean()
        return {'loss': loss, 'valid_samples': count}


def load_or_build_pair_targets(net, dataset, checkpoint_hash, preprocessing,
                               cache_path, device, batch_size=2, num_workers=0):
    """Independent signed Day targets; no geographic labels are fabricated."""
    size = _image_size(preprocessing, dataset)
    sources = dataset.day_sources
    identity = {'format_version': 1, 'purpose': 'reviewed_real_pair_day_targets',
                'checkpoint_sha256': checkpoint_hash, 'preprocessing': preprocessing,
                'dataset_signature': dataset.sources_signature(), 'source_images': sources}
    signature = _signature(identity)
    expected_dim = _descriptor_dim(net)
    path = Path(cache_path).expanduser().resolve() if cache_path is not None else None
    status, reason = 'memory', None
    if path is not None and (path.exists() or path.with_suffix(path.suffix + '.json').exists()):
        try:
            saved = json.loads(path.with_suffix(path.suffix + '.json').read_text())
            if not isinstance(saved, dict):
                raise ValueError('Paired cache metadata must be a dictionary')
            if (saved.get('cache_signature') != signature
                    or any(saved.get(k) != v for k, v in identity.items())):
                raise ValueError('paired target identity changed')
            tensors = _load_verified_tensors(path, saved['tensor_sha256'])
            if set(tensors) != {'descriptors'}:
                raise ValueError('Paired cache must only contain descriptors')
            descriptors = tensors['descriptors']
            shape = saved.get('descriptor_shape')
            if (not isinstance(shape, list) or len(shape) != 2
                    or any(type(v) is not int or v < 1 for v in shape)):
                raise ValueError('Invalid paired cache descriptor shape')
            _validate_descriptors(descriptors, len(dataset), shape, expected_dim)
            return descriptors, dict(saved, cache_status='hit', rebuild_reason=None)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError) as error:
            reason = str(error)
            warnings.warn(f'Rebuilding paired Day targets: {reason}', RuntimeWarning, stacklevel=2)
            status = 'rebuilt'
    elif path is not None:
        status = 'built'
    # A dedicated generator avoids even DataLoader iterator setup changing
    # the main model RNG; callers additionally sandbox the entire build path.
    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=torch.device(device).type == 'cuda',
                         generator=torch.Generator().manual_seed(0))
    if num_workers:
        loader_kwargs['prefetch_factor'] = 1
    loader = DataLoader(_DayImages(sources, size), **loader_kwargs)
    chunks = []
    with torch.no_grad():
        for images in loader:
            result = net.teacher(images.to(device, non_blocking=True), return_aux=False)
            if not isinstance(result, torch.Tensor) or result.ndim != 2 or len(result) != len(images):
                raise ValueError('Teacher must return paired descriptors [batch,dim]')
            result = result.detach().float().cpu()
            if not torch.isfinite(result).all() or (result.norm(dim=1) <= 1e-12).any():
                raise ValueError('Invalid Teacher paired descriptors')
            chunks.append(F.normalize(result, dim=1))
            if len(chunks) % 25 == 0:
                print(f'Real Day target extraction: {sum(len(c) for c in chunks)}/{len(dataset)}', flush=True)
    descriptors = torch.cat(chunks).contiguous()
    _validate_descriptors(descriptors, len(dataset), expected_dim=expected_dim)
    # Do not sign a cache if the audit manifest changed during extraction.
    if dataset.sources_signature() != identity['dataset_signature']:
        raise RuntimeError('Real pair review/source manifest changed during target extraction')
    metadata = dict(identity, cache_signature=signature,
                    descriptor_shape=list(descriptors.shape),
                    cache_status=status, rebuild_reason=reason)
    if path is not None:
        metadata = _publish_cache(path, metadata, descriptors)
    return descriptors, metadata


class AuxiliaryBranch:
    def __init__(self, args, dataset, targets, target_info, groups, source_info):
        self.args, self.dataset = args, dataset
        self.targets, self.target_info = targets, target_info
        self.groups = tuple(groups)
        self.source_info = source_info
        self.sampler = GroupBalancedCycle(groups, auxiliary_seed(args.seed, 'sampler', args.auxiliary_mode))
        self.batches = 0

    def rng_context(self, epoch, step):
        return auxiliary_rng_context(self.args.device,
                                     auxiliary_seed(self.args.seed, 'branch', epoch, step))

    def next_batch(self, epoch, step):
        indices = self.sampler.draw(self.args.auxiliary_samples)
        tensors, ids, records = [], [], []
        if self.args.auxiliary_mode == 'real':
            for index in indices:
                # Teacher Day targets are already signed/cached. Do not decode
                # another full-resolution Day JPEG for every Night update.
                tensors.append(self.dataset.read_night_tensor(index))
                ids.append(self.dataset.rows[index]['sample_id'])
                records.append({'applied': False, 'operations': {}})
        else:
            # Indexing AcceptedWeatherPairs would also construct Source Night
            # and read unused views. Read just the signed Qwen image instead.
            for position, index in enumerate(indices):
                row = self.dataset.rows[index]
                path = Path(row['target_file'])
                payload = path.read_bytes()
                expected = row.get('output_sha256')
                if expected and hashlib.sha256(payload).hexdigest() != expected:
                    raise RuntimeError(f'Auxiliary Qwen image changed: {path}')
                with Image.open(io.BytesIO(payload)) as image:
                    tensor, record = student_training_tensor(
                        image, self.dataset.size, self.dataset.augmentation,
                        seed=auxiliary_seed(self.args.seed, 'synthetic_view', epoch, step, position, index))
                tensors.append(tensor)
                ids.append(str(row.get('sample_id', index)))
                records.append(record)
        self.batches += 1
        return {'images': torch.stack(tensors), 'targets': self.targets[indices],
                'weights': torch.ones(len(indices), dtype=torch.float32),
                'sample_ids': ids, 'indices': indices,
                'groups': [self.groups[i] for i in indices],
                'augmentation': sampled_augmentation_counts(records)}

    def metadata(self):
        return {'enabled': True, 'mode': self.args.auxiliary_mode,
                'weight': self.args.auxiliary_weight, 'samples_per_microbatch': self.args.auxiliary_samples,
                'source': self.source_info, 'day_targets': self.target_info,
                'sampling': self.sampler.metadata(), 'batches': self.batches,
                'rng_policy': 'sha256 private Python/NumPy/Torch context; forward and backward isolated',
                'loss': 'auxiliary_weight * binary-weighted mean paired global cosine; no InfoNCE/Relation',
                'negative_bank': 'unchanged GSV bank; real paired targets never enqueued'}


def prepare_auxiliary_dataset(args, image_size):
    """Fail a bad/unreviewed real manifest before allocating the model."""
    if args.auxiliary_mode != 'real' or args.auxiliary_weight == 0:
        return None
    from weather_vpr.data.real_pairs import ReviewedRealPairs
    with auxiliary_rng_context(args.device, auxiliary_seed(args.seed, 'review')):
        return ReviewedRealPairs(args.real_pairs_root, args.real_pairs_review,
                                 image_size=tuple(image_size))


def build_auxiliary_branch(args, dataset, net, checkpoint_hash, preprocessing,
                           *, day_descriptors=None, day_bank_info=None, reviewed_pairs=None):
    """Zero weight/none is a true bypass, without reading auxiliary files."""
    if args.auxiliary_mode == 'none' or args.auxiliary_weight == 0:
        return None
    enable_backend_imports()
    with auxiliary_rng_context(args.device, auxiliary_seed(args.seed, 'build')):
        if args.auxiliary_mode == 'real':
            pairs = (reviewed_pairs if reviewed_pairs is not None else
                     prepare_auxiliary_dataset(args, preprocessing['image_size']))
            targets, info = load_or_build_pair_targets(
                net, pairs, checkpoint_hash, preprocessing, args.auxiliary_day_cache,
                args.device, batch_size=(getattr(args, 'auxiliary_cache_batch_size', 0)
                                         or args.auxiliary_samples),
                num_workers=getattr(args, 'auxiliary_cache_workers', 0))
            groups = [row['scene_group'] for row in pairs.rows]
            source_info = {'review_manifest': str(pairs.manifest),
                           'signature': pairs.sources_signature(), 'training_pairs': len(pairs)}
            return AuxiliaryBranch(args, pairs, targets, info, groups, source_info)
        if day_descriptors is None:
            day_descriptors, _, day_bank_info = load_or_build_day_bank(
                net, dataset, checkpoint_hash, preprocessing, args.auxiliary_day_cache,
                args.device, batch_size=args.auxiliary_samples, num_workers=0)
        elif args.auxiliary_day_cache is not None:
            raise ValueError('Synthetic auxiliary reuses --fixed-day-bank; do not set --auxiliary-day-cache')
        groups = [json.dumps(list(row['place']), ensure_ascii=False) for row in dataset.rows]
        return AuxiliaryBranch(args, dataset, day_descriptors, day_bank_info, groups,
                               {'manifest': str(dataset.manifest),
                                'manifest_sha256': sha256(dataset.manifest),
                                'training_pairs': len(dataset)})


class AuxiliaryGradientProbe:
    """Observe per-microbatch gradient increments without modifying .grad."""

    def __init__(self, parameters, accumulation_steps=1):
        self.parameters = tuple(parameters)
        self.scale = accumulation_steps
        self.baseline = self._snapshot()
        self.main = None

    def _snapshot(self):
        return [p.grad.detach().float().clone() if p.grad is not None else None
                for p in self.parameters]

    def after_main(self):
        now = self._snapshot()
        self.main = [(value if old is None else -old if value is None else value-old)
                     for old, value in zip(self.baseline, now)]
        self.baseline = now

    def after_auxiliary(self):
        if self.main is None:
            raise RuntimeError('Capture GSV+Source gradients before auxiliary gradients')
        square_main = square_aux = dot = 0.0
        for p, old, main in zip(self.parameters, self.baseline, self.main):
            value = p.grad.detach().float() if p.grad is not None else None
            aux = value if old is None else -old if value is None else value-old
            if main is not None:
                square_main += float(main.double().square().sum())
            if aux is not None:
                square_aux += float(aux.double().square().sum())
            if main is not None and aux is not None:
                dot += float((main.double()*aux.double()).sum())
        norm_main, norm_aux = square_main ** .5, square_aux ** .5
        return {'main_norm': norm_main*self.scale, 'auxiliary_norm': norm_aux*self.scale,
                'norm_ratio': norm_aux/norm_main if norm_main > 1e-12 else None,
                'cosine': max(-1., min(1., dot/(norm_main*norm_aux)))
                if norm_main > 1e-12 and norm_aux > 1e-12 else None,
                'scope': 'first microbatch of selected update; unclipped weighted increments; K rescaled'}

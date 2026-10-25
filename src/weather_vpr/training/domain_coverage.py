"""Train-only frozen-prefix statistics and matched Token interventions.

No source classifier, embedding plot, or scalar distance is a test of pure
illumination. All outputs of this module remain descriptive pending content
review. Prefix outputs use absolute, zero-based block numbers, before norm.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from safetensors.torch import save_file

from weather_vpr.data.source_night import SourceNightAugmentation, SourceNightConfig
from weather_vpr.data.student_augmentation import sample_augmentation_seed
from weather_vpr.training.day_bank import _load_verified_tensors


SOURCES = frozenset(('qwen_night', 'source_night', 'darkdriving_train_night',
                     'bdd_train_night', 'nightcity_train'))
PREPROCESSING = {'version': 'rgb_bilinear_imagenet_v1', 'image_size': [224, 224],
                 'color_space': 'RGB', 'resize': 'PIL_bilinear', 'input_range': [0., 1.],
                 'mean': [.485, .456, .406], 'std': [.229, .224, .225]}
STATISTICS = {'tokens': 'patch-only; CLS excluded', 'precision': 'FP32',
              'mean_axis': 'spatial patches', 'sigma': 'sqrt(var(correction=0)+1e-6)',
              'features': 'concatenate(mu, log(sigma)); pre-final-norm', 'eps': 1e-6}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def private_rng(seed, domain):
    return random.Random(int(signature([seed, domain]), 16))


def validate_train_rows(rows):
    """Defense in depth in addition to the source inventory's admission rules."""
    seen = set()
    for row in rows:
        if row.get('source') not in SOURCES or row.get('split') != 'train':
            raise ValueError('Only registered train-only source rows are permitted')
        if not row.get('sample_id') or not row.get('group_id'):
            raise ValueError('Every diagnostic row needs sample_id and group_id')
        key = row['source'], row['sample_id']
        if key in seen:
            raise ValueError(f'Duplicate source sample: {key}')
        seen.add(key)
        path = Path(row['path']).resolve()
        components = {part.casefold() for part in path.parts}
        forbidden = {'test', 'testing', 'val', 'validation', 'eval', 'evaluation'}
        evaluation_datasets = {'svox', 'tokyo', 'tokyo247', 'robotcar',
                               'robotcar-seasons', 'aachen', 'aachen-day-night'}
        if components & (forbidden | evaluation_datasets):
            raise ValueError(f'Evaluation/validation images cannot become donors: {path}')
        if row['source'] == 'source_night' and row.get('metadata', {}).get('transform') != 'source_night':
            raise ValueError('Source Night must explicitly declare its source_night transform')
    if not rows:
        raise ValueError('No admitted training sources')


def balanced_sample(rows, maximum=512, seed=42):
    """Group-balanced sampling, with exactly paired Qwen/Source scenes when present.

    A joint pool must contain the same canonical sample IDs and place groups for
    both synthetic views. Sharing the private RNG then selects identical scenes
    without weakening group-disjoint reference calibration.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError('maximum must be positive')
    validate_train_rows(rows)
    domains = defaultdict(list)
    for row in rows:
        domains[row['source']].append(row)
    paired_synthetic = {'qwen_night', 'source_night'} <= domains.keys()
    if paired_synthetic:
        pair_groups = {
            source: {row['sample_id']: str(row['group_id']) for row in domains[source]}
            for source in ('qwen_night', 'source_night')
        }
        if pair_groups['qwen_night'] != pair_groups['source_night']:
            raise ValueError('Qwen/Source diagnostic pools must have identical canonical sample IDs and place groups')
    selected, counts = [], {}
    for source, source_rows in sorted(domains.items()):
        rng_domain = ('paired-qwen-source' if paired_synthetic
                      and source in ('qwen_night', 'source_night') else source)
        rng = private_rng(seed, f'group-balanced:{rng_domain}')
        groups = defaultdict(list)
        for row in sorted(source_rows, key=lambda item: item['sample_id']):
            groups[str(row['group_id'])].append(row)
        group_order = sorted(groups)
        rng.shuffle(group_order)
        for group in groups.values():
            rng.shuffle(group)
        result = []
        while len(result) < min(maximum, len(source_rows)):
            for name in group_order:
                if groups[name] and len(result) < maximum:
                    result.append(groups[name].pop())
        selected.extend(result)
        counts[source] = {'full_rows': len(source_rows), 'full_groups': len(groups),
                          'sampled_rows': len(result),
                          'sampled_groups': len({str(row['group_id']) for row in result})}
        if paired_synthetic and source in ('qwen_night', 'source_night'):
            counts[source]['paired_synthetic_scenes'] = True
    return selected, counts


def signed_image_rows(rows, seed):
    """Hash actual selected bytes; record original training index for Source Night."""
    result = []
    for row in rows:
        digest = sha256(row['path'])
        if row.get('sha256') and row['sha256'] != digest:
            raise ValueError(f'Manifest image hash changed: {row["path"]}')
        item = {**row, 'path': str(Path(row['path']).resolve()), 'sha256': digest}
        if row['source'] == 'source_night':
            metadata = row.get('metadata', {})
            original_index = metadata.get('original_sample_index', metadata.get('training_index'))
            if type(original_index) is not int or original_index < 0:
                raise ValueError('Source Night needs metadata.original_sample_index from canonical Qwen manifest')
            item['augmentation_seed'] = sample_augmentation_seed(seed, 0, original_index, view='source_night')
        result.append(item)
    return result


def verify_row_provenance(rows, project_root):
    """Recheck annotation/review bytes, including on a statistics-cache hit."""
    root = Path(project_root).resolve()
    files = {}
    for row in rows:
        metadata = row.get('metadata', {})
        for key in ('annotation', 'review_manifest'):
            if key in metadata:
                expected = metadata.get(key + '_sha256')
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError(f'Missing {key} SHA-256 in source metadata')
                path = Path(metadata[key])
                path = path.resolve() if path.is_absolute() else (root/path).resolve()
                if path in files and files[path] != expected:
                    raise ValueError(f'Contradictory provenance hashes for {path}')
                files[path] = expected
        for name, expected in metadata.get('dataset_metadata', {}).items():
            path = Path(name)
            path = path.resolve() if path.is_absolute() else (root/path).resolve()
            if path in files and files[path] != expected:
                raise ValueError(f'Contradictory dataset metadata hashes for {path}')
            files[path] = expected
    for path, expected in files.items():
        if sha256(path) != expected:
            raise ValueError(f'Training source annotation/review changed: {path}')
    return {str(path): digest for path, digest in sorted(files.items())}


class DiagnosticImages(Dataset):
    """Hash then strict decode the same bytes; no Patch tensors cached on disk."""

    def __init__(self, rows, source_config):
        self.rows = rows
        self.source = SourceNightAugmentation(SourceNightConfig.from_dict(source_config))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = Path(row['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError(f'Selected image changed during extraction: {row["path"]}')
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            resized = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        if row['source'] == 'source_night':
            resized, _ = self.source(resized, seed=row['augmentation_seed'])
        pixels = np.array(resized, dtype=np.float32) / 255.
        brightness = float(np.mean(pixels @ np.array([.2126, .7152, .0722], dtype=np.float32)))
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        tensor = (tensor - tensor.new_tensor(PREPROCESSING['mean'])[:, None, None]) / tensor.new_tensor(
            PREPROCESSING['std'])[:, None, None]
        return tensor, brightness


def patch_statistics(tokens, eps=1e-6):
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError('Expected [batch,CLS+patches,channels]')
    patches = tokens[:, 1:].detach().float()
    mean = patches.mean(1)
    sigma = (patches.var(1, correction=0) + eps).sqrt()
    result = torch.cat((mean, sigma.log()), 1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite prefix statistics')
    return result


@torch.no_grad()
def extract_prefix_statistics(base, images, layers=(3, 5, 7)):
    layers = tuple(layers)
    if (not layers or len(set(layers)) != len(layers)
            or any(type(layer) is not int or not 0 <= layer <= 7 for layer in layers)):
        raise ValueError('Statistics layers must be unique absolute block indices in [0,7]')
    model = base.backbone.model
    tokens = model.prepare_tokens_with_masks(images)
    result = {}
    for index, block in enumerate(model.blocks):
        tokens = block(tokens)
        if index in layers:
            result[f'block_{index}'] = patch_statistics(tokens)
        if index == max(layers):
            break
    if len(result) != len(layers):
        raise ValueError('Backbone lacks a requested prefix block')
    return result


def model_state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_forward_provenance(model, project_root):
    """Bind local loader/backbone/attention/aggregator implementation bytes.

    Checkpoint bytes alone do not detect an edited forward implementation.
    Framework versions supplement local code hashes without reading framework
    source trees or including unrelated project files in cache identity.
    """
    root = Path(project_root).resolve()
    files = set()
    for module in model.modules():
        filename = inspect.getsourcefile(type(module))
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(root):
            files.add(path)
    return {'local_module_files_sha256': {
        str(path.relative_to(root)): sha256(path) for path in sorted(files)},
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
        'cuda_version': torch.version.cuda,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}


def write_stats_cache(path, tensors, identity):
    path = Path(path)
    metadata = path.with_suffix(path.suffix + '.json')
    if path.exists() or metadata.exists():
        raise FileExistsError(f'Never overwrite a statistics cache: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        save_file({key: tensor.cpu().contiguous() for key, tensor in tensors.items()}, str(temporary))
        document = {'identity': identity, 'signature': signature(identity), 'tensor_sha256': sha256(temporary),
                    'shapes': {key: list(value.shape) for key, value in tensors.items()}}
        # Exclusive publication keeps interrupted/incomplete caches fail-closed.
        with metadata.open('x', encoding='utf-8') as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_stats_cache(path, identity):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + '.json').read_text())
    if metadata['identity'] != identity or metadata['signature'] != signature(identity):
        raise ValueError('Statistics cache identity differs: weights/images/split/config/layers/precision changed')
    tensors = _load_verified_tensors(path, metadata['tensor_sha256'])
    expected = {f'block_{layer}' for layer in identity['layers']} | {'brightness'}
    if set(tensors) != expected or set(metadata['shapes']) != expected:
        raise ValueError('Statistics cache keys differ')
    count = len(identity['selected_rows'])
    for key, tensor in tensors.items():
        if (tensor.dtype != torch.float32 or tensor.shape[0] != count
                or list(tensor.shape) != metadata['shapes'][key] or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Statistics cache tensor shape/dtype/finite check failed')
        if ((key == 'brightness' and tensor.ndim != 1)
                or (key != 'brightness' and (tensor.ndim != 2 or tensor.shape[1] % 2))):
            raise ValueError('Statistics cache has invalid channel layout')
    return tensors


def group_weights(groups):
    counts = defaultdict(int)
    for group in groups:
        counts[str(group)] += 1
    return torch.tensor([1. / len(counts) / counts[str(group)] for group in groups], dtype=torch.float64)


def weighted_summary(values, groups):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if len(values) != len(groups) or not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError('Finite nonempty values and matching groups required')
    weights = group_weights(groups)
    order = values.argsort()
    cumulative = weights[order].cumsum(0)
    quantiles = {}
    for name, quantile in (('p50', .5), ('p95', .95), ('p99', .99)):
        position = int(torch.searchsorted(cumulative, torch.tensor(quantile, dtype=torch.float64)))
        quantiles[name] = float(values[order[min(position, len(values)-1)]])
    return {'count': len(values), 'groups': len(set(map(str, groups))),
            'group_weighted_mean': float((values * weights).sum()),
            'image_mean': float(values.mean()), 'min': float(values.min()),
            **quantiles, 'max': float(values.max())}


def reference_group_split(rows, seed=42, fit_fraction=.8):
    groups = sorted({str(row['group_id']) for row in rows
                     if row['source'] in ('qwen_night', 'source_night')})
    if len(groups) < 2:
        raise ValueError('Need at least two synthetic reference groups for disjoint calibration')
    private_rng(seed, 'reference-group-fit-heldout').shuffle(groups)
    cut = min(len(groups)-1, max(1, math.floor(len(groups) * fit_fraction)))
    fit, heldout = set(groups[:cut]), set(groups[cut:])
    fit_indices = [index for index, row in enumerate(rows)
                   if row['source'] in ('qwen_night', 'source_night') and str(row['group_id']) in fit]
    heldout_indices = [index for index, row in enumerate(rows)
                       if row['source'] in ('qwen_night', 'source_night') and str(row['group_id']) in heldout]
    return fit_indices, heldout_indices


def fit_projection(values, groups, dimensions=32):
    """Weighted standardization and PCA fitted exclusively to reference-fit rows."""
    values = values.double()
    weights = group_weights(groups)
    mean = (values * weights[:, None]).sum(0)
    scale = ((values-mean).square() * weights[:, None]).sum(0).sqrt().clamp_min(1e-6)
    standardized = (values-mean) / scale
    _, singular, directions = torch.linalg.svd(standardized * weights.sqrt()[:, None], full_matrices=False)
    dimension = min(dimensions, len(directions), len(values)-1)
    return {'mean': mean, 'scale': scale, 'directions': directions[:dimension].T.contiguous(),
            'explained_variance_fraction': float(singular[:dimension].square().sum()
                                                 / singular.square().sum().clamp_min(1e-12))}


def project(values, fitted):
    return ((values.double()-fitted['mean']) / fitted['scale'] @ fitted['directions']).float()


def knn_distances(target, reference, neighbors=5):
    if len(reference) < 1 or len(target) < 1:
        raise ValueError('kNN needs nonempty target and reference')
    chunks = []
    for start in range(0, len(target), 128):
        distances = torch.cdist(target[start:start+128].float(), reference.float())
        chunks.append(distances.topk(min(neighbors, len(reference)), largest=False).values.mean(1))
    return torch.cat(chunks)


def coverage_report(tensors, rows, layers=(3, 5, 7), seed=42, pca_dimensions=32, neighbors=5):
    fit_indices, heldout_indices = reference_group_split(rows, seed)
    group = lambda indices: [str(rows[index]['group_id']) for index in indices]
    fit_groups, heldout_groups = group(fit_indices), group(heldout_indices)
    if set(fit_groups) & set(heldout_groups):
        raise AssertionError('Reference fit/calibration group overlap')
    result = {'reference_sources': ['qwen_night', 'source_night'],
              'reference_fit': {'rows': len(fit_indices), 'groups': sorted(set(fit_groups))},
              'reference_heldout': {'rows': len(heldout_indices), 'groups': sorted(set(heldout_groups))},
              'threshold_rule': 'group-weighted empirical 95th percentile of heldout-synthetic kNN distance',
              'metric': f'mean of {neighbors} nearest reference-fit Euclidean distances in reference-fit standardized PCA',
              'decision': 'requires_content_review', 'layers': {},
              'limitations': ['Statistics contain scene, camera and semantic content as well as lighting.',
                             'Geographic independence is not established by technical/group-disjoint splitting.',
                             'Brightness strata and group weighting do not control all content confounds.',
                             'This report cannot authorize a training/Recall search or prove pure style.']}
    sources = sorted({row['source'] for row in rows})
    for layer in layers:
        values = tensors[f'block_{layer}']
        fitted = fit_projection(values[fit_indices], fit_groups, pca_dimensions)
        reference = project(values[fit_indices], fitted)
        heldout_distances = knn_distances(project(values[heldout_indices], fitted), reference, neighbors)
        calibration = weighted_summary(heldout_distances, heldout_groups)
        threshold = calibration['p95']
        domains = {}
        for source in sources:
            indices = [index for index, row in enumerate(rows) if row['source'] == source
                       and (source not in ('qwen_night', 'source_night') or index in heldout_indices)]
            if not indices:
                domains[source] = {'status': 'no_heldout_samples'}
                continue
            distances = knn_distances(project(values[indices], fitted), reference, neighbors)
            groups = group(indices)
            weights = group_weights(groups)
            brightness = tensors['brightness'][indices]
            strata = {}
            for lower, upper in ((0., .1), (.1, .25), (.25, .5), (.5, 1.000001)):
                selected = ((brightness >= lower) & (brightness < upper)).nonzero().flatten().tolist()
                if selected:
                    strata[f'[{lower},{min(upper,1.)})'] = weighted_summary(distances[selected], [groups[i] for i in selected])
            domains[source] = {'distance': weighted_summary(distances, groups),
                               'within_reference_heldout_p95_group_weighted_fraction': float(
                                   ((distances <= threshold).double() * weights).sum()),
                               'brightness': weighted_summary(brightness, groups), 'brightness_strata': strata,
                               'geographic_independence_known': all(
                                   rows[index].get('metadata', {}).get('geographic_independence_known') is True
                                   for index in indices),
                               'largest_distance_samples_for_content_review': [
                                   {'sample_id': rows[indices[position]]['sample_id'],
                                    'group_id': rows[indices[position]]['group_id'],
                                    'path': rows[indices[position]]['path'],
                                    'distance': float(distances[position]),
                                    'brightness': float(brightness[position])}
                                   for position in distances.argsort(descending=True)[:12].tolist()]}
        result['layers'][str(layer)] = {'pca_dimensions': fitted['directions'].shape[1],
                                       'pca_explained_variance_fraction': fitted['explained_variance_fraction'],
                                       'heldout_calibration_distance': calibration,
                                       'overlap_threshold': threshold, 'sources': domains}
    return result


def intervention_plan(batch, patches, donors, *, seed, step, probability=.5,
                      fraction=.3, strength=.2, beta=.1, forbidden=None):
    if donors < 2 or batch < 1 or patches < 1:
        raise ValueError('Need at least two donors, one recipient and one patch')
    rng = private_rng(seed, ['matched-intervention', step])
    chosen = [index for index in range(batch) if rng.random() < probability]
    donor_indices = []
    for index in range(batch):
        candidates = [i for i in range(donors) if not forbidden or i != forbidden[index]]
        donor_indices.append(rng.choice(candidates))
    return {'selected_images': chosen, 'donor_indices': donor_indices,
            'patch_indices': [sorted(rng.sample(range(patches), math.floor(patches*fraction))) for _ in chosen],
            'lambdas': [strength * (1-rng.betavariate(beta, beta)) for _ in chosen],
            'probability': probability, 'fraction': fraction, 'strength': strength, 'beta': beta}


def apply_statistic_intervention(tokens, donor_statistics, plan, eps=1e-6):
    """Same indices/masks/lambdas may be reused across matched-size donor banks."""
    if not plan['selected_images'] or not any(plan['lambdas']):
        return tokens, {'applied_images': 0, 'relative_centered_rms_delta': [0.] * len(tokens),
                        'gain': [], 'cls_unchanged': True, 'unselected_patches_unchanged': True}
    if (donor_statistics.ndim != 2 or donor_statistics.shape[1] != tokens.shape[2]*2
            or not bool(torch.isfinite(donor_statistics).all())):
        raise ValueError('Donor statistics require finite [donors,2*channels]')
    patches = tokens[:, 1:].detach().float()
    mean = patches.mean(1, keepdim=True)
    sigma = (patches.var(1, correction=0, keepdim=True)+eps).sqrt()
    output = tokens.clone()
    mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    gains = []
    for image, patch_indices, coefficient in zip(plan['selected_images'], plan['patch_indices'], plan['lambdas']):
        donor = donor_statistics[plan['donor_indices'][image]].to(tokens.device).float()
        donor_mean, donor_logsigma = donor.chunk(2)
        mixed_mean = mean[image] + coefficient * (donor_mean[None]-mean[image])
        mixed_sigma = sigma[image] + coefficient * (donor_logsigma.exp()[None]-sigma[image])
        gains.extend((mixed_sigma/sigma[image]).flatten().cpu().tolist())
        ids = torch.tensor(patch_indices, dtype=torch.long, device=tokens.device)
        transformed = (patches[image, ids]-mean[image]) / sigma[image] * mixed_sigma + mixed_mean
        output[image, ids+1] = transformed.to(tokens.dtype)
        mask[image, ids+1] = True
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Nonfinite donor intervention')
    delta = output[:, 1:].float()-patches
    denominator = (patches-mean).square().mean((1, 2)).sqrt().clamp_min(eps)
    relative = delta.square().mean((1, 2)).sqrt()/denominator
    cls_equal = torch.equal(output[:, 0], tokens[:, 0])
    unselected_equal = torch.equal(output[~mask], tokens[~mask])
    if not cls_equal or not unselected_equal:
        raise AssertionError('Intervention changed CLS or an unselected patch')
    return output, {'applied_images': len(plan['selected_images']),
                    'relative_centered_rms_delta': relative.cpu().tolist(), 'gain': gains,
                    'cls_unchanged': cls_equal, 'unselected_patches_unchanged': unselected_equal}


@torch.no_grad()
def prefix_tokens(base, images, after_block=7):
    tokens = base.backbone.model.prepare_tokens_with_masks(images)
    for block in base.backbone.model.blocks[:after_block+1]:
        tokens = block(tokens)
    return tokens


@torch.no_grad()
def descriptor_from_prefix(base, tokens, after_block=7):
    for block in base.backbone.model.blocks[after_block+1:]:
        tokens = block(tokens)
    if base.backbone.norm_layer:
        tokens = base.backbone.model.norm(tokens)
    side = math.isqrt(tokens.shape[1]-1)
    if side*side != tokens.shape[1]-1:
        raise ValueError('SALAD diagnostic requires a square patch grid')
    patches = tokens[:, 1:].reshape(len(tokens), side, side, -1).permute(0, 3, 1, 2)
    return base.aggregator((patches, tokens[:, 0]), return_aux=True)


@torch.no_grad()
def intervention_report(base, dataset, tensors, rows, *, device='cuda', precision='fp32',
                        batch_size=16, recipients=64, seed=42, after_block=7):
    banks = {source: [i for i, row in enumerate(rows) if row['source'] == source] for source in SOURCES}
    banks = {source: indices for source, indices in sorted(banks.items()) if indices}
    count = min(map(len, banks.values()))
    if 'qwen_night' not in banks or count < 2:
        raise ValueError('Need Qwen and at least two matched donors per available source')
    banks = {source: indices[:count] for source, indices in banks.items()}
    recipient_ids = banks['qwen_night'][:recipients]
    qwen_positions = {index: position for position, index in enumerate(banks['qwen_night'])}
    values = {source: defaultdict(list) for source in banks}
    enabled_values = {source: defaultdict(list) for source in banks}
    enabled_recipient_ids, gain_groups = [], []
    records = []
    device = torch.device(device)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    for step, start in enumerate(range(0, len(recipient_ids), batch_size)):
        indices = recipient_ids[start:start+batch_size]
        images = torch.stack([dataset[index][0] for index in indices]).to(device)
        with amp():
            tokens = prefix_tokens(base, images, after_block)
            plain, plain_aux = descriptor_from_prefix(base, tokens, after_block)
            direct, direct_aux = base(images, return_aux=True)
            identity, _ = apply_statistic_intervention(tokens, tensors[f'block_{after_block}'][banks['qwen_night']],
                                                       {'selected_images': [], 'lambdas': []})
        if (identity is not tokens or not torch.equal(plain, direct)
                or any(not torch.equal(plain_aux[key], direct_aux[key]) for key in plain_aux)):
            raise AssertionError('Manual prefix/suffix or identity intervention differs from original Day model')
        plan = intervention_plan(len(images), tokens.shape[1]-1, count, seed=seed, step=step,
                                 forbidden=[qwen_positions[index] for index in indices])
        enabled_recipient_ids.extend(indices[index] for index in plan['selected_images'])
        gain_groups.extend(str(rows[indices[index]]['group_id'])
                           for index in plan['selected_images'] for _ in range(tokens.shape[-1]))
        records.append({'recipient_sample_ids': [rows[index]['sample_id'] for index in indices], **plan})
        for source, donor_ids in banks.items():
            with amp():
                changed, measured = apply_statistic_intervention(tokens, tensors[f'block_{after_block}'][donor_ids], plan)
                styled, styled_aux = descriptor_from_prefix(base, changed, after_block)
            drift = .5 * (F.normalize(plain.float(), dim=1)-F.normalize(styled.float(), dim=1)).square().sum(1)
            observations = {'descriptor_cosine_drift': drift.cpu().tolist(),
                            'relative_centered_rms_delta': measured['relative_centered_rms_delta'],
                            'gain': measured['gain']}
            for key in ('assignment_keep', 'log_assignment'):
                observations[key + '_mean_absolute_drift'] = (
                    (plain_aux[key]-styled_aux[key]).abs().flatten(1).mean(1).cpu().tolist())
            for metric, observed in observations.items():
                values[source][metric].extend(observed)
                enabled_values[source][metric].extend(
                    observed if metric == 'gain' else [observed[index] for index in plan['selected_images']])
        print(f'Matched intervention {min(start+batch_size,len(recipient_ids))}/{len(recipient_ids)} recipients', flush=True)
    summaries = {}
    groups = [str(rows[index]['group_id']) for index in recipient_ids]
    enabled_groups = [str(rows[index]['group_id']) for index in enabled_recipient_ids]
    for source, metrics in values.items():
        summaries[source] = {}
        for metric, observations in metrics.items():
            metric_groups = groups if metric != 'gain' else gain_groups
            summaries[source][metric] = weighted_summary(observations, metric_groups) if observations else {'count': 0}
        summaries[source]['enabled_only'] = {
            metric: (weighted_summary(observations, gain_groups if metric == 'gain' else enabled_groups)
                     if observations else {'count': 0})
            for metric, observations in enabled_values[source].items()}
        gains = torch.tensor(metrics['gain'])
        summaries[source]['gain_fractions'] = {f'gt_{threshold}': float((gains > threshold).float().mean())
                                               if len(gains) else None for threshold in (2, 4, 8)}
    return {'after_block': after_block, 'matched_donor_count_per_source': count,
            'recipient_count': len(recipient_ids), 'enabled_recipient_count': len(enabled_recipient_ids),
            'identity_equals_original_day': True,
            'same_recipients_donor_indices_masks_lambdas': True, 'sources': summaries, 'records': records,
            'decision': 'descriptive_only', 'gain_clipping': None,
            'summary_populations': {'per_image_metrics': 'all recipients plus enabled_only recipients',
                                    'gain': 'enabled recipient channels; grouped by recipient place'},
            'limitation': ('Equal lambda does not match actual perturbation. Preservation relative to the '
                           'original Qwen descriptor is not evidence of matched clean-Day place alignment. '
                           'No loss, optimizer, Recall or automatic gate.')}

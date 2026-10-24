#!/usr/bin/env python3
"""Read-only SVOX failure diagnosis with fixed-gallery attribution/interventions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ('day', 'source_night', 'late4', 'global_control', 'sgpd', 'sgpd_attention')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def utm_distance(first, second):
    def point(name):
        fields = Path(name).name.split('@')
        return float(fields[1]), float(fields[2])
    return math.dist(point(first), point(second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--top-patches', type=int, default=26)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('analysis.json', 'original_analysis.json', 'case_features.safetensors'):
        if (out / name).exists():
            raise FileExistsError(out / name)
    if args.batch_size < 1 or not 1 <= args.top_patches < 256:
        raise ValueError('Invalid batch size or top-patch count')
    import numpy as np
    from PIL import Image
    from scipy.stats import spearmanr
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from weather_vpr.utils.backend import enable_backend_imports
    from weather_vpr.inference.svox import load_night_dataset, gallery_fingerprint, _cache_metadata, _load_gallery_cache
    from weather_vpr.models.adapter import WeatherAdapterSALAD
    from weather_vpr.models.attention import capture_cls_attention
    from weather_vpr.inference.patch_attribution import signed_patch_margin
    enable_backend_imports()
    from workflow.model import load_checkpoint_model, input_preprocessing
    from workflow.evaluation import exact_retrieval

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    inventory = json.loads(args.inventory.read_text())
    source_results = {}
    for method, identity in inventory['source_evaluations'].items():
        if sha(identity['path']) != identity['sha256']:
            raise ValueError('Saved evaluation changed: ' + method)
        source_results[method] = json.loads(Path(identity['path']).read_text())
    dataset = load_night_dataset(ROOT / 'data/svox', split='test')
    query_lookup = {item.id: index for index, item in enumerate(dataset.queries)}
    gallery_lookup = {item.id: index for index, item in enumerate(dataset.references)}
    known = {row['query_id']: row for row in inventory['queries']}
    selected = inventory['recommended_inference_cases']
    query_indices = [query_lookup[row['query_id']] for row in selected]
    positives = [dataset.positives[index] for index in query_indices]
    checkpoint = ROOT / 'weights/dino_salad.ckpt'
    checkpoint_hash = sha(checkpoint)
    if checkpoint_hash != inventory['shared_checkpoint_sha256']:
        raise ValueError('Original Teacher changed')
    base = load_checkpoint_model(checkpoint, device=args.device, backbone_repo=ROOT / 'third_party/dinov2').eval()
    base.requires_grad_(False)
    preprocessing = input_preprocessing(base.image_size)
    print('Validating original Day Gallery cache...', flush=True)
    gallery_hash = gallery_fingerprint(dataset.references)
    expected = _cache_metadata(dataset.references, gallery_hash, checkpoint_hash, preprocessing)
    gallery_cpu = _load_gallery_cache(ROOT / 'runs/cache/svox_test_gallery.safetensors', expected, 8448)
    if gallery_cpu is None or gallery_hash != inventory['shared_gallery_sha256']:
        raise ValueError('Gallery cache/data do not match saved evaluations')
    gallery = gallery_cpu.to(args.device)
    gallery_norm = gallery.square().sum(1)
    gallery_ids = torch.arange(len(gallery), device=args.device)
    net = WeatherAdapterSALAD(base, ('night',), adapter_type='full').to(args.device).eval()
    net.requires_grad_(False)
    model_metadata = {'day': {'checkpoint_sha256': checkpoint_hash}}
    tensors = {}

    def image_tensor(path):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        return (torch.from_numpy(rgb).permute(2, 0, 1) - torch.tensor(preprocessing['mean'])[:, None, None]) / torch.tensor(preprocessing['std'])[:, None, None]

    inputs = torch.stack([image_tensor(dataset.queries[index].path) for index in query_indices])
    cases = []
    for item, image in zip(selected, inputs):
        rgb = image * torch.tensor(preprocessing['std'])[:, None, None] + torch.tensor(preprocessing['mean'])[:, None, None]
        luma = (rgb * torch.tensor([.2126, .7152, .0722])[:, None, None]).sum(0)
        cases.append({**item, 'methods': {}, 'input_statistics': {
            'mean_srgb_luma': float(luma.mean()), 'near_black_pixel_fraction': float((luma <= .01).float().mean()),
            'all_channel_clipped_fraction': float((rgb >= .99).all(0).float().mean())}})

    def select_model(method):
        if method == 'day':
            return base.backbone
        saved = source_results[method]
        directory = Path(saved['adapter']['directory'])
        meta = json.loads((directory / 'adapter.json').read_text())
        if meta != saved['adapter']['metadata'] or meta['checkpoint_sha256'] != checkpoint_hash or meta['preprocessing'] != preprocessing:
            raise ValueError('Night metadata mismatch: ' + method)
        state = load_file(str(directory / 'adapter.safetensors'), device='cpu')
        net.adapters['night'].load_state_dict(state, strict=True)
        del state
        gc.collect()
        model_metadata[method] = {'directory': str(directory), 'weights_sha256': sha(directory / 'adapter.safetensors'),
                                  'metadata_sha256': sha(directory / 'adapter.json')}
        return net.adapters['night']

    def forward(method, batch, auxiliary=False):
        if method == 'day':
            return base(batch, return_aux=auxiliary)
        return net.student(batch, 'night', return_aux=auxiliary)

    def score(vectors, positive_sets):
        vectors = vectors.to(args.device).float()
        outputs = []
        for start in range(0, len(vectors), args.batch_size):
            q = vectors[start:start + args.batch_size]
            distances = []
            for left in range(0, len(gallery), 4096):
                g = gallery[left:left + 4096]
                distances.append((q.square().sum(1)[:, None] + gallery_norm[left:left + len(g)][None]
                                  - 2 * q @ g.T).clamp_min(0))
            distance = torch.cat(distances, 1)
            for offset, positive_ids in enumerate(positive_sets[start:start + len(q)]):
                row = distance[offset]
                positive_mask = torch.zeros(len(gallery), dtype=torch.bool, device=args.device)
                positive_mask[positive_ids] = True
                pos_distance, pos_index = row.masked_fill(~positive_mask, torch.inf).min(0)
                neg_distance, neg_index = row.masked_fill(positive_mask, torch.inf).min(0)
                top_distance, top_index = row.min(0)
                rank = int(((row < pos_distance) | ((row == pos_distance) & (gallery_ids < pos_index))).sum()) + 1
                dp, dn = float(pos_distance.sqrt()), float(neg_distance.sqrt())
                outputs.append({'rank': rank, 'positive_index': int(pos_index), 'negative_index': int(neg_index),
                    'top1_index': int(top_index), 'positive_distance': dp, 'negative_distance': dn,
                    'margin': dn - dp, 'top1_distance': float(top_distance.sqrt())})
        return outputs

    for method in METHODS:
        print('Extracting original cases:', method, flush=True)
        backbone = select_model(method)
        records = {}
        all_descriptors = []
        selected_lookup = {native: local for local, native in enumerate(query_indices)}
        # Preserve the original extraction's native batch-16 layout. A small
        # case-only batch can move a dense hard query by one rank through FP32
        # kernel rounding, even while all Top-1 decisions are unchanged.
        for start in range(0, len(dataset.queries), 16):
            images = dataset.queries[start:start + 16]
            batch = torch.stack([image_tensor(item.path) for item in images]).to(args.device)
            wanted = [(native - start, selected_lookup[native]) for native in range(start, start + len(images))
                      if native in selected_lookup]
            if not wanted:
                all_descriptors.append(forward(method, batch).cpu())
                continue
            norm_values = []
            prefix = 1 + getattr(backbone.model, 'num_register_tokens', 0)
            hook = backbone.model.norm.register_forward_pre_hook(
                lambda module, values: norm_values.append(values[0][:, prefix:].float().norm(dim=-1).cpu()))
            try:
                with capture_cls_attention(backbone) as capture:
                    descriptor, auxiliary = forward(method, batch, True)
                attention = capture.attention.cpu()
            finally:
                hook.remove()
            all_descriptors.append(descriptor.cpu())
            for position, local in wanted:
                records[local] = {'descriptor': descriptor[position].cpu(), 'attention': attention[position],
                    'norm': norm_values[0][position], 'keep': auxiliary['assignment_keep'][position].flatten().cpu(),
                    'content': auxiliary['content_features'][position].cpu(),
                    'assignment': auxiliary['log_assignment'][position].cpu()}
        for key in ('descriptor', 'attention', 'keep', 'norm', 'content', 'assignment'):
            tensors[method + '.' + key] = torch.stack([records[index][key] for index in range(len(cases))])
        full_descriptors = torch.cat(all_descriptors)
        canonical = exact_retrieval(gallery_cpu, full_descriptors, dataset.positives, device=args.device,
            query_chunk_size=32, reference_chunk_size=4096, reference_cache_bytes=1024 * 1024**2)
        for image, row in zip(dataset.queries, canonical):
            if row['rank'] != known[image.id]['models'][method]['rank']:
                raise ValueError(f'Native full rerun rank mismatch: {image.id}/{method}')
        results = score(tensors[method + '.descriptor'], positives)
        for index, (case, result) in enumerate(zip(cases, results)):
            row = canonical[query_indices[index]]
            result['case_batch_rank'] = result['rank']
            result['rank'] = row['rank']
            result['canonical_native_batch_rank'] = row['rank']
            if result['top1_index'] != row['predicted_index']:
                raise ValueError('Native/case ranking disagrees on top prediction')
            query_id = case['query_id']
            for prefix_name, key in [('pos', 'positive_index'), ('neg', 'negative_index'), ('top1', 'top1_index')]:
                ref = dataset.references[result[key]]
                result[prefix_name + '_ref_id'] = ref.id
                result[prefix_name + '_ref_path'] = str(ref.path)
                result[prefix_name + '_utm_m'] = utm_distance(query_id, ref.id)
            if method == 'day':
                case['fixed_positive_ref_id'] = result['pos_ref_id']
                case['fixed_positive_ref_path'] = result['pos_ref_path']
                candidates = known[query_id]['models']
                error = candidates['sgpd_attention'] if candidates['sgpd_attention']['rank'] > 1 else candidates['global_control']
                wrong_id = error.get('predicted') if error['rank'] > 1 else result['neg_ref_id']
                if gallery_lookup[wrong_id] in positives[index]:
                    raise ValueError('Fixed wrong reference is positive')
                case['fixed_negative_ref_id'] = wrong_id
                case['fixed_negative_ref_path'] = str(dataset.references[gallery_lookup[wrong_id]].path)
            pos = torch.stack([gallery_cpu[gallery_lookup[case['fixed_positive_ref_id']]]])
            neg = torch.stack([gallery_cpu[gallery_lookup[case['fixed_negative_ref_id']]]])
            attribution = signed_patch_margin(tensors[method + '.descriptor'][index:index + 1],
                tensors[method + '.content'][index:index + 1], tensors[method + '.assignment'][index:index + 1], pos, neg)
            attention = tensors[method + '.attention'][index]
            attention = attention / attention.sum()
            keep = tensors[method + '.keep'][index]
            norm = tensors[method + '.norm'][index]
            signed = attribution['patch_contributions'][0]
            k = args.top_patches
            top_attn, top_keep, top_norm = (torch.topk(value, k).indices for value in (attention, keep, norm))
            def rho(a, b):
                return float(spearmanr(a.numpy(), b.numpy()).statistic) if a.std() > 0 and b.std() > 0 else None
            result.update({'attention_map': attention.reshape(16, 16).tolist(), 'keep_map': keep.reshape(16, 16).tolist(),
                'patch_norm_map': norm.reshape(16, 16).tolist(), 'patch_norm_definition': 'before final LayerNorm',
                'signed_margin_map': signed.reshape(16, 16).tolist(),
                'cls_contribution': float(attribution['cls_contribution'][0]),
                'fixed_cosine_margin': float(attribution['direct_margin'][0]),
                'attribution_residual': float(attribution['residual'][0]),
                'cls_energy_fraction': float(attribution['cls_energy_fraction'][0]),
                'map_statistics': {'attention_keep_spearman': rho(attention, keep),
                    'attention_abs_contribution_spearman': rho(attention, signed.abs()),
                    'attention_prenorm_spearman': rho(attention, norm),
                    'attention_keep_top26_overlap': len(set(top_attn.tolist()) & set(top_keep.tolist())) / k,
                    'attention_mass_on_norm_top26': float(attention[top_norm].sum()),
                    'keep_mass_on_attention_top26_fraction': float(keep[top_attn].sum() / keep.sum()),
                    'signed_contribution_on_attention_top26': float(signed[top_attn].sum()),
                    'attention_entropy': float(-(attention * attention.clamp_min(1e-30).log()).sum())}})
            case['methods'][method] = result
        print('Reproduced all native exact ranks:', method, len(canonical), flush=True)

    figure_ids = ['q0134', 'q0798', 'q0233', 'q0431', 'q0448', 'q0761', 'q0031', 'q0001']
    analysis = {'created_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'read-only failure diagnosis; no model selection or training',
        'inventory': str(args.inventory.resolve()), 'inventory_sha256': sha(args.inventory),
        'checkpoint_sha256': checkpoint_hash, 'gallery_sha256': gallery_hash,
        'map_preprocessing': {'image_size': [224, 224], 'resize': 'bilinear_stretch', 'exif_transpose': False},
        'methods': model_metadata, 'cases': cases, 'figure_case_ids': figure_ids,
        'attribution_definition': 'Exact additive cosine(positive)-cosine(fixed wrong) decomposition given current content and transport; not causal saliency',
        'intervention_definition': {'patch_count': args.top_patches, 'pixel_fill': 'per-query RGB mean, same positions across all methods',
            'mask_source': 'sgpd_attention maps for every method', 'random_seed': 42, 'random_repeats': 3,
            'readout_delete': 'Zero selected post-OT patch contributions, keep CLS and transport of other patches fixed, renormalize clusters/descriptor',
            'scope': 'sensitivity to fixed readout/pixel perturbations; no proof of a unique real-world cause'}}
    write(out / 'original_analysis.json', analysis)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(out / 'case_features.safetensors'))

    conditions = ('attention_top', 'keep_top', 'negative_contribution', 'positive_contribution', 'random_0', 'random_1', 'random_2')
    masks = []
    for index, case in enumerate(cases):
        values = case['methods']['sgpd_attention']
        maps = [torch.tensor(values[key]).flatten() for key in ('attention_map', 'keep_map', 'signed_margin_map')]
        chosen = [torch.topk(maps[0], args.top_patches).indices, torch.topk(maps[1], args.top_patches).indices,
                  torch.topk(-maps[2], args.top_patches).indices, torch.topk(maps[2], args.top_patches).indices]
        for repeat in range(3):
            generator = torch.Generator().manual_seed(42 + query_indices[index] * 100 + repeat)
            chosen.append(torch.randperm(256, generator=generator)[:args.top_patches])
        mask = torch.zeros(len(conditions), 256, dtype=torch.bool)
        for row, ids in enumerate(chosen):
            mask[row, ids] = True
        masks.append(mask)
        case['intervention_patch_indices'] = {name: ids.tolist() for name, ids in zip(conditions, chosen)}

    for method in METHODS:
        print('Fixed patch interventions:', method, flush=True)
        select_model(method)
        pixel_inputs, readout_descriptors, intervention_positives = [], [], []
        for index, case in enumerate(cases):
            mask = masks[index]
            pixels = inputs[index:index + 1].repeat(len(conditions), 1, 1, 1)
            image_mask = mask.reshape(-1, 16, 16).repeat_interleave(14, 1).repeat_interleave(14, 2)
            fill = inputs[index].mean((1, 2), keepdim=True)[None]
            pixels = torch.where(image_mask[:, None], fill, pixels)
            pixel_inputs.append(pixels)
            content = tensors[method + '.content'][index]
            assignment = tensors[method + '.assignment'][index, :-1].exp()
            masked_assignment = assignment[None] * (~mask[:, None])
            aggregated = (content[None, :, None] * masked_assignment[:, None]).sum(-1)
            token = F.normalize(tensors[method + '.descriptor'][index, :256], dim=0).repeat(len(conditions), 1)
            rebuilt = F.normalize(torch.cat([token, F.normalize(aggregated, dim=1).flatten(1)], 1), dim=1)
            readout_descriptors.append(rebuilt)
            intervention_positives.extend([positives[index]] * len(conditions))
        pixel_inputs = torch.cat(pixel_inputs)
        descriptors = []
        for start in range(0, len(pixel_inputs), args.batch_size):
            descriptors.append(forward(method, pixel_inputs[start:start + args.batch_size].to(args.device)).cpu())
        pixel_results = score(torch.cat(descriptors), intervention_positives)
        readout_results = score(torch.cat(readout_descriptors), intervention_positives)
        for index, case in enumerate(cases):
            original = case['methods'][method]
            original['counterfactuals'] = {}
            for offset, name in enumerate(conditions):
                point = index * len(conditions) + offset
                original['counterfactuals'][name] = {}
                for perturbation, result in [('pixel_mean_fill', pixel_results[point]), ('post_ot_readout_delete', readout_results[point])]:
                    result['delta_margin'] = result['margin'] - original['margin']
                    result['delta_rank'] = result['rank'] - original['rank']
                    result['top1_ref_id'] = dataset.references[result['top1_index']].id
                    original['counterfactuals'][name][perturbation] = result
        del pixel_inputs, descriptors, readout_descriptors
    analysis['verification'] = {'fresh_original_ranks_reproduced': len(dataset.queries) * len(METHODS),
        'native_extraction_batch_size': 16, 'native_retrieval_query_chunk': 32,
        'case_batch_rank_rounding_differences': [
            {'case_id': case['case_id'], 'method': method, 'native': case['methods'][method]['rank'],
             'case_batch': case['methods'][method]['case_batch_rank']}
            for case in cases for method in METHODS if case['methods'][method]['rank'] != case['methods'][method]['case_batch_rank']],
        'all_case_method_attributions_verified': True, 'max_absolute_attribution_residual': max(
            abs(case['methods'][method]['attribution_residual']) for case in cases for method in METHODS),
        'model_weights_unchanged': True}
    for method in METHODS[1:]:
        if sha(Path(model_metadata[method]['directory']) / 'adapter.safetensors') != model_metadata[method]['weights_sha256']:
            raise ValueError('Weights changed during diagnosis')
    write(out / 'analysis.json', analysis)
    print('FAILURE ANALYSIS COMPLETE:', out / 'analysis.json', flush=True)


if __name__ == '__main__':
    main()

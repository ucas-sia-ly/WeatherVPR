"""Train admission, frozen-statistics coordinates, calibration and interventions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from weather_vpr.training.domain_coverage import (
    DiagnosticImages, apply_statistic_intervention, balanced_sample, coverage_report,
    descriptor_from_prefix, extract_prefix_statistics, fit_projection, group_weights,
    intervention_plan, intervention_report, model_forward_provenance, model_state_digest, patch_statistics,
    prefix_tokens, project, read_stats_cache, reference_group_split, sha256,
    signed_image_rows, validate_train_rows, verify_row_provenance, weighted_summary, write_stats_cache,
)
from weather_vpr.data.source_night import SourceNightConfig


def row(index, source='qwen_night', group=None):
    return {'sample_id': f'{source}:{index}', 'source': source, 'split': 'train',
            'path': f'/train/images/{source}/{index}.jpg',
            'group_id': str(index//2 if group is None else group),
            'metadata': {'geographic_independence_known': False}}


class ToyTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, images):
        patches = images.flatten(2).transpose(1, 2)
        return torch.cat((patches.mean(1, keepdim=True), patches), 1)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.norm_layer = ToyTokenModel(), True

    def forward(self, images):
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        patches = tokens[:, 1:].reshape(len(images), 2, 2, 3).permute(0, 3, 1, 2)
        return patches, tokens[:, 0]


class ToyAggregator(nn.Module):
    def forward(self, features, return_aux=True):
        patches, cls = features
        descriptor = F.normalize(torch.cat((patches.flatten(1), cls), 1), dim=1)
        aux = {'assignment_keep': patches.mean(1).sigmoid(),
               'log_assignment': patches.log_softmax(1)}
        return (descriptor, aux) if return_aux else descriptor


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.aggregator = ToyBackbone(), ToyAggregator()

    def forward(self, images, return_aux=True):
        return self.aggregator(self.backbone(images), return_aux=return_aux)


class DomainCoverageTests(unittest.TestCase):
    def test_train_only_admission_and_source_night_transform(self):
        validate_train_rows([row(1)])
        # NightCity's official training filenames include city names. A city
        # substring is not the Tokyo247 evaluation dataset's root directory.
        validate_train_rows([{**row(2, 'nightcity_train'),
                              'path': '/data/NightCity/images/train/Tokyo_0476.png'}])
        for mutation in ({'split': 'test'}, {'source': 'svox'}, {'group_id': ''},
                         {'path': '/data/BDD/val/images/a.jpg'},
                         {'path': '/data/RobotCar/train/a.jpg'},
                         {'path': '/data/SVOX/train/a.jpg'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_train_rows([{**row(1), **mutation}])
        with self.assertRaises(ValueError):
            validate_train_rows([row(1), row(1)])
        with self.assertRaises(ValueError):
            validate_train_rows([row(1, 'source_night')])
        source = row(1, 'source_night')
        source['metadata']['transform'] = 'source_night'
        validate_train_rows([source])

    def test_group_balanced_sampling_is_worker_independent_and_rng_private(self):
        rows = [row(i, group='large') for i in range(100)] + [row(101, group='small')]
        before = random.getstate()
        selected, counts = balanced_sample(rows, maximum=4, seed=42)
        self.assertEqual(before, random.getstate())
        self.assertEqual(selected, balanced_sample(list(reversed(rows)), maximum=4, seed=42)[0])
        self.assertIn('small', {item['group_id'] for item in selected[:2]})
        self.assertEqual(len({item['sample_id'] for item in selected}), 4)
        self.assertEqual(counts['qwen_night'], {'full_rows': 101, 'full_groups': 2,
                                               'sampled_rows': 4, 'sampled_groups': 2})

    def test_synthetic_sources_select_identical_scenes_and_hold_out_whole_places(self):
        qwen = [row(i, group=f'place_{i//2}') for i in range(40)]
        source = [{**item, 'source': 'source_night',
                   'metadata': {'transform': 'source_night'}} for item in qwen]
        selected, counts = balanced_sample(list(reversed(qwen + source)), maximum=12, seed=42)
        ids = {name: [item['sample_id'] for item in selected if item['source'] == name]
               for name in ('qwen_night', 'source_night')}
        self.assertEqual(ids['qwen_night'], ids['source_night'])
        self.assertTrue(counts['source_night']['paired_synthetic_scenes'])
        fit, heldout = reference_group_split(selected)
        for item in selected:
            memberships = [index in fit for index, other in enumerate(selected)
                           if other['group_id'] == item['group_id']]
            self.assertEqual(len(set(memberships)), 1)
        self.assertFalse(set(fit) & set(heldout))
        source[0] = {**source[0], 'group_id': 'incorrect_other_place'}
        with self.assertRaises(ValueError):
            balanced_sample(qwen + source, maximum=12)

    def test_patch_statistics_exclude_cls_and_use_population_variance_fp32(self):
        tokens = torch.tensor([[[10000., -9999.], [1., 4.], [3., 8.]]], dtype=torch.bfloat16)
        stats = patch_statistics(tokens)
        self.assertEqual(stats.dtype, torch.float32)
        self.assertTrue(torch.equal(stats[:, :2], torch.tensor([[2., 6.]])))
        self.assertTrue(torch.allclose(stats[:, 2:], torch.tensor([[1.000001, 4.000001]]).sqrt().log()))
        altered = tokens.clone()
        altered[:, 0] = -3333
        self.assertTrue(torch.equal(stats, patch_statistics(altered)))

    def test_absolute_prefix_outputs_and_state_unchanged(self):
        base = ToyBase().eval()
        images = torch.arange(24).float().reshape(2, 3, 2, 2)
        before = model_state_digest(base)
        stats = extract_prefix_statistics(base, images)
        expected = patch_statistics(base.backbone.model.prepare_tokens_with_masks(images))
        self.assertEqual(set(stats), {'block_3', 'block_5', 'block_7'})
        for value in stats.values():
            self.assertTrue(torch.equal(value, expected))
        self.assertEqual(before, model_state_digest(base))
        with self.assertRaises(ValueError):
            extract_prefix_statistics(base, images, layers=(8,))

    def test_forward_provenance_binds_local_model_code_and_backend_precision(self):
        root = Path(__file__).resolve().parents[1]
        provenance = model_forward_provenance(ToyBase(), root)
        self.assertEqual(provenance['local_module_files_sha256']['tests/test_domain_coverage.py'], sha256(__file__))
        self.assertIn('torch_version', provenance)
        self.assertIn('cuda_matmul_allow_tf32', provenance)

    def test_signed_image_bytes_source_index_seed_and_modified_input_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'image.png'
            Image.new('RGB', (8, 8), (140, 150, 160)).save(path)
            source = row(1, 'source_night')
            source['path'] = str(path)
            source['metadata'].update(transform='source_night', original_sample_index=7)
            signed = signed_image_rows([source], 42)
            self.assertEqual(signed[0]['sha256'], sha256(path))
            dataset = DiagnosticImages(signed, SourceNightConfig().to_dict())
            tensor, brightness = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertGreater(brightness, 0)
            self.assertTrue(torch.equal(tensor, dataset[0][0]))
            Image.new('RGB', (8, 8), (1, 2, 3)).save(path)
            with self.assertRaises(ValueError):
                dataset[0]
            with self.assertRaises(ValueError):
                signed_image_rows([{**source, 'sha256': 'a'*64}], 42)
            source['metadata'].pop('original_sample_index')
            with self.assertRaises(ValueError):
                signed_image_rows([source], 42)

    def test_source_diagnostic_pixels_exactly_match_existing_training_pipeline(self):
        from weather_vpr.utils.backend import enable_backend_imports
        from weather_vpr.data.source_night import SourceNightAugmentation
        from weather_vpr.data.student_augmentation import sample_augmentation_seed, student_training_tensor
        enable_backend_imports()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'day.png'
            pixels = np.arange(9*11*3, dtype=np.uint8).reshape(9, 11, 3)
            Image.fromarray(pixels).save(path)
            source = row(1, 'source_night')
            source['path'] = str(path)
            source['metadata'].update(transform='source_night', original_sample_index=7)
            config = SourceNightConfig()
            signed = signed_image_rows([source], 42)
            diagnostic, _ = DiagnosticImages(signed, config.to_dict())[0]
            with Image.open(path) as image:
                actual_training, _ = student_training_tensor(
                    image, (224, 224), SourceNightAugmentation(config),
                    seed=sample_augmentation_seed(42, 0, 7, view='source_night'))
            self.assertTrue(torch.equal(diagnostic, actual_training))

    def test_cache_binds_every_identity_field_and_actual_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'stats.safetensors'
            identity = {'layers': [7], 'selected_rows': [row(0), row(1)], 'checkpoint_sha256': 'teacher',
                        'forward_precision': 'fp32', 'source_config': {'strength': 1}}
            tensors = {'block_7': torch.arange(12).float().reshape(2, 6), 'brightness': torch.tensor([.1, .2])}
            write_stats_cache(path, tensors, identity)
            restored = read_stats_cache(path, identity)
            self.assertTrue(torch.equal(restored['block_7'], tensors['block_7']))
            changed = copy.deepcopy(identity)
            changed['source_config']['strength'] = 2
            with self.assertRaises(ValueError):
                read_stats_cache(path, changed)
            with self.assertRaises(FileExistsError):
                write_stats_cache(path, tensors, identity)
            with path.open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaises(ValueError):
                read_stats_cache(path, identity)

    def test_changed_annotation_or_review_rejects_even_with_unchanged_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'annotation.json'
            path.write_text('{"split":"train"}')
            source = row(0)
            source['metadata'].update(annotation='annotation.json', annotation_sha256=sha256(path))
            checked = verify_row_provenance([source], temporary)
            self.assertIn(str(path), checked)
            path.write_text('{"split":"test"}')
            with self.assertRaises(ValueError):
                verify_row_provenance([source], temporary)

    def test_reference_fit_groups_exclude_heldout_across_sources(self):
        rows = [row(i, source, group=f'place_{i}') for source in ('qwen_night', 'source_night')
                for i in range(20)] + [row(50, 'bdd_train_night')]
        fit, heldout = reference_group_split(rows)
        self.assertFalse({rows[i]['group_id'] for i in fit} & {rows[i]['group_id'] for i in heldout})
        self.assertEqual(len(fit)+len(heldout), 40)
        self.assertNotIn(len(rows)-1, fit+heldout)
        with self.assertRaises(ValueError):
            reference_group_split([row(1, group='one'), row(2, group='one')])

    def test_group_weighted_summaries_do_not_count_repeats_as_independent_groups(self):
        weights = group_weights(['large']*100+['small'])
        self.assertAlmostEqual(float(weights[:100].sum()), .5)
        summary = weighted_summary([0.]*100+[10.], ['large']*100+['small'])
        self.assertAlmostEqual(summary['group_weighted_mean'], 5.)
        self.assertEqual(summary['groups'], 2)

    def test_projection_uses_reference_only_and_detects_outside_coverage(self):
        generator = torch.Generator().manual_seed(1)
        rows = [row(i, source, group=f'p{i}') for source in ('qwen_night', 'source_night') for i in range(20)]
        rows += [row(i, 'bdd_train_night', group=f'road{i}') for i in range(20)]
        values = torch.randn(60, 8, generator=generator)
        values[-20:] += 100
        tensors = {'block_7': values, 'brightness': torch.full((60,), .2)}
        report = coverage_report(tensors, rows, layers=[7], pca_dimensions=4)
        self.assertEqual(report['decision'], 'requires_content_review')
        outside = report['layers']['7']['sources']['bdd_train_night']
        self.assertEqual(outside['within_reference_heldout_p95_group_weighted_fraction'], 0)
        fit, _ = reference_group_split(rows)
        fitted = fit_projection(values[fit], [rows[i]['group_id'] for i in fit], 4)
        altered = values.clone()
        altered[-20:] += 10000
        fitted2 = fit_projection(altered[fit], [rows[i]['group_id'] for i in fit], 4)
        self.assertTrue(torch.equal(fitted['directions'], fitted2['directions']))
        self.assertTrue(torch.equal(project(values[fit], fitted), project(altered[fit], fitted2)))

    def test_matched_intervention_changes_only_selected_patches_and_preserves_cls(self):
        generator = torch.Generator().manual_seed(2)
        tokens = torch.randn(4, 5, 3, generator=generator).bfloat16()
        donor = patch_statistics(tokens.float()+5.)
        plan = intervention_plan(4, 4, 4, seed=9, step=0, probability=1, fraction=.5)
        before = torch.random.get_rng_state().clone()
        changed, measured = apply_statistic_intervention(tokens, donor, plan)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(measured['cls_unchanged'])
        self.assertTrue(measured['unselected_patches_unchanged'])
        self.assertTrue(torch.equal(changed[:, 0], tokens[:, 0]))
        self.assertTrue(any(value > 0 for value in measured['relative_centered_rms_delta']))
        identical, _ = apply_statistic_intervention(tokens, donor, {'selected_images': [], 'lambdas': []})
        self.assertIs(identical, tokens)
        stronger, _ = apply_statistic_intervention(tokens, donor+2., plan)
        self.assertFalse(torch.equal(changed, stronger))

    def test_intervention_plan_matches_banks_avoids_self_donors_and_global_rng(self):
        before = random.getstate()
        plan = intervention_plan(4, 256, 4, seed=42, step=0, forbidden=[0, 1, 2, 3])
        self.assertEqual(before, random.getstate())
        self.assertEqual(plan, intervention_plan(4, 256, 4, seed=42, step=0, forbidden=[0, 1, 2, 3]))
        self.assertTrue(all(i != donor for i, donor in enumerate(plan['donor_indices'])))

    def test_full_intervention_manual_identity_matches_original_and_no_weight_updates(self):
        base = ToyBase().eval()
        images = torch.arange(96).float().reshape(8, 3, 2, 2)/10.
        rows = [row(i) for i in range(4)] + [row(i, 'bdd_train_night') for i in range(4)]
        statistics = patch_statistics(base.backbone.model.prepare_tokens_with_masks(images))
        dataset = [(image, .2) for image in images]
        before = model_state_digest(base)
        report = intervention_report(base, dataset, {'block_7': statistics}, rows,
                                     device='cpu', batch_size=2, recipients=4)
        self.assertEqual(report['matched_donor_count_per_source'], 4)
        self.assertTrue(report['identity_equals_original_day'])
        self.assertTrue(report['same_recipients_donor_indices_masks_lambdas'])
        self.assertEqual(report['decision'], 'descriptive_only')
        enabled = sum(len(record['selected_images']) for record in report['records'])
        self.assertEqual(report['enabled_recipient_count'], enabled)
        for source in report['sources'].values():
            self.assertEqual(source['relative_centered_rms_delta']['count'], 4)
            self.assertEqual(source['enabled_only']['relative_centered_rms_delta']['count'], enabled)
            self.assertLessEqual(source['gain'].get('groups', 0), enabled)
        self.assertIn('not evidence of matched clean-Day place alignment', report['limitation'])
        self.assertEqual(before, model_state_digest(base))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))


if __name__ == '__main__':
    unittest.main()

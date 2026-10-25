"""Focused CPU-only checks for frozen external comparison bookkeeping."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('revised_night_queue', ROOT / 'scripts/run_revised_night_evaluation.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def evaluation(ranks):
    count = len(ranks)
    errors = [{'query_id': name, 'rank': rank} for name, rank in ranks.items() if rank > 1]
    return {'num_evaluated_queries': count, 'num_queries_without_positives': 0,
            'recall': {f'R@{k}': sum(rank <= k for rank in ranks.values()) / count for k in (1, 5, 10)},
            'hard_cases_saved': True, 'error_queries': errors, 'num_error_queries': len(errors),
            'protocol': {'ground_truth': 'public_fixture', 'split': 'test'}}


def compact(ranks):
    return queue.compact_evaluation(evaluation(ranks), list(ranks))


class RevisedNightQueueTests(unittest.TestCase):
    def test_complete_errors_reproduce_all_topk(self):
        result = compact({'a': 1, 'b': 3, 'c': 9, 'd': 22})
        self.assertEqual(result['hits'], {'R@1': 1, 'R@5': 2, 'R@10': 3})
        self.assertEqual(result['error_ranks'], {'b': 3, 'c': 9, 'd': 22})

    def test_missing_errors_are_rejected_even_with_plausible_recall(self):
        data = evaluation({'a': 1, 'b': 8})
        data['error_queries'] = []
        with self.assertRaisesRegex(RuntimeError, 'per-query errors'):
            queue.compact_evaluation(data, ['a', 'b'])

    def test_wrong_query_identity_is_rejected(self):
        data = evaluation({'a': 1, 'b': 8})
        data['error_queries'][0]['query_id'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'per-query errors'):
            queue.compact_evaluation(data, ['a', 'b'])

    def test_unevaluated_queries_are_rejected(self):
        data = evaluation({'a': 1, 'b': 8})
        data['num_queries_without_positives'] = 1
        with self.assertRaisesRegex(RuntimeError, 'public positives'):
            queue.compact_evaluation(data, ['a', 'b'])

    def test_command_fixed_native_and_bounded(self):
        args = queue.parse_args(['--output', '/tmp/night-fixture', '--models', 'seed43_A',
                                 '--benchmarks', 'svox_test'])
        models = {'seed43_A': {'directory': '/tmp/adapter'}}
        benchmarks = {'svox_test': {'root': '/tmp/svox', 'cache': '/tmp/gallery'}}
        command = queue.bounded_command(args, 'seed43_A', 'svox_test', models, benchmarks)
        self.assertEqual(command[command.index('--reference-cache-mib') + 1], '4096')
        self.assertEqual(command[command.index('--max-gib') + 1], '7')
        self.assertIn('--save-hard-cases', command)
        self.assertNotIn('--query-scales', command)
        self.assertNotIn('--query-fusion-alpha', command)
        self.assertEqual(command[command.index('--split') + 1], 'test')

    def test_aggregate_matches_query_repairs_and_three_seed_statistics(self):
        baseline = compact({'a': 1, 'b': 1, 'c': 1, 'd': 8})
        plan = {'models': {f'seed{s}_{a}': {} for s in (42, 43, 44) for a in 'ABC'},
                'benchmarks': {'svox_test': {'expected_num_queries': 4}},
                'intent': 'exploratory', 'limitations': 'fixture', 'reserved': 'SF-XL'}
        state = {'state': 'complete', 'jobs': {}}
        for seed in (42, 43, 44):
            for arm in 'ABC':
                ranks = {'a': 1, 'b': 1, 'c': 8, 'd': 8} if arm == 'A' else {'a': 1, 'b': 8, 'c': 1, 'd': 1}
                item = {'checkpoint_sha256': 'day', 'gallery_sha256': 'gallery', 'query_sha256': None,
                        'preprocessing': {}, 'query_preprocessing': {}, 'protocol': {},
                        'baseline': baseline, 'night': compact(ranks), 'historical_baseline_matches': True}
                state['jobs'][f'seed{seed}_{arm}.svox_test'] = {'evaluation': item}
        result = queue.aggregate(plan, state)
        dataset = result['benchmarks']['svox_test']
        paired = dataset['paired_changes']['seed42_B_vs_seed42_A']['R@1']
        self.assertEqual(paired['repaired'], ['c', 'd'])
        self.assertEqual(paired['regressed'], ['b'])
        self.assertEqual(dataset['P0_three_seed']['B']['metrics']['R@1']['mean'], .75)
        self.assertEqual(dataset['P0_three_seed']['B']['metrics']['R@1']['sample_std'], 0.)
        self.assertEqual(result['completed_jobs'], 9)
        changed = copy.deepcopy(state)
        changed['jobs']['seed43_A.svox_test']['evaluation']['gallery_sha256'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'gallery_sha256'):
            queue.aggregate(plan, changed)

    def test_result_must_use_bound_adapter_and_native_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'result.json'
            ids = ['a', 'b']
            baseline = evaluation({'a': 1, 'b': 8})
            night = evaluation({'a': 1, 'b': 2})
            raw = {'checkpoint_sha256': 'day', 'gallery_sha256': 'gallery', 'image_size': [224, 224],
                   'preprocessing': {}, 'query_preprocessing': {'mode': 'native_single_scale'},
                   'split': 'test', 'query_group': 'queries_night', 'baseline': baseline,
                   'adapter': {'directory': temporary, 'evaluation': night, 'scale': 1}}
            plan = {'models': {'seed42_A': {'directory': temporary, 'adapter_sha256': 'weights',
                                           'metadata_sha256': 'meta', 'checkpoint_sha256': 'day'}},
                    'benchmarks': {'svox_test': {'gallery_sha256': 'gallery', 'query_ids': ids,
                                                'historical_baseline_hits_for_audit_only': {'R@1': 1, 'R@5': 1, 'R@10': 2}}}}
            path.write_text(json.dumps(raw))
            result = queue.read_evaluation(path, plan, 'seed42_A', 'svox_test')
            self.assertTrue(result['historical_baseline_matches'])
            raw['adapter']['scale'] = .5
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RuntimeError, 'scaling'):
                queue.read_evaluation(path, plan, 'seed42_A', 'svox_test')
            raw['adapter']['scale'] = 1
            raw['query_preprocessing']['mode'] = 'query_multiscale'
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(RuntimeError, 'preprocessing'):
                queue.read_evaluation(path, plan, 'seed42_A', 'svox_test')

    def test_frozen_phase_verification_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'adapter'
            path.write_bytes(b'original')
            plan = {'shared_files': [], 'model_files': {'A': [str(path)]},
                    'benchmarks': {'night': {'files': []}},
                    'required_files_sha256': {str(path): queue.common.digest(path)}, 'code_sha256': {}}
            queue.verify_plan(plan, 'A', 'night')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Frozen evaluation input'):
                queue.verify_plan(plan, 'A', 'night')


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/summarize_stage1_strata.py'
SPEC = importlib.util.spec_from_file_location('stage1_strata', SCRIPT)
strata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(strata)


def evaluation(errors):
    return {'num_queries': 2, 'error_ranks': errors,
            'hits': {f'R@{k}': 2 - sum(rank > k for rank in errors.values()) for k in (1, 5, 10)}}


class Stage1StrataTests(unittest.TestCase):
    def test_predeclared_bins_include_boundaries(self):
        self.assertEqual(strata.brightness_bin(0), '[0,0.1)')
        self.assertEqual(strata.brightness_bin(.1), '[0.1,0.25)')
        self.assertEqual(strata.brightness_bin(.25), '[0.25,0.5)')
        self.assertEqual(strata.brightness_bin(1), '[0.5,1]')
        self.assertEqual(strata.spatial_bin('@0199.9@0400@date@.jpg'), 'utm200m_0_2')
        with self.assertRaises(ValueError):
            strata.spatial_bin('unknown.jpg')

    def test_real_pixels_and_exact_canonical_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (4, 4), (0, 0, 0)).save(root / '@0200@0400@time@.jpg')
            row = strata.query_metadata(root)[0]
            self.assertEqual(row['query_id'], 'images/train/queries_night/@0200@0400@time@.jpg')
            self.assertEqual(row['brightness'], 0)
            self.assertEqual(row['spatial_bin'], 'utm200m_1_2')
            self.assertEqual(len(row['sha256']), 64)

    def test_paired_topk_changes_group_without_losing_success_queries(self):
        queries = [{'query_id': 'q1', 'brightness_bin': '[0,0.1)', 'spatial_bin': 'a'},
                   {'query_id': 'q2', 'brightness_bin': '[0.5,1]', 'spatial_bin': 'b'}]
        a, b = evaluation({'q1': 3}), evaluation({'q2': 8})
        def record(value):
            return {'evaluation': {'gallery_sha256': 'gallery', 'checkpoint_sha256': 'day',
                                    'baseline': {}, 'night': value}}
        summary = {'state': 'complete', 'smoke': False,
                   'records': {'seed43_A': record(a), 'seed43_B': record(b), 'seed43_C': record(a)}}
        report = strata.build_report(summary, queries)
        changes = report['paired_changes_by_stratum']['seed43_B_vs_A']
        self.assertEqual(changes['R@1']['spatial_bin']['a']['repaired'], ['q1'])
        self.assertEqual(changes['R@1']['spatial_bin']['b']['regressed'], ['q2'])
        self.assertEqual(changes['R@5']['spatial_bin']['a']['repaired_count'], 0)
        self.assertEqual(changes['R@5']['spatial_bin']['b']['regressed_count'], 1)
        self.assertEqual(changes['R@10']['spatial_bin']['b']['regressed_count'], 0)
        self.assertEqual(sum(g['queries'] for g in changes['R@1']['brightness_bin'].values()), 2)

    def test_incomplete_or_unknown_query_results_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown'):
            strata.validate_evaluation(evaluation({'not_a_query': 3}), {'q1', 'q2'})
        bad = evaluation({})
        bad['hits']['R@1'] = 1
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            strata.validate_evaluation(bad, {'q1', 'q2'})
        with self.assertRaisesRegex(ValueError, 'completed'):
            strata.build_report({'state': 'running', 'records': {}}, [])


if __name__ == '__main__':
    unittest.main()

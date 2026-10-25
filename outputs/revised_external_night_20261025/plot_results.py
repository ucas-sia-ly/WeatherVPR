#!/usr/bin/env python3
"""Plot only the complete, frozen 13-model by 4-protocol external evaluation.

This output-only helper imports no training, inference, or dataset code. It
checks the complete summary and its 52 result-file hashes before plotting.
No significance tests or confidence intervals are computed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = (
    'svox_test', 'tokyo247_night', 'robotcar_night', 'robotcar_night_rain',
)
COUNTS = dict(zip(BENCHMARKS, (823, 105, 197, 224)))
LABELS = (
    'SVOX test\n823 queries', 'Tokyo247 Night\n105 queries',
    'RobotCar Night*\n197 queries', 'RobotCar Night-rain*\n224 queries',
)
SEEDS = (42, 43, 44)
MODELS = tuple(f'seed{seed}_{arm}' for seed in SEEDS for arm in 'ABC') + (
    'G1_A', 'G1_B', 'G2_A', 'G2_B',
)
METRICS = ('R@1', 'R@5', 'R@10')


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_compact(evaluation: dict, count: int) -> None:
    if evaluation['num_queries'] != count:
        raise ValueError('Evaluation denominator differs from the frozen protocol')
    ranks = evaluation['error_ranks']
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank <= 1
           for rank in ranks.values()):
        raise ValueError('Invalid full Top-1 error-rank map')
    for k in (1, 5, 10):
        metric = f'R@{k}'
        hits = count - sum(rank > k for rank in ranks.values())
        recall = evaluation['recall'][metric]
        if (evaluation['hits'][metric] != hits or not math.isfinite(recall)
                or not math.isclose(recall, hits / count, abs_tol=1e-12)):
            raise ValueError(f'Ranks, hits, and recall disagree for {metric}')


def load_complete_summary(path: Path) -> dict:
    summary = json.loads(path.read_text())
    if (summary.get('state') != 'complete' or summary.get('completed_jobs') != 52
            or summary.get('planned_jobs') != 52):
        raise ValueError('Refusing partial results: require state=complete and 52/52 jobs')
    if set(summary['benchmarks']) != set(BENCHMARKS):
        raise ValueError('Require exactly the four frozen external protocols')
    for name in BENCHMARKS:
        benchmark = summary['benchmarks'][name]
        count = COUNTS[name]
        if benchmark['num_queries'] != count or set(benchmark['models']) != set(MODELS):
            raise ValueError(f'{name}: unexpected model set or denominator')
        check_compact(benchmark['baseline'], count)
        for model, record in benchmark['models'].items():
            result_path = Path(record['path'])
            if not result_path.is_absolute():
                result_path = ROOT / result_path
            if sha256(result_path) != record['result_sha256']:
                raise ValueError(f'{name}/{model}: saved result-file hash mismatch')
            if record['baseline'] != benchmark['baseline']:
                raise ValueError(f'{name}/{model}: original Day baseline differs')
            check_compact(record['night'], count)
        for seed in SEEDS:
            pair = benchmark['paired_changes'][f'seed{seed}_B_vs_seed{seed}_A']['R@1']
            before = benchmark['models'][f'seed{seed}_A']['night']
            after = benchmark['models'][f'seed{seed}_B']['night']
            old, new = set(before['error_ranks']), set(after['error_ranks'])
            if (set(pair['repaired']) != old - new
                    or set(pair['regressed']) != new - old
                    or pair['repaired_count'] != len(old - new)
                    or pair['regressed_count'] != len(new - old)):
                raise ValueError(f'{name}/seed{seed}: paired Top-1 changes disagree')
    return summary


def plot_data(summary: dict) -> dict:
    result = {}
    for name in BENCHMARKS:
        benchmark = summary['benchmarks'][name]
        count = benchmark['num_queries']
        baseline_hits = benchmark['baseline']['hits']['R@1']
        models = benchmark['models']
        hits = {key: record['night']['hits']['R@1'] for key, record in models.items()}
        paired = []
        for seed in SEEDS:
            pair = benchmark['paired_changes'][f'seed{seed}_B_vs_seed{seed}_A']['R@1']
            net = hits[f'seed{seed}_B'] - hits[f'seed{seed}_A']
            if net != pair['repaired_count'] - pair['regressed_count']:
                raise ValueError('Paired repairs/regressions disagree with the hit difference')
            paired.append({'seed': seed, 'net_correct_queries': net,
                           'delta_pp': 100 * net / count,
                           'repaired': pair['repaired_count'],
                           'regressed': pair['regressed_count']})
        arms = {}
        for arm in 'ABC':
            seed_hits = [hits[f'seed{seed}_{arm}'] for seed in SEEDS]
            deltas = [100 * (value - baseline_hits) / count for value in seed_hits]
            arms[arm] = {'seed_hits': seed_hits, 'seed_delta_vs_day_pp': deltas,
                         'mean_delta_vs_day_pp': statistics.mean(deltas),
                         'sample_sd_pp': statistics.stdev(deltas)}
            recorded = benchmark['P0_three_seed'][arm]['metrics']['R@1']
            if (recorded['hits'] != seed_hits
                    or not math.isclose(recorded['mean'], statistics.mean(seed_hits) / count,
                                        abs_tol=1e-12)):
                raise ValueError('Recorded three-seed aggregate differs from the plot data')
        scale = {}
        for arm in 'AB':
            net = hits[f'G2_{arm}'] - hits[f'G1_{arm}']
            scale[arm] = {'seed': 42, 'G1_hits': hits[f'G1_{arm}'],
                          'G2_hits': hits[f'G2_{arm}'], 'net_correct_queries': net,
                          'delta_pp': 100 * net / count}
        result[name] = {'num_queries': count, 'original_day_top1_hits': baseline_hits,
                        'paired_B_minus_A': paired, 'P0_vs_day': arms,
                        'source_scale_G2_minus_G1': scale}
    return result


def render(data: dict, directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'savefig.dpi': 200,
    })
    figure, axes = plt.subplots(3, 1, figsize=(12.2, 11.7), sharex=True)
    figure.subplots_adjust(left=.105, right=.98, top=.90, bottom=.16, hspace=.35)
    figure.suptitle('Fixed models on external night protocols', x=.105, y=.975,
                   ha='left', fontsize=18, fontweight='bold')
    figure.text(.105, .944, 'Top-1 recall differences in percentage points (pp); '
                'positive values indicate more correct queries.', fontsize=11, color='#475569')
    x = list(range(len(BENCHMARKS)))
    for axis in axes:
        axis.axhline(0, color='#334155', linewidth=1.1, zorder=1)
        axis.grid(axis='y', color='#e2e8f0', linewidth=.7, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylabel('Top-1 recall difference (pp)')
        axis.set_xlim(-.6, len(x) - .4)

    seed_colors = ('#0072B2', '#D55E00', '#009E73')
    seed_markers = ('o', 's', 'D')
    for index, (seed, color, marker) in enumerate(zip(SEEDS, seed_colors, seed_markers)):
        values = [data[name]['paired_B_minus_A'][index]['delta_pp'] for name in BENCHMARKS]
        axes[0].scatter([value + (index - 1) * .16 for value in x], values,
                        color=color, marker=marker, s=57, label=f'Seed {seed}', zorder=4)
    means = [statistics.mean(row['delta_pp'] for row in data[name]['paired_B_minus_A'])
             for name in BENCHMARKS]
    axes[0].scatter(x, means, marker='_', s=360, linewidths=2.4, color='#111827',
                    label='Three-seed mean', zorder=5)
    axes[0].set_title('(a) Paired real auxiliary effect: B minus A', loc='left', pad=12)
    axes[0].legend(loc='best', frameon=False, ncols=4, fontsize=9)

    arm_colors = {'A': '#64748b', 'B': '#0072B2', 'C': '#D55E00'}
    for index, arm in enumerate('ABC'):
        positions = [value + (index - 1) * .23 for value in x]
        means = [data[name]['P0_vs_day'][arm]['mean_delta_vs_day_pp'] for name in BENCHMARKS]
        deviations = [data[name]['P0_vs_day'][arm]['sample_sd_pp'] for name in BENCHMARKS]
        for pos, name in zip(positions, BENCHMARKS):
            seed_values = data[name]['P0_vs_day'][arm]['seed_delta_vs_day_pp']
            axes[1].scatter([pos - .035, pos, pos + .035], seed_values,
                            color=arm_colors[arm], alpha=.35, s=19, zorder=3)
        axes[1].errorbar(positions, means, yerr=deviations, fmt='o', markersize=6,
                         color=arm_colors[arm], capsize=4, linewidth=1.6,
                         label=f'{arm}: mean ± sample SD', zorder=4)
    axes[1].set_title('(b) A/B/C versus original Day: three training seeds', loc='left', pad=12)
    axes[1].legend(loc='best', frameon=False, ncols=3, fontsize=9)

    for index, arm in enumerate('AB'):
        positions = [value + (index - .5) * .24 for value in x]
        rows = [data[name]['source_scale_G2_minus_G1'][arm] for name in BENCHMARKS]
        values = [row['delta_pp'] for row in rows]
        axes[2].scatter(positions, values, color=arm_colors[arm], marker='o' if arm == 'A' else 's',
                        s=60, label=f'Arm {arm}: G2 minus G1', zorder=4)
        for pos, row in zip(positions, rows):
            axes[2].annotate(f"{row['net_correct_queries']:+d}", (pos, row['delta_pp']),
                             xytext=(0, 8 if arm == 'A' else -16), textcoords='offset points',
                             ha='center', fontsize=9, color=arm_colors[arm])
    axes[2].set_title('(c) Independent Source pool scaling: one seed (42); labels show net correct queries',
                       loc='left', pad=12)
    axes[2].legend(loc='best', frameon=False, ncols=2, fontsize=9)
    axes[2].set_xticks(x, LABELS)
    axes[2].tick_params(axis='x', length=0, pad=12)

    figure.text(.105, .092, 'A: no extra auxiliary; B: paired real auxiliary; C: extra synthetic auxiliary.',
                fontsize=9, color='#475569')
    figure.text(.105, .069, 'Exploratory external comparisons; no tuning or winner selection. '
                '*RobotCar uses public-pose diagnostics, not official test.', fontsize=9, color='#475569')
    figure.text(.105, .046, 'Sample SD describes three training seeds, not uncertainty across independent places. '
                'No confidence intervals or significance tests.', fontsize=9, color='#475569')
    figure.text(.105, .023, 'Panel (c) uses an independent Source stream and bank; '
                'it is a different Source protocol from panels (a–b).', fontsize=9, color='#475569')
    outputs = []
    for extension in ('png', 'svg'):
        path = directory / f'external_night_top1_comparisons.{extension}'
        metadata = {'Description': 'Fixed external night evaluation; no significance or confidence interval claims'}
        if extension == 'svg':
            metadata['Date'] = None
        figure.savefig(path, facecolor='white', metadata=metadata)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path,
                        default=ROOT / 'runs/revised_external_night_20261025/summary.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'figures')
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve(strict=True)
    summary = load_complete_summary(summary_path)
    data = plot_data(summary)
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    outputs = render(data, directory)
    import matplotlib
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'summary': str(summary_path), 'summary_sha256': sha256(summary_path),
        'plot_helper_sha256': sha256(Path(__file__).resolve()),
        'matplotlib_version': matplotlib.__version__,
        'metric': 'Top-1 recall difference in percentage points',
        'three_seed_spread': 'sample SD, not a confidence interval',
        'inference': 'None; plotting reads existing completed retrieval results only',
        'data': data,
        'figures': {str(path): sha256(path) for path in outputs},
        'baseline_audit_warnings': summary.get('baseline_audit_warnings', []),
    }
    (directory / 'plot_data.json').write_text(json.dumps(provenance, indent=2) + '\n')
    notes = [
        '# External night comparison figures', '',
        'Figures show every fixed model on the four frozen protocols. The helper requires all 52 results '
        'and verifies their file hashes before plotting.', '',
        '- Panel (a): B minus A separately for seeds 42, 43, and 44; black ticks show their mean.',
        '- Panel (b): A/B/C minus the original Day baseline, mean ± sample SD over three training seeds. '
        'Faint dots show the individual seeds. These are not confidence intervals.',
        '- Panel (c): G2 minus G1 within A and B, seed 42 only. Labels show net correct-query differences.',
        '',
        'Positive differences mean higher Top-1 recall. One correct query corresponds to 100/N percentage points; '
        'query counts differ substantially across protocols. Results do not establish statistical significance '
        'or independence across geographically correlated queries.', '',
        'H1 uses same-main-batch Source Night and the Qwen bank. H3 uses an independent Source stream and '
        'a separate Source-only bank; cross-axis differences do not isolate only pool size.', '',
        summary['limitations'], '', summary['reserved'], '',
        f'Input summary SHA256: `{provenance["summary_sha256"]}`. '
        'Exact plotted values and figure hashes are in `plot_data.json`.',
    ]
    (directory / 'README.md').write_text('\n'.join(notes) + '\n')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()

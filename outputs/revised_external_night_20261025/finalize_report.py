#!/usr/bin/env python3
"""Create an output-only Chinese report after all 52 frozen evaluations finish.

No model, image, dataset, or training module is imported. A partial queue, a
baseline audit mismatch, inconsistent rank bookkeeping, or a failed memory guard
causes a hard failure before either deliverable is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/revised_external_night_20261025'
BENCHMARKS = ('svox_test', 'tokyo247_night', 'robotcar_night', 'robotcar_night_rain')
LABELS = {'svox_test': 'SVOX test', 'tokyo247_night': 'Tokyo247 Night',
          'robotcar_night': 'RobotCar Night*', 'robotcar_night_rain': 'RobotCar Night-rain*'}
COUNTS = dict(zip(BENCHMARKS, (823, 105, 197, 224)))
SEEDS = (42, 43, 44)
MODELS = tuple(f'seed{seed}_{arm}' for seed in SEEDS for arm in 'ABC') + ('G1_A', 'G1_B', 'G2_A', 'G2_B')
METRICS = ('R@1', 'R@5', 'R@10')
EXPECTED_BASELINE = dict(zip(BENCHMARKS, ((763, 810, 816), (91, 95, 98),
                                       (179, 187, 190), (214, 216, 218))))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def relative(path):
    return Path(os.path.relpath(path, OUTPUT)).as_posix()


def metrics(evaluation, count):
    if evaluation['num_queries'] != count:
        raise ValueError('Wrong query denominator')
    errors = evaluation['error_ranks']
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank <= 1 for rank in errors.values()):
        raise ValueError('Invalid exact positive ranks')
    values = {}
    for k in (1, 5, 10):
        metric = f'R@{k}'
        hits = count - sum(rank > k for rank in errors.values())
        recall = evaluation['recall'][metric]
        if evaluation['hits'][metric] != hits or not math.isclose(recall, hits / count, abs_tol=1e-12):
            raise ValueError('Inconsistent exact ranks, hits, and Recall')
        values[metric] = {'hits': hits, 'recall': recall, 'percent': recall * 100}
    return values


def paired(before, after, count, recorded=None):
    result = {}
    for k in (1, 5, 10):
        metric = f'R@{k}'
        old = {qid for qid, rank in before['error_ranks'].items() if rank > k}
        new = {qid for qid, rank in after['error_ranks'].items() if rank > k}
        repaired, regressed = old - new, new - old
        delta = after['hits'][metric] - before['hits'][metric]
        if delta != len(repaired) - len(regressed):
            raise ValueError('Paired changes disagree with the Recall delta')
        if recorded is not None:
            expected = recorded[metric]
            if (set(expected['repaired']) != repaired or set(expected['regressed']) != regressed
                    or expected['repaired_count'] != len(repaired)
                    or expected['regressed_count'] != len(regressed)):
                raise ValueError('Saved paired query identities/counts are inconsistent')
        result[metric] = {'delta_hits': delta, 'delta_pp': 100 * delta / count,
                          'repaired_count': len(repaired), 'regressed_count': len(regressed)}
    return result


def validate_memory(status):
    peaks, available = [], []
    for name, job in status['jobs'].items():
        memory = job.get('memory_guard', {})
        events = memory.get('memory_events', {})
        if (job.get('state') != 'complete' or job.get('returncode') != 0
                or memory.get('event') != 'finish' or memory.get('returncode') != 0
                or memory.get('termination_reason') is not None
                or int(events.get('oom', 0)) or int(events.get('oom_kill', 0))
                or int(events.get('oom_group_kill', 0))):
            raise ValueError(f'Unsuccessful/OOM evaluation memory guard: {name}')
        peak = max(memory.get('peak_cgroup_bytes_sampled') or 0,
                   memory.get('peak_cgroup_bytes_kernel') or 0)
        if peak <= 0:
            raise ValueError(f'Missing measured RAM peak: {name}')
        peaks.append(peak)
        if memory.get('minimum_system_available_bytes') is not None:
            available.append(memory['minimum_system_available_bytes'])
    return {'num_successful_guarded_jobs': len(peaks), 'oom_events': 0,
            'peak_cgroup_bytes': max(peaks), 'peak_cgroup_gib': max(peaks) / 2**30,
            'minimum_system_available_bytes': min(available) if available else None,
            'minimum_system_available_gib': min(available) / 2**30 if available else None,
            'queue_elapsed_seconds': status.get('elapsed_seconds'),
            'memory_limit_gib': 7, 'memory_high_gib': 6, 'minimum_reserve_gib': 2,
            'reference_device_cache_mib': 4096,
            'note': 'cgroup peak measures host RAM of each evaluation subprocess, not GPU VRAM; device reference cache uses GPU'}


def complete_inputs(run):
    summary = json.loads((run / 'summary.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    plan = json.loads((run / 'experiment_plan.json').read_text())
    if (summary.get('state') != 'complete' or status.get('state') != 'complete'
            or summary.get('completed_jobs') != 52 or summary.get('planned_jobs') != 52
            or len(status['jobs']) != 52 or len(plan['commands']) != 52):
        raise ValueError('Refusing partial results: require completed frozen queue with exactly 52/52 jobs')
    if summary.get('baseline_audit_warnings'):
        raise ValueError('Historical baseline audit warning requires review before report publication')
    if set(summary['benchmarks']) != set(BENCHMARKS) or set(plan['models']) != set(MODELS):
        raise ValueError('Wrong predeclared model or benchmark set')
    protocol = plan['protocol']
    if (protocol['image_size'] != [224, 224] or protocol['precision'] != 'fp32'
            or protocol['query_fusion'] is not None or protocol['test_time_augmentation']
            or protocol['training']):
        raise ValueError('Frozen inference protocol differs from the declared comparison')
    resources = validate_memory(status)
    datasets, result_hashes = {}, {}
    for name in BENCHMARKS:
        dataset = summary['benchmarks'][name]
        count = COUNTS[name]
        if (dataset['num_queries'] != count or set(dataset['models']) != set(MODELS)
                or not dataset.get('historical_baseline_matches')):
            raise ValueError(f'{name}: wrong count/model set or baseline audit')
        day = metrics(dataset['baseline'], count)
        if tuple(day[metric]['hits'] for metric in METRICS) != EXPECTED_BASELINE[name]:
            raise ValueError(f'{name}: historical baseline audit mismatch')
        model_metrics = {}
        first = next(iter(dataset['models'].values()))
        num_references = None
        for model in MODELS:
            record = dataset['models'][model]
            path = Path(record['path'])
            if not path.is_absolute():
                path = ROOT / path
            if digest(path) != record['result_sha256']:
                raise ValueError(f'{name}/{model}: raw result has changed')
            result_hashes[str(path)] = record['result_sha256']
            job = status['jobs'][f'{model}.{name}']
            if job['evaluation'] != record:
                raise ValueError(f'{name}/{model}: status and summary disagree')
            for key in ('baseline', 'checkpoint_sha256', 'gallery_sha256', 'query_sha256',
                        'preprocessing', 'query_preprocessing', 'protocol'):
                if record[key] != first[key]:
                    raise ValueError(f'{name}/{model}: shared evaluation identity differs in {key}')
            if record['baseline'] != dataset['baseline']:
                raise ValueError('Original Day baseline differs across models')
            raw = json.loads(path.read_text())
            if 'fusion' in raw or raw['query_preprocessing']['mode'] != 'native_single_scale':
                raise ValueError('Unexpected fusion or non-native query processing')
            num_references = raw['baseline']['num_references']
            model_metrics[model] = {'metrics': metrics(record['night'], count),
                                    'adapter_sha256': plan['models'][model]['adapter_sha256'],
                                    'metadata_sha256': plan['models'][model]['metadata_sha256'],
                                    'result_path': str(path), 'result_sha256': record['result_sha256']}
        arms = {}
        for arm in 'ABC':
            arms[arm] = {}
            for metric in METRICS:
                recalls = [model_metrics[f'seed{seed}_{arm}']['metrics'][metric]['recall'] for seed in SEEDS]
                hits = [model_metrics[f'seed{seed}_{arm}']['metrics'][metric]['hits'] for seed in SEEDS]
                recorded = dataset['P0_three_seed'][arm]['metrics'][metric]
                if (recorded['hits'] != hits or not math.isclose(recorded['mean'], statistics.mean(recalls), abs_tol=1e-12)
                        or not math.isclose(recorded['sample_std'], statistics.stdev(recalls), abs_tol=1e-12)):
                    raise ValueError('Saved three-seed statistics are inconsistent')
                arms[arm][metric] = {'seeds': list(SEEDS), 'hits': hits, 'mean': statistics.mean(recalls),
                                     'sample_std': statistics.stdev(recalls),
                                     'mean_percent': 100 * statistics.mean(recalls),
                                     'sample_std_pp': 100 * statistics.stdev(recalls),
                                     'mean_hits': statistics.mean(hits),
                                     'mean_delta_vs_day_pp': 100 * (statistics.mean(recalls) - day[metric]['recall']),
                                     'mean_delta_vs_day_hits': statistics.mean(hits) - day[metric]['hits']}
        pairs = {}
        for seed in SEEDS:
            for before_arm in ('A', 'C'):
                before, after = f'seed{seed}_{before_arm}', f'seed{seed}_B'
                label = f'{after}_vs_{before}'
                pairs[label] = paired(dataset['models'][before]['night'], dataset['models'][after]['night'],
                                      count, dataset['paired_changes'][label])
        scale = {}
        for before, after in (('G1_A', 'G2_A'), ('G1_B', 'G2_B'), ('G1_A', 'G1_B'), ('G2_A', 'G2_B')):
            label = f'{after}_vs_{before}'
            scale[label] = paired(dataset['models'][before]['night'], dataset['models'][after]['night'],
                                  count, dataset['paired_changes'][label])
        contrast_means = {}
        for before_arm in ('A', 'C'):
            label = f'B_minus_{before_arm}'
            contrast_means[label] = {}
            for metric in METRICS:
                values = [pairs[f'seed{seed}_B_vs_seed{seed}_{before_arm}'][metric]['delta_pp'] for seed in SEEDS]
                delta_hits = [pairs[f'seed{seed}_B_vs_seed{seed}_{before_arm}'][metric]['delta_hits'] for seed in SEEDS]
                contrast_means[label][metric] = {'seed_delta_hits': delta_hits,
                                                 'mean_delta_hits': statistics.mean(delta_hits),
                                                 'mean_delta_pp': statistics.mean(values),
                                                 'sample_std_delta_pp': statistics.stdev(values)}
        datasets[name] = {'label': LABELS[name], 'num_queries': count, 'num_references': num_references,
                          'baseline': day, 'models': model_metrics, 'P0_three_seed': arms,
                          'per_seed_paired_topk': pairs, 'P0_contrast_means': contrast_means,
                          'Source_scale_paired_topk': scale, 'protocol': first['protocol'],
                          'gallery_sha256': first['gallery_sha256'], 'query_sha256': first['query_sha256']}
    return summary, status, plan, resources, datasets, result_hashes


def signed(value, digits=2):
    return f'{value:+.{digits}f}'


def pair_cell(pair):
    return f"{pair['delta_hits']:+d}（{pair['repaired_count']}/{pair['regressed_count']}）"


def report_text(result, run, zip_path):
    datasets, resources = result['benchmarks'], result['resources']
    lines = ['# 既定模型的外部夜间检索评测', '',
             '**52/52 次新评测已完成：13 个固定模型 × 4 个夜间协议。** '
             '本轮只执行推理，训练权重没有更新；所有模型均使用 FP32、原生 224 输入、原始 Day checkpoint 的固定图库，未使用 TTA 或 Teacher/Student 融合。', '',
             'A 为无额外辅助，B 增加真实配对辅助，C 增加合成配对辅助。P0 使用 seed 42/43/44；'
             'G1/G2 使用 seed 42 的独立 Source 流和独立 Source bank，保持训练步数及 Source 曝光数一致。'
             '13 个 checkpoint 均完成 1 个 epoch、908 次更新，主分支 7263 个视图、Source 1816 个视图；'
             'B/C 各自另有 1816 个真实/合成辅助视图。'
             'P0 与 G1/G2 的 Source 采样及候选 bank 不同，跨轴比较不能单独归因于池规模。', '',
             '## 结果判断', '']
    for name in BENCHMARKS:
        data = datasets[name]
        b, a = data['P0_three_seed']['B']['R@1'], data['P0_three_seed']['A']['R@1']
        ba = data['P0_contrast_means']['B_minus_A']['R@1']
        bc = data['P0_contrast_means']['B_minus_C']['R@1']
        sequence = '/'.join(f'{delta:+d}' for delta in ba['seed_delta_hits'])
        lines.append(f"- **{LABELS[name]}**：B 的平均 Top-1 为 {b['mean_percent']:.2f}% ± {b['sample_std_pp']:.2f} pp，"
                     f"较原始 Day {signed(b['mean_delta_vs_day_pp'])} pp，较 A {signed(ba['mean_delta_pp'])} pp；"
                     f"三个 seed 的 B−A 命中变化为 {sequence}，B−C 平均 {signed(bc['mean_delta_pp'])} pp。")
    wins = [name for name in BENCHMARKS if datasets[name]['P0_three_seed']['B']['R@1']['mean_delta_vs_day_pp'] > 1e-12]
    equal = [name for name in BENCHMARKS if abs(datasets[name]['P0_three_seed']['B']['R@1']['mean_delta_vs_day_pp']) <= 1e-12]
    losses = [name for name in BENCHMARKS if datasets[name]['P0_three_seed']['B']['R@1']['mean_delta_vs_day_pp'] < -1e-12]
    lines += ['', f"B 相对原始 Day 的平均 Top-1 在 {len(wins)}/4 个协议提高、{len(equal)}/4 持平、{len(losses)}/4 降低。"
              'B−A 用于衡量加真实辅助后的整体变化；B−C 还涉及真实/合成数据源差异，不能证明“真实配对关系”本身的因果贡献。'
              '每个 seed 只有一次训练；均值和样本标准差描述这三个 checkpoint 的离散程度，没有显著性结论。', '']
    source_parts = []
    for name in BENCHMARKS:
        scales = datasets[name]['Source_scale_paired_topk']
        a = scales['G2_A_vs_G1_A']['R@1']['delta_hits']
        b = scales['G2_B_vs_G1_B']['R@1']['delta_hits']
        source_parts.append(f"{LABELS[name]}：A {a:+d}、B {b:+d}")
    lines += ['扩大 Source 池的 Top-1 命中变化（G2−G1）为：' + '；'.join(source_parts) + '。'
              '这些是同一 seed 的固定预算比较，尚不足以给出多 seed 的规模规律。', '',
              '这轮外部结果用于检验预先固定的方案。默认模型保持由开发集流程决定；'
              '后续权重、供体或超参数改动应在开发集完成，再使用另设的未使用验证集检查，避免把本轮外部分数用于反复选择。', '',
              '![固定模型在四个夜间协议上的 Top-1 对比](figures/external_night_top1_comparisons.png)', '',
              '图中依次展示 B−A 的逐 seed 变化、A/B/C 相对原始 Day 的变化，以及 G2−G1 的变化。'
              '纵轴为 Top-1 百分点，误差线仅表示三个 seed 的样本标准差；图用于区分辅助收益、原始模型退化和 Source 规模变化。', '',
              '## 全部模型结果', '',
              '下表每格为 **R@1 / R@5 / R@10 的命中查询数**，各列使用固定查询总数。', '',
              '| 模型 | ' + ' | '.join(f'{LABELS[name]}（{COUNTS[name]}）' for name in BENCHMARKS) + ' |',
              '|---|' + '|'.join('---:' for _ in BENCHMARKS) + '|']
    for model in ('Original Day', *MODELS):
        cells = []
        for name in BENCHMARKS:
            data = datasets[name]
            values = data['baseline'] if model == 'Original Day' else data['models'][model]['metrics']
            cells.append(' / '.join(str(values[metric]['hits']) for metric in METRICS))
        lines.append('| ' + model + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## P0 三 seed 均值与波动', '',
              '均值为百分比；± 后为样本标准差，单位为百分点（pp），n=3。', '',
              '| 协议 | 组别 | R@1 | R@5 | R@10 |', '|---|---|---:|---:|---:|']
    for name in BENCHMARKS:
        for arm in 'ABC':
            values = datasets[name]['P0_three_seed'][arm]
            cells = [f"{values[metric]['mean_percent']:.2f} ± {values[metric]['sample_std_pp']:.2f}" for metric in METRICS]
            lines.append(f'| {LABELS[name]} | {arm} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'B 相对原始 Day 的三个 seed 平均差值（pp）：', '',
              '| 协议 | R@1 | R@5 | R@10 |', '|---|---:|---:|---:|']
    for name in BENCHMARKS:
        b = datasets[name]['P0_three_seed']['B']
        lines.append(f'| {LABELS[name]} | ' + ' | '.join(signed(b[metric]['mean_delta_vs_day_pp']) for metric in METRICS) + ' |')
    lines += ['', '## P0 逐查询修复与退化', '',
              '单元格为 **命中数净变化（修复数/退化数）**，均使用同一批查询配对计算；顺序为 seed 42/43/44。'
              '完整查询 ID 和 positive rank 见原始队列汇总。', '',
              '| 协议 | 对比 | R@1 | R@5 | R@10 |', '|---|---|---:|---:|---:|']
    for name in BENCHMARKS:
        pairs = datasets[name]['per_seed_paired_topk']
        for seed in SEEDS:
            for before in ('A', 'C'):
                pair = pairs[f'seed{seed}_B_vs_seed{seed}_{before}']
                lines.append(f'| {LABELS[name]} | seed {seed}，B−{before} | ' +
                             ' | '.join(pair_cell(pair[metric]) for metric in METRICS) + ' |')
    lines += ['', '## 独立 Source 流的规模与辅助对比', '',
              'G1/G2 只比较同一独立 Source 流轴；均为 seed 42。单元格仍为净变化（修复/退化）。', '',
              '| 协议 | 对比 | R@1 | R@5 | R@10 |', '|---|---|---:|---:|---:|']
    labels = {'G2_A_vs_G1_A': 'A：G2−G1', 'G2_B_vs_G1_B': 'B：G2−G1',
              'G1_B_vs_G1_A': 'G1：B−A', 'G2_B_vs_G2_A': 'G2：B−A'}
    for name in BENCHMARKS:
        for label, pair in datasets[name]['Source_scale_paired_topk'].items():
            lines.append(f'| {LABELS[name]} | {labels[label]} | ' +
                         ' | '.join(pair_cell(pair[metric]) for metric in METRICS) + ' |')
    elapsed = resources['queue_elapsed_seconds']
    elapsed_text = f"{elapsed / 60:.1f} 分钟" if elapsed is not None else '见队列状态'
    lines += ['', '## 协议、资源与边界', '',
              f"52 个受内存限制的独立评测进程全部正常结束，无 OOM。单进程 cgroup 主机内存峰值 "
              f"{resources['peak_cgroup_gib']:.2f} GiB；每进程上限 7 GiB，memory.high 为 6 GiB，系统预留 2 GiB。"
              f"队列总耗时 {elapsed_text}，包含反复验证输入字节和图库的时间。4 GiB retrieval cache 位于 GPU；"
              'cgroup 峰值不表示显存占用。', '',
              '新增队列的 8 项专项单元检查已通过，覆盖固定命令、完整 TopK 错误、逐查询配对、三 seed 统计、'
              '原生输入约束和冻结输入修改检测。所有结果均由本轮重新计算；历史指标仅用于核对原始 Day 基线，未替代任何一次评测。', '',
              '- SVOX 使用原生 test 夜间 823 查询及 25 m UTM 正样本。',
              '- Tokyo247 使用官方 NetVLAD MAT 的完整图库与 CSV 夜间标签，共 105 查询。当地查询文件来自 v3 发布；'
              '尚未独立逐字节确认其与早期 v2 查询图像完全一致，此限制保留。',
              '- *RobotCar Night / Night-rain 使用公开 train pose 的 rear-camera 查询及 49 个对齐模型的完整 overcast-reference 图库，'
              '按 25 m 三维位置定义正样本。它们是仅用于评测的位置检索诊断，不是官方 RobotCar withheld test，'
              '也未加入朝向或可见性筛选。',
              '- 这四套协议过去已被查看，不能称为全新的盲测。SF-XL 仍保留，未计算模型描述符或检索结果；'
              'Aachen 缺少本地可用的夜间检索正样本真值，官方 RobotCar test pose 也未公开，均未据图像相似度臆造真值。', '',
              '## 复现与审计', '',
              f"- [机器可读精简结果](results_summary.json)、[原始完整队列汇总]({relative(run / 'summary.json')})、"
              f"[52 次评测状态及内存记录]({relative(run / 'status.json')})。",
              f"- [冻结评测计划与所有输入哈希]({relative(run / 'experiment_plan.json')})、"
              '[13 个模型的权重及训练预算审计](model_audit.md)、[模型审计 JSON](model_audit.json)。',
              '- [数据与真值预检](../../runs/revised_external_night_preflight_20261025/)、'
              '[原始方案的数据审计](../revised_plan_20261025/darkdriving_source_audit.json)。',
              '- [P0 训练汇总](../../runs/revised_stage1_multiseed_20261025/summary.json)、'
              '[Source 规模训练汇总](../../runs/revised_gsv_scale_20261025/summary.json)。',
              '- [冻结队列实现](../../scripts/run_revised_night_evaluation.py)、'
              '[8 项专项检查](../../tests/test_revised_night_evaluation.py)、'
              '[专项检查执行日志](../../runs/revised_external_night_units.console.log)、'
              '[图表原始数据与出处](figures/plot_data.json)。']
    if zip_path is not None:
        lines.append(f'- [冻结代码副本]({relative(zip_path)})。')
    else:
        lines.append('- 冻结代码逐文件哈希见评测计划；代码压缩副本路径待附。')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--code-zip', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary, status, plan, resources, datasets, result_hashes = complete_inputs(run)
    zip_path = args.code_zip.resolve() if args.code_zip is not None else None
    if zip_path is None:
        candidates = sorted({path for directory in (OUTPUT, run) for path in directory.glob('*.zip')
                             if any(word in path.name.lower() for word in ('code', 'frozen', 'source'))})
        if len(candidates) == 1:
            zip_path = candidates[0]
    if zip_path is not None and not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    figure = OUTPUT / 'figures/external_night_top1_comparisons.png'
    if not figure.is_file():
        raise ValueError('Render the complete comparison figure before finalizing the report')
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'state': 'complete',
        'completed_fresh_evaluations': 52, 'num_models': 13, 'seeds': list(SEEDS),
        'protocol': plan['protocol'], 'intent': plan['intent'], 'resources': resources,
        'unit_checks': {'focused_cases_passed': 8,
                        'source': str(ROOT / 'tests/test_revised_night_evaluation.py'),
                        'execution_log': str(ROOT / 'runs/revised_external_night_units.console.log')},
        'benchmarks': datasets,
        'limits': ['Exploratory, historically inspected external protocols; no significance or blind-test claim',
                   'RobotCar public train-pose retrieval diagnostics, not official withheld test',
                   'Tokyo local v3 query bytes not independently verified against v2',
                   'SF-XL remains reserved; Aachen and official RobotCar test lack local public positives',
                   'P0 and Source-scale axes use different Source sampling/banks',
                   'Real versus synthetic auxiliary does not isolate pairing causality'],
        'provenance': {
            'queue_summary': {'path': str(run / 'summary.json'), 'sha256': digest(run / 'summary.json')},
            'queue_status': {'path': str(run / 'status.json'), 'sha256': digest(run / 'status.json')},
            'frozen_plan': {'path': str(run / 'experiment_plan.json'), 'sha256': digest(run / 'experiment_plan.json')},
            'raw_result_sha256': result_hashes,
            'figure': {'path': str(figure), 'sha256': digest(figure)},
            'frozen_code_zip': {'path': str(zip_path), 'sha256': digest(zip_path)} if zip_path else None,
        },
    }
    write_json(OUTPUT / 'results_summary.json', result)
    report = OUTPUT / 'report.md'
    temporary = report.with_name(report.name + '.tmp')
    temporary.write_text(report_text(result, run, zip_path))
    os.replace(temporary, report)
    print(f'Wrote {report} and {OUTPUT / "results_summary.json"}')


if __name__ == '__main__':
    main()

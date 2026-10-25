# 其他夜间数据集评估（2026-10-24）

2026-10-25 补充：修订方案的 13 个固定模型已完成全部 **52 次**外部夜间评测，见 [新一轮完整报告](../outputs/revised_external_night_20261025/report.md) 和 [三 seed / 规模对比图](../outputs/revised_external_night_20261025/figures/external_night_top1_comparisons.png)。以下保留 2026-10-24 历史模型的原始记录，不将其混入本轮三 seed 统计。

新一轮真实配对 B 的三 seed 平均 Top-1：SVOX test 92.71%、Tokyo247 Night 84.76%、RobotCar public Night 90.86%、Night-rain 95.09%；相对原始 Day 分别为 0、−1.90、0、−0.45 个百分点，尚未呈现稳定超越原始模型的收益。独立 Source 池 G2−G1 的 B Top-1 净变化依次为 0、0、−1、0 张；扩大池的互补性仍未确立。

本轮使用原生 224、FP32、固定原始 Day Gallery，无融合或 TTA，没有更新权重。可用 `python scripts/run_revised_night_evaluation.py --output runs/new_external_replay` 在全新目录复现；`--plan-only` 仅冻结计划。队列逐项经内存保护运行，记录权重、查询、真值、图库和代码哈希。RobotCar 仍为公开位姿检索诊断；Aachen / RobotCar 官方 test 缺少本地可信正样本真值，SF-XL 继续保留。

## 固定模型与评估设置

本轮评估已有 Night 专用全参数 DINOv2，没有训练、微调或根据新测试分数调整配置。

本文保留历史全部参数微调模型 `source_night` 的评估结果。当前默认改为独立 Night 主干的后四层微调，详见 [SALAD 训练设置](SALAD_TRAINING.md)；此次默认值调整不改写下表历史数值。

- Day：`weights/dino_salad.ckpt`，SHA-256 `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada`。
- Night：`runs/night_specialist_20261024/source_night/adapter.safetensors`，SHA-256 `1ee611a2c2aad86baa330e07751a0643c6c53599bb09cd1b1a355ce4b0ef0617`。
- Day/Night 均用原始 RGB 图、224×224 PIL bilinear、ImageNet normalization，无 TTA。
- 每个数据集仅提取一次原始 Day Gallery；两模型使用完全相同的 Gallery。RobotCar 两天气还复用同一个 Gallery 缓存。
- 描述符与精确检索距离为 FP32；计算完整 Gallery 中首个正样本的真实排名，报告 R@1/5/10。
- 所有下表 Query 都有正样本；没有丢弃困难 Query。

固定权重与协议选择记录：`runs/night_external_20261024/selection.json`。三次结果的 Night 权重哈希一致，RobotCar 两组 Gallery 图像字节哈希一致。

## 实测结果

单位为百分比；括号中的 Query 数为 Recall 分母。

| 夜间集合 | Gallery / Query | 原始 Day R@1 / R@5 / R@10 | Night R@1 / R@5 / R@10 | ΔR@1（百分点） |
| --- | ---: | --- | --- | ---: |
| Tokyo247，官方查询中的夜间子集 | 75,984 / 105 | 86.67 / 90.48 / 93.33 | 83.81 / 90.48 / 94.29 | −2.86 |
| RobotCar v2 public train，Night rear，诊断 | 20,862 / 197 | 90.86 / 94.92 / 96.45 | 91.88 / 95.94 / 96.45 | +1.02 |
| RobotCar v2 public train，Night-rain rear，诊断 | 20,862 / 224 | 95.54 / 96.43 / 97.32 | 95.98 / 96.43 / 97.77 | +0.45 |

**结果尚未显示一致的跨数据集收益。** Tokyo Top-1 减少 3 张；RobotCar Night 增加 2 张，Night-rain 增加 1 张。不能由这轮小样本变化宣称稳定提升，也不能把 RobotCar 的自定义检索指标与官方定位成功率直接比较。

| 集合 | Day 命中数 R@1 / R@5 / R@10 | Night 命中数 R@1 / R@5 / R@10 | Top-1 修复 / 新增错误 |
| --- | --- | --- | --- |
| Tokyo247 Night | 91 / 95 / 98 | 88 / 95 / 99 | 0 / 3 |
| RobotCar public Night | 179 / 187 / 190 | 181 / 189 / 190 | 5 / 3 |
| RobotCar public Night-rain | 214 / 216 / 218 | 215 / 216 / 219 | 1 / 0 |

Tokyo 的三个新增错误为 `00018.jpg`、`00159.jpg`、`00693.jpg`，正样本排名均由 1 降到 2。完整逐 Query 排名、错误检索路径和目标正样本保存在对应 JSON 中。

既有 SVOX test 的 92.71→93.07% R@1 结果见 [Night 专用实验](NIGHT_SPECIALIST_RESULTS.md)，本轮没有重跑 SVOX。

## 真值与数据版本

### Tokyo247

采用 [NetVLAD 官方发布页](https://www.di.ens.fr/willow/research/netvlad/) 的 `netvlad_v100_datasets.tar.gz` 中 `datasets/tokyo247.mat`，已随项目保存在 `configs/evaluation_sources/`，SHA-256 与来源记录一致。

- 使用完整 75,984 Gallery，保留 MAT 原始 ID 和顺序；MAT 的 `.jpg` 名称映射到本地对应 `.png` 文件，不重新编码图片。
- 只使用 MAT 指定的 315 Query 中 CSV 第 6 列（index 5）等于 `N` 的 105 张，不通过亮度猜夜间标签。
- Query/Gallery 坐标都来自官方 MAT。正样本为二维 UTM 距离 ≤25m，每个 Night Query 有 48–240 个正样本。
- 本地 Query 目录有 1,125 张扩展版 v3 图。这里遵循官方 MAT 的子集和坐标，但尚未逐字节核验本地 v3 图像与原 v2 发布图像完全一致；因此该结果带有这一数据版本限制。

### RobotCar：公开位姿诊断，不是官方 test

[官方 v2 README](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md) 提供约半数地点的 train 位姿，test 列表没有位姿。[原始 README](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar-Seasons.md) 明确全部 49 个 aligned 模型使用同一个世界坐标系。

本轮仅在 **public train 图像上进行前向评估**，沿用已经训练好的 Night 权重。自定义协议固定为：

- Gallery：全部 49 模型注册的 20,862 张 overcast-reference 图，包含 left/rear/right，相机和路径无重复、图像齐全。
- Query：公开位姿列表中的 rear 图，Night 197 张，Night-rain 224 张。
- Query 位姿为 camera-to-world，中心取矩阵最后列；COLMAP 是 world-to-camera，中心取 `−Rᵀt`。
- 以三维世界坐标中心距离 ≤25m 定义正样本，没有方向、可见性或相机侧别过滤。Night 每图有 147–314 个正样本，Night-rain 有 147–328 个。
- 官方定位指标需要位姿估计及位置/旋转误差评价；本轮只计算上述位置检索 Recall，不提供 6DoF 定位结果。

本地 train 元数据与官方文件字节一致，SHA-256 为 `842cb03eccf2b10b7ba0a2fe431c1f063f4a1c4459f042b536c9fb39a0315744`。49 个 `images.txt` 的逐文件哈希及组合哈希也在结果协议中保存。

### 尚不能计算可靠 Recall 的官方 test

| 集合 | 本地 Query | 缺少内容 |
| --- | ---: | --- |
| Aachen-Day-Night v1.1 Night | 官方清单 191，图像齐全 | Query pose / 可见性 / Query→Gallery positives |
| RobotCar v2 official test Night rear | 226，图像齐全 | Query pose / Query→Gallery positives |
| RobotCar v2 official test Night-rain rear | 203，图像齐全 | Query pose / Query→Gallery positives |

[Aachen 官方 v1.1 README](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/README_Aachen-Day-Night_v1_1.md) 明确 Query 参考位姿不公开；本地 NVM/COLMAP 的 6,697 个相机仅包含参考图，没有夜间 Query。内参清单不能代替位姿或检索真值。[GLACE 作者仓库](https://github.com/cvg/glace#cambridge-landmarks--aachen-day-night) 的 identity dummy poses 也是占位数据，不能用于 Recall。

这几项不能靠下载更多图像/权重补齐检索真值。需要可信的 Query→Gallery positives；补齐后可使用 `--dataset manifest --root <images> --manifest <evaluation_manifest.json>`。不能把模型检索结果自身当真值，也不能把同一个 RobotCar 粗地点分组的全部图直接当正样本。

本地 Nordland 使用帧对应协议，未提供夜间查询标签，本轮没有把它计为夜间测试。

## 复现

从项目根目录执行；新环境先安装兼容的 PyTorch，再安装 `requirements-extra.txt`（新增 SciPy）。以下命令按 32 GB 主机配置，**逐条串行执行**，使用新的复现目录保留历史结果。训练或其它评估运行时不要同时启动这些命令。

`run_bounded.py` 为每项任务设置 `8 GiB` cgroup RAM 硬上限、禁止 swap，并监控系统可用内存；不足时会拒绝启动或结束该任务。它需要 Linux cgroup v2 与 systemd user scope，不应绕过限制重试。该上限涵盖实验进程及其后代，不能控制浏览器等其它程序的内存；详细约束见 [32 GB 内存设置](SALAD_TRAINING.md)。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export REPLAY_ROOT=runs/night_external_replay_02

# 先检查图像和真值，不加载模型
"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/tokyo_check.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_night_benchmarks.py --dataset tokyo247 --check-data
"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/robotcar_check.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_night_benchmarks.py --dataset robotcar-public \
  --condition night --check-data

"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/tokyo247.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_night_benchmarks.py --dataset tokyo247 \
  --adapter-dir runs/night_specialist_20261024/source_night \
  --device cuda --retrieval-device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/tokyo247_day_gallery.safetensors \
  --reference-cache-mib 4096 \
  --output "$REPLAY_ROOT/tokyo247_night.json"

"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/robotcar_night.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_night_benchmarks.py --dataset robotcar-public --condition night \
  --adapter-dir runs/night_specialist_20261024/source_night \
  --device cuda --retrieval-device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/robotcar-public_day_gallery.safetensors \
  --reference-cache-mib 4096 \
  --output "$REPLAY_ROOT/robotcar_public_night.json"

"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/robotcar_night_rain.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_night_benchmarks.py --dataset robotcar-public --condition night-rain \
  --adapter-dir runs/night_specialist_20261024/source_night \
  --device cuda --retrieval-device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/robotcar-public_day_gallery.safetensors \
  --reference-cache-mib 4096 \
  --output "$REPLAY_ROOT/robotcar_public_night_rain.json"
```

上述显式缓存为 `runs/cache/tokyo247_day_gallery.safetensors` 和 `runs/cache/robotcar-public_day_gallery.safetensors`，绑定完整 Day checkpoint、预处理、有序 Gallery ID/图像字节及缓存张量哈希。更换 Night 权重可以复用原始 Day 缓存。减小推理 batch 和数据加载 worker 数不改变检索协议；复现结果另存新目录，不改写历史表格。

结果目录 `runs/night_external_20261024/` 中：

- `results_summary.json`：三组指标、整数命中数、协议及缺失真值状态。
- 三个单独结果 JSON：模型元数据、图像/权重哈希、精确逐 Query 排名和错误列表。
- `selection.json`：本轮固定评估对象与设置。
- `tests.log`：141 项项目单元测试通过；新增测试覆盖夜间标签列、官方查询子集、坐标阈值、位姿方向、跨模型共享 Gallery、缓存复用及无正样本 Query。

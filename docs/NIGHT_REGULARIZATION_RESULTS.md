# Night 正则、训练范围与融合实验结果（2026-10-24）

**本轮没有取得新的 Recall 突破。** 弱 Day anchor、仅前六层或后六层微调均未超过已有 `source_night` 的开发集结果；开发集选出的 Query 融合在 SVOX test 和 Tokyo 上损失 Top-1，未显示一致收益。因此保留这些能力供消融，不用本轮候选或融合设置替换默认推理方案。

目标仍为 Night Query 检索原始 Day Gallery。Gallery 使用冻结的原始 Day DINOv2 + SALAD；Night 是独立 DINOv2，训练时 SALAD 参数冻结且允许输入梯度。真实夜图只用于开发选择和前向评估，没有进入优化器、训练描述符 bank、分布适应或测试时适应。方法、损失含义及 Claude 建议的审核见 [实验设计](NIGHT_REGULARIZATION.md)。

本轮为单 seed、小预算探索。此前已看过部分测试结果，不能称为完全盲测；以下区分开发选择和固定设置后的测试，全部结果均保留。

## 开发集：五个纯 Student 候选

SVOX train 的 702 张真实夜间 Query 仅作为只读开发集，所有方法共用原始 Day Gallery。表中百分比和整数命中数均按 R@1 / R@5 / R@10 排列。

| 编码器 / 候选 | Recall（%） | 命中数 / 702 |
| --- | --- | --- |
| 原始 Day，参考 | 88.46 / 95.16 / 96.01 | 621 / 668 / 674 |
| `prior_source`，已有模型 | 88.32 / 95.16 / 96.44 | 620 / 668 / 677 |
| `full_control`，全参数 | 87.89 / 94.87 / 95.87 | 617 / 666 / 673 |
| `full_day_anchor`，全参数 + 弱 Day 正则 | 87.89 / 94.87 / 95.87 | 617 / 666 / 673 |
| `early6`，仅前六层 | 87.04 / 94.87 / 96.44 | 611 / 666 / 677 |
| `late6`，仅后六层 | 87.89 / 94.87 / 96.30 | 617 / 666 / 676 |

四个新候选各从同一 Day 权重初始化，使用相同 7,263 训练配对、seed 42、microbatch 8、1 epoch / 908 次优化器更新；共同包含 Qwen 配对分支、由训练 Day 原片模拟的 Source Night 分支和训练 Day bank。四组的 layer decay 统一为 1。Day anchor 权重为 `0.03 → 0`，每 batch 最多取两张干净 Day 图；它的命中数与无 anchor 对照完全相同。这只表明当前强度和预算下没有检索收益，不能据此断言所有 Day 正则都无效。已有 `prior_source` 使用过不同的 layer decay，不能把新旧差异全部归因于一项设置。

选择规则在开发评估前记录于 [selection_policy_before_development.json](../runs/night_regularization_20261024/selection_policy_before_development.json)：先在五个**纯 Student** 中按 R@1、R@5、R@10 的顺序选择，再仅为该 Student 从固定网格 `α ∈ {0, 0.25, 0.5, 0.75, 1}` 选择融合。完全同分时优先纯端点，再优先原始 Day 端点；候选间固定同分顺序见记录。没有跨候选选取测试表现最好的组合。

选中已有 `prior_source`；其开发集最佳融合为 `α=0.25`，即：

```text
q = normalize(0.25 × normalize(S(night)) + 0.75 × normalize(T(night)))
```

融合开发集为 **88.89 / 95.44 / 96.15%**，命中 **624 / 670 / 675**。虽然比原始 Day 多命中 3 张 Top-1，这个开发收益没有迁移到 SVOX test。混合权重（0 < α < 1）在部署时需要两次 Query 编码。

固定选择、元数据哈希和时间先写入 [selection_before_test.json](../runs/night_regularization_20261024/selection_before_test.json)，随后才运行本轮最终评估；各测试集均使用相同 `α=0.25`。关键 SHA-256：

| 对象 | SHA-256 |
| --- | --- |
| 原始 `weights/dino_salad.ckpt` | `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada` |
| 选中 Night `source_night/adapter.safetensors` | `1ee611a2c2aad86baa330e07751a0643c6c53599bb09cd1b1a355ce4b0ef0617` |
| 开发前选择规则文件 | `1ddd3bd56f13251668d3204efdf074c656731274b2403d1b69a614fb32afa4d1` |

## 固定设置后的四组评估

每组都使用同一原始 Day Gallery，固定 224×224 预处理，无多尺度 TTA；全部 Query 都有真值正样本。原始 Day、纯 Night、固定融合一起报告，百分比和命中数仍按 R@1 / R@5 / R@10 排列。

| 集合（Gallery / Query） | Query 编码 | Recall（%） | 整数命中数 |
| --- | --- | --- | --- |
| SVOX test（17,166 / 823） | 原始 Day | 92.71 / 98.42 / 99.15 | 763 / 810 / 816 |
| SVOX test | 纯 Night | 93.07 / 98.66 / 99.03 | 766 / 812 / 815 |
| SVOX test | 融合 α=0.25 | 92.47 / 98.42 / 99.15 | 761 / 810 / 816 |
| Tokyo247 Night（75,984 / 105） | 原始 Day | 86.67 / 90.48 / 93.33 | 91 / 95 / 98 |
| Tokyo247 Night | 纯 Night | 83.81 / 90.48 / 94.29 | 88 / 95 / 99 |
| Tokyo247 Night | 融合 α=0.25 | 85.71 / 90.48 / 92.38 | 90 / 95 / 97 |
| RobotCar public Night（20,862 / 197） | 原始 Day | 90.86 / 94.92 / 96.45 | 179 / 187 / 190 |
| RobotCar public Night | 纯 Night | 91.88 / 95.94 / 96.45 | 181 / 189 / 190 |
| RobotCar public Night | 融合 α=0.25 | 92.39 / 95.43 / 95.94 | 182 / 188 / 189 |
| RobotCar public Night-rain（20,862 / 224） | 原始 Day | 95.54 / 96.43 / 97.32 | 214 / 216 / 218 |
| RobotCar public Night-rain | 纯 Night | 95.98 / 96.43 / 97.77 | 215 / 216 / 219 |
| RobotCar public Night-rain | 融合 α=0.25 | 95.54 / 96.43 / 97.32 | 214 / 216 / 218 |

以下按相同 Query 与原始 Day 配对比较 Top-1。“修复”指 Day 错、该方法对；“新增错误”指 Day 对、该方法错。净增量等于两者之差。

| 集合 | 纯 Night 修复 / 新增错误 / 净增 | 融合修复 / 新增错误 / 净增 |
| --- | --- | --- |
| SVOX test | 11 / 8 / +3 | 3 / 5 / −2 |
| Tokyo247 Night | 0 / 3 / −3 | 0 / 1 / −1 |
| RobotCar public Night | 5 / 3 / +2 | 3 / 0 / +3 |
| RobotCar public Night-rain | 1 / 0 / +1 | 0 / 0 / 0 |

纯 Night 的 SVOX Top-1 小幅收益是已有模型的重测结果，本轮四组新训练并未进一步改进。融合在 RobotCar Night 上增加 3 张 Top-1，但 R@10 减少 1 张，同时在 SVOX 和 Tokyo 退化；不足以支持将融合设为默认。

协议限制延续 [外部夜间评估说明](EXTERNAL_NIGHT_RESULTS.md)：SVOX 与 Tokyo 使用 25 m 二维 UTM 正样本；Tokyo 遵循官方 MAT 的 315 Query 中 105 张夜间子集及完整 Gallery，但本地扩展 v3 图像尚未与原 v2 发布图逐字节核验。RobotCar 是 **public train 位姿上的只读自定义检索诊断**，不是 official test：使用 25 m 三维相机中心距离，Gallery 含全部相机侧别，无朝向或可见性过滤，不能与官方 6DoF 定位指标直接比较。Aachen 夜间 Query 缺可信检索真值，本轮不提供伪造的 Recall。

## 固定亮度分桶诊断

亮度仅为图像代理量：`RGB → bilinear 64×64 → L 均值 / 255`。只用 702 张 SVOX 开发 Query 的三分位数确定阈值 **0.2562212776 / 0.3316629647**，之后对各评估集固定应用；没有利用测试分数重新划桶。它受曝光、后处理和画面内容影响，不能解释为实测照度或“黄昏 / 路灯 / 极暗”的语义类别。

| 集合的较低均值灰度桶 | Query 数 | Day / 纯 Night / 融合 Top-1 命中数 |
| --- | ---: | --- |
| SVOX test | 223 | 202 / 202 / 201 |
| Tokyo247 Night | 82 | 71 / 69 / 70 |
| RobotCar public Night | 50 | 41 / 41 / 42 |
| RobotCar public Night-rain | 207 | 197 / 198 / 197 |

SVOX 较低亮度桶中，纯 Night 的净命中增量为 0；其整体 +3 来自中间桶 +2、较高桶 +1。Tokyo 较低桶反而减少 2 张 Top-1。因此尚无证据表明当前方法已解决较暗 Query 的主要瓶颈。Tokyo 较高桶只有 1 张，不能据其 100% 推断泛化能力。RobotCar Night-rain 的较高桶为空，Recall 记为不可计算，不能把零命中数当作 0% 或除零后的指标。完整分桶见 [brightness.json](../runs/night_regularization_20261024/brightness.json)。

## 32 GB 内存、训练完成性与检查

重跑全程串行，使用 `scripts/run_bounded.py` 设置 **8 GiB cgroup RAM 硬上限、6 GiB 回收阈值、swap 关闭**，worker 数为 0；启动时保留系统可用内存，运行中若低于 4 GiB 则结束实验。受控任务整体观测的最高 cgroup 内核峰值为 **5.43174 GiB**，系统可用内存最低为 **13.33299 GiB**，全部完成任务的 OOM / OOM kill 计数均为 0。cgroup 包含任务及其后代和所计费的文件缓存，这不是模型算法独占 RAM，也不能用于比较各训练范围的固有内存优势。

| 新训练候选 | 完成更新数 | 受控任务耗时 |
| --- | ---: | ---: |
| `full_control` | 908 | 329.21 s（5.49 min） |
| `full_day_anchor` | 908 | 342.22 s（5.70 min） |
| `early6` | 908 | 313.70 s（5.23 min） |
| `late6` | 908 | 301.19 s（5.02 min） |

这些时间来自受控任务日志，包含进程启动等开销，未把前向成本不同的四组称为相同计算量。此前未加限制的并行尝试曾发生 OOM，故障记录保留于 `interrupted_before_ram_limit/`，不计作成功训练或以上资源结论。

[control_checks.json](../runs/night_regularization_20261024/control_checks.json) 状态为 `passed`：四组步数、公共配置、主分支和 Source Night 逐步增强记录一致；Day anchor 首尾权重、冻结参数逐张量一致性、可训练层确实发生变化、张量有效性与任务正常完成均通过核验。[unit_tests.log](../runs/night_regularization_20261024/unit_tests.log) 记录 **158 项测试通过**，涵盖训练范围、冻结层输入梯度、权重日程、Query 融合和缓存等实现。

## 复现固定选择的评估

从项目根目录依次执行，使用新的输出目录。带 `--query-fusion-alpha 0.25` 的评估同时输出原始 Day、纯 Night 和融合结果，无需各自再加载模型重跑。不要与训练或其它模型进程并行运行。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ADAPTER=runs/night_specialist_20261024/source_night
export REPLAY_ROOT=runs/night_regularization_eval_replay_02

"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/svox_test.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_svox.py --split test \
  --adapter-dir "$ADAPTER" --query-fusion-alpha 0.25 \
  --device cuda --retrieval-device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/svox_test_gallery.safetensors --reference-cache-mib 1024 \
  --save-hard-cases --output "$REPLAY_ROOT/svox_test.json"

```

Tokyo 和 RobotCar 的入口与显式缓存参数见 [外部评估复现](EXTERNAL_NIGHT_RESULTS.md#复现)，复现本轮融合时增加 `--query-fusion-alpha 0.25` 并使用新的输出目录。不得为各数据集另选权重或融合系数。

可选：对本轮已完成的四份结果重新生成亮度诊断。它只逐图读图，不加载模型或描述符。

```bash
"$PYTHON" scripts/run_bounded.py --log "$REPLAY_ROOT/brightness.memory.jsonl" \
  -- "$PYTHON" scripts/analyze_night_brightness.py \
  --development runs/night_regularization_20261024/prior_source.svox_train.json \
  --results runs/night_regularization_20261024/svox_test.json \
    runs/night_regularization_20261024/tokyo247_night.json \
    runs/night_regularization_20261024/robotcar_public_night.json \
    runs/night_regularization_20261024/robotcar_public_night_rain.json \
  --output "$REPLAY_ROOT/brightness.json"
```

机器可读汇总为 [results_summary.json](../runs/night_regularization_20261024/results_summary.json)。原始结果及逐 Query 排名分别见 [SVOX test](../runs/night_regularization_20261024/svox_test.json)、[Tokyo247 Night](../runs/night_regularization_20261024/tokyo247_night.json)、[RobotCar Night](../runs/night_regularization_20261024/robotcar_public_night.json)、[RobotCar Night-rain](../runs/night_regularization_20261024/robotcar_public_night_rain.json)；每份最终结果另有 provenance 文件绑定选择记录、模型与结果哈希。

# Night 专用目标的对照实验（2026-10-24）

## 本轮回答的问题

Night Student 只负责夜间 Query→原始 Day Gallery 检索。Synthetic Guidance 和 Day Preservation 都是可选正则，不是跨模型检索的必要条件；即使某个正则有助于 Night，也应由 Night Recall 验证，而不以 Student 白天表现为目标。

本轮两组都关闭旧 Synthetic Guidance、Day Preservation/Restoration。第二组只增加从训练 Day 照片产生的强制低照度辅助视图。Student 不直接接收干净 Day 输入，干净 Day 只供冻结 Teacher 和固定 Day 特征库使用。方法及损失定义见 [Night 专用模型说明](NIGHT_SPECIALIST.md)。

## 实验控制

- 独立从原始 Day checkpoint 初始化完整 Night DINOv2；Day Teacher 和共享 SALAD 冻结。
- 同一 7,263 对 Day/Qwen 数据，1 epoch / 908 次更新，batch 8、BF16、AdamW、seed 42。
- Global / Contrast / Relation 权重均为 0.1；旧 Guidance、Day Preservation 均为 0。
- 固定 Day 库、64 个难负样本、逐层 LR 衰减和 warmup/cosine 日程相同。
- 两组逐步 Night 相机增强采样完全一致。
- `night_only`：辅助权重 0；`source_night`：辅助权重 0.5，每步最多 2 张低照度源图，共 1,816 张，全部执行强制曝光衰减。
- SGPD / Keep / OT / 幅度本轮关闭，原有可信 Qwen Patch 蒸馏代码和独立开关保持可用。
- 所有真实 Night 图像仅用于只读评估，没有恢复真实夜景微调。

两组并行运行于同一 GPU。记录的过程耗时约 160 秒和 213 秒，峰值 CUDA allocated 均约 2.21 GiB；这是进程观测值，不是隔离的速度比较。辅助分支增加计算，顺序反向避免叠加两条激活图。

## 开发集与辅助跨视角验证

SVOX train 为只读开发集：702 张 Night Query，22,232 张固定原始 Day Gallery，25 m UTM 正样本协议。

| 模型 | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 621/702，88.46% | 668/702，95.16% | 674/702，96.01% |
| 仅 Night 主分支 | 618/702，88.03% | 668/702，95.16% | 678/702，96.58% |
| 加入低照度源图辅助 | 620/702，88.32% | 668/702，95.16% | 677/702，96.44% |

低照度辅助比同设置对照多命中 2 张 R@1，但仍比原始 Day 少 1 张。历史 `restore_01` 开发集 R@1 为 626/702，高于本轮两组；因此本轮不能证明默认关闭两项旧正则必然提高 Night Recall。历史模型与本轮同时存在多个配置差异，也不能由此独立判定哪项旧正则有效。

固定相机退化、7,648 张 Day Gallery 的合成跨视角子集排除每条 Query 的精确白天源图，仅检查另一个同地点视角；共 90 张 Query / 45 个地点：

| 模型 | 跨视角 R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 68/90 | 82/90 | 84/90 |
| 仅 Night 主分支 | 67/90 | 83/90 | 84/90 |
| 加入低照度源图辅助 | 70/90 | 83/90 | 84/90 |

辅助分支的跨视角 R@1 有小幅改善，但子集存在选择偏差，不代表全部真实夜景。

## 所选新候选的最终 test

按本轮两个新候选的开发集 R@1 选择 `source_night`，测试前写入权重哈希和输入策略。历史模型只作参考，本轮并未将它称为被新模型全面超过。以下均使用原始 224×224 预处理、同一冻结 Day Gallery，无多尺度、无测试时增强。

| 模型 | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 763/823，92.71% | 810/823，98.42% | 816/823，99.15% |
| 新 Night 专用候选 | 766/823，93.07% | 812/823，98.66% | 815/823，99.03% |

R@1 修复 11 个原始错误，同时新增 8 个错误，净增加 3 张命中；R@5 净增 2 张，R@10 少 1 张。当前结果仅支持小幅 R@1/R@5 改善，不能称为全面、稳定突破。只有所选新候选测了 test，不能把 test 差值单独归因于辅助损失。

这仍是单 seed、短训练预算的探索性结果；历史实验已查看过 SVOX test，本轮没有据本次 test 继续调参。

## 产物与复现

输出根目录：`runs/night_specialist_20261024/`。

- `night_only/`、`source_night/`：完整 Night 权重、配置、训练日志、开发及跨视角验证。
- `source_night/svox_test.json`：最终 test 与错误详情。
- `experiment_plan.json`、`selection_before_test.json`、`control_checks.json`、`results_summary.json`：方案、选择记录、控制检查与机器可读汇总。
- `source_night_preview.jpg` 和 `.json`：干净 Day、Qwen 与源图低照度样例；只是低照度模拟，不生成真实夜间语义灯光。
- `unit_tests.log`：127 项测试通过。

所选权重 SHA-256：`1ee611a2c2aad86baa330e07751a0643c6c53599bb09cd1b1a355ce4b0ef0617`。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# 使用两个新的目录，均从原 Day 初始化
OUTPUT=runs/night_specialist_control_02 SOURCE_NIGHT_WEIGHT=0 \
  bash configs/night_specialist.sh
OUTPUT=runs/night_specialist_source_02 SOURCE_NIGHT_WEIGHT=0.5 \
  bash configs/night_specialist.sh

"$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir runs/night_specialist_source_02 --device cuda \
  --batch-size 16 --num-workers 2 --reference-cache-mib 768 \
  --output runs/night_specialist_source_02/svox_train.json
```

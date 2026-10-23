# 合成训练的检索指导实验（2026-10-23）

## 结论

本轮完成两组 Night DINOv2 全参数训练。`restore_01` 在 SVOX 开发集的 R@1 从原始 Day 的 621/702 提到 626/702，但最终 test 的 R@1 仍为 763/823，未实现 test R@1 突破。R@5、R@10 各多命中 1 张查询，不能据此声称稳定收益。

所有训练仅使用 7,263 对训练 Day/Qwen Night。真实夜图只用于检索评估，没有恢复真实夜景微调。所有结果共用原始 Day SALAD Gallery，Query 和 Gallery 都保持原来的 224×224 RGB/bilinear/ImageNet 预处理；未使用多尺度或测试时增强。

方法依据、论文与本项目协议的差别以及独立开关见 [方法说明](GUIDED_RETRIEVAL.md)。

## 两组候选

两组均从原始 Day checkpoint 独立初始化 Night，训练 1 epoch，3,632 个 microbatch / 908 次优化器更新，batch 2、累积 4、BF16、AdamW。最后 Block/Norm 起始 LR 为 1e-6，逐层倍率 0.8，32 次 warmup 后余弦衰减至 0.1 倍。全部 Night 参数可训练，Teacher/SALAD 冻结。

共同使用固定训练 Day 库的 7,263 个 key / 6,441 个地点，InfoNCE 保留同地点全部正样本、选 64 个难负样本，关系 KL 使用 Teacher 选出的 64 个异地点支持项。平均每 Query 有 1.2264 个正 key。SGPD、Keep、OT 和幅度项本轮均关闭；尚未验证它们在新方案上的增量。

| 设置 | `guided_01` | `restore_01` |
| --- | ---: | ---: |
| Global | 1 | 0.1 |
| Contrast | 0.1 | 0.1 |
| Relation KL | 0.1 | 0.1 |
| 干净 Qwen 冻结指导 | 0.1 | 1 |
| Day 保留/恢复 | 0.5，干净输入 | 0.5，相机退化输入 |

两组的 Qwen 相机增强采样逐步完全一致。第二组增加的 Day Student 相机视图使用独立 `:day` 种子，Teacher 和可信对应路径保持干净。由于同时修改了多个设置，这是候选方案比较，不能单独归因于某一项损失。

## SVOX 开发集：仅评估

官方 train 划分有 22,232 张 Day Gallery、702 张 Night Query，采用 25 m UTM 协议。本表用于候选比较，没有真实图像参数更新。

| 模型 | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 621/702，88.46% | 668/702，95.16% | 674/702，96.01% |
| 上一轮相机增强 Global | 612/702，87.18% | 666/702，94.87% | 675/702，96.15% |
| `guided_01` | 615/702，87.61% | 666/702，94.87% | 678/702，96.58% |
| `restore_01` | **626/702，89.17%** | 667/702，95.01% | 677/702，96.44% |

按开发集 R@1 选择 `restore_01`。输入策略、权重和 Gallery 在查看本轮 test 结果前冻结；未根据 test 重选模型或调参。选择 JSON 首次写入因嵌套字段读取错误失败，修正版在 test 启动后、读取结果前补写，记录中保留了这一时间说明。

## 最终 SVOX test

17,166 张原始 Day Gallery、823 张 Night Query，同样采用 25 m UTM 协议。

| 模型 | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 763/823，92.71% | 810/823，98.42% | 816/823，99.15% |
| `restore_01` | 763/823，92.71% | 811/823，98.54% | 817/823，99.27% |

Night 修复了 10 个原始 R@1 错误，同时新增 10 个 R@1 错误，净变化为零。这说明开发集的 R@1 收益没有稳定迁移到 test；不能将开发集的 +0.71 个百分点表述为最终真实夜景收益。

本轮只有单 seed、短训练预算，且历史实验已查看过 SVOX test，因此属于探索性结果。后续不应据本次 test 选择损失权重或增强范围。

## 合成验证与跨视角检查

385 张地点隔离 Query 加 7,263 张训练 Day distractor 后，Gallery 共 7,648 张。Query 使用固定相机退化，baseline/Night 读取相同张量。原始 Day、旧相机 Global、两个新候选的同地点 R@1 均为 383/385，仍接近饱和。

辅助跨视角指标仅评估有另一张同地点 Day 源图的 90 张 Query / 45 个地点，并从每条 Query 的所有候选中移除自己的精确 Day 源图：

| 模型 | 跨视角 R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day | 68/90 | 82/90 | 84/90 |
| 上一轮相机 Global | 67/90 | 83/90 | 84/90 |
| `guided_01` | 66/90 | 83/90 | 84/90 |
| `restore_01` | 67/90 | 83/90 | 86/90 |

该子集有选择偏差，不能代表全部 Query，但能揭示精确配对验证掩盖的跨视角退化。恢复同源图描述符的能力与跨视角地点检索能力需要分开检查。

## 产物与复现

全部输出在 `runs/night_guided_20261023/`：

- `guided_01/`、`restore_01/`：Night 权重、实际配置、训练日志、开发集及合成验证；最终 test 只评估所选 `restore_01`。
- `experiment_plan.json`、`experiment_plan_restore.json`：候选设置与开发阶段调整理由。
- `selection_before_test.json`：选中权重哈希及冻结的 224 输入策略。
- `results_summary.json`：机器可读汇总、逐步增强一致性及错误变化。
- `unit_tests.log`：代码验证记录。

所选权重 SHA-256：`c6b39b5af8a77e4a64e436f9d80ede1083315417e7cd457d9cc44381d58fff1b`。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# 使用新的输出目录，默认原始 Day 初始化
OUTPUT=runs/night_guided_restore_reproduce_01 bash configs/night_guided_restore.sh

"$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir runs/night_guided_restore_reproduce_01 --device cuda \
  --batch-size 16 --num-workers 2 --reference-cache-mib 768 \
  --output runs/night_guided_restore_reproduce_01/svox_train.json
```

可选 `--query-scales 224,336` 已实现并完成 CPU 行为测试，本轮未做 GPU 收益验证，也未用于上述模型选择或结果。它保持 Day Gallery 不变，同时对 baseline 和 Night 使用相同 Query 策略；其可能的推理收益必须与训练收益分别报告。

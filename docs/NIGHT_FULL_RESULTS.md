# Night 全参数实验记录（2026-10-22）

这是无相机退化增强的历史基线。2026-10-23 的新增强重跑见 [相机增强结果](NIGHT_CAMERA_RESULTS.md)。

## 当前结论与可用权重

完整 Night DINOv2-B/14 已实现并实际训练。Night 的 Patch Embedding、CLS/位置编码、全部 Transformer Block 与最终 Norm 都进入优化器；原始 Day DINOv2 和共享 SALAD 冻结。Night 权重约 346 MB，只保存 Night 主干，仍直接检索原来的白天 Gallery。

最终选择 `runs/night_full_real_day_layerdecay07_anchor100_01/adapter.safetensors`。该模型**直接从 Day 初始化**，未经过 Qwen SGPD。选择依据是 SVOX train 内部空间验证的 Recall@1、Recall@5、Recall@10 依次比较，选择记录在 test 前写入 `runs/night_full_selection_20261022_v2.json`。权重 SHA-256 为 `864f9c82bca8dd2064d1165b877de634eefa133c157c1c09d96c9b8227eb006a`。

| SVOX test，823 夜图 / 17,166 原始白天 Gallery | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day SALAD | 763/823，92.71% | 810/823，98.42% | 816/823，99.15% |
| 所选 Full Night DINOv2 | 765/823，92.95% | 810/823，98.42% | 816/823，99.15% |

Night 修复 2 个 R@1 错误，没有新增 R@1 错误。结果文件包含全部 R@1 错误的路径和精确排名：`runs/night_full_real_day_layerdecay07_anchor100_01/svox_test.json`。机器可读汇总在 `runs/night_full_results_20261022.json`。

提升为 **0.24 个百分点，仅 2 张查询**。本轮没有用 test 继续调参；但先前 LoRA 实验已查看过 test，而且真实训练过滤使用了 test 文件名坐标。因此这是探索性官方文件划分结果，不能称为首次盲测或已证实稳定泛化收益。

## 合成配对训练

训练为 7,263 对 Qwen night/day，地点隔离的合成验证为 385 对。两阶段均 batch 8、BF16、AdamW、weight decay 0.01、seed 42；只优化 Full Night DINOv2。全局损失权重 1、地点多正样本 InfoNCE 权重 1、白天描述符队列容量 256。

| 实验 | 配对余弦（合成验证） | 合成验证 R@1 | 完整 SVOX train R@1 / R@5 / R@10 |
| --- | ---: | ---: | --- |
| 原始 Day SALAD | 0.6778 | 384/385 | 88.46% / 95.16% / 96.01% |
| `night_full_global_01`，LR 1e-5，1 epoch | 0.7199 | 385/385 | 84.19% / 93.16% / 94.59% |
| `night_full_global_lr1e6_01`，LR 1e-6，1 epoch | 0.7413 | 385/385 | 87.46% / 94.59% / 96.15% |
| `night_full_sgpd_lr1e6_01`，从上一行接续 1 epoch | 0.7474 | 385/385 | 87.46% / 94.30% / 96.01% |

SGPD 使用 Patch 权重 0.1、saliency 模式、Stage2B A 类可信对应；Keep、OT、Magnitude 都关闭。该阶段见到 196,535 个可信 Patch，约占训练 Patch 的 10.57%。所有 Teacher 目标、对应索引和置信度均被 detach。

合成余弦提高而真实域 Recall 未超过基线。SGPD 行还多训练了 1 epoch，因此它与第一阶段的差异不能独立归因于局部损失；需要从相同起点、相同步数另做 `PATCH_WEIGHT=0` 对照，才能判断 SGPD 的增量贡献。当前未证明 SGPD 对真实夜景有效。

## 真实夜景训练与内部验证

SVOX 官方 train/test 文件不重复，但存在接近的连续帧。训练前根据 test Gallery 和 night Query **文件名的 UTM 坐标**设置 250 米排除带；不解码 test 图像、不使用 test 描述符。剩余 20,486 张 Day train Gallery、378 张 Night train Query。按北向坐标留出 76 张内部验证，255 张用于优化，47 张作为空间间隔；每张保留夜图有 4–44 个 25 米内正样本。

最终模型使用以下设置：

- 直接复制 Day DINOv2，FP32，batch 4，3 epoch，共 192 次更新；内部验证选第 3 epoch。
- 25 米内全部为正样本，25–250 米的 Gallery 项不参与 InfoNCE，250 米外为负样本；平均每个 Query 有约 20,101 个有效负样本。
- 末层 LR 1e-6，逐层衰减 0.7，最早层 LR 约 1.38e-8，全部参数仍可训练；AdamW、weight decay 0.01。
- 加入同一夜图的冻结 Day Teacher 描述符约束，权重 100；同白天图 anchor 权重 0。
- 真实夜图与白天 Gallery 没有逐 Patch 可信对应，此阶段不使用 SGPD、Keep 或 OT。

| 模型 / 固定 Day train Gallery | 内部验证 R@1 / R@5 / R@10（76 张） | 优化子集 R@1（255 张） |
| --- | --- | ---: |
| Day 基线 | 59/76 / 70/76 / 70/76 | 233/255 |
| 最终所选 Day→真实夜景 | 59/76 / 70/76 / 71/76 | 235/255 |
| 正确加载 SGPD 的初始模型 | 49/76 / 65/76 / 68/76 | 230/255 |
| SGPD→真实夜景，最佳 epoch 1 | 59/76 / 69/76 / 70/76 | 230/255 |

正确执行 SGPD→真实阶段的目录为 `runs/night_full_real_sgpd_verified_01`，其 `adapter.json` 记录了 SGPD 初始化的路径、权重哈希和 `precision=bf16`。它未超过所选 Day 初始化基线，未用其 test 分数选模，也未对它额外测 test。

## 配置脚本修正与来源核验

收尾审查发现，旧 `configs/night_full_real.sh` 构造了 `EXTRA_ARGS` 却没有传给 Python，导致 `INIT_ADAPTER_DIR` 和 `BF16` 环境变量被忽略。修正前通过该脚本运行的真实阶段实际都是 **Day 初始化、FP32**；即使目录名含 `qwen` 或 `sgpd` 也不能据此判断来源。旧 `adapter.json` 正确记录了 `initial_night_backbone=null` 与 `precision=fp32`，现已据此重新解释结果。

该问题已修复并加入参数转发回归测试。原始 `runs/night_full_selection_20261022.json` 在 test 前失效，修正版选择记录为 `runs/night_full_selection_20261022_v2.json`。最终所选目录明确标注 `day`，其元数据与真实训练过程一致；SGPD 初始化的真实阶段已在修正后另跑。

新训练会在开始打印有效参数，并保存 `training_config.json`。始终以 `adapter.json` 中的 `initial_night_backbone`、`checkpoint_sha256`、`precision` 和损失配置确认来源。

## 复现

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python

# 与最终所选模型相同配置；请使用未存在的输出目录
OUTPUT=runs/night_full_real_reproduce_01 INIT_ADAPTER_DIR= BF16=0 bash configs/night_full_real.sh

"$PYTHON" scripts/evaluate_svox.py --split test \
  --adapter-dir runs/night_full_real_reproduce_01 --device cuda \
  --batch-size 8 --num-workers 4 --reference-cache-mib 768 \
  --save-hard-cases --output runs/night_full_real_reproduce_01/svox_test.json

# 合成蒸馏研究路线，现默认低学习率各 1 epoch
OUTPUT=runs/night_full_global_reproduce_01 bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_reproduce_01 \
  INIT_ADAPTER_DIR=runs/night_full_global_reproduce_01 bash configs/night_full_sgpd.sh

# 独立验证 SGPD 初始化的真实阶段；此命令会真实传入初始化与 BF16
OUTPUT=runs/night_full_real_sgpd_reproduce_01 \
  INIT_ADAPTER_DIR=runs/night_full_sgpd_reproduce_01 BF16=1 \
  bash configs/night_full_real.sh
```

单 seed、76 张开发集与当前测试协议限制了结论强度。后续研究应先准备独立真实夜景开发集，再做相同步数的 SGPD 消融和多 seed 验证；当前工程已经保留独立损失、严格来源核验和固定白天检索空间。

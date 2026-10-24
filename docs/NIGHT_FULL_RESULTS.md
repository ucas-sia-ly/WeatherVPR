# Night 全参数配对训练记录（2026-10-22）

这是无相机退化增强的历史全部参数 Qwen 配对训练基线。当前训练默认已改为独立 Night 主干的后四层微调，见 [SALAD 训练设置](SALAD_TRAINING.md)；以下历史模型、数字及全部参数消融入口保留。相机增强的同预算重跑见 [相机增强结果](NIGHT_CAMERA_RESULTS.md)。

## 模型与训练

完整 Night DINOv2-B/14 从白天 `dino_salad.ckpt` 精确复制初始化。Night 的 Patch Embedding、CLS/位置编码、全部 Transformer Block 与最终 Norm 都进入优化器；原始 Day DINOv2 和共享 SALAD 冻结。Night 权重约 346 MB，只保存 Night 主干，直接检索原来的白天 Gallery。

训练使用 7,263 对 Qwen night/day，地点隔离的合成验证为 385 对。两阶段均 batch 8、BF16、AdamW、weight decay 0.01、seed 42；全局损失和地点多正样本 InfoNCE 权重各 1，白天描述符队列容量 256。每个 epoch 有 908 次更新。

## 配对验证与真实检索诊断

表中完整 SVOX train 指标是对 **22,232 张原始白天 Gallery 和 702 张夜间 Query 的只读检索评估**。这些真实图像未用于以下模型的参数更新。

| 实验 | 配对余弦（合成验证） | 合成验证 R@1 | SVOX train R@1 / R@5 / R@10 |
| --- | ---: | ---: | --- |
| 原始 Day SALAD | 0.6778 | 384/385 | 88.46% / 95.16% / 96.01% |
| `night_full_global_01`，LR 1e-5，1 epoch | 0.7199 | 385/385 | 84.19% / 93.16% / 94.59% |
| `night_full_global_lr1e6_01`，LR 1e-6，1 epoch | 0.7413 | 385/385 | 87.46% / 94.59% / 96.15% |
| `night_full_sgpd_lr1e6_01`，从上一行接续 1 epoch | 0.7474 | 385/385 | 87.46% / 94.30% / 96.01% |

SGPD 使用 Patch 权重 0.1、saliency 模式、Stage2B A 类可信对应；Keep、OT、Magnitude 关闭。该阶段见到 196,535 个可信 Patch，约占训练 Patch 的 10.57%。Teacher 目标、对应索引和置信度均被 detach。

合成余弦提高而真实域 Recall 未超过基线。SGPD 行比第一阶段多训练 1 epoch，差异不能独立归因于局部损失；相机增强实验补充了相同起点、相同步数的 Global 与 SGPD 接续对照。当前没有证据表明 SGPD 已稳定改善真实夜景检索。

原始 Day SALAD 的 SVOX test 基线为 R@1 **763/823，92.71%**；R@5 **810/823，98.42%**；R@10 **816/823，99.15%**。本页配对模型的真实域诊断使用上表完整 SVOX train 协议。

## 权重与复现

可用的配对训练结果保存在 `runs/night_full_global_01/`、`runs/night_full_global_lr1e6_01/` 和 `runs/night_full_sgpd_lr1e6_01/`。各目录的 `adapter.json`、`training_config.json` 与训练日志记录实际模型来源、精度及损失配置。

两个历史 full 入口显式默认 `FULL_TRAIN_SCOPE=all`。32 GB 主机使用以下命令，逐项串行执行；每次使用新目录：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

OUTPUT=runs/night_full_global_reproduce_01 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_global_reproduce_01.memory.jsonl \
  -- bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_reproduce_01 \
  INIT_ADAPTER_DIR=runs/night_full_global_reproduce_01 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_sgpd_reproduce_01.memory.jsonl \
  -- bash configs/night_full_sgpd.sh

"$PYTHON" scripts/run_bounded.py --log runs/night_full_sgpd_reproduce_01.pairs.memory.jsonl \
  -- "$PYTHON" scripts/evaluate_pairs.py \
  --adapter-dir runs/night_full_sgpd_reproduce_01 --device cuda --num-workers 0 \
  --output runs/night_full_sgpd_reproduce_01/synthetic_val.json

# 仅评估真实夜景，不更新模型
"$PYTHON" scripts/run_bounded.py --log runs/night_full_sgpd_reproduce_01.svox.memory.jsonl \
  -- "$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir runs/night_full_sgpd_reproduce_01 --device cuda \
  --batch-size 8 --num-workers 0 --reference-cache-mib 768 \
  --gallery-cache runs/cache/svox_train_eval_day.safetensors \
  --output runs/night_full_sgpd_reproduce_01/svox_train.json
```

当前实验为单 seed，合成验证已接近饱和。真实夜景 test 在历史实验中已被查看，后续比较应明确属于探索性评估，并避免据 test 调整超参数。

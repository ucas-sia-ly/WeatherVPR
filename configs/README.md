# Night 训练配置

## 新增：检索指导与知识保留

`night_guided_retrieval.sh` 用全部训练白天图构建固定 Teacher 特征库（7,263 keys / 6,441 地点），新增多正样本难负 InfoNCE、排除同地点的 Teacher 检索关系 KL、干净 Qwen 冻结指导及训练 Day 知识保留。仅使用合成图对与训练 Day，原 Day/SALAD/Gallery 保持冻结；真实夜图不参与梯度更新。

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
OUTPUT=runs/night_guided_retrieval_01 bash configs/night_guided_retrieval.sh
```

默认 `GLOBAL_WEIGHT=1`、`CONTRAST_WEIGHT=0.1`、`RELATION_WEIGHT=0.1`、`SYNTHETIC_GUIDANCE_WEIGHT=0.1`、`DAY_PRESERVATION_WEIGHT=0.5`，局部损失默认关闭；各权重设为 `0` 可独立关闭。`CONTRAST_HARD_NEGATIVES=64`，设 `0` 使用全部负 key；关系支持集另由 `RELATION_HARD_NEGATIVES` 控制。固定库缓存路径为 `DAY_BANK`。

默认 microbatch 2、累积 4 次，BF16、activation checkpoint、`LAYER_DECAY=0.8`、cosine 日程及 32 次优化器更新 warmup。短流程的 `MAX_STEPS` 按 microbatch 计数，需设 `WARMUP_STEPS=0` 或保证其小于总更新数。复现、独立消融、两篇论文与当前改造的区别、更难合成验证命令见 [检索指导说明](../docs/GUIDED_RETRIEVAL.md)。Recall 以实测输出为准。

`night_guided_restore.sh` 是另一候选，默认 `GLOBAL_WEIGHT=0.1`、`SYNTHETIC_GUIDANCE_WEIGHT=1`、`DAY_PRESERVATION_WEIGHT=0.5`、`DAY_VIEW_AUGMENTATION=camera`；其余继承 guided 设置。它对训练白天原片生成独立相机退化视图，让 Night 主干恢复干净 Teacher 描述符，Teacher / 可信匹配 / Gallery 不受该退化影响。原 guided 的 `DAY_VIEW_AUGMENTATION` 仍默认 `off`；设置 `DAY_VIEW_AUGMENTATION=off bash configs/night_guided_restore.sh` 可关闭新视图做消融。

两组候选已完成训练和评估：所选 restore 的开发集 R@1 为 89.17%，test R@1 为 92.71%，与原始 Day 持平。完整数字和复现产物见 [实测记录](../docs/GUIDED_RETRIEVAL_RESULTS.md)。

`evaluate_pairs.py --gallery-cache <path>` 可复用冻结 Day Gallery，自动附加 `cross_view_baseline` / `cross_view_adapter`：排除 Query 精确源图，仅用同地点其它 Day 视角作正样本。该指标只覆盖现有 90 Query / 45 地点的多视角子集；原配对 Recall 保留。评估示例使用 batch 16，显存紧张时可进一步改为 8。

## 原有全参数与相机增强对照

当前默认方案是从 `weights/dino_salad.ckpt` 中的白天 DINOv2-B/14 初始化一个独立 Night DINOv2，并全参数微调；白天 Teacher 与 SALAD 冻结。`night_full_global.sh` 用 Global + 跨域多正样本 InfoNCE，`night_full_sgpd.sh` 接续加入 Stage2B A 类可信 Patch 的 SGPD；两阶段默认学习率 1e-6、各 1 epoch。只保存 Night DINOv2 权重，不复制 SALAD。每次运行请使用新的 `OUTPUT` 目录。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
OUTPUT=runs/night_full_global_01 bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_01 \
  INIT_ADAPTER_DIR=runs/night_full_global_01 bash configs/night_full_sgpd.sh
```

全参数脚本默认 batch 8、BF16、AdamW，并保留容量 256 的白天描述符队列与地点分散采样；显存不足时可用 `BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 保持有效 batch。`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 可分别配置，后三项默认关闭。旧 `night_global.sh` 和 `night_sgpd.sh` 继续支持 LoRA（rank 8、alpha 16、最后 4 个 Block 的 attention `qkv/proj`）以及 `ADAPTER_TYPE=output` 消融。

`night_camera_replay.sh` 显式启用 `configs/night_camera.json`，重跑增强 Global、同预算 Global 接续和 SGPD 接续三组 Qwen 配对训练，并在原图合成验证集评估。真实夜景检索由 `scripts/evaluate_svox.py` 单独执行：

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_02 bash configs/night_camera_replay.sh
```

两个 full 训练脚本均支持 `NIGHT_AUGMENTATION=off|camera`、`AUGMENTATION_CONFIG`、`AUGMENTATION_STRENGTH`、`AUGMENTATION_PROBABILITY`、`AUGMENTATION_EPOCH_OFFSET`。单独提供 JSON 不会打开增强，需显式指定 `camera`。两个合成接续阶段都用 offset 1，保持同一批新采样。操作范围、独立关闭方式见 [增强说明](../docs/NIGHT_CAMERA_AUGMENTATION.md)，本轮数字见 [重跑结果](../docs/NIGHT_CAMERA_RESULTS.md)。

`configs/dino_salad_stage2b_calibration.json` 只与当前完整 checkpoint 和 224×224 预处理匹配；训练脚本默认排除 `data/splits/heldout_places.json` 的地点。其它天气还没有训练数据与独立实验结果，需先准备相应配对清单和测试集。

# Night 训练配置

## 默认：独立 Night 主干的后四层微调

`night_specialist.sh` 从原始 Day checkpoint 初始化独立 Night DINOv2-B/14，默认 `FULL_TRAIN_SCOPE=late FULL_TRAIN_BLOCKS=4`，直接更新 blocks 8–11 的原始权重。其余 Night 参数（包括最终 Norm）冻结，Day Teacher 与共享 SALAD 全部冻结；输入梯度可穿过 Norm 与 SALAD。权重仍保存完整 Night 主干，便于独立推理；这不是 LoRA。

默认 `LAYER_DECAY=1`、LR `1e-6`、batch 8、1 epoch、AdamW、BF16、activation checkpoint、cosine 日程与 32 次更新 warmup，`NUM_WORKERS=0`。`FULL_TRAIN_SCOPE=all` 可切回全部参数消融；`FULL_TRAIN_SCOPE=early|late FULL_TRAIN_BLOCKS=N` 可比较其它层范围。SALAD 原始训练设置与当前蒸馏任务的区别见 [SALAD 训练说明](../docs/SALAD_TRAINING.md)。

主分支读取 Qwen Night（可叠加相机退化）。辅助分支从训练 Day 原图生成必做曝光衰减的低照度视图；干净 Day 仅供冻结 Teacher 和 Day bank。参数见 `source_night.json`，`SOURCE_NIGHT_WEIGHT=0` 可关闭辅助分支，`SOURCE_NIGHT_SAMPLES` 控制每步辅助样本上限。真实夜图仅评估。

默认 Global / Contrast / Relation 权重为 `0.1 / 0.1 / 0.1`、Source Night 为 `0.5`；SGPD / Keep / OT / Magnitude 默认关闭，可独立赋予正权重。固定训练 Day bank 有 7,263 keys / 6,441 地点；InfoNCE 对同一 `(city, place_id)` 的全部条目视为正样本，默认 64 个难负样本。关系蒸馏排除同地点，单独由 `RELATION_WEIGHT` 和 `RELATION_HARD_NEGATIVES` 控制。

32 GB 主机应逐项串行运行，并使用内存限制：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUTPUT=runs/night_late4_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_late4_02.memory.jsonl \
  -- bash configs/night_specialist.sh
```

历史 `source_night` 是**全部参数微调、layer decay 0.8**，其 SVOX test R@1/5/10 为 93.07% / 98.66% / 99.03%；默认值改变后，原命令不再代表该历史配置。复现必须显式设置 `FULL_TRAIN_SCOPE=all LAYER_DECAY=0.8`，完整命令见 [历史实验](../docs/NIGHT_SPECIALIST_RESULTS.md)。用户 `runs/night_late4_01` 保留原始记录，不将其当前 test R@1 解释为超过历史 `source_night`。

`evaluate_pairs.py --gallery-cache <path>` 可缓存冻结 Day Gallery，并附加排除精确源图的 `cross_view_baseline` / `cross_view_adapter`。该指标目前仅覆盖 90 Query / 45 地点的多视角子集，不能替代真实夜间评估。

## 当前 Patch / 注意力对照

`night_patch.sh` 复用后四层 Night 配置，改用 `night_patch_camera.json`：整张增强和曝光操作概率均为 1，使每张 Qwen 输入都经过相机处理。默认 `PATCH_WEIGHT=0.1 ATTENTION_WEIGHT=0 PATCH_VISIBILITY=photometric`；设 `ATTENTION_WEIGHT=0.05` 可加入最后一层 DINOv2 CLS 注意力 KL。SGPD 和注意力仅监督原 Stage 2B A 类可信对应，退化输入的黑场/裁剪像素比例只降低置信度，不创建对应。Teacher 输入保持干净，Source Night 分支不变。

三组固定对照为 `PATCH_WEIGHT=0 ATTENTION_WEIGHT=0`、`0.1 / 0`、`0.1 / 0.05`。各组必须使用独立新输出目录并通过内存保护串行运行；本轮上限 6 GiB、high 5 GiB、系统可用内存保留 4 GiB。公式、CLI、复现和开发集选择规则见 [Patch 与注意力蒸馏](../docs/NIGHT_PATCH_ATTENTION.md)。

## 历史全部参数与相机增强对照

`night_full_global.sh` 和 `night_full_sgpd.sh` 显式默认 `FULL_TRAIN_SCOPE=all`，保留全部参数微调的历史消融；可以环境变量覆盖范围。第一阶段为 Global + 跨域多正样本 InfoNCE，第二阶段接续加入 Stage 2B A 类可信对应上的 SGPD。两阶段默认 LR `1e-6`、batch 8、BF16、AdamW、各 1 epoch，使用容量 256 的白天描述符队列和地点分散采样。

```bash
OUTPUT=runs/night_full_global_replay_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_global_replay_02.memory.jsonl \
  -- bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_replay_02 \
  INIT_ADAPTER_DIR=runs/night_full_global_replay_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_sgpd_replay_02.memory.jsonl \
  -- bash configs/night_full_sgpd.sh
```

`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 分别配置，后三项默认关闭。使用 `BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 可在显存较小时保持有效 batch。旧 `night_global.sh` / `night_sgpd.sh` 保留 LoRA（rank 8、alpha 16、最后四层 attention `qkv/proj`）和 `ADAPTER_TYPE=output` 消融。

两个 full 脚本支持 `NIGHT_AUGMENTATION=off|camera`、`AUGMENTATION_CONFIG`、`AUGMENTATION_STRENGTH`、`AUGMENTATION_PROBABILITY`、`AUGMENTATION_EPOCH_OFFSET`。仅指定 JSON 不会打开增强，还需指定 `camera`。`night_camera_replay.sh` 保留增强 Global 与同预算 Global / SGPD 接续的历史流程；32 GB 主机运行每阶段时均应通过 `run_bounded.py`，避免绕过内存约束。操作参数见 [增强说明](../docs/NIGHT_CAMERA_AUGMENTATION.md)，已有数字见 [相机增强结果](../docs/NIGHT_CAMERA_RESULTS.md)。

`configs/dino_salad_stage2b_calibration.json` 只与当前 checkpoint 和 224×224 预处理匹配；训练默认排除 `data/splits/heldout_places.json`。其它天气尚未训练，需先准备各自的配对清单和测试集。

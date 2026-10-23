# Night 训练配置

当前默认方案是从 `weights/dino_salad.ckpt` 中的白天 DINOv2-B/14 初始化一个独立 Night DINOv2，并全参数微调；白天 Teacher 与 SALAD 冻结。`night_full_global.sh` 用 Global + 跨域多正样本 InfoNCE，`night_full_sgpd.sh` 接续加入 Stage2B A 类可信 Patch 的 SGPD；两阶段默认学习率 1e-6、各 1 epoch。只保存 Night DINOv2 权重，不复制 SALAD。每次运行请使用新的 `OUTPUT` 目录。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
OUTPUT=runs/night_full_global_01 bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_01 \
  INIT_ADAPTER_DIR=runs/night_full_global_01 bash configs/night_full_sgpd.sh
```

全参数脚本默认 batch 8、BF16、AdamW，并保留容量 256 的白天描述符队列与地点分散采样；显存不足时可用 `BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 保持有效 batch。`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 可分别配置，后三项默认关闭。旧 `night_global.sh` 和 `night_sgpd.sh` 继续支持 LoRA（rank 8、alpha 16、最后 4 个 Block 的 attention `qkv/proj`）以及 `ADAPTER_TYPE=output` 消融。

`night_full_real.sh` 是可选的 SVOX train 真实夜景阶段。设置 `INIT_ADAPTER_DIR` 可从已训练的完整 Night 主干继续；省略则从白天 DINOv2 初始化。脚本按 test 文件名坐标排除 250 米内的 train 图像，并在剩余 train 夜图中按空间位置留出内部验证集。只使用真实夜图的 25 米地理正样本和固定白天 Gallery 训练，不使用 SGPD Patch Loss；最佳验证 epoch 保存在独立 `OUTPUT` 目录。运行前可调用 `python scripts/train_real_night.py --check-data` 查看样本数。

真实阶段默认 FP32、batch 4、3 epoch、学习率 1e-6、`LAYERWISE_LR_DECAY=0.7`、`NIGHT_BASELINE_WEIGHT=100`、`NEGATIVE_RADIUS_M=250`，对应当前选定的 Day→真实夜景基线。`NIGHT_BASELINE_WEIGHT=0` 关闭同夜图 Teacher 约束，`LAYERWISE_LR_DECAY=1` 关闭逐层衰减，`BF16=1` 启用混合精度。不同初始化和精度属于不同实验，应先在 train 内部验证后比较；当前 SGPD 初始化的额外真实收益尚未成立。

`night_camera_replay.sh` 显式启用 `configs/night_camera.json`，重跑增强 Global、同预算 Global 接续、SGPD 接续，以及 Day/SGPD 初始化的真实阶段。训练外的验证和测试均使用原图：

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_02 bash configs/night_camera_replay.sh
```

三个 full 脚本均支持 `NIGHT_AUGMENTATION=off|camera`、`AUGMENTATION_CONFIG`、`AUGMENTATION_STRENGTH`、`AUGMENTATION_PROBABILITY`、`AUGMENTATION_EPOCH_OFFSET`。单独提供 JSON 不会打开增强，需显式指定 `camera`。两个合成接续阶段都用 offset 1，保持同一批新采样。操作范围、独立关闭方式见 [增强说明](../docs/NIGHT_CAMERA_AUGMENTATION.md)，本轮数字见 [重跑结果](../docs/NIGHT_CAMERA_RESULTS.md)。

`configs/dino_salad_stage2b_calibration.json` 只与当前完整 checkpoint 和 224×224 预处理匹配；训练脚本默认排除 `data/splits/heldout_places.json` 的地点。其它天气还没有训练数据与独立实验结果，需先准备相应配对清单和测试集。

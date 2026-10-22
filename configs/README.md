# Night 训练配置

`night_global.sh` 训练第一阶段的 LoRA：全局描述符对齐与跨域多正样本 InfoNCE，默认 1 epoch。`night_sgpd.sh` 在第二阶段加入 Stage2B A 类可信 Patch 的 SGPD，默认 2 epoch。两个脚本默认共用 `weights/dino_salad.ckpt` 的冻结 DINOv2-B/14 与 SALAD；只保存夜间 LoRA。首次运行请使用新的 `OUTPUT` 目录。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
OUTPUT=runs/night_lora_global_01 EPOCHS=1 bash configs/night_global.sh
OUTPUT=runs/night_lora_sgpd_01 EPOCHS=2 \
  INIT_ADAPTER_DIR=runs/night_lora_global_01 bash configs/night_sgpd.sh
```

默认 LoRA rank 8、alpha 16、dropout 0，注入 DINOv2 最后 4 个 Block 的 attention `qkv` 和 `proj`。`ADAPTER_TYPE=output` 可运行旧输出端 Adapter 消融。`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 等环境变量可独立设定；后三项默认关闭。容量 256 的白天描述符队列与地点分散采样减少 batch size 4 的负样本不足问题。

`configs/dino_salad_stage2b_calibration.json` 只与当前完整 checkpoint 和 224×224 预处理匹配；训练脚本默认排除 `data/splits/heldout_places.json` 的地点。其它天气还没有训练数据与独立实验结果，需先准备相应配对清单和测试集。

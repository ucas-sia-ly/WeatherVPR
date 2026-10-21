# Night 实验入口

`night_global.sh` 用两步训练检查模型与数据链路。`night_sgpd.sh` 默认对 `data/splits/night_train.jsonl` 做完整 3 epoch SGPD 训练：同一冻结主干、独立 Adapter、与 checkpoint 一致的 Stage2B 校准，并排除历史校准和合成验证地点。

```bash
conda activate a2b-lightvpr
OUTPUT=runs/night_global_smoke_new bash configs/night_global.sh
OUTPUT=runs/night_sgpd_01 bash configs/night_sgpd.sh
```

`night_sgpd.sh` 可通过环境变量控制 `OUTPUT`、`MANIFEST`、`CALIBRATION`、`EXCLUDE_PLACES`、`CKPT`、`DINO_DIR`、`PATCH_WEIGHT`、`KEEP_WEIGHT`、`GLOBAL_WEIGHT`、`CONTRAST_WEIGHT`、`IMPORTANCE_MODE`、`BATCH_SIZE`、`EPOCHS`、`MAX_STEPS`、`SEED`、`NUM_WORKERS`、`BF16`、`VERIFY_HASHES`、`DEVICE`、`PYTHON`。所有损失权重均可单独置零，但至少一个须为正。

其它天气脚本是原型模板；当前只准备了 night 数据、校准与 SVOX night 评估。每次实验必须使用新的输出目录，避免覆盖已有 Adapter。

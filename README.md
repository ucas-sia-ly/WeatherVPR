# WeatherVPR：DINOv2-SALAD 的 Night SGPD Adapter

本项目以完整的白天 `dino_salad.ckpt` 为共享冻结主干，只训练约 0.8 MB 的夜间残差 Adapter。真实白天图作为 Teacher，逐一对应的 Qwen 生成夜图作为 Student。SGPD 只在可信 Patch 对应上蒸馏局部特征，并用白天 SALAD 的 `assignment_keep` 加权。推理时夜间描述符直接检索**原始白天 SALAD Gallery**。

参考论文 [*Distillation Improves Visual Place Recognition for Low Quality Images*](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Yang_Distillation_Improves_Visual_Place_Recognition_for_Low_Quality_Images_ICCVW_2025_paper.html) 的 Teacher–Student 思路；本实现的 SGPD 和跨域 InfoNCE 是新的实验设计，并非论文 ICKD、MSE、Triplet 的逐项复现。

## 当前就绪状态

- `weights/dino_salad.ckpt`、`third_party/salad/`、`third_party/dinov2/` 已在项目内；运行时不查找外部 WORKSPACE。
- `data/train/manifest.jsonl` 是 7,719 对 SHA-256 核验过的 night/day 精确配对。`data/splits/night_train.jsonl` 有 7,263 对训练图，`night_val.jsonl` 有 385 对地点隔离的合成验证图；历史保留地点另排除 71 对。
- `configs/dino_salad_stage2b_calibration.json` 与当前 checkpoint 和 224×224 预处理匹配。相应地点排除已纳入 split；SGPD 的 A 类可信 Patch 才参与局部损失。
- `data/svox/images/test/` 有 17,166 张白天 Gallery 和 823 张真实夜间 Query。评估采用 SVOX 25 m UTM 正样本协议，报告 Recall@1/5/10。
- 已在 GPU 跑通两步 Global 与两步 SGPD 训练、Adapter 描述符导出、完整 SVOX 检索和合成验证。原始 SALAD 在 SVOX night 的 25 m R@1/5/10 为 92.71% / 98.42% / 99.15%；两步 SGPD smoke Adapter 的 Recall 与其相同。两步训练只是工程检查，不代表完整实验结果。

当前目录所需的模型、源码和图像均已具备，**无需再下载文件**。另一台机器上的资源来源见 [下载与来源](docs/DOWNLOADS.md)。

新 Python 环境先安装与本机驱动兼容的 PyTorch，再执行 `pip install -r requirements-extra.txt`。无需安装 DINOv2 仓库中针对旧 CUDA 版本固定的整套依赖。

## 运行

以下命令从项目根目录执行。本机 `base` 环境的 PyTorch CUDA 构建与驱动不兼容；已验证 `a2b-lightvpr` 环境可用。

```bash
conda activate a2b-lightvpr
python scripts/check_setup.py --with-train-data
PYTHONPATH=src python -m unittest discover -s tests -v

# 这些文件当前已生成；复制项目或重新整理数据时可重跑
python scripts/prepare_calibration.py
python scripts/make_night_split.py

# 两步链路检查
OUTPUT=runs/night_global_smoke_new bash configs/night_global.sh

# 完整 night SGPD 训练；默认 3 epoch、batch 4、bf16
OUTPUT=runs/night_sgpd_01 bash configs/night_sgpd.sh

# 合成夜图的地点隔离验证；用于开发阶段比较消融
python scripts/evaluate_pairs.py --adapter-dir runs/night_sgpd_01 \
  --device cuda --output runs/night_sgpd_01/synthetic_val.json

# 与未适配 SALAD baseline 共用同一白天图库，比较真实夜景 R@1/5/10
python scripts/evaluate_svox.py --adapter-dir runs/night_sgpd_01 \
  --device cuda --batch-size 32 --num-workers 4 \
  --output runs/night_sgpd_01/svox_test.json
```

`evaluate_svox.py` 默认在 `runs/cache/` 保存经过 checkpoint、预处理和图库图像字节校验的白天描述符缓存。后续 Adapter 消融可复用它，无需重新提取白天 Gallery。只检查测试集协议可运行 `python scripts/evaluate_svox.py --check-data`。

`evaluate_pairs.py` 使用 385 对未参与训练的 Qwen 合成夜图与其精确白天源图，报告同地点 Recall@1/5/10 和平均配对描述符余弦相似度。它与真实 SVOX 测试分开。`--limit-pairs 8` 可做短流程检查，输出明确标记为子集。

## 独立消融

`configs/night_sgpd.sh` 接受环境变量覆盖各损失和训练预算；每次设置新的 `OUTPUT`。建议所有实验使用相同 split、seed、epoch 与 Gallery 缓存。

```bash
# Global + 跨域 InfoNCE
OUTPUT=runs/night_global_01 PATCH_WEIGHT=0 KEEP_WEIGHT=0 bash configs/night_sgpd.sh

# 同一可信对应，但 Patch 权重均匀
OUTPUT=runs/night_uniform_01 IMPORTANCE_MODE=uniform bash configs/night_sgpd.sh

# SALAD Keep 引导 SGPD
OUTPUT=runs/night_sgpd_01 bash configs/night_sgpd.sh

# 加 Keep Alignment
OUTPUT=runs/night_sgpd_keep_01 KEEP_WEIGHT=0.01 bash configs/night_sgpd.sh
```

可用 `PATCH_WEIGHT`、`KEEP_WEIGHT`、`GLOBAL_WEIGHT`、`CONTRAST_WEIGHT`、`IMPORTANCE_MODE`、`BATCH_SIZE`、`EPOCHS`、`MAX_STEPS`、`SEED`、`BF16`、`DEVICE`、`PYTHON` 等变量调整。训练日志、Adapter 权重和配置保存在各自 `runs/` 目录，完整主干只存一份。

## 目录

```text
configs/                  night 命令、校准及排除地点
data/train/               已核验的原图、生成图和配对清单
data/splits/              地点隔离的训练/合成验证清单
data/svox/                真实天气测试图
src/weather_vpr/models/  冻结共享模型与轻量 Adapter
src/weather_vpr/losses/  SGPD、Keep、全局和地点判别损失
src/weather_vpr/data/    精确配对和可信 Patch 对应
src/weather_vpr/inference/ SVOX 检索与描述符导出
scripts/                  数据准备、训练、评估入口
third_party/              项目内部的 SALAD/DINOv2 源码
weights/                  原始完整 checkpoint
runs/                     实验输出和 Gallery 缓存
```

方法边界与评估协议见 [设计说明](docs/ARCHITECTURE.md)、[数据和评估](docs/DATA_AND_EVALUATION.md)。`assignment_keep` 表示 Sinkhorn 有效簇分配质量，不等同于建筑语义标签；完整训练和真实夜景 Recall 才能验证实际收益。

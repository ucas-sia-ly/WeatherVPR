# WeatherVPR：Night 专属 DINOv2 + 冻结 SALAD

本项目从白天训练好的 `dino_salad.ckpt` 初始化**独立的 Night DINOv2-B/14**，全参数微调 Night 主干；原始白天 DINOv2 作为冻结 Teacher，SALAD 聚合器仍由 Day/Night 共用且冻结。真实白天图与其 Qwen 生成夜图构成训练图对。SGPD 只在可信 Patch 对应上蒸馏局部特征，并用白天 SALAD 的 `assignment_keep` 加权。Night 描述符直接检索**原始白天 SALAD Gallery**，无需重建图库。

参考论文 [*Distillation Improves Visual Place Recognition for Low Quality Images*](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Yang_Distillation_Improves_Visual_Place_Recognition_for_Low_Quality_Images_ICCVW_2025_paper.html) 的 Teacher–Student 思路；本实现的 SGPD 和跨域 InfoNCE 是新的实验设计。先前的 LoRA 和输出端 Patch/CLS Adapter 均保留为消融基线；现在 `--adapter-type full` 是默认配置。Night DINOv2 有 86,580,480 个可训练参数，单独的权重文件约 346 MB；其它天气模型尚未训练。

## 当前就绪状态

- `weights/dino_salad.ckpt`、`third_party/salad/`、`third_party/dinov2/` 已在项目内；运行时不查找外部 WORKSPACE。
- `data/train/manifest.jsonl` 是 7,719 对 SHA-256 核验过的 night/day 精确配对。`data/splits/night_train.jsonl` 有 7,263 对训练图，`night_val.jsonl` 有 385 对地点隔离的合成验证图；历史保留地点另排除 71 对。
- `configs/dino_salad_stage2b_calibration.json` 与当前 checkpoint 和 224×224 预处理匹配。相应地点排除已纳入 split；SGPD 的 A 类可信 Patch 才参与局部损失。
- SVOX train 有 22,232 张白天 Gallery 和 702 张夜间 Query，可用于授权的真实夜景微调；test 有 17,166 张白天 Gallery 和 823 张夜间 Query。评估采用 25 m UTM 正样本协议。
- 已完成的 LoRA 对照保存在 `runs/night_lora_global_01` 和 `runs/night_lora_sgpd_01`；`runs/night_sgpd_01` 是旧 Output Adapter 的独立结果。Night 全参数 DINOv2 使用新的 `runs/night_full_*` 实验目录。

### 本轮全参数结果

| 模型 / 原始白天 Gallery | SVOX test R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始白天 SALAD | 92.71%（763/823） | 98.42% | 99.15% |
| Day 初始化 → 真实夜景全参数微调 | **92.95%（765/823）** | 98.42% | 99.15% |

所选权重是 `runs/night_full_real_day_layerdecay07_anchor100_01/adapter.safetensors`，从白天 DINOv2 直接初始化，使用 SVOX train、逐层学习率衰减与同夜图 Teacher 描述符约束；未经过 Qwen SGPD。模型在 train 内部空间验证集选定后再测 test，修复 2 个 R@1 错误，没有新增 R@1 错误。增益只有 0.24 个百分点，且过滤使用了 test 文件名坐标、先前 LoRA 也查看过 test，结论属于探索性实验。

Qwen SGPD 的合成配对余弦从 0.6778 提高到 0.7474，但其完整 SVOX train R@1 为 87.46%，低于原始模型的 88.46%；目前**未证明 SGPD 对真实夜景的额外收益**。训练来源、脚本修正、逐阶段结果和复现命令见 [全参数实验记录](docs/NIGHT_FULL_RESULTS.md)，历史 LoRA 对照见 [数据和评估](docs/DATA_AND_EVALUATION.md)。

当前目录所需的模型、源码和图像均已具备，**无需再下载文件**。另一台机器上的资源来源见 [下载与来源](docs/DOWNLOADS.md)。

新 Python 环境先安装与本机驱动兼容的 PyTorch，再执行 `pip install -r requirements-extra.txt`。无需安装 DINOv2 仓库中针对旧 CUDA 版本固定的整套依赖。

## 运行

以下命令从项目根目录执行。本机 `base` 环境的 PyTorch CUDA 构建与驱动不兼容；下列 `PYTHON` 指向已验证可用的 `a2b-lightvpr` 环境。每次训练必须使用新的 `OUTPUT` 目录。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
"$PYTHON" scripts/check_setup.py --with-train-data
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -v

# 这些文件当前已生成；复制项目或重新整理数据时可重跑
"$PYTHON" scripts/prepare_calibration.py
"$PYTHON" scripts/make_night_split.py

# 生成未使用过的实验目录；以下命令在同一个 shell 执行
EXPERIMENT="night_full_$(date +%Y%m%d_%H%M%S)"

# 可选：先运行 4 个 batch，得到一个实际更新后的全参数 Night 权重
OUTPUT="runs/${EXPERIMENT}_smoke" MAX_STEPS=4 bash configs/night_full_global.sh

# 阶段 1：Global Alignment + 跨 batch day-descriptor queue 的地点 InfoNCE
OUTPUT="runs/${EXPERIMENT}_global" bash configs/night_full_global.sh

# 阶段 2：从阶段 1 的 Night DINOv2 初始化，继续训练 SGPD + 全局损失
OUTPUT="runs/${EXPERIMENT}_sgpd" \
  INIT_ADAPTER_DIR="runs/${EXPERIMENT}_global" bash configs/night_full_sgpd.sh

# 训练命令结束并出现 adapter.safetensors 与 adapter.json 后再评估

# 合成夜图的地点隔离验证；用于开发阶段比较消融
"$PYTHON" scripts/evaluate_pairs.py --adapter-dir "runs/${EXPERIMENT}_sgpd" \
  --device cuda --output "runs/${EXPERIMENT}_sgpd/synthetic_val.json"

# 真实夜景基线：直接从 Day 初始化，仅使用 SVOX train；内部空间验证选最佳 epoch
OUTPUT="runs/${EXPERIMENT}_real_day" INIT_ADAPTER_DIR= BF16=0 bash configs/night_full_real.sh

# 如需验证 SGPD 初始化的额外作用，另开独立实验；当前尚无真实收益证据
# OUTPUT="runs/${EXPERIMENT}_real_sgpd" INIT_ADAPTER_DIR="runs/${EXPERIMENT}_sgpd" \
#   BF16=1 bash configs/night_full_real.sh

# 最终比较：与未适配 SALAD baseline 共用原始白天 Gallery
"$PYTHON" scripts/evaluate_svox.py --adapter-dir "runs/${EXPERIMENT}_real_day" \
  --device cuda --batch-size 32 --num-workers 4 \
  --output "runs/${EXPERIMENT}_real_day/svox_test.json"
```

两个合成阶段脚本默认使用 batch 8、BF16、AdamW、学习率 1e-6，各训练 1 epoch，优化 Night DINOv2 的所有参数（含 Patch Embedding、位置/CLS Token、全部 Block 与 Norm）。白天 DINOv2 和 SALAD 始终冻结。地点分散 batch 与容量 256 的 detached 白天描述符队列提供跨域多正样本 InfoNCE；同一 `(city, place_id)` 的多个白天描述符不会变成伪负例。训练日志记录有效负样本数。`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 默认均为 0；SGPD 脚本的 `PATCH_WEIGHT` 默认为 0.1。`INIT_ADAPTER_DIR` 只接受天气、checkpoint、预处理及模型类型一致的已有结果。

`evaluate_svox.py` 默认在 `runs/cache/` 保存经过 checkpoint、预处理和图库图像字节校验的**原始白天**描述符缓存。全参数 Night、LoRA 和 Output 消融都可复用它。只检查测试集协议可运行 `"$PYTHON" scripts/evaluate_svox.py --check-data`。

可传 `--split train` 使用 SVOX 的独立 train/day Gallery 与 train/night Query 诊断真实夜景；该 split 有独立的 Gallery 缓存，默认 test 结果不会被覆盖。`--adapter-scale` 只适用于旧 LoRA 实验，不适用于全参数 Night 主干。

`evaluate_pairs.py` 使用 385 对未参与训练的 Qwen 合成夜图与其精确白天源图，报告同地点 Recall@1/5/10 和平均配对描述符余弦相似度。它与真实 SVOX 测试分开。`--limit-pairs 8` 可做短流程检查，输出明确标记为子集。

`configs/night_full_real.sh` 使用 SVOX train 的真实夜图，以**冻结的原始白天 Gallery**作多正样本检索目标：25 米内为正样本，25–250 米不参与损失，250 米外为负样本。默认 batch 4、FP32、3 epoch、末层学习率 1e-6、逐层衰减 0.7、同夜图冻结 Teacher 描述符约束权重 100；全部 Night 参数仍有非零学习率。`NIGHT_BASELINE_WEIGHT=0` 可关闭约束，`LAYERWISE_LR_DECAY=1` 可关闭逐层衰减，`BF16=1` 可启用混合精度。真实图没有逐 Patch 可信对应，该阶段不计算 SGPD。

真实阶段默认从 test Gallery 与 Query 的**文件名坐标**建立 250 米排除带，在剩余 train 夜图中按空间位置留出内部验证集并保存最佳 epoch。训练不读取 test 图像内容。直接从 Day 初始化时省略 `INIT_ADAPTER_DIR`；有效初始化和精度写入 `adapter.json`，应以元数据确认训练来源。

## 独立消融

`configs/night_full_sgpd.sh` 接受环境变量覆盖各损失和训练预算；每次设置新的 `OUTPUT`。比较损失时请使用相同 split、seed、epoch、起始 Night 权重与 Gallery 缓存。旧 `configs/night_global.sh` 和 `configs/night_sgpd.sh` 仍支持 LoRA/Output 消融。

```bash
# 阶段 1 的 Global + queue InfoNCE 本身是无 SGPD 对照
"$PYTHON" scripts/evaluate_pairs.py --adapter-dir runs/night_full_global_01 \
  --device cuda --output runs/night_full_global_01/synthetic_val.json

# 同一可信对应，但 Patch 权重均匀
OUTPUT=runs/night_full_uniform_01 INIT_ADAPTER_DIR=runs/night_full_global_01 \
  IMPORTANCE_MODE=uniform bash configs/night_full_sgpd.sh

# 额外开启 Keep Alignment；OT / Magnitude 也可分别设置非零权重
OUTPUT=runs/night_full_keep_01 INIT_ADAPTER_DIR=runs/night_full_global_01 \
  KEEP_WEIGHT=0.01 bash configs/night_full_sgpd.sh

# 旧输出端 Adapter 的公平消融：两阶段都选 output，使用独立目录
OUTPUT=runs/night_output_global_01 ADAPTER_TYPE=output bash configs/night_global.sh
OUTPUT=runs/night_output_sgpd_01 ADAPTER_TYPE=output \
  INIT_ADAPTER_DIR=runs/night_output_global_01 bash configs/night_sgpd.sh

# 旧 LoRA 对照
OUTPUT=runs/night_lora_new_01 ADAPTER_TYPE=lora bash configs/night_global.sh
```

可用 `PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT`、`GLOBAL_WEIGHT`、`CONTRAST_WEIGHT`、`IMPORTANCE_MODE`、`BATCH_SAMPLING`、`CONTRAST_QUEUE_SIZE`、`BATCH_SIZE`、`GRADIENT_ACCUMULATION_STEPS`、`LEARNING_RATE`、`WEIGHT_DECAY`、`EPOCHS`、`MAX_STEPS`、`SEED`、`BF16`、`DEVICE`、`PYTHON` 等变量调整。训练日志、Night DINOv2 权重和配置保存在各自 `runs/` 目录；白天完整 checkpoint 保持一份。

## 目录

```text
configs/                  night 命令、校准及排除地点
data/train/               已核验的原图、生成图和配对清单
data/splits/              地点隔离的训练/合成验证清单
data/svox/                真实天气测试图
src/weather_vpr/models/  Day Teacher、完整 Night DINOv2、LoRA / Output 消融
src/weather_vpr/losses/  SGPD、可选 Keep/OT/Magnitude、全局和 queue InfoNCE
src/weather_vpr/data/    精确配对、地点分散采样和可信 Patch 对应
src/weather_vpr/inference/ SVOX 检索与描述符导出
scripts/                  数据准备、训练、评估入口
third_party/              项目内部的 SALAD/DINOv2 源码
weights/                  原始完整 checkpoint
runs/                     实验输出和 Gallery 缓存
```

方法边界与评估协议见 [设计说明](docs/ARCHITECTURE.md)、[数据和评估](docs/DATA_AND_EVALUATION.md)。`assignment_keep` 表示 Sinkhorn 有效簇分配质量，不等同于建筑语义标签；完整训练和真实夜景 Recall 才能验证实际收益。

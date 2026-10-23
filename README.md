# WeatherVPR：Night 专属 DINOv2 + 冻结 SALAD

本项目从白天训练好的 `dino_salad.ckpt` 初始化**独立的 Night DINOv2-B/14**，全参数微调 Night 主干；原始白天 DINOv2 作为冻结 Teacher，SALAD 聚合器仍由 Day/Night 共用且冻结。真实白天图与其 Qwen 生成夜图构成训练图对。SGPD 只在可信 Patch 对应上蒸馏局部特征，并用白天 SALAD 的 `assignment_keep` 加权。Night 描述符直接检索**原始白天 SALAD Gallery**，无需重建图库。

参考论文 [*Distillation Improves Visual Place Recognition for Low Quality Images*](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Yang_Distillation_Improves_Visual_Place_Recognition_for_Low_Quality_Images_ICCVW_2025_paper.html) 的 Teacher–Student 思路；本实现的 SGPD 和跨域 InfoNCE 是新的实验设计。先前的 LoRA 和输出端 Patch/CLS Adapter 均保留为消融基线；现在 `--adapter-type full` 是默认配置。Night DINOv2 有 86,580,480 个可训练参数，单独的权重文件约 346 MB；其它天气模型尚未训练。

## 当前就绪状态

- `weights/dino_salad.ckpt`、`third_party/salad/`、`third_party/dinov2/` 已在项目内；运行时不查找外部 WORKSPACE。
- `data/train/manifest.jsonl` 是 7,719 对 SHA-256 核验过的 night/day 精确配对。`data/splits/night_train.jsonl` 有 7,263 对训练图，`night_val.jsonl` 有 385 对地点隔离的合成验证图；历史保留地点另排除 71 对。
- `configs/dino_salad_stage2b_calibration.json` 与当前 checkpoint 和 224×224 预处理匹配。相应地点排除已纳入 split；SGPD 的 A 类可信 Patch 才参与局部损失。
- SVOX 仅用于真实夜景检索评估：test 有 17,166 张白天 Gallery 和 823 张夜间 Query；`--split train` 可对独立的 22,232 张白天 Gallery、702 张夜间 Query 做只读诊断。评估采用 25 m UTM 正样本协议。
- 已完成的 LoRA 对照保存在 `runs/night_lora_global_01` 和 `runs/night_lora_sgpd_01`；`runs/night_sgpd_01` 是旧 Output Adapter 的独立结果。全参数合成基线使用 `runs/night_full_global*` 与 `runs/night_full_sgpd*`；相机增强的三组配对训练保存在 `runs/night_camera_20261023/`。

### Night 专用目标（2026-10-24）

Student 的职责是夜间 Query→原始 Day Gallery 检索。新增 `configs/night_specialist.sh` 默认关闭 Synthetic Guidance 和 Day Preservation；白天照片的辅助用途是产生强制低照度视图，再由 Night Student 检索干净 Day Teacher 特征库。Student 不必保持干净白天输入的表现。旧正则保留作独立消融，其价值以 Night Recall 判断。

两组同预算实验已完成。所选 `runs/night_specialist_20261024/source_night` 在 SVOX test 的 R@1/5/10 为 **93.07% / 98.66% / 99.03%**，原始 Day 为 92.71% / 98.42% / 99.15%。R@1 多命中 3 张、R@5 多 2 张、R@10 少 1 张，属于小幅且不一致的收益。训练仅使用 Day/Qwen，Gallery 模型和特征库不变。

[目标与损失说明](docs/NIGHT_SPECIALIST.md) · [完整实验与复现](docs/NIGHT_SPECIALIST_RESULTS.md)

### 相机增强配对训练（2026-10-23）

| 配置 | 原图合成验证 R@1 | 平均配对描述符余弦 |
| --- | ---: | ---: |
| 原始白天 SALAD | 384/385 | 0.6778 |
| 增强 Global，`global_01` | 385/385 | 0.7295 |
| 增强 Global→Global，`global_02` | 385/385 | 0.7358 |
| 增强 Global→SGPD，`sgpd_02` | 385/385 | 0.7359 |

新增曝光/Gamma、光晕过曝、运动模糊、泊松高斯噪声、JPEG、降饱和、暗角和随机光度策略，只增强训练 Student；Teacher、可信对应、验证/test 与原白天 Gallery 使用原图。每项概率和范围可以独立消融，见 [增强策略](docs/NIGHT_CAMERA_AUGMENTATION.md) 和 [预览](runs/night_camera_20261023/augmentation_preview.jpg)。

Global 接续和 SGPD 接续从相同权重出发，训练预算与增强采样一致。合成验证已接近饱和，配对余弦的差异不能证明真实夜景检索收益。原始 Day SALAD 在 SVOX test 的 R@1/5/10 为 **92.71% / 98.42% / 99.15%**；当前不宣称相机增强或 SGPD 已带来稳定的真实域收益。完整数字见 [相机增强实验记录](docs/NIGHT_CAMERA_RESULTS.md)、[全参数合成基线](docs/NIGHT_FULL_RESULTS.md) 和 [评估协议](docs/DATA_AND_EVALUATION.md)。

当前目录所需的模型、源码和图像均已具备，**无需再下载文件**。另一台机器上的资源来源见 [下载与来源](docs/DOWNLOADS.md)。

### 面向 Recall 的新增实验

新增 [检索指导方案](docs/GUIDED_RETRIEVAL.md)：用 7,263 个训练白天描述符（6,441 个地点）组成固定库，提供多正样本难负 InfoNCE 和排除同地点的 Teacher 检索关系 KL；增加干净 Qwen 的冻结指导与训练 Day 知识保留。训练仅使用 Qwen 图对与训练白天图，Night 主干全参数更新，原 Day Teacher、SALAD 和 Gallery 保持冻结。

该方案参考本地 GCISG 与低质 VPR 蒸馏论文，并根据原 Day Gallery 约束改造；原论文方法、当前实现及差异见上述说明。采用小 batch 梯度累积、activation checkpoint、逐层学习率衰减与 cosine warmup；各损失独立开关。固定相机退化与训练 Day distractors 的精确配对验证仍接近饱和，因此增加排除精确源图的跨视角检查。

另有 `configs/night_guided_restore.sh` 候选，强化干净 Qwen 的冻结指导，并训练 Night 主干从训练白天原片的相机退化视图恢复干净 Day 描述符。`DAY_VIEW_AUGMENTATION` 独立控制该视图；Teacher 和可信匹配始终干净。辅助验证可缓存冻结 Day Gallery，并自动报告排除精确源图的跨视角子集（当前 90 Query / 45 地点），帮助识别精确配对验证饱和。方法、限制及命令见 [检索指导方案](docs/GUIDED_RETRIEVAL.md)。

两组完整实验已完成，开发集选择 `runs/night_guided_20261023/restore_01`。以下均为原始 224 输入及同一 Day Gallery，无多尺度增强：

| 模型 | SVOX 开发 R@1 | test R@1 / R@5 / R@10 |
| --- | ---: | --- |
| 原始 Day | 88.46% | 92.71% / 98.42% / 99.15% |
| 新 `restore_01` | 89.17% | 92.71% / 98.54% / 99.27% |

**test R@1 尚未突破**；R@5、R@10 各增加 1 张命中，不能称为稳定提升。两组配置、跨视角验证、错误变化与权重路径见 [实测记录](docs/GUIDED_RETRIEVAL_RESULTS.md)。

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
OUTPUT=runs/night_guided_retrieval_01 bash configs/night_guided_retrieval.sh
```

新 Python 环境先安装与本机驱动兼容的 PyTorch，再执行 `pip install -r requirements-extra.txt`。无需安装 DINOv2 仓库中针对旧 CUDA 版本固定的整套依赖。

## 运行

以下命令从项目根目录执行。本机 `base` 环境的 PyTorch CUDA 构建与驱动不兼容；下列 `PYTHON` 指向已验证可用的 `a2b-lightvpr` 环境。每次训练必须使用新的 `OUTPUT` 目录。

相机增强配对训练重跑（三组训练及原图合成验证）：

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_02 bash configs/night_camera_replay.sh
```

以下是无增强基线和独立阶段命令。给训练命令添加 `NIGHT_AUGMENTATION=camera AUGMENTATION_CONFIG=configs/night_camera.json` 可启用相机增强；原有入口默认 `off`，方便复现对照。

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

# 真实夜景检索评估：与未适配 SALAD baseline 共用原始白天 Gallery
"$PYTHON" scripts/evaluate_svox.py --adapter-dir "runs/${EXPERIMENT}_sgpd" \
  --device cuda --batch-size 16 --num-workers 2 \
  --output "runs/${EXPERIMENT}_sgpd/svox_test.json"
```

两个合成阶段脚本默认使用 batch 8、BF16、AdamW、学习率 1e-6，各训练 1 epoch，优化 Night DINOv2 的所有参数（含 Patch Embedding、位置/CLS Token、全部 Block 与 Norm）。白天 DINOv2 和 SALAD 始终冻结。地点分散 batch 与容量 256 的 detached 白天描述符队列提供跨域多正样本 InfoNCE；同一 `(city, place_id)` 的多个白天描述符不会变成伪负例。训练日志记录有效负样本数。`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 默认均为 0；SGPD 脚本的 `PATCH_WEIGHT` 默认为 0.1。`INIT_ADAPTER_DIR` 只接受天气、checkpoint、预处理及模型类型一致的已有结果。

`evaluate_svox.py` 默认在 `runs/cache/` 保存经过 checkpoint、预处理和图库图像字节校验的**原始白天**描述符缓存。全参数 Night、LoRA 和 Output 消融都可复用它。只检查测试集协议可运行 `"$PYTHON" scripts/evaluate_svox.py --check-data`。

可传 `--split train` 使用 SVOX 的独立 train/day Gallery 与 train/night Query 诊断真实夜景；该 split 有独立的 Gallery 缓存，默认 test 结果不会被覆盖。`--adapter-scale` 只适用于旧 LoRA 实验，不适用于全参数 Night 主干。

`evaluate_pairs.py` 使用 385 对未参与训练的 Qwen 合成夜图与其精确白天源图，报告同地点 Recall@1/5/10 和平均配对描述符余弦相似度。它与真实 SVOX 测试分开。`--limit-pairs 8` 可做短流程检查，输出明确标记为子集。

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

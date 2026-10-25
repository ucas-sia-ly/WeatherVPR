# WeatherVPR：Night 专属 DINOv2 + 冻结 SALAD

本项目从白天训练好的 `dino_salad.ckpt` 初始化**独立的 Night DINOv2-B/14**，默认直接微调其最后 **4 个 Transformer Block** 的原始权重。原始 Day DINOv2 与共享 SALAD 保持冻结；Night 描述符检索**原始白天 SALAD Gallery**，无需重建图库。该方案使用独立 Night 主干，不使用 LoRA。

参考论文 [*Distillation Improves Visual Place Recognition for Low Quality Images*](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Yang_Distillation_Improves_Visual_Place_Recognition_for_Low_Quality_Images_ICCVW_2025_paper.html) 的 Teacher–Student 思路；本实现增加可信 Patch 上的 SGPD 和跨域多正样本 InfoNCE。旧 LoRA、Output Adapter 和全部参数微调仍可作为独立消融。

## 当前方案

```text
Day Gallery ── 原始冻结 Day DINOv2 ── 冻结 SALAD ── 固定 Day 描述符
Night Query ── 独立 Night DINOv2   ── 同一 SALAD ── 检索上述描述符
                默认更新 blocks 8–11
```

- 默认 `--adapter-type full --full-train-scope late --full-train-blocks 4`，更新 **28,357,632** 个参数。Patch Embedding、位置/CLS Token、前八层和最终 Norm 冻结；最终 Norm 与 SALAD 允许梯度传回可训练层。
- `full` 表示保存独立完整 Night 主干，约 346 MB；是否更新全部参数由 `full-train-scope` 决定。显式 `--full-train-scope all` 可训练全部 86,580,480 个参数。
- `configs/night_specialist.sh` 使用 Qwen Night 主分支，以及由训练 Day 原图产生的强制低照度辅助视图。干净 Day 只供冻结 Teacher 和 Day bank 使用，默认真实 Night 仅用于评估。新增可选 [真实配对阶段一](docs/NIGHT_REAL_PAIRS.md)：`night_real_stage1.sh` 保留原主分支，加入审核过的 DarkDriving 全局蒸馏，并提供 A/B/C/D 对照。
- 默认 Global / Contrast / Relation 权重为 `0.1 / 0.1 / 0.1`，Source Night 权重 `0.5`；SGPD / Keep / OT / Magnitude 分别独立控制，默认关闭。推理默认只用 Night Query 描述符；融合保留为可选诊断。
- 默认 AdamW、LR `1e-6`、`LAYER_DECAY=1`、BF16、activation checkpoint、batch 8、1 epoch、`NUM_WORKERS=0`。SALAD 原始训练设置与当前设置的区别见 [SALAD 训练设置](docs/SALAD_TRAINING.md)。

### 已有结果

以下使用相同原始 Day Gallery 与 224×224 输入，单位为整数命中数；它们是保留的历史结果，未因默认值调整而重新训练。

| 模型 | SVOX 开发 R@1 / R@5 / R@10（702 Query） | SVOX test R@1 / R@5 / R@10（823 Query） |
| --- | --- | --- |
| 原始 Day | 621 / 668 / 674 | 763 / 810 / 816 |
| `source_night`，历史全部参数微调 | 620 / 668 / 677 | 766 / 812 / 815 |
| 用户 `night_late4_01`，后四层微调 | 618 / 668 / 676 | 761 / 812 / 816 |

`source_night` 的 test R@1 为 **93.07%**，原始 Day 为 **92.71%**；R@10 略降。用户后四层结果的 test R@1 为 **92.47%**。当前将后四层设为后续训练起点，不将这些数字解释为其已全面超过全部参数微调。保留用户权重及完整训练、开发、测试记录。Source Night 的实验与复现见 [实测结果](docs/NIGHT_SPECIALIST_RESULTS.md)。

其他夜间评估支持 Tokyo247 官方 Night 查询子集和 RobotCar v2 公开位姿诊断集。历史 `source_night` 在 Tokyo247 上退化、RobotCar 上小幅改善，尚无一致跨数据集收益；协议与完整数字见 [其他夜间评估](docs/EXTERNAL_NIGHT_RESULTS.md)。

真实配对阶段一 seed 42 的 [A/B/C 首轮结果](outputs/stage1_results_20261025/report.md) 仍未取得稳定跨数据集提升。参考 TFS-ViT 的 [部分 Patch 模块](docs/NIGHT_TOKEN_STAGE2.md) 已完成 T0–T3 首轮，增强前后的开发 R@1/5/10 与命中集合完全相同，当前保持默认关闭，见 [阶段二结果评价](docs/NIGHT_TOKEN_STAGE2_RESULTS.md)。[论文及源码分析](docs/TOKEN_STYLE_DG_REVIEW.md) 与 [研究计划](idea.md) 保留真实统计供体和伪域采样的后续假说，不将工程检查通过解释为 Recall 增益。

### 可信 Patch 与 DINO 注意力实验

新增 `configs/night_patch.sh`：每张训练 Qwen 图都经过相机退化，再在 Stage 2B A 类对应上监督 Student Patch。可见比例只对黑场和全通道过曝区域降权，不创建对应。独立的 `ATTENTION_WEIGHT` 对齐最后一个 DINOv2 Block 的 CLS→Patch 注意力分布，Teacher 目标与权重冻结，区别于 SALAD Keep。

`PATCH_WEIGHT=0.1 ATTENTION_WEIGHT=0.05` 同时启用两项；各自设为 `0` 可关闭，`PATCH_VISIBILITY=off` 可关闭可靠性降权。本轮按相同后四层配置、数据和预算比较全局对照、SGPD、SGPD+Attention；方法、固定选择规则和受限内存命令见 [Patch 与注意力蒸馏](docs/NIGHT_PATCH_ATTENTION.md)。

三组完整训练与评估已完成：SVOX test 的 R@1 都为 **759/823（92.22%）**，低于原始 Day 的 763/823；SGPD 的 R@10 相对同设置全局对照增加 1 张，注意力项没有增加 Top-1。**不将这组强制增强或局部项改为默认。** 183 项测试与冻结参数检查通过，实测任务内存峰值 5.00 GiB。完整三项 Recall、权重和注意力图见 [本轮实测结果](docs/NIGHT_PATCH_ATTENTION_RESULTS.md)。

随后已复现六个模型的全部 **4,938 个测试排名**，对照分析 23 张图片并导出真实注意力、Keep、精确有符号检索贡献和固定遮挡结果。Attention 失败但旧方案成功的 17 张中，16 张在加入 Attention 前就失败；当前条件 KL 也不约束可信区域的总注意力质量。具体图片、量化证据及解释限制见 [失败原因诊断](docs/NIGHT_FAILURE_ANALYSIS.md)。此次未重新训练，190 项测试通过，诊断任务峰值 1.40 GiB。

## 数据与独立工程

- `weights/dino_salad.ckpt`、`third_party/salad/`、`third_party/dinov2/` 已在项目内；运行时不查找外部 WORKSPACE。
- `data/train/manifest.jsonl` 有 7,719 对经 SHA-256 核验的 Day/Qwen Night 图；地点隔离训练集为 7,263 对，合成验证集 385 对，另排除历史保留地点 71 对。
- Stage 2B 校准绑定当前 checkpoint 与 224×224 预处理；SGPD 仅使用 A 类可信对应。Qwen 编辑不能保证逐 Patch 同坐标匹配。
- SVOX test：17,166 张 Day Gallery、823 张 Night Query；独立开发划分：22,232 张 Day Gallery、702 张 Night Query。检索采用 25 m UTM 正样本协议。
- DarkDriving_Base 已清理为 9,520 对真实日夜图像，提供相对路径清单、独立加载器及全量校验命令，见 [使用说明](docs/DARKDRIVING.md)。
- 已保留完整的全部参数、相机增强、LoRA、Output 基线；无效分支及其重复训练输出已清理。数据、基础权重和可复用 Gallery/训练 bank 缓存保留。

当前运行所需资源已具备，**无需新增下载**。其它机器上的来源见 [下载与来源](docs/DOWNLOADS.md)。

## 32 GB 主机运行

本机为 4090 48 GiB 显存；优先吞吐时使用 [阶段一加速入口](configs/night_real_stage1_fast.sh)，白天库驻显存、关闭重计算并启用 4 个数据 worker。当前后台任务使用 7 GiB RAM 上限和 2 GiB 系统余量；具体选项与实测见 [吞吐配置](docs/SALAD_TRAINING.md)。以下为原保守默认运行方式。

训练、测试、评估**逐项串行执行**。`scripts/run_bounded.py` 设置 8 GiB cgroup RAM 硬上限、6 GiB 回收阈值和禁用任务 swap；启动要求系统可用内存至少 12 GiB，运行中低于 4 GiB 会结束当前任务。它还使用进程锁防止并行实验。完整约束见 [SALAD 训练与内存设置](docs/SALAD_TRAINING.md)。

从项目根目录执行，每次训练使用新的 `OUTPUT`。本机使用以下已验证的环境；新环境先安装驱动兼容的 PyTorch，再安装 `requirements-extra.txt`。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OUTPUT=runs/night_late4_02

"$PYTHON" scripts/run_bounded.py --log "$OUTPUT.memory.jsonl" \
  -- bash configs/night_specialist.sh

# 训练完成且已保存 adapter.safetensors / adapter.json 后再执行
"$PYTHON" scripts/run_bounded.py --log "$OUTPUT.svox_train.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir "$OUTPUT" --device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/svox_train_eval_day.safetensors \
  --output "$OUTPUT/svox_train.json"

# 固定开发阶段选择的配置后，再评估 test
"$PYTHON" scripts/run_bounded.py --log "$OUTPUT.svox_test.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_svox.py --adapter-dir "$OUTPUT" \
  --device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/svox_test_gallery.safetensors \
  --output "$OUTPUT/svox_test.json"
```

原始 Day Gallery 缓存校验 checkpoint、预处理、有序 ID 和图像字节，不因更换 Night 权重而失效。默认训练不使用真实夜图；显式阶段一仅将审核训练图对用于额外全局蒸馏，不加入 GSV 负样本 bank。test 的历史结果已被查看，后续实验属于探索性比较，不应据 test 反复选配置。

## 损失与消融

`night_specialist.sh` 接受 `GLOBAL_WEIGHT`、`CONTRAST_WEIGHT`、`RELATION_WEIGHT`、`SOURCE_NIGHT_WEIGHT`、`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT`、`ATTENTION_WEIGHT` 等独立权重。SGPD 使用 Stage 2B 可信对应和 `positive_conf × teacher.assignment_keep`，支持 `IMPORTANCE_MODE=saliency|uniform`；未知、无效和不可靠 Patch 权重为零。

```bash
# 与后四层默认方案比较全部参数微调
OUTPUT=runs/night_all_02 FULL_TRAIN_SCOPE=all \
  "$PYTHON" scripts/run_bounded.py --log runs/night_all_02.memory.jsonl \
  -- bash configs/night_specialist.sh

# 后四层方案中加入 SGPD，另存目录
OUTPUT=runs/night_late4_sgpd_02 PATCH_WEIGHT=0.1 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_late4_sgpd_02.memory.jsonl \
  -- bash configs/night_specialist.sh
```

比较时固定数据、初始化、seed、训练预算与增强。`configs/night_full_global.sh` / `night_full_sgpd.sh` 显式保留**全部参数微调**的历史两阶段消融；其 Global / Contrast 默认各 1，SGPD 阶段 Patch 默认 0.1，Keep / OT / Magnitude 默认 0。旧 `night_global.sh` / `night_sgpd.sh` 支持 LoRA 和 Output 消融。详见 [配置说明](configs/README.md)。

相机增强包含曝光/Gamma、光晕过曝、运动模糊、泊松高斯噪声、JPEG、降饱和、暗角和随机光度操作；只作用于 Student。Teacher、可信匹配输入、评估及原始 Day Gallery 均保持干净。参数与历史对照见 [增强策略](docs/NIGHT_CAMERA_AUGMENTATION.md)、[相机增强结果](docs/NIGHT_CAMERA_RESULTS.md)、[全部参数基线](docs/NIGHT_FULL_RESULTS.md)。合成验证接近饱和，不能以配对余弦提高代替真实 Night Recall。

## 目录

```text
configs/                   训练入口、增强、校准及排除地点
src/weather_vpr/models/    独立 Night 主干、Day Teacher、LoRA / Output 消融
src/weather_vpr/losses/    SGPD、Keep/OT/Magnitude、全局和多正样本 InfoNCE
src/weather_vpr/data/      精确配对、地点分散采样和可信对应
src/weather_vpr/inference/ SVOX、Tokyo247、RobotCar 检索与描述符导出
scripts/                   数据准备、受限内存运行、训练和评估
third_party/               项目内部 SALAD / DINOv2 源码
weights/                   原始白天完整 checkpoint
runs/                      独立 Night 权重、有效实验和缓存
```

方法与协议见 [架构](docs/ARCHITECTURE.md)、[数据与评估](docs/DATA_AND_EVALUATION.md)。`assignment_keep` 表示分配到 SALAD 有效簇的质量，不等同于建筑物语义标签。

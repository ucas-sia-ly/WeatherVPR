# 阶段一：额外真实昼夜配对监督

本实现对应 [idea.md](../idea.md) 阶段一，支持真实配对辅助和匹配的合成辅助对照。默认 `night_specialist.sh` 仍关闭新增分支；显式执行 `night_real_stage1.sh` 才选择 A/B/C/D。seed 42 首轮 A/B/C 训练及固定图库评估已完成，见 [首轮结果](../outputs/stage1_results_20261025/report.md)；多 seed 复核仍待完成。

## 模型与优化

从同一原始 `dino_salad.ckpt` 初始化独立 Night DINOv2，只训练 blocks 8–11。Day Teacher、最终 Norm 和共享 SALAD 参数冻结，Student 仍通过冻结 Norm/SALAD 反传。真实图对不进入 GSV 负样本 bank，不生成虚构 `place_id`。

每个主 microbatch 默认 8 个 Qwen 视图及 2 个 Source Night 视图；B/D 额外抽取 2 个真实 Night，C 额外抽取 2 个 GSV/Qwen 视图。真实 Night 只做 bilinear + ImageNet 预处理，不叠加强制低照度退化。各分支依次前向并反传，最后统一裁剪和更新。

```text
L_GSV = 0.1 Global_Q + 0.1 InfoNCE_Q + 0.1 Relation_Q
      + 0.5 (0.1 Global_Source + 0.1 InfoNCE_Source)

L_total = L_GSV + auxiliary_weight × mean(1 − cosine(Student_extra, Teacher_day))
```

额外分支只计算 FP32 配对全局余弦，不加 InfoNCE、Relation、Patch 或 Attention。B/D 目标为对应真实 Day，C 目标为对应 GSV Day。每项分别取有效样本均值，不再次除以视图总数或分支数。累积 K 时各分支统一除以 K，短尾窗口修正后更新并正确写入日志。

真实流按审核的 `scene_group` 均衡循环，C 按既有 `(city, place_id)` 分组均衡循环；训练长度继续由主 GSV 流决定。辅助数据、前向和反向重算使用独立 Python/NumPy/Torch RNG；主/真实 Day 缓存的 DataLoader 也使用独立 generator，避免冷热缓存扰动主流。

## 审核清单

清理数据见 [DarkDriving](DARKDRIVING.md)。生成未审核模板（Fish Shell）：

```fish
set -gx PYTHON /home/admin123/miniconda3/envs/a2b-lightvpr/bin/python

"$PYTHON" scripts/prepare_real_pair_review.py \
    --root data/DarkDriving_Base \
    --output data/splits/darkdriving_review.template.jsonl
```

本机已生成上述模板，9,520 行、接受数为零；脚本拒绝覆盖已有文件。可用 `--sample-ids <txt>` 生成明确 ID 子集。

保留模板，复制到新的审核文件并填写：

| 字段 | 规则 |
| --- | --- |
| `sample_id` | 必须存在于清理后的 source manifest |
| `source_manifest_sha256`、`day_sha256`、`night_sha256` | 模板自动填写，必须与来源及实际字节一致 |
| `scene_group` | 可靠路线/近重复审核确定，不能直接由文件名前缀推断地点 |
| `split` | train / dev / test，一个连通分组只能分配一个集合 |
| `pair_valid` | 布尔值，背景与视角是否适合配对监督 |
| `visibility_valid` | 布尔值；困难夜景不自动无效，近乎无信息情况单独审核 |
| `review_status` | manual / automatic / unreviewed；仅实际人工核查写 manual |
| `reject_reason` | 接受项为 null；审核后拒绝项必须注明原因 |
| `difficulty`、`visibility_status`、`review_notes` | 可选分层和证据记录，不改变基本准入规则 |

仅 train、两个有效性均 true、状态为 manual/automatic 的记录进入真实训练。automatic 不等于人工认证。未审核模板不能直接训练，加载器在构模前检查清单。

加载器合并 scene、content_group、文件/像素哈希的复用关系后检查跨 split 冲突，包括经未审核或拒绝样本桥接的冲突，并核验审核图像字节。**这些检查不证明没有空间近重复**；路线分组仍需可靠审核，不能随机分帧。真实训练不把不同文件当作负地点。

不加载模型的数据检查（Fish Shell）：

```fish
env PYTHONPATH=src "$PYTHON" -c '
from weather_vpr.data.real_pairs import ReviewedRealPairs

pairs = ReviewedRealPairs(
    "data/DarkDriving_Base",
    "data/splits/darkdriving_review.jsonl",
)

print("accepted training pairs:", len(pairs))
print("scene groups:", len({r["scene_group"] for r in pairs.rows}))
print(pairs.sources_signature()["review_manifest_sha256"])
'
```

Fish 不支持 Bash 的 `<<'PY'` heredoc，因此这里改用 Python 的多行 `-c` 参数，执行逻辑不变。

## A/B/C/D 运行

本机 4090 48 GiB 优先使用 `night_real_stage1_fast.sh`。它只调整执行开销：CUDA 常驻 bank、关闭重计算、紧凑 Tensor 预取、独立 Teacher 缓存提取批量；不会自动改变有效训练批量和预算。资源设置与实测见 [本机吞吐配置](SALAD_TRAINING.md)。下方保留保守默认入口示例。

| STAGE1_ARM | 额外分支 | 权重 |
| --- | --- | --- |
| A | 无 | 0 |
| B | 2 个真实配对 | 0.1 |
| C | 2 个合成配对全局对齐 | 0.1 |
| D | 与 B 相同的真实流 | 0.03 |

以下逐条串行运行，每次使用新目录。默认内存保护需 12 GiB 可用 RAM；保留保护和系统余量。

Fish Shell 环境设置：

```fish
set -gx PYTHON /home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
set -gx OMP_NUM_THREADS 4
set -gx MKL_NUM_THREADS 4
set -gx SEED 42
```

**A：无额外辅助分支**

```fish
env STAGE1_ARM=A OUTPUT=runs/night_real_A_seed42 \
    "$PYTHON" scripts/run_bounded.py \
    --log runs/night_real_A_seed42.memory.jsonl \
    -- bash configs/night_real_stage1.sh
```

**B：真实昼夜配对辅助，权重 0.1**

```fish
env STAGE1_ARM=B \
    OUTPUT=runs/night_real_B_seed42 \
    REAL_PAIRS_REVIEW=data/splits/darkdriving_review.jsonl \
    "$PYTHON" scripts/run_bounded.py \
    --log runs/night_real_B_seed42.memory.jsonl \
    -- bash configs/night_real_stage1.sh
```

**C：合成配对全局对齐，权重 0.1**

```fish
env STAGE1_ARM=C OUTPUT=runs/night_real_C_seed42 \
    "$PYTHON" scripts/run_bounded.py \
    --log runs/night_real_C_seed42.memory.jsonl \
    -- bash configs/night_real_stage1.sh
```

**D：真实昼夜配对辅助，权重 0.03**

```fish
env STAGE1_ARM=D \
    OUTPUT=runs/night_real_D_seed42 \
    REAL_PAIRS_REVIEW=data/splits/darkdriving_review.jsonl \
    "$PYTHON" scripts/run_bounded.py \
    --log runs/night_real_D_seed42.memory.jsonl \
    -- bash configs/night_real_stage1.sh
```

B/D 默认根目录 `data/DarkDriving_Base`，独立目标缓存 `runs/cache/stage1_real_day.safetensors`，可由 `REAL_PAIRS_ROOT`、`AUXILIARY_DAY_CACHE` 覆盖。C 复用原固定 GSV bank。C 额外分支仅与 B 匹配全局损失形式及权重，不增加 InfoNCE/Relation/Source Night；它增加合成曝光，不能称完全隔离计算量的对照。

入口禁止 `INIT_ADAPTER_DIR`，固定 Late4 并关闭局部项。其它环境变量可改预算及权重，但不同组必须同步固定主配置，修改后需如实记录。

底层 CLI：

- `--auxiliary-mode none|real|synthetic`
- `--auxiliary-weight`
- `--auxiliary-samples`
- `--real-pairs-root`
- `--real-pairs-review`
- `--auxiliary-day-cache`
- `--auxiliary-gradient-every`

none 或零权重严格旁路，不读真实数据、不额外前向。零有效样本跳过并计数；连续三次为空时报数据管线异常。

## 缓存与日志

真实 Day 目标缓存不带地点标签，绑定 checkpoint、预处理、source/review 清单、split、有序 ID 和图像哈希。safetensors 有独立 SHA-256 与 JSON 签名，身份变化或损坏时重建。

- `experiment_plan.json`：首个更新前保存预算、来源及模型/代码指纹。
- `training_config.json`：解析参数与有效辅助配置。
- `training_log.jsonl`：原有损失、额外原始/加权损失、有效数、采样 ID/组、主样本索引和分域视图账目。
- `adapter.json`：辅助来源、缓存签名、组曝光/重复次数、主流顺序摘要及实际更新数。
- `adapter.safetensors`：仅保存独立 Night 主干，旧权重加载兼容。

入口默认每 100 个 updates 在第一个主 microbatch 抽样检查梯度，直接 CLI 默认关闭。分支前后 `.grad` 差扣除已有累积，GSV 包含加权 Source，记录裁剪前加权辅助范数、比值和方向余弦，再按 K 还原 microbatch 量级；零范数余弦为 null。

它不是整个 K 窗口的合计梯度，负余弦也不直接证明泛化受损。诊断有额外开销，需单列。

## 验证与解释范围

实现及吞吐优化共 235 项测试通过，覆盖独立均值/权重、K=1/2 及尾窗口、冻结目标/SALAD、主随机流、缓存冷热/损坏/失效、实际 B/C 读取、审核隔离、入口、旧权重兼容和数据加载优化。

A/B/C 实际模型训练与 B 推理回读均完成，见 [实测验证报告](../outputs/stage1_implementation/report.md)。原始记录保存在 `runs/stage1_implementation_checks/`，仅验证工程，不提供 Recall 增益结论。

首轮使用 DarkDriving 整场地 train-only 分组，没有构造内部开发/测试集。A/B/C 均完成 908 次更新，主样本、Source Night 及其增强日志一致；B 暴露 1816 个真实样本，未遍历全部 9520 对。B 的 SVOX 开发 / test Top-1 为 622/702、764/823，相对 A 净增 5/4 张，但相对原始 SALAD 均只净增 1 张，Tokyo 和 RobotCar 尚无一致提升。正式实验仍需完成路线审核及相同多 seeds 比较；DarkDriving 缺少可信负地点时，内部配对排名只能作诊断。

外部评估沿用固定原始 Day Gallery，报告 R@1/5/10、整数命中数及修复/新增错误；已查看的测试集只作探索性比较，不据其反复调参。

## 参考的作者实现

除了本地 `third_party/salad/workflow/`，本次核对了固定版本的源码：

| 来源 | 采用范围 |
| --- | --- |
| [SALAD DINOv2](https://github.com/serizba/salad/blob/6aede13a3f6c25750bf7fde10209c06cb73060bb/models/backbones/dinov2.py) | 冻结前缀、可微后层；我们额外冻结聚合器以保持原 Gallery |
| [MixVPR 数据集](https://github.com/amaralibey/MixVPR/blob/4043915cef24818003ece1a8112bc8a24e69abe0/dataloaders/GSVCitiesDataset.py)、[聚合器](https://github.com/amaralibey/MixVPR/blob/4043915cef24818003ece1a8112bc8a24e69abe0/models/aggregators/mixvpr.py) | 分组采样，区分地点与图像计数，描述符归一化 |
| [CosPlace 数据集](https://github.com/gmberton/CosPlace/blob/52b56e95ea62245281281f3bafd7b9390d19a0fd/datasets/train_dataset.py)、[训练循环](https://github.com/gmberton/CosPlace/blob/52b56e95ea62245281281f3bafd7b9390d19a0fd/train.py) | 先均衡分组再抽图，固定迭代预算的循环流 |

未照搬 MultiSimilarity/CosFace 损失，也未把 DarkDriving 文件编号当地点类别；阶段一保持原检索约束，只新增配对全局蒸馏。

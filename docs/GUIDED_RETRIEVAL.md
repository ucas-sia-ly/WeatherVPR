# 面向 Recall 的合成训练改造

本轮优先解决两个问题：原图合成验证已经接近饱和，难以筛选模型；仅拟合配对描述符，未充分约束 Night Query 在原白天检索空间中的邻域结构。新增固定训练 Day 特征库、检索关系蒸馏和知识保留约束，配合相机退化与较保守的全参数优化。两组实验已完成：开发集 R@1 有所改善，最终 test R@1 持平，实测见 [实验结果](GUIDED_RETRIEVAL_RESULTS.md)。

训练只使用 `night_train.jsonl` 的训练白天图和 Qwen 夜图。原始 Day Teacher、SALAD 和白天 Gallery 均保持不变；真实夜图只用于独立检索评估。

## 论文依据与本项目的改造

页码均指 PDF 文件页码。

| 来源 | 原论文内容 | 本项目采用的改造 |
| --- | --- | --- |
| [Nam 等，GCISG，第 4–5 页](<../third_party/Nam 等 - 2022 - GCISG Guided causal invariant learning for improved syn-to-real generalization.pdf#page=4>) | 强风格增强产生两个视图；在线编码器与 EMA 目标编码器相对队列中的支持样本形成关系分布，通过双向交叉熵学习风格不变性。 | 使用冻结的白天 VPR Teacher 和固定训练 Day 特征库，蒸馏 Day→Day 与 Night→Day 的检索关系分布；不引入 EMA 编码器和原论文的双向视图目标。 |
| [GCISG，第 6–7 页](<../third_party/Nam 等 - 2022 - GCISG Guided causal invariant learning for improved syn-to-real generalization.pdf#page=6>) | 用冻结 ImageNet 预训练模型提供归一化池化特征指导，并提出自注意力池化，保留预训练知识。 | 用冻结 Day SALAD 对干净 Qwen 夜图产生描述符指导，并用训练白天图约束 Night 模型保留原描述符空间；这是描述符级改造，没有复现自注意力池化指导。 |
| [Yang 等，第 3–5 页](<../third_party/Yang 等 - Distillation improves visual place recognition for low quality images.pdf#page=3>) | 从同一 VPR 权重初始化 Teacher/Student，实验比较 ICKD、描述符 MSE 和 Triplet 的组合；主要退化为 JPEG quality=10。第 4 页的低质 Query 和 Gallery 均由 Student 处理，第 5 页表 1 的 DINOv2-SALAD 最优组合为 ICKD 单独使用。 | 保持 Teacher→原始白天 Gallery、Night Student→Query 的跨模型检索约束。借鉴蒸馏与知识保留思路，独立消融各损失；不直接照搬原论文权重，也不据其结果推断本项目必然获益。 |

SALAD Keep 是有效簇分配质量，不是建筑物语义标签。已有 SGPD 仍只使用 Stage2B A 类可信对应，不将同坐标 Patch 自动视为正确匹配。

## 新增训练目标

记 `T` 为冻结 Day DINOv2+SALAD，`S` 为全参数训练的 Night DINOv2+冻结 SALAD，`d` 为干净训练白天图，`q` 为其干净 Qwen 夜图，`a(q)` 为相机退化后的夜图。所有描述符均 L2 归一化，Teacher 目标全部 detached。

### 固定训练 Day 特征库与多正样本 InfoNCE

从训练 manifest 构建 **7,263 个 Day key，覆盖 6,441 个 `(city, place_id)`**。缓存默认位于 `runs/cache/qwen_training_day.safetensors`，不含合成验证或 SVOX 图像。缓存校验 checkpoint、预处理、manifest、地点标签、实际 Day 图像字节及张量 SHA-256；不一致时重建。

InfoNCE 的 Query 是 `S(a(q))`，key 来自固定库，不重复添加已在库中的当前 batch key。相同 `(city, place_id)` 的全部 key 都保留为正样本；不同地点才是负样本。`CONTRAST_HARD_NEGATIVES=64` 按 detached Student 相似度为每条 Query 选最多 64 个难负 key；设置为 `0` 使用全部候选负 key。多个训练图可属于同一地点，日志中的 key 数量不等于不同地点数量。

固定库使小 batch 也能看到足够的候选地点。记录候选负样本数、选中负样本数和 InfoNCE 损失，以判断检索监督是否有效。

### 检索关系 KL

对每个配对样本，从库中**排除同地点全部 key**，用 Teacher 的 `T(d)` 相似度选支持集 `H`，默认最多 64 个难负 key：

```text
pT = softmax(T(d) · T(day_key_H) / teacher_temperature)
pS = softmax(S(a(q)) · T(day_key_H) / student_temperature)
L_relation = KL(pT || pS)
```

支持集至少包含两个 key、两个不同负地点时才计算；否则安全返回零。Teacher、key 和支持集选择都不接收梯度，KL 在 FP32 中计算。此项保留相对邻域，不代替 InfoNCE 的地点判别。

### Qwen 冻结指导与 Day 知识保留

```text
L_guidance     = 1 − cosine(S(a(q)), T(q))
L_preservation = 1 − cosine(S(d),    T(d))
```

`T(q)` 使用干净 Qwen 夜图，指导相机退化后的 Student 保留冻结模型已有表征。Day preservation 只读取训练白天图，默认每个 microbatch 取 1 张，单独前向与反向，限制同时保留的激活量。它不会更新 Teacher 或 Gallery，也不是使用真实夜图微调。

可选 `DAY_VIEW_AUGMENTATION=camera` 将第二项改为 `1 − cosine(S(a_day(d)), T(d))`：Night 模型读取训练白天原片的相机退化视图，恢复到其干净 Teacher 描述符。Day 视图与 Night 视图使用相同增强配置，但 Day 随机种子额外附加 `:day`，两者采样分离；原 Day Teacher、Qwen 可信匹配和 Gallery 始终使用干净图。原 `night_guided_retrieval.sh` 默认 `DAY_VIEW_AUGMENTATION=off`，不改变前一候选的训练定义。

全局配对目标 `1 − cosine(S(a(q)), T(d))` 继续保留。SGPD、Keep、Conditional OT 和幅度损失仍各自独立；本轮新入口先将它们关闭，以便先检查全局检索监督。

## 优化与默认值

入口为 [night_guided_retrieval.sh](../configs/night_guided_retrieval.sh)。参数是实验起点，不是已选出的最优组合。

| 项目 | 默认值 / 控制变量 |
| --- | --- |
| Global / InfoNCE | `GLOBAL_WEIGHT=1` / `CONTRAST_WEIGHT=0.1` |
| 检索关系 KL | `RELATION_WEIGHT=0.1`，`RELATION_HARD_NEGATIVES=64` |
| Qwen 冻结指导 / Day preservation | `SYNTHETIC_GUIDANCE_WEIGHT=0.1` / `DAY_PRESERVATION_WEIGHT=0.5` |
| SGPD / Keep / OT / 幅度 | `PATCH_WEIGHT=0` / `KEEP_WEIGHT=0` / `OT_WEIGHT=0` / `MAGNITUDE_WEIGHT=0` |
| 相机增强 | `NIGHT_AUGMENTATION=camera`，概率 `0.8`，强度 `1` |
| Day Student 视图 | `DAY_VIEW_AUGMENTATION=off`，可单独设为 `camera` |
| 主干与初始化 | 独立 Night DINOv2，全参数训练；默认由原始白天权重初始化 |
| 优化 | AdamW；最后 Block/Norm 学习率 `1e-6`，`LAYER_DECAY=0.8` 向前逐层降低；所有 Night 参数仍可训练 |
| 学习率日程 | `LR_SCHEDULE=cosine`，`WARMUP_STEPS=32` 个优化器更新，`MIN_LR_RATIO=0.1` |
| 显存与 batch | `BATCH_SIZE=2`，`GRADIENT_ACCUMULATION_STEPS=4`，有效 batch 8；BF16 与 activation checkpoint 默认开启 |

梯度累积不会扩大每次前向的 microbatch，候选负样本来自 Day 库。日程按优化器更新计数，`MAX_STEPS` 则按 microbatch 计数；短流程检查需相应减少 warmup，保证 warmup 小于总优化器更新数。

另提供 [night_guided_restore.sh](../configs/night_guided_restore.sh) 候选：`GLOBAL_WEIGHT=0.1`、`SYNTHETIC_GUIDANCE_WEIGHT=1`、`DAY_PRESERVATION_WEIGHT=0.5`、`DAY_VIEW_AUGMENTATION=camera`；InfoNCE 与关系 KL 权重仍各为 `0.1`，其余预算及优化设置与上表相同。它同时减轻跨 Day/Night 配对拉近的约束、加强冻结 Qwen 表征保留，并增加训练 Day 退化恢复。这些改变共同构成候选配置，其收益不能单独归因于其中一项；仍需独立消融。

## 复现与独立消融

从项目根目录执行，每次使用未占用的输出目录：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python

# 4 个 microbatch 的流程检查：1 次优化器更新，不代表训练效果
OUTPUT=runs/night_guided_smoke_01 MAX_STEPS=4 WARMUP_STEPS=0 \
  NUM_WORKERS=0 bash configs/night_guided_retrieval.sh

# 完整合成训练；默认 1 epoch
OUTPUT=runs/night_guided_retrieval_01 bash configs/night_guided_retrieval.sh

# 另一候选：强化知识保留 + 训练 Day 退化恢复，默认也从原 Day 权重初始化
OUTPUT=runs/night_guided_restore_01 bash configs/night_guided_restore.sh

# 只关闭 restore 候选的 Day 视图退化，其它设置保持一致
OUTPUT=runs/night_guided_restore_clean_day_01 DAY_VIEW_AUGMENTATION=off \
  bash configs/night_guided_restore.sh

# 相同训练设置下，独立关闭关系 KL
OUTPUT=runs/night_guided_no_relation_01 RELATION_WEIGHT=0 \
  bash configs/night_guided_retrieval.sh

# 独立关闭两种知识保留约束
OUTPUT=runs/night_guided_no_guidance_01 SYNTHETIC_GUIDANCE_WEIGHT=0 \
  bash configs/night_guided_retrieval.sh
OUTPUT=runs/night_guided_no_day_preservation_01 DAY_PRESERVATION_WEIGHT=0 \
  bash configs/night_guided_retrieval.sh

# 全部候选负样本；关系支持集的大小仍由独立变量控制
OUTPUT=runs/night_guided_all_negatives_01 CONTRAST_HARD_NEGATIVES=0 \
  bash configs/night_guided_retrieval.sh
```

`INIT_ADAPTER_DIR` 可接续兼容的已有 Night 权重；比较消融时，各分支应使用相同初始化、split、seed、增强采样和训练预算。令某个损失权重为 `0` 即关闭该项；关闭相机增强用 `NIGHT_AUGMENTATION=off`。固定库之外的旧 FIFO 对照可使用原 `night_full_global.sh`，新入口始终传入固定库。

## 更难的合成验证与真实评估

保留原图合成验证，另增加固定退化和更大 Day Gallery 的辅助开发验证：

```bash
# 原图验证，保持与历史结果可比
"$PYTHON" scripts/evaluate_pairs.py --adapter-dir runs/night_guided_retrieval_01 \
  --device cuda --batch-size 16 --num-workers 2 \
  --gallery-cache runs/cache/synthetic_val_day.safetensors \
  --output runs/night_guided_retrieval_01/synthetic_val.json

# 辅助验证：相同固定退化 Query + 训练 Day distractors
"$PYTHON" scripts/evaluate_pairs.py --adapter-dir runs/night_guided_retrieval_01 \
  --device cuda --batch-size 16 --num-workers 2 --query-augmentation camera \
  --augmentation-config configs/night_camera.json --augmentation-seed 20261024 \
  --distractor-manifest data/splits/night_train.jsonl \
  --gallery-cache runs/cache/synthetic_distractor_day.safetensors \
  --output runs/night_guided_retrieval_01/synthetic_camera_distractors.json

# 真实检索：原 Day Gallery，Night 仅处理 Query
"$PYTHON" scripts/evaluate_svox.py --adapter-dir runs/night_guided_retrieval_01 \
  --device cuda --batch-size 16 --num-workers 2 \
  --output runs/night_guided_retrieval_01/svox_test.json
```

辅助验证的相机退化按 `(seed, sample_id)` 固定，baseline 与 Night 读取同一退化张量，Gallery 仍为干净 Day。加入的 distractor 只读取训练白天图，拒绝与整个验证 manifest 的地点重叠；不会把训练 Qwen 夜图加入验证 Query。固定种子和更大的 Gallery 有助于暴露原图小库验证的饱和问题，但仍不能替代真实夜景评估。

`--gallery-cache` 复用冻结 Day 描述符，按有序 Gallery 图像字节、checkpoint 与预处理校验。原图小库和加入 distractor 的大库使用不同路径；多个 Night 模型可共享同一有效缓存，Night Query 始终重新提取。它是评估缓存，不是训练的固定 Day bank，不会使验证图参与训练。

### 排除精确源图的跨视角子集

每次配对评估额外输出 `cross_view_baseline` 和 `cross_view_adapter`，复用已有描述符，无需新网络前向。只选取验证集中同一 `(city, place_id)` 有另一张不同 Day 源图的 Query；对每条 Query，将生成该夜图的**精确 Day 源图从全部候选中移除**，只把同地点其它 Day 源图设为正样本。这样可检查模型是否仅恢复精确配对图，还是能够匹配同地点的其它视角。

当前完整验证 split 的该子集为 **90 条 Query / 45 个地点**，其它 295 条单视角 Query 不计入此指标；结果记录选取规则与排除数量。原图与固定相机退化验证都自动报告该项，大库仍保留其它地点的 Day distractors。这个已有多视角子集存在选择偏差，不能代表全部 385 条 Query，也不能当作真实夜景 Recall。

先在原图及辅助合成验证上固定候选模型和设置，再做 SVOX test 的 Recall@1/5/10 比较。结果以对应输出 JSON 为准，本文不预填未完成实验的 Recall。

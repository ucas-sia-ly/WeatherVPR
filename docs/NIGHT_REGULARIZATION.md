# Night 专用模型：正则、训练范围与 Query 融合

## 本轮问题与边界

目标仍是 **Night Query → 原始 Day Gallery** 的全局 VPR。白天 Gallery 始终由冻结的原始 Day DINOv2 + SALAD 编码；Night Student 只处理 Query，所有候选从同一 Day 权重重新初始化。本轮不恢复真实夜景训练，不用测试 Query 做无监督适应。

```text
Day Gallery ── 原始冻结 Day DINOv2 ── 冻结 SALAD ── 固定检索库
Night Query ── 独立 Night DINOv2   ── 同一冻结 SALAD ── 检索固定库
```

本轮是相同小预算下的探索，检验弱 Day 正则、可训练层范围和 Query 融合是否值得继续。四组完整训练与开发选择已经完成；结果和限制记录在文末。

四项固定设置的最终评估、亮度诊断与内存记录见 [完整结果](NIGHT_REGULARIZATION_RESULTS.md)。本预算下新候选没有突破，开发集选出的融合也没有一致泛化收益，默认仍保留纯 `source_night`。

## 对 Claude 建议的审核

记 `d` 为训练白天原图，`q` 为对应的 Qwen 夜图，`a` 为相机退化，`n(d)` 为从真实训练 Day 原片生成的低照度视图。`T` 是冻结 Day 模型，`S` 是 Night Student，输出为归一化 SALAD 全局描述符。

| 建议 | 与当前实现的关系及本轮处理 |
| --- | --- |
| 非对称 Teacher Gallery / Student Query | 已实现。Teacher 参数和已有 Day Gallery 都不随训练变化。这里的“保持”指固定编码器与预处理下保留原 Gallery；不代表 Student 对白天输入也逐位一致。 |
| Synthetic Guidance 是 `S(q) → T(d)` | 这是合理的通用命名，但代码中已有同名旧项，二者需区分。当前**主 Global** 已是 `S(a(q)) → T(d)`；旧 `synthetic_guidance` 则是 `S(a(q)) → T(q)`，本轮后者关闭。 |
| Day Preservation 可作为小权重正则 | 值得单独检验。它约束 `S(d) ≈ T(d)`，不承担保持 Gallery 的职责。此次用 `0.03 → 0` 的线性衰减，避免把建议的 `0.1–0.3` 当作已验证范围。 |
| 若 Restoration 重建像素则去掉 | 本项目的旧 Restoration 是 `S(a_day(d)) → T(d)` 的**描述符**恢复，没有像素解码器或重建损失。不能据像素重建的风险判断此项；本轮 clean-Day anchor 只用干净训练 Day 图，另有 Source Night 分支。 |
| 全局回归加多正样本对比、相似度关系蒸馏 | 已有三项独立损失。固定训练 Day bank 提供小 batch 之外的负样本；同一 `(city, place_id)` 的全部 key 是正样本，避免 false negatives。 |
| 合成图可直接逐坐标对齐 Patch | 对 Qwen 编辑不能默认成立。生成过程可能移动物体、改动立面或局部几何，仍只接受 Stage 2B 验证的 A 类对应。基于亮度的可靠性也不能替代几何验证。此轮 Patch/Keep/OT 保持关闭。 |
| Teacher 显著性给 Patch 降权 | 可作为有效对应上的软权重，但 SALAD Keep 不是建筑物或稳定地标的语义标签；天空、灯光和低纹理的处理不能只靠它作语义判断。 |
| 冻结后层能保持 Teacher 空间 | 只能作为待检验的归纳偏置。前层变化会改变后层输入，冻结深层仍会改变最终描述符，不能保证与 Teacher 同空间。通过直接对齐和真实检索结果检验兼容性。 |
| 浅层 LoRA/Adapter | 用户已选择独立 Night DINOv2。此次比较前六层和后六层的**原权重微调**，不切回 LoRA。二者保留完整独立 Night 权重。 |
| 增加真实夜景 MMD、真实正样本训练或测试时适应 | 用户此前明确移除真实微调，故本轮不恢复。SVOX、Tokyo、RobotCar 等真实夜图只作推理评估，不进优化器、训练 bank 或统计适应过程。 |
| Query 融合成本几乎为零 | 离线已有两套描述符时，向量加权成本很小；部署时 `0 < α < 1` 需要 Teacher、Student **两次 Query 编码**及相应模型存储。应同时报告纯 Student 和融合结果。 |

理论参考来自 [GCISG](https://arxiv.org/abs/2208.10024) 和 [Yang 等的低质量 VPR 蒸馏论文](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Yang_Distillation_Improves_Visual_Place_Recognition_for_Low_Quality_Images_ICCVW_2025_paper.html)。前者研究分类领域的合成到真实泛化，后者主要研究图像质量退化下的 VPR；它们为关系蒸馏、指导和正则提供思路，不能直接证明本项目 Night→Day 的收益。此前阅读与实现对应关系见 [GUIDED_RETRIEVAL.md](GUIDED_RETRIEVAL.md)。

## 四组受控候选

共同设置为 Day 全新初始化、seed `42`、同一训练 split、同一主分支增强随机流、`1 epoch / 908` 次优化器更新、microbatch `8`、梯度累积 `1`、BF16 和 activation checkpoint。学习率 `1e-6`，cosine 日程、warmup `32`、最小学习率比例 `0.1`、weight decay `0.01`。本轮四组统一 `LAYER_DECAY=1`，避免层学习率衰减混淆 early/late 比较。

四组共享以下检索目标：

```text
L_main = 0.1 L_global + 0.1 L_contrast + 0.1 L_relation
L_source = 0.1 L_global(S(n(d)), T(d))
         + 0.1 L_contrast(S(n(d)), frozen_training_day_bank)
L_total = L_main + 0.5 L_source + λday(u) L_anchor(S(d), T(d))
```

Source Night 每个 microbatch 最多取 `2` 张训练 Day 原图；它保留真实照片纹理，再模拟低照度和相机退化，仍不等于真实夜景。旧 Synthetic Guidance、SGPD、Keep、OT、幅度项均为零。Day Teacher、训练 Day bank、SALAD 全程冻结；SALAD 对 Student 保留输入梯度。

| 候选 | Night 可训练范围 | Day anchor |
| --- | --- | --- |
| `full_control` | 全部 DINOv2 参数，包括 embedding、tokens、最终 norm | 关闭 |
| `full_day_anchor` | 同 `full_control` | 初始 `0.03`，按优化器更新线性衰减到 `0`；每个 microbatch 最多 `2` 张干净 Day 图 |
| `early6` | 仅前六个 Transformer blocks，即索引 `0–5` | 关闭 |
| `late6` | 仅后六个 Transformer blocks，即索引 `6–11` | 关闭 |

`early6` / `late6` 都冻结 embedding、tokens 和最终 norm。冻结层允许梯度穿过输入：尤其 `early6` 不能把后六层或最终 norm 放进截断梯度的 `no_grad` 区域。每组只优化选定参数，Teacher 和 SALAD 参数不更新。完整推理模型仍使用全部十二层。

设总更新数为 `U=908`，更新索引为 `u=0,…,U−1`，Day anchor 权重为：

```text
λday(u) = 0.03 × (1 − u / (U − 1))
```

衰减按**优化器更新**计数，梯度累积窗口中不改变权重。Day anchor 与 Source Night 增加额外前向计算，因此更新次数相同并不表示计算成本完全相同。旧 `source_night` 使用过 `layer_decay=0.8`，本轮 `full_control` 是新基线，不能把与旧结果的差异单项归因于正则或层范围。

## 32 GB 主机内存约束

本轮使用 [run_bounded.py](../scripts/run_bounded.py) 串行启动训练和评估。它依赖 Linux cgroup v2 与 systemd user scope，并在缺少有效硬限制时拒绝启动：

- 实验进程及其后代合计 `MemoryMax=8 GiB`，`MemoryHigh=6 GiB`，`MemorySwapMax=0`。
- 启动前要求系统可用内存至少为硬限额加保留量，即默认 `12 GiB`。
- 运行时每 `0.5` 秒记录 cgroup 用量和系统 `MemAvailable`；系统可用内存低于 `4 GiB` 时结束当前实验任务。
- 通过 `runs/.experiment_memory.lock` 排他锁阻止受控任务并行；本轮 `NUM_WORKERS=0`。
- 内存日志记录内核峰值、采样峰值、最低系统可用内存、退出码及 OOM 事件，结果必须结合日志检查。

先前四任务并行尝试发生 OOM；这些日志保留作故障记录，未完成输出不作为本轮结果。四组现在在上述限制下从相同 Day 初始化串行重跑。`MemoryHigh` 是回收阈值，`MemoryMax` 才是实验的硬上限；该限制针对本项目进程，不控制浏览器等外部应用，所以不能承诺整个系统的其它程序总用量。

## 复现训练

从项目根目录执行。输出目录必须是新的；不要把先前失败目录当作已完成训练。下列四条命令依次执行，任一失败先检查其训练日志与 `.memory.jsonl`，不要绕过限制并行重启。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUM_WORKERS=0
export LAYER_DECAY=1 SEED=42 BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1
# 本轮实际产物在 runs/night_regularization_20261024；复现使用新目录
export RUN_ROOT=runs/night_regularization_replay_02

OUTPUT="$RUN_ROOT/full_control" FULL_TRAIN_SCOPE=all \
  "$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/full_control.memory.jsonl" \
  -- bash configs/night_specialist.sh

OUTPUT="$RUN_ROOT/full_day_anchor" FULL_TRAIN_SCOPE=all \
DAY_PRESERVATION_WEIGHT=0.03 DAY_PRESERVATION_SAMPLES=2 \
DAY_PRESERVATION_SCHEDULE=linear DAY_PRESERVATION_FINAL_RATIO=0 \
  "$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/full_day_anchor.memory.jsonl" \
  -- bash configs/night_specialist.sh

OUTPUT="$RUN_ROOT/early6" FULL_TRAIN_SCOPE=early FULL_TRAIN_BLOCKS=6 \
  "$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/early6.memory.jsonl" \
  -- bash configs/night_specialist.sh

OUTPUT="$RUN_ROOT/late6" FULL_TRAIN_SCOPE=late FULL_TRAIN_BLOCKS=6 \
  "$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/late6.memory.jsonl" \
  -- bash configs/night_specialist.sh
```

训练入口的默认参数仍可被环境变量覆盖；比较时应保存实际 `training_config.json`、adapter 元数据、训练日志、权重 SHA-256 和内存日志，以实际记录为准。此轮不设 `INIT_ADAPTER_DIR`，避免从前一 Night 候选继续训练。

## 开发选择、融合与最终评估

先在只读 **SVOX train 的 702 张夜间 Query** 上比较四组新模型及既有 `prior_source`，共五个候选。Gallery 使用同一原始 Day 缓存；所有模型使用 standard `224×224` 预处理，不开多尺度 TTA。开发标签可用于选择配置，但不用于反向传播或构建训练 bank。

选择规则在开发评估前保存为 `runs/night_regularization_20261024/selection_policy_before_development.json`：先按 R@1、R@5、R@10 的字典序选择纯 Student；完全相同时按 `prior_source → full_control → full_day_anchor → early6 → late6` 的顺序。再只为选中的纯 Student 选择融合权重，指标完全相同则优先纯端点，再优先 `α=0`。其它候选的融合曲线只作诊断，不据此改变候选选择。历史 `restore_01` 仅作参考，不在本轮预定候选池中。

融合定义为：

```text
qα = normalize(α × normalize(S(night)) + (1−α) × normalize(T(night)))
```

只在开发集搜索 `α ∈ {0, 0.25, 0.5, 0.75, 1}`，`α` 是 Student 权重。`α=0` 为纯原 Day 编码器，`α=1` 为纯 Night 编码器。选择完成后记录候选权重哈希、`α`、选择规则和时间，再对 SVOX test 及外部夜间集合按同一设置评估；不得按各测试集重新选择 `α`。同时报告原始 Day、纯 Night、固定融合三者的 R@1/5/10 和命中数。

例如，对一个候选进行受控开发评估并导出两套 Query 描述符：

```bash
"$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/full_control.dev.memory.jsonl" \
  -- "$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir "$RUN_ROOT/full_control" --device cuda \
  --gallery-cache runs/cache/svox_train_eval_day.safetensors \
  --batch-size 16 --num-workers 0 --reference-cache-mib 0 \
  --save-query-descriptors "$RUN_ROOT/full_control.dev_queries.safetensors" \
  --output "$RUN_ROOT/full_control.dev.json"
```

导出两套描述符后可离线搜索融合权重，无需为每个 `α` 重复图像编码。缓存与输出需验证模型、图像、预处理和顺序的来源一致；跨 split 不复用 Query 缓存。CLI 也可用 `--query-fusion-alpha <固定值>` 在单次评估中计算融合结果。

离线搜索入口只接受 SVOX train 诊断结果，固定搜索上述五个权重：

```bash
"$PYTHON" scripts/run_bounded.py --log "$RUN_ROOT/full_control.sweep.memory.jsonl" \
  -- "$PYTHON" scripts/sweep_query_fusion.py \
  --evaluation "$RUN_ROOT/full_control.dev.json" \
  --output "$RUN_ROOT/full_control.fusion_train.json" --device cuda
```

此前 SVOX test 和外部测试结果已经查看，本轮整体应称为**探索性比较**，不能宣称盲测，也不能把反复实验中选出的少量 Top-1 增益视为稳定结论。当前结果仅覆盖一个随机种子和这一训练预算，不能据此推广为所有 Day 正则或层范围选择的结论。

## 已完成的开发结果

以下命中数的共同分母为 702。每个新候选都从原始 Day 权重初始化，完成 908 次更新。

| 模型 | R@1 / R@5 / R@10 命中数 | R@1 / R@5 / R@10（%） |
| --- | --- | --- |
| 原始 Day | 621 / 668 / 674 | 88.46 / 95.16 / 96.01 |
| 既有 `prior_source` | 620 / 668 / 677 | 88.32 / 95.16 / 96.44 |
| `full_control` | 617 / 666 / 673 | 87.89 / 94.87 / 95.87 |
| `full_day_anchor` | 617 / 666 / 673 | 87.89 / 94.87 / 95.87 |
| `early6` | 611 / 666 / 677 | 87.04 / 94.87 / 96.44 |
| `late6` | 617 / 666 / 676 | 87.89 / 94.87 / 96.30 |

本预算下弱 Day anchor 没有改变三项 Recall，前六层微调退化最大；没有证据支持把它们替换成默认方案。这也不能证明所有权重、训练时长下这些方法都无效。

按照预先规则，保留 `prior_source`：`runs/night_specialist_20261024/source_night/adapter.safetensors`。其固定融合网格在开发集上的命中数如下。

| Student 权重 α | R@1 / R@5 / R@10 命中数 |
| --- | --- |
| 0 | 621 / 668 / 674 |
| 0.25（开发选择） | 624 / 670 / 675 |
| 0.5 | 623 / 668 / 677 |
| 0.75 | 621 / 668 / 677 |
| 1 | 620 / 668 / 677 |

`selection_before_test.json` 保存所选模型、融合权重、开发指标、文件哈希、规则和时间。实验控制及冻结参数逐项比较通过，见 `control_checks.json`：前/后六层候选的其它参数与原始 Day 完全相等；四组训练配置和逐步增强记录符合受控设计。

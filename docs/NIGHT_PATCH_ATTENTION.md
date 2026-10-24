# Night：可信 Patch 与 DINOv2 注意力蒸馏

## 目标与模型

夜间 Query 使用独立 Night DINOv2，白天 Gallery 始终使用原始冻结 Day Teacher。Night 从 Day 权重初始化，仅更新最后四层 blocks 8–11；其余参数、最终 Norm 和共享 SALAD 均冻结，Norm 与 SALAD 允许梯度传回 Night。没有 Day Anchor、真实夜图训练或测试时适应。

这次在当前后四层配置上，重新验证「相机退化后的 Qwen Night → 冻结 Day 特征」的局部蒸馏，并增加真正的 DINOv2 注意力对齐。SALAD Keep 是聚合分配质量，不是建筑分割或 DINO 注意力。

三组完整训练与评估已结束，**当前 SGPD 和注意力配置没有取得 Top-1 增益**；完整消融、外部评估、注意力图及资源记录见 [实测结果](NIGHT_PATCH_ATTENTION_RESULTS.md)。

## 每张 Qwen Night 都经过相机退化

入口为 [night_patch.sh](../configs/night_patch.sh)，使用 [night_patch_camera.json](../configs/night_patch_camera.json)。CLI 增强概率与曝光操作概率都设为 `1`，使每个训练 Qwen 视图都进入相机处理且必做曝光操作；旧默认 `0.8` 允许整张跳过增强。

曝光同时模拟目标亮度、增益和 gamma；另随机组合光晕、运动模糊、泊松–高斯噪声、JPEG、降饱和、暗角及亮度/对比度。曝光可能提亮或压暗，不保证每张都更暗；其余操作也不保证逐张全部开启。所有操作保持空间网格，不新增几何变换。

Day Teacher 的输入保持干净。可信对应仍由冻结 Teacher 编码的**干净 Day / 干净 Qwen**经过原 Stage 2B 校准产生，仅接受验证过的 A 类对应；不将相同坐标默认为正确对应，也不把退化后的图像重新当作更可靠的匹配证据。

## SGPD：监督实际退化后的 Student

令夜间 patch `i` 的可信白天对应为 `M(i)`，原始对应置信度为 `q_i`。在实际送入 Student 的退化图像上，计算无梯度的可见比例 `v_i`：patch 内 sRGB 亮度大于 `0.01`，且不是 RGB 三通道都达到 `0.99` 的像素占比。

```text
c_i = q_i × v_i
w_i = c_i × keep_day[M(i)]
L_patch = sum_i w_i × (1 − cosine(F_night[i], F_day[M(i)])) / sum_i w_i
```

这是保守的黑场/全通道裁剪代理，不是语义掩码或信息可恢复性的标定结果；普通低照度像素仍可参与。它只能降低既有对应的权重，不能创建对应。Teacher、对应、置信度及可见比例全部 detach。无可信对应时安全返回零局部损失并记录覆盖率。

`saliency` 模式使用 Teacher Keep；`uniform` 模式去掉 Keep 因子。Keep、Conditional OT、幅度蒸馏仍由各自权重控制，本轮均关闭。

## DINOv2 CLS 注意力对齐

[models/attention.py](../src/weather_vpr/models/attention.py) 从最后一个 DINOv2 block 已有的 QKV 输出提取 CLS 到 patch 的注意力，按 head 求平均。softmax 包含原始 CLS/注册 token，然后取 patch 项。只计算 CLS 行，不构造额外的 patch×patch 注意力矩阵，也不增加第二次 Backbone 前向。

Day 注意力来自冻结 Teacher；Night 注意力来自本次退化输入的 Student，允许梯度通过 Q/K 及可训练 blocks。提取兼容当前非重入 activation checkpoint，不更改原注意力前向。

在同一批 A 类对应上，用相同固定权重构造两张分布：

```text
P_T(i) ∝ c_i × keep_day[M(i)] × attention_day[M(i)]
P_S(i) ∝ c_i × keep_day[M(i)] × attention_night[i]
L_attention = KL(P_T || P_S)
```

两者都只在可信支持集上归一化；`uniform` 模式同样移除 Keep。KL 在 FP32 计算，每张图至少需要两个正权重对应且 Teacher 注意力质量非零才参与。Student 分布加小量平滑以避免零概率；Teacher 目标与所有权重 detach。该损失约束可信支持内的相对关注分布，不要求所有 patch 对齐，也不宣称恢复完整注意力图。

它不约束可信区域获得的总注意力质量。后续固定验证对照中，条件 KL 降低与原始可信注意力总量下降同时出现；这两个量需要分别记录，不能将训练日志中的 confidence×Keep 加权质量当作原始总量。数值反例、验证实测和真实夜景失败图见 [失败诊断](NIGHT_FAILURE_ANALYSIS.md)。

它与 SGPD 特征余弦损失独立开关，可单独消融；只有使用独立 Night 主干的 `full` 模式支持此注意力项。

## 预先固定的三组对照

计划在训练前写入 [experiment_plan_before_training.json](../runs/night_patch_20261024/experiment_plan_before_training.json)，绑定 checkpoint、清单、校准、增强和训练代码哈希。

| 组别 | SGPD | Attention |
|---|---:|---:|
| `global_control` | 0 | 0 |
| `sgpd` | 0.1 | 0 |
| `sgpd_attention` | 0.1 | 0.05 |

共同设置：7,263 组配对、seed 42、batch 8、1 epoch / 908 次更新、LR `1e-6`、layer decay `1`、BF16、checkpointing。Global / 多正样本 Contrast / Relation 权重均为 `0.1`；固定 Day bank 与每 Query 64 个难负样本保持一致。三组使用同一相机增强随机流。

Source Night 辅助分支保持原配置：从训练 Day 原图模拟低照度视图，每步最多 2 张、权重 `0.5`。它不是新引入的真实夜间监督。

SVOX train 的 702 张夜图仅用于只读开发评估，按纯 Student 的 R@1、R@5、R@10 依次比较；完全相同时依次优先 `global_control`、`sgpd`、`sgpd_attention`。选定模型和权重哈希在 test 前记录。三组均做预声明的 SVOX test 消融，选定组再测 Tokyo247 Night 和 RobotCar 公开 Night / Night-rain 诊断集。无 Query 融合或按测试集调参。历史测试已查看，本轮属于探索性实验，不声称盲测。

## 独立控制与复现

| 环境变量 | 对应训练 CLI | 用途 |
|---|---|---|
| `PATCH_WEIGHT` | `--patch-weight` | SGPD，0 关闭 |
| `ATTENTION_WEIGHT` | `--attention-weight` | 注意力 KL，0 关闭 |
| `PATCH_VISIBILITY` | `--patch-visibility off\|photometric` | 退化输入可靠性降权 |
| `IMPORTANCE_MODE` | `--importance-mode uniform\|saliency` | 是否使用 Teacher Keep |
| `KEEP_WEIGHT` / `OT_WEIGHT` / `MAGNITUDE_WEIGHT` | 对应 `--*-weight` | 独立局部消融 |
| `AUGMENTATION_PROBABILITY` | `--augmentation-probability` | 整张增强概率 |

以下只运行三次更新，验证连通性，**不是完整训练结果**。请使用新的输出路径，且不要与其他训练/评估并行运行：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUTPUT=runs/night_patch_smoke_02 MAX_STEPS=3 WARMUP_STEPS=0 \
  PATCH_WEIGHT=0.1 ATTENTION_WEIGHT=0.05 NUM_WORKERS=0 \
  "$PYTHON" scripts/run_bounded.py --max-gib 6 --high-gib 5 \
  --reserve-gib 4 --log runs/night_patch_smoke_02.memory.jsonl \
  -- bash configs/night_patch.sh
```

完整运行将 `MAX_STEPS=0 WARMUP_STEPS=32`，依次设置表中三组权重及各自全新 `OUTPUT`。本轮内存策略为串行、workers 0、任务 cgroup 上限 6 GiB / high 5 GiB、系统可用内存保留 4 GiB；测试任务上限 2 GiB。通用保护脚本拒绝超过 8 GiB 的任务上限，保护范围是其子任务，不限制其他桌面应用。

已完成 183 项单元测试、三次更新 smoke，以及三组完整训练和评估。每组全部 7,263 张 Qwen 输入均执行了相机处理和曝光。单组可信 Patch 覆盖率约 10.54%，注意力有效图像比例约 99.67%；局部项有效参与训练但没有增加本轮 SVOX test Top-1。完整数字见 [实测结果](NIGHT_PATCH_ATTENTION_RESULTS.md)。

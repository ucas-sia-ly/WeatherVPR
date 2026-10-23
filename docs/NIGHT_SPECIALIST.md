# Night 专用 Query 模型

## 任务边界

目标是让 Night DINOv2 处理夜间 Query，匹配由原始 Day DINOv2 + SALAD 提取的白天 Gallery。Student 不承担白天推理任务，也不要求一个模型同时解决白天和夜晚。

```text
白天 Gallery ── Frozen Day DINOv2 ── Frozen SALAD ── 固定 Day 描述符
夜间 Query   ── Trainable Night DINOv2 ── 同一冻结 SALAD ── 检索上述描述符
```

Night 主干从已有 Day 权重初始化并全参数训练；Day Teacher 与 SALAD 的参数冻结。SALAD 对 Student 保留输入梯度。白天 Gallery 的提取模型、描述符维度和归一化方式不变，无需重建 Gallery。

## 调整两项旧约束

记 `T` 为冻结 Day 模型，`S` 为 Night Student，`d` 为真实训练白天原图，`q` 为其 Qwen 夜图，`a` 为相机退化。描述符均归一化。

| 旧损失 | 实际约束 | Night 专用配置的处理 |
| --- | --- | --- |
| Synthetic Guidance：`1 − cosine(S(a(q)), T(q))` | 保留原 Day 模型在干净 Qwen 夜图上的表征；目标本身仍是该模型对合成夜图的响应。 | 默认关闭。它可能限制 Night 适配，并不是匹配 Day Gallery 的必要条件；收益尚未由独立消融证实。 |
| Day Preservation：`1 − cosine(S(d), T(d))` | 给 Student 增加白天表征任务。 | 默认关闭，推理目标不要求 Student 处理白天。 |
| Day Restoration：`1 − cosine(S(a_day(d)), T(d))` | 从训练白天图的相机退化恢复干净 Day 描述符。 | 旧实现保留作历史消融；普通相机增强不保证产生低照度输入。 |

旧 camera 配置的曝光 gain 下限为 `0.6`，且变换可能被概率门控跳过，因此不能把这种输入统称为夜景。历史实验同时改变了多个目标，不能把其联合结果归因于某一项损失。旧入口和结果继续保留，便于复现。

## Source Night 辅助分支

为了利用真实照片的结构和成像统计，新增从**训练 Day 原图**构造低照度视图的辅助分支。它不让 Student 读取干净白天图：

```text
训练 Day 原图 d ── 强制低照度变换 n(d) ── Night Student ── 辅助检索损失
              └────────────────────── Frozen Day Teacher ── 干净目标 T(d)
```

`n(d)` 先在线性光空间施加强制曝光衰减，默认 gain 为 `0.03–0.30`，再可选添加照明变化、白平衡和 gamma 变化，并叠加已有相机退化。辅助分支关闭 camera 自身的曝光操作，避免其自动增益抵消低照度构造。强制曝光不受 camera 总概率门控控制。

这只是由真实白天照片产生的低照度模拟，不能等同于真实夜景。它无法完整生成夜间灯源、真实照明分布、阴影和传感器响应；这些差异仍需真实夜景的只读评估检验。

辅助分支使用与主分支相同的冻结训练 Day 特征库，以 `(city, place_id)` 定义多正样本 InfoNCE，并对齐干净 `T(d)`：

```text
L_source = λg × [1 − cosine(S(n(d)), T(d))]
         + λc × InfoNCE(S(n(d)), frozen_training_day_bank)
L_total  = L_main + λsource × L_source
```

同地点的所有库条目是正样本，不作为负样本。Teacher、Day bank 和目标不接收梯度；只有 Night 主干更新。辅助分支本轮只使用全局目标，没有新增 Patch 监督路径；原 Qwen 分支的 SGPD 仍依赖已验证的 A 类对应。

## 主分支与初始配置

主分支仍为 `S(a(q)) → T(d)`：跨域全局对齐、固定 Day bank 多正样本 InfoNCE、检索关系蒸馏各自独立。Qwen 夜图的 SGPD、Keep、Conditional OT 和幅度损失保持独立开关，SGPD 仅使用已验证的 Stage 2B A 类对应；SALAD Keep 不表示建筑语义。

新入口为 `configs/night_specialist.sh`，辅助低照度参数为 `configs/source_night.json`。初始建议如下，均不是已验证的最优超参数：

| 项目 | 初始值 |
| --- | --- |
| 主分支 Global / Contrast / Relation | `0.1 / 0.1 / 0.1` |
| Synthetic Guidance / Day Preservation | `0 / 0` |
| SGPD / Keep / OT / 幅度 | `0 / 0 / 0 / 0`，可单独开启 |
| Source Night 辅助权重 | `--source-night-weight 0.5` |
| 每个 microbatch 的辅助视图上限 | `--source-night-samples 2` |
| Source Night 配置 | `--source-night-config configs/source_night.json` |

Student 的主分支输入为 Qwen 夜图（可叠加相机退化），辅助分支输入为强制低照度 Day 视图。干净 Day 只供冻结 Teacher 和固定库使用。训练不读取 SVOX 真实夜图，不恢复已移除的真实夜景微调流程。

## 比较与评估

先比较以下两组，使用相同初始化、训练地点、主分支 Night 增强随机种子、microbatch 顺序和优化器更新预算：

1. Night 专用主分支，关闭两项旧约束，`source-night-weight=0`。
2. 相同主分支，增加 `source-night-weight=0.5` 的辅助分支。

辅助分支增加前向计算量，应同时记录运行时间与显存；不能把相同更新次数解释为相同计算成本。辅助采样使用独立随机流，避免改变主分支 Night 增强。此比较检验 Source Night 的增量作用；与旧 restore 候选的对比包含多个变化，不能单项归因。

先用只读 SVOX train 检索开发结果选择候选、固定配置，再运行 SVOX test。真实图像不进入训练、不构造训练负样本库；test 不用于调参。报告 R@1/5/10、命中数量、训练预算和使用的初始化。需要明确：目标是 Night Query Recall 提升，设计调整本身不保证取得提升。

两组实验已完成，最终 test R@1 从 92.71% 提到 93.07%，R@5 提高而 R@10 略降；完整对照、限制和复现见 [实测记录](NIGHT_SPECIALIST_RESULTS.md)。不能据此认定某项正则必然有害，或将整个模型的 test 差值归因于单一损失。

# Night 专用 Query 模型

## 任务与模型

Night DINOv2 处理夜间 Query，检索由原始 Day DINOv2 + SALAD 提取的白天 Gallery。Student 不承担白天推理任务。

```text
白天 Gallery ── Frozen Day DINOv2 ── Frozen SALAD ── 固定 Day 描述符
夜间 Query   ── 独立 Night DINOv2 ── 同一冻结 SALAD ── 检索上述描述符
```

Night 主干从 Day checkpoint 初始化，默认直接微调最后四层 blocks 8–11 的原始权重，其余参数包括最终 Norm 冻结。最终 Norm 和 SALAD 保留输入梯度。Day Teacher 始终 `no_grad`。Night 权重保存完整主干，`FULL_TRAIN_SCOPE=all` 可显式训练其全部参数；默认层范围与 SALAD 训练设置的比较见 [SALAD 训练说明](SALAD_TRAINING.md)。

白天 Gallery 的模型、描述符维度和归一化方式不变，无需重建 Gallery。

## Source Night 辅助分支

从**训练 Day 原图**构造低照度视图，利用真实照片的结构和成像统计。Student 的辅助输入始终经过强制曝光衰减：

```text
训练 Day 原图 d ── 强制低照度变换 n(d) ── Night Student ── 辅助检索损失
              └────────────────────── Frozen Day Teacher ── 干净目标 T(d)
```

`n(d)` 在线性光空间施加强制曝光衰减，默认 gain `0.03–0.30`，再可选添加照明、白平衡和 gamma 变化，并叠加相机退化。辅助分支关闭 camera 自身的曝光操作，避免自动增益抵消低照度构造。强制曝光不受 camera 总概率门控控制。

这是白天照片的低照度模拟，不能等同于真实夜景；它无法完整生成真实夜间灯源、照明分布、阴影和传感器响应。默认真实夜间数据仅用于只读开发和评估；可选 [真实配对阶段一](NIGHT_REAL_PAIRS.md) 显式引入审核训练图对。

辅助分支复用冻结训练 Day bank，以 `(city, place_id)` 定义多正样本 InfoNCE，对齐干净 `T(d)`：

```text
L_source = λg × [1 − cosine(S(n(d)), T(d))]
         + λc × InfoNCE(S(n(d)), frozen_training_day_bank)
L_total  = L_main + λsource × L_source
```

同地点的所有库条目均为正样本。Teacher、Day bank 和监督目标不接收梯度；只有选定的 Night 参数更新。辅助分支仅使用全局目标，不新增 Patch 对应假设；Qwen 主分支的 SGPD 仍需通过 Stage 2B A 类可信对应。

## 主分支与默认配置

主分支为 `S(a(q)) → T(d)`：Qwen 夜图 `q` 经过相机退化 `a` 后输入 Student，目标来自其真实白天源图 `d`。全局对齐、固定 Day bank 多正样本 InfoNCE、检索关系蒸馏各自独立。SGPD、Keep、Conditional OT 和特征幅度损失均有独立开关；SALAD Keep 不表示建筑语义。

入口为 `configs/night_specialist.sh`，辅助低照度参数为 `configs/source_night.json`。以下是后续实验的起点，不是已证实的最优参数：

| 项目 | 默认值 |
| --- | --- |
| Night 更新范围 | 最后 4 个 Transformer Block；最终 Norm 冻结 |
| 主分支 Global / Contrast / Relation | `0.1 / 0.1 / 0.1` |
| SGPD / Keep / OT / 幅度 | `0 / 0 / 0 / 0`，可单独开启 |
| Source Night 权重 | `--source-night-weight 0.5` |
| 每个 microbatch 辅助视图上限 | `--source-night-samples 2` |
| Source Night 配置 | `--source-night-config configs/source_night.json` |
| LR / Layer decay | `1e-6 / 1` |
| 数据加载 worker | `0` |

干净 Day 只供冻结 Teacher 和固定库使用，默认训练不读取真实 Night 数据。真实配对阶段一需显式启用辅助分支，真实目标独立缓存、不进入 GSV 负样本库。默认推理使用纯 Night 描述符，可选 Query 融合属于独立诊断，需要额外 Teacher 编码。

## 比较与评估

使用相同初始化、训练地点、主分支增强随机种子、样本顺序、层范围和更新预算比较 `SOURCE_NIGHT_WEIGHT=0` 与 `0.5`。辅助采样使用独立随机流，避免改变主分支增强。辅助分支增加计算量，应记录运行时间与显存，不能将相同更新次数解释为相同计算成本。

先用只读 SVOX train 开发集固定候选，再评估 test；真实图像不进入训练 bank，test 不参与调参。报告 Recall@1/5/10、整数命中、训练预算和初始化。

保留的历史两组实验为全部参数微调，所选 `source_night` 在 test R@1/5 上小幅改善、R@10 略降，不能称为稳定突破。详细控制、限制和显式复现参数见 [实测记录](NIGHT_SPECIALIST_RESULTS.md)。

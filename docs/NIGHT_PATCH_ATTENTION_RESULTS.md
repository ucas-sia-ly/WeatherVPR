# 后四层 SGPD 与注意力蒸馏：实测结果

**本轮未取得 Recall@1 增益，不将强制全图增强、SGPD 或注意力项替换为默认方案。** 已完成三组完整训练、开发选择、三组 SVOX test 消融，以及选定模型的外部夜景评估。模型和损失实现保留为可独立控制的实验能力。

方法、公式和命令见 [Patch 与 DINO 注意力蒸馏](NIGHT_PATCH_ATTENTION.md)。训练仅使用 7,263 对 Day/Qwen 图及训练 Day 模拟低照度视图；真实夜图只用于开发与评估，没有恢复真实微调、Day Anchor 或测试时适应。

## 相同预算的三个候选

所有模型独立从原始 Day 权重初始化，仅更新 blocks 8–11，最终 Norm、其它 Backbone 参数、Teacher 和 SALAD 冻结。共同使用 seed 42、batch 8、908 次更新、LR `1e-6`、layer decay 1、BF16、activation checkpoint。Global / Contrast / Relation 均为 0.1，Source Night 为 0.5。Keep / OT / Magnitude 均关闭。

| 候选目录 | SGPD 权重 | Attention 权重 |
| --- | ---: | ---: |
| `runs/night_patch_20261024/global_control` | 0 | 0 |
| `runs/night_patch_20261024/sgpd` | 0.1 | 0 |
| `runs/night_patch_20261024/sgpd_attention` | 0.1 | 0.05 |

三组每张 Qwen 图均经过相机处理，强制曝光操作并随机组合其它退化。**每组 7,263/7,263 张实际执行了增强**，没有按局部覆盖率丢弃图像；各组相机及 Source Night 的逐步增强计数一致。Source Night 每组共 1,816 个辅助视图。原始 Day/Qwen Teacher 输入、可信对应验证和 Gallery 都保持干净。

## 开发选择

以下为 SVOX train 的只读开发指标，分母 702，按 R@1 / R@5 / R@10 排列。

| 模型 | 命中数 | Recall（%） |
| --- | --- | --- |
| 原始 Day | 621 / 668 / 674 | 88.46 / 95.16 / 96.01 |
| 全局对照 | 623 / 668 / 676 | 88.75 / 95.16 / 96.30 |
| SGPD | 623 / 668 / 676 | 88.75 / 95.16 / 96.30 |
| SGPD + Attention | 623 / 668 / 676 | 88.75 / 95.16 / 96.30 |

三者指标完全相同，按照训练前记录的同分顺序选择 **`global_control`**。选择及三个候选的权重 SHA-256 均在本轮 test 前固定，见 [selection_before_test.json](../runs/night_patch_20261024/selection_before_test.json)。历史测试结果已经查看，本轮属于单随机种子、短预算的探索性比较，不宣称盲测。

## SVOX test：预先约定的三组消融

全部使用原始 Day Gallery，224×224 单尺度、纯 Student Query，无融合或测试时增强，分母 823。

| Query 模型 | R@1 / R@5 / R@10（%） | 命中数 |
| --- | --- | --- |
| 原始 Day | 92.71 / 98.42 / 99.15 | 763 / 810 / 816 |
| 全局对照 | 92.22 / 98.78 / 99.15 | 759 / 813 / 816 |
| SGPD | 92.22 / 98.78 / 99.27 | 759 / 813 / 817 |
| SGPD + Attention | 92.22 / 98.66 / 99.27 | 759 / 812 / 817 |

三组 R@1 均比原始 Day 少命中 4 张。SGPD 相对同设置全局对照的 Top-1 成功集合完全一致，仅 R@10 增加 1 张；加入 Attention 后 Top-1 修复 1 张、退化 1 张，净变化为零，R@5 比 SGPD 少 1 张。因此不能用局部损失下降或较高阶 Recall 的微小变化宣称 Top-1 突破。

| 相对原始 Day | Top-1 修复 / 新增错误 | R@5 净增加 | R@10 净增加 |
| --- | --- | ---: | ---: |
| 全局对照 | 6 / 10 | +3 | 0 |
| SGPD | 6 / 10 | +3 | +1 |
| SGPD + Attention | 5 / 9 | +2 | +1 |

用户已有的 `night_late4_01` 为 761/823 Top-1，历史全部参数 `source_night` 为 766/823；这轮没有超过它们。不能根据本次 test 反选另一个训练配置，也不把这组强制增强设为当前默认。

## 外部集合：仅评估开发选定的全局对照

以下结果属于 `global_control`，**不是 SGPD 或 Attention 的外部评估**。每行按 R@1 / R@5 / R@10 排列，Teacher Gallery 固定。

| 集合 / Query 数 | 原始 Day 命中数 | 选定 Night 命中数 | Night Recall（%） |
| --- | --- | --- | --- |
| Tokyo247 Night / 105 | 91 / 95 / 98 | 90 / 97 / 100 | 85.71 / 92.38 / 95.24 |
| RobotCar public Night / 197 | 179 / 187 / 190 | 178 / 187 / 192 | 90.36 / 94.92 / 97.46 |
| RobotCar public Night-rain / 224 | 214 / 216 / 218 | 213 / 215 / 218 | 95.09 / 95.98 / 97.32 |

三组 Top-1 均少 1 张，没有一致优于原始 Day。Tokyo 使用官方 MAT 的夜间查询子集和完整 Gallery，本地 v3 图像仍存在与原始发布版本未逐字节核验的限制；RobotCar 为公开 train 位姿上的只读自定义检索诊断，不是官方定位 test。完整协议与缺少真值的数据集状态见 [外部夜景说明](EXTERNAL_NIGHT_RESULTS.md)。

## 局部项确实有效参与了训练

- 两个局部候选各有 196,535 个验证 A 类 Patch；退化可靠性降权后 195,973 个仍为正权重，覆盖全部 Patch 位置的 **10.54%**。
- Attention 有效图像 **7,239/7,263（99.67%）**，平均 KL 为 **0.30170**；SGPD + Attention 的平均 Patch 损失为 **0.15880**，SGPD 单项为 **0.15896**。这些是训练批次统计，不是检索指标。
- 每个 Query 始终有 64 个有效难负样本。三份模型权重哈希不同，检索错误排名也不同；等 Recall 并非重复评估同一模型。
- [逐张量检查通过](../runs/night_patch_20261024/control_checks.json)：每组后四层的 56 个参数张量均发生变化，另外 **119 个张量逐位保持原值**，包括最终 Norm 与 embedding；Night 文件不包含 SALAD。

当前实验只说明这组权重与预算下没有 Top-1 增益，不能推广为所有局部或注意力蒸馏都无效。注意力 KL 约束的是可信支持内的相对分布，不监督未知区域的完整注意力或语义分割。

## 注意力图与资源

已导出验证集前四个配对的 [注意力对比图](../runs/night_patch_20261024/attention_preview.png) 和 [图像、模型、增强溯源](../runs/night_patch_20261024/attention_preview.png.json)，没有按视觉效果挑图。对比干净 Day、退化 Qwen、Teacher 的两种输入和训练后的 Night 注意力。每行使用共同色阶；该图只是定性诊断，不能代替可信匹配或真实 Night Recall。

183 项单元测试通过；三步集成检查、完整训练和所有评估均正常结束。任务串行运行，最大允许 6 GiB RAM；测试、状态核验和预览分别使用更低上限。**整轮任务实测最高 cgroup 内存约 5.00 GiB，系统可用内存最低约 8.61 GiB，没有 OOM。** 这些是任务及计费缓存的观测值，不是模型独占内存比较。三步检查的临时权重已删除，仅保留小型日志和元数据。

完整结果、各阶段资源记录、三项 Recall、配对错误变化、模型哈希见 [results_summary.json](../runs/night_patch_20261024/results_summary.json)。每个候选目录保留完整 Night 权重与训练日志；`*.svox_train.json`、`*.svox_test.json` 保留各自评估及错误路径。复现时使用新输出目录，按 [方法文档](NIGHT_PATCH_ATTENTION.md#独立控制与复现) 分别设置三组权重并串行运行。

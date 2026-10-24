# 阶段一实现验证（2026-10-25）

实现已接入，但未运行完整训练和 Recall 实验。以下均为技术检查，不能据 loss 或梯度变化宣称泛化收益。

## 自动化检查

- 最终 232 项测试通过，覆盖配对准入、缓存、随机隔离、三分支优化、累积尾窗口、配置及旧权重兼容。
- 完整日志：`runs/stage1_implementation_checks/tests_final.log`。
- 代码、Markdown 链接及 shell 语法另作静态检查。

## 实际权重短程训练

三组均使用原始 Day checkpoint、Late4、BF16、activation checkpoint、LR 1e-6、seed 42；主 batch 2、Source 1、辅助 2、K=2、3 个主微批、2 次更新，包含短尾窗口。warmup 为 0，仅为缩小检查预算，不是完整阶段一默认协议。

| 组别 | Student 视图 Qwen / Source / 额外 | RAM 峰值（GiB） | 结果 |
| --- | --- | ---: | --- |
| A | 6 / 3 / 0 | 2.67 | 2 次更新，权重保存成功 |
| B | 6 / 3 / 6 | 3.35 | 2 次更新，权重保存成功 |
| C | 6 / 3 / 6 | 2.90 | 2 次更新，权重保存成功 |

- 采用 5 GiB 任务 RAM 硬上限、4 GiB 回收阈值、禁止任务 swap，保持 4 GiB 系统余量；所有模型任务串行。未发生 OOM。
- 三组均仅 blocks 8–11 的 56 个张量发生变化，其余 119 个主干张量与原始 checkpoint 逐元素相同。
- 三组主样本/Source 索引、增强操作记录相同，主流摘要相同；第一次更新前的 GSV/Source 损失一致（容差 1e-5）。
- B/C 额外梯度非零；B 首个真实余弦 loss 约 0.3842，加权后约 0.03842。这是接入证据，不是收益证据。
- A 以 `A_final/` 记录修复后的最终实现；更早的 `A/` 是初次工程检查，保留但不纳入上表。

## 推理回读

已通过原有 `scripts/describe.py` 加载 B 的权重，对真实 Night 输出 `[1, 8448]` FP32 描述符，值均有限，L2 范数为 1。新增训练元数据未破坏既有推理加载协议。

## 真实样本边界

B 使用 4 对通过助手目视检查的样本，状态记为 automatic，而非人工认证；全部归入 `all_unknown_routes_smoke_only`，不建立 dev/test，不宣称不同前缀代表不同地点。样本及检查图位于 `runs/stage1_implementation_checks/real_smoke_review.jsonl` 和 `real_smoke_review.jpg`，仅用于技术验证。

全量 9,520 对的未审核模板已保存到 `data/splits/darkdriving_review.template.jsonl`，接受数为零。正式实验仍需可靠路线/近重复审核与固定划分。

运行方式和固定版本的 SALAD/MixVPR/CosPlace 源码依据见 [阶段一说明](../../docs/NIGHT_REAL_PAIRS.md)。

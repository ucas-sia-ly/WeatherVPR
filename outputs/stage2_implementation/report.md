# 阶段二实现与预检查

部分 Patch 统计增强已实现，T0–T3 首轮完整训练及 SVOX 开发评估已启动。运行状态以 [正式队列状态](../../runs/night_token_stage2_20261025_075002_seed42/status.json) 为准；本报告仅记录启动前的工程验证，不声明 Recall 提升。

- 264 项测试通过：统计轴、CLS/未选 Patch、独立 RNG、梯度、模型插入点、checkpoint 重算、训练分支隔离、旧格式推理恢复、预算和队列检查。
- T0/T1/T2/T3 均完成真实 CUDA 三微批、两次优化更新；主/Source 流、T1/T3 真实流及 T2/T3 风格随机选择一致。
- 各组 119 个冻结张量与原始 checkpoint 位级相同，后四层 56 个张量发生更新；这些是保存的实际 Night 权重检查。
- 64 张训练图、16 微批的无更新诊断：gain p99=1.0579、max=1.1849，无 gain>2；中心化特征相对 RMS 变化 p95=3.93%、max=5.08%；描述符余弦漂移最大 0.00273。模型完整 state hash 未变，普通推理确定且等于原始 Day 初始化。
- 基于上述训练数据诊断，保留预定 p=.5、Patch 比例 .3、最大混合系数 .2，无额外限幅。该决定早于开发 Recall，诊断证据哈希纳入正式计划。

正式四组均从原始 Day 独立初始化，采用相同 fast 后端，每组 908 次更新、主有效 batch8、Source2、启用真实辅助时 Real2；使用 BF16、CUDA bank、4 workers、预取1、不做激活重计算。任务 RAM 上限7 GiB/high6 GiB/系统余量2 GiB，所有模型任务串行受保护。

运行说明见 [阶段二文档](../../docs/NIGHT_TOKEN_STAGE2.md)，机器可读证据见 [verification.json](verification.json)。原始检查保存在 `runs/stage2_implementation_checks/`，短程队列保存在 `runs/night_token_stage2_smoke_20261025_seed42/`。正式结果由队列自动写入 `runs/night_token_stage2_20261025_075002_seed42/report.md`，只评估 SVOX 开发集；没有运行扩散伪域、真实统计供体或外部测试搜索。

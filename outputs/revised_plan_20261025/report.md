# 修订方案落地与首轮结果

**已完成方案调整、训练数据审计、三 seed A/B/C、冻结域覆盖诊断，以及独立 Source 的 G1/G2×A/B 首轮完整训练。** 新训练与评估均使用固定预算，只评估 SVOX dev。

## H1：真实配对有小幅 Top-1 排序增量，同时存在 Top-5 退化

三个 seed 的 B−A Top-1 为 +5/+7/+2 张，平均 +0.665 个百分点；Top-5 为 −1/−2/−2。所有 Top-1 修复原本已在 A 的前五名内，不能称检索候选覆盖提升。B 仍为可选实验，未设为默认。见 [多 seed 结果](p0_multiseed.md)、[逐 Query 复核](p0_review.md)、[亮度/空间分层](stage1_strata.md)。

## H2：已测量真实/合成覆盖，暂不训练真实供体

每域 512 张训练图，共 2,560 张；原始 Day 前缀及 SALAD 全冻结，0 次更新、0 次 Recall。block 7 的合成联合参考阈值包含率：DarkDriving94.92%、NightCity95.12%、BDD65.62%。BDD 的差异仍混有道路、视场、相机与亮度；PCA32只保留参考方差54%，参考也未覆盖 camera-augmented Qwen 的全部训练输入。该测量不是风格因果证明。

已对相同64张接收图、等量供体、相同掩码和λ测量真实ΔF及描述符/SALAD变化，实际29张启用；未根据Recall调强度。结合25张训练样本的有限内容检查，当前证据不足以打开真实供体TFS、GUIDE或配对残差。见 [定量与图像报告](domain_coverage/report.md)。

## H3：独立 Source 地点规模首轮

四组从原始Day全新初始化，seed42；每组908次更新、7,263个主Qwen视图、1,816个Source视图；B再有1,816个真实辅助视图。Source使用该档独立bank，主Qwen流与bank保持一致。原始Day开发命中数为621/668/674。

| 单元 | 可用Source地点 | 实际Source地点 | 实际新增地点 | R@1 / R@5 / R@10 命中数 |
|---|---:|---:|---:|---|
| G1_A | 6441 | 1816 | 0 | 621 / 668 / 677 |
| G1_B | 6441 | 1816 | 0 | 624 / 666 / 674 |
| G2_A | 11441 | 1816 | 792 | 623 / 668 / 676 |
| G2_B | 11441 | 1816 | 792 | 624 / 666 / 676 |

| 比较 | ΔTop-1 | ΔTop-5 | ΔTop-10 | Top-1修复 / 退化 |
|---|---:|---:|---:|---:|
| G1_B−G1_A | +3 | -2 | -3 | 6 / 3 |
| G2_B−G2_A | +1 | -2 | +0 | 5 / 4 |
| G2_A−G1_A | +2 | +0 | -1 | 3 / 1 |
| G2_B−G1_B | +0 | +0 | +2 | 0 / 0 |

本轮扩大 Source 池使 A 的 Top-1 净增 2 张；B 的 Top-1 和 Top-5 命中集合完全相同，Top-10 净增 2 张。真实辅助在 G1/G2 上的 Top-1 增量分别为 +3/+1，尚未显示明确的规模互补证据；这些单 seed 结果也不足以断言扩展无效。

G2池为17,263张Day、11,441个地点；一个固定预算epoch实际只曝光1,816个Source地点，其中792个不在G1。主Qwen已覆盖6,441个G1地点，故实际Student训练地点并集为7233，不能写成11,441个地点都经过Night训练。

规模轴同时改变Source可采样地点和Source-only固定候选bank；采样量与优化预算匹配，但不能将差异单独归因于地理多样性。独立G1与旧同batch Source的A/B不是同一个实验底座。单seed、同一开发集的几张差异不足以确认“大规模GSV与真实监督互补”，不因本轮结果追加已看过test的搜索。完整配置/权重哈希/配对Query见 [规模结果](../../runs/revised_gsv_scale_20261025/report.md) 和 [JSON](../../runs/revised_gsv_scale_20261025/summary.json)。

## 实现、资源与保留限制

新增训练池来源清单、CSV/图像/排除证据签名、可复现域覆盖缓存、独立Source采样及bank、串行队列、多seed/配对Top-k/亮度与200m空间分组统计。GSV原朝向值−11到731按原SALAD文件规则保留。321项检查通过，包含实际训练循环的梯度累积、尾批、多正样本与RNG隔离数值对照；四单元真实数据短跑及完整预算亦通过。

本轮10次新完整训练均通过7GiB RAM上限、6GiB high、2GiB系统余量保护，无OOM；完整训练最大任务RAM 5.04GiB，最大GPU已分配张量 2.92GiB。保持microbatch4×累积2以保留对比批次语义，采用GPU bank、4workers、prefetch1及关闭重计算。实际内存峰值不是设备整体驻留或利用率，缓存构建开销另见短跑日志。

DarkDriving的9,520条接受记录均为automatic；9,286个精确内容组不等于独立路线，24组跨文件前缀。现有审核只有一个全场地scene_group，实际为文件行内循环，未完成路线/近重复认证。保留该协议复核H1，后续修改采样须单独对照。

本地SF-XL的466张night query仅盘点为最终测试候选，未提取描述符、未计算Recall、未按失败样本选模型；完整协议与历史使用核验仍待完成。未重新运行SVOX test/Tokyo247/RobotCar。

[主方案](../../idea.md) · [实现与复现](../../docs/REVISED_RESEARCH_IMPLEMENTATION.md) · [数据清单](data_inventory/inventory.json) · [DarkDriving审计](darkdriving_source_audit.json) · [测试保留记录](reserved_night_test.json)

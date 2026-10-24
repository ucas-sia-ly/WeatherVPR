# 夜间检索失败诊断：注意力为什么没有转化为 Recall

日期：2026-10-24。此次只做推理、可视化与固定干预；未训练、修改权重或用 test 选择新配置。

## 1. 目前证据支持的解释

**在这 17 张对照失败中，现有证据支持优先检查微调后相似地点的排序变化；新增注意力项未能纠正共同失败。不能将全部退化归因于注意力项，也不能根据热图断言模型只在看灯光。**

- Attention 模型失败的 64 张图中，47 张六个模型均失败，另外 **17 张至少有一个旧方案成功**。这 17 张是本轮重点。
- 其中 **16 张在 Global 对照和 SGPD 中也已失败**。Attention 相对 SGPD 只新增 1 张失败，同时修复 1 张，Top-1 净变化为零。SGPD 与同配置 Global 对照的 Top-1 成败集合完全相同。
- 失败对照里可见连续街屋、重复立面和相似道路布局。多张图把相似地点、同一路段的另一位置排在正确地点前。局部特征“像白天”还不等于能区分这些位置。
- 当前注意力损失对齐的是**可信支持集内的相对分布**。它既不保证这些 Patch 获得足够总注意力，也不直接监督哪些特征支持正确 Gallery、哪些特征支持错误 Gallery。

“合成域微调影响了细粒度地点区分”是符合这些观察的解释；尚不能单独确定曝光增强、Qwen 风格、训练层数或某个损失是共同退化的原因。历史 Source Night、Late 4 与本轮的配置不完全相同，不能用它们替代受控因果消融。

## 2. 检查范围与完整性

使用 SVOX test 的全部 **823 Night Query / 17,166 Day Gallery**，正样本仍按原生 **25 m UTM** 协议定义。Gallery、预处理及原始 Day checkpoint 均固定。

| Query 编码器 | R@1 命中数 | R@1 |
| --- | ---: | ---: |
| 原始 Day | 763 | 92.71% |
| 历史 Source Night，全部参数微调 | 766 | 93.07% |
| 用户 Late 4 | 761 | 92.47% |
| 本轮 Global control | 759 | 92.22% |
| 本轮 SGPD | 759 | 92.22% |
| 本轮 SGPD + Attention | 759 | 92.22% |

重新提取全部 Query 并检索，**4,938 个原始排名逐项复现**；之后分析 23 个固定案例：上述 17 张、Attention 修复的 1 张、3 张共同失败、2 张共同成功。其中 8 张导出完整检索图和热图。已验证所有模型权重前后不变。

17 张对照失败的错误描述符距离优势中位数仅 **0.00385**。但不主要是 25 m 阈值边界：错误 Top-1 距 Query 在 25–30 m 的只有 1 张，在 25–35 m 的有 3 张。部分案例来自相邻连续帧，不能当作 17 个独立场景。

## 3. 具体图片上发生了什么

下表的排名是**全图库中任意有效正样本的最佳排名**；并非固定某一张白天图的排名。

| 案例 | Day | Source | Late 4 | Global | SGPD | +Attention | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q0134 | 2 | 2 | 1 | 1 | 1 | 2 | Attention 唯一新增的 Top-1 退化，误检距 Query 2,397 m |
| q0798 | 1 | 2 | 2 | 2 | 2 | 1 | Attention 唯一修复；原始 Day 本就正确 |
| q0233 | 1 | 3 | 3 | 3 | 3 | 3 | 连续重复街屋；正图距 Query 3.3 m，错误 Top-1 为 32.5 m |
| q0431 | 1 | 1 | 2 | 2 | 2 | 2 | 相似成排街屋；错误地点距 Query 74.5 m |
| q0448 | 2 | 1 | 2 | 2 | 2 | 2 | Source 独自成功；其余混淆到 66.1 m 外的地点 |
| q0761 | 3 | 2 | 1 | 2 | 2 | 2 | Late 4 独自成功；本轮误检距 Query 537.6 m |
| q0031 | 86 | 147 | 111 | 117 | 117 | 118 | 六模型共同困难例；夜图明显模糊，注意力聚焦仍不能恢复地点区分 |

### q0134：注意力项确实翻转了一次分数接近的竞争

SGPD 的“最近负样本距离 − 最近正样本距离”为 **+0.001127**，加 Attention 后为 **−0.001514**，正确地点从第 1 降到第 2。错误图具有相似道路透视与两侧建筑布局，但不是同一地点。

六个模型的热图都主要覆盖远处建筑及局部高响应区域，不能据此说 Attention 模型突然“只看路灯”。Attention 模型的分布更分散，但分散本身没有确保正确排序。

![q0134 检索对照](../runs/night_failure_analysis_20261024/figures/q0134_retrieval.png)

![q0134 注意力、Keep 与有符号检索贡献](../runs/night_failure_analysis_20261024/figures/q0134_maps.png)

### q0233 / q0431：共同训练退化，局部项没有救回

q0233 中所有微调模型都把正确图排到第 3，该退化在 Global 对照中就已出现。正确图和错误图均含连续的相似街屋，错误图位于附近。q0431 中原始 Day 和 Source 能保住正确地点，本轮三组均混淆到另一段相似街景。

这说明“聚焦建筑”仍可能不够：重复建筑也会提供很强响应，检索需要保留区分具体立面组合、道路位置和视角的特征。**这一点是图片与排名支持的解释，不是热图提供的语义标注。**

![q0431 相似地点混淆](../runs/night_failure_analysis_20261024/figures/q0431_retrieval.png)

其余图：[q0233](../runs/night_failure_analysis_20261024/figures/q0233_retrieval.png)、[q0448](../runs/night_failure_analysis_20261024/figures/q0448_retrieval.png)、[q0761](../runs/night_failure_analysis_20261024/figures/q0761_retrieval.png)、[q0798](../runs/night_failure_analysis_20261024/figures/q0798_retrieval.png)。全部图的路径和哈希见 [图像清单](../runs/night_failure_analysis_20261024/figures/manifest.json)。

### q0031：信息退化与可见内容差异，所有方案都困难

这张夜图有明显运动模糊，窗、墙面和道路边缘细节已损失。六个模型的注意力仍集中在照亮的建筑区域，却全部把另一个地点排在第一，最佳有效正图排名为 86–147。所示 GPS 正图与夜图的可见内容也有较大差异；25 m 标签保证位置邻近，不保证像素或视角对应。因此不能期待注意力对齐凭空恢复丢失细节，也不能将这一例归因于新增注意力项。这只是共同失败的一例，未对全部 47 张做同类归因。

图：[检索对照](../runs/night_failure_analysis_20261024/figures/q0031_retrieval.png)、[注意力与贡献](../runs/night_failure_analysis_20261024/figures/q0031_maps.png)。

## 4. 为什么仅看 CLS 注意力图会误判

本轮同时计算三种不同的量，热图在同一行使用共同绝对色标：

1. 最后一个 DINOv2 Block 的真实 CLS→Patch 注意力；绘图时只在 Patch 上归一化。
2. SALAD `assignment_keep`：Patch 分配给有效簇的质量，不是建筑物分割。
3. **固定正确 Gallery 与固定错误 Gallery 的有符号余弦差贡献**：红色支持前者，蓝色支持后者。

第三项按实际 SALAD 聚合、各簇归一化和最终归一化分解。所有 Patch 贡献与直接 CLS 贡献之和等于完整描述符的固定正负余弦差，最大绝对残差 **2.24×10⁻⁸**。这是给定当前特征和 OT 的精确代数分解，不能解释成唯一因果归因。

SALAD 将一个归一化 CLS 向量与 64 个归一化簇向量拼接再归一化，直接 CLS 部分通常占描述符平方范数的 **1/65≈1.54%**。这不代表它无关：q0134 的 SGPD→Attention 固定余弦差下降中，约 **43% 来自直接 CLS，57% 来自聚合 Patch**。注意力计算还会影响 Patch 表征，不能把它的作用限制在这 1.54%。

在选取的 23 张图上，Attention 相对 SGPD 的注意力熵均增加，中位变化 **+0.285 nat**；与 Keep 的 Spearman 均下降，但仍然正相关。17 张重点失败中，Attention 最大的 26 个 Patch 的**合计有符号贡献有 9 张支持正确图**。因此，“热图亮的地方都是干扰”不成立。

固定贡献图为便于跨模型比较，使用同一正图与同一错误图。固定正图来自原始 Day 的最佳有效正图；它未必是其他模型的最佳正图。因此固定余弦差可能为负，而该模型仍然全库 Top-1 正确；不要混用这两个指标。

## 5. 固定干预：哪些区域真的影响排序

对每张案例固定 **26/256 个 Patch（10.16% 面积）**，位置均由该图原始 SGPD+Attention 模型的对应热图确定，六个模型使用相同位置。执行两种操作并重新检索全图库：

- **像素均值填充**：遮挡对应 14×14 输入块，重跑 DINOv2 和 SALAD。
- **OT 后读出删除**：保留原 CLS 和 OT，删除选中 Patch 的聚合贡献，再归一化；不重跑 DINOv2。

以下只统计 Attention 模型原本失败、旧方案至少一个成功的 17 张图：

| 删除/遮挡区域 | OT 后删除变为 Top-1 的数量 | 像素填充变为 Top-1 的数量 |
| --- | ---: | ---: |
| CLS 注意力最大的 26 块 | 6 / 17 | 6 / 17 |
| Keep 最大的 26 块 | 3 / 17 | 4 / 17 |
| 对正确减错误余弦差最负的 26 块 | 17 / 17 | 13 / 17 |
| 对正确减错误余弦差最正的 26 块 | 0 / 17 | 1 / 17 |
| 随机，固定重复 0 / 1 / 2 | 3 / 2 / 3（各 17 张） | 8 / 3 / 8（各 17 张） |

**正负贡献区域用到了测试真值和错误候选，是 oracle 诊断，不是可部署的选块方法；这些数字不是模型 Recall 提升。** 它表明当前特征中正确信息与干扰贡献并存，Attention/Keep 的大小不足以区分它们。像素删除与读出删除结果不同，也说明特征、其他 Patch 和 OT 会随输入一起变化。

例如 q0134 遮挡最大注意力块后，排名 **2→59**；删掉最负的聚合贡献则 **2→1**，但同一区域的像素遮挡仅到第 3。高注意力区域不能简单作为应删除的干扰区域。

## 6. 局部监督覆盖稀疏，条件 KL 有明确的盲区

### 训练中实际覆盖

从 908 个训练更新、7,263 张图的日志重新汇总：可信且可见 Patch 为 **195,973 / 1,859,328 = 10.54%**。这是按实际 batch 图像数加权的覆盖率，不是对各步简单平均。

另外固定使用地点隔离的合成验证集前 32 对，复用实际相机增强、Stage 2B A 类对应及可见性权重：

| 指标 | 实测 |
| --- | ---: |
| 可信覆盖率均值 / 中位数 | 9.56% / 9.375% |
| 可信 Patch 数均值 | 24.47 / 256 |
| confidence×visibility×Keep 有效加权数量均值 / 中位数 | 8.11 / 7.13 |
| 条件 KL，原始模型处理退化图 → 训练后模型 | 0.2757 → 0.2043 |
| 可信区域原始注意力总质量 | 17.57% → 16.03% |
| KL 下降 / 原始总质量下降 / 两者同时下降 | 21 / 22 / 15 对，分母均为 32 |

有效加权数量定义为 `(sum w)² / sum(w²)`；它不是新增的匹配数量。训练日志原有的 confidence×Keep 加权 attention mass 也不是上表的原始注意力总质量，二者已分别计算。

### 为什么 KL 下降，关注总量仍能下降

当前目标在可信支持集内分别归一化 Teacher 和 Student。若只提高未知 Patch 的 logits，可信 Patch 的相对比例可以不变，而它们的总质量显著下降。

脚本中的四 Patch 数值例子：只将两个未知 Patch 的 logits 增加 8，可信区域总质量 **95.26%→0.67%**，条件 KL 仍约 **0.0100808**，实际实现差值仅 **1.79×10⁻⁷**。对应测试也已加入。

这证明目标函数不保证总注意力集中到可信区域；32 对验证也确实出现平均 KL 与总质量同时下降。**验证对照包含整个训练过程的影响，不能据此把下降单独归因于 Attention 项，更不能直接宣称这就是某张真实夜景误检的唯一原因。**

## 7. 接下来最值得验证的方向

1. **先处理共同退化。** 暂不提高 Attention 权重。用同一后四层训练设置，在训练/开发集受控比较“部分原始 Qwen + 部分相机退化”与“全部强制退化”，并检查近邻地点的正负 margin。当前注意力配置未解决这 16 张共同失败，宜先检查共享训练设置。
2. **把局部监督与 SALAD 判别联系起来。** 以已有冻结 Teacher、可信 A 类对应及训练 Day bank 为基础，优先比较有效簇条件分配或相对正负描述符关系；不能将 CLS 热图更相似当作成功标准。现有 InfoNCE、Relation 和硬负样本银行应作为受控基线，避免重复叠加等价目标。
3. **若继续注意力蒸馏，单独验证支持集总质量约束。** 在可信且可见区域上加入可开关的总质量项，同时记录原始总质量与条件 KL；不可对未知 Patch 强行造标签，也不能恢复同坐标对应假设。是否改善仍需开发集检验。

这些是由诊断提出的下一轮假设，本轮未据 test 图片调整损失或重新训练。原始权重、失败案例及诊断结果保留供复核。

## 8. 代码、复现与内存

- [失败分析脚本](../scripts/analyze_failure_cases.py)：全量原始排名复现、特征捕获、固定区域干预与权重核验。
- [聚合贡献模块](../src/weather_vpr/inference/patch_attribution.py)：精确有符号 Patch 余弦差分解。
- [支持集检查](../scripts/analyze_attention_support.py)：固定 held-out 配对、实际增强、原始质量/加权质量与条件 KL。
- [绘图脚本](../scripts/render_failure_analysis.py)：统一色标、原图/候选与溯源哈希。
- [汇总数据](../runs/night_failure_analysis_20261024/findings_summary.json)、[逐图原始结果](../runs/night_failure_analysis_20261024/analysis.json)、[对应支持集原始结果](../runs/night_failure_analysis_20261024/attention_support.json)。

所有 Torch 任务串行，`NUM_WORKERS=0`，cgroup 上限不超过 6 GiB。本轮成功任务最高内核记录峰值 **1.40 GiB**，系统最低可用内存 **10.26 GiB**，无 OOM。该数值为任务 cgroup 实测，不是整个桌面系统的总占用。全套单元测试 **190 passed**。

复现时使用新的输出目录，避免覆盖证据；已有 `query_cohorts.json` 绑定源评估、模型及 Gallery 哈希：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OUT=runs/night_failure_analysis_replay_01

"$PYTHON" scripts/run_bounded.py --max-gib 6 --high-gib 5 \
  --log "$OUT.memory.jsonl" -- "$PYTHON" scripts/analyze_failure_cases.py \
  --inventory runs/night_failure_analysis_20261024/query_cohorts.json \
  --output-dir "$OUT" --batch-size 8

"$PYTHON" scripts/run_bounded.py --max-gib 4 --high-gib 3 \
  --log "$OUT/support.memory.jsonl" -- "$PYTHON" scripts/analyze_attention_support.py \
  --adapter-dir runs/night_patch_20261024/sgpd_attention \
  --output "$OUT/attention_support.json" --pairs 32

"$PYTHON" scripts/run_bounded.py --max-gib 2 --high-gib 1.5 \
  --log "$OUT/render.memory.jsonl" -- "$PYTHON" scripts/render_failure_analysis.py \
  --analysis "$OUT/analysis.json" --output-dir "$OUT/figures"
```

原始排名对 batch 数值细节敏感，因此分析脚本还按原评估 batch 16 和精确检索分块重新验证全部排名。初次未通过的浮点一致性检查保留在输出的 `numerical_precision_check/`，未混入最终结果；它没有改变 Top-1 结论。

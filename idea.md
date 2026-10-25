# WeatherVPR：真实配对蒸馏与 Token 风格泛化研究方案

更新：2026-10-25。本文是实验设计文档。阶段一 seed 42 的 A/B/C 已完成，见 [首轮结果](outputs/stage1_results_20261025/report.md)：B 在 SVOX 有微弱增量，尚无稳定跨数据集收益。阶段二 T0–T3 也已完成：本轮批内部分 Patch 增强未改变开发集 R@1/5/10 或各阈值成功集合，见 [结果评价](docs/NIGHT_TOKEN_STAGE2_RESULTS.md)。模块与运行方式见 [阶段二实现](docs/NIGHT_TOKEN_STAGE2.md)，论文依据和后续研究见第 12 节及 [专项分析](docs/TOKEN_STYLE_DG_REVIEW.md)。当前增强保持默认关闭；正式路线审核与多 seed 复核仍待完成，真实统计供体、扩散伪域和配对残差尚未验证。

## 1. 研究目标与当前决策

本项目已有 GSV/Qwen 合成夜图及 Source Night 低照度辅助训练。下一步先回答：

> 保持原有 GSV 训练流程与固定 Day Gallery 不变，额外加入有限场景的真实昼夜配对监督，能否提高 Night Student 在未参与训练的真实夜景上的检索能力？

**阶段一采用最小真实配对蒸馏：保留原有 8 个 Qwen 主视图和 2 个 Source Night 辅助视图，额外加入 2 个 DarkDriving Night，只新增全局配对余弦损失。** 三个分支串行计算、累积梯度后统一更新。阶段一不要求生成 DarkDriving 的 Qwen 图，不引入新的局部监督或特征统计操作。

首先验证真实监督的增量价值，再决定是否研究“真实配对测得的合成—真实统计残差，能否帮助更多 GSV 训练地点”。后者是阶段二的研究假设，不能提前作为阶段一成立的机制解释。

**首轮结果后的调整：** 暂不直接扩大配对损失或实现复杂残差模块。先用 TFS 启发的训练时 Token 增强检验外观多样性的价值，再分别考察真实统计来源和 GUIDE 启发的伪域覆盖。它们是独立机制，应分别消融；当前不承诺效果，也不据外部 test 结果增加搜索。

## 2. 现有基础与可行性

### 2.1 模型和数据

- 从 `weights/dino_salad.ckpt` 初始化独立 Night DINOv2-B/14；仅训练 blocks 8–11，共 28,357,632 个参数。Patch Embedding、位置/CLS Token、前八层及最终 Norm 冻结。
- 原始 Day DINOv2 与共享 SALAD 始终冻结。Student 对齐原始 Day 描述符空间，推理仍为单个 Night Student 检索固定 Day Gallery，无需重建图库。
- GSV/Qwen 地点隔离训练集为 7,263 对，已有 Global、固定 Day bank 多正样本 InfoNCE、Relation，以及由训练 Day 构造的 Source Night 辅助分支。
- DarkDriving_Base 清理后有 9,520 对真实昼夜图像，已有相对路径清单、严格解码校验及精确重复记录。已接入配对辅助优化，首轮整场地只作训练使用，尚无可靠地点/路线隔离开发集；真实目标不进入 GSV 检索负样本 bank。见 [DarkDriving 使用说明](docs/DARKDRIVING.md)。

DarkDriving 原论文通过轨迹跟踪和位姿匹配构造真实配对，报告厘米级位姿对齐。这支持将其用于配对监督，但不能自动推出全部像素或 Patch 同坐标对应。本地文件清理也不等于独立复核了全部图对的空间关系。[DarkDriving 原论文](https://arxiv.org/html/2603.18067v2#S3.SS3)

### 2.2 风险与历史依据

历史全参数 `source_night` 在 SVOX test 的 R@1 从 763/823 增至 766/823，在 Tokyo247 Night 则从 91/105 降至 88/105，尚无一致的跨数据集收益。加入 Patch/Attention 的后四层实验也未获得 Top-1 改善。见 [Night 历史结果](docs/NIGHT_SPECIALIST_RESULTS.md)、[外部夜间结果](docs/EXTERNAL_NIGHT_RESULTS.md)和 [Patch 实验结果](docs/NIGHT_PATCH_ATTENTION_RESULTS.md)。

因此阶段一优先控制变量。已有结果只作为背景，新实验必须重跑同设置 Late4 基线，不能将相对历史全参数模型的差值归因于真实配对监督。

工程可行性较高：复用现有编码器、冻结聚合器及全局损失即可。效果仍未知，主要风险是封闭场地过拟合、近重复样本主导、真实目标缺乏地点判别性，以及辅助梯度损害城市街景泛化。

## 3. 固定架构与初始化

```text
训练：
GSV Qwen Night ── 原有相机增强 ──┐
GSV Day ── Source Night 构造 ────┼→ 同一个 Night Student → 冻结 SALAD
DarkDriving Real Night ─────────┘                          ↓
                                                     各分支损失
GSV Day / DarkDriving Day → 冻结原始 Day Teacher → 冻结 SALAD

推理：
真实 Night Query → Night Student → 同一冻结 SALAD → 固定原始 Day Gallery
```

首轮 A/B/C/D 所有组均从同一份原始 `dino_salad.ckpt` 复制 Student，**不加载历史 Night adapter**。采用相同初始化、相同 Late4 范围和全新优化器。

若以后研究在现有 Night 权重上继续适配，应另开实验，所有组加载同一 Night checkpoint；无真实配对的控制组也必须继续训练同样步数。不能将继续训练的实验组与未经继续训练的旧模型直接作为唯一对照。

Teacher 只在 `no_grad` 下生成目标。SALAD 和最终 Norm 参数虽冻结，Student 分支仍须保留其输入梯度。训练和推理均不新增编码器、聚合器或可学习质量模块。

## 4. DarkDriving 数据准入与划分

### 4.1 区分配对质量、夜间难度和可见信息

按文件名前缀、道路内容与光照分层抽查约 200 对，记录背景一致性、视角偏移、遮挡、动态内容、模糊及明显错误。抽查用于评估清单可靠性并建立审核规则，不意味着其余图对已获人工逐一认证；疑似异常需扩展检查。

每个样本分别记录：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `pair_valid` / `reject_reason` | 昼夜背景是否对应，是否存在严重错配 | 决定是否可用于配对蒸馏 |
| `difficulty` | 暗度、眩光、噪声等夜间困难程度 | 分层采样与性能分析，不直接等同于质量 |
| `visibility_status` | 是否接近完全无信息，或仍保留可辨认结构 | 标记黑场等极端输入，单独统计处理 |
| `review_status` | 人工审核、自动筛查或尚未复核 | 明确监督质量的证据来源 |

很暗、有眩光不构成直接排除理由。结构正确且仍有可辨认信息的困难样本应保留。接近完全无信息的图像按预先固定的规则标记并单独统计；不要要求模型恢复不可见的地点信息，也不要事后只删除模型失败的困难图。

第一轮质量权重采用二值，依据配对与输入有效性确定，不用 Teacher 相似度高低作为唯一筛选标准。

### 4.2 路线与内容隔离

1. 先按 `content_group` 绑定任一侧精确复用的图组，再结合路线、采集批次、近重复和空间区域合并分组。
2. `sequence_id` 是文件名前缀，`content_group` 是精确内容复用组，都不是 `place_id`。历史审计发现不同前缀间复用 Day 内容，单独按前缀划分并不充分。
3. 有可信位姿时按空间区域划分并设置缓冲；无位姿时，用检索辅助人工核验路线重叠，保存审核记录。检索候选不直接充当地点真值。
4. 同一路段的重复经过、不同照明版本和近重复帧不得跨训练与开发分组。数据划分后再建立 Teacher 目标缓存。
5. 不将论文的官方训练数量直接当成本地可用训练数量。实际数量取决于清理、审核与本项目隔离结果，全部写入实验清单。

若封闭场地无法形成可靠独立开发区，可以只把 DarkDriving 用作训练来源，使用现有外部开发协议选择配置；此时不声称具备地点隔离的 DarkDriving 开发结果。

### 4.3 地点多样性与负样本

审计应报告独立路线/场景数、重复程度、光照覆盖及每个分组的采样占比。按场景平衡抽样，避免长序列和邻近帧主导。道路与照明类别只有在有可靠标注时才用于分类统计；亮度不能代替灯光条件标签。

首轮 DarkDriving 只提供配对全局监督，**不加入固定 GSV Day bank，不计算以 sample_id 为类别的真实域 InfoNCE**。不同文件可能是同地点邻帧，不能默认互为负样本。

## 5. 统一预处理与监督目标

所有干净 Day 目标、真实 Night 输入及评估图使用现有模型协议：RGB、224×224 PIL bilinear 缩放、ImageNet normalization。Teacher 输入和 Gallery 不加训练增强。

当前 `DarkDrivingPairs` 默认使用 bicubic 并输出 `[0,1]`，不等于模型预处理。应从原始 PIL 图直接调用项目模型预处理，或传入覆盖默认流程的 transform；避免先默认缩放再二次处理，也避免重复归一化。

Qwen 主分支保留 `configs/night_camera.json` 的既有相机增强，Source Night 保留 `configs/source_night.json`。真实 Night 首轮仅规范预处理，不叠加强制低照度或其他额外退化；几何增强与真实图额外退化留作独立消融。

定义 L2 归一化描述符：

\[
t(x)=\operatorname{norm}(A(T(x))),\qquad
s(x)=\operatorname{norm}(A(S_\theta(x))).
\]

其中 \(T\) 为冻结原始 Day DINOv2，\(S_\theta\) 为独立 Night Student，\(A\) 为共享冻结 SALAD。

真实配对的目标始终是对应真实 Day 的 \(t(d_i)\)，不是合成图描述符。允许缓存审核训练 Day 的目标向量，以减少在线 Teacher 开销；缓存绑定原始 checkpoint、预处理、划分和图像哈希，与 GSV bank 分开管理。

## 6. 损失设计

### 6.1 保持已有 GSV 监督

沿用当前默认权重：

\[
\begin{aligned}
L_{\mathrm{GSV}}={}&0.1L_{\mathrm{Global}}^Q
+0.1L_{\mathrm{InfoNCE}}^Q
+0.1L_{\mathrm{Relation}}^Q\\
&+0.5\left(0.1L_{\mathrm{Global}}^{\mathrm{Source}}
+0.1L_{\mathrm{InfoNCE}}^{\mathrm{Source}}\right).
\end{aligned}
\]

Qwen 和 Source Night 均检索原有冻结 GSV Day bank，正样本沿用 `(city, place_id)` 定义。Relation 与 InfoNCE 温度、hard-negative 配置保持原默认值。Source Night 不额外增加 Relation。

Patch、Keep、OT、Magnitude、Attention 和 Synthetic Guidance 权重均为 0。具体默认与实现以 [训练配置](configs/night_specialist.sh)和 [训练循环](src/weather_vpr/training/train.py)为依据。

### 6.2 新增真实配对项

\[
L_{\mathrm{real}}=
\frac{\sum_i w_i\left[1-s(n_i)^\top\operatorname{sg}(t(d_i))\right]}{\sum_iw_i},
\qquad w_i\in\{0,1\}.
\]

\(\operatorname{sg}\) 为停止梯度。各分支按各自有效样本数归一化，不把 GSV 和真实图混为一次跨域样本平均。

\[
\boxed{L_B=L_{\mathrm{GSV}}+\lambda_{\mathrm{real}}L_{\mathrm{real}}}
\]

首轮 \(\lambda_{\mathrm{real}}=0.1\)，作为明确起点而非最优值。正常情况下从已接受清单采样，保证辅助分支有有效图对；若临时有效数为零，跳过该项并记录原因。重复跳过应视为数据管线异常处理，不允许悄然退化为纯 GSV 训练。

损失取平均只控制数量带来的机械缩放，不能消除辅助采样频率、数据重复或梯度方向对优化的影响。

## 7. 训练组织：保持原分支，串行增加真实辅助

默认 gradient accumulation 为 1 时，每次 optimizer update：

| 分支 | Student 视图数 | 输入来源 | 监督 |
| --- | ---: | --- | --- |
| Qwen 主分支 | 8 | 原 GSV/Qwen 训练样本流 | Global + InfoNCE + Relation |
| Source Night | 2 | 从上述 GSV batch 按原策略选取 Day 构造低照度视图 | Global + InfoNCE |
| Real Pair | 2 | 独立采样的 DarkDriving 训练配对 | 真实全局配对余弦 |

这是 **12 个 Student 视图、三次分支前向**，不是 12 个独立地点，也不要求一次驻留全部激活。Source Night 来自主 GSV batch。真实 Day 的 Teacher 前向或目标缓存另计。

执行顺序：

1. Qwen 前向、计算其已有损失并反传。
2. Source Night 前向，按既有权重反传，及时释放分支计算图。
3. DarkDriving Night 前向，反传 \(\lambda_{\mathrm{real}}L_{\mathrm{real}}\)。
4. 对所有分支累积后的梯度统一裁剪，再 `optimizer.step()` 和清零。裁剪阈值沿用现有 1.0，不逐分支分别裁剪。

**不将总损失再次除以 12 或分支数。** 各项已按分支样本平均并带有明确权重，再平均会改变原 GSV 训练强度。

若 gradient accumulation 为 \(K\)，每个主 microbatch 都关联对应辅助分支，各分支损失统一除以 \(K\)，每 \(K\) 个主 microbatch 更新一次；不足 \(K\) 的末尾窗口沿用现有修正逻辑。默认完整窗口的视图数分别为 \(8K/2K/2K\)，不把 12 当作新的 GSV batch size。

训练长度由固定 GSV 主样本流决定，不随 DarkDriving 长度变化。真实辅助使用独立可复现的循环采样器，记录重复曝光次数；GSV 与真实数据、增强均使用各自随机流，新增分支不能改变原有主样本顺序或增强结果。

“6 GSV + 2 Real”只保留为将来固定总输入预算下的数据替换实验，不作为阶段一主对照；若执行，必须另配对应的 6 GSV 控制，并继续单列 Source Night 开销。

## 8. 阶段一实验矩阵

| 组别 | 原有 GSV 训练 | 额外辅助 | 权重 | 主要用途 |
| --- | --- | --- | --- | --- |
| A：匹配基线 | 8 Qwen + 2 Source Night | 无 | 无 | 当前 Late4 基线 |
| B：真实配对 | 与 A 完全相同 | 2 个 DarkDriving Night，目标为对应 Day | 0.1 | 真实监督的增量价值 |
| C：合成辅助对照 | 与 A 完全相同 | 2 个额外 GSV/Qwen 视图，目标为对应 Day | 0.1 | 额外合成监督能否获得类似收益 |
| D：低权重真实配对，可选 | 与 A 完全相同 | 与 B 相同的真实样本流 | 0.03 | 检查对真实约束强度的敏感性 |

C 的额外分支**只用与 B 相同形式的平均全局余弦损失**，不额外增加 InfoNCE、Relation 或 Source Night。辅助样本来自现有 GSV 训练清单，按固定独立流采样，输入增强沿用 Qwen 主分支流程。这样匹配辅助视图数、全局损失形式及权重；它也增加了 GSV 曝光，因此称为“合成辅助对照”，不称完全隔离计算量的实验。

B/C 的 Day 目标均优先使用各自缓存，使在线 Teacher 开销尽量一致；仍报告实际时间与内存，不把相同输入数量宣称为严格相同 FLOPs 或端到端成本。缓存构建开销单列。

关键比较：B−A 检验额外真实监督的实际收益；C−A 检验额外合成辅助的收益；B−C 比较两种辅助来源；D−B 诊断真实权重敏感性。阶段一不单独证明真实配对对应关系优于所有无配对方法。

### 8.1 固定项

| 项目 | 首轮设置 |
| --- | --- |
| 初始化 | 同一原始 Day `dino_salad.ckpt`；不加载 Night adapter |
| 训练范围 | Late4，blocks 8–11；最终 Norm 和 SALAD 冻结 |
| 主 batch / accumulation | 8 / 1；需要调整时各组同步 |
| AdamW LR / weight decay / layer decay | `1e-6 / 0.01 / 1` |
| 调度 | cosine，warmup 32 updates，min LR ratio 0.1 |
| 主训练预算 | 以 A 的一个 GSV epoch 实际主 batch 流锁定，保存精确 updates |
| 精度与加载 | BF16、activation checkpoint、`NUM_WORKERS=0` |
| 初筛 seed | 42；关键结论复核使用相同的 42、43、44 |
| 第一轮选择的模型 | 固定预算结束的最终 checkpoint，不按 test 挑中间权重 |

A/B/C 在同一 seed 下必须保持主样本、Source Night 选择与增强、初始化、更新次数、调度完全一致。D 是可选诊断，不是首轮必须扩展的搜索；如仍需考察 0.3，应先更新实验计划再运行，不据 test 追加权重扫描。

## 9. 日志与低频梯度诊断

每步记录各分支有效样本数、原始/加权损失、累计曝光、optimizer update、LR、裁剪前总梯度范数、运行时间与峰值内存。日志将 GSV 主视图数、Source Night 视图数、真实辅助数分开，不只报告一个 `effective_batch_size`。

梯度诊断使用向量定义：

\[
\mathbf g_G=\nabla_\theta L_{\mathrm{GSV}},\qquad
\mathbf g_R=\nabla_\theta(\lambda_{\mathrm{real}}L_{\mathrm{real}}).
\]

在固定诊断步上计算 \(\|\mathbf g_G\|\)、\(\|\mathbf g_R\|\)、范数比和两向量余弦，覆盖最后四个可训练 block；可按 block 汇总。\(L_{\mathrm{GSV}}\) 必须包含 Source Night 加权项，不能只用 Qwen 损失代表整个旧方案。任一范数接近零时方向余弦记为未定义，不伪造为 0。

建议预先固定每 100 个 updates 采样一次，统计裁剪前梯度。诊断可通过复制/分离分支梯度或额外独立前后向实现，但必须不改变用于更新的累计梯度、不推进数据/增强随机流，并单列开销；不要求每步执行。

负梯度余弦仅说明当前样本在当前参数处的局部方向冲突，不足以判定真实分支损害泛化；最终判断依据是检索表现，而非某一次梯度指标。

## 10. 评估、配置选择与结论边界

### 10.1 三层证据

| 层次 | 检查内容 | 能支持的解释 |
| --- | --- | --- |
| 训练接入正确 | 真实 loss 有限、梯度到达 Late4、冻结参数不变 | 真实监督确实进入优化 |
| 地点判别未退化 | 可靠隔离开发数据上的排名/可信正负 Margin，以及外部检索 | 配对余弦改善是否伴随判别能力 |
| 真实夜间泛化 | 固定原始 Day Gallery 的 R@1/5/10，多 seed 和逐 Query 对照 | 当前设置下真实配对训练是否值得保留 |

DarkDriving 内部 Margin 仅在正样本与负地点有可信标注时计算；不能把所有非同名 Day 当负样本。无法满足时只报告严格配对诊断和目标描述符分布，主要结论依赖外部检索。还应检查原始 Teacher 在场地 Day 图上的区分能力；若目标本身难以区分路线，继续提高配对余弦未必有价值。

### 10.2 选择规则与测试使用

首轮先固定 seed 42 的 A/B/C，仅用 SVOX 开发集和可靠的 DarkDriving 开发诊断判断是否推进。若需要在预定候选中选配置，按 SVOX 开发 R@1、R@5、R@10 依次比较，全部相同则优先较低开销配置。该规则在新结果产生前写入实验计划。

进入结论的关键 A/B/C 比较使用相同 3 个 seeds，报告均值、离散程度、整数命中数、修复/新增错误及困难光照分桶。原始 Day 模型始终保留为参照，避免只胜过较弱的 Night 基线。

SVOX test、Tokyo247 Night 与 RobotCar public 的历史结果已经被查看，新结果属于探索性比较。开发阶段冻结候选配置和权重哈希后，再统一运行这些外部评估；不据其结果追加权重搜索。RobotCar public 使用自定义公开位姿检索协议，不等同官方定位成功率。

同一 Query 的模型差异采用配对比较；存在路线/场景相关性时按可信分组 bootstrap，无分组时注明区间可能偏窄。严格确认贡献还需未用于选择的独立测试场景；不能把已反复查看的 test 重新称为完全未见。

### 10.3 成功与停止条件

| 结果 | 结论与下一步 |
| --- | --- |
| B 在多个 seed 优于 A/C，外部表现无一致退化 | 支持在当前设置下，额外真实配对监督带来增量价值 |
| B/C 都优于 A，B 与 C 相当 | 额外辅助训练有效，真实来源的独特收益尚不明确 |
| 仅真实配对余弦改善，外部 Recall 不改善 | 不认定泛化成功，检查目标判别性和场地过拟合 |
| B 下降、D 改善 | 真实约束可能过强，优先复核低权重方案 |
| B/D 都下降 | 检查图对、预处理、采样和 Teacher，暂缓扩大真实数据训练 |
| 只改善 SVOX、Tokyo 等持续退化 | 报告数据集依赖性，不宣称稳定跨域提升 |

一两张净增、单 seed 最优或只有训练 loss 下降，不足以认定成功。报告对 R@5/10 的影响，不能只挑有利指标。

阶段一最多支持“额外真实昼夜配对监督改善了当前训练设置下的真实夜间检索”。**不能仅凭 B 提升就声称纠正了 Qwen 生成域偏差，或证明同地点配对优于无配对真实数据。** 改善也可能来自真实成像覆盖、额外场景监督或正则化，机制需要后续专门对照。

## 11. 实现清单与资源约束

| 工作项 | 已有基础 | 阶段一新增内容 |
| --- | --- | --- |
| 数据准入 | 清理清单、`DarkDrivingPairs` | 配对审核、场景分组、隔离清单、独立采样器 |
| 预处理 | 模型原有 bilinear/normalization | 真实图直接复用，避免双重缩放和归一化 |
| Teacher 目标 | GSV Day bank 缓存思路 | 独立真实 Day 目标缓存及签名，不写虚构 place_id |
| 训练循环 | Qwen/Source 串行反传 | 在 optimizer update 之前新增 Real 或额外 Qwen 分支 |
| 记录与评估 | 原有日志及固定 Gallery 脚本 | 分域曝光、梯度诊断、实验计划、多 seed 统计 |

实现后验证：真实分支关闭时复现 A；只有 blocks 8–11 有可训练参数和梯度；Teacher/SALAD 参数及 Gallery 输出不变；真实权重为零或无有效样本时行为正确；所有分支在同一次更新前完成反传；累计窗口与末尾修正正确；额外采样不改变主 GSV 流；缓存不会跨划分或错误预处理复用。

训练和模型评估统一经 `scripts/run_bounded.py` 串行执行，依据实际可用 RAM 设置任务限制和余量检查。首轮 fast 配置使用 7 GiB 上限、6 GiB high、2 GiB 系统余量，不将设备的 32 GiB 总量当成全部空闲。各分支及时释放计算图，缓存按需要分批读取；若内存不足，先减小驻留微批并保持相同有效分支预算。

不承诺新增分支仍有旧实验的峰值内存，也不将 12/10 的视图比例直接等同于实际耗时比例。分别记录在线训练、目标缓存构建及评估开销，所有输出使用新目录保留旧结果。

执行顺序：数据审核与划分 → 最小批次/冻结检查 → seed 42 的 A/B/C → 必要时 D → 关键比较多 seed → 冻结配置后统一外部评估。

## 12. 后续优先方向：Token 风格泛化

TFS-ViT 用中间 Token 统计混合扩展训练域；GUIDE 用冻结辅助表示发现伪域并辅助分类。前者没有验证 DINOv2–SALAD 检索，后者默认主模型甚至是 ResNet-50，不能将两者理解为“换更先进 ViT 就能解决问题”。详细公式、源码差异与证据见 [论文及实现分析](docs/TOKEN_STYLE_DG_REVIEW.md)。

### 12.1 先验证普通 Token 增强

只在 Qwen 主分支、零起算 block 7 输出到 block 8 输入之间增强。Patch 为 `F[B,256,768]`，沿 256 个 Patch 计算每通道 FP32 均值/标准差。保留网格位置，不把供体的空间内容搬入；CLS 排除统计和直接替换。供体先来自当前 Qwen microbatch 的不同样本，统计停止梯度。

\[
\mu_\lambda=(1-\lambda)\mu+\lambda\mu_d,\quad
\sigma_\lambda=(1-\lambda)\sigma+\lambda\sigma_d,\qquad
F'=F+M\odot\left[\sigma_\lambda\frac{F-\mu}{\sigma}+\mu_\lambda-F\right].
\]

每个主 Qwen 样本独立以概率 p 启用，启用后 M 每图随机选择比例 d 的 Patch。首个未验证候选可固定 `p=0.5、d=0.3、λ=0.2×(1−α)、α~Beta(0.1,0.1)`，不同时搜索层、强度和 attention。小 λ 不保证小扰动：接收端方差低时 `σλ/σ` 仍可能很大，须先在训练样本上诊断实际增益与 `ΔF`；若需截断，在评估前固定统一规则。方差用 `correction=0` 加 epsilon/下界；`p=0`、`d=0` 或强度为零时直接走原路径，不读供体、不消耗增强 RNG。它是 VPR 改造，不是 TFS 作者实现的严格复现：作者随机版本可能替换 CLS，统计也包含 CLS。

原有 Global/InfoNCE/Relation、Source Night 和真实配对损失保持原样，不增加额外前向或新一致性损失。只有主 Qwen 分支启用；Source、真实输入、Day Teacher、SALAD 参数、固定 Gallery 和推理不增加此操作。后四层继续接受原始 Day 描述子的检索监督，检验其能否在扰动下保持与固定图库兼容。

| 组别 | 真实配对余弦 | Qwen Token 增强 |
| --- | --- | --- |
| T0 | 关闭 | 关闭 |
| T1 | 与 B 相同 | 关闭 |
| T2 | 关闭 | 普通批内 Qwen 统计混合 |
| T3 | 与 B 相同 | 与 T2 相同 |

全部使用相同 fast 后端与初始化、主输入/Source 随机流、有效预算、908 次更新及学习率。T1/T3 真实曝光一致。严格比较应重跑 fast T0；旧 A 的 CPU bank/检查点路径只能作历史参考。混合池由实际 microbatch 决定，梯度累积不扩大一次前向的供体池；B=1 时跳过。先开发验证，再对关键比较补齐相同 3 seeds，不能把单 seed 一两张净增当成成功。

**首轮实测：** T0/T2 均为 617/668/676，T1/T3 均为 622/667/675（702 Query 的 R@1/5/10 整数命中数）；两种底座上增强均为 0 修复、0 新增错误。增强对 50.31% 主图实际启用，保存权重和少量精确排名确实改变，不能解释为开关失效；当前配置没有提供检索收益。优先复核真实配对的多 seed 信号；若继续增强，先诊断合成/真实统计覆盖和实际扰动，再决定单因素实验，不自动推进复杂组合。

### 12.2 再考察真实统计来源

若普通增强有效，比较匹配大小、采样与公式的 Qwen 统计库和训练真实 Night 统计库。真实库是额外的真实训练数据访问，即使不增加在线反传视图也须明确报告。固定前缀下 9520 图单层 `μ/σ` FP32 约 55.8 MiB，是容量估计，不是实测峰值。缓存必须绑定前缀权重、层号、预处理、审核清单和统计定义。

逐通道统计混有场景内容；同 λ 也不保证不同供体产生同幅度扰动。记录实际特征变化、SALAD 分配与开发 Recall，必要时做同扰动强度控制。保留 Token 空间顺序不等于地点语义保持，分类高 attention 也未必代表 VPR 稳定结构，因此 ATFS 延后单独研究。

### 12.3 GUIDE 用于独立的覆盖诊断与采样

先保持 B 的模型、损失和真实样本总数不变，单独比较原采样与伪域均衡采样。优先用已有 DINO 的配对 `Δμ、Δlogσ` 聚类；低成本表示不足时，再比较离线冻结 SD-2.1 的 `Ψ(Night)−Ψ(Day)`。差分只是降低内容干扰的假说，不保证纯风格，也不等于 Qwen→真实偏差。

只在训练池拟合标准化、PCA、聚类和中心；伪域不当 place label 或地理划分。检查是否只是按建筑、道路、场地分组。扩散特征只用于训练采样/供体选择，不直接加到 DINO 通道、不拼接到 8448 维 SALAD 描述符，推理不新增模型。

GUIDE 本地脚本会重算测试群体中心，与正文固定训练中心描述有差异；论文还按 test accuracy 选 K 报部分结果。本项目不沿用这些评估行为，只按开发协议冻结配置。先分别验证采样和 Token 增强，再讨论组合；不把多个变化合并成无法解释的收益。

## 13. 远期备选：配对统计残差迁移

只有普通 Token 增强、真实统计供体或独立低成本诊断提供支持时，再研究如何把少量真实配对信息迁移到更多 GSV 合成夜图。暂称 PRSA（Paired Residual Statistics Augmentation，配对残差统计增强），不作为已经成立的新方法。优先级低于第 12 节，不先为全部 DarkDriving 批量生成 Qwen 图。

对 DarkDriving 训练 Day 生成与 GSV 尽量一致流程的 Qwen Night，构造 \((d_i,q_i,n_i)\)，记录生成版本、prompt、seed、图像哈希与结构审核。开发/测试图不进入残差库，统计变换也无法修复生成器改变的建筑和地点内容。

初始在冻结 blocks 0–7 的末端提取 Patch Token，排除 CLS，用 FP32 沿空间 Token 维计算每通道均值 \(\mu\) 和标准差 \(\sigma=\sqrt{\operatorname{var}+\epsilon}\)。定义逐对有界残差：

\[
a_i=\operatorname{clip}\left(\frac{\mu(n_i)-\mu(q_i)}{\sigma(q_i)},-a_{\max},a_{\max}\right),
\quad
b_i=\operatorname{clip}\left(\log\sigma(n_i)-\log\sigma(q_i),-b_{\max},b_{\max}\right).
\]

在 GSV/Qwen Student 的 block 7 后、block 8 前采样完整残差向量：

\[
F'_g=\mu_g+\beta\sigma_g a_i+(F_g-\mu_g)\exp(\beta b_i).
\]

统计与残差停止梯度，\(\beta=0\) 恒等，CLS 在注入点不变。只在训练时扰动 Qwen 分支，真实分支和推理关闭；缓存绑定模型、冻结范围、层定义和预处理。若此前层开始更新，必须重新设计缓存与特征坐标系，不能继续静态复用。

必须保留逐对残差，而非只用全局平均差，因为：

\[
\frac1M\sum_i[\mu(n_i)-\mu(q_i)]
=\mathbb E[\mu(n)]-\mathbb E[\mu(q)]
\]

与配对置换无关，平均 log-std 差同理。仅平均修正不能证明利用了配对优势。

在同一个已冻结训练协议上比较：无统计模块、普通真实统计插值、正确配对残差、打乱配对残差、同强度随机扰动。保持真实训练曝光和实际特征扰动强度可比，检验正确配对是否有额外收益。阶段一的 A/B/C/D 编号不复用于这些后续消融。

AdaIN 已研究均值/方差对齐，MixStyle 已研究统计混合，因此通道仿射本身不构成新贡献。DINOv2 的通道统计也混有内容语义，LayerNorm 不保证消除或保留有效风格变化。可能的研究贡献在于：**配对残差的迁移收益，能否超过普通增强与无配对对照，并在固定 Gallery 的真实 VPR 中成立。** [AdaIN 原论文](https://arxiv.org/abs/1703.06868)、[MixStyle 作者实现](https://github.com/KaiyangZhou/mixstyle-release)

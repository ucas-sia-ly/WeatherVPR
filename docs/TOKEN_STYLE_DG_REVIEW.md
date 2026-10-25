# TFS-ViT 与 GUIDE 对 WeatherVPR 的启发

更新：2026-10-25。本文核对两篇本地论文、GUIDE 本地源码、TFS 作者固定版本源码及当前 DINOv2–SALAD 前向。部分 Patch 增强已完成 T0–T3 首轮，见 [阶段二实现](NIGHT_TOKEN_STAGE2.md) 与 [结果评价](NIGHT_TOKEN_STAGE2_RESULTS.md)：该配置未获得开发 Recall 增益，保持默认关闭。下文其余路线仍为研究假说；真实统计库、扩散伪域与配对残差未验证，未提取扩散特征。

**建议先验证训练时的部分 Patch 统计增强，再研究真实统计供体与伪域采样。** 两篇工作的价值在于域信息的使用方式，并不是更换一个更大的 ViT。当前首轮真实配对结果只有 SVOX 小幅增量，不能据此宣布合成—真实偏差已被纠正。见 [阶段一结果](../outputs/stage1_results_20261025/report.md)。

阶段一原有的多 seed 复核与路线审核仍待完成；新研究路线不替代这些证据，也不改写旧实验结果。

## 1. 两篇论文实际解决的问题

| 项目 | TFS-ViT | GUIDE |
| --- | --- | --- |
| 核心操作 | 混合中间特征的均值/标准差，只替换部分 Token | 冻结辅助编码器发现伪域，映射伪域中心后与主模型特征拼接 |
| 主任务 | 类别分类的领域泛化 | 类别分类的领域泛化 |
| 实验主干 | DeiT-Small、T2T-ViT-14 | 默认主任务模型为 AugMix ResNet-50；辅助比较 SD、DiT、DINOv2、MAE、CLIP 等 |
| 是否新增生成图 | 否 | 否；扩散模型用于特征提取 |
| 原法的推理代价 | 风格化关闭，没有新增推理模块 | 新查询仍需辅助编码器及伪域分配 |
| 对本项目最直接的启发 | 扩展训练外观，保持原有 Day 检索监督 | 审查训练风格覆盖，改进供体或训练样本采样 |
| 尚未验证的部分 | DINOv2–SALAD 地点检索、固定 Day 图库 | 以 DINOv2–SALAD 为主任务模型的地点检索及固定图库兼容性；论文已比较 DINOv2 辅助表示 |

TFS-ViT 正文第 4–5 页在五个分类基准上报告三 seeds、源域验证选择结果。表 1 的 DeiT 平均准确率为 ERM 64.8、TFS 66.95、ATFS 66.37；ATFS 并非一直优于随机 TFS。GUIDE 正文第 4 页明确区分主任务编码器 Φ 与冻结伪域编码器 Ψ，DiT 是其中一种辅助表示，不能将其收益解释为“ViT 越先进越有效”。这些数值不是 VPR Recall，也不是对本项目的收益估计。

## 2. TFS：统计轴、位置和作者代码细节

依据正文第 3–4 页 §3.2–3.3、Eq. 1–3、Fig. 1–2：

- 激活为 `X[B,S,C]`。每图、每通道沿序列 S 求统计，得到 `[B,1,C]`；不是跨 batch 统计，也不是每个 Token 沿 C 做 LayerNorm。
- 对样本 i 与供体 π(i) 的均值、标准差分别做凸组合，α 来自 `Beta(0.1,0.1)`；用组合统计重新缩放原图的标准化激活。
- 在 block K 输出与 K+1 输入之间，只将比例 d 的 Token 改成对应的风格化结果。Token 的位置索引不交换，也不是把另一张图的 Patch 内容搬进来。
- 每次训练前向选择若干层。ATFS 使用当前层、跨 heads 平均的 CLS→Patch 注意力选择高分 Patch；它不是给前景/背景分别计算一套统计。
- 风格化在推理关闭，原分类交叉熵之外不要求新增风格损失。

作者代码固定版本为 `a7eba85d2f6486a0de0ac0252a4731f7cbac4cda`。以下细节决定了我们的版本只能称为“受 TFS 启发的 VPR 改造”：

| 问题 | 作者实现 | 本项目拟采用 |
| --- | --- | --- |
| CLS 统计 | 完整序列统计，包含 CLS | 只统计 Patch |
| CLS 替换 | 随机 TFS 可能选 CLS；ATFS 明确不选 | 不直接修改 CLS |
| 方差 | `x.var(dim=1)` 默认样本方差；公式分母为 S | 明确 `correction=0`，FP32，加 epsilon/下界 |
| 供体 | `randperm(B)`，允许自配，不保证跨域 | 首轮仅主 Qwen microbatch 内选择不同样本；B=1 明确跳过 |
| 随机强度 | 每图一个 α，共享所有通道/Token | 保留每图共享原则，先限制最大扰动强度 |
| 注入层 | 默认从前 `floor(0.7*depth)` 层抽样 | 首轮固定 block 7 后、block 8 前 |

证据：[DeiT Token 选择](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/blob/a7eba85d2f6486a0de0ac0252a4731f7cbac4cda/domainbed/pretrained_models/DeiT_models/models.py#L46)、[统计与 detach](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/blob/a7eba85d2f6486a0de0ac0252a4731f7cbac4cda/domainbed/pretrained_models/DeiT_models/models.py#L205)、[训练监督](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/blob/a7eba85d2f6486a0de0ac0252a4731f7cbac4cda/domainbed/algorithms.py#L181)。论文“前 75% 层”与代码默认 0.7、DeiT 前 8 层并不严格一致，应按采用的实现写清。

论文第 7 页表 4 报告其设置下约 1.08% 训练时间增量，但这不能作为我们蒸馏、数据加载和缓存流程的实测开销。

## 3. GUIDE：值得借鉴的机制和不能照搬的部分

论文正文第 3–5 页与补充第 16 页算法 1 的流程：

1. 用冻结 Ψ 提取训练特征，K-Means++ 得到伪域中心 `c_k`。
2. 按训练簇计算当前主模型 Φ 的特征均值 `m_k`。
3. 用 RBF kernel ridge regression 学习 `T(c_k)≈m_k`。
4. 将 `[Φ(x),T(c_k)]` 送入分类器；随 Φ 更新，周期刷新映射，初始聚类保持固定。
5. 论文描述的新样本推理使用训练时的伪域中心。

SD-2.1 的例子是空提示、t=50、UNet `up_ft:1`；无需生成新图片、训练扩散模型或做 textual inversion。主模型不是必须采用 ViT。论文表 3 在 PACS 上 DiT 的域 NMI 较高，而 TerraIncognita 上 SD 明显高于 DiT；作者也指出预训练数据等不同，不能把差异只归因于架构。

本地 GUIDE revision：`24fd5474fd4c51e6c97d9c1c66bcf34564307f75`。研究参考时须记录以下差异：

- [algorithms.py](../third_party/GUIDE/domainbed/algorithms.py#L254) 明确拼接 image/clust 特征并做分类。直接照搬会改变当前 8448 维检索坐标，不能直接与既有 Day 图库比较；扩展到检索需要另行设计兼容映射和目标。
- [train_precompute.py](../third_party/GUIDE/domainbed/scripts/train_precompute.py#L400) 将测试样本按训练 KMeans 分组后，重新用整组测试特征计算中心，再送入训练拟合的映射。这是依赖测试群体的传导式特征处理，与正文的固定训练中心推理描述不同。类别标签未用于中心计算，不能称为测试标签泄漏。我们的方案不使用这种测试群体统计。
- 同脚本 420–485 行用当前 Φ 的训练簇均值作回归目标；508–540 行在初始及约 1、2、4、8…epoch 刷新。只抄聚类而忽略主空间映射，并非复现 GUIDE。
- 论文第 5 页明确说按各域 best test accuracy 选择 K 来报告表 6；这与本项目只用开发集选择配置的规则不同。`GUIDE-BEST` 还取不同数据集上 SD/DiT 中的较优结果，不应当成单一固定配置的收益。
- 论文 K 公式与“200 为上限”的文字有矛盾；本地脚本也没有相同的 200 截断。无需把这套依赖类别数的 K 规则移植到地点检索。
- [diffusion_features.py](../third_party/GUIDE/domainbed/feature_models/diffusion_features.py#L649) 的 SD 路径确实向 VAE latent 加噪；同文件 792–808 行的 DiT 路径却注释了加噪，直接输入干净 latent 和 t。两者不能未经核查当作相同噪声提取协议。

因此本项目借鉴 GUIDE 的首选方式是：**离线域表示 → 训练覆盖诊断/采样**。不新增线上扩散分支，不拼接域中心到最终描述符，不在任何外部查询上拟合中心或标准化。

## 4. 最小 VPR 改造

### 4.1 插入点与空间约束

当前实际前向在 [workflow/model.py](../third_party/salad/workflow/model.py#L142)。Late4 的参数冻结在 [optimization.py](../src/weather_vpr/training/optimization.py#L100) 完成；其 `num_trainable_blocks=12` 是为了保持完整可微前向，不能据现有 `split` 变量寻找冻结边界。

显式选择零起算 block 7 输出、block 8 输入之间。224×224 的 B/14 此处为 `[B,257,768]`；取 256 个 Patch，排除 CLS。空间索引保留，注入时 CLS 不变，但后四层注意力仍会改变它。若以后换成带 register Token 的模型，也必须显式排除这些 Token。

### 4.2 混合系数受限的部分 Patch 统计增强

令 `F∈R^{B×256×768}`，沿 256 个 Patch 求 FP32 的 detached `μ,σ`。供体同样只提供 detached `μ_d,σ_d`，不传入其空间激活。

\[
\lambda=\eta(1-\alpha),\quad \alpha\sim\mathrm{Beta}(0.1,0.1),\qquad
\mu_\lambda=(1-\lambda)\mu+\lambda\mu_d,\quad
\sigma_\lambda=(1-\lambda)\sigma+\lambda\sigma_d,
\]
\[
\widetilde F=\sigma_\lambda\frac{F-\mu}{\sigma}+\mu_\lambda,\qquad
F'=F+M\odot(\widetilde F-F).
\]

每个主 Qwen 样本独立以概率 p 启用；启用后，`M[B,256,1]` 每图独立随机选比例 d 的 Patch。同一图的供体、λ 在通道和所选 Patch 间共享。σ 使用明确下界，统计与变换计算使用 FP32，之后按主干精度转回。`p=0`、`η=0`、`d=0` 或非指定训练分支时直接返回原 Tensor，跳过 RNG 与供体访问，避免仅靠浮点往返实现“恒等”。

首个候选可预先固定为 `p=0.5、d=0.3、η=0.2、单层 index=7`。这些是未验证的小混合系数起点，不是论文最优参数，也不保证适合 VPR。η 只限制 λ，不限制实际仿射增益 `σλ/σ`：接收端方差小时仍可能强放大。先用训练数据诊断 gain、标准化均值偏移和实际 `ΔF`；若需截断增益/偏移，须在检索评估前固定同一规则并用于全部增强组，将其明确记为额外改造。先不同时搜索层数、attention 策略和新损失。

### 4.3 训练和推理

仅增强 Qwen 主分支的已有前向，不增加干净/增强双前向和一致性新损失。Global、InfoNCE、Relation、Source Night 及其权重保持原样；若使用 B 底座，真实配对余弦也保持原样。Source Night、真实 Night、Day Teacher 与推理均关闭 Token 增强。

通过显式分支参数传递到独立 Night 副本，不能只检查 `self.training`，否则三个训练分支都会被增强。也不修改共享 Day backbone 或 SALAD。原始 Teacher 目标与 Day 图库缓存可继续复用。

### 4.4 为什么不能把它当作“只改风格”

- DINOv2 的 LayerNorm 沿每个 Token 的 C 维归一化，统计混合沿 Patch 维，逐通道变换通常不会被下一层 LN 完全消除；残差路径也保留变化。具体有效强度仍须测量。
- [SALAD.forward](../third_party/salad/models/aggregators/salad.py#L241) 同时将 Patch 送入 `cluster_features` 和 `score→Sinkhorn`。增强会改变描述内容及聚类分配，并不保证地点信息不变。
- 通道统计包含天空、建筑、道路、植被比例。保留网格位置不等于保持地点语义；对部分 Patch 强增强也可能产生不自然的局部混合。
- 分类中的高 CLS attention 不等于 VPR 的稳定结构。车灯、车辆、灯牌可能很显著。现有 [attention.py](../src/weather_vpr/models/attention.py#L160) 只捕获末层注意力，不能用它给更早的在线注入直接选 mask。因此先用随机 Patch，ATFS 单独研究。

## 5. 实验顺序：先回答机制，再增加真实统计

### 第一步：普通 Token 增强与真实配对是否互补

| 组别 | 底座 | 主 Qwen Token 增强 | 回答的问题 |
| --- | --- | --- | --- |
| T0 | A，同 fast 后端 | 无 | 新实验的匹配基线 |
| T1 | B，同 fast 后端 | 无 | 真实配对基线 |
| T2 | 同 T0 | 批内 Qwen 统计供体 | 普通 Token 增强是否有效 |
| T3 | 同 T1 | 与 T2 相同 | Token 增强是否与真实配对互补 |

首轮供体只来自当前主 microbatch 的 Qwen 特征，不需要新统计库或生成 DarkDriving/Qwen。当前主 microbatch=4、梯度累积 K=2；混合池是 4，不能当作有效 batch 8。Source/extra 的 microbatch=1 不参加混合。不同样本的供体可用无固定点排列；不足 2 个样本时明确跳过并记数。

所有组使用相同 fast 后端、初始化、主输入与 Source 随机流、908 次更新、学习率、损失权重和原始 Day 图库；T1/T3 的真实曝光也一致。旧 A 使用 CPU bank/检查点而 B/C 使用 fast 后端，严格新比较应补 T0，不能把旧 A 与新方案的微小差值全归因于方法。

先只在开发协议上比较预定配置，若有可复现信号，再补齐相同 3 seeds，报告 R@1/5/10、整数命中数与逐查询修复/退化。外部集合已有历史查看，只作冻结后的探索性报告，不能据其扫描强度。

### 第二步：真实 Night 统计库

若普通 Token 增强有效，固定底座、公式、强度和预算，比较：

1. 训练 Qwen 统计库供体；
2. 训练真实 Night 统计库供体；
3. 必要时按供体数、采样概率和实际扰动幅度匹配的控制。

两种库均用同一冻结前缀和规范预处理提取，供体存完整 `μ/σ` 向量。库大小应匹配，例如从两池预先固定等量训练供体；不能在换真实来源时同时增加供体数或改变混合强度。即使同 λ，不同统计来源也可能产生不同实际扰动，须记录归一化特征变化量，并在后续机制对照中匹配。

真实统计库明确使用真实训练数据，但不增加真实 Night 的在线反传视图。它检验真实外观统计能否转移到 GSV 内容，不证明“同地点配对”有独特贡献，也不能把绝对统计当作已去内容的风格。

单层完整 9520 图的 `μ/σ` FP32 库为 `9520×2×768×4` bytes，约 **55.8 MiB**，只是容量计算，不是实测峰值。与存完整 Patch 激活相比，更适合当前设备；提取时间、文件审核、GPU 与 RAM 峰值仍要单列测量。

### 第三步：GUIDE 启发的伪域采样

先单独保持 B 的模型、损失和真实样本总曝光不变，只比较原采样与伪域均衡采样；不要同时叠加 Token 增强。若有收益，再用于统计库供体选择。

优先试已有冻结 DINO 的成对统计差分：

\[
r_i=[\mu(n_i)-\mu(d_i),\;\log\sigma(n_i)-\log\sigma(d_i)].
\]

只有低成本表示不足以描述照明覆盖时，再比较离线冻结 SD-2.1 的 `Ψ(n_i)−Ψ(d_i)`，先固定 t=50、`up_ft:1`、提取种子及噪声策略。该差分试图降低内容干扰，**并不保证内容被消除**。它描述 Day→真实 Night 变化，不等于 Qwen→真实 Night 残差。

对差分聚类是我们的假说，不是 GUIDE 原法。检查每簇图例、近重复/内容组分布和照明因素；具备可信标签时才计算域/地点相关性。单纯按数据集或场地分开的高纯度，并不是有效风格解耦证据。

所有标准化、PCA、KMeans 和中心仅在训练池拟合；K 预先固定或只由开发协议选择。伪域不作为 place label、不用于制造地理划分。扩散特征的坐标也不能直接当 DINO 通道的 μ/σ 使用，二者并不对齐。

如果只用于离线采样，推理仍为一个 Night DINOv2 + 原 SALAD，新增线上模型成本为零；离线特征抽取并不免费。原版 GUIDE 的线上辅助编码成本不能用离线预计算来掩盖。

## 6. 工程验收与停止条件

- 关闭增强时描述符、主随机流和训练行为回归旧路径；eval 无供体访问、不消耗增强 RNG。Day 权重、原始图库及缓存签名不变。
- 检查只在 Qwen Night 路径、指定 block 后生效；CLS 未直接替换，Patch 数与索引不变，统计轴正确，弱/零方差输入有限。
- 检查 Late4 梯度、冻结范围、最终 Norm 与 SALAD 输入梯度，不能为了冻结参数而把下游包进 `no_grad`。
- 若新增静态统计库，签名包含冻结前缀权重、层号、pre/post Norm、排除特殊 Token、预处理、图像/审核清单、方差定义和 epsilon。最终 `return_aux` 特征不能冒充 block 7 输出；解冻前缀后静态库失效。
- 增强 RNG 与主增强/采样隔离；checkpoint 重算若覆盖随机模块，复用完全相同的供体、mask 和 λ。记录供体 ID、层号、有效 Patch 数、应用概率和实际扰动量。
- 有界诊断测注入前后、下一层 LN 后、最终描述符和 SALAD 分配的变化。数值变化、配对余弦变好或 t-SNE 分开，都不能替代开发 Recall。
- 若只有真实训练场地诊断变好、开发检索不改善，先停下复核地点信息损伤与统计内容污染；不通过扩大强度、层数和模型规模无限搜索。

配对 Qwen→真实统计残差继续保留为远期备选，见 [idea.md 第 13 节](../idea.md#13-远期备选配对统计残差迁移)。只有在普通增强、无配对真实统计和扰动强度控制之后，才能评估正确配对本身的增量。

## 7. 来源

- [TFS-ViT 本地论文](<../third_party/Noori 等 - 2024 - TFS-ViT Token-level feature stylization for domain generalization.pdf>)：方法第 3–4 页；表 1 第 5 页；Token/层选择消融第 6 页；计算开销第 7 页。作者代码链接与固定 revision 见第 2 节。
- [GUIDE 本地论文](<../third_party/Thomas和Ghadiyaram - What’s in a latent Leveraging diffusion latent space for domain generalization.pdf>)：图 2/方法第 3–5 页；域表示分析第 5–7 页；主结果/聚类第 8 页；映射消融第 13 页；算法第 16 页。[本地源码说明](../third_party/GUIDE/README.md)、[作者仓库](https://github.com/XThomasBU/GUIDE)。
- [当前 Night 前向](../src/weather_vpr/models/adapter.py#L244)、[实际 DINO 前向](../third_party/salad/workflow/model.py#L142)、[Late4 冻结](../src/weather_vpr/training/optimization.py#L60)、[SALAD 聚合](../third_party/salad/models/aggregators/salad.py#L241)。

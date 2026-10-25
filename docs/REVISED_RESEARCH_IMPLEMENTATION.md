# 修订研究方案：实现与实验账目

更新：2026-10-25。研究依据为用户提供的 `WeatherVPR_idea_revised_20261025_copy.md`；主方案已同步到 [idea.md](../idea.md)。原始修订文件与历史实验保持可追溯。

## 三条独立研究轴

- H1：真实配对监督的增量。固定原始 Day 初始化、Late4、Qwen/Source 流与优化预算，复核 A/B/C 的 seed 42、43、44。
- H2：真实夜间统计覆盖。先进行训练池内的冻结模型测量；Qwen 批内 TFS 的空结果不否定真实供体。距离差异也不能直接证明光照风格或泛化收益。
- H3：更多 GSV 地点。固定 7,263 对 Qwen 主流，独立抽取有 CSV 地点标签的 Day 构造 Source Night。Source 使用该档独立 bank，Qwen bank、Teacher、SALAD 与评估 Gallery 均固定。

## 数据清单与准入

执行 `scripts/prepare_research_sources.py` 后生成 [完整审计](../outputs/revised_plan_20261025/data_inventory/inventory.json)。原数据未搬移或修改。

| 来源 | 可用训练图像/配对 | 已知分组 | 用途与限制 |
| --- | ---: | ---: | --- |
| GSV/Qwen Night | 7,263 | 6,441 个 CSV 地点 | 原 Qwen 主流及合成统计来源 |
| 对应 GSV Day → Source Night | 7,263 | 同上 | 原低照度构造及合成统计来源 |
| DarkDriving | 9,520 对 | 9,286 个精确内容连通组 | 自动准入、整场地 train-only；没有认证的独立路线 |
| BDD100K train-night | 27,971 | 单图标识 | 仅官方 train 标注中 night 与本地图像的交集；不把单图标识当路线 |
| NightCity train | 2,998 | 单图标识 | 仅官方 train 图像与标签交集，排除 val/test |
| G1 Day Source 池 | 7,263 | 6,441 个地点 | 保留当前全部 Day 视图 |
| G2 Day Source 池 | 17,263 | 11,441 个地点 | G1 加 5,000 个新地点，每新地点 2 视图 |

GSV 原 CSV 共 529,506 行、64,394 个地点，所有对应本地图像均存在。合并 heldout、合成验证和校准等排除证据后保留 480 个排除地点，扩展池与之无交集。G1 原复制图像还逐字节匹配原 GSV 图；新池绑定原 CSV 行、排除证据和清单签名。消费新增图像时计算实际 SHA-256，并绑定 Teacher 缓存；默认盘点不预先哈希所有新增图像。数量增加不代表新的地理独立性证明。

DarkDriving 9,520 项全部为 `automatic`，不能称为逐对人工确认。文件名前缀共 23 个；24 个精确内容组跨前缀复用。组大小分布为 9,064 个单例、214 个双例、4 个三例、4 个四例，见 [补充审计](../outputs/revised_plan_20261025/darkdriving_source_audit.json)。前缀和内容连通组都不能当作地点标签。已有约 200 对缩略图抽查不替代路线/近重复复核。训练仍只使用配对余弦，不将真实样本作为互斥负地点。

现有真实采样器平衡的是审核记录的 `scene_group`，但当前记录只有一个 `whole_closed_driving_field_train_only` 组，因此实际是在全场地图像行内循环抽样，未实现独立路线或精确内容组均衡。多 seed 复核保留这个协议；若改采样，应单开对照，不能混入已有 seed 结果。

BDD 的磁盘容器名称包含 `testA/testB`，但它们位于本地 `images/train` 内；准入依据是官方 **train 标注的精确图像名**，而不是这些内部容器名。诊断前再次校验原标注文件哈希和选中图像字节。NightCity 同样绑定训练标签及数据元信息。这里只对现有本地数据作研究测量，未作重新分发许可的法律认定。

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
"$PYTHON" scripts/run_bounded.py --max-gib 7 --high-gib 6 --reserve-gib 2 \
  --log runs/research_inventory.memory.jsonl -- \
  "$PYTHON" scripts/prepare_research_sources.py --output outputs/new_inventory \
  --real-review runs/night_real_stage1_20261025_053154_seed42/real_review.jsonl \
  --include-bdd --include-nightcity --include-gsv \
  --gsv-new-places 5000 --gsv-views-per-place 2
```

输出目录必须为空。默认盘点不对全部 BDD/NightCity 图像作解码认证；`--verify-images` 可进行全量严格解码及 SHA。统计提取会严格核验实际选中的图像。

## P0：统一后端多 seed

`scripts/run_stage1_multiseed.py` 冻结实验计划、代码和输入签名，顺序执行每个 seed 的 A/B/C 与 SVOX dev 评估。每组独立从原始 Day 初始化，使用最终权重；不继续旧 Night adapter，不运行历史外部 test。

实际固定配置为 BF16、microbatch 4、累积 2、Source 每 microbatch 1、B/C 额外辅助 1，908 次更新；LR 1e-6、cosine、warmup 32，GPU bank、4 workers、prefetch 1、关闭激活重计算。有效完整窗口为主 8 / Source 2 / 额外辅助 2。保留 microbatch 是因为它会影响对比集合与 Token 供体，不能仅以有效 batch 相同宣称等价。

seed 42 引用同 fast 后端的 T0/T1 作为 A/B，C 引用 `C_fast`，不重复算作新 seed。汇总保留种子间离散程度、整数命中与逐 Query Top-k 修复/退化；源流签名比较样本 ID 和操作计数，不等于独立哈希每张增强后的图像。

三 seed 已完成，见 [本轮 P0 评价](../outputs/revised_plan_20261025/p0_multiseed.md)：B−A 的 Top-1 为 +5/+7/+2，平均 +0.665 个百分点；Top-5 为 −1/−2/−2。完整输入与代码哈希见运行目录；另保存与冻结清单逐文件匹配的 `code_snapshot.zip`，便于在后续修正独立 Source 加载器后复现旧 P0。

## P1：冻结前缀域覆盖与匹配干预

本轮已完成，见 [定量结果与训练内容复核](../outputs/revised_plan_20261025/domain_coverage/report.md)。每域 512 张，block 7 的参考阈值包含率为 DarkDriving 94.92%、NightCity 95.12%、BDD 65.62%；PCA32 只保留参考方差的 54.00%，不能解释成完整空间覆盖。64 张接收图中 29 张实际启用，真实供体的实际扰动与 Qwen 并未严格匹配。当前判断是继续训练池内容控制，暂不开启 P2。

`scripts/analyze_domain_coverage.py` 只加载已登记的原始 Day DINOv2–SALAD，冻结全部参数。对五个来源各取最多 512 个组均衡训练样本，在零起算 blocks 3、5、7 的输出提取 Patch-only FP32 `μ,logσ`。CLS 排除。Source 使用实际训练相同的 epoch 0 构造规则。

标准化、PCA 和 kNN 均仅拟合合成参考 fit 组，阈值取合成 heldout 的距离 p95。Qwen 与 Source 选择同一批场景，同地点永不跨 fit/heldout。报告不同真实来源落在参考阈值内的比例、亮度分层及离群样本。不同道路、相机和内容仍会造成真实与合成来源的统计差异，单图分组也不能消除路线相关性。

对相同 64 张 Qwen 接收图使用等量各来源供体，固定供体索引、Patch mask 与 λ，测量实际相对 RMS `ΔF`、gain、描述符余弦变化和 SALAD 分配变化，同时报告全部接收图与真正启用图的统计。**相同 λ 不保证实际扰动匹配**；相对原 Qwen 描述符的漂移也不证明与干净 Day 的地点对齐。本轮记录差异，不根据 Recall 追调强度，也不自动触发真实供体训练。缓存绑定模型、代码、预处理、来源证据和实际图像字节。前后模型状态哈希必须相同，梯度为空，optimizer updates 与 Recall 评估次数都为 0。

```bash
"$PYTHON" scripts/run_bounded.py --max-gib 7 --high-gib 6 --reserve-gib 2 \
  --log runs/domain_coverage.memory.jsonl -- \
  "$PYTHON" scripts/analyze_domain_coverage.py \
  --manifest outputs/revised_plan_20261025/data_inventory/train_sources.jsonl \
  --output outputs/new_coverage/report.json --cache runs/cache/new_coverage.safetensors \
  --device cuda --max-per-source 512 --batch-size 64 --num-workers 4
```

## 独立 Source 规模实验

`configs/night_gsv_scale.sh` 选择 G1/G2 与 A/B。`SOURCE_DAY_MANIFEST`、`SOURCE_DAY_CACHE` 显式区分每档，缓存不能覆盖主 Qwen bank 或真实 Day 目标。独立 Source 输入构造、采样、前向及反传均在专用随机上下文内，防止改变主 Qwen/真实分支随机流。Source 保留原 Global + 多正样本 InfoNCE 权重，不增加 Relation 或真实负样本。

完整 G1/G2×A/B 必须保持主 Qwen 顺序、初始化、Source 曝光量、更新预算一致。G1 独立抽样和历史同 batch Source 在抽样含义上不同，应重新训练 G1-A/B；不能将旧 A/B 直接当作这个规模轴的 G1。

统一队列为 `scripts/run_gsv_scale.py`，不复用旧 A/B 权重。先在新目录使用 `--max-steps 3 --skip-evaluation` 验证并构建缺失 Source 目标缓存；完整运行在计划冻结前要求缓存及签名已存在。两种模式都核验跨组主流、同规模 Source 流、真实曝光与实际地点曝光，完整模式仅评估 SVOX dev。

```bash
"$PYTHON" scripts/run_gsv_scale.py --output runs/new_scale_smoke \
  --max-steps 3 --skip-evaluation
"$PYTHON" scripts/run_gsv_scale.py --output runs/new_scale_seed42 --seed42
```

实际 CSV 的原始 `northdeg` 范围为 −11 到 731；加载器按照原 SALAD 的整数补零命名规则保留，不能以常见角度范围做截断、取模或只允许 −1。四单元真实数据短跑和完整一轮均已完成，321 项检查通过；[总结果](../outputs/revised_plan_20261025/report.md) 中 G1-A/B Top-1 为 621/624，G2-A/B 为 623/624。G2 实际曝光新增地点 792 个，尚无明确规模互补证据。

## 保留新的夜间测试候选

本地 SF-XL 有 `processed/test/queries_night`，盘点为 466 张，数据库及路径清单存在；当前代码和研究记录未发现其检索评估。它暂列最终测试候选，**未提取夜间 Query 描述符、未计算 Recall、未挑选失败案例**。最终使用前还须冻结完整 Gallery、坐标正样本协议、查询有效性、数据重叠及训练历史核验。当前记录不能证明用户或其它工程从未使用过它，因此不直接宣称已获得严格盲测。

## 后续门槛

真实统计供体、GUIDE 扩散特征和配对残差继续默认关闭。P1 的来源差异必须结合内容复核及实际干预强度解释；只有出现可利用证据，才预登记等规模、等曝光、尽量匹配实际 ΔF 的供体训练对照。工程检查、统计距离和训练 loss 均不能代替真实夜间检索收益。

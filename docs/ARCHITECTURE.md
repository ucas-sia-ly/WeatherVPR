# Night 全参数 DINOv2-SGPD 设计

## 模型与梯度

`dino_salad.ckpt` 提供白天训练好的 DINOv2-B/14 和 SALAD。Teacher 保留原始白天 DINOv2；Student 在初始化时精确复制这份主干，然后对**整个 Night DINOv2**进行微调。两支 DINOv2 各有独立参数，SALAD 是共用的冻结聚合器。优化器只接收 Night DINOv2 的 86,580,480 个参数；白天 Teacher 使用 `no_grad`。Night 的 Patch Embedding、位置/CLS Token、全部 Transformer Block 和最终 Norm 都保留反向图，SALAD 虽然冻结仍允许梯度穿过聚合计算。

训练和评估白天 Gallery 时始终使用原始白天主干，因此 Night DINOv2 对齐**原始白天 SALAD 描述符空间**，不需要重训或重建 Gallery。Night 权重单独保存到 `adapter.safetensors`，只包含 Night DINOv2，不包含 SALAD；文件约 346 MB。`adapter_type=lora` 和 `adapter_type=output` 保留先前消融，旧 `runs/night_sgpd_01` 属于 Output。框架预留 rain、fog、snow、sun 接口，目前仅训练 night。

## 分阶段训练

1. `configs/night_full_global.sh` 默认用地点隔离的 night 训练集，仅优化全局 Descriptor Alignment 与跨域 InfoNCE。batch 尽量覆盖不同的 `(city, place_id)`；容量 256 的跨 batch 队列保存 detached 的白天 Teacher 描述符。当前 batch 和队列内的同地点描述符全是正样本，不会充当伪负例。训练日志记录每个 query 的有效负样本数。
2. `configs/night_full_sgpd.sh` 从第一阶段完整 Night 主干初始化，继续训练全局损失与 SGPD。加载时检查天气、原始 checkpoint、预处理和模型类型。若要从原始白天权重直接训练 SGPD，可调用 `scripts/train.py` 并省略 `--init-adapter-dir`，以检验阶段训练的作用。
3. `configs/night_full_real.sh` 可选地使用 SVOX train 的真实夜图训练完整 Night 主干。白天 SALAD Gallery 固定，25 米内所有白天图是正样本，25–250 米从分母排除，250 米以外提供负样本。另用冻结 Day 模型对**同一夜图**的描述符作余弦约束，缓解小样本训练漂移；它与合成图对的 Day→Night Descriptor Alignment 是不同的监督来源。真实夜图没有可信逐 Patch 对应，所以此阶段不启用 SGPD。默认按北向坐标保留内部空间验证集，以 Recall@1/5/10 选择最佳 epoch；不指定 `INIT_ADAPTER_DIR` 时从原始白天 DINOv2 初始化。

两个合成阶段均默认全参数 Night DINOv2、batch size 8、BF16、AdamW、学习率 1e-6，各 1 epoch。真实阶段默认 batch 4、FP32、3 epoch，末层学习率 1e-6、逐层衰减 0.7，最早层学习率约 1.38e-8；所有参数均可训练。同夜图 Teacher 约束默认权重 100，可用 `NIGHT_BASELINE_WEIGHT=0` 独立关闭。可通过 `GRADIENT_ACCUMULATION_STEPS` 在显存较小时维持有效 batch 大小。当前项目提供匹配这份白天 checkpoint 与 224×224 预处理的 Stage2B 对应校准，仅启用局部损失时需要它。

可选 `--night-augmentation camera` 在缩放之后、归一化之前添加曝光/Gamma、光晕过曝、模糊、噪声、JPEG、降饱和、暗角和随机光度操作。Dataset 保留 clean Teacher/对应输入与单独的 Student 增强输入；真实阶段的描述符 anchor 也以原始夜图为 Teacher 目标。策略不改变几何，原图验证/test 不调用增强。固定种子、逐项消融和相机计算顺序见 [增强设计](NIGHT_CAMERA_AUGMENTATION.md)。

## 损失与消融

| 损失 | 实现 | 开关 |
|---|---|---|
| SGPD Patch | 对经过 Stage2B 校准的 A 类可信匹配，用 `confidence × teacher.assignment_keep` 加权 `1 − cosine(student_patch, day_patch)` | `--patch-weight`，默认第二阶段 0.1；`--importance-mode saliency/uniform` |
| Keep Alignment | 在相同可信匹配上对齐 Student/Teacher 的 `assignment_keep` | `--keep-weight`，默认 0 |
| 条件 OT | 在可信 Patch 的有效 SALAD 簇上，对齐 Teacher/Student 分配分布 | `--ot-weight`，默认 0 |
| Patch Magnitude | 对齐可信 Patch 的特征范数 | `--magnitude-weight`，默认 0 |
| Descriptor Alignment | 归一化夜间描述符与同图白天 Teacher 描述符的半平方距离 | `--global-weight`，默认 1 |
| 地点判别 | 夜间查询与当前 batch 加历史队列的白天描述符做多正样本 InfoNCE；正样本由 `(city, place_id)` 定义 | `--contrast-weight`，默认 1；`--contrast-queue-size` 默认 256 |

权重设为 0 可独立消融。`uniform` 使用同一批可信匹配但不乘 `assignment_keep`，可检验显著性权重是否有效。`--contrast-queue-size 0` 退化为 batch 内 InfoNCE；不同地点才计为有效负样本。Keep、条件 OT 与 Patch Magnitude 默认关闭，只有显式赋予正权重才启用。当前地点判别是跨域 InfoNCE，并非原论文的弱监督 Triplet；若做与原文严格对照，需另实现其 ICKD/Triplet 训练。

`assignment_keep` 是 SALAD Sinkhorn 分配到有效簇的质量，不是建筑物语义标签。它仅提供候选 Patch 重要度；真正参与局部监督的 Patch 还必须通过同一教师权重、同一预处理下标定的可信对应筛选。不能宣称它自动识别稳定建筑。

## 对应关系与防泄漏

训练图对由 Qwen `sample_id` 和真实源图 basename 显式匹配，并核验 SHA-256。Stage2B 的 A 类对应经过双向局部特征匹配、相对结构和校准阈值检查。Student 的监督目标始终来自**真实白天图**；对生成夜图做的额外冻结 Teacher 前向仅用于判断哪些局部对应可信。未知或不可靠的 Patch 不计算局部损失。

项目内的 `configs/dino_salad_stage2b_calibration.json` 绑定当前完整 checkpoint 和 224×224 预处理。其历史标定地点与当前 night 原始训练清单有 4 个地点重叠；默认 `data/splits/night_train.jsonl` 已排除历史保留的验证/校准地点，并与 `night_val.jsonl` 按地点隔离。直接改用未经划分的 `data/train/manifest.jsonl` 时，必须通过 `--exclude-places` 排除这些地点。SVOX 官方 train/test 划分存在空间近邻；真实夜景阶段使用 test 文件名坐标建立 250 米隔离带，既不解码 test 图像，也不使用 test 描述符或 Recall 优化参数。由于过滤使用了 test 几何信息，最终结果应标记为探索性官方划分评估。

## 评估

SVOX 使用完整 test/day Gallery 和真实 `queries_night`，从文件名读取 UTM 坐标，25 m 半径内任一 Gallery 图为正样本；计算精确检索 R@1/5/10。全参数 Night、LoRA 与旧 Output Adapter 共用**原始白天** Gallery 描述符和同一缓存。`data/svox/README.md` 提及的 10 m 是构建数据集时的查询筛选阈值；本项目评估采用随附 SALAD workflow 的 25 m 检索协议。

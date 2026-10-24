# Night DINOv2-SGPD 设计

## 模型与梯度

`dino_salad.ckpt` 提供白天训练好的 DINOv2-B/14 与 SALAD。Teacher 保留原始冻结白天主干；Student 初始化时复制这份主干，随后默认只微调最后四层 blocks 8–11 的原始权重，共 28,357,632 个参数。Patch Embedding、位置/CLS Token、前八层与最终 Norm 冻结。两支 DINOv2 独立，SALAD 为共用冻结聚合器；Teacher 使用 `no_grad`，冻结的最终 Norm 和 SALAD 允许梯度传回 Student 可训练层。

`--adapter-type full` 表示独立完整天气主干，不意味着必须训练所有参数。`--full-train-scope late --full-train-blocks 4` 是默认，`all` 显式启用所有 86,580,480 个 Night 参数；`early|late` 配合 block 数可做范围消融。各范围仍保存完整 Night state dict。默认最终 Norm 冻结，与用户 `night_late4_01` 一致；SALAD 原始训练设置和当前实现区别见 [训练说明](SALAD_TRAINING.md)。

训练和评估白天 Gallery 时始终使用原始白天主干，因此 Night DINOv2 对齐**原始白天 SALAD 描述符空间**，不需要重训或重建 Gallery。Night 权重单独保存到 `adapter.safetensors`，只包含 Night DINOv2，不包含 SALAD；文件约 346 MB。`adapter_type=lora` 和 `adapter_type=output` 保留先前消融，旧 `runs/night_sgpd_01` 属于 Output。框架预留 rain、fog、snow、sun 接口，目前仅训练 night。

## 默认与历史分阶段训练

当前入口 `configs/night_specialist.sh` 默认后四层微调，Qwen 主分支使用 Global / Contrast / Relation，另可使用训练 Day 原图生成的强制低照度 Source Night 视图。固定 Day bank 的同一 `(city, place_id)` 条目均视为正样本，避免伪负例。真实夜图仅用于只读评估，具体损失见 [Night 专用模型](NIGHT_SPECIALIST.md)。

以下两个 full 脚本显式保留历史**全部参数微调**消融，不随默认范围调整而改变：

1. `configs/night_full_global.sh` 默认用地点隔离的 night 训练集，仅优化全局 Descriptor Alignment 与跨域 InfoNCE。batch 尽量覆盖不同的 `(city, place_id)`；容量 256 的跨 batch 队列保存 detached 的白天 Teacher 描述符。当前 batch 和队列内的同地点描述符全是正样本，不会充当伪负例。训练日志记录每个 query 的有效负样本数。
2. `configs/night_full_sgpd.sh` 从第一阶段完整 Night 主干初始化，继续训练全局损失与 SGPD。加载时检查天气、原始 checkpoint、预处理和模型类型。若要从原始白天权重直接训练 SGPD，可调用 `scripts/train.py` 并省略 `--init-adapter-dir`，以检验阶段训练的作用。

两个历史配对阶段均显式默认全部参数 Night DINOv2、batch size 8、BF16、AdamW、学习率 1e-6，各 1 epoch。可通过 `GRADIENT_ACCUMULATION_STEPS` 在显存较小时维持有效 batch 大小。当前项目提供匹配这份白天 checkpoint 与 224×224 预处理的 Stage2B 对应校准，仅启用局部损失时需要它。

可选 `--night-augmentation camera` 在缩放之后、归一化之前添加曝光/Gamma、光晕过曝、模糊、噪声、JPEG、降饱和、暗角和随机光度操作。Dataset 保留 clean Teacher/对应输入与单独的 Student 增强输入。策略不改变几何，原图验证/test 不调用增强。固定种子、逐项消融和相机计算顺序见 [增强设计](NIGHT_CAMERA_AUGMENTATION.md)。

## 损失与消融

| 损失 | 实现 | 开关 |
|---|---|---|
| SGPD Patch | 对经过 Stage2B 校准的 A 类可信匹配，用 `confidence × teacher.assignment_keep` 加权 `1 − cosine(student_patch, day_patch)` | `--patch-weight`，默认第二阶段 0.1；`--importance-mode saliency/uniform` |
| Keep Alignment | 在相同可信匹配上对齐 Student/Teacher 的 `assignment_keep` | `--keep-weight`，默认 0 |
| 条件 OT | 在可信 Patch 的有效 SALAD 簇上，对齐 Teacher/Student 分配分布 | `--ot-weight`，默认 0 |
| Patch Magnitude | 对齐可信 Patch 的特征范数 | `--magnitude-weight`，默认 0 |
| DINO Attention | 最后层 CLS→Patch 注意力，在同一 A 类对应上条件归一化后计算 FP32 Teacher→Student KL；不改变注意力原前向 | `--attention-weight`，默认 0，仅独立 Night 模式 |
| Descriptor Alignment | 归一化夜间描述符与对应白天 Teacher 描述符的 `1 − cosine` | `--global-weight`；专用配置默认 0.1，历史 full 阶段默认 1 |
| 地点判别 | 夜间查询与冻结 Day bank 或当前 batch 加历史队列的白天描述符做多正样本 InfoNCE；正样本由 `(city, place_id)` 定义 | `--contrast-weight`；专用配置默认 0.1，历史 full 阶段默认 1；队列默认 256 |
| 检索关系 | 对齐 Teacher 与 Student 对其它地点 Day 描述符的相对相似度分布 | `--relation-weight`；专用配置默认 0.1 |
| Source Night | 模拟低照度训练 Day 视图的全局对齐与多正样本 InfoNCE | `--source-night-weight`；专用配置默认 0.5 |

权重设为 0 可独立消融。`uniform` 使用同一批可信匹配但不乘 `assignment_keep`，可检验显著性权重是否有效。`--contrast-queue-size 0` 退化为 batch 内 InfoNCE；不同地点才计为有效负样本。Keep、条件 OT 与 Patch Magnitude 默认关闭，只有显式赋予正权重才启用。当前地点判别是跨域 InfoNCE，并非原论文的弱监督 Triplet；若做与原文严格对照，需另实现其 ICKD/Triplet 训练。

`assignment_keep` 是 SALAD Sinkhorn 分配到有效簇的质量，不是建筑物语义标签。它仅提供候选 Patch 重要度；真正参与局部监督的 Patch 还必须通过同一教师权重、同一预处理下标定的可信对应筛选。不能宣称它自动识别稳定建筑。

`configs/night_patch.sh` 把所有训练 Qwen 图送入相机退化；`--patch-visibility photometric` 根据实际 Student 输入的非黑场、非全通道裁剪像素比例，对既有对应降权。注意力提取只重算最后层 QKV 的 CLS 行，兼容非重入 checkpoint，不构造额外的完整注意力矩阵。它与 SGPD、Keep、OT 分别控制，完整定义和对照见 [Patch 与 DINO 注意力](NIGHT_PATCH_ATTENTION.md)。

## 对应关系与防泄漏

训练图对由 Qwen `sample_id` 和真实源图 basename 显式匹配，并核验 SHA-256。Stage2B 的 A 类对应经过双向局部特征匹配、相对结构和校准阈值检查。Student 的监督目标始终来自**真实白天图**；对生成夜图做的额外冻结 Teacher 前向仅用于判断哪些局部对应可信。未知或不可靠的 Patch 不计算局部损失。

项目内的 `configs/dino_salad_stage2b_calibration.json` 绑定当前完整 checkpoint 和 224×224 预处理。其历史标定地点与当前 night 原始训练清单有 4 个地点重叠；默认 `data/splits/night_train.jsonl` 已排除历史保留的验证/校准地点，并与 `night_val.jsonl` 按地点隔离。直接改用未经划分的 `data/train/manifest.jsonl` 时，必须通过 `--exclude-places` 排除这些地点。SVOX 图像只进入检索评估；训练输入仅来自上述 Qwen 配对清单及其中 Day 原图产生的低照度视图。

## 评估

SVOX 使用完整 test/day Gallery 和真实 `queries_night`，从文件名读取 UTM 坐标，25 m 半径内任一 Gallery 图为正样本；计算精确检索 R@1/5/10。全参数 Night、LoRA 与旧 Output Adapter 共用**原始白天** Gallery 描述符和同一缓存。`data/svox/README.md` 提及的 10 m 是构建数据集时的查询筛选阈值；本项目评估采用随附 SALAD workflow 的 25 m 检索协议。

32 GB 主机下训练和评估逐项串行执行，使用 `scripts/run_bounded.py` 限制实验进程及其子进程内存，默认数据加载 worker 为 0。运行命令与限制见 [SALAD 训练设置](SALAD_TRAINING.md)。

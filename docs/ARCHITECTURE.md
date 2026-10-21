# Night LoRA-SGPD 设计

## 模型与梯度

`dino_salad.ckpt` 提供已训练的 DINOv2-B/14 和 SALAD。Teacher 与 Student 使用**同一个**主干实例；DINOv2 原权重与 SALAD 始终冻结。默认夜间 Adapter 是 LoRA：在 DINOv2 最后 4 个 Transformer block 的 `attn.qkv` 和 `attn.proj` 上加入 rank 8、alpha 16 的低秩增量。Teacher 读取真实白天图时关闭 LoRA；Student 读取 Qwen 夜图时启用 night LoRA，并经过同一冻结 SALAD。优化器只接收 night LoRA 参数。

训练和评估白天 Gallery 时也关闭天气 LoRA，因此所有夜间 Adapter 均对齐**原始白天 SALAD 描述符空间**，不需要重训或重建 Gallery。LoRA 初始化时增量为零，Adapter 单独保存，完整 checkpoint 只存一份。`ADAPTER_TYPE=output` 保留旧版 DINOv2 输出端的 Patch/CLS 残差 Adapter 作为消融；`runs/night_sgpd_01` 就是旧 Output 实验结果，不是 LoRA 结果。框架预留了 rain、fog、snow、sun 接口，但目前只准备和训练 night。

## 分阶段训练

1. `configs/night_global.sh` 默认用地点隔离的 night 训练集，仅优化全局 Descriptor Alignment 与跨域 InfoNCE。batch 尽量覆盖不同的 `(city, place_id)`；容量 256 的跨 batch 队列保存 detached 的白天 Teacher 描述符。当前 batch 和队列内的同地点描述符全是正样本，不会充当伪负例。训练日志记录每个 query 的有效负样本数。
2. `configs/night_sgpd.sh` 用相同 LoRA 架构继续训练全局损失与 SGPD；设置 `INIT_ADAPTER_DIR` 指向第一阶段输出。加载时检查天气、完整 checkpoint、预处理和 Adapter 参数配置。也可不传 `INIT_ADAPTER_DIR`，从零增量直接训练 SGPD，以检验阶段训练的作用。

两个阶段均默认 LoRA rank 8、alpha 16、最后 4 个 block 的 `qkv/proj`，batch size 4、3 epoch、BF16。当前项目提供匹配这份完整 checkpoint 与 224×224 预处理的 Stage2B 对应校准。仅第二阶段启用局部损失时需要它。

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

项目内的 `configs/dino_salad_stage2b_calibration.json` 绑定当前完整 checkpoint 和 224×224 预处理。其历史标定地点与当前 night 原始训练清单有 4 个地点重叠；默认 `data/splits/night_train.jsonl` 已排除历史保留的验证/校准地点，并与 `night_val.jsonl` 按地点隔离。直接改用未经划分的 `data/train/manifest.jsonl` 时，必须通过 `--exclude-places` 排除这些地点。SVOX test 与 GSV 训练图来源不同；真实 SVOX night 测试集只用于最终检索评估。

## 评估

SVOX 使用完整 test/day Gallery 和真实 `queries_night`，从文件名读取 UTM 坐标，25 m 半径内任一 Gallery 图为正样本；计算精确检索 R@1/5/10。baseline、LoRA 与旧 Output Adapter 共用**原始白天** Gallery 描述符和同一缓存。`data/svox/README.md` 提及的 10 m 是构建数据集时的查询筛选阈值；本项目评估采用随附 SALAD workflow 的 25 m 检索协议。现有 `runs/night_sgpd_01` 的真实夜间结果只说明旧 Output 配置的表现，不能作为 LoRA 结果。

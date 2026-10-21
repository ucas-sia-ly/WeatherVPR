# Night SGPD 设计

## 模型与梯度

`dino_salad.ckpt` 提供已训练的 DINOv2-B/14 和 SALAD。Teacher 与 Student 使用**同一个**冻结实例。Teacher 将真实白天图直接输入主干；Student 将 Qwen 夜图输入同一 DINOv2，在输出的 16×16 Patch 特征和 CLS token 上应用残差 Adapter，再经过同一冻结 SALAD。只有 Adapter 的参数加入优化器。推理时白天 Gallery 仍用原始 SALAD 提取，因此白天图库可在各夜间训练实验之间复用。

当前 Adapter 位于 DINOv2 输出端，不是 Transformer 内部 LoRA。它只保存约 0.8 MB 的参数；相对于 336 MB 的完整 checkpoint，适合独立保存天气模块。初始化为恒等映射。

## 损失与消融

| 损失 | 实现 | 开关 |
|---|---|---|
| SGPD Patch | 对经过 Stage2B 校准的 A 类可信匹配，用 `confidence × teacher.assignment_keep` 加权 `1 − cosine(student_patch, day_patch)` | `--patch-weight`；`--importance-mode saliency/uniform` |
| Keep Alignment | 在相同可信匹配上对齐 Student/Teacher 的 `assignment_keep` | `--keep-weight` |
| Descriptor Alignment | 归一化夜间描述符与同图白天 Teacher 描述符的半平方距离 | `--global-weight` |
| 地点判别 | 夜间查询与 batch 内白天描述符的多正样本 InfoNCE；正样本由 `(city, place_id)` 定义 | `--contrast-weight` |

权重设为 0 可独立消融。`uniform` 使用同一批可信匹配但不乘 `assignment_keep`，可检验显著性权重是否有效。当前地点判别是跨域 InfoNCE，并非原论文的弱监督 Triplet；若做与原文严格对照，需另实现其 ICKD/Triplet 训练。

`assignment_keep` 是 SALAD Sinkhorn 分配到有效簇的质量，不是建筑物语义标签。它仅提供候选 Patch 重要度；真正参与局部监督的 Patch 还必须通过同一教师权重、同一预处理下标定的可信对应筛选。不能宣称它自动识别稳定建筑。

## 对应关系与防泄漏

训练图对由 Qwen `sample_id` 和真实源图 basename 显式匹配，并核验 SHA-256。Stage2B 的 A 类对应经过双向局部特征匹配、相对结构和校准阈值检查。Student 的监督目标始终来自**真实白天图**；对生成夜图做的额外冻结 Teacher 前向仅用于判断哪些局部对应可信。未知或不可靠的 Patch 不计算局部损失。

项目内的 `configs/dino_salad_stage2b_calibration.json` 绑定当前完整 checkpoint 和 224×224 预处理。其历史标定地点与当前 night 训练集有 4 个地点重叠，必须在训练时传 `configs/dino_salad_calibration_excluded_places.json`，或使用已经排除这些地点的训练 split。SVOX test 与 GSV 训练图来源不同；真实 SVOX night 测试集只用于最终检索评估。

## 评估

SVOX 使用完整 test/day Gallery 和真实 `queries_night`，从文件名读取 UTM 坐标，25 m 半径内任一 Gallery 图为正样本；计算精确检索 R@1/5/10。baseline 与 Adapter 共用**原始白天** Gallery 描述符。`data/svox/README.md` 提及的 10 m 是构建数据集时的查询筛选阈值；本项目评估采用随附 SALAD workflow 的 25 m 检索协议。

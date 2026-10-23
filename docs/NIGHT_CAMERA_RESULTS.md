# Night 相机退化增强重跑（2026-10-23）

## 测试结果

新增相机增强后，已完成 5 组主训练，以及原图合成验证、SVOX train 空间验证和两条预先登记的最终 test 对比。增强参数在训练前固定为 `configs/night_camera.json`，未依据这轮 test 调整强度、损失或 epoch。

| 模型 / 固定原始白天 Gallery | SVOX test R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 原始 Day SALAD | 763/823，92.71% | 810/823，98.42% | 816/823，99.15% |
| 上一轮无增强 Day→真实 Night 微调 | 765/823，92.95% | 810/823，98.42% | 816/823，99.15% |
| 相机增强 Day→真实 Night 微调 | 765/823，92.95% | 809/823，98.30% | 817/823，99.27% |
| 相机增强 Global→SGPD→真实 Night 微调 | 766/823，93.07% | 810/823，98.42% | 816/823，99.15% |

直接 Day 初始化的增强模型没有提高上一轮的 test R@1；R@5 少命中 1 张、R@10 多命中 1 张。完整 SGPD 路线的 test R@1 比上一轮多命中 **1 张**，比原始 Day 模型多 3 张。它同时修复了原始 Day 的 15 个 R@1 错误并引入 12 个新错误；相对上一轮 Night 则修复 13 个、引入 12 个。不能将这种单 seed 的净变化称为稳定显著提升。

开发阶段按固定 76 张空间验证集的 R@1、R@5、R@10 依次选择，`real_day` 为主模型；`real_sgpd` 是预先登记的第二条最终训练路线。两者均在 test 前冻结权重，第 1 epoch 被选中，后续未用 test 重新选择 epoch。测试上 SGPD 路线稍高这一点没有被用来回头调参或改写开发集选择。

## 增强内容与接线

新增自动曝光增益与 Gamma、光源周围光晕及裁剪过曝、运动模糊、泊松高斯噪声、JPEG、降饱和、暗角，以及随机亮度/对比度策略。整体触发概率 0.8、强度 1；每项还有独立采样概率和范围，均可通过 JSON 消融。具体值、计算顺序与复现方式见 [相机增强说明](NIGHT_CAMERA_AUGMENTATION.md)。

增强在原 SALAD 的 224×224 RGB 双线性缩放之后、归一化之前执行。Teacher、Gallery、验证与 test 均使用原始干净预处理。合成阶段先在 clean Day/Qwen 图对上验证 Stage2B A 类对应，再只将退化图送入 Student。真实阶段使用 Student(增强夜图) 对齐 Teacher(同一原始夜图)，避免强约束目标也被随机退化。

策略没有引入独立裁剪、旋转或透视变换。所有随机性由训练 seed、epoch、样本索引导出，与 DataLoader worker 数无关。范围是待验证的域随机化假设，并非相机标定结果；这里也没有调用包含几何变换的完整 torchvision RandAugment。

增强预览与采样参数在 `runs/night_camera_20261023/augmentation_preview.jpg`、`augmentation_preview.json`。

## 同预算训练和内部验证

合成训练仍为 7,263 对训练图、385 对地点隔离合成验证，batch 8、BF16、AdamW、LR 1e-6、seed 42。Global/Contrast 权重各 1，队列 256；SGPD 权重 0.1，Keep/OT/Magnitude 关闭。每个合成阶段 1 epoch、908 更新。

真实阶段仍为 255 张优化夜图、76 张空间验证、47 张间隔，20,486 张固定 Day Gallery；FP32、batch 4、3 epoch / 192 更新、LR 1e-6、逐层衰减 0.7、同原始夜图 Teacher anchor 100。两条真实路线采用相同采样视图、同 batch 顺序和相同预算，均按内部验证保存最佳 epoch。

| 配置 | 内部验证 R@1 / R@5 / R@10（76 张） | 原图合成验证配对余弦 |
| --- | --- | ---: |
| 原始 Day SALAD | 59 / 70 / 70 | 0.6778 |
| 上一轮无增强 Global，1 epoch | 49 / 64 / 70 | 0.7413 |
| 上一轮无增强 Global→SGPD | 49 / 65 / 68 | 0.7474 |
| 增强 Global，`global_01` | 54 / 66 / 71 | 0.7295 |
| 增强 Global→Global，`global_02` | 50 / 62 / 69 | 0.7358 |
| 增强 Global→SGPD，`sgpd_02` | 50 / 62 / 69 | 0.7359 |
| 上一轮无增强 Day→真实微调 | 59 / 70 / 71 | — |
| 增强 Day→真实微调，`real_day` | 61 / 70 / 71 | — |
| 增强 Global→SGPD→真实微调，`real_sgpd` | 61 / 69 / 71 | — |

所有合成训练模型的原图合成验证 R@1 均为 385/385，该验证已接近饱和。相机增强把合成 Global 的真实内部验证 R@1 从 49/76 提到 54/76，但再训练一轮后降至 50/76；合成余弦变好也不能替代真实域验证。

新增 `global_02` 与 `sgpd_02` 从**相同的 `global_01` 权重**出发，使用相同的下一轮增强采样和相同 908 更新，区别是 Patch 权重 0 与 0.1。两者内部 Recall 完全相同，目前没有 SGPD 的额外检索收益证据。完整 SGPD→真实阶段的 test 多命中 1 张也不能单独归因于 SGPD：本轮没有再增加 Global 接续→真实阶段的对应最终对照。

## 控制检查与限制

- 两个合成接续的 908 步增强采样计数逐步相同；两条真实阶段的 192 步也逐步相同。
- SGPD 仍见到 196,535 个 clean A 类可信 Patch，与上一轮相同，说明随机相机退化没有污染匹配验证路径。
- 真实优化/验证/间隔的 Query ID 与上一轮逐项一致；基础 checkpoint、预处理和 Day Gallery 缓存经过严格校验。
- `selection_before_test.json` 保存了 test 前的全部候选权重 SHA-256、内部指标和两条最终对比路线；test 后已核对权重没有改变。
- 这是单 seed、76 张开发集的探索性实验；先前已查看过 test，训练过滤也使用 test 文件名坐标。没有使用 test 图像内容优化参数，但不能称为全新盲测。

可参考的 `WORKSPACE/salad/dataloaders/GSVCitiesDataloader.py` 中确有 `Resize→RandAugment(num_ops=3)`，但没有当前 checkpoint 的完整训练记录或该增强的独立消融，无法证明它是原始 Day 模型夜景泛化能力的主要原因；DINOv2 预训练与 SALAD 地点训练也未被单独消融。本轮只检验相机退化增强在既定框架和预算下的效果。

## 权重与复现

所有产物在 `runs/night_camera_20261023/`：

| 路线 | 权重 |
| --- | --- |
| 开发集所选：Day→增强真实 Night | `real_day/adapter.safetensors` |
| 完整增强蒸馏路线 | `real_sgpd/adapter.safetensors` |
| 增强合成 Global / Global 接续 / SGPD 接续 | `global_01/`、`global_02/`、`sgpd_02/` |

每个目录包含实际配置、模型来源、增强范围和计数、训练日志及评估 JSON。统一结果为 `results_summary.json`；匹配采样与可信 Patch 检查在 `control_checks.json`。

```bash
# 完整重跑五组主训练与干净开发集评估；必须使用新的实验名
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_02 bash configs/night_camera_replay.sh

# 仅重新评估已完成的完整路线；不会启用测试增强
/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python scripts/evaluate_svox.py \
  --adapter-dir runs/night_camera_20261023/real_sgpd --device cuda \
  --batch-size 8 --reference-cache-mib 768 --save-hard-cases \
  --output runs/night_camera_20261023/real_sgpd/svox_test_recheck.json
```

原有训练入口的增强默认仍为 `off`，保留干净基线；`night_camera_replay.sh` 显式启用固定策略。当前结果不足以支持将相机增强或 SGPD 宣称为稳定的普遍收益来源。

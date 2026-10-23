# Night 相机退化增强配对训练（2026-10-23）

## 实验设置

相机增强使用训练前固定的 `configs/night_camera.json`。保留三组 Qwen 配对训练：Global、从同一 Global 起点继续的 Global 对照、以及 SGPD 接续。数据为 7,263 对训练图和 385 对地点隔离的合成验证图；batch 8、BF16、AdamW、LR 1e-6、seed 42，每阶段 1 epoch / 908 次更新。

Global/Contrast 权重各 1，白天描述符队列容量 256；SGPD 权重 0.1，Keep/OT/Magnitude 关闭。`global_02` 与 `sgpd_02` 从相同的 `global_01` 权重开始，使用相同的下一轮增强采样和同等更新预算，分别设置 Patch 权重 0 与 0.1。

## 原图合成验证

| 配置 | R@1 | 平均配对描述符余弦 |
| --- | ---: | ---: |
| 原始 Day SALAD | 384/385 | 0.6778 |
| 无增强 Global，1 epoch | 385/385 | 0.7413 |
| 无增强 Global→SGPD | 385/385 | 0.7474 |
| 相机增强 Global，`global_01` | 385/385 | 0.7295 |
| 相机增强 Global→Global，`global_02` | 385/385 | 0.7358 |
| 相机增强 Global→SGPD，`sgpd_02` | 385/385 | 0.7359 |

合成验证 Recall 已接近饱和。相机增强训练得到的配对余弦低于对应无增强路线；同预算 Global 与 SGPD 接续余弦接近。以上指标只衡量合成域，当前不足以证明相机增强或 SGPD 能稳定提高真实夜景检索。

原始 Day SALAD 在 SVOX test（823 张夜间 Query、17,166 张原始白天 Gallery）的 R@1/5/10 为 **92.71% / 98.42% / 99.15%**。真实夜景检索协议与独立评估命令见 [数据和评估](DATA_AND_EVALUATION.md)。

## 增强内容与控制检查

新增自动曝光增益与 Gamma、光源周围光晕及裁剪过曝、运动模糊、泊松高斯噪声、JPEG、降饱和、暗角，以及随机亮度/对比度策略。整体触发概率 0.8、强度 1，各操作有独立概率和范围，见 [相机增强说明](NIGHT_CAMERA_AUGMENTATION.md)。

增强在原 SALAD 的 224×224 RGB 双线性缩放之后、归一化之前执行。Day Teacher、Gallery 和评估均使用原图。Stage2B 在干净 Day/Qwen 图对上验证 A 类对应，退化后的夜图仅送入 Student。策略没有独立裁剪、旋转或透视变换，随机性由 seed、epoch、样本索引导出。

- `global_02` 与 `sgpd_02` 的 908 步增强采样计数逐步相同。
- SGPD 见到 196,535 个 clean A 类可信 Patch，与无增强路线相同，增强没有污染对应验证路径。
- 配置、训练日志和权重元数据记录实际增强参数、采样 offset 和累计计数。

可参考的 `WORKSPACE/salad/dataloaders/GSVCitiesDataloader.py` 中存在 `Resize→RandAugment(num_ops=3)`，但缺少当前 checkpoint 的完整训练记录和该增强的独立消融，无法确认它是原始 Day 模型夜景泛化能力的主要原因；DINOv2 预训练与 SALAD 地点训练也未被单独消融。相机退化范围是可验证的域随机化假设，尚未针对具体相机标定。

## 产物与复现

结果位于 `runs/night_camera_20261023/`：

| 配对训练路线 | 权重目录 |
| --- | --- |
| 相机增强 Global | `global_01/` |
| 相机增强 Global→Global | `global_02/` |
| 相机增强 Global→SGPD | `sgpd_02/` |

每个目录保留权重、实际配置、模型来源、增强计数、训练日志及原图合成验证 `synthetic_val.json`。对应采样与可信 Patch 检查为 `control_checks.json`；增强预览为 `augmentation_preview.jpg` 与 `augmentation_preview.json`。

```bash
# 重跑三组配对训练与原图合成验证；使用新的实验名
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_02 bash configs/night_camera_replay.sh

# 独立评估真实夜景，不启用训练增强
/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python scripts/evaluate_svox.py \
  --adapter-dir runs/night_camera_20261023/sgpd_02 --device cuda \
  --batch-size 8 --reference-cache-mib 768 --save-hard-cases \
  --output runs/night_camera_20261023/sgpd_02/svox_test.json
```

原有训练入口的增强默认 `off`；`night_camera_replay.sh` 显式启用固定策略。test 不能用于挑选 epoch 或调整增强参数。当前是单 seed 配对训练，仍需通过相同预算的增强及损失消融、独立真实域检索评估来判断收益。

# Night 训练相机退化增强

此增强用于缩小高质量 Qwen 夜图与真实夜间相机图像之间的外观差异。实现位于 [night_augmentation.py](../src/weather_vpr/data/night_augmentation.py)，固定策略在 [night_camera.json](../configs/night_camera.json)。这些范围是可消融的增强假设，尚未依据具体相机标定，不能视为真实曝光、噪声或传感器参数。

增强只作用于训练中的 Student 夜图。默认配置仍为 `off`，显式指定 `camera` 后启用；验证、测试和 Gallery 特征提取均使用原有干净预处理。

## 数据流与监督目标

Student 输入先按原 SALAD 方式执行 RGB 转换和双线性缩放，默认网络分辨率为 224×224，再执行相机增强，最后使用原 ImageNet 归一化。因此模糊长度、光晕半径和 JPEG 作用于网络输入分辨率。

Qwen 配对训练保留三份用途明确的输入：

- 冻结 Day Teacher 读取干净白天图像，生成全局和局部监督目标。
- Stage 2B 使用干净白天图像和干净 Qwen 夜图验证可信对应；增强结果不参与对应关系计算。
- Night Student 读取增强后的 Qwen 夜图，接受全局、地点判别和可选 SGPD 等损失。

增强不执行裁剪、旋转、透视、翻转或其他几何变换，Patch 网格位置和原可信对应仍然一致。模糊与噪声可能降低局部特征可辨识度，因此增强强度和 SGPD 权重需要独立验证。

真实夜图阶段同样只增强训练 Night Student 输入。`--night-baseline-weight` 启用时，冻结原 Day 模型读取**同一张干净真实夜图**生成 Descriptor anchor；Student 使用增强视图。该约束帮助模型在学习地点判别时保留原描述符空间。可选 `--anchor-weight` 的同图白天 anchor 也保留干净输入。真实夜图不因加入相机增强而获得像素级 Day–Night 对应，不在该阶段凭相同坐标增加 SGPD。

## 默认策略

整个 `camera` 策略以 `probability=0.8` 触发。触发后，各操作按下表独立采样；表中概率是**条件于策略已触发**的概率。参数范围对应 `strength=1`，并非每张图都执行全部操作。

| 操作键 | 条件概率 | 参数范围与行为 |
|---|---:|---|
| `exposure` | 0.90 | 线性亮度目标 0.06–0.20；自动增益限制为 0.6–2.0；Gamma 0.75–1.30 |
| `vignette` | 0.40 | 边缘衰减强度 0.15–0.55；中心在归一化坐标内各偏移 ±0.08 |
| `motion_blur` | 0.35 | 3–9 像素奇数长度线核；方向在 0–180° 随机采样；边界复制、不环绕 |
| `bloom` | 0.40 | 线性高光阈值 0.50–0.85；光晕强度 0.15–0.55；半径为图像短边的 0.004–0.018 |
| `noise` | 0.80 | 泊松等效计数 500–2400；高斯读出噪声标准差 0.0005–0.0035，均作用于线性信号 |
| `desaturation` | 0.45 | PIL 饱和度系数 0.35–0.85 |
| `photometric` | 0.60 | 随机选择 1 或 2 项：亮度系数 0.80–1.20、对比度系数 0.75–1.25 |
| `jpeg` | 0.65 | JPEG quality 40–90，4:2:0 色度采样，随后解码回 RGB |

`photometric` 是随机亮度/对比度策略；项目没有在此处调用完整 torchvision RandAugment。此处保留了其随机光度变化的思路，并单独加入相机退化，便于分别消融。

默认噪声范围经过 Qwen 夜图的 224×224 接触表检查：保留地标轮廓，覆盖暗部噪点、不同亮度、部分模糊和降饱和。检查时降低了过强的线性读出噪声，避免多数暗部被彩色噪点淹没。视觉检查只能确认增强形态和强度，是否提升真实夜景检索由干净验证数据决定。

## 相机计算顺序

实现的顺序固定如下，各步骤可以不被采样到：

1. 将 RGB sRGB 数值转换为线性信号。
2. 根据当前平均线性亮度和采样目标计算自动曝光增益，并限制在配置范围；同时采样 Gamma，留待后面的色调映射使用。
3. 在线性信号上施加暗角和运动模糊。
4. 从增益后的高亮信号生成高光掩膜，经高斯扩散后加回信号，形成光晕。高光及其叠加可以在随后裁剪中产生过曝。
5. 在线性信号上采样泊松 Shot Noise，再加入高斯 Read Noise；不直接在 sRGB 像素上套用泊松分布。
6. 将信号裁剪至 [0, 1]，执行 `signal ** gamma`，再转回 sRGB。Gamma 小于 1 对应提亮。
7. 依次执行可选降饱和、亮度/对比度策略和 JPEG 编码解码。

所有输出均为同尺寸 RGB、有限的 8 位像素；原输入 PIL 图像不会被修改。每次采样记录操作名称和实际参数，以及色调映射前达到裁剪阈值的像素比例。

`strength` 接受 [0, 2]。增益按 `gain ** strength` 插值；Gamma、饱和度、亮度和对比度向恒等值 1 插值；光晕、暗角和读出噪声等随强度缩放；泊松等效计数按 `1 / strength²` 调整。`strength <= 1e-6` 明确返回恒等结果，以避免极小强度下泊松参数溢出；`off`、概率 0 和强度 0 也保持原像素。

## 开关和消融

两个训练入口 `scripts/train.py` 和 `scripts/train_real_night.py` 共用参数：

```text
--night-augmentation off|camera
--augmentation-strength 1.0
--augmentation-probability 0.8
--augmentation-config configs/night_camera.json
--augmentation-epoch-offset 0
```

CLI 的 `profile`、`strength` 和 `probability` 始终覆盖 JSON 中对应字段，包括其 CLI 默认值。因此只传 JSON 文件不会自动启用策略，需同时指定 `--night-augmentation camera`。范围和 `operation_probabilities` 从 JSON 读取。

现有 full 训练配置脚本支持对应环境变量：

```bash
NIGHT_AUGMENTATION=camera \
AUGMENTATION_CONFIG=configs/night_camera.json \
AUGMENTATION_STRENGTH=1 \
AUGMENTATION_PROBABILITY=0.8 \
OUTPUT=runs/night_camera_global \
bash configs/night_full_global.sh
```

`NIGHT_AUGMENTATION=off` 可恢复干净输入。为保证对照清晰，应保留相同的数据划分、模型初始化、训练步数和随机种子，并分别改变增强或损失开关。

独立操作消融可复制 JSON，仅修改相应概率。例如关闭噪声、保留其他默认概率：

```json
{
  "operation_probabilities": {
    "noise": 0
  }
}
```

JSON 中未声明的操作沿用模块默认概率；若只验证一个操作，需显式把其他操作概率全部设为 0。所有表中范围均有对应的 `*_range` 配置字段，完整字段名以 `configs/night_camera.json` 为准。`off`、概率 0、零强度及每项概率置 0 均有测试覆盖。

## 复现完整训练流程

从项目根目录运行 [night_camera_replay.sh](../configs/night_camera_replay.sh)：

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python \
EXPERIMENT=night_camera_replay_01 \
bash configs/night_camera_replay.sh
```

脚本使用同一套增强配置，依次运行全局训练、同等追加训练的 Global 对照、加入 SGPD 的追加训练，以及 Day 初始化和 SGPD 初始化的真实夜图阶段。它在新建目录已存在时退出，以免覆盖实验。`PYTHON=python` 或其他环境中的 Python 均可；训练数据、冻结基础权重和已有 Stage 2B 校准配置需要按项目说明准备好。

Global 与 SGPD 两个追加阶段从相同 `global_01` 权重开始，均设置 `AUGMENTATION_EPOCH_OFFSET=1`，使用对应的下一轮采样。真实夜图阶段默认从 offset 0 开始。脚本最后仅执行干净 synthetic validation 和 SVOX train 内部开发划分评估；最终 test 评估应在依据开发集完成模型选择后单独运行。

## 随机性与日志

集成层通过 `SHA256(training_seed, zero_based_epoch, original_sample_index)` 的前 64 位生成每样本 seed。实际 epoch 为当前零起始 epoch 加 `--augmentation-epoch-offset`，因此同 seed、同 epoch 和同原始样本索引可重现同一增强，与 DataLoader worker 数量、调度和 batch 顺序无关。阶段续训时设置 offset 可避免重复第一轮增强；offset 不会恢复优化器或改变其他训练随机过程。

模块使用局部 NumPy Generator，不修改全局 NumPy RNG。训练配置和权重元数据保存有效增强参数、seed 策略、offset、分辨率和累积操作计数；训练日志记录 batch 内的增强次数。比较实验时，应同时保留配置、训练日志与模型目录。

推理和所有评估入口均不调用该训练增强；Night Query 和原白天 Gallery 继续使用原 SALAD 预处理，检索空间由冻结 Teacher、共享冻结 SALAD 和训练损失约束。

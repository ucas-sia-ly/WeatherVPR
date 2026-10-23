# Night 数据与评估协议

## 精确配对

`data/train/manifest.jsonl` 的每行包含 `sample_id`、`city`、`place_id`、`condition=night`、`source_path`、`output_path`、两个 SHA-256 及质量筛选状态。路径相对清单所在目录。生成脚本读取历史 Qwen `images/metadata.jsonl`，按 `sample_id` 对应本地生成图，并按元数据中的源图 basename 对应本地白天图；7,719 对全部经哈希核验。生成后运行只依赖本项目文件，不访问历史 WORKSPACE。

```bash
python scripts/build_night_manifest.py \
  --metadata /home/admin123/github/WORKSPACE/outputs/qwen_curriculum/images/metadata.jsonl \
  --verify-hashes
```

## 地点隔离

`python scripts/make_night_split.py` 排除随附历史 workflow 的 140 个 validation/calibration 保留地点，其中 61 个出现在当前 night 配对中，共移除 71 对图。剩余数据按 `(city, place_id)` 分组、固定 seed 42，在每个城市抽约 5% 地点作合成验证：7,263 对训练、385 对验证，地点不重叠。`data/splits/heldout_places.json` 同时包含这些外部保留地点和新验证地点。训练入口默认读 `data/splits/night_train.jsonl`，并传 heldout 列表作再次防泄漏校验。

SGPD 校准与当前完整 `dino_salad.ckpt` 的 SHA-256、DINOv2-B/14/SALAD 配置和 224×224 RGB 双线性归一化严格匹配。其 12 个校准地点均在 heldout 中。`scripts/prepare_calibration.py` 能重验并打包校准与地点排除文件。更换 checkpoint 或预处理后必须重新标定，不能修改 JSON 的哈希来绕过检查。

合成验证可运行 `python scripts/evaluate_pairs.py --adapter-dir runs/night_full_sgpd_01 --device cuda`。它使用验证清单中的原始白天图作为 Gallery，未适配与适配夜图共享这份 Gallery；同 `(city, place_id)` 判为检索正样本，并额外比较每对夜图与其**精确白天源图**的描述符余弦相似度。该指标只监控合成域，不能替代真实夜景 Recall。

## 真实夜间检索

`scripts/evaluate_svox.py` 使用 `data/svox/images/test/gallery` 中全部 17,166 张图作原始白天 SALAD Gallery，`queries_night` 中 823 张真实夜图作为 Query。SVOX 文件名提供 UTM Easting/Northing；以**25 米**内的 Gallery 图作为正样本，精确计算 Recall@1/5/10。823 张夜图都有至少一个正样本。这个协议与本项目随附的 SALAD `workflow/evaluation.py` 一致。SVOX README 的 10 米用于构建数据集时筛选查询，不是这里的检索评估半径。

```bash
python scripts/evaluate_svox.py --check-data
python scripts/evaluate_svox.py --adapter-dir runs/night_full_real_01 \
  --device cuda --output runs/night_full_real_01/svox_test.json
```

评估先提取一次白天 Gallery，再用同一 Gallery 分别计算未适配夜图 baseline 与 Night 模型夜图。缓存绑定完整 checkpoint、预处理和有序图像字节；不同 Night 实验可以安全复用原始 Gallery。勿使用 SVOX test Recall 挑选 epoch 或调整超参数；真实阶段从 SVOX train 内部留出的空间验证集选择 epoch。先前 LoRA 实验已经查看过 SVOX test 分数，因此本项目目前的 test 结果不应描述为首次盲测。

需要在真实夜景上做开发阶段诊断时，可使用 `--split train`：它读取 SVOX 的 22,232 张 train/day Gallery 与 702 张 train/night Query，默认另存 `runs/cache/svox_train_gallery.safetensors`，不会混用 test Gallery。`--split test` 仍是默认的最终评估协议。

### SVOX train 真实夜景微调

用户已允许将 SVOX train 用作 Night DINOv2 的额外监督数据。`configs/night_full_real.sh` 默认先按 test/day Gallery 和 test/night Query **文件名中的 UTM 坐标**排除 250 米内的 train 图像，避免官方划分中临近帧跨集合。这个过滤不打开 test 图像文件，但使用了 test 坐标元数据，所以研究结论应标注为**探索性官方文件划分评估**。当前数据保留 20,486 张 train/day Gallery 和 378 张 train/night Query；夜图进一步分为 255 张优化、76 张内部验证、47 张空间间隔缓冲。被保留的每张夜图有 4–44 张 25 米内的白天正样本。

真实夜景阶段以冻结的原始白天 Gallery 描述符为目标，使用地理多正样本 InfoNCE：25 米内均为正样本，默认遮蔽 25–250 米的模糊图像，250 米外为负样本。另有可独立关闭的同夜图 Day Teacher 描述符约束和逐层学习率衰减。日志记录有效正负样本数、优化子集和内部验证集的 Recall，按内部验证 Recall@1/5/10 保存最佳 Night 主干。真实夜图和白天 Gallery **不是像素配对**，因此不为它们构造 SGPD Patch 对应。默认从原始白天主干初始化，也可传 `INIT_ADAPTER_DIR` 接续合成阶段的完整 Night DINOv2。

## 2026-10-22 全参数基线

在内部验证选定的 Day→真实夜景全参数模型，test R@1 从 763/823 提高到 765/823（92.71%→92.95%），R@5/10 保持 98.42%/99.15%。仅修复 2 张查询，不能据此声称稳定显著收益。当前所选模型未使用 Qwen SGPD 初始化；合成 SGPD 的真实域额外收益尚未证实。全部来源、配置脚本修正与复现命令见 [全参数实验记录](NIGHT_FULL_RESULTS.md)。

## 2026-10-23 相机增强重跑

新实验固定相同数据划分和训练预算，仅为 Student 添加相机退化；可信对应和所有评估保持原图。直接 Day 初始化的增强真实模型 test 为 765/823、809/823、817/823；增强 Global→SGPD→真实阶段为 766/823、810/823、816/823。完整路线相对上一轮 R@1 仅净增 1 张，SGPD 的独立收益尚未证实。详细内部对照、预先登记的测试路线与错误变化见 [相机增强结果](NIGHT_CAMERA_RESULTS.md)。

`scripts/evaluate_svox_development.py` 可将任一 Night checkpoint 放到与真实训练一致的固定空间验证协议（76 张 Query、20,486 张原始 Day Gallery）中比较；它只解码 train 图像，test 文件名坐标仍仅用于排除带。新增强不改变 `evaluate_pairs.py` 或 `evaluate_svox.py` 的原图预处理。

## 先前 LoRA 对照记录

先前 `adapter_type=lora` 在默认推理强度 `--adapter-scale 1` 的两阶段训练均已跑完：阶段 1 是 Global + Contrast（1 epoch、1,816 步），阶段 2 从该 LoRA 接续 Global + Contrast + SGPD（2 epoch、3,632 步），Keep、OT 和幅度蒸馏均关闭。阶段 2 记录了 393,244 个 A 类可信 Patch，末段每批平均覆盖约 10.9%。每个阶段的 LoRA 权重约 0.59 MB。这些数字与新的全参数 Night DINOv2 是不同实验。

| Night Query / 白天 Gallery | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| SVOX test，原始 SALAD，823 Query | 92.71% | 98.42% | 99.15% |
| SVOX test，阶段 1 LoRA 强度 1 | 92.22% | 98.06% | 98.66% |
| SVOX test，阶段 2 LoRA 强度 1 | 91.86% | 98.06% | 98.66% |
| SVOX train，原始 SALAD，702 Query | 88.46% | 95.16% | 96.01% |
| SVOX train，阶段 1 LoRA 强度 0.05 | 88.46% | 95.30% | 96.30% |
| SVOX train，阶段 1 LoRA 强度 1 | 87.32% | 94.73% | 95.73% |
| SVOX train，阶段 2 LoRA 强度 0.1 | 88.18% | 95.30% | 96.30% |
| SVOX train，阶段 2 LoRA 强度 1 | 85.19% | 93.87% | 96.01% |
| SVOX test，阶段 1 LoRA 强度 0.05 | 92.71% | 98.42% | 99.15% |

强度 0.05 是在 SVOX train 上依次比较 R@1、R@5、R@10，并在指标同分时取较小的非零增量；随后单独评估 test。强度 1 的 test 结果在这项强度诊断之前已经查看过，因此强度 0.05 的 test 数字也应视为**探索性诊断**，不能作为全新盲测结果。阶段 1 的 train 强度 0.25 达到相同的 R@1/5/10；阶段 2 在已试的 0.1、0.25、0.5、0.75、1.0 中均未超过原始模型的 R@1。完整逐次 JSON 存放在相应 `runs/night_lora_global_01/` 和 `runs/night_lora_sgpd_01/` 目录。

合成验证本身几乎饱和：原始模型在 385 对中已命中 384 对 R@1，两阶段 LoRA 均命中 385 对。其配对余弦相似度确实提升，但真实 SVOX 的 R@1 没有提升。这说明当前合成验证不足以判定真实夜景收益；造成差距的具体因素仍需通过额外真实夜景训练或验证数据实验确认。

阶段 2 同时增加了 SGPD 和训练步数，现有阶段 1/2 对比不能单独归因于 SGPD。若继续研究局部损失，应从同一阶段 1 权重出发，以相同步数另训 `PATCH_WEIGHT=0` 的对照实验，并在独立真实夜景验证划分上选择配置。

仓库还保留了 `scripts/evaluate.py`，用于用户自行提供 `path,place_id` 的普通 CSV 数据集。它的同 ID 判正协议**不适用于** SVOX 地理半径真值；SVOX 应使用专门入口。

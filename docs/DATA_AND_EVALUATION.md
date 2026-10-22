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

合成验证可运行 `python scripts/evaluate_pairs.py --adapter-dir runs/night_lora_sgpd_01 --device cuda`。它使用验证清单中的原始白天图作为 Gallery，未适配与适配夜图共享这份 Gallery；同 `(city, place_id)` 判为检索正样本，并额外比较每对夜图与其**精确白天源图**的描述符余弦相似度。该指标只监控合成域，不能替代真实夜景 Recall。

## 真实夜间检索

`scripts/evaluate_svox.py` 使用 `data/svox/images/test/gallery` 中全部 17,166 张图作原始白天 SALAD Gallery，`queries_night` 中 823 张真实夜图作为 Query。SVOX 文件名提供 UTM Easting/Northing；以**25 米**内的 Gallery 图作为正样本，精确计算 Recall@1/5/10。823 张夜图都有至少一个正样本。这个协议与本项目随附的 SALAD `workflow/evaluation.py` 一致。SVOX README 的 10 米用于构建数据集时筛选查询，不是这里的检索评估半径。

```bash
python scripts/evaluate_svox.py --check-data
python scripts/evaluate_svox.py --adapter-dir runs/night_lora_sgpd_01 \
  --device cuda --output runs/night_lora_sgpd_01/svox_test.json
```

评估先提取一次白天 Gallery，再用同一 Gallery 分别计算未适配夜图 baseline 与 Adapter 夜图。缓存绑定完整 checkpoint、预处理和有序图像字节；不同 Adapter 实验可以安全复用原始 Gallery。勿使用 SVOX test Recall 挑选 epoch 或调整超参数；合成验证划分用于开发阶段监控，最终固定配置后再测 SVOX。

需要在真实夜景上做开发阶段诊断时，可使用 `--split train`：它读取 SVOX 的 22,232 张 train/day Gallery 与 702 张 train/night Query，默认另存 `runs/cache/svox_train_gallery.safetensors`，不会混用 test Gallery。`--split test` 仍是默认的最终评估协议。

## 当前 Night 实验记录

默认推理强度 `--adapter-scale 1` 的两阶段训练均已跑完：阶段 1 是 Global + Contrast（1 epoch、1,816 步），阶段 2 从该 LoRA 接续 Global + Contrast + SGPD（2 epoch、3,632 步），Keep、OT 和幅度蒸馏均关闭。阶段 2 记录了 393,244 个 A 类可信 Patch，末段每批平均覆盖约 10.9%。每个阶段的 Adapter 权重约 0.59 MB。

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

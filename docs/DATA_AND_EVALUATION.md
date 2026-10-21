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

合成验证可运行 `python scripts/evaluate_pairs.py --adapter-dir runs/night_sgpd_01 --device cuda`。它使用验证清单中的原始白天图作为 Gallery，未适配与适配夜图共享这份 Gallery；同 `(city, place_id)` 判为检索正样本，并额外比较每对夜图与其**精确白天源图**的描述符余弦相似度。该指标只监控合成域，不能替代真实夜景 Recall。

## 真实夜间检索

`scripts/evaluate_svox.py` 使用 `data/svox/images/test/gallery` 中全部 17,166 张图作原始白天 SALAD Gallery，`queries_night` 中 823 张真实夜图作为 Query。SVOX 文件名提供 UTM Easting/Northing；以**25 米**内的 Gallery 图作为正样本，精确计算 Recall@1/5/10。823 张夜图都有至少一个正样本。这个协议与本项目随附的 SALAD `workflow/evaluation.py` 一致。SVOX README 的 10 米用于构建数据集时筛选查询，不是这里的检索评估半径。

```bash
python scripts/evaluate_svox.py --check-data
python scripts/evaluate_svox.py --adapter-dir runs/night_sgpd_01 \
  --device cuda --output runs/night_sgpd_01/svox_test.json
```

评估先提取一次白天 Gallery，再用同一 Gallery 分别计算未适配夜图 baseline 与 Adapter 夜图。缓存绑定完整 checkpoint、预处理和有序图像字节；不同 Adapter 实验可以安全复用原始 Gallery。勿使用 SVOX test Recall 挑选 epoch 或调整超参数；合成验证划分用于开发阶段监控，最终固定配置后再测 SVOX。

仓库还保留了 `scripts/evaluate.py`，用于用户自行提供 `path,place_id` 的普通 CSV 数据集。它的同 ID 判正协议**不适用于** SVOX 地理半径真值；SVOX 应使用专门入口。

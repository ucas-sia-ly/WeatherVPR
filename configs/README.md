# Night 训练配置

## 可选：真实配对阶段一

`night_real_stage1.sh` 保留 8 个 Qwen + 2 个 Source Night 视图，B/D 再加入 2 个审核真实配对全局监督；A 为无额外分支，C 为等量合成配对全局对齐。入口固定原始 Day 初始化与 Late4，禁止加载历史 Night adapter，默认不改变 `night_specialist.sh` 的训练行为。数据审核格式、参数和逐项运行命令见 [阶段一说明](../docs/NIGHT_REAL_PAIRS.md)。

## 默认：独立 Night 主干的后四层微调

`night_specialist.sh` 从原始 Day checkpoint 初始化独立 Night DINOv2-B/14，默认 `FULL_TRAIN_SCOPE=late FULL_TRAIN_BLOCKS=4`，直接更新 blocks 8–11 的原始权重。其余 Night 参数（包括最终 Norm）冻结，Day Teacher 与共享 SALAD 全部冻结；输入梯度可穿过 Norm 与 SALAD。权重仍保存完整 Night 主干，便于独立推理；这不是 LoRA。

默认 `LAYER_DECAY=1`、LR `1e-6`、batch 8、1 epoch、AdamW、BF16、activation checkpoint、cosine 日程与 32 次更新 warmup，`NUM_WORKERS=0`。`FULL_TRAIN_SCOPE=all` 可切回全部参数消融；`FULL_TRAIN_SCOPE=early|late FULL_TRAIN_BLOCKS=N` 可比较其它层范围。SALAD 原始训练设置与当前蒸馏任务的区别见 [SALAD 训练说明](../docs/SALAD_TRAINING.md)。

主分支读取 Qwen Night（可叠加相机退化）。辅助分支从训练 Day 原图生成必做曝光衰减的低照度视图；干净 Day 仅供冻结 Teacher 和 Day bank。参数见 `source_night.json`，`SOURCE_NIGHT_WEIGHT=0` 可关闭辅助分支，`SOURCE_NIGHT_SAMPLES` 控制每步辅助样本上限。真实夜图仅评估。

默认 Global / Contrast / Relation 权重为 `0.1 / 0.1 / 0.1`、Source Night 为 `0.5`；SGPD / Keep / OT / Magnitude 默认关闭，可独立赋予正权重。固定训练 Day bank 有 7,263 keys / 6,441 地点；InfoNCE 对同一 `(city, place_id)` 的全部条目视为正样本，默认 64 个难负样本。关系蒸馏排除同地点，单独由 `RELATION_WEIGHT` 和 `RELATION_HARD_NEGATIVES` 控制。

32 GB 主机应逐项串行运行，并使用内存限制：

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUTPUT=runs/night_late4_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_late4_02.memory.jsonl \
  -- bash configs/night_specialist.sh
```

历史 `source_night` 是**全部参数微调、layer decay 0.8**，其 SVOX test R@1/5/10 为 93.07% / 98.66% / 99.03%；默认值改变后，原命令不再代表该历史配置。复现必须显式设置 `FULL_TRAIN_SCOPE=all LAYER_DECAY=0.8`，完整命令见 [历史实验](../docs/NIGHT_SPECIALIST_RESULTS.md)。用户 `runs/night_late4_01` 保留原始记录，不将其当前 test R@1 解释为超过历史 `source_night`。

`evaluate_pairs.py --gallery-cache <path>` 可缓存冻结 Day Gallery，并附加排除精确源图的 `cross_view_baseline` / `cross_view_adapter`。该指标目前仅覆盖 90 Query / 45 地点的多视角子集，不能替代真实夜间评估。

## 当前 Patch / 注意力对照

`night_patch.sh` 复用后四层 Night 配置，改用 `night_patch_camera.json`：整张增强和曝光操作概率均为 1，使每张 Qwen 输入都经过相机处理。默认 `PATCH_WEIGHT=0.1 ATTENTION_WEIGHT=0 PATCH_VISIBILITY=photometric`；设 `ATTENTION_WEIGHT=0.05` 可加入最后一层 DINOv2 CLS 注意力 KL。SGPD 和注意力仅监督原 Stage 2B A 类可信对应，退化输入的黑场/裁剪像素比例只降低置信度，不创建对应。Teacher 输入保持干净，Source Night 分支不变。

三组固定对照为 `PATCH_WEIGHT=0 ATTENTION_WEIGHT=0`、`0.1 / 0`、`0.1 / 0.05`。各组必须使用独立新输出目录并通过内存保护串行运行；本轮上限 6 GiB、high 5 GiB、系统可用内存保留 4 GiB。公式、CLI、复现和开发集选择规则见 [Patch 与注意力蒸馏](../docs/NIGHT_PATCH_ATTENTION.md)。

## 历史全部参数与相机增强对照

`night_full_global.sh` 和 `night_full_sgpd.sh` 显式默认 `FULL_TRAIN_SCOPE=all`，保留全部参数微调的历史消融；可以环境变量覆盖范围。第一阶段为 Global + 跨域多正样本 InfoNCE，第二阶段接续加入 Stage 2B A 类可信对应上的 SGPD。两阶段默认 LR `1e-6`、batch 8、BF16、AdamW、各 1 epoch，使用容量 256 的白天描述符队列和地点分散采样。

```bash
OUTPUT=runs/night_full_global_replay_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_global_replay_02.memory.jsonl \
  -- bash configs/night_full_global.sh
OUTPUT=runs/night_full_sgpd_replay_02 \
  INIT_ADAPTER_DIR=runs/night_full_global_replay_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_full_sgpd_replay_02.memory.jsonl \
  -- bash configs/night_full_sgpd.sh
```

`PATCH_WEIGHT`、`KEEP_WEIGHT`、`OT_WEIGHT`、`MAGNITUDE_WEIGHT` 分别配置，后三项默认关闭。使用 `BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 可在显存较小时保持有效 batch。旧 `night_global.sh` / `night_sgpd.sh` 保留 LoRA（rank 8、alpha 16、最后四层 attention `qkv/proj`）和 `ADAPTER_TYPE=output` 消融。

两个 full 脚本支持 `NIGHT_AUGMENTATION=off|camera`、`AUGMENTATION_CONFIG`、`AUGMENTATION_STRENGTH`、`AUGMENTATION_PROBABILITY`、`AUGMENTATION_EPOCH_OFFSET`。仅指定 JSON 不会打开增强，还需指定 `camera`。`night_camera_replay.sh` 保留增强 Global 与同预算 Global / SGPD 接续的历史流程；32 GB 主机运行每阶段时均应通过 `run_bounded.py`，避免绕过内存约束。操作参数见 [增强说明](../docs/NIGHT_CAMERA_AUGMENTATION.md)，已有数字见 [相机增强结果](../docs/NIGHT_CAMERA_RESULTS.md)。

`configs/dino_salad_stage2b_calibration.json` 只与当前 checkpoint 和 224×224 预处理匹配；训练默认排除 `data/splits/heldout_places.json`。其它天气尚未训练，需先准备各自的配对清单和测试集。

## 阶段二部分 Patch 风格增强

`night_token_stage2.sh` 用 `STAGE2_ARM=T0|T1|T2|T3` 固定普通 Token 增强与真实配对监督的 2×2 对照；只有 Qwen 主分支在 block 7 后增强部分 Patch，CLS 不直接修改，推理关闭。四组均从原始 Day 权重初始化，沿用 fast 后端与 4×2 的主有效批次。

推荐运行 `python scripts/run_token_style_stage2.py --run-root runs/night_token_stage2_new_seed42`，由队列统一冻结参数、串行内存保护、训练和 SVOX 开发评估；真实审核与 Day 缓存默认从阶段一 latest 复用。实施细节及预检查见 [阶段二运行](../docs/NIGHT_TOKEN_STAGE2.md)，该实验不自动引入扩散模型或依据外部 test 搜索参数。

## 修订方案：多 seed、域覆盖与独立 Source 地点扩展

统一后端多 seed 使用 `scripts/run_stage1_multiseed.py --output runs/new_multiseed`：新增 seeds 43、44 的 A/B/C，复用历史 seed 42，固定 microbatch4×累积2、Source1、额外辅助1，只评估开发集。`--plan-only` 生成冻结计划；`--max-steps 3 --skip-evaluation` 只作实现验证。

独立 Source 规模入口为 `night_gsv_scale.sh`，使用 `GSV_SCALE=G1|G2` 和 `GSV_SCALE_ARM=A|B`。每档配置 `GSV_G1_MANIFEST` / `GSV_G2_MANIFEST` 及不同的 `GSV_G1_CACHE` / `GSV_G2_CACHE`；manifest 必须附带来源/排除证据签名。Source 独立采样，使用该档 Source-only 训练 bank，主 Qwen 流和 bank 固定。B 还须设置 `REAL_PAIRS_REVIEW` 与真实目标缓存，默认真实权重0.1。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
GSV_SCALE=G2 GSV_SCALE_ARM=A \
GSV_G2_MANIFEST=outputs/revised_plan_20261025/data_inventory/g2_day.jsonl \
GSV_G2_CACHE=runs/cache/new_G2_source_day.safetensors \
OUTPUT=runs/new_G2_A \
  "$PYTHON" scripts/run_bounded.py --max-gib 7 --high-gib 6 --reserve-gib 2 \
  --log runs/new_G2_A.memory.jsonl -- bash configs/night_gsv_scale.sh
```

先使用 `scripts/prepare_research_sources.py` 生成训练来源清单；`scripts/analyze_domain_coverage.py` 只测量冻结模型的统计覆盖与实际扰动，不运行检索或自动打开真实供体。完整边界与本地数据数量见 [修订方案实现](../docs/REVISED_RESEARCH_IMPLEMENTATION.md)。

四单元队列可使用 `python scripts/run_gsv_scale.py --output runs/new_scale_smoke --max-steps 3 --skip-evaluation` 先验证并构建 Source 缓存，再用新目录执行 `python scripts/run_gsv_scale.py --output runs/new_scale_seed42 --seed42`。完整模式要求缓存已就绪，冻结输入与代码后逐组训练和评估 SVOX dev；不复用历史 A/B 作为独立 Source 的 G1。

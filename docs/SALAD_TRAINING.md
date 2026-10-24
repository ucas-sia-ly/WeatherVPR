# 对照 SALAD 的 Night 后四层训练

## 当前默认

Night 仍是从 Day checkpoint 初始化的**独立 DINOv2**，默认只更新最后四个 Transformer Blocks 的原始参数，即 ViT-B/14 的 `8–11`，共 **28,357,632** 个可训练参数。前八层、embedding、tokens、最终 LayerNorm、Day Teacher 和共享 SALAD 参数冻结；SALAD 与最终 norm 允许输入梯度通过。保存完整独立 Night backbone，推理没有 LoRA 依赖。

```text
Day Gallery → 冻结原始 Day DINOv2 → 冻结 SALAD → 固定 Gallery
Night Query → 独立 Night DINOv2   → 同一 SALAD → 检索原始 Gallery
                  仅 blocks 8–11 更新
```

训练 CLI 与 `configs/night_specialist.sh` 默认 `--adapter-type full --full-train-scope late --full-train-blocks 4`。`full` 表示完整独立 Night 模型的存储形式，实际更新范围由 scope 控制；`--full-train-scope all` 才更新全部 DINOv2 参数。旧 `night_full_global.sh` / `night_full_sgpd.sh` 显式保留 `all` 默认，作为历史全参数对照，也接受 scope 覆盖。

## SALAD 源码的设置与本项目的区别

核对了本地原始 [main.py](../third_party/salad/main.py)、[DINOv2 forward](../third_party/salad/models/backbones/dinov2.py) 和项目实际使用的 [workflow/model.py](../third_party/salad/workflow/model.py)。原始训练入口与 `workflow` 扩展入口不是同一套超参数；下表的 SALAD 列取原始 `main.py`。

| 项目 | 原始 SALAD 入口 | Night 默认 |
| --- | --- | --- |
| Backbone | DINOv2-B/14，最后 4 个 Block | 独立 Night DINOv2-B/14，最后 4 个 Block |
| 最终 LayerNorm | 启用并训练 | 启用、参数冻结，保持用户已有 late4 设定 |
| SALAD | 与 Backbone 联合训练 | 参数冻结，固定 Day 检索空间 |
| 输入 | 224×224 | 224×224 |
| 采样 | 60 地点 × 每地点 4 张 | microbatch 8，地点分散；固定 Day bank 提供多正样本和难负样本 |
| 主判别目标 | MultiSimilarityLoss + Miner，margin 0.1 | 跨域多正样本 InfoNCE；另有 Global、Relation 等独立项 |
| 增强 | RandAugment，3 个操作 | Qwen Night 相机退化；训练 Day 模拟低照度辅助分支 |
| 优化器 / LR | AdamW / `6e-5` | AdamW / `1e-6` |
| weight decay | `9.5e-9` | `0.01` |
| LR 日程 | 4000 次更新线性降至 0.2 | 32 次更新 warmup + cosine，最低比例 0.1 |
| epoch / 精度 | 4 / FP16 | 1 / BF16 + activation checkpoint |

后四层范围参考 SALAD；最终 norm 冻结与原版存在明确差异，不称为完全复现 SALAD。也不直接复制原版的大 batch、学习率或聚合器联合训练，因为本项目必须匹配不变的 Day Gallery。

冻结前缀在没有可训练参数、输入无需梯度时不建立有效反向图；训练层、最终 norm 和 SALAD 之间保留输入梯度。对可选 `early` 消融，后部冻结层同样不能用 `no_grad` 截断前层梯度。测试覆盖选定层有梯度、其它参数无梯度且优化后不变。

## 用户已有的后四层结果

保留 `runs/night_late4_01`，没有重写其权重、配置或评估。它完成 908 次更新，训练 7,263 对 Day/Qwen 图像，使用相同训练 Day bank 与 Source Night 辅助分支。权重 SHA-256：

```text
b6748afc4c026c46bb2d8057ee4636cf88a2bb882af51eeae17fbef29ad61190
```

以下均为原始 224 单尺度、同一 Day Gallery，按 R@1 / R@5 / R@10 顺序列出。

| 模型 | SVOX 开发命中数 / 702 | SVOX test 命中数 / 823 | test Recall（%） |
| --- | --- | --- | --- |
| 原始 Day | 621 / 668 / 674 | 763 / 810 / 816 | 92.71 / 98.42 / 99.15 |
| 用户 `night_late4_01` | 618 / 668 / 676 | 761 / 812 / 816 | 92.47 / 98.66 / 99.15 |
| 已有 `source_night`，全部参数更新 | 620 / 668 / 677 | 766 / 812 / 815 | 93.07 / 98.66 / 99.03 |

后四层的开发 Top-1 优于此前同设置的全量、前六层和后六层候选，但现有记录尚未超过 `source_night` 或原始 Day 的 Top-1。将后四层作为下一轮训练默认，不将它宣称为已验证的全局最优模型；保留 `source_night` 作为已有检索参考。

## 训练目标与清理

主目标仍为 `S(aug(Qwen Night)) → T(clean Day)`，辅以训练 Day 模拟低照度视图的相同跨域目标。保留可独立关闭的 Global、Contrast、Relation、SGPD、Keep、Conditional OT、特征幅度项。固定 Day bank 使用 `(city, place_id)` 处理多正样本；SGPD 只接受 Stage 2B 验证的 A 类对应，不能按相同坐标假设 Qwen 编辑对齐。

已移除 Day Anchor / Preservation / Restoration 的 CLI、数据视图、训练分支及调度器，旧参数会报错。Source Night 是强制低照度输入的检索训练分支，继续保留。真实夜景数据只作只读评估。

失败的正则/旧 Guided 实验、相关文档与专用审计入口及重复 smoke 输出已清理；清单和保留模型哈希见 [清理记录](../outputs/maintenance/cleanup_20261024.json)。基础权重、数据、Stage 2B 校准、完整有效对照及 Gallery/训练 bank 缓存均保留。旧保留模型中的历史元数据不改写。

## 32 GB 主机上的运行方式

每次使用新的 `OUTPUT`，所有训练和评估串行执行。`run_bounded.py` 要求 Linux cgroup v2 + systemd user scope；任务及后代 `MemoryMax=8 GiB`、`MemoryHigh=6 GiB`、swap 关闭，启动需至少 12 GiB 系统可用内存，运行时低于 4 GiB 则结束实验任务。限制不控制浏览器等其它程序。默认 worker 为 0。

```bash
export PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# 默认独立 Night 模型，后四层；保持 layer decay=1 与现有 late4 一致
OUTPUT=runs/night_late4_02 \
  "$PYTHON" scripts/run_bounded.py --log runs/night_late4_02.memory.jsonl \
  -- bash configs/night_specialist.sh

# 只读开发集诊断，沿用原始 Day Gallery 缓存
"$PYTHON" scripts/run_bounded.py --log runs/night_late4_02.dev.memory.jsonl \
  -- "$PYTHON" scripts/evaluate_svox.py --split train \
  --adapter-dir runs/night_late4_02 --device cuda --batch-size 16 --num-workers 0 \
  --gallery-cache runs/cache/svox_train_eval_day.safetensors --reference-cache-mib 1024 \
  --output runs/night_late4_02/svox_train.json
```

全参数消融显式设置 `FULL_TRAIN_SCOPE=all`；改变学习率、日程或损失时分别对照，不从不同设置混合归因。开发选择后固定配置再评估 test，不根据测试分数选择层数或融合权重。

## 本次修改的验证

- [158 项单元测试通过](../outputs/maintenance/tests_20261024.log)，覆盖 CLI 默认与消融覆盖、退役参数拒绝、梯度和冻结参数等。
- 默认 `night_specialist.sh` 完成 [2 次真实优化器更新检查](../outputs/maintenance/late4_validation_20261024.json)：Blocks 8–11 均发生变化，其它 119 个张量与原始 Day 逐位一致；Source Night 每步有 128 个有效负样本对。临时模型检查后已删除，不作为新的实验结果。
- 本次集成检查 cgroup 内存峰值 **3.78 GiB**，系统可用内存最低 **11.01 GiB**，没有 OOM；原有两份保留模型的权重哈希复核一致。

# 阶段二首轮：部分 Patch 统计增强

本实现对应 [idea.md 第 12.1 节](../idea.md#121-先验证普通-token-增强)。论文/作者源码依据、迁移限制与后续统计库/伪域方向见 [专项分析](TOKEN_STYLE_DG_REVIEW.md)。本轮仅运行普通批内 Qwen 统计混合，不提取扩散特征，也不引入真实风格供体库。

2026-10-25：264 项测试及 T0–T3 各三微批/两次更新的实际训练通过。四组均有 119 个冻结张量与原始 checkpoint 完全相同，后四层有权重更新；主/Source/真实流和 T2/T3 风格选择核对一致。64 张训练图诊断的 gain 最大 1.185、中心化相对 RMS 变化最大 5.1%，无非有限值，未加入额外限幅。seed 42 正式训练与开发评估队列位于 `runs/night_token_stage2_20261025_075002_seed42/`，进度以其 `status.json` 为准，启动不代表已获得 Recall 提升。

正式四组现已全部完成，各 908 次更新。T0/T2 为 617/668/676，T1/T3 为 622/667/675；普通 Token 增强没有增加任何 Top-k 命中，也没有逐查询修复。当前配置默认保持关闭，完整评价与建议见 [阶段二结果](NIGHT_TOKEN_STAGE2_RESULTS.md)。

## 固定实验

| 组别 | 真实配对辅助 | Qwen Token 增强 |
| --- | --- | --- |
| T0 | 无 | 无 |
| T1 | 0.1 × 真实配对余弦 | 无 |
| T2 | 无 | 有 |
| T3 | 与 T1 相同 | 与 T2 相同 |

四组从原始 Day checkpoint 独立初始化，不继续训练旧 Night 权重。使用同一 fast 后端：CUDA Day bank、不启用激活重计算、4 个加载 worker、预取 1 批。保持 microbatch 4、梯度累积 2、Source 每微批 1、真实辅助每微批 1；完整一轮为 7263 个主视图、1816 个 Source 视图、908 次更新，T1/T3 另有 1816 次真实曝光。BF16、LR 1e-6、32 次更新 warmup、其余损失和原始固定 Day 图库保持原样。

新增增强参数固定：每个 Qwen 样本独立 `p=0.5`；每个启用样本选择 `floor(256×0.3)=76` 个 Patch；混合系数 `0.2×(1−Beta(0.1,0.1))`。供体由当前微批的无固定点排列选出，不允许自供体；实际混合池为 4，不是累计后的 8。

## 实现与开关

- [token_style.py](../src/weather_vpr/training/token_style.py)：纯训练变换，不注册模型参数、buffer 或长期 hook。均值/标准差为 FP32、每图沿 Patch 轴计算，`sqrt(var(correction=0)+1e-6)`，统计停止梯度。
- [实际 DINO 前向](../third_party/salad/workflow/model.py)：显式在零起算 block 7 完整输出后、block 8 前调用变换。回调在 block checkpoint 包装之外，不在反向重算中重抽增强。
- [Night 接口](../src/weather_vpr/models/adapter.py)：只允许独立 full Night 显式传入回调。只在训练模式接受变换，普通推理保持原调用和 state_dict 结构。
- [训练入口](../src/weather_vpr/training/train.py)：只有 Qwen 主分支传回调；Source、真实配对、Teacher 不传。所有分支原本的损失缩放、串行反传和更新顺序不变。

CLS、未选样本及未选 Patch 在注入点位级保留，空间索引不交换；后续注意力仍会影响全部 Token。这不保证地点信息或 SALAD 聚类分配不变。

增强使用由 seed/epoch/step 的 SHA-256 派生的局部 Python RNG，不消耗全局 Python/NumPy/Torch 随机流。零概率、零比例、零强度和微批不足两图均直接返回原 Tensor。主数据相机增强与 Source/真实分支的随机流继续独立。

CLI 默认关闭，可用 `--token-style-probability`、`--token-style-fraction`、`--token-style-strength`、`--token-style-beta`、`--token-style-after-block` 配置。目前激活时要求 full Night Late4、block 7、仅全局监督；不将未经验证的组合悄悄扩展到其它架构。

## 检查与运行

统一由 [run_bounded.py](../scripts/run_bounded.py) 串行执行模型工作，不同时启动诊断与训练。当前设备其它程序占用较多 RAM，任务限制为 7 GiB、high 6 GiB、系统余量 2 GiB，禁用任务 swap。

```bash
PYTHON=/home/admin123/miniconda3/envs/a2b-lightvpr/bin/python

# 工程测试
"$PYTHON" scripts/run_bounded.py --max-gib 7 --high-gib 6 --reserve-gib 2 \
  --log runs/stage2_tests.memory.jsonl -- \
  env PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -m unittest discover -s tests -v

# 新目录，三微批冒烟；只在冒烟中关闭 warmup，不作性能结论
"$PYTHON" scripts/run_token_style_stage2.py \
  --run-root runs/night_token_stage2_smoke_new --max-steps 3 --skip-evaluation

# 四组完整训练与 SVOX 开发评估，输出目录必须为空
"$PYTHON" scripts/run_token_style_stage2.py \
  --run-root runs/night_token_stage2_new_seed42
```

队列自动读取阶段一 latest 中的真实审核清单，并优先复用已有真实 Day 缓存；也可通过 `--real-pairs-review`、`--real-day-cache` 显式指定。缓存消费者仍独立验证来源签名。每个子训练/评估任务均有内存日志和退出检查。

首次完整训练前，还应在纯训练样本上运行 [诊断脚本](../scripts/diagnose_token_style.py)，测量 raw gain 分位数、`gain>2/4/8` 比例、标准化偏移、中心化 RMS 相对变化，以及最终描述符/SALAD 分配变化。小混合系数不保证小扰动；任何限幅规则须在检索评估前冻结并统一用于所有增强组。该诊断不计算开发/测试 Recall，不调整模型权重。

## 输出与解释

`runs/stage2_training_latest.json` 指向最新队列。每个完整队列目录包含：

- `experiment_plan.json`：训练前冻结预算、参数、输入/代码哈希、命令及开发评估规则；运行间发现文件改变则停止。
- `status.json`、`report.md`、`summary.json`：阶段状态、R@1/5/10 命中数、逐查询修复/退化及流一致性检查。
- `T0`–`T3`：普通 Night 权重、元数据、日志、独立训练计划和 `svox_train.json`。
- 每组 `*.train.console.log`、`*.eval.console.log` 和内存日志。

日志包含逐图启用、供体、λ、Patch 索引、gain、实际 RMS 变化；元数据保存累计曝光与最大观测变化。队列核对四组主/Source 流，T1/T3 的真实流，T2/T3 的风格选择流，评估前后最终权重哈希，以及相同原始 Day 图库。

本轮只评估 SVOX 开发集，不根据 Tokyo/RobotCar 或 SVOX test 调参。单 seed 的微小变化不足以认定成功；若有信号，再在固定配置下补齐相同多 seeds。真实统计供体、配对残差与 GUIDE 伪域采样属于后续独立实验。

# 资源状态

当前工作目录已具备 night 实验所需的全部大文件，无需额外下载：

| 资源 | 当前路径 | 用途 |
|---|---|---|
| 完整 DINOv2-B/14 + SALAD checkpoint | `weights/dino_salad.ckpt` | 提供冻结 Day Teacher、冻结共享 SALAD，以及可全参数训练的 Night DINOv2 初始权重；SHA-256 `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada` |
| SALAD workflow 源码 | `third_party/salad/` | 本项目修改过的完整 Night 前向、预处理、Stage2B 对应和 SVOX 协议 |
| DINOv2 源码 | `third_party/dinov2/` | 随项目提供的源码副本，具体 revision 尚未核验；本地构建 Backbone，不联网加载 |
| 白天/生成夜间训练图 | `data/train/day/night/`、`data/train/weather/night/` | 7,719 对已核验图像 |
| SVOX 真实夜间测试集 | `data/svox/images/test/` | 17,166 Gallery、823 night Query |

校准文件已从仓库内现有历史诊断核验并复制到 `configs/dino_salad_stage2b_calibration.json`。Qwen 生成器及其模型权重**不用于** Night DINOv2 训练。基础 Python 环境中的 PyTorch CUDA 构建与当前驱动不兼容；已验证 `a2b-lightvpr` conda 环境可用。

其他夜间评估也已具备 Tokyo247 图像、官方 MAT 和 RobotCar 公开 train 位姿/白天模型，无需再下载大文件；新环境依赖包含 SciPy。RobotCar 官方 test 与 Aachen 的 Query 真值不公开，下载更多图像或权重不能补齐 Recall 真值；若需要这两项 test Recall，须提供可信的 Query→Gallery positives。可用协议与官方来源见 [其他夜间评估](EXTERNAL_NIGHT_RESULTS.md)。

若将项目复制到另一台机器，请保留随发行包提供的运行源码：`third_party/salad/` 下的 `workflow/`、`models/`、`configs/`、`LICENSE` 和 README 文件，以及 `third_party/dinov2/` 的源码目录。SALAD 约 26 GB 的历史 `outputs/`、旧数据与权重无需复制；当前训练数据和权重使用项目根目录的 `data/`、`weights/`。

`bash scripts/setup_sources.sh` 仅检查 SALAD 与 DINOv2 的必要运行文件和接口，不下载或替换源码，也不声称已核验其 revision。任一源码目录缺失或不兼容时，必须从此项目发行副本或备份恢复上述运行源码；不能用上游 SALAD 或未修改的 WORKSPACE 替代本地 SALAD 修改。DINOv2 原始来源为 [官方仓库](https://github.com/facebookresearch/dinov2)，当前随附副本的具体 revision 尚未核验。需要补齐权重时，可从 [SALAD 官方权重](https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt) 下载相同完整 checkpoint，模型加载及校准会检查 checkpoint 匹配。具体源码恢复说明见 [第三方源码](../third_party/README.md)，许可证保留在各自目录。

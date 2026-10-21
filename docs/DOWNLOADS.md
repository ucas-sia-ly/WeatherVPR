# 资源状态

当前工作目录已具备 night 实验所需的全部大文件，无需额外下载：

| 资源 | 当前路径 | 用途 |
|---|---|---|
| 完整 DINOv2-B/14 + SALAD checkpoint | `weights/dino_salad.ckpt` | 冻结 Teacher/Student 主干，SHA-256 `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada` |
| SALAD workflow 源码 | `third_party/salad/` | 模型、预处理、Stage2B 对应和 SVOX 协议 |
| DINOv2 源码 | `third_party/dinov2/` | 本地构建 Backbone，不联网加载 |
| 白天/生成夜间训练图 | `data/train/day/night/`、`data/train/weather/night/` | 7,719 对已核验图像 |
| SVOX 真实夜间测试集 | `data/svox/images/test/` | 17,166 Gallery、823 night Query |

校准文件已从仓库内现有历史诊断核验并复制到 `configs/dino_salad_stage2b_calibration.json`。Qwen 生成器及其模型权重**不用于** Adapter 训练。基础 Python 环境中的 PyTorch CUDA 构建与当前驱动不兼容；已验证 `a2b-lightvpr` conda 环境可用。

若将项目复制到另一台机器，可运行 `bash scripts/setup_sources.sh` 获取缺失源码，并从 [SALAD 官方权重](https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt) 下载相同完整 checkpoint；模型加载及校准会检查 checkpoint 匹配。第三方许可证保留在各自目录。

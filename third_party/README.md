# Third-party code

本项目随附两份第三方源码，在项目内部运行，不导入外部 WORKSPACE 模块：

- `third_party/salad/`：最初来自 `WORKSPACE` 的 `claude` 分支，并包含本项目对完整 Night DINOv2 梯度路径的修改，以及 SALAD 辅助输出、可信 Patch 对应和预处理接口。保留原始 `salad/LICENSE`。
- `third_party/dinov2/`：来自 [官方 DINOv2 仓库](https://github.com/facebookresearch/dinov2) 的源码副本，保留上游许可证。当前副本没有独立 Git 元数据，尚未核验其具体 revision。

复制项目到另一台机器时，请保留随发行包提供的运行源码：SALAD 的 `workflow/`、`models/`、`configs/`、`LICENSE` 和 README 文件，以及 DINOv2 的源码目录。SALAD 中约 26 GB 的历史 `outputs/`、旧数据和权重不属于这些运行源码，无需复制；当前模型权重与数据另放在项目根目录的 `weights/`、`data/`。

`bash scripts/setup_sources.sh` 虽沿用 setup 名称，实际只检查两个随附目录的必要文件与接口，包括 SALAD 的 `full_finetune` 前向和 DINOv2 的 `dinov2_vitb14`、Token 准备接口；不下载或替换任何源码。检查通过仅表明这些接口存在，不代表已核验源码 revision 或全部运行行为。

任一运行源码目录缺失或不兼容时，脚本直接停止，并提示从此 WeatherVPR 发行副本或备份恢复上述运行源码。原版 SALAD 或未修改的 WORKSPACE 代码不能直接替代本项目修改过的 SALAD；DINOv2 也应恢复随本项目提供的源码副本，以复现当前已验证的运行环境。

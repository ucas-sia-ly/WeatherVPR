# Third-party code

`bash scripts/setup_sources.sh` 会在**本独立项目内部**下载：

- `third_party/salad/`：复用 SALAD、Patch 对应及预处理代码，来自 `WORKSPACE` 的 `claude` 分支。保留原始 `salad/LICENSE`。
- `third_party/dinov2/`：官方 DINOv2 代码，保留上游许可证。

不需要将这个项目放在外部 WORKSPACE 下，也不导入外部工作区模块。
如果不能访问 GitHub，也可以把已有 WORKSPACE 的 `salad/` 文件夹**复制到** `third_party/salad/`，再复制已有 DINOv2 源码到 `third_party/dinov2/`。
注意：请使用包含 `workflow/model.py`、`pair_correspondence.py` 的 `claude` 分支代码，原版 serizba/salad 并不直接提供这些接口。

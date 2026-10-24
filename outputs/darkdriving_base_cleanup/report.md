# DarkDriving_Base 清理完成

当前数据目录：`data/DarkDriving_Base/`。

- 原始 9,538 对，成对移出含损坏图像的 8 对，以及完整日夜内容重复的 10 对冗余图组。
- 保留 9,520 对，Day/Night 各 9,520 张，无孤立图像。双方图像内容同时重复的样本已去重。
- 只有单侧内容重复的图组继续保留，`content_group` 标明精确内容复用关系；不将其推断为地点标签。
- 全量 19,040 张图像完成严格解码、文件及解码 RGB SHA-256 校验，错误 0。完成时间：2026-10-24T23:11:13.126606+08:00。
- `pairs.csv` / `manifest.jsonl` 均使用数据集内相对路径；清单 SHA-256：`b02e4d06cc24e489a44b18252eb4509ca29dc535823a2ce656adb8a0786b3761`。
- 4 项清理/加载单元测试通过；真实数据 DataLoader 以 2 workers 连续读取 3 个 batch，每侧形状 `[8,3,224,224]`，float32 且数值有限、位于 [0,1]。
- 36 张移出文件全部保留在 `20261024/quarantine/`，备份 SHA-256 与原审计逐项一致；恢复记录见 `20261024/moves.json`，按图组汇总见 `20261024/removed_pairs.csv`。
- 原图未重新编码、改名或改变分辨率，原始压缩包未修改。

加载示例与完整说明：`docs/DARKDRIVING.md` 或数据集内 `README.md`。复查命令：

```bash
python scripts/prepare_darkdriving.py --root data/DarkDriving_Base --check-only
```

本数据仍未作地点/路线隔离的训练测试划分，未新增地理或像素级真值，未加入现有 Qwen 训练清单。历史审计目录中的 CSV 描述清理前状态，实际使用请读当前数据目录内清单。

> 历史快照：本报告及本目录清单描述清理前状态。数据集后续已清理为 9,520 对；当前可用清单位于 `data/DarkDriving_Base/`，说明见 `docs/DARKDRIVING.md`。

# DarkDriving_Base 数据集检查报告

检查时间：2026-10-24T23:00:59.629295+08:00（Asia/Shanghai）

结论：当前数据集含 8 张无法严格解码的 JPEG，不能认定全部正常。排除涉及的 8 个同名日夜图组后，保留 9,530 组双侧可读图像。原始数据未修改。

| 项目 | Day | Night |
| --- | ---: | ---: |
| 文件数 | 9,538 | 9,538 |
| 严格解码通过 | 9,536 | 9,532 |

- 总计 19,076 个文件，20,108,997,552 字节（18.73 GiB）。
- 同名覆盖 9,538 组，Day 单侧 0，Night 单侧 0。
- 分辨率：Day {'2448x2048': 9538}；Night {'2448x2048': 9538}。
- 图像格式：Day {'JPEG': 9538} / {'RGB': 9538}；Night {'JPEG': 9538} / {'RGB': 9538}。
- 文件字节重复组 228；可解码 RGB 像素重复组 228。
- 解码警告 0；极端黑白/近常量图候选：Day 0，Night 0。

## 损坏文件

| 路径 | 严格解码错误 |
| --- | --- |
| `Day/in_1_1728606329132532736.jpg` | OSError: broken data stream when reading image file |
| `Day/in_4_1728610437458285824.jpg` | OSError: broken data stream when reading image file |
| `Night/in_1_1728606258599845888.jpg` | OSError: broken data stream when reading image file |
| `Night/in_1_1728606259499429888.jpg` | OSError: broken data stream when reading image file |
| `Night/in_1_1728606707996788992.jpg` | OSError: broken data stream when reading image file |
| `Night/in_2_1728607961134389760.jpg` | OSError: broken data stream when reading image file |
| `Night/in_4_1728610004382670848.jpg` | OSError: broken data stream when reading image file |
| `Night/in_4_1728610182769377024.jpg` | OSError: broken data stream when reading image file |

## 原始压缩包核验

本地压缩包：`/home/admin123/my_software/网盘下载地址/DarkDriving_Base.zip`。
中央目录中共有 19,076 个图像文件，解压目录无缺失、无额外文件、无大小差异。全部 19,076 个本地文件的 CRC32 与压缩包记录一致，差异 0。
另外逐个解压了上述 8 个损坏成员，验证解压内容 CRC32，并与本地图像比较 SHA-256；它们均字节相同且原包成员同样无法严格解码。因此重新解压该压缩包无法修复这些图像，需要另一个可靠来源，或在使用时排除。压缩包一致性不等于已核验发布者官方校验和。

## 可用清单与使用边界

- `valid_pairs.csv`：9,530 个同名图组；两侧均完整解码通过，含绝对路径和 SHA-256。
- `excluded_pairs.csv`：8 个排除图组，包含失败侧及原因。
- `files.jsonl`：逐文件检查、SHA-256、尺寸、模式和缩略图亮度统计。
- `pairs.csv`：全部同名图组的亮度、相关性和尺寸诊断；未通过解码者没有亮度指标。
- `archive_inventory.json`：原包文件清单、CRC32 和损坏成员核验记录。
- `sequences.json`：23 个文件名前缀组的数量与时间戳跨度；`in_5` 仅有 1 组，该数量与本地原包一致，不能据此认定解压丢失。
- `contact_*.jpg`：每个文件名前缀的首/中/末图组，以及亮度和相关性极值候选。

同名关系只证明文件结构覆盖；没有将文件名推断为官方 place_id，也没有将其声明为像素级配准真值。现有训练加载器需要带来源与审核字段的 manifest，通用评估加载器需要带 `place_id` 的 CSV；本次输出不虚构这些标注。若后续划分训练/测试集，应考虑连续帧和路线重叠，不能把随机分帧当作地点隔离。

检查方法：Pillow `verify()` 加全分辨率 `load()`，关闭容错截断；计算压缩文件和解码 RGB 的 SHA-256；亮度指标采用 128×128 BOX 缩略图、0.299R+0.587G+0.114B。OpenCV 容错预览只用于显示损坏现象，未用于接受坏图。

## 重复与人工抽查

完全相同的图像分布：Day 28 组，涉及 66 个文件、冗余 38 个；Night 200 组，涉及 416 个文件、冗余 216 个。两侧合计 228 组、254 个冗余副本；Day 与 Night 之间没有字节相同的图像。重复文件逐项见 `duplicate_images.csv`。

双侧图像内容都完全相同的有效图组重复集合有 4 组。不能把单侧重复直接等同于整个日夜图组重复，本次只记录，不自动删除。同一 Day 内容还出现在 `3_1` 与 `3_2` 两个文件名前缀下，因此仅按此前缀划分也不能保证图像内容隔离。

人工查看了 7 张联系表，覆盖 79 个不同的同名图组（各前缀首/中/末及亮度/相关性极值）。抽查中道路和建筑场景能对应，存在视点差异；未逐一人工确认全部 9,538 组，也未验证像素级配准。低日夜亮度相关性在这些抽样中与昼夜照明分布变化相符，不能单独据此判为错配。

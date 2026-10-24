# DarkDriving_Base：清理后直接使用

当前目录为 `data/DarkDriving_Base/`，保留 **9,520 对**同名 Day/Night 图像，双方均为 2448×2048 RGB JPEG。共 19,040 张，保留图像没有重新编码、改名或改变尺寸。

原始 9,538 对中，移出了 8 个含损坏 JPEG 的图组（连同正常的另一侧），以及 10 个完整日夜内容均重复的冗余图组。每个重复集合按文件名字典序保留首个图组。只有一侧相同、另一侧不同的样本继续保留；重复关系写在 `content_group` 字段中。

## 文件与清单

```text
DarkDriving_Base/
  Day/                  # 9,520 张白天图
  Night/                # 9,520 张夜间图，与 Day 完全同名
  pairs.csv             # 每行一个图组，路径相对本目录
  manifest.jsonl        # 同一清单的 JSONL 版本，含字节/解码像素 SHA-256
  dataset_info.json     # 版本、处理数量、来源审计指纹与备份位置
  validation.json       # 最近一次全量严格解码和校验结果
  README.md             # 本说明的副本
```

`sample_id` 是不含扩展名的原文件名；`sequence_id` 是去除末尾时间戳后的文件名前缀。`day_path` / `night_path` 均为相对路径，可以整体移动数据集目录。`content_group` 将任一侧图像内容完全相同的样本连接在一起，便于划分时避免完全相同的图像跨集合。

## PyTorch 批量加载

在 WeatherVPR 项目根目录运行，使用已安装 PyTorch、NumPy 和 Pillow 的 Python 环境：

```bash
PYTHONPATH=src python - <<'PY'
from torch.utils.data import DataLoader
from weather_vpr.data.darkdriving import DarkDrivingPairs

pairs = DarkDrivingPairs('data/DarkDriving_Base', image_size=(224, 224))
loader = DataLoader(pairs, batch_size=8, shuffle=False, num_workers=2)
batch = next(iter(loader))
print(len(pairs))           # 9520
print(batch['day'].shape)   # torch.Size([8, 3, 224, 224])
print(batch['night'].shape) # torch.Size([8, 3, 224, 224])
print(batch['sample_id'])
PY
```

默认输出为 RGB、float32、`[0, 1]` 范围的张量；尺寸参数为 `(高度, 宽度)`，两侧使用相同的双三次缩放。模型所需的均值/标准差归一化由调用方补上。无需载入 SALAD、DINOv2 或下载模型。

如需原始分辨率 PIL 图像：

```python
pairs = DarkDrivingPairs('data/DarkDriving_Base', image_size=None)
sample = pairs[0]
day_image, night_image = sample['day'], sample['night']
```

如需自己的预处理，可传入 `transform(day_image, night_image)`，返回处理后的 `(day, night)`；该函数会覆盖默认缩放/张量转换，可在函数内对两侧共享随机几何变换参数。PIL 模式的批量加载需要自行转换为张量或提供 `collate_fn`。

## 不使用项目加载器

直接从 CSV 读取即可，不依赖原机器上的绝对路径：

```python
import csv
from pathlib import Path
from PIL import Image

root = Path('data/DarkDriving_Base')
with (root / 'pairs.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        with Image.open(root / row['day_path']) as image:
            day = image.convert('RGB')
        with Image.open(root / row['night_path']) as image:
            night = image.convert('RGB')
        # 使用 day、night；不要把整个数据集一次性读进内存。
```

## 完整性复查

```bash
python scripts/prepare_darkdriving.py --root data/DarkDriving_Base --check-only
```

该命令核对两侧目录与清单完全一致、CSV 与 JSONL 一致、清单指纹及每张图像的文件/解码像素 SHA-256，并逐张做全分辨率严格解码。失败时退出码非零；只有全部通过才更新 `validation.json`。加载器本身在初始化时检查目录和配对结构，在读取样本时严格解码。

## 备份与用途

所有移出的 **36 张文件（18 个图组）**保存在项目的 `outputs/darkdriving_base_cleanup/20261024/quarantine/`，`moves.json` 记录原路径、备份路径、校验值及移出原因。原始压缩包未修改。上一轮审计保存在 `outputs/darkdriving_base_audit/`，其中清单描述清理前状态；实际使用应读取当前数据目录中的 `pairs.csv` / `manifest.jsonl`。

这是未划分的真实日夜图组数据。文件名关系沿用本地原包；清理仅验证可读性和内容重复，未新增地理位置真值或像素级配准真值。`sequence_id` 和 `content_group` 均不是地点标签，连续帧和重复路线仍可能十分相近。若用于训练/评估，应按实际地点或路线制定划分，不能仅随机分帧。

当前项目默认训练仍使用已有 Qwen 合成夜图清单；此真实夜间数据没有自动加入训练，也没有被写入虚构 `place_id` 的检索评估 CSV。

可选 [真实配对阶段一](NIGHT_REAL_PAIRS.md) 已提供来源绑定的审核模板和独立全局蒸馏分支。只有明确审核、分组并分配为 train 的接受项才会进入真实辅助训练；本页的通用加载器不会自行建立这些标注。

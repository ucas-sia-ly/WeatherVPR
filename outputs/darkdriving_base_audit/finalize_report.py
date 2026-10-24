from pathlib import Path
import csv, hashlib, io, json, lzma, struct, zipfile, zlib
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
out=Path('outputs/darkdriving_base_audit').resolve(); root=Path('data/DarkDriving_Base').resolve()
s=json.loads((out/'summary.json').read_text()); archive=json.loads((out/'archive_inventory.json').read_text())
rows=[json.loads(line) for line in (out/'files.jsonl').read_text().splitlines()]
comparisons=[]
with zipfile.ZipFile(archive['archive']) as z, open(archive['archive'],'rb') as f:
 for row in s['errors']:
  info=z.getinfo(row['path']); f.seek(info.header_offset); h=f.read(30); nlen,xlen=struct.unpack_from('<HH',h,26); f.seek(nlen+xlen,1); compressed=f.read(info.compress_size)
  assert info.compress_type==95
  raw=lzma.decompress(compressed)
  assert len(raw)==info.file_size and zlib.crc32(raw)==info.CRC
  result={'path':row['path'],'archive_crc_verified':True,'archive_sha256':hashlib.sha256(raw).hexdigest(),'local_sha256':row['sha256']}
  result['same_archive_and_local_bytes']=result['archive_sha256']==result['local_sha256']
  try:
   with Image.open(io.BytesIO(raw)) as im: im.load()
   result['archive_decode_error']=None
  except Exception as exc: result['archive_decode_error']=f'{type(exc).__name__}: {exc}'
  comparisons.append(result)
archive['confirmed_corrupt_source_comparisons']=comparisons
(out/'archive_inventory.json').write_text(json.dumps(archive,indent=2,ensure_ascii=False)+'\n')
by_split={k:{r['name']:r for r in rows if r['split']==k} for k in ('Day','Night')}
invalid_names={r['name'] for r in s['errors']}
valid=[]; excluded=[]
for name in sorted(set(by_split['Day'])&set(by_split['Night'])):
 d,n=by_split['Day'][name],by_split['Night'][name]
 item={'filename':name,'day_path':str(root/'Day'/name),'night_path':str(root/'Night'/name),'day_sha256':d['sha256'],'night_sha256':n['sha256']}
 if name in invalid_names:
  excluded.append({**item,'day_error':'; '.join(d['errors']),'night_error':'; '.join(n['errors'])})
 else: valid.append(item)
for name,items,fields in [('valid_pairs.csv',valid,['filename','day_path','night_path','day_sha256','night_sha256']),('excluded_pairs.csv',excluded,['filename','day_path','night_path','day_sha256','night_sha256','day_error','night_error'])]:
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(items)
s['checked_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
s['usable_filename_pairs']=len(valid); s['excluded_filename_pairs']=len(excluded)
s['archive_verification']={'archive':archive['archive'],'file_count':archive['archive_file_count'],'missing_local':len(archive['missing_local']),'extra_local':len(archive['extra_local']),'size_mismatches':len(archive['size_mismatches']),'crc_checked_files':archive['crc_comparison']['checked_files'],'crc_mismatches':len(archive['crc_comparison']['mismatches']),'corrupt_members_sha256_match':all(r['same_archive_and_local_bytes'] and r['archive_decode_error'] for r in comparisons)}
s['limitations']=['Filename matching is structural coverage, not verified place IDs or pixel correspondences.','Archive comparison verifies this local archive, not a publisher checksum or official completeness.','Photometric statistics use thumbnails; no per-image human review.','Original images were not modified, deleted, re-encoded, or permissively decoded for valid-pair acceptance.']
(out/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
lines=[f'# DarkDriving_Base 数据集检查报告', '',f'检查时间：{s["checked_at"]}（Asia/Shanghai）', '',f'结论：当前数据集含 {len(s["errors"])} 张无法严格解码的 JPEG，不能认定全部正常。排除涉及的 {len(excluded)} 个同名日夜图组后，保留 {len(valid):,} 组双侧可读图像。原始数据未修改。','', '| 项目 | Day | Night |','| --- | ---: | ---: |']
for label,key in [('文件数','count'),('严格解码通过','valid_count')]:
 lines.append(f'| {label} | {s["splits"]["Day"][key]:,} | {s["splits"]["Night"][key]:,} |')
lines += ['',f'- 总计 {s["total_files"]:,} 个文件，{s["total_bytes"]:,} 字节（{s["total_bytes"]/2**30:.2f} GiB）。',f'- 同名覆盖 {s["filename_coverage"]["common_names"]:,} 组，Day 单侧 {len(s["filename_coverage"]["day_only"])}，Night 单侧 {len(s["filename_coverage"]["night_only"])}。',f'- 分辨率：Day {s["splits"]["Day"]["dimensions"]}；Night {s["splits"]["Night"]["dimensions"]}。',f'- 图像格式：Day {s["splits"]["Day"]["formats"]} / {s["splits"]["Day"]["modes"]}；Night {s["splits"]["Night"]["formats"]} / {s["splits"]["Night"]["modes"]}。',f'- 文件字节重复组 {len(s["duplicates"]["file_bytes"])}；可解码 RGB 像素重复组 {len(s["duplicates"]["decoded_rgb"])}。',f'- 解码警告 {len(s["warnings"])}；极端黑白/近常量图候选：Day {len(s["splits"]["Day"]["extreme_images"])}，Night {len(s["splits"]["Night"]["extreme_images"])}。','', '## 损坏文件','', '| 路径 | 严格解码错误 |','| --- | --- |']
for r in s['errors']: lines.append(f'| `{r["path"]}` | {"; ".join(r["errors"])} |')
lines += ['', '## 原始压缩包核验','',f'本地压缩包：`{archive["archive"]}`。',f'中央目录中共有 {archive["archive_file_count"]:,} 个图像文件，解压目录无缺失、无额外文件、无大小差异。全部 {archive["crc_comparison"]["checked_files"]:,} 个本地文件的 CRC32 与压缩包记录一致，差异 {len(archive["crc_comparison"]["mismatches"])}。',f'另外逐个解压了上述 {len(comparisons)} 个损坏成员，验证解压内容 CRC32，并与本地图像比较 SHA-256；它们均字节相同且原包成员同样无法严格解码。因此重新解压该压缩包无法修复这些图像，需要另一个可靠来源，或在使用时排除。压缩包一致性不等于已核验发布者官方校验和。','', '## 可用清单与使用边界','',f'- `valid_pairs.csv`：{len(valid):,} 个同名图组；两侧均完整解码通过，含绝对路径和 SHA-256。',f'- `excluded_pairs.csv`：{len(excluded)} 个排除图组，包含失败侧及原因。', '- `files.jsonl`：逐文件检查、SHA-256、尺寸、模式和缩略图亮度统计。','- `pairs.csv`：全部同名图组的亮度、相关性和尺寸诊断；未通过解码者没有亮度指标。','- `archive_inventory.json`：原包文件清单、CRC32 和损坏成员核验记录。','- `sequences.json`：23 个文件名前缀组的数量与时间戳跨度；`in_5` 仅有 1 组，该数量与本地原包一致，不能据此认定解压丢失。','- `contact_*.jpg`：每个文件名前缀的首/中/末图组，以及亮度和相关性极值候选。','', '同名关系只证明文件结构覆盖；没有将文件名推断为官方 place_id，也没有将其声明为像素级配准真值。现有训练加载器需要带来源与审核字段的 manifest，通用评估加载器需要带 `place_id` 的 CSV；本次输出不虚构这些标注。若后续划分训练/测试集，应考虑连续帧和路线重叠，不能把随机分帧当作地点隔离。','', '检查方法：Pillow `verify()` 加全分辨率 `load()`，关闭容错截断；计算压缩文件和解码 RGB 的 SHA-256；亮度指标采用 128×128 BOX 缩略图、0.299R+0.587G+0.114B。OpenCV 容错预览只用于显示损坏现象，未用于接受坏图。']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'corrupt_images':len(s['errors']),'excluded_pairs':len(excluded),'valid_pairs':len(valid),'errors':[r['path'] for r in s['errors']],'warnings':len(s['warnings']),'duplicate_file_groups':len(s['duplicates']['file_bytes']),'duplicate_pixel_groups':len(s['duplicates']['decoded_rgb']),'extreme_images':{k:v['extreme_images'] for k,v in s['splits'].items()},'night_not_darker_count':len(s['pair_diagnostics']['night_not_darker'])},indent=2))

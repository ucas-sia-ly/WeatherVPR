from __future__ import annotations
import csv, hashlib, io, json, os, re, time, warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile

ROOT = Path('data/DarkDriving_Base').resolve()
OUT = Path('outputs/darkdriving_base_audit').resolve()
ImageFile.LOAD_TRUNCATED_IMAGES = False
START = time.monotonic()

def scan(path):
    row = {'path': str(path.relative_to(ROOT)), 'split': path.parent.name,
           'name': path.name, 'errors': [], 'warnings': []}
    try:
        before = path.stat()
        row.update(bytes=before.st_size, symlink=path.is_symlink())
        raw = path.read_bytes()
        row['sha256'] = hashlib.sha256(raw).hexdigest()
        row['jpeg_soi'] = raw.startswith(b'\xff\xd8')
        row['jpeg_eoi'] = raw.endswith(b'\xff\xd9')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with Image.open(io.BytesIO(raw)) as im:
                row.update(width=im.width, height=im.height, mode=im.mode,
                           format=im.format, exif_orientation=im.getexif().get(274))
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()  # Full-resolution strict decode, not header-only or draft decode.
                rgb = im.convert('RGB')
                row['pixel_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                thumb = np.asarray(rgb.resize((128, 128), Image.Resampling.BOX))
                gray = (thumb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
                row.update(mean_luma=float(gray.mean()), std_luma=float(gray.std()),
                           black_fraction=float(np.mean(np.max(thumb, axis=2) <= 5)),
                           white_fraction=float(np.mean(np.min(thumb, axis=2) >= 250)),
                           thumbnail_sha256=hashlib.sha256(thumb.tobytes()).hexdigest())
                row['_gray'] = gray
                row['_thumb'] = thumb
            row['warnings'] = [str(w.message) for w in caught]
        if not row['jpeg_soi'] or not row['jpeg_eoi']:
            row['warnings'].append('JPEG boundary marker missing')
        after = path.stat()
        row['changed_during_read'] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        if row['changed_during_read']:
            row['errors'].append('File changed during read')
    except Exception as exc:
        row['errors'].append(f'{type(exc).__name__}: {exc}')
    return row

files = sorted(p for p in ROOT.rglob('*') if p.is_file())
non_files = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if not p.is_file() and not p.is_dir()]
rows = []
thumbs = {}
with (OUT/'files.jsonl').open('w') as log, ThreadPoolExecutor(max_workers=8) as pool:
    for i, row in enumerate(pool.map(scan, files), 1):
        gray = row.pop('_gray', None)
        thumb = row.pop('_thumb', None)
        if gray is not None:
            thumbs[(row['split'], row['name'])] = (gray, thumb)
        rows.append(row)
        log.write(json.dumps(row, ensure_ascii=False)+'\n')
        if i % 500 == 0 or i == len(files):
            log.flush()
            print(f'{i}/{len(files)} files; errors={sum(bool(r["errors"]) for r in rows)}; elapsed={time.monotonic()-START:.1f}s', flush=True)

def duplicates(key):
    groups = defaultdict(list)
    for r in rows:
        if key in r:
            groups[r[key]].append(r['path'])
    return [v for v in groups.values() if len(v)>1]

def distribution(values):
    return dict(zip(('min','p01','p05','p25','median','p75','p95','p99','max'),
                    map(float, np.percentile(values, (0,1,5,25,50,75,95,99,100))))) if values else {}

by_split = {s:{r['name']:r for r in rows if r['split']==s} for s in ('Day','Night')}
names = sorted(set(by_split['Day']) & set(by_split['Night']))
pairs = []
for name in names:
    d,n = by_split['Day'][name], by_split['Night'][name]
    pair = {'name':name}
    pair['same_bytes'] = d.get('sha256') == n.get('sha256')
    pair['same_pixels'] = d.get('pixel_sha256') is not None and d.get('pixel_sha256') == n.get('pixel_sha256')
    pair['dimensions_match'] = (d.get('width'),d.get('height')) == (n.get('width'),n.get('height'))
    if ('Day',name) in thumbs and ('Night',name) in thumbs:
        gd,td = thumbs['Day',name]; gn,tn = thumbs['Night',name]
        pair['day_luma'] = d['mean_luma']; pair['night_luma'] = n['mean_luma']
        pair['night_day_luma_ratio'] = n['mean_luma']/max(d['mean_luma'],1e-8)
        pair['luma_correlation'] = float(np.corrcoef(gd.ravel(),gn.ravel())[0,1]) if gd.std()>0 and gn.std()>0 else None
        pair['thumbnail_rgb_mae'] = float(np.abs(td.astype(float)-tn.astype(float)).mean())
    pairs.append(pair)
with (OUT/'pairs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for p in pairs for k in p})); w.writeheader(); w.writerows(pairs)

summary = {'dataset':str(ROOT), 'checked_at':datetime.now().astimezone().isoformat(),
           'method':{'strict_full_resolution_decode':True,'pil_verify':True,'sha256_file_and_decoded_rgb':True,
                     'photometric_statistics':'128x128 BOX thumbnails; sRGB luma 0.299R+0.587G+0.114B, range 0..255',
                     'workers':8},
           'total_files':len(rows),'total_bytes':sum(r.get('bytes',0) for r in rows),
           'errors':[r for r in rows if r['errors']], 'warnings':[r for r in rows if r['warnings']],
           'non_files':non_files, 'unexpected_paths':[r['path'] for r in rows if r['split'] not in ('Day','Night') or not r['name'].endswith('.jpg')],
           'symlinks':[r['path'] for r in rows if r.get('symlink')],
           'splits':{},
           'filename_coverage':{'common_names':len(names),'day_only':sorted(set(by_split['Day'])-set(by_split['Night'])),
                                'night_only':sorted(set(by_split['Night'])-set(by_split['Day']))},
           'duplicates':{'file_bytes':duplicates('sha256'),'decoded_rgb':duplicates('pixel_sha256')},
           'pair_diagnostics':{'same_bytes':[p['name'] for p in pairs if p['same_bytes']],
                               'same_pixels':[p['name'] for p in pairs if p['same_pixels']],
                               'dimension_mismatch':[p['name'] for p in pairs if not p['dimensions_match']],
                               'night_not_darker':[p['name'] for p in pairs if p.get('night_luma',0)>=p.get('day_luma',1)],
                               'night_day_luma_ratio':distribution([p['night_day_luma_ratio'] for p in pairs if 'night_day_luma_ratio' in p]),
                               'luma_correlation':distribution([p['luma_correlation'] for p in pairs if p.get('luma_correlation') is not None])}}
for split in ('Day','Night'):
    group=list(by_split[split].values()); valid=[r for r in group if not r['errors']]
    sequences=Counter(); malformed=[]
    for r in group:
        m=re.fullmatch(r'(.+)_(\d{19})\.jpg',r['name'])
        if m: sequences[m.group(1)]+=1
        else: malformed.append(r['name'])
    summary['splits'][split]={'count':len(group),'valid_count':len(valid),'bytes':sum(r['bytes'] for r in group),
        'dimensions':dict(Counter(f'{r.get("width")}x{r.get("height")}' for r in group)),
        'modes':dict(Counter(r.get('mode') for r in group)), 'formats':dict(Counter(r.get('format') for r in group)),
        'sequences':dict(sorted(sequences.items())),'unparsed_names':malformed,
        'mean_luma':distribution([r['mean_luma'] for r in valid]),
        'std_luma':distribution([r['std_luma'] for r in valid]),
        'extreme_images':[r['path'] for r in valid if r['std_luma']<1 or r['black_fraction']>0.95 or r['white_fraction']>0.95],
        'nonstandard_orientation':[r['path'] for r in valid if r.get('exif_orientation') not in (None,1)]}
summary['duration_seconds']=time.monotonic()-START
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
# Stratified first/middle/last scenes across every filename prefix, plus diagnostic extremes.
by_prefix=defaultdict(list)
for name in names:
    by_prefix[name.rsplit('_',1)[0]].append(name)
selected=[]
for prefix,group in sorted(by_prefix.items()):
    selected += [(f'{prefix}: {label}',group[idx]) for label,idx in [('first',0),('middle',len(group)//2),('last',len(group)-1)]]
valid_pairs=[p for p in pairs if p.get('luma_correlation') is not None]
selected += [('lowest correlation',p['name']) for p in sorted(valid_pairs,key=lambda p:p['luma_correlation'])[:6]]
selected += [('brightest Night',p['name']) for p in sorted(valid_pairs,key=lambda p:p['night_luma'],reverse=True)[:3]]
selected += [('darkest Night',p['name']) for p in sorted(valid_pairs,key=lambda p:p['night_luma'])[:3]]
from PIL import ImageDraw
for page, start in enumerate(range(0,len(selected),12),1):
    chunk=selected[start:start+12]
    sheet=Image.new('RGB',(1000, len(chunk)*180),(28,28,28)); draw=ImageDraw.Draw(sheet)
    for j,(label,name) in enumerate(chunk):
        draw.text((8,j*180+4),f'{label} | {name}',fill='white')
        for k,split in enumerate(('Day','Night')):
            if (split,name) not in thumbs: continue
            # Open original only for presentation; audit decode above already covers all files.
            with Image.open(ROOT/split/name) as im:
                im.thumbnail((460,150))
                sheet.paste(im,(8+k*500,j*180+25))
            draw.text((400+k*500,j*180+4),split,fill='white')
    sheet.save(OUT/f'contact_{page:02d}.jpg',quality=88)
print(json.dumps({k:summary[k] for k in ('total_files','total_bytes','splits','filename_coverage','pair_diagnostics','duration_seconds')},ensure_ascii=False,indent=2),flush=True)
print('duplicate byte groups:',len(summary['duplicates']['file_bytes']),'duplicate pixel groups:',len(summary['duplicates']['decoded_rgb']),flush=True)

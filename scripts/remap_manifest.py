#!/usr/bin/env python3
"""EXPLICIT old->new path prefix rewrite; never infer a paired image."""
from pathlib import Path
import argparse, json
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-prefix',required=True)
    p.add_argument('--new-prefix',required=True)
    a=p.parse_args()
    if a.input.resolve()==a.output.resolve(): p.error('Do not overwrite source manifest')
    keys=('source_path','output_path','edit_mask_path','occlusion_mask_path','destruction_mask_path')
    transformed=0; rows=[]
    for line in a.input.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        r=json.loads(line)
        for key in keys:
            value=r.get(key)
            if not value:continue
            if value.startswith(a.old_prefix):
                r[key]=a.new_prefix+value[len(a.old_prefix):]
                if not Path(r[key]).is_file():
                    raise FileNotFoundError(f'Path after rewrite missing: {r[key]}')
                transformed+=1
        rows.append(r)
    if not transformed: raise ValueError('No paths matched --old-prefix; nothing saved')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'Wrote {len(rows)} records with {transformed} explicit path rewrites: {a.output}')
if __name__=='__main__':main()

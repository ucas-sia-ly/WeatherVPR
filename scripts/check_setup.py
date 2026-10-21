#!/usr/bin/env python3
"""Only verify standalone project directories; no model loading/network."""
from pathlib import Path
import argparse, os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from weather_vpr.utils.backend import project_root

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=project_root()/'weights/dino_salad.ckpt')
    p.add_argument('--backbone-repo', type=Path, default=project_root()/'third_party/dinov2')
    p.add_argument('--manifest', type=Path, default=project_root()/'data/train/manifest.jsonl')
    p.add_argument('--with-train-data', action='store_true', help='Require train manifest')
    a = p.parse_args()
    root = project_root()
    print('Standalone project:', root)
    print('No outer WORKSPACE directory is needed.')
    backend = Path(os.environ.get('WEATHERVPR_SALAD_SOURCE', root/'third_party/salad'))
    checks = [
        ('SALAD backend',backend/'workflow/model.py'),
        ('Official DINOv2 source',a.backbone_repo/'hubconf.py'),
        ('Full SALAD checkpoint',a.checkpoint),
    ]
    if a.with_train_data: checks.append(('Paired training manifest',a.manifest))
    ok = True
    for label, path in checks:
        exists = path.is_file()
        ok &= exists
        print(('OK     ' if exists else 'MISSING'), label, ':', path)
    if not a.manifest.is_file() and not a.with_train_data:
        print('NOTE: data/train/manifest.jsonl is not present.')
    raise SystemExit(0 if ok else 2)
if __name__ == '__main__': main()

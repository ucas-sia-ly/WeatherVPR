#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/train/manifest.jsonl}" \
  --weather sun --patch-weight 0 --keep-weight 0 \
  --max-steps 2 --batch-size 2 \
  --output-dir "${OUTPUT:-runs/sun_global_smoke}"

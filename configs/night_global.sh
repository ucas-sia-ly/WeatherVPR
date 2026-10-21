#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-python}" scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/splits/night_train.jsonl}" \
  --weather night --patch-weight 0 --keep-weight 0 \
  --max-steps 2 --batch-size 2 \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_global_smoke}"

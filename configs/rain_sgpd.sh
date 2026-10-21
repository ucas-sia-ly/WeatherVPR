#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CALIBRATION:?Set CALIBRATION to the teacher-matched Stage2B calibration JSON}"
: "${EXCLUDE_PLACES:?Set EXCLUDE_PLACES to train/val/test held-out places JSON}"
python scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/train/manifest.jsonl}" \
  --pair-calibration "$CALIBRATION" --exclude-places "$EXCLUDE_PLACES" \
  --weather rain --verify-hashes --bf16 \
  --importance-mode saliency --patch-weight 0.1 --keep-weight 0 \
  --global-weight 1 --contrast-weight 1 --batch-size 4 \
  --output-dir "${OUTPUT:-runs/rain_sgpd_01}"

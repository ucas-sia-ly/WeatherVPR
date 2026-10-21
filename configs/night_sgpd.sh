#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRA_ARGS=()
if [[ "${BF16:-1}" == 1 ]]; then EXTRA_ARGS+=(--bf16); fi
if [[ "${VERIFY_HASHES:-1}" == 1 ]]; then EXTRA_ARGS+=(--verify-hashes); fi
"${PYTHON:-python}" scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/splits/night_train.jsonl}" \
  --pair-calibration "${CALIBRATION:-configs/dino_salad_stage2b_calibration.json}" \
  --exclude-places "${EXCLUDE_PLACES:-data/splits/heldout_places.json}" \
  --weather night "${EXTRA_ARGS[@]}" \
  --importance-mode "${IMPORTANCE_MODE:-saliency}" \
  --patch-weight "${PATCH_WEIGHT:-0.1}" --keep-weight "${KEEP_WEIGHT:-0}" \
  --global-weight "${GLOBAL_WEIGHT:-1}" --contrast-weight "${CONTRAST_WEIGHT:-1}" \
  --batch-size "${BATCH_SIZE:-4}" --epochs "${EPOCHS:-3}" \
  --max-steps "${MAX_STEPS:-0}" --seed "${SEED:-42}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_sgpd_01}"

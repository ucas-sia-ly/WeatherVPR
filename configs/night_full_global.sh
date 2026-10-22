#!/usr/bin/env bash
# Stage 1: initialize an independent Night DINOv2 from the trained Day backbone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRA_ARGS=()
if [[ "${BF16:-1}" == 1 ]]; then EXTRA_ARGS+=(--bf16); fi
if [[ "${VERIFY_HASHES:-1}" == 1 ]]; then EXTRA_ARGS+=(--verify-hashes); fi
if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then EXTRA_ARGS+=(--init-adapter-dir "$INIT_ADAPTER_DIR"); fi
"${PYTHON:-python}" scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/splits/night_train.jsonl}" \
  --exclude-places "${EXCLUDE_PLACES:-data/splits/heldout_places.json}" \
  --weather night "${EXTRA_ARGS[@]}" \
  --adapter-type full \
  --global-weight "${GLOBAL_WEIGHT:-1}" --contrast-weight "${CONTRAST_WEIGHT:-1}" \
  --patch-weight 0 --keep-weight 0 --ot-weight 0 --magnitude-weight 0 \
  --batch-sampling "${BATCH_SAMPLING:-distinct}" \
  --contrast-queue-size "${CONTRAST_QUEUE_SIZE:-256}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --epochs "${EPOCHS:-1}" --learning-rate "${LEARNING_RATE:-0.000001}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --max-steps "${MAX_STEPS:-0}" --seed "${SEED:-42}" \
  --save-every-steps "${SAVE_EVERY_STEPS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_full_global_01}"

#!/usr/bin/env bash
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
  --adapter-type "${ADAPTER_TYPE:-lora}" \
  --lora-rank "${LORA_RANK:-8}" --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0}" --lora-blocks "${LORA_BLOCKS:-4}" \
  --hidden "${HIDDEN:-64}" \
  --global-weight "${GLOBAL_WEIGHT:-1}" --contrast-weight "${CONTRAST_WEIGHT:-1}" \
  --patch-weight 0 --keep-weight 0 --ot-weight 0 --magnitude-weight 0 \
  --batch-sampling "${BATCH_SAMPLING:-distinct}" \
  --contrast-queue-size "${CONTRAST_QUEUE_SIZE:-256}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --batch-size "${BATCH_SIZE:-4}" --epochs "${EPOCHS:-3}" \
  --learning-rate "${LEARNING_RATE:-0.0001}" \
  --max-steps "${MAX_STEPS:-0}" --seed "${SEED:-42}" \
  --save-every-steps "${SAVE_EVERY_STEPS:-200}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_lora_global_01}"

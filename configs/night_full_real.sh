#!/usr/bin/env bash
# Optional real-night stage. SVOX test image bytes are never used in training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRA_ARGS=()
if [[ -n "${AUGMENTATION_CONFIG:-}" ]]; then EXTRA_ARGS+=(--augmentation-config "$AUGMENTATION_CONFIG"); fi
if [[ "${BF16:-0}" == 1 ]]; then EXTRA_ARGS+=(--bf16); fi
if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then EXTRA_ARGS+=(--init-adapter-dir "$INIT_ADAPTER_DIR"); fi
"${PYTHON:-python}" scripts/train_real_night.py \
  "${EXTRA_ARGS[@]}" \
  --svox-root "${SVOX_ROOT:-data/svox}" \
  --night-augmentation "${NIGHT_AUGMENTATION:-off}" \
  --augmentation-strength "${AUGMENTATION_STRENGTH:-1}" \
  --augmentation-probability "${AUGMENTATION_PROBABILITY:-0.8}" \
  --augmentation-epoch-offset "${AUGMENTATION_EPOCH_OFFSET:-0}" \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --gallery-cache "${GALLERY_CACHE:-runs/cache/svox_train_gallery.safetensors}" \
  --test-buffer-m "${TEST_BUFFER_M:-250}" \
  --negative-radius-m "${NEGATIVE_RADIUS_M:-250}" \
  --validation-fraction "${VALIDATION_FRACTION:-0.2}" \
  --validation-gap-m "${VALIDATION_GAP_M:-50}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --epochs "${EPOCHS:-3}" --max-steps "${MAX_STEPS:-0}" \
  --learning-rate "${LEARNING_RATE:-0.000001}" \
  --layerwise-lr-decay "${LAYERWISE_LR_DECAY:-0.7}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --anchor-weight "${ANCHOR_WEIGHT:-0}" \
  --night-baseline-weight "${NIGHT_BASELINE_WEIGHT:-100}" \
  --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_full_real_01}"

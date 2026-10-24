#!/usr/bin/env bash
# Stage 2: continue Stage 1 with trusted SALAD guided patch distillation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRA_ARGS=()
if [[ -n "${AUGMENTATION_CONFIG:-}" ]]; then EXTRA_ARGS+=(--augmentation-config "$AUGMENTATION_CONFIG"); fi
if [[ "${BF16:-1}" == 1 ]]; then EXTRA_ARGS+=(--bf16); fi
if [[ "${VERIFY_HASHES:-1}" == 1 ]]; then EXTRA_ARGS+=(--verify-hashes); fi
"${PYTHON:-python}" scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/splits/night_train.jsonl}" \
  --pair-calibration "${CALIBRATION:-configs/dino_salad_stage2b_calibration.json}" \
  --exclude-places "${EXCLUDE_PLACES:-data/splits/heldout_places.json}" \
  --weather night "${EXTRA_ARGS[@]}" \
  --adapter-type full \
  --full-train-scope "${FULL_TRAIN_SCOPE:-all}" --full-train-blocks "${FULL_TRAIN_BLOCKS:-4}" \
  --night-augmentation "${NIGHT_AUGMENTATION:-off}" \
  --augmentation-strength "${AUGMENTATION_STRENGTH:-1}" \
  --augmentation-probability "${AUGMENTATION_PROBABILITY:-0.8}" \
  --augmentation-epoch-offset "${AUGMENTATION_EPOCH_OFFSET:-0}" \
  --init-adapter-dir "${INIT_ADAPTER_DIR:-runs/night_full_global_01}" \
  --importance-mode "${IMPORTANCE_MODE:-saliency}" \
  --patch-weight "${PATCH_WEIGHT:-0.1}" --keep-weight "${KEEP_WEIGHT:-0}" \
  --ot-weight "${OT_WEIGHT:-0}" --magnitude-weight "${MAGNITUDE_WEIGHT:-0}" \
  --min-teacher-keep "${MIN_TEACHER_KEEP:-0.01}" \
  --global-weight "${GLOBAL_WEIGHT:-1}" --contrast-weight "${CONTRAST_WEIGHT:-1}" \
  --batch-sampling "${BATCH_SAMPLING:-distinct}" \
  --contrast-queue-size "${CONTRAST_QUEUE_SIZE:-256}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --epochs "${EPOCHS:-1}" --learning-rate "${LEARNING_RATE:-0.000001}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --max-steps "${MAX_STEPS:-0}" --seed "${SEED:-42}" \
  --save-every-steps "${SAVE_EVERY_STEPS:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT:-runs/night_full_sgpd_01}"

#!/usr/bin/env bash
# Train Night queries against a fixed Day gallery, including only simulated
# low-light views of real training Day photos in the optional source branch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRA_ARGS=()
if [[ "${BF16:-1}" == 1 ]]; then EXTRA_ARGS+=(--bf16); fi
if [[ "${VERIFY_HASHES:-1}" == 1 ]]; then EXTRA_ARGS+=(--verify-hashes); fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == 1 ]]; then EXTRA_ARGS+=(--gradient-checkpointing); fi
if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then EXTRA_ARGS+=(--init-adapter-dir "$INIT_ADAPTER_DIR"); fi
if [[ -n "${REAL_PAIRS_ROOT:-}" ]]; then EXTRA_ARGS+=(--real-pairs-root "$REAL_PAIRS_ROOT"); fi
if [[ -n "${REAL_PAIRS_REVIEW:-}" ]]; then EXTRA_ARGS+=(--real-pairs-review "$REAL_PAIRS_REVIEW"); fi
if [[ -n "${AUXILIARY_DAY_CACHE:-}" ]]; then EXTRA_ARGS+=(--auxiliary-day-cache "$AUXILIARY_DAY_CACHE"); fi
if [[ "${COMPACT_BATCHES:-0}" == 1 ]]; then EXTRA_ARGS+=(--compact-batches); fi
"${PYTHON:-python}" scripts/train.py \
  --checkpoint "${CKPT:-weights/dino_salad.ckpt}" \
  --backbone-repo "${DINO_DIR:-third_party/dinov2}" \
  --manifest "${MANIFEST:-data/splits/night_train.jsonl}" \
  --exclude-places "${EXCLUDE_PLACES:-data/splits/heldout_places.json}" \
  --weather night --adapter-type full "${EXTRA_ARGS[@]}" \
  --full-train-scope "${FULL_TRAIN_SCOPE:-late}" --full-train-blocks "${FULL_TRAIN_BLOCKS:-4}" \
  --fixed-day-bank "${DAY_BANK:-runs/cache/qwen_training_day.safetensors}" \
  --day-bank-device "${DAY_BANK_DEVICE:-cpu}" \
  --night-augmentation "${NIGHT_AUGMENTATION:-camera}" \
  --augmentation-config "${AUGMENTATION_CONFIG:-configs/night_camera.json}" \
  --augmentation-strength "${AUGMENTATION_STRENGTH:-1}" \
  --augmentation-probability "${AUGMENTATION_PROBABILITY:-0.8}" \
  --augmentation-epoch-offset "${AUGMENTATION_EPOCH_OFFSET:-0}" \
  --global-weight "${GLOBAL_WEIGHT:-0.1}" --contrast-weight "${CONTRAST_WEIGHT:-0.1}" \
  --contrast-hard-negatives "${CONTRAST_HARD_NEGATIVES:-64}" \
  --relation-weight "${RELATION_WEIGHT:-0.1}" \
  --relation-temperature "${RELATION_TEMPERATURE:-0.1}" \
  --relation-teacher-temperature "${RELATION_TEACHER_TEMPERATURE:-0.1}" \
  --relation-hard-negatives "${RELATION_HARD_NEGATIVES:-64}" \
  --synthetic-guidance-weight "${SYNTHETIC_GUIDANCE_WEIGHT:-0}" \
  --source-night-weight "${SOURCE_NIGHT_WEIGHT:-0.5}" \
  --source-night-samples "${SOURCE_NIGHT_SAMPLES:-2}" \
  --source-night-config "${SOURCE_NIGHT_CONFIG:-configs/source_night.json}" \
  --auxiliary-mode "${AUXILIARY_MODE:-none}" \
  --auxiliary-weight "${AUXILIARY_WEIGHT:-0.1}" \
  --auxiliary-samples "${AUXILIARY_SAMPLES:-2}" \
  --auxiliary-gradient-every "${AUXILIARY_GRADIENT_EVERY:-0}" \
  --auxiliary-cache-batch-size "${AUXILIARY_CACHE_BATCH_SIZE:-0}" \
  --auxiliary-cache-workers "${AUXILIARY_CACHE_WORKERS:-0}" \
  --token-style-probability "${TOKEN_STYLE_PROBABILITY:-0}" \
  --token-style-fraction "${TOKEN_STYLE_FRACTION:-0.3}" \
  --token-style-strength "${TOKEN_STYLE_STRENGTH:-0.2}" \
  --token-style-beta "${TOKEN_STYLE_BETA:-0.1}" \
  --token-style-after-block "${TOKEN_STYLE_AFTER_BLOCK:-7}" \
  --pair-calibration "${CALIBRATION:-configs/dino_salad_stage2b_calibration.json}" \
  --patch-weight "${PATCH_WEIGHT:-0}" --keep-weight "${KEEP_WEIGHT:-0}" \
  --ot-weight "${OT_WEIGHT:-0}" --magnitude-weight "${MAGNITUDE_WEIGHT:-0}" \
  --attention-weight "${ATTENTION_WEIGHT:-0}" --patch-visibility "${PATCH_VISIBILITY:-off}" \
  --importance-mode "${IMPORTANCE_MODE:-saliency}" \
  --min-teacher-keep "${MIN_TEACHER_KEEP:-0.01}" \
  --batch-sampling distinct --batch-size "${BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --epochs "${EPOCHS:-1}" --learning-rate "${LEARNING_RATE:-0.000001}" \
  --layer-decay "${LAYER_DECAY:-1}" --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --lr-schedule "${LR_SCHEDULE:-cosine}" --warmup-steps "${WARMUP_STEPS:-32}" \
  --min-lr-ratio "${MIN_LR_RATIO:-0.1}" \
  --max-steps "${MAX_STEPS:-0}" --seed "${SEED:-42}" \
  --save-every-steps "${SAVE_EVERY_STEPS:-0}" \
  --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-cuda}" \
  --prefetch-factor "${PREFETCH_FACTOR:-2}" \
  --output-dir "${OUTPUT:-runs/night_late4_02}"

#!/usr/bin/env bash
# Stage-one A/B/C/D: preserve the main GSV stream, add only paired cosine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then
  echo 'Stage one starts from the original Day checkpoint; unset INIT_ADAPTER_DIR.' >&2
  exit 2
fi
ARM="${STAGE1_ARM:-B}"
case "$ARM" in
  A) export AUXILIARY_MODE=none AUXILIARY_WEIGHT=0 AUXILIARY_GRADIENT_EVERY=0 ;;
  B) export AUXILIARY_MODE=real AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.1}" ;;
  C) export AUXILIARY_MODE=synthetic AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.1}" ;;
  D) export AUXILIARY_MODE=real AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.03}" ;;
  *) echo 'STAGE1_ARM must be A, B, C or D.' >&2; exit 2 ;;
esac
if [[ "$AUXILIARY_MODE" == real ]]; then
  : "${REAL_PAIRS_REVIEW:?Set REAL_PAIRS_REVIEW to the completed source-bound review JSONL.}"
  export REAL_PAIRS_ROOT="${REAL_PAIRS_ROOT:-data/DarkDriving_Base}"
  export AUXILIARY_DAY_CACHE="${AUXILIARY_DAY_CACHE:-runs/cache/stage1_real_day.safetensors}"
else
  # C always shares the existing fixed GSV bank. A does not touch real data.
  unset AUXILIARY_DAY_CACHE
fi
export FULL_TRAIN_SCOPE=late FULL_TRAIN_BLOCKS=4
export PATCH_WEIGHT=0 KEEP_WEIGHT=0 OT_WEIGHT=0 MAGNITUDE_WEIGHT=0 ATTENTION_WEIGHT=0
export SYNTHETIC_GUIDANCE_WEIGHT=0
export AUXILIARY_SAMPLES="${AUXILIARY_SAMPLES:-2}"
export AUXILIARY_GRADIENT_EVERY="${AUXILIARY_GRADIENT_EVERY:-100}"
export OUTPUT="${OUTPUT:-runs/night_real_stage1_${ARM}_seed${SEED:-42}}"
exec bash configs/night_specialist.sh

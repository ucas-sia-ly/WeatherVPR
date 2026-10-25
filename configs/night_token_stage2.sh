#!/usr/bin/env bash
# Stage two: a fixed 2x2 test of reviewed real pairs and partial-token style.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARM="${STAGE2_ARM:-T3}"
case "$ARM" in
  T0) export STAGE1_ARM=A TOKEN_STYLE_PROBABILITY=0 ;;
  T1) export STAGE1_ARM=B TOKEN_STYLE_PROBABILITY=0 ;;
  T2) export STAGE1_ARM=A TOKEN_STYLE_PROBABILITY=0.5 ;;
  T3) export STAGE1_ARM=B TOKEN_STYLE_PROBABILITY=0.5 ;;
  *) echo 'STAGE2_ARM must be T0, T1, T2 or T3.' >&2; exit 2 ;;
esac
export TOKEN_STYLE_AFTER_BLOCK=7 TOKEN_STYLE_FRACTION=0.3
export TOKEN_STYLE_STRENGTH=0.2 TOKEN_STYLE_BETA=0.1
export BATCH_SIZE="${BATCH_SIZE:-4}" GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SOURCE_NIGHT_SAMPLES="${SOURCE_NIGHT_SAMPLES:-1}" AUXILIARY_SAMPLES="${AUXILIARY_SAMPLES:-1}"
export EPOCHS="${EPOCHS:-1}" SEED="${SEED:-42}" BF16=1
export LEARNING_RATE="${LEARNING_RATE:-0.000001}"
export OUTPUT="${OUTPUT:-runs/night_token_stage2_${ARM}_seed${SEED}}"
exec bash configs/night_real_stage1_fast.sh

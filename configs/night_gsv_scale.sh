#!/usr/bin/env bash
# Fixed-budget G1/G2 x A/B coverage experiment. Qwen inputs/keys stay fixed;
# only the independent CSV-verified GSV Source pool changes with GSV_SCALE.
# Required: GSV_G1_MANIFEST, GSV_G2_MANIFEST (signed sidecars), and for B
# REAL_PAIRS_REVIEW. Scale A/B uses an identical Source manifest/cache.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "${GSV_SCALE:-G1}" in
  G1)
    : "${GSV_G1_MANIFEST:?Set GSV_G1_MANIFEST to the signed current-place Day manifest.}"
    export SOURCE_DAY_MANIFEST="$GSV_G1_MANIFEST"
    export SOURCE_DAY_CACHE="${GSV_G1_CACHE:-runs/cache/gsv_scale_G1_source_day.safetensors}"
    ;;
  G2)
    : "${GSV_G2_MANIFEST:?Set GSV_G2_MANIFEST to the signed expanded-place Day manifest.}"
    export SOURCE_DAY_MANIFEST="$GSV_G2_MANIFEST"
    export SOURCE_DAY_CACHE="${GSV_G2_CACHE:-runs/cache/gsv_scale_G2_source_day.safetensors}"
    ;;
  *) echo 'GSV_SCALE must be G1 or G2.' >&2; exit 2 ;;
esac
case "${GSV_SCALE_ARM:-A}" in
  A|B) export STAGE1_ARM="${GSV_SCALE_ARM:-A}" ;;
  *) echo 'GSV_SCALE_ARM must be A (no real) or B (paired real cosine).' >&2; exit 2 ;;
esac
# Keep these identical in all four cells. Larger microbatches change contrast
# semantics; use an explicitly separate protocol for such throughput trials.
export BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=2 SOURCE_NIGHT_SAMPLES=1
export SOURCE_NIGHT_WEIGHT=0.5 AUXILIARY_SAMPLES=1
export EPOCHS="${EPOCHS:-1}" LEARNING_RATE="${LEARNING_RATE:-0.000001}" WARMUP_STEPS="${WARMUP_STEPS:-32}"
export AUXILIARY_WEIGHT=0.1 TOKEN_STYLE_PROBABILITY=0
export DAY_BANK="${DAY_BANK:-runs/cache/qwen_training_day.safetensors}"
export SOURCE_DAY_BANK_DEVICE="${SOURCE_DAY_BANK_DEVICE:-cuda}"
export SOURCE_DAY_CACHE_BATCH_SIZE="${SOURCE_DAY_CACHE_BATCH_SIZE:-64}"
export SOURCE_DAY_CACHE_WORKERS="${SOURCE_DAY_CACHE_WORKERS:-4}"
export OUTPUT="${OUTPUT:-runs/night_gsv_scale_${GSV_SCALE:-G1}_${STAGE1_ARM}_seed${SEED:-42}}"
exec bash configs/night_real_stage1_fast.sh

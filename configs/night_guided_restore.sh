#!/usr/bin/env bash
# A second synthetic-only candidate: preserve the original image's descriptor
# while learning camera robustness on both real Day photos and Qwen Night.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GLOBAL_WEIGHT="${GLOBAL_WEIGHT:-0.1}"
export SYNTHETIC_GUIDANCE_WEIGHT="${SYNTHETIC_GUIDANCE_WEIGHT:-1}"
export DAY_PRESERVATION_WEIGHT="${DAY_PRESERVATION_WEIGHT:-0.5}"
export DAY_VIEW_AUGMENTATION="${DAY_VIEW_AUGMENTATION:-camera}"
export OUTPUT="${OUTPUT:-runs/night_guided_restore_01}"
exec bash "$ROOT/configs/night_guided_retrieval.sh"

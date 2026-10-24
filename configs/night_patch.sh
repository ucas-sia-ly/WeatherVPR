#!/usr/bin/env bash
# Last-four-block Night training with every Qwen view camera-degraded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AUGMENTATION_CONFIG="${AUGMENTATION_CONFIG:-configs/night_patch_camera.json}"
export AUGMENTATION_PROBABILITY="${AUGMENTATION_PROBABILITY:-1}"
export PATCH_WEIGHT="${PATCH_WEIGHT:-0.1}"
export ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-0}"
export PATCH_VISIBILITY="${PATCH_VISIBILITY:-photometric}"
export OUTPUT="${OUTPUT:-runs/night_patch_01}"
exec bash "$ROOT/configs/night_specialist.sh"

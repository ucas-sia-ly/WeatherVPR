#!/usr/bin/env bash
# Replay paired Night experiments with a fixed, Student-only camera policy.
# Compare Global and SGPD continuations on clean synthetic validation pairs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
EXPERIMENT="${EXPERIMENT:-night_camera_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="runs/$EXPERIMENT"
for stage in global_01 global_02 sgpd_02; do
  if [[ -e "$RUN_ROOT/$stage" ]]; then
    echo "Experiment stage already exists: $RUN_ROOT/$stage; choose a new EXPERIMENT" >&2
    exit 1
  fi
done
export NIGHT_AUGMENTATION=camera
export AUGMENTATION_CONFIG="${AUGMENTATION_CONFIG:-configs/night_camera.json}"
export AUGMENTATION_STRENGTH="${AUGMENTATION_STRENGTH:-1}"
export AUGMENTATION_PROBABILITY="${AUGMENTATION_PROBABILITY:-0.8}"
export SEED="${SEED:-42}"

OUTPUT="$RUN_ROOT/global_01" INIT_ADAPTER_DIR= AUGMENTATION_EPOCH_OFFSET=0 \
  EPOCHS=1 BF16=1 bash configs/night_full_global.sh

# Both continuations start from the same weights and use the same new draws.
OUTPUT="$RUN_ROOT/global_02" INIT_ADAPTER_DIR="$RUN_ROOT/global_01" \
  AUGMENTATION_EPOCH_OFFSET=1 EPOCHS=1 BF16=1 bash configs/night_full_global.sh
OUTPUT="$RUN_ROOT/sgpd_02" INIT_ADAPTER_DIR="$RUN_ROOT/global_01" \
  AUGMENTATION_EPOCH_OFFSET=1 EPOCHS=1 BF16=1 bash configs/night_full_sgpd.sh

for stage in global_01 global_02 sgpd_02; do
  "$PYTHON" scripts/evaluate_pairs.py --adapter-dir "$RUN_ROOT/$stage" \
    --device "${DEVICE:-cuda}" --batch-size 16 --num-workers "${NUM_WORKERS:-4}" \
    --output "$RUN_ROOT/$stage/synthetic_val.json"
done
echo "Replay complete: $RUN_ROOT. Clean paired validation results are saved in each stage directory."

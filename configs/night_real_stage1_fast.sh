#!/usr/bin/env bash
# Throughput preset for a 48 GiB GPU: same optimizer/data budget, no recompute.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DAY_BANK_DEVICE="${DAY_BANK_DEVICE:-cuda}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export COMPACT_BATCHES=1
export AUXILIARY_CACHE_BATCH_SIZE="${AUXILIARY_CACHE_BATCH_SIZE:-64}"
export AUXILIARY_CACHE_WORKERS="${AUXILIARY_CACHE_WORKERS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec bash configs/night_real_stage1.sh

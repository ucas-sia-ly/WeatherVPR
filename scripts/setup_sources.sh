#!/usr/bin/env bash
# Download third-party *source code* into this standalone project, no outside WORKSPACE needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
if [[ ! -f "$ROOT/third_party/salad/workflow/model.py" ]]; then
  echo '[1/2] Sparse-cloning only SALAD-related files from the claude branch'
  SOURCE_DIR="$(mktemp -d "$ROOT/third_party/.workspace_source.XXXXXX")"
  trap 'rm -rf "$SOURCE_DIR"' EXIT
  git clone --filter=blob:none --depth 1 --sparse --single-branch --branch claude https://github.com/ucas-sia-ly/WORKSPACE.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" sparse-checkout set salad
  mv "$SOURCE_DIR/salad" "$ROOT/third_party/salad"
  rm -rf "$SOURCE_DIR"
  trap - EXIT
else
  echo '[1/2] SALAD backend already present; skipped'
fi
if [[ ! -f "$ROOT/third_party/dinov2/hubconf.py" ]]; then
  echo '[2/2] Cloning official DINOv2 source'
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$ROOT/third_party/dinov2"
else
  echo '[2/2] DINOv2 source already present; skipped'
fi
echo 'Sources ready. Place full weights/dino_salad.ckpt separately.'

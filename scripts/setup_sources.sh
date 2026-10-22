#!/usr/bin/env bash
# Verify interfaces in the bundled source snapshots; never download or replace them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_bundled_salad() {
  local relative
  for relative in workflow/model.py workflow/pair_correspondence.py workflow/reliability.py \
      workflow/evaluation.py workflow/reliability_cache.py \
      workflow/synthetic_losses.py models/aggregators/salad.py \
      configs/splits/ccr_reserved_places.json; do
    [[ -f "$ROOT/third_party/salad/$relative" ]] || return 1
  done
  # The bundled forward preserves gradients through all Night DINOv2 layers.
  # Downloading an upstream SALAD directory would discard this local change.
  grep -Fq 'self.full_finetune = False' "$ROOT/third_party/salad/workflow/model.py" &&
    grep -Fq 'if self.full_finetune:' "$ROOT/third_party/salad/workflow/model.py" &&
    grep -Fq 'nullcontext() if self.full_finetune else torch.no_grad()' \
      "$ROOT/third_party/salad/workflow/model.py" &&
    grep -Fq '"assignment_keep"' "$ROOT/third_party/salad/models/aggregators/salad.py"
}

if ! require_bundled_salad; then
  echo 'Missing or incompatible bundled SALAD backend.' >&2
  echo 'Restore the bundled third_party/salad/workflow/, models/, configs/, LICENSE and README files from this WeatherVPR distribution or its backup, including the local full_finetune forward changes.' >&2
  echo 'Historical SALAD outputs/, datasets and weights are not required runtime sources.' >&2
  echo 'An upstream SALAD or WORKSPACE checkout alone is insufficient. No sources were replaced.' >&2
  exit 1
fi
echo '[1/2] Bundled modified SALAD backend validated; kept in place'

require_bundled_dinov2() {
  local relative
  for relative in hubconf.py dinov2/hub/backbones.py \
      dinov2/models/vision_transformer.py dinov2/layers/attention.py \
      dinov2/layers/block.py dinov2/layers/patch_embed.py; do
    [[ -f "$ROOT/third_party/dinov2/$relative" ]] || return 1
  done
  grep -Fq 'dinov2_vitb14' "$ROOT/third_party/dinov2/hubconf.py" &&
    grep -Fq 'def dinov2_vitb14(' "$ROOT/third_party/dinov2/dinov2/hub/backbones.py" &&
    grep -Fq 'def prepare_tokens_with_masks(' \
      "$ROOT/third_party/dinov2/dinov2/models/vision_transformer.py"
}

if ! require_bundled_dinov2; then
  echo 'Missing or incompatible bundled DINOv2 source snapshot.' >&2
  echo 'Restore the bundled third_party/dinov2/ runtime source files from this WeatherVPR distribution or its backup; historical outputs and model weights are not needed here.' >&2
  echo 'This verifier does not download or replace source files; the bundled snapshot revision has not been independently verified.' >&2
  exit 1
fi
echo '[2/2] Bundled DINOv2 interfaces validated; snapshot revision not verified'
echo 'Bundled source interface checks passed. Place full weights/dino_salad.ckpt separately.'

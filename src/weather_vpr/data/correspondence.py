"""Stage 2B A-only verified patch correspondence with exact teacher checksum."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from weather_vpr.data.paired_dataset import sha256, resolve_file

def load_calibration(path: Path | None, checkpoint: Path, image_size, index=0):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        payload = payload[index]
    elif index != 0:
        raise ValueError('calibration-index only applies to a list')
    from workflow.pair_correspondence import CorrespondenceCalibration
    from workflow.model import input_preprocessing
    calibration = CorrespondenceCalibration(**payload)
    if calibration.search_radius != 1:
        raise ValueError('Starter expects existing Stage2B radius=1 calibration')
    source = calibration.provenance.get('teacher', {})
    expected_sha = sha256(checkpoint)
    if source.get('checkpoint_sha256') != expected_sha:
        raise ValueError('Calibration was computed using a different teacher checkpoint')
    if source.get('image_preprocessing') != input_preprocessing(image_size):
        raise ValueError('Calibration preprocessing differs from current model inputs')
    return calibration


def mask_image(row, field, expected_size, manifest):
    name = row.get(field + '_path')
    if not name:
        return None
    import numpy as np
    path = resolve_file(name, manifest)
    with Image.open(path) as im:
        if im.size != expected_size:
            raise ValueError(f'{field} mask size differs from generated image: {path}')
        return torch.from_numpy(np.array(im.convert('L'), dtype=np.float32).copy() / 255)


@torch.no_grad()
def correspondence(batch, source_aux, generated_aux, calibration, manifest):
    """Produce Stage 2B A-only pair indices [B,N], confidence [B,N]."""
    from workflow.pair_correspondence import build_pair_correspondence
    indices, confidences = [], []
    for i, item in enumerate(batch):
        src = {k: v[i:i+1] for k, v in source_aux.items() if isinstance(v, torch.Tensor)}
        dst = {k: v[i:i+1] for k, v in generated_aux.items() if isinstance(v, torch.Tensor)}
        row = item['row']
        gen = item['weather_image']
        kwargs = {key: mask_image(row, key, gen.size, manifest)
                  for key in ('edit_mask', 'occlusion_mask', 'destruction_mask')}
        result = build_pair_correspondence(
            item['day_image'], gen, src, dst, calibration=calibration,
            search_radius=1, alignment_tolerance=1, quality=row,
            # Never treat B candidate as a verified positive:
            **kwargs)
        indices.append(result['match_index'].flatten())
        a_only = result['positive_conf'] * result['match_diagnostics']['positive_a']
        confidences.append(a_only.flatten())
    return torch.stack(indices).long(), torch.stack(confidences).float()


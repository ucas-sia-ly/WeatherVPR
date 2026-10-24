"""Two-update integration check; discard its temporary Night weights."""
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
with tempfile.TemporaryDirectory(prefix='weather_vpr_late4_check_') as temporary:
    output = Path(temporary) / 'night'
    env = {key: value for key, value in os.environ.items()
           if key not in {'FULL_TRAIN_SCOPE', 'FULL_TRAIN_BLOCKS', 'INIT_ADAPTER_DIR'}}
    env.update(PYTHON=sys.executable, MAX_STEPS='2', WARMUP_STEPS='0',
               OUTPUT=str(output), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', NUM_WORKERS='0')
    subprocess.run(['bash', 'configs/night_specialist.sh'], env=env, check=True)
    metadata = json.loads((output / 'adapter.json').read_text())
    configuration = json.loads((output / 'training_config.json').read_text())
    rows = [json.loads(line) for line in (output / 'training_log.jsonl').read_text().splitlines()]
    scope = metadata['full_train_scope']
    assert scope['scope'] == 'late' and scope['block_indices'] == [8, 9, 10, 11], scope
    assert scope['trainable_parameters'] == 28357632, scope
    assert metadata['optimizer_steps'] == len(rows) == 2
    assert not any('day_preservation' in key or 'day_view' in key for key in configuration)
    assert all('day_preservation_loss' not in row for row in rows)
    assert all(row['source_night_negative_pairs'] > 0 for row in rows)

    # Training subprocess has exited. Load just CPU state, never a GPU model.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from safetensors.torch import load_file
    saved = torch.load(ROOT / 'weights/dino_salad.ckpt', map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved)
    day = {key.removeprefix('backbone.'): value for key, value in state.items()
           if key.startswith('backbone.')}
    del saved, state
    gc.collect()
    night = load_file(str(output / 'adapter.safetensors'), device='cpu')
    assert set(day) == set(night)
    protected = changed = 0
    changed_blocks = set()
    for key, value in night.items():
        selected = any(key.startswith(f'model.blocks.{index}.') for index in (8, 9, 10, 11))
        assert torch.isfinite(value).all(), key
        if selected:
            if not torch.equal(value, day[key]):
                changed += 1
                changed_blocks.add(int(key.split('.')[2]))
        else:
            assert torch.equal(value, day[key]), key
            protected += 1
    assert changed_blocks == {8, 9, 10, 11}
    report = {'status': 'passed', 'optimizer_updates': len(rows), 'training_scope': scope,
              'changed_selected_tensors': changed, 'frozen_tensors_bitwise_equal': protected,
              'source_night_negative_pairs': [row['source_night_negative_pairs'] for row in rows],
              'retired_day_options_absent': True, 'temporary_checkpoint_deleted': True}
    (ROOT / 'outputs/maintenance/late4_validation_20261024.json').write_text(json.dumps(report, indent=2) + '\n')
    print('LATE4 VALIDATION PASSED', json.dumps(report), flush=True)

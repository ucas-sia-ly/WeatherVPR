"""Throughput options preserve image tensors, key values and gradients."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader

from weather_vpr.data.paired_dataset import AcceptedWeatherPairs, compact_training_batch
from weather_vpr.data.night_augmentation import NightAugmentationConfig
from weather_vpr.data.source_night import SourceNightConfig
from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.utils.backend import enable_backend_imports


class ThroughputTests(unittest.TestCase):
    def test_compact_parallel_prefetch_preserves_exact_views_and_main_rng(self):
        enable_backend_imports()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(6):
                day, night = root / f'day{i}.png', root / f'night{i}.png'
                Image.new('RGB', (27, 21), (30+i*11, 120-i*7, 210)).save(day)
                Image.new('RGB', (27, 21), (20+i*5, 30+i*8, 70)).save(night)
                rows.append(dict(sample_id=str(i), city='city', place_id=i, condition='night',
                                 passed=True, eligible_for_training=True,
                                 source_path=str(day), output_path=str(night)))
            manifest = root / 'pairs.jsonl'
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            pairs = AcceptedWeatherPairs(
                manifest, 'night', (14, 14), seed=42,
                night_augmentation=NightAugmentationConfig(profile='camera', probability=1),
                source_night_config=SourceNightConfig())
            pairs.set_epoch(3)
            normal = list(DataLoader(pairs, batch_size=2, num_workers=0, collate_fn=list,
                                     generator=torch.Generator().manual_seed(77)))
            before = torch.random.get_rng_state().clone()
            parallel = list(DataLoader(pairs, batch_size=2, num_workers=2, prefetch_factor=1,
                                       collate_fn=compact_training_batch,
                                       generator=torch.Generator().manual_seed(77)))
            torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
            for old_batch, new_batch in zip(normal, parallel):
                for old, new in zip(old_batch, new_batch):
                    self.assertNotIn('day_image', new)
                    self.assertNotIn('weather_image', new)
                    self.assertNotIn('row', new)
                    for key in ('day_tensor', 'weather_tensor', 'student_tensor', 'source_night_tensor'):
                        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
                    for key in ('index', 'place', 'augmentation_record', 'source_night_record'):
                        self.assertEqual(old[key], new[key])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA bank comparison')
    def test_gpu_resident_keys_match_transferred_keys_loss_and_gradients(self):
        torch.manual_seed(44)
        keys = torch.randn(6, 5)
        places = [('c', str(i)) for i in range(6)]
        queues = [DayDescriptorQueue(6, storage_device=device, include_current_keys=False)
                  for device in ('cpu', 'cuda')]
        for queue in queues:
            queue.enqueue(keys, places)
        query = torch.randn(2, 5, device='cuda')
        targets = keys[[0, 2]].cuda()
        losses, gradients = [], []
        for queue in queues:
            student = query.clone().requires_grad_()
            loss = (queue.contrastive_loss(student, targets, [places[0], places[2]], hard_negatives=3)['loss']
                    + queue.relational_distillation_loss(
                        student, targets, [places[0], places[2]], hard_negatives=3)['loss'])
            loss.backward()
            losses.append(loss.detach())
            gradients.append(student.grad)
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

"""Independent Source integration: target ownership, gradients and main RNG."""
from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from weather_vpr.losses.contrast_queue import DayDescriptorQueue
from weather_vpr.training import source_stream, train


def _features(index):
    return torch.tensor([.4 + index * .13, .9 - index * .07])


def _image(kind, index, features):
    return torch.cat((torch.tensor([float(kind), float(index)]), features))


class _Pairs:
    def __init__(self, manifest, *args, **kwargs):
        self.manifest = manifest
        self.rows = [{'place': ('qwen_city', str(i))} for i in range(6)]
        self.source_config = kwargs.get('source_night_config')

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        features = _features(index)
        main = _image(1, index, features + torch.tensor([.25, -.2]))
        return dict(index=index, day_tensor=_image(0, index, features),
                    weather_tensor=main, student_tensor=main,
                    augmentation_record={'applied': False, 'operations': {}},
                    place=self.rows[index]['place'])


class _SourceDataset:
    """Distinct cities/targets ensure accidental use of Qwen keys is rejected."""
    def __init__(self, args, config, count):
        self.manifest = args.source_day_manifest
        self.source_night_config = config
        self.rows = [dict(place=('source_city', str(i // 2)), sample_id=f'source:{i}')
                     for i in range(count)]

    def __len__(self):
        return len(self.rows)

    def assert_current(self):
        pass

    def sources_signature(self):
        return {'places': [list(row['place']) for row in self.rows]}

    def source_view(self, index, *, seed):
        # Production SourceStream must isolate construction even if a decoder
        # or future augmentation consumes these process-wide RNG streams.
        torch.rand(2)
        random.random()
        np.random.rand(2)
        image = _image(2, index + 20, _features(index + 20) * torch.tensor([.4, .7]))
        return image, {'applied': False, 'operations': {}}


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (224, 224)
        self.config = {'agg_config': {'num_channels': 2}}
        self.weight = nn.Parameter(torch.tensor([[1., .15], [-.1, .85]]))

    def forward(self, images):
        return F.linear(images[:, 2:], self.weight)


class _Net(nn.Module):
    def __init__(self, base, *args, **kwargs):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.adapters = nn.ModuleDict({'night': nn.Linear(2, 2, bias=False)})
        with torch.no_grad():
            self.adapters['night'].weight.copy_(torch.tensor([[.7, -.2], [.1, 1.1]]))
        self.events, self.draws = [], []
        self.initial_base = copy.deepcopy(base.state_dict())

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_parameters(self, weather):
        return self.adapters[weather].parameters()

    @torch.no_grad()
    def teacher(self, images, return_aux=False):
        assert not return_aux
        return self.base(images)

    def student(self, images, weather, return_aux=False):
        assert not return_aux
        kind = int(images[0, 0])
        self.draws.append((kind, torch.rand(1).item(), random.random(), np.random.rand()))
        self.events.append(('forward', kind, images.detach().clone()))
        output = self.adapters[weather](images[:, 2:])

        def backward(gradient):
            self.events.append(('backward', kind))
            if kind == 2:
                # This also tests isolation of checkpoint-like stochastic
                # recomputation during the complete Source backward pass.
                torch.rand(3)
                random.random()
                np.random.rand(3)
            return gradient

        if output.requires_grad:
            output.register_hook(backward)
        return output


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, *args, events, **kwargs):
        super().__init__(*args, **kwargs)
        self.events, self.gradients = events, []

    def step(self, closure=None):
        self.events.append(('step',))
        self.gradients.append([p.grad.detach().clone() for group in self.param_groups
                               for p in group['params'] if p.grad is not None])
        return super().step(closure)


class SourceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'qwen.jsonl'
        self.manifest.write_text('tiny fixture\n')
        self.checkpoint = self.root / 'base.ckpt'
        self.checkpoint.write_bytes(b'tiny frozen model')
        self.counter = 0

    def run_tiny(self, count=8, accumulation=2, enabled=True):
        np.random.seed(31)
        self.counter += 1
        output = self.root / f'run_{self.counter}'
        holder = {}

        def make_net(base, *args, **kwargs):
            holder['net'] = _Net(base)
            return holder['net']

        def make_optimizer(*args, **kwargs):
            holder['optimizer'] = _RecordingSGD(*args, events=holder['net'].events, **kwargs)
            return holder['optimizer']

        def prepare(args, size, source_night_config=None):
            if not enabled:
                return None
            holder['source_dataset'] = _SourceDataset(args, source_night_config, count)
            return holder['source_dataset']

        def main_bank(net, dataset, *args, **kwargs):
            holder['pairs'] = dataset
            targets = net.teacher(torch.stack([dataset[i]['day_tensor'] for i in range(len(dataset))]))
            holder['main_targets'] = targets.clone()
            return targets, [row['place'] for row in dataset.rows], {'cache_status': 'fixture_main'}

        def source_bank(net, dataset, *args, **kwargs):
            # Building frozen targets also consumes RNG, as an implementation
            # might while constructing a DataLoader. The outer builder owns it.
            torch.rand(2)
            random.random()
            np.random.rand(2)
            targets = net.teacher(torch.stack([
                _image(0, i + 20, _features(i + 20)) for i in range(len(dataset))]))
            holder['source_targets'] = targets.clone()
            return targets, [row['place'] for row in dataset.rows], {'cache_status': 'fixture_source'}

        actual_build = source_stream.build_source_stream

        def build(*args, **kwargs):
            result = actual_build(*args, **kwargs)
            holder['stream'] = result
            return result

        arguments = ['--weather', 'night', '--device', 'cpu', '--adapter-type', 'full',
                     '--checkpoint', str(self.checkpoint), '--manifest', str(self.manifest),
                     '--output-dir', str(output), '--batch-size', '2', '--num-workers', '0',
                     '--gradient-accumulation-steps', str(accumulation), '--save-every-steps', '0',
                     '--global-weight', '.1', '--contrast-weight', '.2', '--patch-weight', '0',
                     '--relation-weight', '0', '--contrast-hard-negatives', '0',
                     '--fixed-day-bank', str(self.root / 'qwen.safetensors'),
                     '--learning-rate', '.05', '--weight-decay', '0',
                     '--source-night-weight', '.5' if enabled else '0',
                     '--source-night-samples', '2', '--source-day-manifest', str(self.root / 'source.jsonl'),
                     '--source-day-cache', str(self.root / 'source.safetensors')]
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            for name, value in [('AcceptedWeatherPairs', _Pairs), ('prepare_source_dataset', prepare),
                                ('WeatherAdapterSALAD', make_net), ('build_source_stream', build)]:
                stack.enter_context(patch.object(train, name, side_effect=value))
            stack.enter_context(patch.object(train, 'load_checkpoint_model', side_effect=lambda *a, **k: _Base()))
            stack.enter_context(patch.object(train.torch.optim, 'AdamW', side_effect=make_optimizer))
            stack.enter_context(patch('weather_vpr.training.day_bank.load_or_build_day_bank', side_effect=main_bank))
            stack.enter_context(patch.object(source_stream, 'load_or_build_day_bank', side_effect=source_bank))
            stack.enter_context(patch('weather_vpr.training.optimization.configure_full_backbone_scope',
                                     return_value={'scope': 'late', 'block_indices': [8, 9, 10, 11]}))
            stack.enter_context(patch('weather_vpr.training.optimization.full_backbone_parameter_groups',
                                     side_effect=lambda adapter, **kwargs: adapter.parameters()))
            train.main(arguments)
        holder['metadata'] = json.loads((output / 'adapter.json').read_text())
        holder['logs'] = [json.loads(line) for line in (output / 'training_log.jsonl').read_text().splitlines()]
        return holder

    def test_source_bank_is_separate_and_main_stream_rng_identical_across_scales(self):
        baseline = self.run_tiny(enabled=False)
        expected_draws = [row for row in baseline['net'].draws if row[0] == 1]
        expected_inputs = [row[2] for row in baseline['net'].events if row[:2] == ('forward', 1)]
        for count in (4, 8):
            with self.subTest(source_images=count):
                result = self.run_tiny(count=count)
                self.assertIsNone(result['pairs'].source_config)
                self.assertIn('microbatch_step', result['metadata']['source_night']['seed_policy'])
                self.assertEqual([row for row in result['net'].draws if row[0] == 1], expected_draws)
                inputs = [row[2] for row in result['net'].events if row[:2] == ('forward', 1)]
                for actual, expected in zip(inputs, expected_inputs):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(result['main_targets'], baseline['main_targets'], rtol=0, atol=0)
                self.assertEqual(result['metadata']['main_stream_sha256'], baseline['metadata']['main_stream_sha256'])
                self.assertEqual(result['metadata']['student_views'],
                                 {'qwen': 6, 'source_night': 6, 'real_auxiliary': 0, 'synthetic_auxiliary': 0})
                self.assertEqual(len(result['stream'].queue), count)
                for log in result['logs']:
                    self.assertEqual(log['source_bank_policy'], 'separate_source_only')
                    self.assertTrue(all(place[0] == 'source_city' for place in log['source_day_places']))
                    self.assertEqual(log['source_night_positive_pairs'], 4)
                    self.assertEqual(log['source_night_negative_pairs'], 2 * (count - 2))
                for name, value in result['net'].base.state_dict().items():
                    torch.testing.assert_close(value, result['net'].initial_base[name], rtol=0, atol=0)
                self.assertFalse(any(p.grad is not None for p in result['net'].base.parameters()))

    def test_source_and_main_means_accumulate_into_exact_reference_gradients(self):
        for accumulation in (1, 2):
            with self.subTest(accumulation=accumulation):
                result = self.run_tiny(accumulation=accumulation)
                adapter = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    adapter.weight.copy_(torch.tensor([[.7, -.2], [.1, 1.1]]))
                optimizer = torch.optim.SGD(adapter.parameters(), lr=.05)
                main_queue = DayDescriptorQueue(6, include_current_keys=False)
                main_queue.enqueue(result['main_targets'], [('qwen_city', str(i)) for i in range(6)])
                source_queue = result['stream'].queue
                losses, microbatches, update = [], 0, 0
                for event in result['net'].events:
                    if event[0] == 'forward':
                        _, kind, images = event
                        indices = images[:, 1].to(torch.long)
                        if kind == 1:
                            targets = result['main_targets'][indices]
                            places = [('qwen_city', str(i)) for i in indices.tolist()]
                            queue, weight = main_queue, 1
                            microbatches += 1
                        else:
                            indices -= 20
                            targets = result['source_targets'][indices]
                            places = [result['source_dataset'].rows[i]['place'] for i in indices.tolist()]
                            queue, weight = source_queue, .5
                        descriptor = adapter(images[:, 2:])
                        alignment = (1 - F.cosine_similarity(descriptor, targets)).mean()
                        contrast = queue.contrastive_loss(descriptor, targets, places)['loss']
                        losses.append(weight * (.1 * alignment + .2 * contrast))
                    elif event[0] == 'step':
                        optimizer.zero_grad(set_to_none=True)
                        (sum(losses) / microbatches).backward()
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
                        torch.testing.assert_close(adapter.weight.grad, result['optimizer'].gradients[update][0],
                                                   rtol=3e-5, atol=3e-7)
                        optimizer.step()
                        update += 1
                        losses, microbatches = [], 0
                self.assertEqual(update, 3 if accumulation == 1 else 2)
                torch.testing.assert_close(adapter.weight, result['net'].adapters['night'].weight,
                                           rtol=3e-5, atol=3e-7)


if __name__ == '__main__':
    unittest.main()

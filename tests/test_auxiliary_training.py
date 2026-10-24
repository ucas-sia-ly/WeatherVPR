"""Exercise serial auxiliary optimization on CPU without loading DINO weights."""
from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

from weather_vpr.training import train
from weather_vpr.training.auxiliary import (
    AuxiliaryGradientProbe, auxiliary_rng_context, build_auxiliary_branch,
    paired_global_alignment,
)


def _day_features(index):
    return torch.tensor([0.4 + index * 0.13, 0.9 - index * 0.07])


def _image(kind, index, features):
    return torch.cat((torch.tensor([float(kind), float(index)]), features))


class _TrainingPairs:
    """The training loop sees real Dataset semantics, with vector-valued images."""

    def __init__(self, manifest, weather, image_size, *args, **kwargs):
        self.manifest = manifest
        self.rows = [{'place': ('city', str(index))} for index in range(6)]
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        features = _day_features(index)
        day = _image(0, index, features)
        main = _image(1, index, features + torch.tensor([0.25, -0.2]))
        source = _image(2, index, features * torch.tensor([0.45, 0.75]))
        record = {'applied': False, 'operations': {}}
        return dict(index=index, day_tensor=day, weather_tensor=main,
                    student_tensor=main, source_night_tensor=source,
                    augmentation_record=record, source_night_record=record,
                    place=self.rows[index]['place'])


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (224, 224)
        self.config = {'agg_config': {'num_channels': 2}}
        self.backbone = nn.Linear(2, 2, bias=False)
        self.aggregator = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.tensor([[1.0, 0.15], [-0.1, 0.85]]))
            self.aggregator.weight.copy_(torch.tensor([[1.1, -0.1], [0.2, 0.8]]))

    def forward(self, images):
        return self.aggregator(self.backbone(images[:, 2:]))


class _TinyNet(nn.Module):
    def __init__(self, base, weather_names, *args, **kwargs):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapters = nn.ModuleDict({'night': nn.Linear(2, 2, bias=False)})
        with torch.no_grad():
            self.adapters['night'].weight.copy_(torch.tensor([[0.7, -0.2], [0.1, 1.1]]))
        self.events = []
        self.draws = []
        self.consume_random = True
        self.initial_base = copy.deepcopy(self.base.state_dict())

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
        if self.consume_random:
            self.draws.append((kind, torch.rand(1).item(), random.random()))
        self.events.append(('forward', kind, images.detach().clone()))
        output = self.base.aggregator(self.adapters[weather](images[:, 2:]))
        if output.requires_grad:
            output.register_hook(lambda gradient: self.events.append(('backward', kind)))
        return output


class _RecordingSGD(torch.optim.SGD):
    """Use ordinary updates and retain the exact gradients that reached step()."""

    def __init__(self, *args, events, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events
        self.gradients = []

    def step(self, closure=None):
        self.events.append(('step',))
        self.gradients.append([parameter.grad.detach().clone() for group in self.param_groups
                               for parameter in group['params'] if parameter.grad is not None])
        return super().step(closure)


class _AuxiliaryBranch:
    def __init__(self, args, base):
        self.mode = args.auxiliary_mode
        self.samples = args.auxiliary_samples
        self.base = base
        self.calls = []
        self.target_by_index = {}
        self.weight_by_index = {}

    def rng_context(self, epoch, step):
        # Deliberately make fake data/forward consume the global streams. The
        # production loop must enter this supplied context around the full pass.
        return auxiliary_rng_context('cpu', 700 + epoch * 100 + step)

    def next_batch(self, epoch, step):
        self.calls.append((epoch, step))
        torch.rand(1)
        random.random()
        indices = [100 + step * self.samples + offset for offset in range(self.samples)]
        images, targets = [], []
        for index in indices:
            features = torch.tensor([0.6 + index * .001, 0.4 - index * .001])
            offset = torch.tensor([-.3, .15]) if self.mode == 'real' else torch.tensor([.25, -.2])
            images.append(_image(3, index, features + offset))
            targets.append(self.base(_image(0, index, features)[None])[0].detach())
        targets = torch.stack(targets)
        weights = torch.ones(self.samples)
        # One ignored sample distinguishes an effective-sample mean from /B.
        if step == 1 and self.samples > 1:
            weights[-1] = 0
        for index, target, weight in zip(indices, targets, weights):
            self.target_by_index[index] = target
            self.weight_by_index[index] = weight
        return dict(images=torch.stack(images), targets=targets, weights=weights,
                    sample_ids=[f'{self.mode}:{index}' for index in indices],
                    groups=['scene'] * self.samples, indices=indices,
                    augmentation={'samples': self.samples, 'applied_samples': 0, 'operations': {}})

    def metadata(self):
        return dict(enabled=True, mode=self.mode, sampled_batches=len(self.calls),
                    source='test-only tiny paired descriptors')


class AuxiliaryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'pairs.jsonl'
        self.manifest.write_text('fixture training manifest\n', encoding='utf-8')
        self.checkpoint = self.root / 'base.ckpt'
        self.checkpoint.write_bytes(b'fixture frozen checkpoint')
        self.run_number = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_tiny(self, mode='none', *, accumulation=1, extra=()):
        self.run_number += 1
        output = self.root / f'run_{self.run_number}'
        holder = {}

        def make_net(base, *args, **kwargs):
            net = _TinyNet(base, *args, **kwargs)
            holder['net'] = net
            return net

        def make_optimizer(*args, **kwargs):
            optimizer = _RecordingSGD(*args, events=holder['net'].events, **kwargs)
            holder['optimizer'] = optimizer
            return optimizer

        def build_auxiliary(args, dataset, net, *unused, **kwargs):
            if args.auxiliary_mode == 'none' or args.auxiliary_weight == 0:
                return None
            # The real builder owns this isolation; reproduce its contract.
            with auxiliary_rng_context('cpu', 689):
                torch.rand(3)
                random.random()
                holder['branch'] = _AuxiliaryBranch(args, net.base)
            return holder['branch']

        arguments = ['--weather', 'night', '--device', 'cpu', '--adapter-type', 'full',
                     '--checkpoint', str(self.checkpoint), '--manifest', str(self.manifest),
                     '--output-dir', str(output), '--batch-size', '2', '--num-workers', '0',
                     '--gradient-accumulation-steps', str(accumulation), '--save-every-steps', '0',
                     '--global-weight', '.1', '--contrast-weight', '0', '--patch-weight', '0',
                     '--source-night-weight', '.5', '--source-night-samples', '1',
                     '--learning-rate', '.05', '--weight-decay', '0',
                     '--auxiliary-mode', mode, '--auxiliary-weight', '.1',
                     '--auxiliary-samples', '2']
        if mode == 'real':
            arguments += ['--real-pairs-root', str(self.root),
                          '--real-pairs-review', str(self.manifest)]
        arguments += list(extra)
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(train, 'AcceptedWeatherPairs', _TrainingPairs))
            stack.enter_context(patch.object(train, 'load_checkpoint_model',
                                            side_effect=lambda *a, **k: _Base()))
            stack.enter_context(patch.object(train, 'WeatherAdapterSALAD', side_effect=make_net))
            stack.enter_context(patch.object(train, 'prepare_auxiliary_dataset', return_value=None))
            stack.enter_context(patch.object(train, 'build_auxiliary_branch', side_effect=build_auxiliary))
            stack.enter_context(patch.object(train.torch.optim, 'AdamW', side_effect=make_optimizer))
            stack.enter_context(patch('weather_vpr.training.optimization.configure_full_backbone_scope',
                                     return_value={'scope': 'late', 'block_indices': [8, 9, 10, 11]}))
            stack.enter_context(patch('weather_vpr.training.optimization.full_backbone_parameter_groups',
                                     side_effect=lambda adapter, **kwargs: adapter.parameters()))
            train.main(arguments)
        holder['output'] = output
        holder['metadata'] = json.loads((output / 'adapter.json').read_text())
        holder['log'] = [json.loads(line) for line in (output / 'training_log.jsonl').read_text().splitlines()]
        return holder

    def assert_reference_updates(self, result):
        actual = result['net']
        reference = _TinyNet(_Base(), ('night',))
        reference.consume_random = False
        optimizer = torch.optim.SGD(reference.adapter_parameters('night'), lr=.05, weight_decay=0)
        branch = result.get('branch')
        expected_gradients = []
        losses, main_batches = [], 0

        def finish_window():
            optimizer.zero_grad(set_to_none=True)
            (sum(losses) / main_batches).backward()
            torch.nn.utils.clip_grad_norm_(reference.adapter_parameters('night'), 1.0)
            expected_gradients.append([parameter.grad.clone()
                                       for parameter in reference.adapter_parameters('night')])
            optimizer.step()

        for event in actual.events:
            if event[0] == 'forward':
                _, kind, images = event
                descriptor = reference.student(images, 'night')
                if kind == 3:
                    indices = [int(value) for value in images[:, 1]]
                    targets = torch.stack([branch.target_by_index[index] for index in indices])
                    weights = torch.stack([branch.weight_by_index[index] for index in indices])
                    valid = weights > 0
                    per_sample = 1 - F.cosine_similarity(descriptor[valid], targets[valid])
                    losses.append(.1 * (per_sample * weights[valid]).sum() / weights[valid].sum())
                else:
                    targets = reference.teacher(torch.stack([
                        _image(0, int(index), _day_features(int(index))) for index in images[:, 1]]))
                    loss = (1 - F.cosine_similarity(descriptor, targets)).mean()
                    losses.append(.1 * loss if kind == 1 else .5 * .1 * loss)
                    main_batches += int(kind == 1)
            elif event[0] == 'step':
                finish_window()
                losses, main_batches = [], 0
        self.assertFalse(losses)
        self.assertEqual(len(expected_gradients), len(result['optimizer'].gradients))
        for actual_gradients, expected in zip(result['optimizer'].gradients, expected_gradients):
            for value, target in zip(actual_gradients, expected):
                torch.testing.assert_close(value, target, rtol=2e-5, atol=2e-7)
        for value, target in zip(actual.adapter_parameters('night'), reference.adapter_parameters('night')):
            torch.testing.assert_close(value, target, rtol=2e-5, atol=2e-7)

    def test_serial_branches_keep_independent_means_and_share_each_update(self):
        for mode in ('none', 'real', 'synthetic'):
            for accumulation in (1, 2):
                with self.subTest(mode=mode, accumulation=accumulation):
                    result = self.run_tiny(mode, accumulation=accumulation)
                    self.assert_reference_updates(result)
                    self.assertEqual(result['metadata']['optimizer_steps'], 3 if accumulation == 1 else 2)
                    expected = [1, 2] if mode == 'none' else [1, 2, 3]
                    outstanding = []
                    for event in result['net'].events:
                        if event[0] == 'backward':
                            outstanding.append(event[1])
                        elif event[0] == 'step':
                            self.assertIn(len(outstanding), (len(expected), accumulation * len(expected)))
                            self.assertEqual(outstanding, expected * (len(outstanding) // len(expected)))
                            outstanding = []
                    self.assertFalse(outstanding)

    def test_extra_data_and_forward_preserve_main_random_draws_and_inputs(self):
        baseline = self.run_tiny('none', accumulation=2)
        expected_draws = [value for value in baseline['net'].draws if value[0] != 3]
        expected_inputs = [event[2] for event in baseline['net'].events if event[0] == 'forward']
        for mode in ('real', 'synthetic'):
            with self.subTest(mode=mode):
                result = self.run_tiny(mode, accumulation=2)
                self.assertEqual([value for value in result['net'].draws if value[0] != 3], expected_draws)
                inputs = [event[2] for event in result['net'].events
                          if event[0] == 'forward' and event[1] != 3]
                self.assertEqual(len(inputs), len(expected_inputs))
                for value, target in zip(inputs, expected_inputs):
                    torch.testing.assert_close(value, target, rtol=0, atol=0)

    def test_saves_student_only_and_old_metadata_remains_loadable(self):
        result = self.run_tiny('real', accumulation=2)
        net = result['net']
        self.assertFalse(net.base.training)
        for name, value in net.base.state_dict().items():
            torch.testing.assert_close(value, net.initial_base[name], rtol=0, atol=0)
        self.assertFalse(any(parameter.grad is not None for parameter in net.base.parameters()))
        saved = load_file(str(result['output'] / 'adapter.safetensors'))
        self.assertEqual(set(saved), set(net.adapters['night'].state_dict()))
        for name, value in saved.items():
            torch.testing.assert_close(value, net.adapters['night'].state_dict()[name], rtol=0, atol=0)
        metadata = result['metadata']
        # Older runs have no auxiliary provenance. It must not become part of
        # the model architecture compatibility contract used by the loader.
        metadata = {key: value for key, value in metadata.items() if 'auxiliary' not in key}
        (result['output'] / 'adapter.json').write_text(json.dumps(metadata))
        restored = _TinyNet(_Base(), ('night',))
        args = train.parse_args(['--weather', 'night', '--device', 'cpu',
                                 '--init-adapter-dir', str(result['output'])])
        train._load_initial_adapter(restored, args, metadata['checkpoint_sha256'],
                                    restored.base, metadata['preprocessing'])
        for name, value in restored.adapters['night'].state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


class AuxiliaryPrimitiveTests(unittest.TestCase):
    def test_effective_sample_mean_detaches_teacher_and_excludes_rejected_pairs(self):
        student = torch.tensor([[2., 1.], [-.5, 2.], [1., 3.]], requires_grad=True)
        teacher = torch.tensor([[1., 0.], [1., 1.], [2., -1.]], requires_grad=True)
        weights = torch.tensor([1., 0., 1.], requires_grad=True)
        result = paired_global_alignment(student, teacher, weights)
        expected = (1 - F.cosine_similarity(student[[0, 2]], teacher.detach()[[0, 2]])).mean()
        torch.testing.assert_close(result['loss'], expected)
        self.assertEqual(result['valid_samples'], 2)
        result['loss'].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)
        torch.testing.assert_close(student.grad[1], torch.zeros(2), rtol=0, atol=0)
        self.assertGreater(student.grad[[0, 2]].abs().sum().item(), 0)

    def test_empty_effective_batch_is_a_finite_noop_and_bad_pairs_fail(self):
        student = torch.ones(2, 3, requires_grad=True)
        teacher = torch.ones_like(student)
        result = paired_global_alignment(student, teacher, torch.zeros(2))
        self.assertEqual(result['valid_samples'], 0)
        self.assertEqual(result['loss'].item(), 0)
        result['loss'].backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student), rtol=0, atol=0)
        for weights in (torch.tensor([1., .5]), torch.tensor([1., float('nan')])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                paired_global_alignment(student, teacher, weights)
        with self.assertRaisesRegex(ValueError, 'zero norm'):
            paired_global_alignment(student, torch.zeros_like(student), torch.ones(2))

    def test_disabled_builder_needs_no_files_and_does_not_advance_rng(self):
        before = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        for mode, weight in (('none', .1), ('real', 0), ('synthetic', 0)):
            args = SimpleNamespace(auxiliary_mode=mode, auxiliary_weight=weight)
            self.assertIsNone(build_auxiliary_branch(args, None, None, None, None))
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)

    def test_private_rng_restores_python_numpy_torch_even_on_exception(self):
        before = torch.random.get_rng_state().clone()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with auxiliary_rng_context('cpu', 1234):
                first = (torch.rand(4), random.random(), np.random.rand(4))
                raise RuntimeError('fixture')
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        for expected, value in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(value, expected)
        with auxiliary_rng_context('cpu', 1234):
            repeated = (torch.rand(4), random.random(), np.random.rand(4))
        torch.testing.assert_close(first[0], repeated[0], rtol=0, atol=0)
        self.assertEqual(first[1], repeated[1])
        np.testing.assert_array_equal(first[2], repeated[2])

    def test_gradient_probe_measures_increments_and_preserves_accumulated_gradients(self):
        parameter = nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([2., -1.])
        probe = AuxiliaryGradientProbe([parameter], accumulation_steps=2)
        parameter.grad.add_(torch.tensor([3., 4.]) / 2)
        probe.after_main()
        parameter.grad.add_(torch.tensor([-1., 2.]) / 2)
        before = parameter.grad.clone()
        result = probe.after_auxiliary()
        self.assertAlmostEqual(result['main_norm'], 5)
        self.assertAlmostEqual(result['auxiliary_norm'], 5 ** .5)
        self.assertAlmostEqual(result['norm_ratio'], 5 ** .5 / 5)
        self.assertAlmostEqual(result['cosine'], 1 / 5 ** .5)
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)

        unused = nn.Parameter(torch.zeros(2))
        probe = AuxiliaryGradientProbe([unused])
        probe.after_main()
        result = probe.after_auxiliary()
        self.assertIsNone(result['cosine'])
        self.assertIsNone(result['norm_ratio'])
        self.assertIsNone(unused.grad)


if __name__ == '__main__':
    unittest.main()

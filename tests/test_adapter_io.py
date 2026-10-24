"""CLI defaults and metadata keep earlier LoRA/Output runs loadable."""
import unittest

from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.training.train import parse_args, _validate


class AdapterIOTests(unittest.TestCase):
    def test_cli_defaults_to_full_and_staged_losses(self):
        args = parse_args(['--weather', 'night'])
        _validate(args)
        self.assertEqual(args.adapter_type, 'full')
        self.assertEqual((args.full_train_scope, args.full_train_blocks), ('late', 4))
        self.assertEqual(args.learning_rate, 1e-6)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual((args.lora_rank, args.lora_alpha, args.lora_dropout,
                          args.lora_blocks), (8, 16, 0, 4))
        self.assertEqual((args.global_weight, args.contrast_weight,
                          args.patch_weight, args.keep_weight, args.ot_weight),
                         (1, 1, 0.1, 0, 0))

    def test_accumulation_and_epoch_only_checkpoint_validation(self):
        args = parse_args(['--weather', 'night', '--gradient-accumulation-steps', '4',
                           '--save-every-steps', '0', '--adapter-type', 'full'])
        _validate(args)
        args.gradient_accumulation_steps = 0
        with self.assertRaises(ValueError):
            _validate(args)

    def test_old_output_metadata_and_new_lora_config(self):
        self.assertEqual(adapter_model_kwargs({'bottleneck': 32})['adapter_type'], 'output')
        self.assertEqual(adapter_model_kwargs({'adapter_type': 'lora'})['adapter_type'], 'lora')
        with self.assertRaises(ValueError):
            adapter_model_kwargs({'adapter_type': 'unknown'})

    def test_retrieval_guidance_flags_are_independent_and_validate(self):
        args = parse_args(['--weather', 'night', '--fixed-day-bank', 'runs/cache/day.safetensors',
                           '--relation-weight', '.1',
                           '--synthetic-guidance-weight', '.1', '--contrast-hard-negatives', '64',
                           '--layer-decay', '.8', '--lr-schedule', 'cosine',
                           '--warmup-steps', '32', '--gradient-checkpointing'])
        _validate(args)
        for name, bad in (('relation_weight', -1), ('relation_temperature', 0),
                          ('layer_decay', 1.1), ('min_lr_ratio', float('nan')),
                          ('contrast_hard_negatives', -1), ('full_train_blocks', 0)):
            old = getattr(args, name)
            setattr(args, name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate(args)
            setattr(args, name, old)
        args.adapter_type = 'lora'
        with self.assertRaises(ValueError):
            _validate(args)

    def test_source_night_loss_has_only_night_retrieval_targets(self):
        args = parse_args(['--weather', 'night', '--source-night-weight', '.5',
                           '--source-night-samples', '2', '--synthetic-guidance-weight', '0'])
        _validate(args)
        self.assertEqual(args.source_night_weight, .5)
        for name, bad in [('source_night_weight', -1), ('source_night_samples', 0),
                          ('weather', 'rain')]:
            old = getattr(args, name)
            setattr(args, name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate(args)
            setattr(args, name, old)
        args.global_weight = args.contrast_weight = 0
        with self.assertRaisesRegex(ValueError, 'Source night retrieval'):
            _validate(args)

    def test_output_and_lora_defaults_keep_valid_scope(self):
        for adapter_type in ('output', 'lora'):
            with self.subTest(adapter_type=adapter_type):
                args = parse_args(['--weather', 'night', '--device', 'cpu',
                                   '--adapter-type', adapter_type])
                _validate(args)
                self.assertEqual(args.full_train_scope, 'all')
                for scope in ('early', 'late'):
                    args.full_train_scope = scope
                    with self.assertRaisesRegex(ValueError, 'scope requires adapter-type full'):
                        _validate(args)


if __name__ == '__main__':
    unittest.main()

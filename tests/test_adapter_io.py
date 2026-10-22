"""CLI defaults and metadata keep earlier LoRA/Output runs loadable."""
import unittest

from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.training.train import parse_args, _validate


class AdapterIOTests(unittest.TestCase):
    def test_cli_defaults_to_full_and_staged_losses(self):
        args = parse_args(['--weather', 'night'])
        _validate(args)
        self.assertEqual(args.adapter_type, 'full')
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


if __name__ == '__main__':
    unittest.main()

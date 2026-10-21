"""Adapter metadata keeps old Output runs loadable while LoRA is the CLI default."""
import unittest

from weather_vpr.models.io import adapter_model_kwargs
from weather_vpr.training.train import parse_args, _validate


class AdapterIOTests(unittest.TestCase):
    def test_cli_defaults_to_lora_and_staged_losses(self):
        args = parse_args(['--weather', 'night'])
        _validate(args)
        self.assertEqual(args.adapter_type, 'lora')
        self.assertEqual((args.lora_rank, args.lora_alpha, args.lora_dropout,
                          args.lora_blocks), (8, 16, 0, 4))
        self.assertEqual((args.global_weight, args.contrast_weight,
                          args.patch_weight, args.keep_weight, args.ot_weight),
                         (1, 1, 0.1, 0, 0))

    def test_old_output_metadata_and_new_lora_config(self):
        self.assertEqual(adapter_model_kwargs({'bottleneck': 32})['adapter_type'], 'output')
        self.assertEqual(adapter_model_kwargs({'adapter_type': 'lora'})['adapter_type'], 'lora')
        with self.assertRaises(ValueError):
            adapter_model_kwargs({'adapter_type': 'unknown'})


if __name__ == '__main__':
    unittest.main()

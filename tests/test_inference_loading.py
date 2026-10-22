"""Weather inference artifact loading without a DINOv2 download."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from weather_vpr.inference.loading import load_weather_model


class TinyInferenceNet:
    def __init__(self):
        self.adapters = {"night": torch.nn.Linear(2, 2)}
        self.scales = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def set_lora_scale(self, weather, scale):
        self.scales.append((weather, scale))


class InferenceLoadingTests(unittest.TestCase):
    def _artifact(self, directory: Path, *, wrong_keys: bool = False):
        source = torch.nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(0.75)
            source.bias.fill_(-0.25)
        state = source.state_dict()
        if wrong_keys:
            state = {"wrong.weight": state["weight"], "wrong.bias": state["bias"]}
        save_file(state, str(directory / "adapter.safetensors"))

    def test_full_backbone_artifact_is_strictly_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._artifact(directory)
            fake = TinyInferenceNet()
            with patch("weather_vpr.inference.loading.WeatherAdapterSALAD", return_value=fake) as factory:
                model = load_weather_model(object(), "night", directory,
                                           {"adapter_type": "full"}, device="cpu")
            self.assertIs(model, fake)
            self.assertEqual(factory.call_args.kwargs["adapter_type"], "full")
            torch.testing.assert_close(fake.adapters["night"].weight,
                                       torch.full((2, 2), 0.75))
            torch.testing.assert_close(fake.adapters["night"].bias,
                                       torch.full((2,), -0.25))
            self.assertEqual(fake.scales, [])

    def test_full_rejects_lora_scale_before_constructing_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("weather_vpr.inference.loading.WeatherAdapterSALAD") as factory:
                with self.assertRaisesRegex(ValueError, "LoRA"):
                    load_weather_model(object(), "night", Path(temporary),
                                       {"adapter_type": "full"}, device="cpu",
                                       adapter_scale=0.5)
                factory.assert_not_called()

    def test_old_output_and_lora_still_route_correctly(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._artifact(directory)
            for metadata, scale, kind in (({}, 1.0, "output"),
                                           ({"adapter_type": "lora"}, 0.25, "lora")):
                fake = TinyInferenceNet()
                with patch("weather_vpr.inference.loading.WeatherAdapterSALAD", return_value=fake) as factory:
                    load_weather_model(object(), "night", directory, metadata,
                                       device="cpu", adapter_scale=scale)
                self.assertEqual(factory.call_args.kwargs["adapter_type"], kind)
                self.assertEqual(fake.scales, [("night", scale)] if kind == "lora" else [])

    def test_missing_or_unexpected_backbone_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._artifact(directory, wrong_keys=True)
            with patch("weather_vpr.inference.loading.WeatherAdapterSALAD",
                       return_value=TinyInferenceNet()):
                with self.assertRaises(RuntimeError):
                    load_weather_model(object(), "night", directory,
                                       {"adapter_type": "full"}, device="cpu")


if __name__ == "__main__":
    unittest.main()

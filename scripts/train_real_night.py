#!/usr/bin/env python3
"""Train a full Night DINOv2 on geographically supervised SVOX train queries."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weather_vpr.training.real_night import main  # noqa: E402


if __name__ == '__main__':
    main()

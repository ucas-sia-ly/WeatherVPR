#!/usr/bin/env python3
"""Evaluate Tokyo247 Night, public RobotCar poses, or a trusted retrieval manifest."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_vpr.inference.benchmarks import main

if __name__ == "__main__":
    main()

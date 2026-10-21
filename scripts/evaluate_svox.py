#!/usr/bin/env python3
"""Evaluate SVOX night queries; use --check-data for a model-free protocol check."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_vpr.inference.svox import main

if __name__ == "__main__":
    main()

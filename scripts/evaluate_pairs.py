#!/usr/bin/env python3
"""Evaluate held-out synthetic night/day pairs; see --help for options."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_vpr.inference.paired_validation import main

if __name__ == "__main__":
    main()

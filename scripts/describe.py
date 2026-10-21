#!/usr/bin/env python3
"""Entrypoint: python WeatherVPR/scripts/describe.py [args]."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weather_vpr.inference.describe import main
if __name__ == '__main__':
    main()

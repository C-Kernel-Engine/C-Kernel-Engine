#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "dataset" / "build_svg_dataset_visualizer_v8.py"
    runpy.run_path(str(target), run_name="__main__")

#!/usr/bin/env python3
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "dataset" / "materialize_svg_stage_artifacts_v8.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")

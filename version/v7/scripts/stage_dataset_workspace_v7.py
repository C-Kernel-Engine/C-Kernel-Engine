#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys


SCRIPT = Path(__file__).resolve().parent / "dataset" / "stage_dataset_workspace_v7.py"
if not SCRIPT.exists():
    raise SystemExit(f"missing script: {SCRIPT}")

sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")

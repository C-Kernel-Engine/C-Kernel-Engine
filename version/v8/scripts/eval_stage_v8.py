#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "eval_stage_v7.py"


if __name__ == "__main__":
    target_parent = str(TARGET.parent)
    if target_parent not in sys.path:
        sys.path.insert(0, target_parent)
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(TARGET), run_name="__main__")

#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "test_visualizer_js_units_v7.py"


if __name__ == "__main__":
    os.environ.setdefault("CK_VIS_VERSION", "v8")
    os.environ.setdefault("CK_VIS_CONTRACTS_DIR", str(ROOT / "version" / "v8" / "tests" / "contracts"))
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(TARGET), run_name="__main__")

#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "test_visualizer_health_v7.py"


if __name__ == "__main__":
    os.environ.setdefault("CK_VIS_VERSION", "v8")
    os.environ.setdefault("CK_VIS_CONTRACTS_DIR", str(ROOT / "version" / "v8" / "tests" / "contracts"))
    os.environ.setdefault("CK_VIS_IR_VIZ_SOURCE", str(ROOT / "version" / "v8" / "tools" / "ir_visualizer.html"))
    os.environ.setdefault("CK_VIS_DV_SOURCE", str(ROOT / "version" / "v8" / "scripts" / "dataset" / "build_svg_dataset_visualizer_v8.py"))
    os.environ.setdefault("CK_VIS_HUB_SOURCE", str(ROOT / "version" / "v8" / "tools" / "open_ir_hub_v8.py"))
    os.environ.setdefault("CK_VIS_MODELS_ROOT", str(Path.home() / ".cache" / "ck-engine-v8" / "models"))
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(TARGET), run_name="__main__")

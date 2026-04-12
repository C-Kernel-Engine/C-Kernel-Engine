#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "test_visualizer_generated_e2e_v7.py"


if __name__ == "__main__":
    os.environ.setdefault("CK_VIS_VERSION", "v8")
    os.environ.setdefault("CK_VIS_MODELS_ROOT", str(Path.home() / ".cache" / "ck-engine-v8" / "models"))
    os.environ.setdefault("CK_VIS_HEALTH_SCRIPT", str(ROOT / "version" / "v8" / "scripts" / "test_visualizer_health_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_VIZ", str(ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"))
    os.environ.setdefault("CK_VIS_PREPARE_VIEWER", str(ROOT / "version" / "v8" / "tools" / "prepare_run_viewer_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_HUB", str(ROOT / "version" / "v8" / "tools" / "open_ir_hub_v8.py"))
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(TARGET), run_name="__main__")

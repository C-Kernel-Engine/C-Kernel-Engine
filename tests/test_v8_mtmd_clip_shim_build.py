#!/usr/bin/env python3
"""Compile and load the Qwen3-VL mtmd adapter against the pinned llama.cpp."""

from __future__ import annotations

import ctypes
import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "numeric_parity_qwen3vl_mmproj_v8.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("qwen3vl_mtmd_shim_build", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import numeric parity harness: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    llama_root = module.LLAMA_CPP_ROOT
    required = [
        llama_root / "tools" / "mtmd" / "clip.h",
        llama_root / "tools" / "mtmd" / "clip-impl.h",
        llama_root / "build" / "bin" / "libmtmd.so",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("mtmd shim build requires: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="v8_mtmd_shim_build_") as tmpdir:
        shim = module._compile_mtmd_shim(Path(tmpdir))
        library = ctypes.CDLL(str(shim))
        for symbol in (
            "ck_mtmd_clip_embd_nbytes_by_img",
            "ck_mtmd_clip_encode_float_image",
        ):
            if not hasattr(library, symbol):
                raise RuntimeError(f"compiled mtmd shim is missing {symbol}")

    print("Qwen3-VL mtmd clip shim build: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

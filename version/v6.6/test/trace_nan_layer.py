#!/usr/bin/env python3
"""
Binary-search style scan to find the first stop-op where NaN appears.
Checks A_EMBEDDED_INPUT and A_LAYER_INPUT.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def resolve_default_model() -> Path:
    root = Path.home() / ".cache/ck-engine-v6.6/models"
    if not root.exists():
        raise FileNotFoundError(f"Model cache not found: {root}")
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "weights.bump").exists() and (d / "layout_decode.json").exists():
            return d
    raise FileNotFoundError(f"No model dirs with required artifacts in {root}")


def load_layout(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_offsets(layout: Dict) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
        off = buf.get("abs_offset", buf.get("offset"))
        if isinstance(off, int):
            if isinstance(buf.get("name"), str):
                offsets[buf["name"]] = off
            if isinstance(buf.get("define"), str):
                offsets[buf["define"]] = off
    return offsets


def check_nan_at_stop(
    model_dir: Path,
    offsets: Dict[str, int],
    embed_dim: int,
    token: int,
    stop_op: int,
) -> Optional[Dict]:
    os.environ["CK_STOP_OP"] = str(stop_op)
    try:
        engine_path = model_dir / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = model_dir / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = model_dir / "libmodel.so"
        lib = ctypes.CDLL(str(lib_path))

        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
        lib.ck_model_get_vocab_size.argtypes = []
        lib.ck_model_get_vocab_size.restype = ctypes.c_int
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        ret = lib.ck_model_init(str(model_dir / "weights.bump").encode())
        if ret != 0:
            return None
        try:
            base_ptr = int(lib.ck_model_get_base_ptr())
            vocab = int(lib.ck_model_get_vocab_size())
            logits = (ctypes.c_float * vocab)()
            ret = lib.ck_model_decode(token, logits)
            if ret != 0:
                return None

            emb_offset = offsets.get("A_EMBEDDED_INPUT", offsets.get("embedded_input"))
            li_offset = offsets.get("A_LAYER_INPUT", offsets.get("layer_input"))
            if emb_offset is None or li_offset is None:
                return None

            emb_ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
            emb_arr = np.ctypeslib.as_array(emb_ptr, shape=(embed_dim,)).copy()

            li_ptr = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
            li_arr = np.ctypeslib.as_array(li_ptr, shape=(embed_dim,)).copy()
            return {
                "embedded_nan": int(np.isnan(emb_arr).sum()),
                "layer_nan": int(np.isnan(li_arr).sum()),
                "embedded_arr": emb_arr,
            }
        finally:
            lib.ck_model_free()
    finally:
        if "CK_STOP_OP" in os.environ:
            del os.environ["CK_STOP_OP"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate first stop-op with NaN")
    parser.add_argument("--model", type=Path, default=None, help="Model directory")
    parser.add_argument("--token", type=int, default=25, help="Token ID")
    args = parser.parse_args()

    model_dir = args.model or resolve_default_model()
    layout = load_layout(model_dir / "layout_decode.json")
    offsets = build_offsets(layout)
    embed_dim = int(layout.get("config", {}).get("hidden_size") or layout.get("config", {}).get("d_model") or 896)

    print("Tracing where NaN first appears...")
    print("Checking A_EMBEDDED_INPUT and A_LAYER_INPUT at various ops\n")
    print(f"Model: {model_dir}")
    print(f"Token: {args.token}\n")

    checkpoints = [1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 624]
    last_good = 0
    first_bad = None

    for cp in checkpoints:
        result = check_nan_at_stop(model_dir, offsets, embed_dim, args.token, cp)
        if result is None:
            continue
        emb_nan = result["embedded_nan"]
        li_nan = result["layer_nan"]
        status = "OK" if emb_nan == 0 else f"NaN({emb_nan})"
        li_status = "OK" if li_nan == 0 else f"NaN({li_nan})"
        print(f"Op {cp:3d}: A_EMBEDDED_INPUT={status:12s}  A_LAYER_INPUT={li_status}")
        if emb_nan == 0:
            last_good = cp
        elif first_bad is None:
            first_bad = cp

    if first_bad is None:
        print("\nNo NaN found in checkpoints.")
        return 0

    print(f"\n--- Fine-grained search between op {last_good} and op {first_bad} ---\n")
    for op in range(last_good, first_bad + 1):
        result = check_nan_at_stop(model_dir, offsets, embed_dim, args.token, op)
        if result is None:
            continue
        emb_nan = result["embedded_nan"]
        li_nan = result["layer_nan"]
        status = "OK" if emb_nan == 0 else f"NaN({emb_nan})"
        li_status = "OK" if li_nan == 0 else f"NaN({li_nan})"
        print(f"Op {op:3d}: A_EMBEDDED_INPUT={status:12s}  A_LAYER_INPUT={li_status}")
        if emb_nan > 0 and last_good < op:
            print(f"\n*** NaN first appears in A_EMBEDDED_INPUT at op {op} ***")
            print("Inspect generated code around this op in model_v6_6.c")
            arr = result["embedded_arr"]
            print(f"Embedded first 20 values: {arr[:20]}")
            break
        if emb_nan == 0:
            last_good = op

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

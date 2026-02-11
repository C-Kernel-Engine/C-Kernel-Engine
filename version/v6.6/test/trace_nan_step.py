#!/usr/bin/env python3
"""Trace NaN signals from key activation buffers for a single decode."""

import argparse
import ctypes
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_default_model() -> Path:
    root = Path.home() / ".cache/ck-engine-v6.6/models"
    if not root.exists():
        raise FileNotFoundError(f"Model cache not found: {root}")
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "weights.bump").exists():
            return d
    raise FileNotFoundError(f"No model dirs with weights.bump in {root}")


def activation_offsets(layout: Dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
        off = buf.get("abs_offset", buf.get("offset"))
        if isinstance(off, int):
            if isinstance(buf.get("name"), str):
                out[buf["name"]] = off
            if isinstance(buf.get("define"), str):
                out[buf["define"]] = off
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace NaN at key buffer offsets")
    parser.add_argument("--model", type=Path, default=None, help="Model directory")
    parser.add_argument("--token", type=int, default=25, help="Token ID")
    args = parser.parse_args()

    model_dir = args.model or resolve_default_model()
    layout = load_json(model_dir / "layout_decode.json")
    offsets = activation_offsets(layout)

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
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    ret = lib.ck_model_init(str(model_dir / "weights.bump").encode())
    if ret != 0:
        raise RuntimeError(f"Failed to init model: {ret}")

    try:
        bump = int(lib.ck_model_get_base_ptr())
        vocab_size = int(lib.ck_model_get_vocab_size())
        out = (ctypes.c_float * vocab_size)()
        lib.ck_model_decode(args.token, out)

        print(f"Model: {model_dir}")
        print(f"Token: {args.token}")
        print(f"Bump base pointer: {hex(bump)}")

        logits_ptr = lib.ck_model_get_logits()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        print("\nLogits from API:")
        print(f"  [0:5]: {logits[:5]}")
        print(f"  NaN count: {np.isnan(logits).sum()}")

        if "A_LOGITS" in offsets:
            logits_direct = ctypes.cast(bump + offsets["A_LOGITS"], ctypes.POINTER(ctypes.c_float))
            logits_arr = np.ctypeslib.as_array(logits_direct, shape=(vocab_size,))
            print("\nLogits from A_LOGITS:")
            print(f"  [0:5]: {logits_arr[:5]}")
            print(f"  NaN count: {np.isnan(logits_arr).sum()}")

        embed_dim = int(
            layout.get("config", {}).get("hidden_size")
            or layout.get("config", {}).get("d_model")
            or 896
        )

        if "A_EMBEDDED_INPUT" in offsets:
            emb_ptr = ctypes.cast(bump + offsets["A_EMBEDDED_INPUT"], ctypes.POINTER(ctypes.c_float))
            emb = np.ctypeslib.as_array(emb_ptr, shape=(embed_dim,))
            print("\nA_EMBEDDED_INPUT:")
            print(f"  [0:5]: {emb[:5]}")
            print(f"  NaN count: {np.isnan(emb).sum()}")

        if "A_LAYER_INPUT" in offsets:
            layer_ptr = ctypes.cast(bump + offsets["A_LAYER_INPUT"], ctypes.POINTER(ctypes.c_float))
            layer = np.ctypeslib.as_array(layer_ptr, shape=(embed_dim,))
            print("\nA_LAYER_INPUT:")
            print(f"  [0:5]: {layer[:5]}")
            print(f"  NaN count: {np.isnan(layer).sum()}")
    finally:
        lib.ck_model_free()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

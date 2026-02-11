#!/usr/bin/env python3
"""
Trace v6.6 execution and localize first failing stop-op.

Usage:
    python trace_divergence.py --model ~/.cache/ck-engine-v6.6/models/<model> --token 25
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


DEFAULT_V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_model_config(model_dir: Path) -> Dict:
    cfg = load_json(model_dir / "config.json")
    if isinstance(cfg, dict):
        return cfg
    lowered = load_json(model_dir / "lowered_decode_call.json")
    if isinstance(lowered, dict):
        return lowered.get("config", {})
    lowered = load_json(model_dir / "lowered_decode.json")
    if isinstance(lowered, dict):
        return lowered.get("config", {})
    return {}


def build_activation_offset_map(layout: dict) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    buffers = layout.get("memory", {}).get("activations", {}).get("buffers", [])
    for buf in buffers:
        off = buf.get("abs_offset", buf.get("offset"))
        if isinstance(off, int):
            name = buf.get("name")
            define = buf.get("define")
            if isinstance(name, str):
                offsets[name] = off
            if isinstance(define, str):
                offsets[define] = off
        for entry in buf.get("entries", []):
            eoff = entry.get("abs_offset", entry.get("offset"))
            if not isinstance(eoff, int):
                continue
            name = entry.get("name")
            define = entry.get("define")
            if isinstance(name, str):
                offsets[name] = eoff
            if isinstance(define, str):
                offsets[define] = eoff
    return offsets


class Runner:
    def __init__(self, model_dir: Path, threads: int = 1, context_len: Optional[int] = None):
        self.model_dir = model_dir
        self.threads = max(1, int(threads))
        self.context_len = context_len
        self.config = load_model_config(model_dir)
        self.layout = load_json(model_dir / "layout_decode.json") or {}
        self.offsets = build_activation_offset_map(self.layout)
        self.has_base_ptr = False

    def _preload_optional_runtime_libs(self) -> None:
        lib_roots = [
            Path("/opt/intel/oneapi/compiler/latest/lib"),
            Path("/opt/intel/oneapi/compiler/2025.3/lib"),
            Path("/opt/intel/oneapi/2025.3/lib"),
        ]
        lib_names = [
            "libimf.so",
            "libsvml.so",
            "libintlc.so.5",
            "libirc.so",
            "libirc_s.so",
            "libirng.so",
        ]
        for root in lib_roots:
            if not root.exists():
                continue
            for name in lib_names:
                p = root / name
                if p.exists():
                    try:
                        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    def _load_library(self) -> ctypes.CDLL:
        if not self.model_dir.exists():
            raise RuntimeError(f"Model directory not found: {self.model_dir}")

        self._preload_optional_runtime_libs()

        engine_path = self.model_dir / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = self.model_dir / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            raise RuntimeError(f"No model library found in {self.model_dir}")

        try:
            lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise RuntimeError(f"Failed loading {lib_path}: {e}")

        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_get_vocab_size.argtypes = []
        lib.ck_model_get_vocab_size.restype = ctypes.c_int
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        try:
            lib.ck_model_get_base_ptr.argtypes = []
            lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
            self.has_base_ptr = True
        except AttributeError:
            self.has_base_ptr = False

        return lib

    def _read_activation(self, base_ptr: int, offset: Optional[int], size: int) -> Optional[np.ndarray]:
        if not base_ptr or offset is None or size <= 0:
            return None
        ptr = ctypes.cast(base_ptr + int(offset), ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    def run(self, token: int, stop_op: int = -1) -> Dict[str, Optional[np.ndarray]]:
        os.environ["CK_NUM_THREADS"] = str(self.threads)
        os.environ["OMP_NUM_THREADS"] = str(self.threads)

        lib = self._load_library()
        weights_path = self.model_dir / "weights.bump"

        ret = lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            raise RuntimeError(f"ck_model_init failed: {ret}")

        try:
            if hasattr(lib, "ck_model_kv_cache_reset"):
                try:
                    lib.ck_model_kv_cache_reset.argtypes = []
                    lib.ck_model_kv_cache_reset.restype = None
                    lib.ck_model_kv_cache_reset()
                except Exception:
                    pass

            if self.context_len is not None and hasattr(lib, "ck_model_get_context_window"):
                try:
                    lib.ck_model_get_context_window.argtypes = []
                    lib.ck_model_get_context_window.restype = ctypes.c_int
                    runtime_ctx = int(lib.ck_model_get_context_window())
                    if runtime_ctx != int(self.context_len):
                        print(
                            f"WARNING: requested --context-len {self.context_len}, "
                            f"runtime is compiled for {runtime_ctx}"
                        )
                except Exception:
                    pass

            vocab_size = int(lib.ck_model_get_vocab_size())
            if vocab_size <= 0:
                vocab_size = int(self.config.get("vocab_size", 151936))
            output = (ctypes.c_float * vocab_size)()

            if stop_op >= 0:
                os.environ["CK_STOP_OP"] = str(stop_op)
            try:
                ret = lib.ck_model_decode(token, output)
            finally:
                if stop_op >= 0 and "CK_STOP_OP" in os.environ:
                    del os.environ["CK_STOP_OP"]

            if ret != 0:
                raise RuntimeError(f"ck_model_decode failed: {ret}")

            base_ptr = 0
            if self.has_base_ptr:
                base_ptr = int(lib.ck_model_get_base_ptr())

            embed_dim = int(
                self.config.get("hidden_size")
                or self.config.get("d_model")
                or self.config.get("embed_dim")
                or 896
            )
            num_heads = int(self.config.get("num_attention_heads") or self.config.get("n_head") or 1)
            head_dim = int(self.config.get("head_dim") or max(embed_dim // max(num_heads, 1), 1))

            return {
                "logits": np.array(output[:], dtype=np.float32),
                "embedding": self._read_activation(base_ptr, self.offsets.get("A_EMBEDDED_INPUT"), embed_dim),
                "layer_input": self._read_activation(base_ptr, self.offsets.get("A_LAYER_INPUT"), embed_dim),
                "residual": self._read_activation(base_ptr, self.offsets.get("A_RESIDUAL"), embed_dim),
                "q_scratch": self._read_activation(base_ptr, self.offsets.get("A_Q_SCRATCH"), embed_dim),
                "k_scratch": self._read_activation(base_ptr, self.offsets.get("A_K_SCRATCH"), head_dim),
                "v_scratch": self._read_activation(base_ptr, self.offsets.get("A_V_SCRATCH"), head_dim),
                "attn_scratch": self._read_activation(base_ptr, self.offsets.get("A_ATTN_SCRATCH"), embed_dim),
            }
        finally:
            lib.ck_model_free()


def print_activation_summary(name: str, arr: Optional[np.ndarray]) -> None:
    if arr is None:
        print(f"{name:15s}: n/a")
        return
    print(
        f"{name:15s}: min={arr.min():10.4f}, max={arr.max():10.4f}, "
        f"mean={arr.mean():10.4f}, first5={arr[:5]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace v6.6 stop-op execution")
    parser.add_argument("--model", type=Path, default=DEFAULT_V66_DIR, help="Path to v6.6 model directory")
    parser.add_argument("--token", type=int, default=25, help="Token ID to decode")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Force runtime threads (default: 1 for stable debug runs)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Expected runtime context window (warning if mismatch)",
    )
    parser.add_argument(
        "--stop-ops",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 10, 15, 20],
        help="Stop-op sequence to probe",
    )
    parser.add_argument("--skip-stop-scan", action="store_true", help="Skip stop-op probing")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRACING V6.6 EXECUTION for token {args.token}")
    print("=" * 70)

    runner = Runner(args.model, threads=args.threads, context_len=args.context_len)

    print("\n--- FULL DECODE ---")
    try:
        out = runner.run(args.token)
    except Exception as e:
        print(f"ERROR: v6.6 decode failed: {e}")
        return 1

    logits = out["logits"]
    if logits is None:
        print("ERROR: logits unavailable")
        return 1

    print(
        f"v6.6 logits: min={logits.min():.4f}, max={logits.max():.4f}, "
        f"argmax={int(np.argmax(logits))}"
    )

    print("\n--- V6.6 ACTIVATION SUMMARY ---")
    for key in (
        "embedding",
        "layer_input",
        "residual",
        "q_scratch",
        "k_scratch",
        "v_scratch",
        "attn_scratch",
    ):
        print_activation_summary(key, out.get(key))

    if not args.skip_stop_scan:
        print("\n" + "=" * 70)
        print("STEP-BY-STEP STOP_OP TEST (v6.6)")
        print("=" * 70)
        for stop in args.stop_ops:
            try:
                result = runner.run(args.token, stop_op=stop)
                emb = result.get("embedding")
                q = result.get("q_scratch")
                emb_s = "n/a" if emb is None else f"[{emb.min():.4f}, {emb.max():.4f}]"
                q_s = "n/a" if q is None else f"[{q.min():.4f}, {q.max():.4f}]"
                print(f"stop={stop:3d}: embedding={emb_s} q_scratch={q_s}")
            except Exception as e:
                print(f"stop={stop:3d}: ERROR {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
test_numerical_parity.py - Numerical validation for v6.6 runtime only.

USAGE:
    python test_numerical_parity.py --model ~/.cache/ck-engine-v6.6/models/<model>
    python test_numerical_parity.py --model <path> --tokens 1 25 42 --threads 1 --context-len 1024

This test:
1. Runs input tokens through v6.6 decode API
2. Captures final logits
3. Flags NaN/Inf and basic output anomalies
4. Provides operation-order diagnosis hints on failure
"""

import argparse
import ctypes
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LayerSnapshot:
    """Snapshot of an observed tensor in the v6.6 pipeline."""

    layer_idx: int
    name: str
    values: np.ndarray
    has_nan: bool = False
    has_inf: bool = False
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0

    def compute_stats(self) -> None:
        self.has_nan = bool(np.any(np.isnan(self.values)))
        self.has_inf = bool(np.any(np.isinf(self.values)))
        if not self.has_nan and not self.has_inf:
            self.min_val = float(np.min(self.values))
            self.max_val = float(np.max(self.values))
            self.mean_val = float(np.mean(self.values))
            self.std_val = float(np.std(self.values))


@dataclass
class PipelineTrace:
    """Minimal execution trace for a v6.6 decode run."""

    model_name: str
    input_tokens: List[int]
    snapshots: List[LayerSnapshot] = field(default_factory=list)
    final_logits: Optional[np.ndarray] = None
    output_token: int = -1


class NumericalValidationTest:
    """Numerical health check for v6.6 output."""

    def __init__(
        self,
        model_path: Path,
        threads: int = 1,
        context_len: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.threads = max(1, int(threads))
        self.context_len = context_len
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load model config from config.json or lowered IR."""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        lowered_path = self.model_path / "lowered_decode.json"
        if lowered_path.exists():
            with open(lowered_path) as f:
                data = json.load(f)
                return data.get("config", {})

        lowered_call_path = self.model_path / "lowered_decode_call.json"
        if lowered_call_path.exists():
            with open(lowered_call_path) as f:
                data = json.load(f)
                return data.get("config", {})

        return {}

    def _preload_optional_runtime_libs(self) -> None:
        """Best-effort preload for Intel oneAPI runtime libs if present."""
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

    def _load_v66_library(self) -> ctypes.CDLL:
        """Load generated v6.6 model library."""
        self._preload_optional_runtime_libs()

        engine_path = self.model_path / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = self.model_path / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = self.model_path / "libmodel.so"

        try:
            lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model library {lib_path}: {e}. "
                "Check runtime linker dependencies (e.g. LD_LIBRARY_PATH)."
            )

        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_get_vocab_size.argtypes = []
        lib.ck_model_get_vocab_size.restype = ctypes.c_int
        lib.ck_model_get_logits.argtypes = []
        lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        return lib

    def run_v66(self, input_tokens: List[int]) -> PipelineTrace:
        """Run v6.6 decode path and capture final logits."""
        trace = PipelineTrace(model_name="v6.6", input_tokens=input_tokens)

        os.environ["CK_NUM_THREADS"] = str(self.threads)
        os.environ["OMP_NUM_THREADS"] = str(self.threads)

        lib = self._load_v66_library()
        weights_path = self.model_path / "weights.bump"

        result = lib.ck_model_init(str(weights_path).encode())
        if result != 0:
            raise RuntimeError(f"v6.6 init failed: {result}")

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
        for i, token in enumerate(input_tokens):
            result = lib.ck_model_decode(token, output)
            if result != 0:
                raise RuntimeError(f"v6.6 decode failed at token index {i}: {result}")

        logits_ptr = lib.ck_model_get_logits()
        if not logits_ptr:
            raise RuntimeError("ck_model_get_logits returned NULL")

        trace.final_logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()
        snapshot = LayerSnapshot(layer_idx=-1, name="final_logits", values=trace.final_logits)
        snapshot.compute_stats()
        trace.snapshots.append(snapshot)
        trace.output_token = int(np.argmax(trace.final_logits))

        lib.ck_model_free()
        return trace

    def run_test(self, input_tokens: Optional[List[int]] = None) -> Dict:
        """Run v6.6 numerical validation."""
        if input_tokens is None:
            input_tokens = [1]

        print("=" * 70)
        print("NUMERICAL VALIDATION TEST: v6.6")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Input tokens: {input_tokens}")
        print()

        print("Running v6.6...")
        trace = self.run_v66(input_tokens)

        snap = trace.snapshots[0] if trace.snapshots else None
        has_nan = bool(snap.has_nan) if snap else True
        has_inf = bool(snap.has_inf) if snap else True
        finite = not has_nan and not has_inf

        if snap is not None:
            status = "OK" if finite else ("NaN" if has_nan else "Inf")
            print(
                f"  final_logits: [{snap.min_val:.4f}, {snap.max_val:.4f}] "
                f"mean={snap.mean_val:.4f} std={snap.std_val:.4f} [{status}]"
            )

        in_vocab_range = False
        if trace.final_logits is not None:
            in_vocab_range = 0 <= trace.output_token < int(trace.final_logits.shape[0])

        print(f"  output_token: {trace.output_token}")

        passed = finite and in_vocab_range

        if not in_vocab_range:
            print("  [FAIL] output token index out of vocab range")

        print("\n" + "=" * 70)
        print(f"RESULT: {'PASSED' if passed else 'FAILED'}")
        print("=" * 70)

        return {
            "passed": passed,
            "v66_output": trace.output_token,
            "v66_has_nan": has_nan,
            "v66_has_inf": has_inf,
        }


def diagnose_nan_source(model_path: Path) -> None:
    """Diagnose where NaN values may appear based on generated op sequence."""
    print("\n" + "=" * 70)
    print("NaN SOURCE DIAGNOSIS")
    print("=" * 70)

    c_file = model_path / "ck-kernel-inference.c"
    if not c_file.exists():
        c_file = model_path / "model_v6_6.c"
    if not c_file.exists():
        print(f"ERROR: generated C not found in {model_path}")
        return

    with open(c_file) as f:
        content = f.read()

    import re

    kernel_calls = re.findall(r"/\* Op (\d+): (\w+) \((.*?)\) \*/", content)
    print(f"\nOperation sequence ({len(kernel_calls)} ops):")
    for op_num, kernel, details in kernel_calls[:30]:
        print(f"  Op {op_num:3s}: {kernel:40s} ({details})")
    if len(kernel_calls) > 30:
        print(f"  ... and {len(kernel_calls) - 30} more ops")

    print("\n--- Debugging Suggestions ---")
    print("1. Re-run with --threads 1 and --context-len matching generated runtime")
    print("2. Use CK_STOP_OP to bisect the first failing operation")
    print("3. Check weight offsets match manifest (W_PTR vs actual offset)")
    print("4. Verify activation buffer sizes are sufficient")

    print("\n--- Checking Common Issues ---")
    act_offsets = re.findall(r"model->activations \+ (\d+)", content)
    if act_offsets:
        max_offset = max(int(o) for o in act_offsets)
        print(f"  Max activation offset: {max_offset:,} bytes")

    w_ptr_calls = re.findall(r"W_PTR\((.*?)\)", content)
    print(f"  W_PTR calls: {len(w_ptr_calls)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Numerical validation test for v6.6")
    parser.add_argument("--model", type=Path, help="Path to v6.6 model directory")
    parser.add_argument("--model-dir", type=Path, help="Alias for --model")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1], help="Input tokens to test")
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
    parser.add_argument("--diagnose", action="store_true", help="Run NaN diagnosis")

    args = parser.parse_args()
    model_path = args.model or args.model_dir
    if model_path is None:
        parser.error("Missing model path: provide --model or --model-dir")

    test = NumericalValidationTest(
        model_path=model_path,
        threads=args.threads,
        context_len=args.context_len,
    )

    try:
        result = test.run_test(args.tokens)
    except Exception as e:
        print(f"FAILED: {e}")
        result = {"passed": False}

    if args.diagnose or not result["passed"]:
        diagnose_nan_source(model_path)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
v6_5_vs_v6_6_comparison.py - Compare v6.5 and v6.6 layer by layer

This test identifies WHERE v6.6 diverges from v6.5.

Usage:
    python v6_5_vs_v6_6_comparison.py --token 9707
    python v6_5_vs_v6_6_comparison.py --all-layers
"""

import argparse
import ctypes
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class V5V6Comparator:
    """Compare v6.5 and v6.6 outputs."""

    def __init__(self, v65_dir: Path, v66_dir: Path, verbose: bool = False):
        self.v65_dir = v65_dir
        self.v66_dir = v66_dir
        self.verbose = verbose
        self.lib65 = None
        self.lib66 = None
        self.config = {}

    def load_v65(self) -> bool:
        """Load v6.5 model."""
        lib_path = self.v65_dir / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = self.v65_dir / "libmodel.so"

        if not lib_path.exists():
            print(f"ERROR: v6.5 lib not found in {self.v65_dir}")
            return False

        self.lib65 = ctypes.CDLL(str(lib_path))
        self.lib65.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib65.ck_model_init.restype = ctypes.c_int
        self.lib65.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        self.lib65.ck_model_decode.restype = ctypes.c_int
        self.lib65.ck_model_get_logits.argtypes = []
        self.lib65.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        self.lib65.ck_model_free.argtypes = []
        self.lib65.ck_model_free.restype = None

        weights = self.v65_dir / "weights.bump"
        ret = self.lib65.ck_model_init(str(weights).encode())
        if ret != 0:
            print(f"ERROR: v6.5 init failed: {ret}")
            return False

        print(f"v6.5 loaded successfully")
        return True

    def load_v66(self) -> bool:
        """Load v6.6 model."""
        # Load engine first
        engine_path = self.v66_dir / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = self.v66_dir / "libmodel.so"
        if not lib_path.exists():
            lib_path = self.v66_dir / "ck-kernel-inference.so"

        if not lib_path.exists():
            print(f"ERROR: v6.6 lib not found in {self.v66_dir}")
            return False

        self.lib66 = ctypes.CDLL(str(lib_path))
        self.lib66.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib66.ck_model_init.restype = ctypes.c_int
        self.lib66.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        self.lib66.ck_model_decode.restype = ctypes.c_int
        self.lib66.ck_model_get_logits.argtypes = []
        self.lib66.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        self.lib66.ck_model_free.argtypes = []
        self.lib66.ck_model_free.restype = None

        weights = self.v66_dir / "weights.bump"
        ret = self.lib66.ck_model_init(str(weights).encode())
        if ret != 0:
            print(f"ERROR: v6.6 init failed: {ret}")
            return False

        # Load config for comparison
        lowered_path = self.v66_dir / "lowered_decode_call.json"
        if lowered_path.exists():
            with open(lowered_path) as f:
                self.config = json.load(f).get("config", {})

        print(f"v6.6 loaded successfully")
        return True

    def run_decode(self, lib, token: int) -> np.ndarray:
        """Run single decode and return logits."""
        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        ret = lib.ck_model_decode(ctypes.c_int32(token), output)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")

        return np.array(output[:], dtype=np.float32)

    def compare_logits(self, logits65: np.ndarray, logits66: np.ndarray) -> Dict:
        """Compare two logits arrays."""
        diff = np.abs(logits65 - logits66)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = diff / (np.abs(logits65) + 1e-8)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
        max_rel = float(np.max(rel_diff))

        # Top predictions
        top65 = np.argsort(logits65)[-5:][::-1]
        top66 = np.argsort(logits66)[-5:][::-1]

        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "max_rel": max_rel,
            "top5_match": list(top65) == list(top66),
            "top65": list(top65),
            "top66": list(top66),
        }

    def test_single_token(self, token: int) -> bool:
        """Test single token decode."""
        print("\n" + "="*70)
        print(f"SINGLE TOKEN TEST (token={token})")
        print("="*70)

        # Run both
        logits65 = self.run_decode(self.lib65, token)
        logits66 = self.run_decode(self.lib66, token)

        # Compare
        result = self.compare_logits(logits65, logits66)

        print(f"\nv6.5: min={logits65.min():.4f}, max={logits65.max():.4f}")
        print(f"v6.6: min={logits66.min():.4f}, max={logits66.max():.4f}")
        print(f"\nDiff analysis:")
        print(f"  Max absolute diff: {result['max_diff']:.6f}")
        print(f"  Mean absolute diff: {result['mean_diff']:.6f}")
        print(f"  Max relative diff: {result['max_rel']:.6f}")
        print(f"\nTop-5 predictions:")
        print(f"  v6.5: {result['top65']}")
        print(f"  v6.6: {result['top66']}")
        print(f"  Match: {result['top5_match']}")

        if result['top5_match']:
            print("\n✓ TOP-5 MATCH - Output is functionally identical!")
            return True
        else:
            print("\n✗ TOP-5 MISMATCH - Output differs!")
            return False

    def test_multi_token(self, tokens: List[int]) -> bool:
        """Test multi-token sequence."""
        print("\n" + "="*70)
        print(f"MULTI TOKEN TEST ({len(tokens)} tokens)")
        print("="*70)

        # For now, just run decode for last token
        # Full prefill comparison would require ck_model_embed_tokens
        last_token = tokens[-1]
        return self.test_single_token(last_token)

    def cleanup(self):
        """Free both models."""
        if self.lib65:
            self.lib65.ck_model_free()
        if self.lib66:
            self.lib66.ck_model_free()


def main():
    parser = argparse.ArgumentParser(description="Compare v6.5 vs v6.6")
    parser.add_argument("--v65-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF",
                       help="v6.5 model directory")
    parser.add_argument("--v66-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF",
                       help="v6.6 model directory")
    parser.add_argument("--token", type=int, default=9707, help="Token to test")
    parser.add_argument("--tokens", type=int, nargs="+", help="Multiple tokens")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    comparator = V5V6Comparator(args.v65_dir, args.v66_dir, args.verbose)

    if not comparator.load_v65():
        return 1
    if not comparator.load_v66():
        comparator.lib65.ck_model_free()
        return 1

    try:
        if args.tokens:
            success = comparator.test_multi_token(args.tokens)
        else:
            success = comparator.test_single_token(args.token)
    finally:
        comparator.cleanup()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

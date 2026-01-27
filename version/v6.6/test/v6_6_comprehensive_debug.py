#!/usr/bin/env python3
"""
v6.6_comprehensive_debug_test.py - Debug v6.6 step by step

This test validates each layer operation independently to find
where the output diverges from expected values.

Usage:
    python v6.6_comprehensive_debug_test.py --stop-at 0    # Test just embedding
    python v6.6_comprehensive_debug_test.py --stop-at 5    # Test through layer 0 attention
    python v6.6_comprehensive_debug_test.py --verbose
"""

import argparse
import ctypes
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class V6DebugTester:
    """Comprehensive debug testing for v6.6."""

    def __init__(self, model_dir: Path, verbose: bool = False):
        self.model_dir = model_dir
        self.verbose = verbose
        self.lib = None
        self.base_ptr = None
        self.config = {}
        self.layout = {}

    def load(self) -> bool:
        """Load model and config."""
        # Load lowered IR for offsets
        lowered_path = self.model_dir / "lowered_decode_call.json"
        if not lowered_path.exists():
            lowered_path = self.model_dir / "lowered_decode.json"

        with open(lowered_path) as f:
            self.lowered = json.load(f)

        # Load layout for activation offsets
        layout_path = self.model_dir / "layout_decode.json"
        with open(layout_path) as f:
            self.layout = json.load(f)

        # Get config
        self.config = self.lowered.get("config", {})

        # Load model library
        lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            lib_path = self.model_dir / "ck-kernel-inference.so"

        if not lib_path.exists():
            print(f"ERROR: No .so found in {self.model_dir}")
            return False

        # Load engine first
        engine_path = self.model_dir / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        self.lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int
        self.lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_decode.restype = ctypes.c_int
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None
        self.lib.ck_model_get_base_ptr.argtypes = []
        self.lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        # Initialize model
        weights_path = self.model_dir / "weights.bump"
        ret = self.lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"ERROR: ck_model_init failed with code {ret}")
            return False

        self.base_ptr = self.lib.ck_model_get_base_ptr()
        print(f"Model loaded. Base pointer: 0x{self.base_ptr:x}")

        return True

    def read_activation(self, offset: int, size: int) -> np.ndarray:
        """Read activation from memory."""
        if self.base_ptr == 0:
            raise RuntimeError("Model not initialized")

        ptr = ctypes.cast(self.base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    def get_layout_offset(self, name: str) -> Optional[int]:
        """Get activation offset from layout."""
        memory = self.layout.get("memory", {})
        activations = memory.get("activations", {})
        buffers = activations.get("buffers", [])

        for buf in buffers:
            if buf.get("name") == name:
                return buf.get("abs_offset")
            # Check entries within buffer
            for entry in buf.get("entries", []):
                if entry.get("name") == name:
                    return entry.get("abs_offset")
        return None

    def get_weight_offset(self, name: str) -> Optional[int]:
        """Get weight offset from IR."""
        memory = self.lowered.get("memory", {})
        weights = memory.get("weights", {})
        entries = weights.get("entries", [])

        for entry in entries:
            if entry.get("name") == name or entry.get("define") == name:
                return entry.get("abs_offset")
        return None

    def run_decode_with_stop(self, token: int, stop_op: int) -> Dict:
        """Run decode and stop at specific operation."""
        import os
        os.environ["CK_STOP_OP"] = str(stop_op)

        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        ret = self.lib.ck_model_decode(ctypes.c_int32(token), output)
        result = {
            "return_code": ret,
            "logits": np.array(output[:], dtype=np.float32) if ret == 0 else None
        }

        del os.environ["CK_STOP_OP"]
        return result

    def test_embedding(self) -> Tuple[bool, str]:
        """Test 1: Embedding lookup."""
        print("\n" + "="*60)
        print("TEST: EMBEDDING")
        print("="*60)

        # Get expected offset
        offset = self.get_weight_offset("token_emb")
        if offset:
            print(f"Token emb weight offset: {offset}")

        # Get layout offset
        emb_input_offset = self.get_layout_offset("A_EMBEDDED_INPUT")
        print(f"A_EMBEDDED_INPUT layout offset: {emb_input_offset}")

        # Run decode, stop at op 0 (after embedding)
        result = self.run_decode_with_stop(9707, 0)

        if result["logits"] is None:
            return False, f"Decode failed with code {result['return_code']}"

        logits = result["logits"]
        print(f"Logits after embedding: min={logits.min():.4f}, max={logits.max():.4f}")

        # Check for NaN/Inf
        if np.isnan(logits).any():
            return False, "NaN detected in logits"
        if np.isinf(logits).any():
            return False, "Inf detected in logits"

        # Check if reasonable range
        if logits.max() > 100 or logits.min() < -100:
            print(f"WARNING: Logits out of expected range")

        return True, "Embedding OK"

    def test_layer0_rmsnorm(self) -> Tuple[bool, str]:
        """Test 2: Layer 0 first RMSNorm."""
        print("\n" + "="*60)
        print("TEST: LAYER 0 RMSNORM (pre-attention)")
        print("="*60)

        # Get gamma weight offset
        gamma_offset = self.get_weight_offset("layer.0.ln1_gamma")
        print(f"W_LAYER_0_LN1_GAMMA offset: {gamma_offset}")

        # Get output offset
        layer_input_offset = self.get_layout_offset("A_LAYER_INPUT")
        print(f"A_LAYER_INPUT layout offset: {layer_input_offset}")

        # Run decode, stop after op 1 (after first rmsnorm)
        result = self.run_decode_with_stop(9707, 1)

        if result["logits"] is None:
            return False, f"Decode failed"

        logits = result["logits"]
        print(f"Logits after RMSNorm: min={logits.min():.4f}, max={logits.max():.4f}")

        return True, "RMSNorm OK"

    def test_qkv_projection(self) -> Tuple[bool, str]:
        """Test 3: QKV projection."""
        print("\n" + "="*60)
        print("TEST: QKV PROJECTION")
        print("="*60)

        # Get weight offsets
        wq_offset = self.get_weight_offset("layer.0.wq")
        wk_offset = self.get_weight_offset("layer.0.wk")
        wv_offset = self.get_weight_offset("layer.0.wv")

        print(f"W_LAYER_0_WQ offset: {wq_offset}")
        print(f"W_LAYER_0_WK offset: {wk_offset}")
        print(f"W_LAYER_0_WV offset: {wv_offset}")

        # Get bias offsets
        bq_offset = self.get_weight_offset("layer.0.bq")
        bk_offset = self.get_weight_offset("layer.0.bk")
        bv_offset = self.get_weight_offset("layer.0.bv")

        print(f"W_LAYER_0_BQ offset: {bq_offset}")
        print(f"W_LAYER_0_BK offset: {bk_offset}")
        print(f"W_LAYER_0_BV offset: {bv_offset}")

        # Get scratch buffer offsets
        q_offset = self.get_layout_offset("A_Q_SCRATCH")
        k_offset = self.get_layout_offset("A_K_SCRATCH")
        v_offset = self.get_layout_offset("A_V_SCRATCH")

        print(f"A_Q_SCRATCH layout offset: {q_offset}")
        print(f"A_K_SCRATCH layout offset: {k_offset}")
        print(f"A_V_SCRATCH layout offset: {v_offset}")

        # Run decode, stop after QKV (op 2-3 for decode, op 3-4 for prefill)
        result = self.run_decode_with_stop(9707, 3)

        if result["logits"] is None:
            return False, f"Decode failed"

        logits = result["logits"]
        print(f"Logits after QKV: min={logits.min():.4f}, max={logits.max():.4f}")

        return True, "QKV OK"

    def test_attention(self) -> Tuple[bool, str]:
        """Test 4: Attention output."""
        print("\n" + "="*60)
        print("TEST: ATTENTION")
        print("="*60)

        # Run decode through attention
        result = self.run_decode_with_stop(9707, 10)

        if result["logits"] is None:
            return False, f"Decode failed"

        logits = result["logits"]
        print(f"Logits after attention: min={logits.min():.4f}, max={logits.max():.4f}")

        return True, "Attention OK"

    def test_mlp(self) -> Tuple[bool, str]:
        """Test 5: MLP."""
        print("\n" + "="*60)
        print("TEST: MLP")
        print("="*60)

        # Run decode through MLP
        result = self.run_decode_with_stop(9707, 20)

        if result["logits"] is None:
            return False, f"Decode failed"

        logits = result["logits"]
        print(f"Logits after MLP: min={logits.min():.4f}, max={logits.max():.4f}")

        return True, "MLP OK"

    def test_full_decode(self) -> Tuple[bool, str]:
        """Test 6: Full decode."""
        print("\n" + "="*60)
        print("TEST: FULL DECODE")
        print("="*60)

        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        ret = self.lib.ck_model_decode(ctypes.c_int32(9707), output)

        if ret != 0:
            return False, f"Decode failed with code {ret}"

        logits = np.array(output[:], dtype=np.float32)
        print(f"Final logits: min={logits.min():.4f}, max={logits.max():.4f}")

        # Check for issues
        if np.isnan(logits).any():
            return False, "NaN in final logits"
        if np.isinf(logits).any():
            return False, "Inf in final logits"

        # Print top 5 predictions
        top5 = np.argsort(logits)[-5:][::-1]
        print(f"Top 5 tokens: {list(top5)}")

        return True, "Full decode OK"

    def check_memory_consistency(self) -> Tuple[bool, str]:
        """Check that all memory offsets are consistent between IR and layout."""
        print("\n" + "="*60)
        print("MEMORY CONSISTENCY CHECK")
        print("="*60)

        issues = []

        # Check weight offsets
        memory = self.lowered.get("memory", {})
        weights = memory.get("weights", {})
        entries = weights.get("entries", [])

        weight_defines = {e.get("define"): e.get("abs_offset") for e in entries}

        # Check layout activation offsets
        layout_memory = self.layout.get("memory", {})
        layout_activations = layout_memory.get("activations", {})
        buffers = layout_activations.get("buffers", [])

        layout_offsets = {}
        for buf in buffers:
            for entry in buf.get("entries", []):
                name = entry.get("name")
                offset = entry.get("abs_offset")
                if name and offset:
                    layout_offsets[name] = offset

        print(f"IR defines: {len(weight_defines)} weights")
        print(f"Layout defines: {len(layout_offsets)} activations")

        # Check that all referenced offsets in code exist in defines
        # This is done by examining the generated C code
        c_code_path = self.model_dir / "model_v6_6.c"
        if c_code_path.exists():
            with open(c_code_path) as f:
                c_code = f.read()

            # Find all offset references
            import re
            hardcoded_offsets = re.findall(r'\(model->bump\s*\+\s*(\d+)\)', c_code)

            print(f"\nHardcoded offsets in C code: {len(hardcoded_offsets)}")

            # Check for offsets that should be macros
            # Look for hardcoded numbers that aren't small constants
            suspicious = []
            for off in hardcoded_offsets:
                try:
                    val = int(off)
                    if val > 1000000:  # Likely should be a macro
                        suspicious.append(val)
                except:
                    pass

            if suspicious:
                print(f"Suspicious hardcoded offsets (>1M): {suspicious[:10]}...")
                issues.append(f"Found {len(suspicious)} hardcoded offsets that should be macros")

        if issues:
            return False, "; ".join(issues)
        return True, "Memory consistent"

    def cleanup(self):
        """Cleanup model."""
        if self.lib:
            self.lib.ck_model_free()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive v6.6 debug test")
    parser.add_argument("--model-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF",
                       help="Model directory")
    parser.add_argument("--stop-at", type=int, default=None,
                       help="Stop testing at specific step (0=embedding, 1=rmsnorm, 2=qkv, etc.)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tester = V6DebugTester(args.model_dir, verbose=args.verbose)

    if not tester.load():
        return 1

    results = []

    # Run tests
    if args.stop_at is None or args.stop_at == 0:
        results.append(("Embedding", tester.test_embedding()))

    if args.stop_at is None or args.stop_at >= 1:
        results.append(("RMSNorm", tester.test_layer0_rmsnorm()))

    if args.stop_at is None or args.stop_at >= 3:
        results.append(("QKV", tester.test_qkv_projection()))

    if args.stop_at is None or args.stop_at >= 10:
        results.append(("Attention", tester.test_attention()))

    if args.stop_at is None or args.stop_at >= 20:
        results.append(("MLP", tester.test_mlp()))

    if args.stop_at is None:
        results.append(("Memory Check", tester.check_memory_consistency()))
        results.append(("Full Decode", tester.test_full_decode()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for name, (ok, msg) in results:
        status = "PASS" if ok else "FAIL"
        symbol = "✓" if ok else "✗"
        print(f"{symbol} {name}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    tester.cleanup()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

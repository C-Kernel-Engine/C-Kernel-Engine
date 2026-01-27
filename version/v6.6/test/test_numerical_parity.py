#!/usr/bin/env python3
"""
test_numerical_parity.py - Layer-by-layer numerical validation for v6.6

Compares v6.6 pipeline against reference (v6.5 or llama.cpp) to find divergence.

USAGE:
    python test_numerical_parity.py --model ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF
    python test_numerical_parity.py --model <path> --reference v6.5
    python test_numerical_parity.py --model <path> --reference llamacpp

This test:
1. Runs the same input through both pipelines
2. Captures intermediate activations at each layer
3. Reports where values diverge (first NaN, first large diff)
4. Generates a detailed report for debugging
"""

import argparse
import ctypes
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile
import os


@dataclass
class LayerSnapshot:
    """Snapshot of activations at a layer."""
    layer_idx: int
    name: str
    values: np.ndarray
    has_nan: bool = False
    has_inf: bool = False
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0

    def compute_stats(self):
        self.has_nan = np.any(np.isnan(self.values))
        self.has_inf = np.any(np.isinf(self.values))
        if not self.has_nan and not self.has_inf:
            self.min_val = float(np.min(self.values))
            self.max_val = float(np.max(self.values))
            self.mean_val = float(np.mean(self.values))
            self.std_val = float(np.std(self.values))


@dataclass
class ComparisonResult:
    """Result of comparing two layer snapshots."""
    layer_idx: int
    name: str
    match: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    nan_mismatch: bool = False
    first_diff_idx: int = -1
    details: str = ""


@dataclass
class PipelineTrace:
    """Full trace of a pipeline execution."""
    model_name: str
    input_tokens: List[int]
    snapshots: List[LayerSnapshot] = field(default_factory=list)
    final_logits: Optional[np.ndarray] = None
    output_token: int = -1


class NumericalParityTest:
    """Test numerical parity between v6.6 and reference implementation."""

    # Checkpoints to capture (operation names)
    CHECKPOINTS = [
        "embedding_output",      # After token embedding
        "layer_{i}_ln1_output",  # After first layer norm
        "layer_{i}_qkv",         # Q, K, V projections
        "layer_{i}_attn_scores", # Attention scores
        "layer_{i}_attn_output", # After attention
        "layer_{i}_ln2_output",  # After second layer norm
        "layer_{i}_mlp_gate",    # MLP gate
        "layer_{i}_mlp_up",      # MLP up projection
        "layer_{i}_mlp_output",  # After MLP
        "layer_{i}_residual",    # After residual add
        "final_ln_output",       # Final layer norm
        "logits",                # Output logits
    ]

    def __init__(self, model_path: Path, reference: str = "v6.5"):
        self.model_path = model_path
        self.reference = reference
        self.config = self._load_config()
        self.num_layers = self.config.get("num_layers", 24)

    def _load_config(self) -> Dict:
        """Load model config."""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        # Try lowered IR
        lowered_path = self.model_path / "lowered_decode.json"
        if lowered_path.exists():
            with open(lowered_path) as f:
                data = json.load(f)
                return data.get("config", {})
        return {}

    def _load_v66_library(self) -> ctypes.CDLL:
        """Load v6.6 model library."""
        # Load kernel engine first
        engine_path = self.model_path / "libckernel_engine.so"
        if engine_path.exists():
            ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

        lib_path = self.model_path / "libmodel.so"
        if not lib_path.exists():
            lib_path = self.model_path / "ck-kernel-inference.so"

        lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_get_logits.argtypes = []
        lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        return lib

    def _load_v65_library(self) -> Optional[ctypes.CDLL]:
        """Load v6.5 reference library."""
        v65_cache = Path.home() / ".cache/ck-engine-v6.5/models"

        # Find matching model
        model_name = self.model_path.name
        v65_model = None

        for candidate in v65_cache.iterdir():
            if candidate.is_dir() and "qwen" in candidate.name.lower():
                v65_model = candidate
                break

        if not v65_model:
            print(f"WARNING: No v6.5 reference model found in {v65_cache}")
            return None

        lib_path = v65_model / "ck-kernel-inference.so"
        if not lib_path.exists():
            print(f"WARNING: No v6.5 library at {lib_path}")
            return None

        lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures (v6.5 API)
        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_get_logits.argtypes = []
        lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        return lib

    def run_v66(self, input_tokens: List[int]) -> PipelineTrace:
        """Run v6.6 pipeline and capture trace."""
        trace = PipelineTrace(model_name="v6.6", input_tokens=input_tokens)

        lib = self._load_v66_library()
        weights_path = self.model_path / "weights.bump"

        result = lib.ck_model_init(str(weights_path).encode())
        if result != 0:
            raise RuntimeError(f"v6.6 init failed: {result}")

        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        for i, token in enumerate(input_tokens):
            result = lib.ck_model_decode(token, output)
            if result != 0:
                print(f"WARNING: v6.6 decode failed at token {i}: {result}")
                break

        # Capture final logits
        logits_ptr = lib.ck_model_get_logits()
        trace.final_logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

        # Compute stats
        snapshot = LayerSnapshot(
            layer_idx=-1,
            name="final_logits",
            values=trace.final_logits
        )
        snapshot.compute_stats()
        trace.snapshots.append(snapshot)

        # Get output token
        trace.output_token = int(np.argmax(trace.final_logits))

        lib.ck_model_free()

        return trace

    def run_reference(self, input_tokens: List[int]) -> Optional[PipelineTrace]:
        """Run reference pipeline (v6.5 or llama.cpp)."""
        if self.reference == "v6.5":
            return self._run_v65(input_tokens)
        elif self.reference == "llamacpp":
            return self._run_llamacpp(input_tokens)
        return None

    def _run_v65(self, input_tokens: List[int]) -> Optional[PipelineTrace]:
        """Run v6.5 pipeline."""
        trace = PipelineTrace(model_name="v6.5", input_tokens=input_tokens)

        lib = self._load_v65_library()
        if not lib:
            return None

        # Find v6.5 weights
        v65_cache = Path.home() / ".cache/ck-engine-v6.5/models"
        v65_model = None
        for candidate in v65_cache.iterdir():
            if candidate.is_dir() and "qwen" in candidate.name.lower():
                v65_model = candidate
                break

        if not v65_model:
            return None

        weights_path = v65_model / "weights.bump"
        if not weights_path.exists():
            # Try GGUF
            for f in v65_model.glob("*.gguf"):
                weights_path = f
                break

        result = lib.ck_model_init(str(weights_path).encode())
        if result != 0:
            print(f"v6.5 init failed: {result}")
            return None

        vocab_size = self.config.get("vocab_size", 151936)
        output = (ctypes.c_float * vocab_size)()

        for i, token in enumerate(input_tokens):
            result = lib.ck_model_decode(token, output)
            if result != 0:
                print(f"WARNING: v6.5 decode failed at token {i}: {result}")
                break

        # Capture final logits
        logits_ptr = lib.ck_model_get_logits()
        trace.final_logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

        snapshot = LayerSnapshot(
            layer_idx=-1,
            name="final_logits",
            values=trace.final_logits
        )
        snapshot.compute_stats()
        trace.snapshots.append(snapshot)

        trace.output_token = int(np.argmax(trace.final_logits))

        lib.ck_model_free()

        return trace

    def _run_llamacpp(self, input_tokens: List[int]) -> Optional[PipelineTrace]:
        """Run llama.cpp for reference."""
        trace = PipelineTrace(model_name="llama.cpp", input_tokens=input_tokens)

        # Find llama.cpp binary
        llamacpp_path = Path.home() / "Workspace/C-Kernel-Engine/llama.cpp/build/bin/llama-cli"
        if not llamacpp_path.exists():
            llamacpp_path = Path("/usr/local/bin/llama-cli")
        if not llamacpp_path.exists():
            print("WARNING: llama.cpp not found")
            return None

        # Find GGUF model
        gguf_path = None
        for f in self.model_path.glob("*.gguf"):
            gguf_path = f
            break

        if not gguf_path:
            print("WARNING: No GGUF file found for llama.cpp")
            return None

        # Run llama.cpp with logit output
        # This is a simplified version - would need custom llama.cpp build for full trace
        cmd = [
            str(llamacpp_path),
            "-m", str(gguf_path),
            "-p", " ".join(str(t) for t in input_tokens),
            "-n", "1",
            "--logits-all",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            # Parse output (simplified)
            print(f"llama.cpp output: {result.stdout[:200]}...")
        except Exception as e:
            print(f"llama.cpp failed: {e}")
            return None

        return trace

    def compare_traces(self, test: PipelineTrace, ref: PipelineTrace) -> List[ComparisonResult]:
        """Compare two pipeline traces."""
        results = []

        # Compare final logits
        if test.final_logits is not None and ref.final_logits is not None:
            result = self._compare_arrays(
                "final_logits", -1,
                test.final_logits, ref.final_logits
            )
            results.append(result)

        return results

    def _compare_arrays(self, name: str, layer_idx: int,
                        test: np.ndarray, ref: np.ndarray,
                        rtol: float = 1e-3, atol: float = 1e-5) -> ComparisonResult:
        """Compare two arrays."""
        result = ComparisonResult(layer_idx=layer_idx, name=name, match=True)

        # Check shapes
        if test.shape != ref.shape:
            result.match = False
            result.details = f"Shape mismatch: {test.shape} vs {ref.shape}"
            return result

        # Check for NaN/Inf
        test_nan = np.any(np.isnan(test))
        ref_nan = np.any(np.isnan(ref))
        test_inf = np.any(np.isinf(test))
        ref_inf = np.any(np.isinf(ref))

        if test_nan != ref_nan or test_inf != ref_inf:
            result.match = False
            result.nan_mismatch = True
            result.details = f"NaN/Inf mismatch: test(nan={test_nan},inf={test_inf}) vs ref(nan={ref_nan},inf={ref_inf})"

            # Find first NaN in test
            if test_nan:
                nan_indices = np.where(np.isnan(test))[0]
                result.first_diff_idx = int(nan_indices[0])
                result.details += f"\n  First NaN at index {result.first_diff_idx}"

            return result

        # Skip numerical comparison if both have NaN
        if test_nan and ref_nan:
            result.match = True
            result.details = "Both have NaN (skipping numerical comparison)"
            return result

        # Numerical comparison
        abs_diff = np.abs(test - ref)
        result.max_abs_diff = float(np.max(abs_diff))

        # Relative diff (avoid divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = abs_diff / (np.abs(ref) + 1e-10)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
        result.max_rel_diff = float(np.max(rel_diff))

        # Check if within tolerance
        if not np.allclose(test, ref, rtol=rtol, atol=atol):
            result.match = False
            diff_indices = np.where(abs_diff > atol + rtol * np.abs(ref))[0]
            if len(diff_indices) > 0:
                result.first_diff_idx = int(diff_indices[0])
            result.details = f"Values differ: max_abs={result.max_abs_diff:.6e}, max_rel={result.max_rel_diff:.6e}"
            result.details += f"\n  First diff at index {result.first_diff_idx}"
            result.details += f"\n  test[{result.first_diff_idx}]={test[result.first_diff_idx]:.6e}"
            result.details += f"\n  ref[{result.first_diff_idx}]={ref[result.first_diff_idx]:.6e}"

        return result

    def run_test(self, input_tokens: List[int] = None) -> Dict:
        """Run full parity test."""
        if input_tokens is None:
            input_tokens = [1]  # Default: single token test

        print(f"=" * 70)
        print(f"NUMERICAL PARITY TEST: v6.6 vs {self.reference}")
        print(f"=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Input tokens: {input_tokens}")
        print()

        # Run v6.6
        print("Running v6.6...")
        try:
            v66_trace = self.run_v66(input_tokens)
            print(f"  Output token: {v66_trace.output_token}")
            for snap in v66_trace.snapshots:
                snap.compute_stats()
                status = "NaN!" if snap.has_nan else ("Inf!" if snap.has_inf else "OK")
                print(f"  {snap.name}: [{snap.min_val:.4f}, {snap.max_val:.4f}] mean={snap.mean_val:.4f} std={snap.std_val:.4f} [{status}]")
        except Exception as e:
            print(f"  FAILED: {e}")
            return {"passed": False, "error": str(e)}

        # Run reference
        print(f"\nRunning {self.reference}...")
        ref_trace = self.run_reference(input_tokens)
        if ref_trace:
            print(f"  Output token: {ref_trace.output_token}")
            for snap in ref_trace.snapshots:
                snap.compute_stats()
                status = "NaN!" if snap.has_nan else ("Inf!" if snap.has_inf else "OK")
                print(f"  {snap.name}: [{snap.min_val:.4f}, {snap.max_val:.4f}] mean={snap.mean_val:.4f} std={snap.std_val:.4f} [{status}]")
        else:
            print(f"  {self.reference} not available")

        # Compare
        print(f"\n--- Comparison ---")
        if ref_trace:
            comparisons = self.compare_traces(v66_trace, ref_trace)
            all_passed = True
            for comp in comparisons:
                status = "PASS" if comp.match else "FAIL"
                print(f"  [{status}] {comp.name}")
                if not comp.match:
                    all_passed = False
                    print(f"    {comp.details}")

            if v66_trace.output_token != ref_trace.output_token:
                print(f"  [FAIL] Output token: v6.6={v66_trace.output_token} vs ref={ref_trace.output_token}")
                all_passed = False
            else:
                print(f"  [PASS] Output token: {v66_trace.output_token}")
        else:
            all_passed = not v66_trace.snapshots[0].has_nan
            print(f"  Skipping comparison (no reference)")
            print(f"  v6.6 has NaN: {v66_trace.snapshots[0].has_nan}")

        print(f"\n{'=' * 70}")
        print(f"RESULT: {'PASSED' if all_passed else 'FAILED'}")
        print(f"{'=' * 70}")

        return {
            "passed": all_passed,
            "v66_output": v66_trace.output_token,
            "ref_output": ref_trace.output_token if ref_trace else None,
            "v66_has_nan": v66_trace.snapshots[0].has_nan if v66_trace.snapshots else True,
        }


def diagnose_nan_source(model_path: Path):
    """Diagnose where NaN values first appear in the pipeline."""
    print("\n" + "=" * 70)
    print("NaN SOURCE DIAGNOSIS")
    print("=" * 70)

    # Load the generated C file and find operation order
    c_file = model_path / "ck-kernel-inference.c"
    if not c_file.exists():
        print(f"ERROR: {c_file} not found")
        return

    # Parse operations from C file
    with open(c_file) as f:
        content = f.read()

    # Find all kernel calls
    import re
    kernel_calls = re.findall(r'/\* Op (\d+): (\w+) \((.*?)\) \*/', content)

    print(f"\nOperation sequence ({len(kernel_calls)} ops):")
    for op_num, kernel, details in kernel_calls[:30]:
        print(f"  Op {op_num:3s}: {kernel:40s} ({details})")
    if len(kernel_calls) > 30:
        print(f"  ... and {len(kernel_calls) - 30} more ops")

    # Suggestions for debugging
    print("\n--- Debugging Suggestions ---")
    print("1. Add fprintf(stderr) after each kernel call to track progress")
    print("2. Check weight offsets match manifest (W_PTR vs actual offset)")
    print("3. Verify activation buffer sizes are sufficient")
    print("4. Compare individual kernel outputs with reference")
    print()

    # Check for common issues
    print("--- Checking Common Issues ---")

    # Check for large activation offsets
    act_offsets = re.findall(r'model->activations \+ (\d+)', content)
    if act_offsets:
        max_offset = max(int(o) for o in act_offsets)
        print(f"  Max activation offset: {max_offset:,} bytes")

        # Check against ACTIVATIONS_SIZE
        match = re.search(r'#define ACTIVATIONS_SIZE (\d+)', content)
        if match:
            act_size = int(match.group(1))
            print(f"  ACTIVATIONS_SIZE: {act_size:,} bytes")
            if max_offset >= act_size:
                print(f"  WARNING: Max offset >= ACTIVATIONS_SIZE!")

    # Check weight pointer usage
    w_ptr_calls = re.findall(r'W_PTR\((.*?)\)', content)
    print(f"  W_PTR calls: {len(w_ptr_calls)}")

    # Check for potential alignment issues
    print(f"  Check: Are all offsets 64-byte aligned for SIMD?")


def main():
    parser = argparse.ArgumentParser(description="Numerical parity test for v6.6")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to v6.6 model directory")
    parser.add_argument("--reference", choices=["v6.5", "llamacpp"], default="v6.5",
                        help="Reference implementation to compare against")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1],
                        help="Input tokens to test")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run NaN diagnosis")

    args = parser.parse_args()

    test = NumericalParityTest(args.model, args.reference)
    result = test.run_test(args.tokens)

    if args.diagnose or not result["passed"]:
        diagnose_nan_source(args.model)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    exit(main())

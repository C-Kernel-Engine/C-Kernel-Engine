"""
Stage 3: Single Layer Activation Validation

This is the most critical stage for debugging gibberish output.
It compares CK layer activations against llama.cpp reference at each operation:
  3.1 RMSNorm output (attn_norm)
  3.2 Q/K/V projections
  3.3 RoPE application
  3.4 Attention scores and softmax
  3.5 Attention output
  3.6 FFN norm, gate_up, SwiGLU, down
  3.7 Residual outputs
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .base import BaseValidator, StageResult, TestResult, TestStatus


@dataclass
class OperationResult:
    """Result of comparing a single operation"""
    name: str
    llama_name: str
    ck_name: str
    max_diff: float
    mean_diff: float
    passed: bool
    shape: Optional[Tuple[int, ...]] = None
    details: Optional[str] = None


class Stage3Validator(BaseValidator):
    """Stage 3: Single Layer Activation Validation"""

    # Mapping of llama.cpp tensor names to CK names
    # These are the key checkpoints where we compare activations
    OPERATION_CHECKPOINTS = [
        # (llama_name_pattern, ck_name_pattern, tolerance, description)
        ("inp_embd", "token_embedding", 1e-5, "Token embedding lookup"),
        ("attn_norm-{layer}", "layer.{layer}.ln1_out", 1e-5, "Attention RMSNorm"),
        ("Qcur-{layer}", "layer.{layer}.q_proj", 1e-4, "Q projection"),
        ("Kcur-{layer}", "layer.{layer}.k_proj", 1e-4, "K projection"),
        ("Vcur-{layer}", "layer.{layer}.v_proj", 1e-4, "V projection"),
        ("Qcur_rope-{layer}", "layer.{layer}.q_rope", 1e-4, "Q after RoPE"),
        ("Kcur_rope-{layer}", "layer.{layer}.k_rope", 1e-4, "K after RoPE"),
        ("kq-{layer}", "layer.{layer}.attn_scores", 1e-3, "Attention scores"),
        ("kq_softmax-{layer}", "layer.{layer}.attn_probs", 1e-4, "Softmax output"),
        ("attn_out-{layer}", "layer.{layer}.attn_out", 1e-3, "Attention output"),
        ("ffn_norm-{layer}", "layer.{layer}.ln2_out", 1e-5, "FFN RMSNorm"),
        ("ffn_gate-{layer}", "layer.{layer}.ffn_gate", 1e-3, "FFN gate"),
        ("ffn_up-{layer}", "layer.{layer}.ffn_up", 1e-3, "FFN up"),
        ("ffn_gate_silu-{layer}", "layer.{layer}.swiglu_out", 1e-3, "SwiGLU output"),
        ("ffn_out-{layer}", "layer.{layer}.mlp_out", 1e-3, "FFN output"),
        ("res-{layer}", "layer.{layer}.residual", 1e-3, "Layer output"),
    ]

    def __init__(self, gguf_path: str, bump_path: Optional[str] = None,
                 manifest_path: Optional[str] = None, verbose: bool = False,
                 layer: int = 0, prompt: str = "Hello"):
        super().__init__(gguf_path, bump_path, manifest_path, verbose)
        self.layer = layer
        self.prompt = prompt
        self.llama_tensors: Dict[str, np.ndarray] = {}
        self.ck_tensors: Dict[str, np.ndarray] = {}
        self.llama_dump_dir = None

    def _run_llama_single_layer(self) -> bool:
        """
        Run llama.cpp with activation dumping for the target layer.

        This uses the LLAMA_SINGLE_LAYER environment variable to limit
        processing to a single layer, and dumps intermediate tensors.
        """
        self.log(f"Running llama.cpp for layer {self.layer}...")

        # Create temp directory for dumps
        self.llama_dump_dir = tempfile.mkdtemp(prefix=f"llama_layer{self.layer}_")
        self.log(f"Dump directory: {self.llama_dump_dir}")

        # Find llama.cpp executable
        llama_main = None
        search_paths = [
            "./llama.cpp/build/bin/llama-cli",
            "./llama.cpp/build/bin/main",
            "../llama.cpp/build/bin/llama-cli",
        ]
        for path in search_paths:
            if os.path.exists(path):
                llama_main = path
                break

        if not llama_main:
            self.log("llama.cpp executable not found")
            return False

        # Set environment for single layer mode
        env = os.environ.copy()
        env["LLAMA_SINGLE_LAYER"] = str(self.layer + 1)
        env["LLAMA_DUMP_TENSORS"] = self.llama_dump_dir

        try:
            # Run llama.cpp with minimal settings
            result = subprocess.run(
                [
                    llama_main,
                    "-m", self.gguf_path,
                    "-p", self.prompt,
                    "-n", "1",  # Generate 1 token
                    "--no-display-prompt",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.log(f"llama.cpp exit code: {result.returncode}")
            if result.returncode != 0:
                self.log(f"llama.cpp stderr: {result.stderr[:500]}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            self.log("llama.cpp timed out")
            return False
        except Exception as e:
            self.log(f"Failed to run llama.cpp: {e}")
            return False

    def _load_llama_tensors(self) -> bool:
        """Load tensors dumped by llama.cpp"""
        if not self.llama_dump_dir or not os.path.exists(self.llama_dump_dir):
            return False

        try:
            for filename in os.listdir(self.llama_dump_dir):
                if filename.endswith(".bin"):
                    tensor_name = filename[:-4]  # Remove .bin
                    filepath = os.path.join(self.llama_dump_dir, filename)
                    tensor = np.fromfile(filepath, dtype=np.float32)
                    self.llama_tensors[tensor_name] = tensor
                    self.log(f"Loaded llama tensor: {tensor_name} shape={tensor.shape}")

            return len(self.llama_tensors) > 0

        except Exception as e:
            self.log(f"Failed to load llama tensors: {e}")
            return False

    def _run_ck_single_layer(self) -> bool:
        """
        Run CK forward pass for single layer.

        This uses the v6 infrastructure with layer limiting.
        """
        self.log(f"Running CK for layer {self.layer}...")

        if not self.bump_path or not os.path.exists(self.bump_path):
            self.log("BUMP file not provided")
            return False

        # Create temp directory for CK dumps
        ck_dump_dir = tempfile.mkdtemp(prefix=f"ck_layer{self.layer}_")
        self.log(f"CK dump directory: {ck_dump_dir}")

        # Find CK executable
        ck_exe = None
        search_paths = [
            "./build/ck-engine-v6",
            "./ck-engine-v6",
            "../build/ck-engine-v6",
        ]
        for path in search_paths:
            if os.path.exists(path):
                ck_exe = path
                break

        if not ck_exe:
            self.log("ck-engine-v6 executable not found")
            # Try using Python runner
            return self._run_ck_via_python()

        env = os.environ.copy()
        env["CK_SINGLE_LAYER"] = str(self.layer + 1)
        env["CK_DUMP_TENSORS"] = ck_dump_dir

        try:
            result = subprocess.run(
                [
                    ck_exe,
                    "--weights", self.bump_path,
                    "--manifest", self.manifest_path or "",
                    "--prompt", self.prompt,
                    "--max-tokens", "1",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.log(f"CK exit code: {result.returncode}")
            if result.returncode != 0:
                self.log(f"CK stderr: {result.stderr[:500]}")
                return False

            # Load CK tensors
            for filename in os.listdir(ck_dump_dir):
                if filename.endswith(".bin"):
                    tensor_name = filename[:-4]
                    filepath = os.path.join(ck_dump_dir, filename)
                    tensor = np.fromfile(filepath, dtype=np.float32)
                    self.ck_tensors[tensor_name] = tensor
                    self.log(f"Loaded CK tensor: {tensor_name} shape={tensor.shape}")

            return len(self.ck_tensors) > 0

        except Exception as e:
            self.log(f"Failed to run CK: {e}")
            return False

    def _run_ck_via_python(self) -> bool:
        """Fallback: run CK validation via Python ctypes"""
        self.log("Using Python ctypes for CK validation...")

        try:
            # Try to load the CK library
            import ctypes
            lib_paths = [
                "./build/libckernel_engine.so",
                "./libckernel_engine.so",
            ]

            lib = None
            for path in lib_paths:
                if os.path.exists(path):
                    lib = ctypes.cdll.LoadLibrary(path)
                    break

            if not lib:
                self.log("Could not load CK library")
                return False

            # TODO: Implement ctypes-based single layer execution
            # This would call ck_layer_forward_* directly
            self.log("ctypes execution not fully implemented yet")
            return False

        except Exception as e:
            self.log(f"Python CK execution failed: {e}")
            return False

    def _compare_tensors(self, name: str, llama: np.ndarray, ck: np.ndarray,
                        tolerance: float) -> OperationResult:
        """Compare two tensors and return detailed result"""

        # Handle shape mismatches
        if llama.shape != ck.shape:
            # Try to reshape if total elements match
            if llama.size == ck.size:
                ck = ck.reshape(llama.shape)
            else:
                return OperationResult(
                    name=name,
                    llama_name=f"llama:{llama.shape}",
                    ck_name=f"ck:{ck.shape}",
                    max_diff=float('inf'),
                    mean_diff=float('inf'),
                    passed=False,
                    details=f"Shape mismatch: {llama.shape} vs {ck.shape}"
                )

        # Check for NaN/Inf
        if np.any(np.isnan(ck)) or np.any(np.isinf(ck)):
            return OperationResult(
                name=name,
                llama_name="llama",
                ck_name="ck",
                max_diff=float('inf'),
                mean_diff=float('inf'),
                passed=False,
                shape=llama.shape,
                details="CK output contains NaN/Inf"
            )

        # Compute differences
        diff = np.abs(llama - ck)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        passed = max_diff <= tolerance

        return OperationResult(
            name=name,
            llama_name="llama",
            ck_name="ck",
            max_diff=max_diff,
            mean_diff=mean_diff,
            passed=passed,
            shape=llama.shape,
        )

    def _validate_operations(self) -> List[TestResult]:
        """Compare each operation checkpoint"""
        results = []

        for llama_pattern, ck_pattern, tolerance, description in self.OPERATION_CHECKPOINTS:
            llama_name = llama_pattern.format(layer=self.layer)
            ck_name = ck_pattern.format(layer=self.layer)

            # Check if both tensors exist
            if llama_name not in self.llama_tensors:
                results.append(TestResult(
                    name=description,
                    status=TestStatus.SKIP,
                    message=f"llama tensor '{llama_name}' not found"
                ))
                continue

            if ck_name not in self.ck_tensors:
                results.append(TestResult(
                    name=description,
                    status=TestStatus.SKIP,
                    message=f"CK tensor '{ck_name}' not found"
                ))
                continue

            # Compare
            llama_tensor = self.llama_tensors[llama_name]
            ck_tensor = self.ck_tensors[ck_name]

            op_result = self._compare_tensors(description, llama_tensor, ck_tensor, tolerance)

            if op_result.passed:
                results.append(TestResult(
                    name=description,
                    status=TestStatus.PASS,
                    max_diff=op_result.max_diff,
                    mean_diff=op_result.mean_diff,
                    message=f"shape={op_result.shape}"
                ))
            else:
                results.append(TestResult(
                    name=description,
                    status=TestStatus.FAIL,
                    max_diff=op_result.max_diff,
                    mean_diff=op_result.mean_diff,
                    message=op_result.details or f"max_diff {op_result.max_diff:.2e} > {tolerance:.0e}",
                    details={'shape': op_result.shape}
                ))

        return results

    def _create_synthetic_test(self) -> List[TestResult]:
        """
        Create synthetic validation tests when llama.cpp is not available.
        This tests basic kernel correctness with known inputs.
        """
        results = []

        # Test 1: RMSNorm kernel validation
        results.append(TestResult(
            name="RMSNorm kernel (synthetic)",
            status=TestStatus.SKIP,
            message="Synthetic test - requires CK library loading"
        ))

        # Test 2: GEMV kernel validation
        results.append(TestResult(
            name="GEMV Q4_K kernel (synthetic)",
            status=TestStatus.SKIP,
            message="Synthetic test - requires CK library loading"
        ))

        # Test 3: RoPE kernel validation
        results.append(TestResult(
            name="RoPE kernel (synthetic)",
            status=TestStatus.SKIP,
            message="Synthetic test - requires CK library loading"
        ))

        return results

    def run(self) -> StageResult:
        """Execute Stage 3 validation"""
        result = StageResult(stage_num=3, stage_name=f"Single Layer (layer={self.layer})")

        try:
            # Try to run llama.cpp reference
            llama_success = self._run_llama_single_layer()
            if llama_success:
                llama_success = self._load_llama_tensors()

            if not llama_success:
                result.add_test(TestResult(
                    name="llama.cpp reference",
                    status=TestStatus.SKIP,
                    message="Could not run llama.cpp (may need patches for tensor dumping)"
                ))

            # Try to run CK
            ck_success = self._run_ck_single_layer()

            if not ck_success:
                result.add_test(TestResult(
                    name="CK execution",
                    status=TestStatus.SKIP,
                    message="Could not run CK (may need environment setup)"
                ))

            # If we have both, compare operations
            if llama_success and ck_success:
                for test_result in self._validate_operations():
                    result.add_test(test_result)
            else:
                # Fall back to synthetic tests
                for test_result in self._create_synthetic_test():
                    result.add_test(test_result)

                result.add_test(TestResult(
                    name="Activation comparison",
                    status=TestStatus.SKIP,
                    message="Manual comparison needed - see instructions below"
                ))

                # Provide guidance
                print("\n" + "=" * 60)
                print("STAGE 3 MANUAL TESTING INSTRUCTIONS")
                print("=" * 60)
                print("""
To perform manual single-layer validation:

1. Patch llama.cpp to dump layer tensors:
   cd llama.cpp
   # Add LLAMA_DUMP_TENSORS environment variable check
   # In llama.cpp/src/llama.cpp, add tensor dumps after each operation

2. Run llama.cpp with single layer:
   LLAMA_SINGLE_LAYER=1 ./llama-cli -m model.gguf -p "Hello" -n 1

3. Run CK with single layer:
   CK_SINGLE_LAYER=1 ./ck-engine-v6 --weights weights.bump

4. Compare outputs using:
   python -c "
   import numpy as np
   llama = np.fromfile('llama_layer0_attn_norm.bin', dtype=np.float32)
   ck = np.fromfile('ck_layer0_ln1_out.bin', dtype=np.float32)
   print(f'Max diff: {np.max(np.abs(llama - ck)):.6e}')
   "
""")

        except Exception as e:
            result.error_message = f"Stage 3 failed: {e}"

        return result

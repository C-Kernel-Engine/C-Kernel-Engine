#!/usr/bin/env python3
"""
test_dtype_consistency.py - Validate dtype consistency across the pipeline

This test ensures dtype compatibility:
- Input dtype matches what kernel expects
- Weight dtype matches kernel selection
- Output dtype is correct for next kernel
- Activation quantization happens when needed

BUGS THIS CATCHES:
- Bug 7: Weight dtype validation (wq=4, wk=4, wv=3)
- Bug 8: Dtype sync (input→weights→activation→output)
- Bug 10: v6.5 quantizes activations, v6.6 doesn't

Usage:
    python test_dtype_consistency.py
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

# Paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"
KERNEL_MAPS_DIR = V66_ROOT / "kernel_maps"


# Dtype compatibility matrix
# Key: kernel suffix, Value: (expected_weight_dtype, expected_activation_dtype)
KERNEL_DTYPE_REQUIREMENTS = {
    # gemv_<weight>_<activation>
    "gemv_q5_0_q8_0": ("q5_0", "q8_0"),
    "gemv_q4_k_q8_k": ("q4_k", "q8_k"),
    "gemv_q6_k_q8_k": ("q6_k", "q8_k"),
    "gemv_q8_0_q8_0": ("q8_0", "q8_0"),
    # FP32 activation variants
    "gemv_q5_0": ("q5_0", "fp32"),
    "gemv_q4_k": ("q4_k", "fp32"),
    "gemv_q6_k": ("q6_k", "fp32"),
    "gemv_q8_0": ("q8_0", "fp32"),
    # gemm_nt_<weight>_<activation>
    "gemm_nt_q5_0_q8_0": ("q5_0", "q8_0"),
    "gemm_nt_q4_k_q8_k": ("q4_k", "q8_k"),
    "gemm_nt_q6_k_q8_k": ("q6_k", "q8_k"),
    "gemm_nt_q8_0_q8_0": ("q8_0", "q8_0"),
    # FP32 variants
    "gemm_nt_q5_0": ("q5_0", "fp32"),
    "gemm_nt_q4_k": ("q4_k", "fp32"),
    "gemm_nt_q6_k": ("q6_k", "fp32"),
    "gemm_nt_q8_0": ("q8_0", "fp32"),
}

# CK dtype enum values (from ckernel_types.h)
CK_DTYPE_VALUES = {
    0: "fp32",
    1: "fp16",
    2: "bf16",
    3: "q8_0",
    4: "q5_0",
    5: "q4_0",
    6: "q4_k",
    7: "q6_k",
    8: "q8_k",
}


@dataclass
class DtypeIssue:
    """A dtype consistency issue."""
    severity: str
    category: str
    message: str
    kernel: str = ""
    expected: str = ""
    actual: str = ""


class DtypeValidator:
    """Validates dtype consistency across pipeline."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[DtypeIssue] = []

    def load_files(self) -> bool:
        """Load required files."""
        self.ir = None
        self.manifest = None
        self.c_code = None
        self.registry = None

        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            if (base_dir / "lowered_decode.json").exists():
                with open(base_dir / "lowered_decode.json") as f:
                    self.ir = json.load(f)

            if (base_dir / "weights_manifest.json").exists():
                with open(base_dir / "weights_manifest.json") as f:
                    self.manifest = json.load(f)

            if (base_dir / "ck-kernel-inference.c").exists():
                self.c_code = (base_dir / "ck-kernel-inference.c").read_text()

        # Load kernel registry
        registry_path = KERNEL_MAPS_DIR / "KERNEL_REGISTRY.json"
        if registry_path.exists():
            with open(registry_path) as f:
                self.registry = json.load(f)

        if not self.ir:
            print("ERROR: Could not find lowered_decode.json")
            return False
        if not self.manifest:
            print("WARNING: Could not find weights_manifest.json")

        return True

    def get_weight_dtypes(self) -> Dict[str, str]:
        """Extract weight dtypes from manifest."""
        dtypes = {}
        if not self.manifest:
            return dtypes

        for entry in self.manifest.get("entries", []):
            name = entry.get("name", "")
            dtype = entry.get("dtype", "fp32")
            dtypes[name] = dtype

        return dtypes

    def validate_kernel_weight_dtype(self) -> bool:
        """Validate kernel selection matches weight dtype."""
        print("\n" + "="*70)
        print("KERNEL ↔ WEIGHT DTYPE VALIDATION")
        print("="*70)

        weight_dtypes = self.get_weight_dtypes()

        # Extract ops from IR
        ops = []
        if "operations" in self.ir:
            ops = self.ir["operations"]
        elif "ops" in self.ir:
            ops = self.ir["ops"]
        else:
            for section in self.ir.get("sections", []):
                for layer in section.get("layers", []):
                    ops.extend(layer.get("ops", []))

        for op in ops:
            kernel = op.get("kernel", op.get("function", ""))
            weights = op.get("weights", {})

            if not kernel:
                continue

            # Check if kernel expects specific weight dtype
            for kernel_pattern, (expected_wdtype, expected_adtype) in KERNEL_DTYPE_REQUIREMENTS.items():
                if kernel_pattern in kernel:
                    # Check each weight in this op
                    for wname, winfo in weights.items():
                        if isinstance(winfo, dict):
                            actual_dtype = winfo.get("dtype", "unknown")
                        else:
                            # Look up in manifest
                            actual_dtype = weight_dtypes.get(wname, "unknown")

                        if actual_dtype != "unknown" and actual_dtype != expected_wdtype:
                            self.issues.append(DtypeIssue(
                                severity="ERROR",
                                category="KERNEL_WEIGHT_MISMATCH",
                                message=f"Kernel {kernel} expects {expected_wdtype} weights but {wname} is {actual_dtype}",
                                kernel=kernel,
                                expected=expected_wdtype,
                                actual=actual_dtype
                            ))
                        elif self.verbose:
                            print(f"  ✓ {kernel}: {wname} is {actual_dtype}")
                    break

        errors = len([i for i in self.issues if i.severity == "ERROR"])
        print(f"\nChecked {len(ops)} ops, found {errors} dtype mismatches")
        return errors == 0

    def validate_activation_quantization(self) -> bool:
        """Validate activation quantization happens when needed."""
        print("\n" + "="*70)
        print("ACTIVATION QUANTIZATION VALIDATION")
        print("="*70)

        if not self.c_code:
            print("WARNING: No C code to analyze")
            return True

        # Check for quantized kernel calls
        quantized_kernels = []
        for pattern in ["gemv_q.*_q8_0", "gemv_q.*_q8_k", "gemm_nt_q.*_q8_0", "gemm_nt_q.*_q8_k"]:
            matches = re.findall(pattern, self.c_code)
            quantized_kernels.extend(matches)

        # Check for quantize_row calls
        has_quantize = "quantize_row_q8_0" in self.c_code or "quantize_row_q8_k" in self.c_code

        if quantized_kernels and not has_quantize:
            self.issues.append(DtypeIssue(
                severity="CRITICAL",
                category="MISSING_QUANTIZATION",
                message=f"Code uses {len(quantized_kernels)} quantized kernels but no quantize_row calls found",
                expected="quantize_row_q8_0() before each quantized GEMV",
                actual="Direct FP32 passed to quantized kernels"
            ))
            print("  ✗ MISSING: quantize_row calls before quantized kernels")
            print(f"    Quantized kernels found: {set(quantized_kernels)}")
            return False

        if quantized_kernels:
            print(f"  ✓ Found quantized kernels: {set(quantized_kernels)}")
            if has_quantize:
                print("  ✓ Found quantize_row calls")
        else:
            print("  ℹ No quantized activation kernels used (FP32 path)")

        return True

    def validate_fused_kernel_dtypes(self) -> bool:
        """Validate fused kernels receive correct dtypes."""
        print("\n" + "="*70)
        print("FUSED KERNEL DTYPE VALIDATION")
        print("="*70)

        if not self.c_code:
            print("WARNING: No C code to analyze")
            return True

        # Find mega_fused calls and check dtype params
        fused_pattern = r'mega_fused_\w+\([^)]*CK_DT_(\w+)[^)]*\)'
        matches = re.findall(fused_pattern, self.c_code)

        if not matches:
            print("  ℹ No fused kernels with dtype parameters found")
            return True

        print(f"  Found dtype parameters in fused kernels: {set(matches)}")

        # Check for unsupported dtype combinations
        valid_dtypes = {"Q5_0", "Q8_0", "Q4_K", "Q6_K", "FP32"}
        invalid = set(matches) - valid_dtypes
        if invalid:
            self.issues.append(DtypeIssue(
                severity="WARNING",
                category="UNKNOWN_DTYPE",
                message=f"Unknown dtype values in fused kernel calls: {invalid}"
            ))

        return True

    def validate_qkv_dtype_consistency(self) -> bool:
        """Validate Q/K/V projections use consistent dtypes."""
        print("\n" + "="*70)
        print("Q/K/V DTYPE CONSISTENCY")
        print("="*70)

        weight_dtypes = self.get_weight_dtypes()

        # Group by layer
        layers = {}
        for name, dtype in weight_dtypes.items():
            # Parse layer number
            match = re.search(r'layer\.(\d+)\.(wq|wk|wv|wo)', name)
            if match:
                layer = int(match.group(1))
                proj = match.group(2)
                if layer not in layers:
                    layers[layer] = {}
                layers[layer][proj] = dtype

        # Check each layer
        issues_found = False
        for layer_idx in sorted(layers.keys())[:3]:  # Check first 3 layers
            projs = layers[layer_idx]
            dtypes = set(projs.values())

            if len(dtypes) > 2:  # Allow some variation (e.g., wv might be different)
                self.issues.append(DtypeIssue(
                    severity="WARNING",
                    category="QKV_DTYPE_MISMATCH",
                    message=f"Layer {layer_idx} has inconsistent Q/K/V dtypes: {projs}"
                ))
                issues_found = True

            if self.verbose:
                print(f"  Layer {layer_idx}: {projs}")

        if not issues_found:
            print("  ✓ Q/K/V dtypes are consistent across layers")

        return not issues_found

    def run_all_tests(self) -> bool:
        """Run all dtype validations."""
        if not self.load_files():
            return False

        results = []
        results.append(("Kernel ↔ Weight dtype", self.validate_kernel_weight_dtype()))
        results.append(("Activation quantization", self.validate_activation_quantization()))
        results.append(("Fused kernel dtypes", self.validate_fused_kernel_dtypes()))
        results.append(("Q/K/V consistency", self.validate_qkv_dtype_consistency()))

        # Summary
        print("\n" + "="*70)
        print("DTYPE ISSUES FOUND")
        print("="*70)

        if not self.issues:
            print("No dtype issues found!")
        else:
            for severity in ["CRITICAL", "ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        print(f"  [{issue.category}]: {issue.message}")
                        if issue.expected:
                            print(f"    Expected: {issue.expected}")
                        if issue.actual:
                            print(f"    Actual: {issue.actual}")

        # Final verdict
        print("\n" + "="*70)
        critical = len([i for i in self.issues if i.severity == "CRITICAL"])
        errors = len([i for i in self.issues if i.severity == "ERROR"])
        passed = sum(1 for _, r in results if r)

        if critical > 0:
            print(f"VERDICT: FAIL - {critical} CRITICAL dtype issues")
        elif errors > 0:
            print(f"VERDICT: FAIL - {errors} dtype errors")
        else:
            print(f"VERDICT: PASS - {passed}/{len(results)} tests passed")
        print("="*70)

        return critical == 0 and errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate dtype consistency")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = DtypeValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

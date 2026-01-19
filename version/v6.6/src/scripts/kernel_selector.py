#!/usr/bin/env python3
"""
Kernel Selector for CK-Engine v6.6 Codegen

Demonstrates how codegen selects the best kernel variant based on:
1. Available hardware features (AVX, AVX2, AVX512, etc.)
2. Available parallelization (OpenMP)
3. Weight quantization format (Q4_K, Q6_K, etc.)
4. Mode (decode vs prefill)

Priority order:
  1. parallel_simd  (best: AVX + parallel + prefetch)
  2. parallel       (good: scalar parallel)
  3. simd           (okay: single-threaded AVX)
  4. ref            (fallback: pure C)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Kernel registry loaded from JSON
KERNEL_REGISTRY: Dict[str, dict] = {}

def load_kernel_maps(kernel_dir: Path) -> Dict[str, dict]:
    """Load all kernel definitions from JSON files."""
    registry = {}
    for json_file in kernel_dir.glob("*.json"):
        with open(json_file) as f:
            kernel = json.load(f)
            registry[kernel["name"]] = kernel
    return registry


def detect_hardware_features() -> Dict[str, bool]:
    """
    Detect available hardware features.
    In real codegen, this would check CPU flags.
    """
    # Example: i7-3630QM (Ivy Bridge)
    return {
        "sse": True,
        "sse4_1": True,
        "avx": True,
        "avx2": False,  # Ivy Bridge doesn't have AVX2
        "avx512": False,
        "vnni": False,
        "amx": False,
        "openmp": True,
        "f16c": True,  # FP16 conversion
    }


def select_best_variant(kernel_name: str,
                        features: Dict[str, bool],
                        mode: str = "decode") -> Tuple[str, str, List[str]]:
    """
    Select the best kernel variant based on available features.

    Returns: (variant_name, function_name, extra_params)
    """
    kernel = KERNEL_REGISTRY.get(kernel_name)
    if not kernel:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    variants = kernel.get("impl", {}).get("variants", {})

    # Sort variants by priority (lower = better)
    sorted_variants = sorted(
        variants.items(),
        key=lambda x: x[1].get("priority", 999)
    )

    for variant_name, variant in sorted_variants:
        requires = variant.get("requires", [])

        # Check if all requirements are met
        all_met = all(features.get(req, False) for req in requires)

        if all_met:
            return (
                variant_name,
                variant["forward"],
                variant.get("extra_params", [])
            )

    # Fallback to default
    return ("default", kernel["impl"]["forward"], [])


def generate_kernel_call(kernel_name: str,
                         features: Dict[str, bool],
                         mode: str = "decode") -> str:
    """
    Generate C code for the optimal kernel call.
    """
    variant_name, func_name, extra_params = select_best_variant(
        kernel_name, features, mode
    )

    kernel = KERNEL_REGISTRY[kernel_name]
    buffers = kernel.get("buffers", [])

    # Build parameter list
    params = []
    for buf in buffers:
        params.append(buf["id"])

    # Add dimension parameters
    params.extend(["M", "K"])

    # Add parallel parameters if needed
    if extra_params:
        params.extend(extra_params)

    param_str = ", ".join(params)

    # Generate code
    code = f"""
    /* Kernel: {kernel_name} (variant: {variant_name})
     * Speedup: {kernel['impl']['variants'].get(variant_name, {}).get('speedup', 'N/A')}
     */
    {func_name}({param_str});
"""
    return code


def generate_parallel_wrapper(kernel_name: str,
                              features: Dict[str, bool]) -> str:
    """
    Generate OpenMP wrapper for parallel kernel variants.
    """
    variant_name, func_name, extra_params = select_best_variant(
        kernel_name, features
    )

    if "ith" not in extra_params:
        # Not a parallel variant, no wrapper needed
        return generate_kernel_call(kernel_name, features)

    kernel = KERNEL_REGISTRY[kernel_name]
    hints = kernel.get("codegen_hints", {})
    optimal_threads = hints.get("optimal_threads", 4)

    code = f"""
    /* Parallel {kernel_name} with {variant_name} variant */
    #pragma omp parallel num_threads({optimal_threads})
    {{
        const int ith = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        {func_name}(y, W, x_q8, M, K, ith, nth);
    }}
"""
    return code


def demo_kernel_selection():
    """Demonstrate kernel selection for different scenarios."""

    # Load kernel maps
    global KERNEL_REGISTRY
    kernel_dir = Path(__file__).parent.parent.parent.parent / "kernel_maps" / "kernels"
    KERNEL_REGISTRY = load_kernel_maps(kernel_dir)

    print("=" * 70)
    print("CK-Engine v6.6 Kernel Selection Demo")
    print("=" * 70)

    # Scenario 1: i7-3630QM (Ivy Bridge) with AVX + OpenMP
    print("\n--- Scenario 1: i7-3630QM (AVX + OpenMP) ---")
    features_ivy = {
        "sse": True, "sse4_1": True, "avx": True,
        "avx2": False, "avx512": False, "openmp": True
    }

    if "gemv_q4k_q8k" in KERNEL_REGISTRY:
        variant, func, params = select_best_variant("gemv_q4k_q8k", features_ivy)
        print(f"Selected: {variant} -> {func}({', '.join(params) if params else 'standard'})")
        print(f"Code:\n{generate_parallel_wrapper('gemv_q4k_q8k', features_ivy)}")

    # Scenario 2: No OpenMP (single-threaded)
    print("\n--- Scenario 2: AVX but no OpenMP ---")
    features_no_omp = {
        "sse": True, "sse4_1": True, "avx": True,
        "avx2": False, "avx512": False, "openmp": False
    }

    if "gemv_q4k_q8k" in KERNEL_REGISTRY:
        variant, func, params = select_best_variant("gemv_q4k_q8k", features_no_omp)
        print(f"Selected: {variant} -> {func}")

    # Scenario 3: Pure reference (no SIMD, no OpenMP)
    print("\n--- Scenario 3: Pure reference (no SIMD, no OpenMP) ---")
    features_ref = {
        "sse": False, "avx": False, "openmp": False
    }

    if "gemv_q4k_q8k" in KERNEL_REGISTRY:
        variant, func, params = select_best_variant("gemv_q4k_q8k", features_ref)
        print(f"Selected: {variant} -> {func}")

    # Show full kernel hierarchy
    print("\n" + "=" * 70)
    print("Full Kernel Hierarchy for gemv_q4k_q8k:")
    print("=" * 70)

    if "gemv_q4k_q8k" in KERNEL_REGISTRY:
        variants = KERNEL_REGISTRY["gemv_q4k_q8k"]["impl"]["variants"]
        for name, v in sorted(variants.items(), key=lambda x: x[1]["priority"]):
            print(f"  {v['priority']}. {name:15} -> {v['forward']}")
            print(f"     Requires: {v['requires']}")
            print(f"     Speedup:  {v['speedup']}")
            print()


if __name__ == "__main__":
    demo_kernel_selection()

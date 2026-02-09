#!/usr/bin/env python3
"""
Roofline analysis for CK kernels.
Computes operational intensity and compares to theoretical hardware limits.
"""
import subprocess
import sys
import json
from typing import Dict, List, Optional


# Theoretical peaks for common CPUs (per core, in GFLOPS)
# FP32 theoretical peak = clock_ghz * cores * 2 (FMUL + FADD) * vector_width (32 for AVX-512)
CPU_PEAKS: Dict[str, Dict] = {
    "skylake": {
        "name": "Intel Xeon Skylake",
        "clock_ghz": 3.0,
        "cores": 4,
        "vector_width": 32,  # AVX-512
        "memory_gbps": 41.6,  # DDR4-3200 typical
        "notes": "AVX-512 capable"
    },
    "cascadelake": {
        "name": "Intel Xeon Cascade Lake",
        "clock_ghz": 2.9,
        "cores": 6,
        "vector_width": 32,
        "memory_gbps": 41.6,
        "notes": "AVX-512, VNNI for int8"
    },
    "icelake": {
        "name": "Intel Ice Lake",
        "clock_ghz": 3.2,
        "cores": 8,
        "vector_width": 32,
        "memory_gbps": 51.2,
        "notes": "AVX-512, VNNI"
    },
    "amd_zen2": {
        "name": "AMD Zen 2",
        "clock_ghz": 3.7,
        "cores": 8,
        "vector_width": 16,  # AVX2
        "memory_gbps": 51.2,
        "notes": "AVX2 (256-bit)"
    },
    "amd_zen3": {
        "name": "AMD Zen 3",
        "clock_ghz": 3.8,
        "cores": 8,
        "vector_width": 16,  # AVX2
        "memory_gbps": 51.2,
        "notes": "AVX2 (256-bit)"
    },
    "amd_zen4": {
        "name": "AMD Zen 4",
        "clock_ghz": 4.0,
        "cores": 12,
        "vector_width": 32,  # AVX-512
        "memory_gbps": 51.2,
        "notes": "AVX-512"
    },
    "apple_m1": {
        "name": "Apple M1",
        "clock_ghz": 3.2,
        "cores": 8,
        "vector_width": 16,  # NEON
        "memory_gbps": 68.3,  # LPDDR5
        "notes": "ARM NEON (128-bit)"
    },
    "apple_m2": {
        "name": "Apple M2",
        "clock_ghz": 3.5,
        "cores": 8,
        "vector_width": 16,
        "memory_gbps": 100.0,  # LPDDR5
        "notes": "ARM NEON (128-bit)"
    },
    "generic": {
        "name": "Generic x86-64",
        "clock_ghz": 2.5,
        "cores": 4,
        "vector_width": 16,
        "memory_gbps": 25.6,
        "notes": "Fallback for unknown CPUs"
    }
}


def get_cpu_info() -> str:
    """Detect CPU type from lscpu output."""
    result = subprocess.run(
        ["lscpu"], capture_output=True, text=True
    )

    cpu_info = result.stdout

    # Check for specific CPU models
    if "Xeon" in cpu_info:
        if "Platinum" in cpu_info or "Gold" in cpu_info or "Silver" in cpu_info:
            if "8275" in cpu_info or "8280" in cpu_info:
                return "cascadelake"
            return "skylake"  # Default Xeon
    elif "AMD" in cpu_info:
        if "Zen 4" in cpu_info or "7940" in cpu_info or "7950" in cpu_info:
            return "amd_zen4"
        elif "Zen 3" in cpu_info or "5900" in cpu_info or "5950" in cpu_info:
            return "amd_zen3"
        elif "Zen 2" in cpu_info or "3900" in cpu_info:
            return "amd_zen2"
        return "amd_zen3"  # Default AMD
    elif "Apple" in cpu_info or "M1" in cpu_info or "M2" in cpu_info:
        if "M2" in cpu_info or "M3" in cpu_info:
            return "apple_m2"
        return "apple_m1"
    elif "Intel" in cpu_info:
        if "i7-" in cpu_info or "i9-" in cpu_info:
            return "skylake"  # Similar to Xeon for our purposes
        return "skylake"

    return "generic"


def estimate_flops_per_byte(kernel: str, mode: str) -> float:
    """
    Estimate FLOPs/byte operational intensity for common kernels.

    Returns:
        Estimated AI (FLOPs per byte of data accessed)
    """
    kernel_lower = kernel.lower()

    # Quantized GEMV operations - very high AI due to low precision weights
    if "gemv" in kernel_lower:
        if "q8" in kernel_lower or "int8" in kernel_lower:
            # Q8_0: 1 byte weight, ~2 FLOPs multiply-add per weight
            # Plus dequantization overhead
            return 4.0  # Dequantize + compute
        elif "q6" in kernel_lower:
            return 3.0
        elif "q5" in kernel_lower:
            return 2.5
        elif "q4" in kernel_lower:
            return 2.0
        return 2.0

    # GEMM operations - typically higher AI
    if "gemm" in kernel_lower:
        return 8.0  # Matmul with accumulation

    # Attention kernels - mix of memory and compute
    if "attention" in kernel_lower or "attn" in kernel_lower:
        return 5.0

    # MLP/FFN layers
    if "mlp" in kernel_lower or "gate_up" in kernel_lower or "down" in kernel_lower:
        return 6.0

    # Normalization
    if "norm" in kernel_lower or "rms" in kernel_lower:
        return 0.5  # Mostly memory bound

    # Embedding lookups
    if "embed" in kernel_lower:
        return 0.1  # Pure memory lookup

    # Default estimate
    return 1.0


def roofline_analysis(profile: dict, cpu: str = "auto") -> List[Dict]:
    """
    Compute AI and compare to roofline for each kernel.

    Args:
        profile: Performance profile dictionary
        cpu: CPU type ("auto" or specific model)

    Returns:
        List of analysis results per kernel
    """
    if cpu == "auto":
        cpu = get_cpu_info()

    peak = CPU_PEAKS.get(cpu, CPU_PEAKS["generic"])

    # Compute theoretical limits
    # FP32 peak = clock * cores * 2 (mul+add) * vector_width / 1e9 (GFLOPS)
    fp32_peak = peak["clock_ghz"] * peak["cores"] * 2 * peak["vector_width"]

    # Memory bandwidth roof (GB/s -> GB/s, divide by 8 for bytes)
    memory_roof = peak["memory_gbps"] / 8.0

    # Ridge point: where compute and memory bounds meet
    ridge_point = fp32_peak / memory_roof if memory_roof > 0 else float('inf')

    results = []
    for kernel in profile["kernels"]:
        kernel_name = kernel.get("kernel", "unknown")
        mode = kernel.get("mode", "unknown")

        # Estimate operational intensity
        ai = estimate_flops_per_byte(kernel_name, mode)

        # Compute attainable performance
        compute_bound_perf = min(fp32_peak, memory_roof * ai)

        # Determine bottleneck
        if ai < ridge_point:
            bottleneck = "memory"
            attainable = memory_roof * ai
        else:
            bottleneck = "compute"
            attainable = fp32_peak

        results.append({
            "kernel": kernel_name,
            "op": kernel.get("op", "unknown"),
            "mode": mode,
            "time_ms": kernel.get("time_ms", 0),
            "ai": ai,
            "fp32_peak_gflops": fp32_peak,
            "memory_roof_gbps": memory_roof,
            "attainable_gflops": round(attainable, 2),
            "ridge_point": round(ridge_point, 2),
            "bottleneck": bottleneck,
            "utilization_pct": round((kernel.get("time_ms", 1) / (attainable * 1000)) * 100, 2) if attainable > 0 else 0
        })

    # Sort by time (slowest first)
    return sorted(results, key=lambda x: -x["time_ms"])


def print_roofline(results: List[Dict], cpu: str):
    """Print roofline analysis results."""
    peak = CPU_PEAKS.get(cpu, CPU_PEAKS["generic"])

    print("\n" + "=" * 70)
    print("ROOFLINE ANALYSIS")
    print("=" * 70)
    print(f"\nDetected CPU: {peak['name']}")
    print(f"  Clock: {peak['clock_ghz']} GHz")
    print(f"  Cores: {peak['cores']}")
    print(f"  Vector width: {peak['vector_width']} (AVX{'2' if peak['vector_width'] == 16 else '-512'})")
    print(f"  Memory: {peak['memory_gbps']} Gbps")
    print(f"\nTheoretical Peaks:")
    print(f"  FP32: {peak['clock_ghz'] * peak['cores'] * 2 * peak['vector_width']:.0f} GFLOPS")
    print(f"  Memory: {peak['memory_gbps'] / 8:.1f} GB/s")
    print(f"  Ridge Point: {results[0]['ridge_point'] if results else 'N/A':.2f} FLOPs/byte")

    print("\n" + "-" * 70)
    print(f"{'Kernel':<30} {'Time':>8} {'AI':>6} {'Attain':>10} {'Bottleneck':<12} {'%Peak':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['kernel'][:30]:<30} {r['time_ms']:>7.2f}ms {r['ai']:>6.1f} "
              f"{r['attainable_gflops']:>8.2f}G {r['bottleneck']:<12} {r['utilization_pct']:>7.2f}%")

    print("-" * 70)

    # Summary
    memory_bound = [r for r in results if r["bottleneck"] == "memory"]
    compute_bound = [r for r in results if r["bottleneck"] == "compute"]

    total_time = sum(r["time_ms"] for r in results)
    memory_time = sum(r["time_ms"] for r in memory_bound)
    compute_time = sum(r["time_ms"] for r in compute_bound)

    print(f"\nBottleneck Summary:")
    print(f"  Memory-bound kernels: {len(memory_bound)} ({memory_time/total_time*100:.1f}% of time)")
    print(f"  Compute-bound kernels: {len(compute_bound)} ({compute_time/total_time*100:.1f}% of time)")

    if memory_bound:
        print(f"\n  Top memory-bound kernels (potential optimization targets):")
        for r in sorted(memory_bound, key=lambda x: -x["time_ms"])[:5]:
            print(f"    - {r['kernel']}: {r['time_ms']:.2f}ms")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Roofline analysis for CK kernels"
    )
    parser.add_argument("profile", help="Profile JSON file")
    parser.add_argument("--cpu", default="auto", choices=list(CPU_PEAKS.keys()),
                       help="CPU type (default: auto-detect)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    results = roofline_analysis(profile, args.cpu)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return 0

    print_roofline(results, args.cpu)

    return 0


if __name__ == "__main__":
    sys.exit(main())

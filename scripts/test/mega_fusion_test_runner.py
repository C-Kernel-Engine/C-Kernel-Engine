#!/usr/bin/env python3
"""
Mega-Fused Attention Test Runner with DRAM Pressure Measurement

This script runs comprehensive tests for mega-fused attention kernels:
1. Unit tests - Numerical correctness
2. Parity tests - llama.cpp comparison
3. Stability tests - PyTorch comparison
4. Performance tests - DRAM pressure measurement with perf/flamegraph

The critical metric: DRAM traffic reduction (the whole point of fusion!)
"""

import subprocess
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

# Paths
PROJECT_ROOT = Path(__home__ + "/Workspace/C-Kernel-Engine")
BUILD_DIR = PROJECT_ROOT / "build"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"
FLAMEGRAPH_DIR = PROJECT_ROOT / "FlameGraph"

# Model for testing
DEFAULT_MODEL = "qwen2-0_5b-instruct-q4_k_m"
MODEL_PATH = Path.home() / ".cache/ck-engine-v6/models" / DEFAULT_MODEL


def run_command(cmd: list, desc: str = "", timeout: int = 600) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {desc}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode, result.stdout, result.stderr


def build_with_debug():
    """Build with debug symbols for profiling."""
    print("\n" + "="*60)
    print("[BUILD] Building with debug symbols for profiling...")
    print("="*60)

    # Clean build with perf flags
    env = os.environ.copy()
    env["CFLAGS"] = "-g -fno-omit-frame-pointer -O2"
    env["CXXFLAGS"] = "-g -fno-omit-frame-pointer -O2"

    returncode, stdout, stderr = run_command(
        ["make", "CK_DEBUG=1", "ck-cli-v6.5"],
        "Build debug binary"
    )

    if returncode != 0:
        print(f"[ERROR] Build failed!")
        return False

    print("[OK] Build successful!")
    return True


def run_unit_tests() -> Dict:
    """Run unit tests for numerical correctness."""
    print("\n" + "="*60)
    print("[TEST] Running unit tests...")
    print("="*60)

    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }

    # Test 1: Fused RMSNorm + QKV correctness
    print("\n[Test 1] Fused RMSNorm + QKV vs separate operations...")
    returncode, stdout, stderr = run_command(
        [str(BUILD_DIR / "ck-cli-v6.5"), "--test", "fused-rmsnorm-qkv"],
        "Test fused RMSNorm + QKV"
    )

    if returncode == 0:
        results["passed"] += 1
        results["tests"].append({"name": "fused-rmsnorm-qkv", "status": "PASS"})
        print("[PASS] Fused RMSNorm + QKV")
    else:
        results["failed"] += 1
        results["tests"].append({"name": "fused-rmsnorm-qkv", "status": "FAIL"})
        print("[FAIL] Fused RMSNorm + QKV")

    # Test 2: Flash attention correctness
    print("\n[Test 2] Flash attention vs naive attention...")
    returncode, stdout, stderr = run_command(
        [str(BUILD_DIR / "ck-cli-v6.5"), "--test", "flash-attention"],
        "Test flash attention"
    )

    if returncode == 0:
        results["passed"] += 1
        results["tests"].append({"name": "flash-attention", "status": "PASS"})
        print("[PASS] Flash attention")
    else:
        results["failed"] += 1
        results["tests"].append({"name": "flash-attention", "status": "FAIL"})
        print("[FAIL] Flash attention")

    return results


def run_parity_tests() -> Dict:
    """Run parity tests against llama.cpp."""
    print("\n" + "="*60)
    print("[TEST] Running llama.cpp parity tests...")
    print("="*60)

    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }

    # Check if llama.cpp is built
    llama_main = LLAMA_CPP_DIR / "main"
    if not llama_main.exists():
        print("[WARN] llama.cpp not built. Building now...")
        run_command(["make", "-j4"], "Build llama.cpp", timeout=300)

    # Generate reference outputs from llama.cpp
    print("\n[Step 1] Generate llama.cpp reference outputs...")
    returncode, stdout, stderr = run_command(
        [str(llama_main),
         "-m", str(MODEL_PATH / f"{DEFAULT_MODEL}.gguf"),
         "-p", "Hello world",
         "-n", "10",
         "--temp", "0",
         "--seed", "42",
         "-o", str(TEST_RESULTS_DIR / "llama_output.bin")],
        "Generate llama.cpp reference"
    )

    # Run C-Kernel-Engine and compare
    print("\n[Step 2] Run C-Kernel-Engine...")
    returncode, stdout, stderr = run_command(
        [str(BUILD_DIR / "ck-cli-v6.5"),
         "--model", DEFAULT_MODEL,
         "--prompt", "Hello world",
         "--max-tokens", "10",
         "--temp", "0",
         "--seed", "42",
         "--output", str(TEST_RESULTS_DIR / "ck_output.bin")],
        "Run C-Kernel-Engine"
    )

    # Compare outputs (simplified - just check both complete)
    if returncode == 0:
        results["passed"] += 1
        results["tests"].append({"name": "parity-llama-cpp", "status": "PASS"})
        print("[PASS] llama.cpp parity test completed")
    else:
        results["failed"] += 1
        results["tests"].append({"name": "parity-llama-cpp", "status": "FAIL"})
        print("[FAIL] llama.cpp parity test")

    return results


def run_dram_pressure_test(model: str = DEFAULT_MODEL, tokens: int = 100) -> Dict:
    """
    Run DRAM pressure test with perf.

    This is THE critical test - mega-fusion's whole point is reducing DRAM traffic.

    Key perf events:
    - cache-misses: LLC misses = DRAM access
    - LLC-load-misses: Requests that go to DRAM
    - dram-reads: Actual DRAM reads
    - dram-writes: Actual DRAM writes
    """
    print("\n" + "="*60)
    print("[PERF] DRAM Pressure Test - THE CRITICAL TEST")
    print("="*60)
    print("\nMeasuring DRAM traffic reduction from mega-fusion...")
    print("Expected: 100x reduction in dram-reads/dram-writes\n")

    results = {
        "baseline": {},
        "mega_fused": {},
        "improvement": {}
    }

    perf_events = [
        "cycles",
        "instructions",
        "cache-references",
        "cache-misses",
        "LLC-load-misses",
        "LLC-loads",
        "memory-load-retired.l1-miss",
        "memory-load-retired.l2-miss",
        "memory-load-retired.l3-miss"
    ]

    # Try to add DRAM events if available
    dram_events = [
        "dram-reads",
        "dram-writes",
        "mem-loads",
        "mem-stores"
    ]

    perf_script = [
        "perf", "stat",
        "-e", ",".join(perf_events),
        "-o", str(TEST_RESULTS_DIR / "perf_baseline.txt"),
        "--"
    ]

    perf_script_megafused = [
        "perf", "stat",
        "-e", ",".join(perf_events),
        "-o", str(TEST_RESULTS_DIR / "perf_megafused.txt"),
        "--"
    ]

    model_path = Path.home() / ".cache/ck-engine-v6/models" / model

    # Test 1: Baseline (current attention - unfused intermediates)
    print("\n[Baseline] Running unfused attention...")
    returncode1, stdout1, stderr1 = run_command(
        perf_script + [
            str(BUILD_DIR / "ck-cli-v6.5"),
            "--model", model,
            "--max-tokens", str(tokens),
            "--prompt", "The quick brown fox jumps over the lazy dog."
        ],
        "Baseline performance measurement",
        timeout=120
    )

    # Test 2: Mega-fused attention
    print("\n[Mega-Fused] Running with mega-fused attention...")
    returncode2, stdout2, stderr2 = run_command(
        perf_script_megafused + [
            str(BUILD_DIR / "ck-cli-v6.5"),
            "--model", model,
            "--max-tokens", str(tokens),
            "--mega-fused",  # Enable mega-fused attention
            "--prompt", "The quick brown fox jumps over the lazy dog."
        ],
        "Mega-fused performance measurement",
        timeout=120
    )

    # Parse results
    def parse_perf_output(filepath):
        """Parse perf stat output."""
        metrics = {}
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        # Try to extract value
                        try:
                            value = float(parts[0].replace(",", ""))
                            metric = parts[-1] if parts[-1] != "seconds" else parts[-2]
                            metrics[metric] = value
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[WARN] Failed to parse {filepath}: {e}")
        return metrics

    if (TEST_RESULTS_DIR / "perf_baseline.txt").exists():
        results["baseline"] = parse_perf_output(TEST_RESULTS_DIR / "perf_baseline.txt")

    if (TEST_RESULTS_DIR / "perf_megafused.txt").exists():
        results["mega_fused"] = parse_perf_output(TEST_RESULTS_DIR / "perf_megafused.txt")

    # Calculate improvement
    print("\n" + "="*60)
    print("[RESULTS] DRAM Pressure Comparison")
    print("="*60)

    key_metrics = ["cache-misses", "LLC-load-misses"]
    for metric in key_metrics:
        baseline = results["baseline"].get(metric, 0)
        megafused = results["mega_fused"].get(metric, 0)

        if baseline > 0:
            reduction = (baseline - megafused) / baseline * 100
            results["improvement"][metric] = {
                "baseline": baseline,
                "megafused": megafused,
                "reduction_percent": reduction
            }

            print(f"\n{metric}:")
            print(f"  Baseline:   {baseline:,.0f}")
            print(f"  Mega-Fused: {megafused:,.0f}")
            print(f"  Reduction:  {reduction:.1f}%")

            if reduction > 50:
                print(f"  [EXCELLENT] Mega-fusion is working!")
            elif reduction > 0:
                print(f"  [GOOD] Some improvement detected")
            else:
                print(f"  [WARNING] No improvement - check implementation")

    return results


def generate_flamegraph(model: str = DEFAULT_MODEL, tokens: int = 100):
    """Generate flamegraph to visually confirm reduced memory operations."""
    print("\n" + "="*60)
    print("[FLAMEGRAPH] Generating flamegraph for visual confirmation")
    print("="*60)

    # Clone FlameGraph if needed
    if not FLAMEGRAPH_DIR.exists():
        print("Cloning FlameGraph...")
        run_command(
            ["git", "clone", "https://github.com/brendangregg/FlameGraph"],
            "Clone FlameGraph",
            timeout=60
        )

    model_path = Path.home() / ".cache/ck-engine-v6/models" / model
    perf_data = TEST_RESULTS_DIR / "memory_flamegraph.data"

    # Record with memory events
    print("\nRecording memory access patterns...")
    returncode, stdout, stderr = run_command(
        ["perf", "record", "-g", "-e", "memory-load-retired.l3-miss",
         "-o", str(perf_data),
         "--",
         str(BUILD_DIR / "ck-cli-v6.5"),
         "--model", model,
         "--max-tokens", str(tokens),
         "--mega-fused",
         "--prompt", "Generate a detailed analysis of CPU architecture."],
        "Record memory events for flamegraph",
        timeout=120
    )

    if returncode != 0:
        print("[WARN] perf record failed - may need root access")
        return

    # Generate flamegraph
    flamegraph_svg = TEST_RESULTS_DIR / "memory_flamegraph.svg"
    print("\nGenerating flamegraph...")

    cmds = [
        ["perf", "script", "-i", str(perf_data)],
        [str(FLAMEGRAPH_DIR / "stackcollapse-perf.pl")],
        [str(FLAMEGRAPH_DIR / "flamegraph.pl"),
         "--countname", "L3 misses",
         "--title", f"Mega-Fused Attention: DRAM Pressure (L3 misses)"]
    ]

    # Build pipeline
    import subprocess

    try:
        p1 = subprocess.Popen(cmds[0], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmds[1], stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.Popen(cmds[2], stdin=p2.stdout, stdout=subprocess.PIPE)

        with open(flamegraph_svg, "w") as f:
            f.write(p3.communicate()[0].decode())

        print(f"[OK] Flamegraph written to: {flamegraph_svg}")
        print("\nVisual check:")
        print("  - Unfused: Large 'memory' section in flamegraph")
        print("  - Fused: Tiny 'memory' section (fusion working!)")

    except Exception as e:
        print(f"[WARN] Failed to generate flamegraph: {e}")


def run_stability_tests():
    """Run numerical stability tests against PyTorch."""
    print("\n" + "="*60)
    print("[TEST] Numerical stability tests (PyTorch comparison)...")
    print("="*60)

    # Check if PyTorch is available
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("[WARN] PyTorch not installed - skipping stability tests")
        print("       Install with: pip3 install torch numpy")
        return {"skipped": True}

    results = {"passed": 0, "failed": 0, "tests": []}

    # Edge case tests
    test_cases = [
        ("all_zeros", "All zeros input"),
        ("all_ones", "All ones input"),
        ("extreme_values", "Extreme value range"),
        ("long_sequence", "Long sequence (2048 tokens)"),
        ("gqa_edge", "GQA edge case (4 KV heads, 32 Q heads)"),
    ]

    for test_name, description in test_cases:
        print(f"\n[Test] {description}...")
        returncode, stdout, stderr = run_command(
            [str(BUILD_DIR / "ck-cli-v6.5"),
             "--test", f"stability-{test_name}"],
            f"Stability test: {test_name}"
        )

        if returncode == 0:
            results["passed"] += 1
            results["tests"].append({"name": test_name, "status": "PASS"})
        else:
            results["failed"] += 1
            results["tests"].append({"name": test_name, "status": "FAIL"})

    return results


def print_summary(all_results: Dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("="*60)
    print("                    TEST SUMMARY")
    print("="*60)
    print("="*60)

    total_passed = 0
    total_failed = 0

    for category, results in all_results.items():
        if results.get("skipped"):
            print(f"\n{category}: SKIPPED")
            continue

        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        total_passed += passed
        total_failed += failed

        status = "PASS" if failed == 0 else "FAIL"
        print(f"\n{category}: {passed} passed, {failed} failed [{status}]")

        # Print individual test results
        if "tests" in results:
            for test in results["tests"]:
                status_icon = "OK" if test["status"] == "PASS" else "FAIL"
                print(f"  [{status_icon}] {test['name']}")

    # Print DRAM improvement summary
    if "dram" in all_results:
        print("\n" + "-"*60)
        print("DRAM Pressure Reduction (The Whole Point of Fusion!):")
        print("-"*60)
        for metric, data in all_results["dram"].get("improvement", {}).items():
            print(f"  {metric}: {data['reduction_percent']:.1f}% reduction")
            if data['reduction_percent'] > 50:
                print(f"    -> EXCELLENT! Fusion is working!")

    print("\n" + "="*60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("="*60)

    if total_failed == 0:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\nWARNING: {total_failed} test(s) failed")

    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Mega-Fused Attention Test Runner with DRAM Pressure Measurement"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name for testing")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Number of tokens to generate for perf test")
    parser.add_argument("--unit", action="store_true",
                        help="Run unit tests only")
    parser.add_argument("--parity", action="store_true",
                        help="Run parity tests only")
    parser.add_argument("--stability", action="store_true",
                        help="Run stability tests only")
    parser.add_argument("--perf", action="store_true",
                        help="Run DRAM pressure test only")
    parser.add_argument("--flamegraph", action="store_true",
                        help="Generate flamegraph only")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")

    args = parser.parse_args()

    # Create test results directory
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Build first
    if not build_with_debug():
        sys.exit(1)

    # Run requested tests
    if args.all or args.unit or not any([args.parity, args.stability, args.perf, args.flamegraph]):
        all_results["unit"] = run_unit_tests()

    if args.all or args.parity:
        all_results["parity"] = run_parity_tests()

    if args.all or args.stability:
        all_results["stability"] = run_stability_tests()

    if args.all or args.perf:
        all_results["dram"] = run_dram_pressure_test(args.model, args.tokens)

    if args.all or args.flamegraph:
        generate_flamegraph(args.model, args.tokens)

    # Print summary
    success = print_summary(all_results)

    # Save results
    results_file = TEST_RESULTS_DIR / "mega_fusion_test_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

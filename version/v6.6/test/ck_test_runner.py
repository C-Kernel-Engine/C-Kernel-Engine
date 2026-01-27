#!/usr/bin/env python3
"""
ck_test_runner.py - Convenience wrapper for running C-Kernel-Engine tests

This script provides quick access to common test workflows without remembering
individual command lines.

USAGE:
    python ck_test_runner.py --quick          # Quick sanity check
    python ck_test_runner.py --parity         # Full parity check
    python ck_test_runner.py --memory         # Memory validation
    python ck_test_runner.py --trace 25       # Trace token 25
    python ck_test_runner.py --divergence     # Find divergence
    python ck_test_runner.py --all            # Run everything

For full documentation, see TEST_README.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Default model path
DEFAULT_MODEL = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

# Test directory
TEST_DIR = Path(__file__).parent


class TestRunner:
    def __init__(self, model_path: Path, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose

    def run_script(self, script: str, args: list = None) -> int:
        """Run a test script."""
        script_path = TEST_DIR / script
        if not script_path.exists():
            print(f"ERROR: {script} not found")
            return 1

        cmd = [sys.executable, str(script_path), "--model", str(self.model_path)]
        if args:
            cmd.extend(args)

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd)
        return result.returncode

    def quick_sanity(self) -> int:
        """Quick sanity check - just run embedding and logits."""
        print("\n" + "=" * 70)
        print("QUICK SANITY CHECK")
        print("=" * 70)

        # Test embedding
        print("\n[1/3] Testing embedding...")
        result = self.run_script("test_embedding_only.py")
        if result != 0:
            print("EMBEDDING FAILED")
            return 1

        # Test weight offsets
        print("\n[2/3] Checking weight offsets...")
        result = self.run_script("test_weight_offset_consistency.py")
        if result != 0:
            print("WEIGHT OFFSET CHECK FAILED")
            return 1

        # Test memory planner
        print("\n[3/3] Validating memory layout...")
        result = self.run_script("test_memory_planner.py", [
            f"--layout={self.model_path}/layout_decode.json"
        ])
        if result != 0:
            print("MEMORY VALIDATION FAILED")
            return 1

        print("\n" + "=" * 70)
        print("QUICK SANITY CHECK: PASSED")
        print("=" * 70)
        return 0

    def parity_check(self, token: int = 25, stop_on_fail: bool = True) -> int:
        """Full layer-by-layer parity check."""
        print("\n" + "=" * 70)
        print(f"NUMERICAL PARITY CHECK (token={token})")
        print("=" * 70)

        args = [f"--token", str(token)]
        if not stop_on_fail:
            args.append("--all-layers")

        return self.run_script("test_layer_by_layer.py", args)

    def memory_validation(self) -> int:
        """Comprehensive memory validation."""
        print("\n" + "=" * 70)
        print("MEMORY VALIDATION")
        print("=" * 70)

        # Quick memory check
        print("\n[1/4] Quick memory layout check...")
        result = self.run_script("test_memory_planner.py", [
            f"--layout={self.model_path}/layout_decode.json"
        ])
        if result != 0:
            print("Quick memory check failed")
            return 1

        # Weight offsets
        print("\n[2/4] Weight offset consistency...")
        result = self.run_script("test_weight_offset_consistency.py")
        if result != 0:
            print("Weight offset check failed")
            return 1

        # Advanced validation
        print("\n[3/4] Advanced memory validation...")
        result = self.run_script("advanced_memory_validator.py", ["--deep"])
        if result != 0:
            print("Advanced validation failed")
            return 1

        # BUMP sync
        print("\n[4/4] BUMP layout sync check...")
        result = self.run_script("test_bump_layout_sync.py")
        if result != 0:
            print("BUMP sync check failed")
            return 1

        print("\n" + "=" * 70)
        print("MEMORY VALIDATION: PASSED")
        print("=" * 70)
        return 0

    def trace_token(self, token: int) -> int:
        """Trace execution for a specific token."""
        print("\n" + "=" * 70)
        print(f"TRACE TOKEN {token}")
        print("=" * 70)

        return self.run_script("v6_6_comprehensive_debug.py", [
            "--verbose",
            f"--token={token}"
        ])

    def find_divergence(self, token: int = 25) -> int:
        """Find divergence between v6.5 and v6.6."""
        print("\n" + "=" * 70)
        print(f"FIND DIVERGENCE (token={token})")
        print("=" * 70)
        print("\nThis requires both v6.5 and v6.6 models to be cached.")

        return self.run_script("trace_divergence.py", [f"--token={token}"])

    def debug_nan(self, token: int = 25) -> int:
        """Debug NaN issues."""
        print("\n" + "=" * 70)
        print(f"DEBUG NaN (token={token})")
        print("=" * 70)

        # Find layer with NaN
        print("\n[1/2] Finding layer with NaN...")
        result = self.run_script("trace_nan_layer.py", [f"--token={token}"])
        if result != 0:
            print("trace_nan_layer failed")

        # Find source
        print("\n[2/2] Finding source of NaN...")
        result = self.run_script("trace_nan_source.py", [f"--token={token}"])

        return result

    def run_all(self) -> int:
        """Run all tests."""
        print("\n" + "=" * 70)
        print("RUNNING ALL TESTS")
        print("=" * 70)

        tests = [
            ("Quick Sanity", lambda: self.quick_sanity()),
            ("Memory Validation", lambda: self.memory_validation()),
            ("Parity Check", lambda: self.parity_check(stop_on_fail=False)),
        ]

        results = []
        for name, func in tests:
            print(f"\n{'='*70}")
            print(f"TEST: {name}")
            print(f"{'='*70}")
            try:
                result = func()
                results.append((name, result))
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                results.append((name, 1))

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        passed = 0
        failed = 0
        for name, result in results:
            status = "PASS" if result == 0 else "FAIL"
            print(f"  {name}: {status}")
            if result == 0:
                passed += 1
            else:
                failed += 1

        print(f"\nTotal: {passed} passed, {failed} failed")
        return 1 if failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ck_test_runner.py --quick              # Quick sanity check
    python ck_test_runner.py --parity --token 25  # Parity check token 25
    python ck_test_runner.py --memory             # Memory validation
    python ck_test_runner.py --trace 25           # Trace token 25
    python ck_test_runner.py --divergence         # Find divergence
    python ck_test_runner.py --nan 25             # Debug NaN at token 25
    python ck_test_runner.py --all                # Run all tests
        """
    )

    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Model cache path (default: {DEFAULT_MODEL})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    # Test options
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check (embedding, offsets, memory)")
    parser.add_argument("--parity", action="store_true",
                        help="Full layer-by-layer parity check")
    parser.add_argument("--memory", action="store_true",
                        help="Comprehensive memory validation")
    parser.add_argument("--trace", type=int, metavar="TOKEN",
                        help="Trace execution for specific token")
    parser.add_argument("--divergence", action="store_true",
                        help="Find v6.5/v6.6 divergence point")
    parser.add_argument("--nan", type=int, metavar="TOKEN",
                        help="Debug NaN for specific token")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")

    args = parser.parse_args()

    # Check model path
    if not args.model.exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Please run the pipeline to generate model files first:")
        print(f"  cd {TEST_DIR.parent.parent}")
        print("  python scripts/ck_run_v6_6.py --download --build --run")
        return 1

    runner = TestRunner(args.model, args.verbose)

    # Run selected test(s)
    if args.quick:
        return runner.quick_sanity()
    elif args.parity:
        return runner.parity_check()
    elif args.memory:
        return runner.memory_validation()
    elif args.trace is not None:
        return runner.trace_token(args.trace)
    elif args.divergence:
        return runner.find_divergence()
    elif args.nan is not None:
        return runner.debug_nan(args.nan)
    elif args.all:
        return runner.run_all()
    else:
        # Default: quick sanity check
        print("No test specified, running quick sanity check...")
        return runner.quick_sanity()


if __name__ == "__main__":
    sys.exit(main())

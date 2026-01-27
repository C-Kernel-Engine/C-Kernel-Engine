#!/usr/bin/env python3
"""
run_all_v66_tests.py - Run all v6.6 validation tests and generate report

This script runs all v6.6 tests and generates a comprehensive report
showing what's working and what's broken.

Usage:
    python run_all_v66_tests.py
    python run_all_v66_tests.py --verbose
    python run_all_v66_tests.py --report output.txt
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Test directories
TEST_DIR = Path(__file__).parent
ALT_TEST_DIR = TEST_DIR.parent / "tests"


@dataclass
class TestSuiteResult:
    """Result from running a test suite."""
    name: str
    passed: bool
    duration: float
    output: str
    error: str = ""


# All test modules to run (in order)
TEST_MODULES = [
    ("test_ir_stitch_validate", "IR Stitch Validate (Lower 3 wiring)"),
    ("test_decode_sanity", "Decode Sanity (runtime wiring checks)"),
    ("test_prefill_sanity", "Prefill Sanity (runtime wiring checks)"),
    ("test_codegen_respects_ir", "Codegen respects IR (ROOT CAUSE)"),
    ("test_ir_lower3", "IR Lower 3 (call-ready)"),
    ("test_io_chain", "I/O Chain (output→input)"),
    ("test_dtype_consistency", "Dtype Consistency"),
    ("test_scratch_buffer", "Scratch Buffer Allocation"),
    ("test_kv_cache", "KV Cache Integration"),
    ("test_residual_flow", "Residual Flow"),
    ("test_dimension_alignment", "Dimension Alignment"),
    ("test_op_naming_consistency", "Op Naming Consistency"),
    ("test_pipeline_validation", "Pipeline Validation"),
    ("test_codegen_ir_builder", "Codegen IR Builder"),
    ("test_memory_planner", "Memory Planner"),
    ("test_kernel_validation", "Kernel Validation"),
    ("test_numerical_parity", "Numerical Parity"),
    ("test_layer_by_layer", "Layer by Layer"),
    ("test_embedding_only", "Embedding Only"),
    ("test_kernel_direct", "Kernel Direct"),
    ("test_pipeline_v2", "Pipeline V2"),
]


def run_test_module(module_name: str) -> TestSuiteResult:
    """Run a single test module and capture results."""
    module_path = TEST_DIR / f"{module_name}.py"
    if not module_path.exists():
        alt_path = ALT_TEST_DIR / f"{module_name}.py"
        if alt_path.exists():
            module_path = alt_path

    if not module_path.exists():
        return TestSuiteResult(
            name=module_name,
            passed=False,
            duration=0,
            output="",
            error=f"Test file not found: {module_path}"
        )

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(module_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=TEST_DIR
        )
        duration = time.time() - start

        passed = result.returncode == 0

        return TestSuiteResult(
            name=module_name,
            passed=passed,
            duration=duration,
            output=result.stdout,
            error=result.stderr if result.stderr else ""
        )

    except subprocess.TimeoutExpired:
        return TestSuiteResult(
            name=module_name,
            passed=False,
            duration=120,
            output="",
            error="Test timed out after 120 seconds"
        )
    except Exception as e:
        return TestSuiteResult(
            name=module_name,
            passed=False,
            duration=time.time() - start,
            output="",
            error=str(e)
        )


def extract_issues_from_output(output: str) -> List[str]:
    """Extract issue lines from test output."""
    issues = []
    in_issues = False

    for line in output.split('\n'):
        if 'ISSUES' in line or 'CRITICAL' in line or 'ERROR' in line:
            in_issues = True
        if in_issues and line.strip().startswith('['):
            issues.append(line.strip())
        if 'VERDICT' in line:
            in_issues = False

    return issues


def generate_report(results: List[TestSuiteResult], verbose: bool = False) -> str:
    """Generate a summary report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 80)
    lines.append("V6.6 VALIDATION REPORT")
    lines.append(f"Generated: {now}")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Tests passed: {passed}/{total}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")

    # Results table
    lines.append("TEST RESULTS")
    lines.append("-" * 80)
    lines.append(f"{'Test':<40} {'Status':<10} {'Time':<10}")
    lines.append("-" * 80)

    for module_name, description in TEST_MODULES:
        result = next((r for r in results if r.name == module_name), None)
        if result:
            status = "PASS" if result.passed else "FAIL"
            symbol = "✓" if result.passed else "✗"
            lines.append(f"{symbol} {description:<38} {status:<10} {result.duration:.1f}s")
        else:
            lines.append(f"? {description:<38} {'SKIP':<10} -")

    lines.append("-" * 80)
    lines.append("")

    # Issues by test
    lines.append("ISSUES BY TEST")
    lines.append("-" * 80)

    all_issues = []
    for result in results:
        if not result.passed:
            issues = extract_issues_from_output(result.output)
            if issues:
                lines.append(f"\n{result.name}:")
                for issue in issues[:5]:  # Limit to 5 per test
                    lines.append(f"  {issue}")
                    all_issues.append((result.name, issue))
                if len(issues) > 5:
                    lines.append(f"  ... and {len(issues) - 5} more")
            elif result.error:
                lines.append(f"\n{result.name}:")
                lines.append(f"  Error: {result.error[:200]}")

    if not all_issues:
        lines.append("No issues found!")

    lines.append("")

    # Root cause analysis
    lines.append("=" * 80)
    lines.append("ROOT CAUSE ANALYSIS")
    lines.append("=" * 80)

    # Check for the most critical issues
    codegen_result = next((r for r in results if r.name == "test_codegen_respects_ir"), None)
    if codegen_result and not codegen_result.passed:
        lines.append("")
        lines.append("⚠ CRITICAL: Codegen does not respect IR")
        lines.append("  This is the ROOT CAUSE of most issues.")
        lines.append("  Codegen ignores lowered IR and uses hardcoded values.")
        lines.append("")
        lines.append("  To fix:")
        lines.append("  1. Make codegen read ops from IR sections (not top-level)")
        lines.append("  2. Use buffer offsets from layout.json")
        lines.append("  3. Handle all kernel types (especially mega_fused)")
        lines.append("  4. Pass scratch buffers (not NULL)")

    io_result = next((r for r in results if r.name == "test_io_chain"), None)
    if io_result and not io_result.passed:
        lines.append("")
        lines.append("⚠ I/O chain broken")
        lines.append("  Output of kernel N doesn't feed kernel N+1.")

    kv_result = next((r for r in results if r.name == "test_kv_cache"), None)
    if kv_result and not kv_result.passed:
        lines.append("")
        lines.append("⚠ KV cache not integrated")
        lines.append("  Autoregressive decoding will not work correctly.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all v6.6 validation tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show test output")
    parser.add_argument("--report", "-r", type=str, help="Save report to file")
    parser.add_argument("--test", "-t", type=str, help="Run single test module")
    args = parser.parse_args()

    print("=" * 80)
    print("V6.6 VALIDATION TEST SUITE")
    print("=" * 80)

    # Filter tests if single test requested
    if args.test:
        tests_to_run = [(args.test, args.test)]
    else:
        tests_to_run = TEST_MODULES

    results = []

    for module_name, description in tests_to_run:
        print(f"\nRunning: {description}...")
        result = run_test_module(module_name)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} ({result.duration:.1f}s)")

        if args.verbose and result.output:
            print("\n" + "-" * 40)
            print(result.output)
            print("-" * 40)

        if result.error:
            print(f"  Error: {result.error[:100]}")

    # Generate report
    report = generate_report(results, args.verbose)

    print("\n")
    print(report)

    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.report}")

    # Exit code
    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

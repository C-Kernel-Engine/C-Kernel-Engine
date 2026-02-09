#!/usr/bin/env python3
"""
Report kernel test coverage.

Tracks which kernels from the registry have corresponding tests.

Usage:
    python3 scripts/kernel_test_coverage.py
    python3 scripts/kernel_test_coverage.py --registry version/v6.6/kernel_maps/KERNEL_REGISTRY.json
    python3 scripts/kernel_test_coverage.py --json
    python3 scripts/kernel_test_coverage.py --threshold 75
"""

import argparse
import json
import re
from pathlib import Path
from typing import Set


def extract_kernel_ids_from_registry(registry_path: str) -> Set[str]:
    """Extract all kernel IDs from the registry."""
    with open(registry_path) as f:
        registry = json.load(f)

    kernel_ids = set()
    for kernel in registry.get("kernels", []):
        kernel_id = kernel.get("id")
        if kernel_id:
            kernel_ids.add(kernel_id)

            # Also add normalized forms
            kernel_ids.add(kernel_id.lower())
            kernel_ids.add(kernel_id.replace("_", "-"))

    return kernel_ids


def find_kernel_tests(test_dirs: list, kernel_ids: Set[str]) -> dict:
    """Find which kernels have corresponding tests."""
    tested = set()
    test_details = {}

    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if not test_path.exists():
            continue

        for path in test_path.rglob("*.py"):
            content = path.read_text()
            path_str = str(path)

            for kernel_id in kernel_ids:
                # Check for kernel ID in test content
                patterns = [
                    kernel_id,
                    kernel_id.replace("_", "-"),
                    kernel_id.replace("-", "_"),
                ]

                for pattern in patterns:
                    if pattern.lower() in content.lower():
                        tested.add(kernel_id)
                        if kernel_id not in test_details:
                            test_details[kernel_id] = []
                        test_details[kernel_id].append(path_str)
                        break

    return {
        "tested": tested,
        "details": test_details
    }


def find_c_tests(test_dirs: list, kernel_ids: Set[str]) -> dict:
    """Find C/C++ tests for kernels."""
    tested = set()
    test_details = {}

    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if not test_path.exists():
            continue

        for path in test_path.rglob("*.[ch]"):
            content = path.read_text()
            path_str = str(path)

            for kernel_id in kernel_ids:
                patterns = [
                    kernel_id,
                    kernel_id.replace("_", "-"),
                    kernel_id.replace("-", "_"),
                ]

                for pattern in patterns:
                    if pattern.lower() in content.lower():
                        tested.add(kernel_id)
                        if kernel_id not in test_details:
                            test_details[kernel_id] = []
                        test_details[kernel_id].append(path_str)
                        break

    return {
        "tested": tested,
        "details": test_details
    }


def kernel_coverage_report(registry_path: str, test_dirs: list,
                            c_test_dirs: list = None) -> dict:
    """Calculate kernel test coverage."""
    kernel_ids = extract_kernel_ids_from_registry(registry_path)

    # Python tests
    py_result = find_kernel_tests(test_dirs, kernel_ids)

    # C tests (if specified)
    if c_test_dirs:
        c_result = find_kernel_tests(c_test_dirs, kernel_ids)
        tested = py_result["tested"] | c_result["tested"]
        # Merge details
        for kernel_id, files in c_result["details"].items():
            if kernel_id in py_result["details"]:
                py_result["details"][kernel_id].extend(files)
            else:
                py_result["details"][kernel_id] = files
    else:
        tested = py_result["tested"]

    # Calculate coverage
    total = len(kernel_ids)
    coverage_pct = (len(tested) / total * 100) if total > 0 else 0

    return {
        "total_kernels": total,
        "tested_kernels": len(tested),
        "untested_kernels": list(kernel_ids - tested),
        "coverage_pct": coverage_pct,
        "test_details": py_result["details"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Report kernel test coverage"
    )
    parser.add_argument("--registry",
                       default="version/v6.6/kernel_maps/KERNEL_REGISTRY.json",
                       help="Path to kernel registry JSON")
    parser.add_argument("--tests", nargs="+",
                       default=["version/v6.6/test", "version/v6.6/unittest",
                               "unittest", "test"],
                       help="Directories containing Python tests")
    parser.add_argument("--c-tests", nargs="+", default=None,
                       help="Directories containing C/C++ tests")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--threshold", type=float, default=50,
                       help="Minimum coverage threshold (default: 50%%)")
    parser.add_argument("--show-untested", action="store_true",
                       help="Show untested kernels")
    args = parser.parse_args()

    report = kernel_coverage_report(
        args.registry,
        args.tests,
        args.c_tests
    )

    if args.json:
        # JSON output - exclude test_details to keep it clean
        clean_report = {k: v for k, v in report.items() if k != "test_details"}
        print(json.dumps(clean_report, indent=2))
        return 0

    # Text output
    print("\n" + "="*60)
    print("KERNEL TEST COVERAGE REPORT")
    print("="*60)
    print(f"\nRegistry: {args.registry}")
    print(f"Total kernels: {report['total_kernels']}")
    print(f"Tested kernels: {report['tested_kernels']}")
    print(f"Untested kernels: {report['total_kernels'] - report['tested_kernels']}")
    print(f"Coverage: {report['coverage_pct']:.1f}%")

    # Visual bar
    bar_width = 40
    filled = int(report['coverage_pct'] / 100 * bar_width)
    bar = "[" + "█" * filled + "░" * (bar_width - filled) + "]"
    print(f"\n{bar} {report['coverage_pct']:.1f}%")

    if report['untested_kernels']:
        print(f"\nUntested kernels ({len(report['untested_kernels'])}):")
        for k in sorted(report['untested_kernels'])[:30]:
            print(f"  - {k}")
        if len(report['untested_kernels']) > 30:
            print(f"  ... and {len(report['untested_kernels']) - 30} more")
    else:
        print("\nAll kernels have tests!")

    # Status based on threshold
    print("\n" + "-"*60)
    if report['coverage_pct'] >= args.threshold:
        print(f"PASS: Coverage ({report['coverage_pct']:.1f}%) meets threshold ({args.threshold}%)")
        return 0
    else:
        print(f"FAIL: Coverage ({report['coverage_pct']:.1f}%) below threshold ({args.threshold}%)")
        return 1


if __name__ == "__main__":
    exit(main())

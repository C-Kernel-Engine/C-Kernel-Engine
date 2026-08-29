#!/usr/bin/env python3
"""
Sync kernel_meta.json with actual source code.

Scans C source files for optimization indicators and updates
the JSON metadata file. Can also report discrepancies.

Usage:
    python3 scripts/sync_kernel_meta.py --check     # Report differences only
    python3 scripts/sync_kernel_meta.py --update    # Update JSON with detected opts
    python3 scripts/sync_kernel_meta.py --init      # Initialize missing kernels
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
META_FILE = ROOT / "version" / "meta" / "kernel_meta.json"
KERNEL_DIR = ROOT / "src" / "kernels"


# Patterns to detect optimization levels in C code
OPT_PATTERNS = {
    "simd_avx512": [
        r"__m512",
        r"_mm512_",
        r"#ifdef\s+__AVX512",
        r"avx512",
    ],
    "simd_avx512bf16": [
        r"__m512bh",
        r"_mm512_cvtne2ps_pbh",
        r"_mm512_dpbf16_ps",
        r"avx512bf16",
    ],
    "simd_avx2": [
        r"__m256[^i]",
        r"_mm256_",
        r"#ifdef\s+__AVX2",
    ],
    "simd_amx": [
        r"_tile_",
        r"__tile",
        r"AMX",
    ],
    "blocked": [
        r"BLOCK_SIZE",
        r"block_",
        r"_blocked",
        r"cache.?block",
        r"tile_",
    ],
    "parallel": [
        r"#pragma\s+omp\s+parallel",
        r"omp_get_",
        r"_parallel\(",
    ],
    "fused": [
        r"_fused",
        r"fused_",
    ],
}


def detect_opts_in_file(filepath):
    """Detect optimization levels used in a C file."""
    try:
        content = filepath.read_text()
    except Exception:
        return set()

    detected = set()
    for opt_level, patterns in OPT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                detected.add(opt_level)
                break

    # If no SIMD detected, assume scalar
    simd_opts = {"simd_avx512", "simd_avx512bf16", "simd_avx2", "simd_amx"}
    if not (detected & simd_opts):
        detected.add("scalar")

    return detected


def detect_functions_in_file(filepath):
    """Detect kernel functions and their directions (forward/backward)."""
    try:
        content = filepath.read_text()
    except Exception:
        return []

    functions = []
    # Match function definitions
    pattern = r"^(?:void|int|float|double|static\s+\w+)\s+(\w+)\s*\([^)]*\)\s*\{"
    for match in re.finditer(pattern, content, re.MULTILINE):
        func_name = match.group(1)
        # Skip internal/helper functions
        if func_name.startswith("_") or func_name in ["main", "init", "cleanup"]:
            continue
        functions.append(func_name)

    return functions


def classify_function_direction(func_name):
    """Classify if function is forward or backward pass."""
    backward_indicators = ["backward", "_bwd", "_grad", "_dx", "_dw", "gemm_nn", "gemm_tn"]
    for indicator in backward_indicators:
        if indicator in func_name.lower():
            return "backward"
    return "forward"


def get_dtype_from_filename(filename):
    """Extract dtype from kernel filename."""
    name = filename.lower()
    if "_bf16" in name:
        return "bf16"
    elif "_f16" in name:
        return "fp16"
    elif "_q4_0" in name:
        return "q4_0"
    elif "_q4k" in name:
        return "q4_k"
    elif "_q8_0" in name:
        return "q8_0"
    elif "_int8" in name:
        return "int8"
    elif "_int4" in name:
        return "int4"
    else:
        return "fp32"


def get_kernel_from_filename(filename):
    """Extract kernel name from filename."""
    name = filename.replace("_kernels", "").replace(".c", "")
    # Remove dtype suffixes
    for suffix in ["_bf16", "_f16", "_q4_0", "_q4k", "_q8_0", "_int8", "_int4"]:
        name = name.replace(suffix, "")
    return name


def scan_source_files():
    """Scan all kernel source files and detect optimizations."""
    results = defaultdict(lambda: defaultdict(lambda: {"forward": set(), "backward": set()}))

    for cfile in KERNEL_DIR.glob("*_kernels*.c"):
        kernel = get_kernel_from_filename(cfile.name)
        dtype = get_dtype_from_filename(cfile.name)
        opts = detect_opts_in_file(cfile)
        funcs = detect_functions_in_file(cfile)

        # Determine if file has forward/backward
        has_forward = False
        has_backward = False
        for func in funcs:
            direction = classify_function_direction(func)
            if direction == "forward":
                has_forward = True
            else:
                has_backward = True

        if has_forward:
            results[kernel][dtype]["forward"].update(opts)
        if has_backward:
            results[kernel][dtype]["backward"].update(opts)

        # If we couldn't determine direction, assume forward
        if not has_forward and not has_backward:
            results[kernel][dtype]["forward"].update(opts)

    return results


def load_meta():
    """Load existing metadata."""
    if not META_FILE.exists():
        return {"kernels": {}}
    with open(META_FILE) as f:
        return json.load(f)


def save_meta(meta):
    """Save metadata."""
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {META_FILE}")


def compare_opts(detected, recorded):
    """Compare detected vs recorded optimization levels."""
    detected_set = set(detected) if detected else set()
    recorded_set = set(recorded) if recorded else set()

    missing = detected_set - recorded_set
    extra = recorded_set - detected_set

    return missing, extra


def check_discrepancies(meta, detected):
    """Check for discrepancies between meta and detected."""
    discrepancies = []

    for kernel, dtypes in detected.items():
        if kernel not in meta.get("kernels", {}):
            discrepancies.append(f"[NEW] Kernel '{kernel}' not in meta")
            continue

        kernel_meta = meta["kernels"][kernel]
        for dtype, directions in dtypes.items():
            if dtype not in kernel_meta.get("variants", {}):
                discrepancies.append(f"[NEW] {kernel}/{dtype} not in meta")
                continue

            variant = kernel_meta["variants"][dtype]
            for direction in ["forward", "backward"]:
                if direction not in directions or not directions[direction]:
                    continue

                detected_opts = directions[direction]
                if direction in variant:
                    recorded_opts = variant[direction].get("opt_level", [])
                else:
                    recorded_opts = []

                missing, extra = compare_opts(detected_opts, recorded_opts)
                if missing:
                    discrepancies.append(
                        f"[MISSING] {kernel}/{dtype}/{direction}: "
                        f"detected {missing} not in meta"
                    )
                if extra:
                    discrepancies.append(
                        f"[EXTRA] {kernel}/{dtype}/{direction}: "
                        f"meta has {extra} not detected in code"
                    )

    return discrepancies


def update_meta(meta, detected):
    """Update meta with detected optimizations."""
    updated = False

    for kernel, dtypes in detected.items():
        if kernel not in meta.get("kernels", {}):
            print(f"  Skipping new kernel '{kernel}' (use --init to add)")
            continue

        kernel_meta = meta["kernels"][kernel]
        for dtype, directions in dtypes.items():
            if dtype not in kernel_meta.get("variants", {}):
                print(f"  Skipping new dtype {kernel}/{dtype} (use --init to add)")
                continue

            variant = kernel_meta["variants"][dtype]
            for direction in ["forward", "backward"]:
                if direction not in directions or not directions[direction]:
                    continue

                detected_opts = sorted(list(directions[direction]))
                if direction in variant:
                    current = variant[direction].get("opt_level", [])
                    if set(current) != set(detected_opts):
                        print(f"  Updating {kernel}/{dtype}/{direction}: {current} -> {detected_opts}")
                        variant[direction]["opt_level"] = detected_opts
                        updated = True

    return updated


def init_missing(meta, detected):
    """Initialize missing kernels/dtypes in meta."""
    if "kernels" not in meta:
        meta["kernels"] = {}

    for kernel, dtypes in detected.items():
        if kernel not in meta["kernels"]:
            print(f"  Adding new kernel: {kernel}")
            meta["kernels"][kernel] = {
                "description": f"{kernel} kernel",
                "category": "compute",
                "variants": {}
            }

        for dtype, directions in dtypes.items():
            if dtype not in meta["kernels"][kernel].get("variants", {}):
                if "variants" not in meta["kernels"][kernel]:
                    meta["kernels"][kernel]["variants"] = {}

                print(f"  Adding new variant: {kernel}/{dtype}")
                meta["kernels"][kernel]["variants"][dtype] = {}

            variant = meta["kernels"][kernel]["variants"][dtype]
            for direction in ["forward", "backward"]:
                if direction not in directions or not directions[direction]:
                    continue

                if direction not in variant:
                    detected_opts = sorted(list(directions[direction]))
                    print(f"  Adding {kernel}/{dtype}/{direction}: {detected_opts}")
                    variant[direction] = {
                        "status": "done",
                        "opt_level": detected_opts
                    }


def main():
    parser = argparse.ArgumentParser(description="Sync kernel_meta.json with source code")
    parser.add_argument("--check", action="store_true", help="Check for discrepancies only")
    parser.add_argument("--update", action="store_true", help="Update meta with detected opts")
    parser.add_argument("--init", action="store_true", help="Initialize missing kernels")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not any([args.check, args.update, args.init]):
        args.check = True  # Default to check

    print("Scanning source files...")
    detected = scan_source_files()

    if args.verbose:
        print("\nDetected optimizations:")
        for kernel, dtypes in sorted(detected.items()):
            for dtype, directions in sorted(dtypes.items()):
                for direction, opts in directions.items():
                    if opts:
                        print(f"  {kernel}/{dtype}/{direction}: {sorted(opts)}")

    meta = load_meta()

    if args.check:
        print("\nChecking for discrepancies...")
        discrepancies = check_discrepancies(meta, detected)
        if discrepancies:
            print("\nDiscrepancies found:")
            for d in discrepancies:
                print(f"  {d}")
            print(f"\nTotal: {len(discrepancies)} discrepancies")
            print("\nRun with --update to sync meta, or --init to add new entries")
        else:
            print("\nNo discrepancies found - meta is in sync!")

    if args.init:
        print("\nInitializing missing entries...")
        init_missing(meta, detected)
        save_meta(meta)

    if args.update:
        print("\nUpdating existing entries...")
        if update_meta(meta, detected):
            save_meta(meta)
        else:
            print("  No updates needed")


if __name__ == "__main__":
    main()

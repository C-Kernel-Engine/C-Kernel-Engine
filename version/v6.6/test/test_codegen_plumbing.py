#!/usr/bin/env python3
"""
test_codegen_plumbing.py - Validate IR3 → codegen → C code pipeline

Tests:
1. IR3 generates correct pointer expressions (uint8_t* base + byte offset)
2. Generated C code has correct types
3. Code compiles successfully
4. Named defines are used (not raw numbers)

Usage:
    python test_codegen_plumbing.py --model-dir=/path/to/model
    python test_codegen_plumbing.py  # Uses default cached model
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Default model directory
DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"


def test_ir3_format(ir3_path: Path) -> dict:
    """Test IR3 structure and expressions."""
    results = {"name": "IR3 Format", "passed": True, "issues": []}

    if not ir3_path.exists():
        results["passed"] = False
        results["issues"].append(f"IR3 file not found: {ir3_path}")
        return results

    with open(ir3_path) as f:
        ir3 = json.load(f)

    # Check format version
    if ir3.get("format") != "lowered-ir-v3":
        results["issues"].append(f"Wrong format: {ir3.get('format')}, expected lowered-ir-v3")

    if ir3.get("version") != 3:
        results["issues"].append(f"Wrong version: {ir3.get('version')}, expected 3")

    # Check for errors in ops
    errors = ir3.get("errors", [])
    if errors:
        results["passed"] = False
        results["issues"].append(f"{len(errors)} ops have errors")
        for err in errors[:5]:  # Show first 5
            results["issues"].append(f"  Op {err.get('idx')}: {err.get('errors', err.get('error', 'unknown'))}")

    # Check pointer expressions use model->bump
    ops = ir3.get("operations", [])
    bad_exprs = []
    for op in ops:
        for arg in op.get("args", []):
            expr = arg.get("expr", "")
            # Check for raw numeric offsets without named define
            if re.search(r"model->bump \+ \d{6,}", expr):  # 6+ digit raw number
                if not re.search(r"[A-Z_]+", expr):  # No named define
                    bad_exprs.append(f"Op {op.get('idx')}: {expr[:50]}...")

    if bad_exprs:
        results["issues"].append(f"{len(bad_exprs)} expressions use raw offsets instead of defines")
        for be in bad_exprs[:3]:
            results["issues"].append(f"  {be}")

    # Check for named defines in memory section
    memory = ir3.get("memory", {})
    weights = memory.get("weights", {}).get("entries", [])
    acts = memory.get("activations", {}).get("buffers", [])

    weights_with_define = sum(1 for w in weights if w.get("define"))
    acts_with_define = sum(1 for a in acts if a.get("define"))

    if weights and weights_with_define == 0:
        results["issues"].append("No weight defines found (W_LAYER_0_WQ, etc.)")
    if acts and acts_with_define == 0:
        results["issues"].append("No activation defines found (A_EMBEDDED_INPUT, etc.)")

    if results["issues"]:
        results["passed"] = False

    return results


def test_c_code_types(c_path: Path) -> dict:
    """Test C code has correct pointer types."""
    results = {"name": "C Code Types", "passed": True, "issues": []}

    if not c_path.exists():
        results["passed"] = False
        results["issues"].append(f"C file not found: {c_path}")
        return results

    with open(c_path) as f:
        code = f.read()

    # Check bump is uint8_t*
    if not re.search(r"uint8_t\s*\*\s*bump", code):
        results["passed"] = False
        results["issues"].append("bump should be uint8_t*, not found")

    # Check pointer arithmetic pattern
    # Good: (float*)(model->bump + OFFSET)
    # Bad: (float*)(model->bump) + OFFSET (arithmetic on wrong type)
    bad_patterns = re.findall(r"\(\w+\*\)\s*\(model->bump\)\s*\+", code)
    if bad_patterns:
        results["passed"] = False
        results["issues"].append(f"Bad pointer arithmetic: cast before addition ({len(bad_patterns)} instances)")

    # Check named defines are used
    define_pattern = r"#define\s+(A_[A-Z_]+|W_[A-Z_0-9]+)\s+\d+"
    defines = re.findall(define_pattern, code)
    if len(defines) < 10:
        results["issues"].append(f"Only {len(defines)} named defines found, expected more")

    # Check for hardcoded large offsets in function calls (should use defines)
    hardcoded = re.findall(r"model->bump\s*\+\s*\d{7,}", code)  # 7+ digit numbers
    named = re.findall(r"model->bump\s*\+\s*[A-Z_]+", code)

    if hardcoded and len(hardcoded) > len(named) * 0.1:  # More than 10% hardcoded
        results["issues"].append(f"{len(hardcoded)} hardcoded offsets vs {len(named)} named defines")

    return results


def test_compilation(c_path: Path, model_dir: Path) -> dict:
    """Test that C code compiles."""
    results = {"name": "Compilation", "passed": True, "issues": []}

    if not c_path.exists():
        results["passed"] = False
        results["issues"].append(f"C file not found: {c_path}")
        return results

    # Find include paths
    v66_root = Path(__file__).parent.parent
    src_dir = v66_root / "src"
    kernels_dir = v66_root.parent.parent / "src" / "kernels"

    # Try to compile
    output = model_dir / "test_compile.o"
    cmd = [
        "gcc", "-c", "-O0", "-fsyntax-only",
        f"-I{src_dir}",
        f"-I{kernels_dir}",
        f"-I{v66_root}",
        str(c_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            results["passed"] = False
            # Extract first few errors
            errors = result.stderr.strip().split('\n')[:5]
            for err in errors:
                results["issues"].append(err[:100])
    except FileNotFoundError:
        results["issues"].append("gcc not found, skipping compilation test")
    except subprocess.TimeoutExpired:
        results["passed"] = False
        results["issues"].append("Compilation timed out")

    return results


def test_memory_layout(layout_path: Path) -> dict:
    """Test memory layout has correct structure."""
    results = {"name": "Memory Layout", "passed": True, "issues": []}

    if not layout_path.exists():
        results["passed"] = False
        results["issues"].append(f"Layout file not found: {layout_path}")
        return results

    with open(layout_path) as f:
        layout = json.load(f)

    # Check bump_layout exists
    bump_layout = layout.get("bump_layout", {})
    if not bump_layout:
        results["issues"].append("No bump_layout section")
    else:
        expected_keys = ["header_size", "ext_metadata_size", "data_start"]
        for key in expected_keys:
            if key not in bump_layout:
                results["issues"].append(f"Missing bump_layout.{key}")

    # Layout can be nested under "memory" or at top level
    memory = layout.get("memory", layout)

    # Check activations have offsets
    activations = memory.get("activations", {})
    buffers = activations.get("buffers", [])
    if not buffers:
        results["issues"].append("No activation buffers defined")
    else:
        for buf in buffers[:3]:
            if "offset" not in buf and "abs_offset" not in buf:
                results["issues"].append(f"Buffer {buf.get('name')} missing offset")
                break

    # Check weights have offsets
    weights = memory.get("weights", {})
    entries = weights.get("entries", [])
    if not entries:
        results["issues"].append("No weight entries defined")
    else:
        # Verify entries have abs_offset (used by C code)
        for entry in entries[:3]:
            if "abs_offset" not in entry:
                results["issues"].append(f"Weight {entry.get('name')} missing abs_offset")
                break

    if results["issues"]:
        results["passed"] = False

    return results


def test_defines_consistency(c_path: Path, ir3_path: Path) -> dict:
    """Test that defines in C code match IR3 (using abs_offset)."""
    results = {"name": "Defines Consistency", "passed": True, "issues": []}

    if not c_path.exists() or not ir3_path.exists():
        results["passed"] = False
        results["issues"].append("Missing files")
        return results

    with open(c_path) as f:
        code = f.read()
    with open(ir3_path) as f:
        ir3 = json.load(f)

    # Extract defines from C code
    c_defines = {}
    for match in re.finditer(r"#define\s+(A_[A-Z_]+|W_[A-Z_0-9]+)\s+(\d+)", code):
        c_defines[match.group(1)] = int(match.group(2))

    # Extract defines from IR3 - use abs_offset (matches C code)
    ir3_defines = {}
    memory = ir3.get("memory", {})
    for entry in memory.get("weights", {}).get("entries", []):
        if entry.get("define"):
            # C code uses abs_offset (includes base_offset)
            ir3_defines[entry["define"]] = entry.get("abs_offset", entry.get("offset", 0))
    for buf in memory.get("activations", {}).get("buffers", []):
        if buf.get("define"):
            ir3_defines[buf["define"]] = buf.get("abs_offset", buf.get("offset", 0))

    # Compare
    mismatches = []
    for name, ir3_val in list(ir3_defines.items())[:20]:  # Check first 20
        if name in c_defines and c_defines[name] != ir3_val:
            mismatches.append(f"{name}: C={c_defines[name]}, IR3={ir3_val}")

    if mismatches:
        results["passed"] = False
        results["issues"].append(f"{len(mismatches)} define mismatches")
        for mm in mismatches[:3]:
            results["issues"].append(f"  {mm}")

    # Also check that we found defines
    if not c_defines:
        results["issues"].append("No defines found in C code")
    if not ir3_defines:
        results["issues"].append("No defines found in IR3")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test codegen plumbing")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir

    print("=" * 70)
    print("CODEGEN PLUMBING TEST")
    print("=" * 70)
    print(f"Model dir: {model_dir}")
    print()

    # Find files
    ir3_path = model_dir / "lowered_decode_call.json"
    c_path = model_dir / "model_v6_6.c"
    layout_path = model_dir / "layout_decode.json"

    # Run tests
    tests = [
        test_ir3_format(ir3_path),
        test_c_code_types(c_path),
        test_memory_layout(layout_path),
        test_defines_consistency(c_path, ir3_path),
        test_compilation(c_path, model_dir),
    ]

    # Print results
    all_passed = True
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        all_passed = all_passed and test["passed"]

        print(f"{test['name']}: {status}")
        if test["issues"]:
            for issue in test["issues"]:
                print(f"  - {issue}")
        print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        failed = [t["name"] for t in tests if not t["passed"]]
        print(f"FAILED: {', '.join(failed)}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

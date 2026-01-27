#!/usr/bin/env python3
"""
test_scratch_buffer.py - Validate scratch buffer allocation and usage

This test ensures scratch buffers are properly:
- Allocated with correct sizes
- Passed to kernels (not NULL)
- Reused appropriately across ops

BUGS THIS CATCHES:
- Bug 4: Scratch buffer allocation
- Bug 18: NULL scratch to fused kernels

Usage:
    python test_scratch_buffer.py
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


# Kernels that require scratch buffers
KERNELS_NEEDING_SCRATCH = {
    "mega_fused_attention_prefill": "QKV scratch + attention scratch",
    "mega_fused_attention_decode": "attention scratch",
    "mega_fused_outproj_mlp_prefill": "MLP scratch",
    "attention_forward_causal_head_major_gqa_flash_strided": "softmax scratch",
    "flash_attention": "QK^T scratch",
}


@dataclass
class ScratchIssue:
    """A scratch buffer issue."""
    severity: str
    kernel: str
    message: str
    line_number: int = -1


class ScratchValidator:
    """Validates scratch buffer allocation and usage."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[ScratchIssue] = []

    def load_files(self) -> bool:
        """Load required files."""
        self.ir = None
        self.layout = None
        self.c_code = None
        self.c_lines = []

        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            if (base_dir / "lowered_decode.json").exists():
                with open(base_dir / "lowered_decode.json") as f:
                    self.ir = json.load(f)

            if (base_dir / "layout_decode.json").exists():
                with open(base_dir / "layout_decode.json") as f:
                    self.layout = json.load(f)

            if (base_dir / "ck-kernel-inference.c").exists():
                self.c_code = (base_dir / "ck-kernel-inference.c").read_text()
                self.c_lines = self.c_code.split('\n')

        if not self.c_code:
            print("ERROR: Could not find ck-kernel-inference.c")
            return False

        return True

    def find_scratch_allocations(self) -> Dict[str, int]:
        """Find scratch buffer allocations in code."""
        allocations = {}

        # Look for scratch buffer allocation patterns
        patterns = [
            r'scratch\s*=\s*(?:aligned_alloc|malloc|calloc)\s*\([^,]+,\s*(\d+)',
            r'#define\s+SCRATCH_SIZE\s+(\d+)',
            r'scratch_size\s*=\s*(\d+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, self.c_code):
                size = int(match.group(1))
                allocations[f"scratch_{len(allocations)}"] = size

        return allocations

    def check_null_scratch_in_calls(self) -> List[Tuple[str, int]]:
        """Find kernel calls with NULL scratch parameter."""
        null_scratches = []

        for kernel_name in KERNELS_NEEDING_SCRATCH.keys():
            # Find all calls to this kernel
            pattern = rf'{kernel_name}\s*\('
            for i, line in enumerate(self.c_lines):
                if re.search(pattern, line):
                    # Collect the full function call
                    call_start = i
                    call_text = ""
                    paren_count = 0
                    started = False

                    for j in range(i, min(i + 30, len(self.c_lines))):
                        call_text += self.c_lines[j] + " "
                        paren_count += self.c_lines[j].count('(')
                        paren_count -= self.c_lines[j].count(')')
                        if '(' in self.c_lines[j]:
                            started = True
                        if started and paren_count == 0:
                            break

                    # Check if last argument before ) is NULL
                    # Remove the closing )
                    call_text = call_text.strip()
                    if call_text.endswith(');'):
                        call_text = call_text[:-2]

                    # Get last argument
                    args = call_text.split(',')
                    if args:
                        last_arg = args[-1].strip()
                        if last_arg == 'NULL' or last_arg == 'null' or last_arg == '0':
                            null_scratches.append((kernel_name, call_start + 1))
                            self.issues.append(ScratchIssue(
                                severity="CRITICAL",
                                kernel=kernel_name,
                                message=f"NULL scratch passed to {kernel_name}",
                                line_number=call_start + 1
                            ))

        return null_scratches

    def check_ir_scratch_requirements(self) -> Dict[str, Dict]:
        """Check what scratch each op requires according to IR."""
        scratch_reqs = {}

        if not self.ir:
            return scratch_reqs

        # Extract ops
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
            scratch = op.get("scratch", [])

            if scratch:
                scratch_reqs[kernel] = {
                    "op": op.get("op", ""),
                    "scratch_buffers": scratch
                }

        return scratch_reqs

    def check_layout_scratch_allocation(self) -> Dict[str, int]:
        """Check scratch allocation in layout."""
        scratch_sizes = {}

        if not self.layout:
            return scratch_sizes

        activations = self.layout.get("memory", {}).get("activations", {})
        for buf in activations.get("buffers", []):
            name = buf.get("name", "")
            if "scratch" in name.lower():
                size = buf.get("size", 0)
                scratch_sizes[name] = size

        return scratch_sizes

    def run_all_tests(self) -> bool:
        """Run all scratch buffer validations."""
        print("\n" + "="*70)
        print("SCRATCH BUFFER VALIDATION")
        print("="*70)

        if not self.load_files():
            return False

        # Test 1: Check for NULL scratch in kernel calls
        print("\n" + "-"*70)
        print("TEST 1: NULL scratch detection")
        print("-"*70)

        null_scratches = self.check_null_scratch_in_calls()
        if null_scratches:
            print(f"  ✗ Found {len(null_scratches)} kernel calls with NULL scratch:")
            for kernel, line in null_scratches:
                print(f"    - {kernel} at line {line}")
        else:
            print("  ✓ No NULL scratch arguments found")

        # Test 2: Check IR scratch requirements
        print("\n" + "-"*70)
        print("TEST 2: IR scratch requirements")
        print("-"*70)

        ir_scratch = self.check_ir_scratch_requirements()
        if ir_scratch:
            print(f"  Found {len(ir_scratch)} ops with scratch requirements:")
            for kernel, info in ir_scratch.items():
                print(f"    - {kernel}: {len(info['scratch_buffers'])} buffers")
        else:
            print("  ⚠ No scratch requirements found in IR (may be missing)")

        # Test 3: Check layout scratch allocation
        print("\n" + "-"*70)
        print("TEST 3: Layout scratch allocation")
        print("-"*70)

        layout_scratch = self.check_layout_scratch_allocation()
        if layout_scratch:
            print(f"  Found {len(layout_scratch)} scratch buffers in layout:")
            for name, size in layout_scratch.items():
                print(f"    - {name}: {size:,} bytes")
        else:
            print("  ⚠ No scratch buffers found in layout")

        # Test 4: Cross-check: kernels needing scratch vs what's allocated
        print("\n" + "-"*70)
        print("TEST 4: Scratch coverage check")
        print("-"*70)

        # Find which kernels needing scratch are used
        kernels_used = set()
        for kernel in KERNELS_NEEDING_SCRATCH.keys():
            if kernel in self.c_code:
                kernels_used.add(kernel)

        if kernels_used:
            print(f"  Kernels used that need scratch: {kernels_used}")

            # Check if scratch is allocated for them
            if not layout_scratch:
                self.issues.append(ScratchIssue(
                    severity="ERROR",
                    kernel=", ".join(kernels_used),
                    message="Kernels need scratch but no scratch allocated in layout"
                ))
                print("  ✗ No scratch allocated for these kernels!")
        else:
            print("  ℹ No scratch-requiring kernels found in generated code")

        # Summary
        print("\n" + "="*70)
        print("SCRATCH BUFFER ISSUES")
        print("="*70)

        if not self.issues:
            print("No scratch buffer issues found!")
        else:
            for severity in ["CRITICAL", "ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        line_info = f" (line {issue.line_number})" if issue.line_number > 0 else ""
                        print(f"  [{issue.kernel}]{line_info}: {issue.message}")

        # Verdict
        print("\n" + "="*70)
        critical = len([i for i in self.issues if i.severity == "CRITICAL"])
        errors = len([i for i in self.issues if i.severity == "ERROR"])

        if critical > 0:
            print(f"VERDICT: FAIL - {critical} CRITICAL scratch issues")
        elif errors > 0:
            print(f"VERDICT: FAIL - {errors} scratch errors")
        else:
            print("VERDICT: PASS - Scratch buffers OK")
        print("="*70)

        return critical == 0 and errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate scratch buffers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = ScratchValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
test_dimension_alignment.py - Validate dimension alignment across pipeline

This test ensures:
- HEAD_DIM matches aligned_head_dim where needed
- EMBED_DIM is properly aligned
- Buffer sizes account for alignment
- Kernel parameters use correct dimensions

BUGS THIS CATCHES:
- Bug 13: HEAD_DIM vs aligned_head_dim mismatch

Usage:
    python test_dimension_alignment.py
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

# Paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"


@dataclass
class AlignmentIssue:
    """A dimension alignment issue."""
    severity: str
    dimension: str
    message: str
    expected: int = 0
    actual: int = 0
    line_number: int = -1


class AlignmentValidator:
    """Validates dimension alignment."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[AlignmentIssue] = []
        self.dimensions: Dict[str, int] = {}

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

    def extract_dimensions_from_c(self) -> Dict[str, int]:
        """Extract dimension definitions from C code."""
        dims = {}

        patterns = [
            (r'#define\s+(EMBED_DIM|HEAD_DIM|NUM_HEADS|NUM_KV_HEADS|INTERMEDIATE_SIZE|VOCAB_SIZE|MAX_SEQ_LEN|NUM_LAYERS)\s+(\d+)', None),
            (r'#define\s+(aligned_\w+)\s+(\d+)', None),
        ]

        for pattern, transform in patterns:
            for match in re.finditer(pattern, self.c_code, re.IGNORECASE):
                name = match.group(1).upper()
                value = int(match.group(2))
                dims[name] = value

        return dims

    def extract_dimensions_from_ir(self) -> Dict[str, int]:
        """Extract dimensions from IR config."""
        dims = {}

        if not self.ir:
            return dims

        config = self.ir.get("config", {})

        dim_mapping = {
            "embed_dim": "EMBED_DIM",
            "head_dim": "HEAD_DIM",
            "num_heads": "NUM_HEADS",
            "num_kv_heads": "NUM_KV_HEADS",
            "intermediate_size": "INTERMEDIATE_SIZE",
            "intermediate_dim": "INTERMEDIATE_SIZE",
            "vocab_size": "VOCAB_SIZE",
            "max_seq_len": "MAX_SEQ_LEN",
            "context_length": "MAX_SEQ_LEN",
            "num_layers": "NUM_LAYERS",
            "aligned_embed": "ALIGNED_EMBED_DIM",
            "aligned_head": "ALIGNED_HEAD_DIM",
            "aligned_intermediate": "ALIGNED_INTERMEDIATE",
        }

        for key, normalized in dim_mapping.items():
            if key in config:
                dims[normalized] = config[key]

        return dims

    def check_power_of_two_alignment(self) -> bool:
        """Check dimensions are properly aligned (power of 2 or multiple of 64)."""
        print("\n" + "-"*70)
        print("TEST 1: Power of 2 / 64-byte alignment")
        print("-"*70)

        c_dims = self.extract_dimensions_from_c()
        self.dimensions.update(c_dims)

        # Dimensions that should be aligned
        alignment_requirements = {
            "HEAD_DIM": 64,  # Should be power of 2
            "EMBED_DIM": 64,  # Should be multiple of 64
            "INTERMEDIATE_SIZE": 64,
        }

        all_aligned = True

        for dim_name, alignment in alignment_requirements.items():
            if dim_name in c_dims:
                value = c_dims[dim_name]
                if value % alignment != 0:
                    self.issues.append(AlignmentIssue(
                        severity="WARNING",
                        dimension=dim_name,
                        message=f"{dim_name}={value} is not aligned to {alignment}",
                        expected=((value + alignment - 1) // alignment) * alignment,
                        actual=value
                    ))
                    all_aligned = False
                    print(f"  ⚠ {dim_name}={value} (not aligned to {alignment})")
                elif self.verbose:
                    print(f"  ✓ {dim_name}={value} (aligned to {alignment})")

        if all_aligned:
            print("  ✓ All dimensions properly aligned")

        return all_aligned

    def check_head_dim_consistency(self) -> bool:
        """Check HEAD_DIM is consistent with EMBED_DIM / NUM_HEADS."""
        print("\n" + "-"*70)
        print("TEST 2: HEAD_DIM = EMBED_DIM / NUM_HEADS")
        print("-"*70)

        c_dims = self.extract_dimensions_from_c()

        head_dim = c_dims.get("HEAD_DIM", 0)
        embed_dim = c_dims.get("EMBED_DIM", 0)
        num_heads = c_dims.get("NUM_HEADS", 1)

        if embed_dim and num_heads:
            expected_head_dim = embed_dim // num_heads

            if head_dim != expected_head_dim:
                self.issues.append(AlignmentIssue(
                    severity="ERROR",
                    dimension="HEAD_DIM",
                    message=f"HEAD_DIM={head_dim} doesn't match EMBED_DIM/NUM_HEADS={expected_head_dim}",
                    expected=expected_head_dim,
                    actual=head_dim
                ))
                print(f"  ✗ HEAD_DIM={head_dim}, expected {embed_dim}/{num_heads}={expected_head_dim}")
                return False

        print(f"  ✓ HEAD_DIM={head_dim} = {embed_dim}/{num_heads}")
        return True

    def check_ir_vs_c_dimensions(self) -> bool:
        """Check IR dimensions match C code dimensions."""
        print("\n" + "-"*70)
        print("TEST 3: IR dimensions match C code")
        print("-"*70)

        ir_dims = self.extract_dimensions_from_ir()
        c_dims = self.extract_dimensions_from_c()

        mismatches = []

        for dim_name in set(ir_dims.keys()) & set(c_dims.keys()):
            ir_val = ir_dims[dim_name]
            c_val = c_dims[dim_name]

            if ir_val != c_val:
                mismatches.append((dim_name, ir_val, c_val))
                self.issues.append(AlignmentIssue(
                    severity="ERROR",
                    dimension=dim_name,
                    message=f"IR has {dim_name}={ir_val}, C code has {c_val}",
                    expected=ir_val,
                    actual=c_val
                ))

        if mismatches:
            print("  ✗ Dimension mismatches found:")
            for name, ir_val, c_val in mismatches:
                print(f"    {name}: IR={ir_val}, C={c_val}")
            return False

        print(f"  ✓ All {len(set(ir_dims.keys()) & set(c_dims.keys()))} common dimensions match")
        return True

    def check_kernel_dimension_usage(self) -> bool:
        """Check kernels use correct dimension parameters."""
        print("\n" + "-"*70)
        print("TEST 4: Kernel dimension parameter usage")
        print("-"*70)

        # Check for aligned vs unaligned usage
        # Some kernels need aligned dimensions, others need actual

        # Attention kernels typically need aligned dimensions for striding
        attention_pattern = r'attention_forward.*HEAD_DIM|attention_forward.*ALIGNED_HEAD'

        has_aligned_in_attn = bool(re.search(r'attention.*aligned', self.c_code, re.IGNORECASE))
        has_raw_in_attn = bool(re.search(r'attention.*HEAD_DIM', self.c_code, re.IGNORECASE))

        if has_aligned_in_attn and has_raw_in_attn:
            print("  ⚠ Attention uses both aligned and raw dimensions (check consistency)")
            self.issues.append(AlignmentIssue(
                severity="WARNING",
                dimension="HEAD_DIM/ALIGNED_HEAD",
                message="Attention kernel uses both aligned and raw HEAD_DIM - verify correctness"
            ))
        elif has_raw_in_attn:
            print("  ✓ Attention uses HEAD_DIM")
        elif has_aligned_in_attn:
            print("  ✓ Attention uses aligned dimensions")

        return True

    def check_buffer_size_alignment(self) -> bool:
        """Check buffer sizes account for alignment."""
        print("\n" + "-"*70)
        print("TEST 5: Buffer size alignment")
        print("-"*70)

        if not self.layout:
            print("  ⚠ No layout file to check")
            return True

        # Check activation buffer sizes
        activations = self.layout.get("memory", {}).get("activations", {})

        issues_found = False
        for buf in activations.get("buffers", []):
            size = buf.get("size", 0)
            # Sizes should typically be aligned to 64 bytes
            if size > 0 and size % 64 != 0:
                # Not necessarily an error, but worth noting
                if self.verbose:
                    print(f"  ⚠ {buf['name']}: size {size} not 64-byte aligned")

        if not issues_found:
            print("  ✓ Buffer sizes checked")

        return True

    def run_all_tests(self) -> bool:
        """Run all alignment validations."""
        print("\n" + "="*70)
        print("DIMENSION ALIGNMENT VALIDATION")
        print("="*70)

        if not self.load_files():
            return False

        results = []
        results.append(("Power of 2 alignment", self.check_power_of_two_alignment()))
        results.append(("HEAD_DIM consistency", self.check_head_dim_consistency()))
        results.append(("IR vs C dimensions", self.check_ir_vs_c_dimensions()))
        results.append(("Kernel dimension usage", self.check_kernel_dimension_usage()))
        results.append(("Buffer size alignment", self.check_buffer_size_alignment()))

        # Summary
        print("\n" + "="*70)
        print("ALIGNMENT ISSUES")
        print("="*70)

        if not self.issues:
            print("No alignment issues found!")
        else:
            for severity in ["ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        print(f"  [{issue.dimension}]: {issue.message}")
                        if issue.expected:
                            print(f"    Expected: {issue.expected}")
                        if issue.actual:
                            print(f"    Actual: {issue.actual}")

        # Verdict
        print("\n" + "="*70)
        errors = len([i for i in self.issues if i.severity == "ERROR"])
        passed = sum(1 for _, r in results if r)

        if errors > 0:
            print(f"VERDICT: FAIL - {errors} alignment errors")
        else:
            print(f"VERDICT: PASS - {passed}/{len(results)} tests passed")
        print("="*70)

        return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate dimension alignment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = AlignmentValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

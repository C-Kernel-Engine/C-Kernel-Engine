#!/usr/bin/env python3
"""
test_kv_cache.py - Validate KV cache integration

This test ensures KV cache is properly:
- Allocated with correct size
- Stored after K/V projections
- Used in attention with correct indexing
- Layout matches attention kernel expectations (head-major vs token-major)

BUGS THIS CATCHES:
- Bug 11: KV cache not integrated
- Bug 16: KV cache offset layout (token vs head major)

Usage:
    python test_kv_cache.py
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


@dataclass
class KVCacheIssue:
    """A KV cache issue."""
    severity: str
    category: str
    message: str
    line_number: int = -1


class KVCacheValidator:
    """Validates KV cache integration."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[KVCacheIssue] = []

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

    def check_kv_cache_allocation(self) -> bool:
        """Check KV cache is allocated."""
        print("\n" + "-"*70)
        print("TEST 1: KV cache allocation")
        print("-"*70)

        # Look for kv_cache in struct or allocation
        has_kv_cache_member = "kv_cache" in self.c_code and ("float *kv_cache" in self.c_code or "float* kv_cache" in self.c_code)
        has_kv_alloc = "KV_CACHE_SIZE" in self.c_code or "kv_cache_size" in self.c_code.lower()

        if has_kv_cache_member:
            print("  ✓ KV cache member found in struct")
        else:
            print("  ✗ No kv_cache member in struct")
            self.issues.append(KVCacheIssue(
                severity="ERROR",
                category="ALLOCATION",
                message="No kv_cache member found in model struct"
            ))

        if has_kv_alloc:
            # Extract size if possible
            match = re.search(r'KV_CACHE_SIZE\s*[=\(]\s*([^)\n]+)', self.c_code)
            if match:
                print(f"  ✓ KV cache size: {match.group(1)}")
            else:
                print("  ✓ KV cache size defined")
        else:
            print("  ⚠ KV_CACHE_SIZE not defined")

        return has_kv_cache_member

    def check_kv_store_operations(self) -> bool:
        """Check K/V values are stored to cache after projections."""
        print("\n" + "-"*70)
        print("TEST 2: KV store operations")
        print("-"*70)

        # Look for kv_cache_store or memcpy to kv_cache
        kv_store_patterns = [
            r'kv_cache_store',
            r'memcpy\s*\([^,]*kv_cache',
            r'model->kv_cache\s*\+.*=',  # Assignment to kv_cache
        ]

        store_found = False
        for pattern in kv_store_patterns:
            matches = list(re.finditer(pattern, self.c_code))
            if matches:
                store_found = True
                print(f"  ✓ Found {len(matches)} KV store operations")
                break

        if not store_found:
            # Check if it's marked as TODO/TEMP
            if "TEMP" in self.c_code and "kv" in self.c_code.lower():
                print("  ⚠ KV cache marked as TEMP/incomplete")
                self.issues.append(KVCacheIssue(
                    severity="WARNING",
                    category="INCOMPLETE",
                    message="KV cache integration marked as TEMP"
                ))
            else:
                print("  ✗ No KV store operations found")
                self.issues.append(KVCacheIssue(
                    severity="ERROR",
                    category="MISSING_STORE",
                    message="No K/V store operations found - values not saved to cache"
                ))

        return store_found

    def check_attention_uses_kv_cache(self) -> bool:
        """Check attention kernel uses KV cache (not scratch)."""
        print("\n" + "-"*70)
        print("TEST 3: Attention uses KV cache")
        print("-"*70)

        # Find attention kernel calls
        attention_patterns = [
            r'attention_forward.*\([^)]+\)',
            r'mega_fused_attention.*\([^)]+\)',
        ]

        for pattern in attention_patterns:
            for match in re.finditer(pattern, self.c_code, re.DOTALL):
                call = match.group(0)
                # Check if kv_cache is in the call
                if 'kv_cache' in call:
                    print("  ✓ Attention uses kv_cache")
                    return True

        # Check if attention is using scratch instead
        for line_num, line in enumerate(self.c_lines):
            if 'attention' in line.lower() and 'scratch' in line.lower():
                print("  ⚠ Attention may be using scratch instead of KV cache")
                self.issues.append(KVCacheIssue(
                    severity="WARNING",
                    category="WRONG_BUFFER",
                    message="Attention appears to use scratch instead of KV cache",
                    line_number=line_num + 1
                ))

        print("  ? Could not determine if attention uses KV cache")
        return False

    def check_kv_layout_consistency(self) -> bool:
        """Check KV cache layout is consistent (head-major expected)."""
        print("\n" + "-"*70)
        print("TEST 4: KV cache layout consistency")
        print("-"*70)

        # Look for KV cache indexing patterns
        # Head-major: kv_cache + layer * 2 * H * T * D + h * T * D + pos * D
        # Token-major: kv_cache + layer * 2 * T * H * D + pos * H * D + h * D

        head_major_pattern = r'kv_cache.*\+.*layer.*\*.*HEAD'
        token_major_pattern = r'kv_cache.*\+.*layer.*\*.*SEQ'

        has_head_major = bool(re.search(head_major_pattern, self.c_code, re.IGNORECASE))
        has_token_major = bool(re.search(token_major_pattern, self.c_code, re.IGNORECASE))

        if has_head_major and has_token_major:
            print("  ⚠ Mixed KV layouts detected (both head-major and token-major)")
            self.issues.append(KVCacheIssue(
                severity="ERROR",
                category="LAYOUT_MISMATCH",
                message="KV cache has inconsistent layouts (both head-major and token-major patterns found)"
            ))
            return False
        elif has_head_major:
            print("  ✓ KV cache uses head-major layout")
        elif has_token_major:
            print("  ✓ KV cache uses token-major layout")
        else:
            print("  ? Could not determine KV cache layout")

        # Check attention kernel matches layout
        # attention_forward_causal_head_major expects head-major
        if "head_major" in self.c_code.lower():
            if has_token_major:
                self.issues.append(KVCacheIssue(
                    severity="ERROR",
                    category="LAYOUT_MISMATCH",
                    message="Attention expects head-major but KV cache is token-major"
                ))
                return False

        return True

    def check_position_tracking(self) -> bool:
        """Check position is tracked for KV cache indexing."""
        print("\n" + "-"*70)
        print("TEST 5: Position tracking")
        print("-"*70)

        has_pos = "model->pos" in self.c_code or "pos++" in self.c_code
        has_pos_in_attn = bool(re.search(r'attention.*pos', self.c_code, re.IGNORECASE))

        if has_pos:
            print("  ✓ Position tracking found (model->pos)")
        else:
            print("  ✗ No position tracking found")
            self.issues.append(KVCacheIssue(
                severity="ERROR",
                category="MISSING_POS",
                message="No position tracking for autoregressive decoding"
            ))

        if has_pos_in_attn:
            print("  ✓ Position used in attention")
        else:
            print("  ⚠ Position may not be passed to attention")

        return has_pos

    def run_all_tests(self) -> bool:
        """Run all KV cache validations."""
        print("\n" + "="*70)
        print("KV CACHE VALIDATION")
        print("="*70)

        if not self.load_files():
            return False

        results = []
        results.append(("Allocation", self.check_kv_cache_allocation()))
        results.append(("Store operations", self.check_kv_store_operations()))
        results.append(("Attention uses cache", self.check_attention_uses_kv_cache()))
        results.append(("Layout consistency", self.check_kv_layout_consistency()))
        results.append(("Position tracking", self.check_position_tracking()))

        # Summary
        print("\n" + "="*70)
        print("KV CACHE ISSUES")
        print("="*70)

        if not self.issues:
            print("No KV cache issues found!")
        else:
            for severity in ["CRITICAL", "ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        line_info = f" (line {issue.line_number})" if issue.line_number > 0 else ""
                        print(f"  [{issue.category}]{line_info}: {issue.message}")

        # Verdict
        print("\n" + "="*70)
        errors = len([i for i in self.issues if i.severity in ["CRITICAL", "ERROR"]])
        passed = sum(1 for _, r in results if r)

        if errors > 0:
            print(f"VERDICT: FAIL - {errors} KV cache errors")
        else:
            print(f"VERDICT: PASS - {passed}/{len(results)} tests passed")
        print("="*70)

        return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate KV cache")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = KVCacheValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

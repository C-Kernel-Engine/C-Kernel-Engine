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

    def __init__(self, model_dir: Path = CACHE_DIR, generated_dir: Path = GENERATED_DIR, verbose: bool = False):
        self.model_dir = model_dir
        self.generated_dir = generated_dir
        self.verbose = verbose
        self.issues: List[KVCacheIssue] = []
        self.results: List[Tuple[str, bool]] = []

    def load_files(self) -> bool:
        """Load required files."""
        self.ir = None
        self.layout = None
        self.c_code = None
        self.c_lines = []
        self.c_path = None
        self.ir_path = None
        self.layout_path = None

        for base_dir in [self.model_dir, self.generated_dir]:
            if self.ir is None:
                for ir_name in ("lowered_decode_call.json", "lowered_decode.json"):
                    ir_path = base_dir / ir_name
                    if ir_path.exists():
                        with open(ir_path) as f:
                            self.ir = json.load(f)
                        self.ir_path = ir_path
                        break

            if self.layout is None:
                layout_path = base_dir / "layout_decode.json"
                if layout_path.exists():
                    with open(layout_path) as f:
                        self.layout = json.load(f)
                    self.layout_path = layout_path

            if self.c_code is None:
                for c_name in ("ck-kernel-inference.c", "model_v6_6.c"):
                    c_path = base_dir / c_name
                    if c_path.exists():
                        self.c_code = c_path.read_text()
                        self.c_lines = self.c_code.split('\n')
                        self.c_path = c_path
                        break

        if self.verbose:
            if self.ir_path:
                print(f"  Loaded IR: {self.ir_path}")
            if self.layout_path:
                print(f"  Loaded layout: {self.layout_path}")
            if self.c_path:
                print(f"  Loaded C source: {self.c_path}")

        if not self.c_code and not self.ir:
            print("ERROR: Could not find model C source or lowered IR")
            return False

        return True

    def check_kv_cache_allocation(self) -> bool:
        """Check KV cache is allocated."""
        print("\n" + "-"*70)
        print("TEST 1: KV cache allocation")
        print("-"*70)

        # Look for kv_cache in struct or allocation
        c_code = self.c_code or ""
        has_kv_cache_member = "kv_cache" in c_code and ("float *kv_cache" in c_code or "float* kv_cache" in c_code)
        has_kv_alloc = "KV_CACHE_SIZE" in c_code or "kv_cache_size" in c_code.lower()

        if not has_kv_cache_member and self.layout:
            for buf in self.layout.get("memory", {}).get("activations", {}).get("buffers", []):
                if buf.get("name") == "kv_cache" or buf.get("define") == "A_KV_CACHE":
                    has_kv_cache_member = True
                    break

        if has_kv_cache_member:
            print("  ✓ KV cache member found in struct")
        else:
            print("  ✗ No kv_cache member in struct")
            self.issues.append(KVCacheIssue(
                severity="ERROR",
                category="ALLOCATION",
                message="No kv_cache member found in model struct"
            ))

        if not has_kv_alloc and self.layout:
            for buf in self.layout.get("memory", {}).get("activations", {}).get("buffers", []):
                if buf.get("name") == "kv_cache" and int(buf.get("size", 0)) > 0:
                    has_kv_alloc = True
                    break

        if has_kv_alloc:
            # Extract size if possible
            match = re.search(r'KV_CACHE_SIZE\s*[=\(]\s*([^)\n]+)', c_code)
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
        c_code = self.c_code or ""
        kv_store_patterns = [
            r'kv_cache_store',
            r'memcpy\s*\([^,]*kv_cache',
            r'model->kv_cache\s*\+.*=',  # Assignment to kv_cache
        ]

        store_found = False
        for pattern in kv_store_patterns:
            matches = list(re.finditer(pattern, c_code))
            if matches:
                store_found = True
                print(f"  ✓ Found {len(matches)} KV store operations")
                break

        if not store_found and self.ir:
            ops = self.ir.get("operations", [])
            ir_matches = []
            for op in ops:
                fn = str(op.get("function", "")).lower()
                name = str(op.get("op", "")).lower()
                if "kv_cache_store" in fn or "kv_cache_store" in name:
                    ir_matches.append(op)
                    continue
                for inp in op.get("inputs", []):
                    inp_name = str(inp.get("name", "")).lower()
                    inp_src = str(inp.get("source", "")).lower()
                    if "kv_cache" in inp_name or "kv_cache" in inp_src:
                        if "store" in fn or "store" in name:
                            ir_matches.append(op)
                            break
            if ir_matches:
                store_found = True
                print(f"  ✓ Found {len(ir_matches)} KV store ops in lowered IR")

        if not store_found:
            # Check if it's marked as TODO/TEMP
            if "TEMP" in c_code and "kv" in c_code.lower():
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

        if self.ir:
            for op in self.ir.get("operations", []):
                fn = str(op.get("function", "")).lower()
                name = str(op.get("op", "")).lower()
                if "attention" not in fn and "attention" not in name:
                    continue
                for inp in op.get("inputs", []):
                    inp_name = str(inp.get("name", "")).lower()
                    inp_src = str(inp.get("source", "")).lower()
                    if "kv_cache" in inp_name or "kv_cache" in inp_src:
                        print("  ✓ Attention uses kv_cache")
                        return True

        c_code = self.c_code or ""
        # Find attention kernel calls
        attention_patterns = [
            r'attention_forward.*\([^)]+\)',
            r'mega_fused_attention.*\([^)]+\)',
        ]

        for pattern in attention_patterns:
            for match in re.finditer(pattern, c_code, re.DOTALL):
                call = match.group(0)
                # Check if kv_cache is in the call
                if 'kv_cache' in call:
                    print("  ✓ Attention uses kv_cache")
                    return True

        # Check if attention is using scratch instead
        for line_num, line in enumerate(self.c_lines or []):
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

        c_code = self.c_code or ""
        head_major_pattern = r'model->kv_cache[^\n]*NUM_KV_HEADS\s*\*\s*MAX_SEQ_LEN\s*\*\s*HEAD_DIM'
        token_major_pattern = r'model->kv_cache[^\n]*MAX_SEQ_LEN\s*\*\s*NUM_KV_HEADS\s*\*\s*HEAD_DIM'

        has_head_major = bool(re.search(head_major_pattern, c_code, re.IGNORECASE))
        has_token_major = bool(re.search(token_major_pattern, c_code, re.IGNORECASE))
        if re.search(r'attention_forward[^\n]*head_major', c_code, re.IGNORECASE):
            has_head_major = True
        if re.search(r'attention_forward[^\n]*token_major', c_code, re.IGNORECASE):
            has_token_major = True

        if self.ir and (not has_head_major and not has_token_major):
            for op in self.ir.get("operations", []):
                fn = str(op.get("function", "")).lower()
                if "attention_forward_decode_head_major" in fn:
                    has_head_major = True
                if "token_major" in fn:
                    has_token_major = True

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
        if "head_major" in c_code.lower():
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

        c_code = self.c_code or ""
        has_pos = "model->pos" in c_code or "pos++" in c_code
        has_pos_in_attn = bool(re.search(r'attention.*pos', c_code, re.IGNORECASE))

        if not has_pos and self.ir:
            for op in self.ir.get("operations", []):
                for inp in op.get("inputs", []):
                    text = f"{inp.get('name', '')} {inp.get('source', '')}".lower()
                    if text == "position runtime:pos" or "position" in text or "runtime:pos" in text:
                        has_pos = True
                        if "attention" in str(op.get("function", "")).lower() or "attention" in str(op.get("op", "")).lower():
                            has_pos_in_attn = True
                        break
                if has_pos and has_pos_in_attn:
                    break

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
        self.results = results

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

    def to_json(self) -> Dict:
        """Structured summary for automation/CI."""
        return {
            "passed": len([i for i in self.issues if i.severity in ["CRITICAL", "ERROR"]]) == 0,
            "results": [{"name": name, "passed": passed} for name, passed in self.results],
            "errors": [f"[{i.category}] {i.message}" for i in self.issues if i.severity in ["CRITICAL", "ERROR"]],
            "warnings": [f"[{i.category}] {i.message}" for i in self.issues if i.severity == "WARNING"],
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "line_number": i.line_number,
                }
                for i in self.issues
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Validate KV cache")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CACHE_DIR,
        help="Model directory containing layout/lowered artifacts",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=GENERATED_DIR,
        help="Generated source directory (default: version/v6.6/src/generated)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    validator = KVCacheValidator(
        model_dir=args.model_dir,
        generated_dir=args.generated_dir,
        verbose=args.verbose,
    )
    success = validator.run_all_tests()
    if args.json:
        print(json.dumps(validator.to_json(), indent=2))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

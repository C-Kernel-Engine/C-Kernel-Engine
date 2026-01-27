#!/usr/bin/env python3
"""
test_codegen_respects_ir.py - CRITICAL: Verify codegen uses IR, not hardcoded values

This is the ROOT CAUSE test for most v6.6 bugs. The codegen MUST:
1. Read ops from lowered IR (not generate its own)
2. Use offsets from IR/layout (not hardcoded defaults)
3. Handle ALL kernel types (not fall back to TODO)
4. Pass correct scratch buffers (not NULL)

BUGS THIS CATCHES:
- Bug 17: Token offset mismatch (0 vs 3604)
- Bug 18: NULL scratch to fused kernels
- Bug 19: Codegen ignores IR lower
- Bug 6: Fused kernel issues

Usage:
    python test_codegen_respects_ir.py
    python test_codegen_respects_ir.py --verbose
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
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class CodegenIssue:
    """An issue found in codegen output."""
    severity: str  # "CRITICAL", "ERROR", "WARNING"
    category: str
    message: str
    line_number: int = -1
    code_snippet: str = ""


class CodegenValidator:
    """Validates that codegen respects IR and layout."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[CodegenIssue] = []
        self.results: List[TestResult] = []

    def load_files(self) -> bool:
        """Load all required files."""
        # Try generated dir first, then cache dir
        dirs_to_try = [GENERATED_DIR, CACHE_DIR]

        self.lowered_ir = None
        self.layout = None
        self.c_code = None
        self.c_code_lines = []

        for base_dir in dirs_to_try:
            lowered_path = base_dir / "lowered_decode.json"
            layout_path = base_dir / "layout_decode.json"
            c_path = base_dir / "ck-kernel-inference.c"

            if lowered_path.exists():
                with open(lowered_path) as f:
                    self.lowered_ir = json.load(f)

            if layout_path.exists():
                with open(layout_path) as f:
                    self.layout = json.load(f)

            if c_path.exists():
                self.c_code = c_path.read_text()
                self.c_code_lines = self.c_code.split('\n')
                self.c_path = c_path

        if not self.lowered_ir:
            print("ERROR: Could not find lowered_decode.json")
            return False
        if not self.layout:
            print("ERROR: Could not find layout_decode.json")
            return False
        if not self.c_code:
            print("ERROR: Could not find ck-kernel-inference.c")
            return False

        print(f"Loaded files from: {self.c_path.parent}")
        return True

    def extract_ir_ops(self) -> List[Dict]:
        """Extract ops from lowered IR (handles nested structure)."""
        ops = []

        # Try top-level first
        if "operations" in self.lowered_ir:
            ops = self.lowered_ir["operations"]
        elif "ops" in self.lowered_ir:
            ops = self.lowered_ir["ops"]
        else:
            # Try nested in sections
            for section in self.lowered_ir.get("sections", []):
                # Header ops
                header = section.get("header", {})
                if "ops" in header:
                    ops.extend(header["ops"])

                # Layer ops
                for layer in section.get("layers", []):
                    if "ops" in layer:
                        ops.extend(layer["ops"])

        return ops

    def extract_layout_buffers(self) -> Dict[str, Dict]:
        """Extract buffer info from layout."""
        buffers = {}

        # Weight entries
        weights = self.layout.get("memory", {}).get("weights", {})
        for entry in weights.get("entries", []):
            buffers[entry["name"]] = {
                "offset": entry.get("offset", 0),
                "size": entry.get("size", 0),
                "dtype": entry.get("dtype", "unknown"),
                "role": "weight"
            }

        # Activation buffers
        activations = self.layout.get("memory", {}).get("activations", {})
        for buf in activations.get("buffers", []):
            offset = buf.get("offset", 0)
            if isinstance(offset, str):
                offset = int(offset, 16) if offset.startswith("0x") else int(offset)
            buffers[buf["name"]] = {
                "offset": offset,
                "size": buf.get("size", 0),
                "dtype": buf.get("dtype", "fp32"),
                "role": "activation"
            }

        return buffers

    def test_ir_has_operations(self) -> TestResult:
        """Test 1: IR must have operations."""
        ops = self.extract_ir_ops()

        if not ops:
            self.issues.append(CodegenIssue(
                severity="CRITICAL",
                category="IR_STRUCTURE",
                message="Lowered IR has no operations! Codegen will generate empty decode function."
            ))
            return TestResult(
                name="IR has operations",
                passed=False,
                message=f"No ops found. IR keys: {list(self.lowered_ir.keys())}"
            )

        return TestResult(
            name="IR has operations",
            passed=True,
            message=f"Found {len(ops)} operations in IR"
        )

    def test_codegen_reads_ops_from_ir(self) -> TestResult:
        """Test 2: Codegen should read ops from IR, not hardcode them."""
        ir_ops = self.extract_ir_ops()
        ir_kernels = set()
        for op in ir_ops:
            kernel = op.get("kernel") or op.get("function") or op.get("op")
            if kernel:
                ir_kernels.add(kernel)

        # Find kernel calls in generated C
        c_kernels = set()
        kernel_pattern = r'^\s*(\w+)\s*\('
        for line in self.c_code_lines:
            # Skip comments and declarations
            if '//' in line or 'void' in line or 'extern' in line:
                continue
            match = re.match(kernel_pattern, line.strip())
            if match:
                func = match.group(1)
                # Filter to likely kernel calls
                if any(k in func.lower() for k in ['gemv', 'gemm', 'attention', 'rmsnorm',
                                                     'embedding', 'rope', 'swiglu', 'residual',
                                                     'mega_fused', 'quantize']):
                    c_kernels.add(func)

        # Check for kernels in C but not in IR
        extra_kernels = c_kernels - ir_kernels
        if extra_kernels:
            self.issues.append(CodegenIssue(
                severity="WARNING",
                category="CODEGEN_OVERRIDE",
                message=f"Codegen uses kernels not in IR: {extra_kernels}"
            ))

        # Check for TODO comments (kernels codegen couldn't handle)
        todo_count = self.c_code.count("/* TODO:")
        if todo_count > 0:
            self.issues.append(CodegenIssue(
                severity="ERROR",
                category="UNHANDLED_KERNEL",
                message=f"Codegen has {todo_count} unhandled kernels (/* TODO: ... */)"
            ))

        passed = len(extra_kernels) == 0 and todo_count == 0
        return TestResult(
            name="Codegen reads ops from IR",
            passed=passed,
            message=f"IR kernels: {len(ir_kernels)}, C kernels: {len(c_kernels)}, Extra: {len(extra_kernels)}, TODOs: {todo_count}",
            details=[f"Extra kernels in C: {extra_kernels}"] if extra_kernels else []
        )

    def test_offsets_match_layout(self) -> TestResult:
        """Test 3: Offsets in generated C must match layout JSON."""
        layout_buffers = self.extract_layout_buffers()

        # Extract hardcoded offsets from C code
        # Pattern: model->activations + OFFSET or ACT + OFFSET
        offset_pattern = r'(?:model->activations|ACT)\s*\+\s*(\d+)'
        c_offsets = set()
        offset_lines = {}

        for i, line in enumerate(self.c_code_lines):
            for match in re.finditer(offset_pattern, line):
                offset = int(match.group(1))
                c_offsets.add(offset)
                if offset not in offset_lines:
                    offset_lines[offset] = []
                offset_lines[offset].append(i + 1)

        # Get layout offsets
        layout_offsets = set()
        for name, info in layout_buffers.items():
            if info["role"] == "activation":
                layout_offsets.add(info["offset"])

        # Find offsets in C that don't match layout
        suspicious_offsets = []
        for offset in c_offsets:
            if offset not in layout_offsets and offset > 100:  # Skip small offsets (likely valid)
                suspicious_offsets.append((offset, offset_lines.get(offset, [])))
                self.issues.append(CodegenIssue(
                    severity="ERROR",
                    category="OFFSET_MISMATCH",
                    message=f"Offset {offset} in C code not found in layout",
                    line_number=offset_lines.get(offset, [-1])[0]
                ))

        passed = len(suspicious_offsets) == 0
        return TestResult(
            name="Offsets match layout",
            passed=passed,
            message=f"C offsets: {len(c_offsets)}, Layout offsets: {len(layout_offsets)}, Mismatches: {len(suspicious_offsets)}",
            details=[f"Suspicious: {o} at lines {l}" for o, l in suspicious_offsets[:5]]
        )

    def test_token_offset_consistency(self) -> TestResult:
        """Test 4: Token storage and read offsets must match."""
        # Find where token is stored
        store_pattern = r'\(\(int32_t\s*\*\)\s*(?:model->activations|ACT)(?:\s*\+\s*(\d+))?\s*\)\s*\[\s*\d*\s*\]\s*=\s*token'
        store_offset = None
        store_line = -1

        for i, line in enumerate(self.c_code_lines):
            match = re.search(store_pattern, line)
            if match:
                store_offset = int(match.group(1)) if match.group(1) else 0
                store_line = i + 1
                break

        # Find where embedding reads token
        read_pattern = r'embedding_forward.*\(\s*\(\s*(?:int32_t\s*\*|void\s*\*)\s*\)\s*\((?:model->activations|ACT)\s*\+\s*(\d+)\)'
        read_offset = None
        read_line = -1

        for i, line in enumerate(self.c_code_lines):
            if 'embedding_forward' in line:
                match = re.search(r'activations\s*\+\s*(\d+)', line)
                if match:
                    read_offset = int(match.group(1))
                    read_line = i + 1
                    break

        if store_offset is not None and read_offset is not None:
            if store_offset != read_offset:
                self.issues.append(CodegenIssue(
                    severity="CRITICAL",
                    category="TOKEN_OFFSET_MISMATCH",
                    message=f"Token stored at offset {store_offset} (line {store_line}) but embedding reads from {read_offset} (line {read_line})",
                    line_number=read_line
                ))
                return TestResult(
                    name="Token offset consistency",
                    passed=False,
                    message=f"MISMATCH: stored at {store_offset}, read at {read_offset}"
                )

        return TestResult(
            name="Token offset consistency",
            passed=True,
            message=f"Token stored and read at offset {store_offset or 'N/A'}"
        )

    def test_no_null_scratch(self) -> TestResult:
        """Test 5: Fused kernels must not receive NULL scratch."""
        # Find fused kernel calls with NULL scratch
        null_scratch_calls = []

        fused_kernels = ['mega_fused_attention', 'mega_fused_outproj']

        for i, line in enumerate(self.c_code_lines):
            for fused in fused_kernels:
                if fused in line:
                    # Look ahead for NULL in the call
                    call_text = line
                    j = i + 1
                    paren_count = line.count('(') - line.count(')')
                    while paren_count > 0 and j < len(self.c_code_lines):
                        call_text += self.c_code_lines[j]
                        paren_count += self.c_code_lines[j].count('(') - self.c_code_lines[j].count(')')
                        j += 1

                    if 'NULL' in call_text or 'null' in call_text.lower():
                        null_scratch_calls.append((fused, i + 1))
                        self.issues.append(CodegenIssue(
                            severity="CRITICAL",
                            category="NULL_SCRATCH",
                            message=f"{fused} receives NULL scratch - kernel will fail silently",
                            line_number=i + 1
                        ))
                    break

        passed = len(null_scratch_calls) == 0
        return TestResult(
            name="No NULL scratch to fused kernels",
            passed=passed,
            message=f"Found {len(null_scratch_calls)} fused kernel calls with NULL scratch",
            details=[f"{k} at line {l}" for k, l in null_scratch_calls[:5]]
        )

    def test_all_kernels_handled(self) -> TestResult:
        """Test 6: All kernel types in IR must be handled by codegen."""
        ir_ops = self.extract_ir_ops()

        # Get unique kernel/op types from IR
        kernel_types = set()
        for op in ir_ops:
            k = op.get("kernel") or op.get("op")
            if k:
                kernel_types.add(k)

        # Check which ones appear as TODO in generated code
        unhandled = []
        for line_num, line in enumerate(self.c_code_lines):
            if '/* TODO:' in line:
                match = re.search(r'/\* TODO:\s*(\w+)', line)
                if match:
                    unhandled.append((match.group(1), line_num + 1))

        # Also check for mega_fused handling
        has_mega_fused_handling = 'mega_fused_attention_prefill' in self.c_code
        ir_has_mega_fused = any('mega_fused' in str(op) for op in ir_ops)

        if ir_has_mega_fused and not has_mega_fused_handling:
            self.issues.append(CodegenIssue(
                severity="ERROR",
                category="MISSING_HANDLER",
                message="IR contains mega_fused ops but codegen doesn't handle them"
            ))

        passed = len(unhandled) == 0
        return TestResult(
            name="All kernels handled",
            passed=passed,
            message=f"IR has {len(kernel_types)} kernel types, {len(unhandled)} unhandled",
            details=[f"Unhandled: {k} at line {l}" for k, l in unhandled[:5]]
        )

    def test_activation_quantization(self) -> TestResult:
        """Test 7: Check if quantization is called before quantized GEMV."""
        # Look for quantize_row calls before gemv_q*_q8_0
        has_quantize = 'quantize_row_q8_0' in self.c_code
        has_quantized_gemv = bool(re.search(r'gemv_q[0-9]+_[0-9k]+_q8_0', self.c_code))

        if has_quantized_gemv and not has_quantize:
            self.issues.append(CodegenIssue(
                severity="ERROR",
                category="MISSING_QUANTIZATION",
                message="Code uses gemv_q*_q8_0 but never calls quantize_row_q8_0 - FP32 data passed to Q8_0 kernel"
            ))
            return TestResult(
                name="Activation quantization",
                passed=False,
                message="Missing quantize_row_q8_0 before quantized GEMV calls"
            )

        return TestResult(
            name="Activation quantization",
            passed=True,
            message="Quantization pattern OK" if has_quantize else "No quantized kernels used"
        )

    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("\n" + "="*70)
        print("CODEGEN VALIDATION: Does codegen respect IR?")
        print("="*70)

        if not self.load_files():
            return False

        tests = [
            self.test_ir_has_operations,
            self.test_codegen_reads_ops_from_ir,
            self.test_offsets_match_layout,
            self.test_token_offset_consistency,
            self.test_no_null_scratch,
            self.test_all_kernels_handled,
            self.test_activation_quantization,
        ]

        for test in tests:
            result = test()
            self.results.append(result)

            status = "PASS" if result.passed else "FAIL"
            symbol = "✓" if result.passed else "✗"
            print(f"\n{symbol} {result.name}: {status}")
            print(f"  {result.message}")
            for detail in result.details:
                print(f"    - {detail}")

        # Summary
        print("\n" + "="*70)
        print("ISSUES FOUND")
        print("="*70)

        if not self.issues:
            print("No issues found!")
        else:
            for severity in ["CRITICAL", "ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        line_info = f" (line {issue.line_number})" if issue.line_number > 0 else ""
                        print(f"  [{issue.category}]{line_info}: {issue.message}")

        # Final verdict
        print("\n" + "="*70)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        critical = sum(1 for i in self.issues if i.severity == "CRITICAL")

        if critical > 0:
            print(f"VERDICT: FAIL - {critical} CRITICAL issues found")
        elif passed == total:
            print(f"VERDICT: PASS - All {total} tests passed")
        else:
            print(f"VERDICT: PARTIAL - {passed}/{total} tests passed")

        print("="*70)

        return critical == 0 and passed == total


def main():
    parser = argparse.ArgumentParser(description="Validate codegen respects IR")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = CodegenValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

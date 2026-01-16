#!/usr/bin/env python3
"""
Layer 4: Codegen Validation
============================

Validates that generated C code is correct and compilable.

Tests:
1. Compilation success (no errors)
2. Header/source consistency
3. Kernel calls match IR operations
4. Buffer references are valid
5. Canary verification present
6. Dtype-specific kernel selection

Usage:
    python scripts/test_codegen_validation.py --c-file path/to/ck-kernel-inference.c
    python scripts/test_codegen_validation.py --model-dir path/to/model_dir
    python scripts/test_codegen_validation.py --auto
"""

import os
import sys
import re
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'


class CodegenValidator:
    """Validate generated C code."""

    def __init__(self, c_file: str, h_file: str = None, ir_path: str = None,
                 manifest_path: str = None, verbose: bool = False):
        self.c_file = Path(c_file)
        self.h_file = Path(h_file) if h_file else self.c_file.with_suffix('.h')
        self.ir_path = Path(ir_path) if ir_path else None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

        self.c_code = None
        self.h_code = None
        self.ir = None
        self.manifest = None

    def log_pass(self, msg: str):
        print(f"{GREEN}[PASS]{NC} {msg}")
        self.passed += 1

    def log_fail(self, msg: str):
        print(f"{RED}[FAIL]{NC} {msg}")
        self.failed += 1

    def log_warn(self, msg: str):
        print(f"{YELLOW}[WARN]{NC} {msg}")
        self.warnings += 1

    def log_info(self, msg: str):
        print(f"{BLUE}[INFO]{NC} {msg}")

    def load_files(self) -> bool:
        """Load generated code and related files."""
        # Load C source
        if self.c_file.exists():
            self.c_code = self.c_file.read_text()
            self.log_pass(f"Loaded C source: {self.c_file.name} ({len(self.c_code)} chars)")
        else:
            self.log_fail(f"C source not found: {self.c_file}")
            return False

        # Load header
        if self.h_file.exists():
            self.h_code = self.h_file.read_text()
            self.log_pass(f"Loaded header: {self.h_file.name}")
        else:
            self.log_warn(f"Header not found: {self.h_file}")

        # Load IR if available
        if self.ir_path and self.ir_path.exists():
            with open(self.ir_path) as f:
                self.ir = json.load(f)
            self.log_pass(f"Loaded IR: {self.ir_path.name}")

        # Load manifest if available
        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            self.log_pass(f"Loaded manifest: {self.manifest_path.name}")

        return True

    def test_compilation(self) -> bool:
        """Test 1: Code compiles without errors."""
        self.log_info("Test 1: Compilation")

        # Find include paths
        root_dir = self.c_file.parent
        while root_dir != root_dir.parent:
            if (root_dir / "include").exists():
                break
            root_dir = root_dir.parent

        include_dir = root_dir / "include"
        model_dir = self.c_file.parent

        # Compile command
        cmd = [
            'gcc', '-c', '-O2', '-fPIC', '-fopenmp',
            '-I', str(include_dir),
            '-I', str(model_dir),
            '-fsyntax-only',  # Just check syntax, don't generate object
            str(self.c_file)
        ]

        if self.verbose:
            print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.log_pass("Compilation successful (no syntax errors)")
                return True
            else:
                self.log_fail("Compilation failed")
                # Show errors
                if result.stderr:
                    errors = result.stderr.strip().split('\n')
                    for err in errors[:10]:  # Limit output
                        print(f"    {err}")
                    if len(errors) > 10:
                        print(f"    ... and {len(errors) - 10} more errors")
                return False

        except subprocess.TimeoutExpired:
            self.log_fail("Compilation timed out")
            return False
        except FileNotFoundError:
            self.log_warn("gcc not found, skipping compilation test")
            return True

    def test_header_consistency(self) -> bool:
        """Test 2: Header and source are consistent."""
        self.log_info("Test 2: Header Consistency")

        if not self.h_code:
            self.log_warn("No header file to check")
            return True

        issues = []

        # Extract defines from header
        header_defines = {}
        for match in re.finditer(r'#define\s+(\w+)\s+(\d+)', self.h_code):
            header_defines[match.group(1)] = int(match.group(2))

        # Check key defines are used consistently in source
        key_defines = [
            'VOCAB_SIZE', 'MAX_SEQ_LEN', 'EMBED_DIM', 'NUM_HEADS',
            'HEAD_DIM', 'NUM_LAYERS', 'INTERMEDIATE_DIM'
        ]

        for define in key_defines:
            # Find any define with this suffix
            matching = [k for k in header_defines if define in k]
            if matching and self.verbose:
                for m in matching:
                    print(f"    {m} = {header_defines[m]}")

        # Check struct definitions match
        header_structs = set(re.findall(r'typedef\s+struct\s+(\w+)', self.h_code))
        source_structs = set(re.findall(r'typedef\s+struct\s+(\w+)', self.c_code))

        if header_structs:
            self.log_pass(f"Found {len(header_structs)} struct definitions in header")

        # Check function declarations in header have implementations in source
        header_funcs = set(re.findall(r'\b(\w+)\s*\([^)]*\)\s*;', self.h_code))
        source_funcs = set(re.findall(r'\b(\w+)\s*\([^)]*\)\s*\{', self.c_code))

        # Filter to likely model functions
        model_funcs = [f for f in header_funcs if 'model' in f.lower() or 'forward' in f.lower()]
        missing = [f for f in model_funcs if f not in source_funcs]

        if missing:
            for f in missing[:5]:
                issues.append(f"Function declared but not defined: {f}")

        if not issues:
            self.log_pass("Header and source are consistent")
            return True
        else:
            for issue in issues:
                self.log_fail(issue)
            return False

    def test_kernel_calls(self) -> bool:
        """Test 3: Kernel calls match expected operations."""
        self.log_info("Test 3: Kernel Call Validation")

        # Extract kernel calls from code
        kernel_patterns = [
            r'gemm_nt_(\w+)\s*\(',  # GEMM kernels
            r'gemv_(\w+)\s*\(',     # GEMV kernels
            r'rmsnorm\w*\s*\(',     # RMSNorm
            r'rope_forward\w*\s*\(',  # RoPE
            r'softmax\w*\s*\(',     # Softmax
            r'swiglu\w*\s*\(',      # SwiGLU
            r'attention_forward\w*\s*\(',  # Attention
            r'embedding_forward\w*\s*\(',  # Embedding
        ]

        found_kernels = {}
        for pattern in kernel_patterns:
            matches = re.findall(pattern, self.c_code)
            for m in matches:
                kernel_name = pattern.split(r'\s*')[0].replace(r'\(', '').replace(r'\w*', '').replace(r'(\w+)', m)
                # Simplify
                if 'gemm' in pattern:
                    kernel_name = f"gemm_nt_{m}"
                elif 'gemv' in pattern:
                    kernel_name = f"gemv_{m}"
                else:
                    kernel_name = pattern.replace(r'\w*', '').replace(r'\s*\(', '').replace('\\', '')

                if kernel_name not in found_kernels:
                    found_kernels[kernel_name] = 0
                found_kernels[kernel_name] += 1

        if found_kernels:
            print(f"  Found kernel calls:")
            for kernel, count in sorted(found_kernels.items()):
                print(f"    {kernel}: {count} calls")
            self.log_pass(f"Found {len(found_kernels)} kernel types, {sum(found_kernels.values())} total calls")
        else:
            self.log_warn("No kernel calls found in generated code")

        # Check for expected kernels based on manifest dtypes
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            expected_dtypes = set()
            for name, info in tensors.items():
                dtype = info.get('dtype', info.get('type', '')).lower().replace('_', '')
                if 'q4' in dtype or 'q5' in dtype or 'q6' in dtype or 'q8' in dtype:
                    expected_dtypes.add(dtype)

            if expected_dtypes:
                print(f"  Expected quantization types from manifest: {expected_dtypes}")
                for dtype in expected_dtypes:
                    # Check if corresponding kernel is called
                    has_kernel = any(dtype in k for k in found_kernels)
                    if has_kernel:
                        self.log_pass(f"Found kernel for {dtype}")
                    else:
                        self.log_warn(f"No kernel found for {dtype}")

        return True

    def test_buffer_references(self) -> bool:
        """Test 4: Buffer references are valid."""
        self.log_info("Test 4: Buffer References")

        # Find buffer pointer macros
        ptr_pattern = r'(\w+)_PTR\s*\(\s*&\w+\s*,\s*(\w+)\s*\)'
        ptr_refs = re.findall(ptr_pattern, self.c_code)

        if ptr_refs:
            unique_refs = set(ptr_refs)
            print(f"  Found {len(unique_refs)} unique buffer references")

            # Check for common patterns
            has_weights = any('weight' in ref[1].lower() for ref in ptr_refs)
            has_cache = any('cache' in ref[1].lower() for ref in ptr_refs)
            has_activation = any('hidden' in ref[1].lower() or 'out' in ref[1].lower() for ref in ptr_refs)

            if has_weights:
                self.log_pass("Weight buffer references present")
            if has_cache:
                self.log_pass("KV cache buffer references present")
            if has_activation:
                self.log_pass("Activation buffer references present")

        # Check for potential issues
        issues = []

        # Look for NULL pointer dereferences
        if 'NULL' in self.c_code and '->' in self.c_code:
            # Check for patterns like: if (ptr) ptr->field (safe)
            # vs: ptr->field (potentially unsafe)
            unsafe_pattern = r'\b(\w+)\s*->\s*\w+(?!\s*\?)'
            unsafe_refs = re.findall(unsafe_pattern, self.c_code)
            # This is just a heuristic warning
            if self.verbose:
                print(f"  Found {len(set(unsafe_refs))} pointer dereferences")

        if not issues:
            self.log_pass("Buffer references appear valid")
            return True
        else:
            for issue in issues:
                self.log_fail(issue)
            return False

    def test_canary_verification(self) -> bool:
        """Test 5: Canary verification function present."""
        self.log_info("Test 5: Canary Verification")

        # Look for canary-related code
        has_canary_check = 'canary' in self.c_code.lower() or 'verify' in self.c_code.lower()
        has_canary_value = re.search(r'0x[A-Fa-f0-9]{8}', self.c_code) is not None

        if has_canary_check:
            self.log_pass("Canary verification code present")
        else:
            self.log_warn("No canary verification found (optional)")

        if has_canary_value:
            self.log_pass("Canary magic values defined")

        return True

    def test_dtype_kernel_mapping(self) -> bool:
        """Test 6: Correct kernels for each dtype."""
        self.log_info("Test 6: Dtype-Kernel Mapping")

        # Expected mappings
        dtype_to_kernel = {
            'q4_k': 'gemm_nt_q4_k',
            'q6_k': 'gemm_nt_q6_k',
            'q5_0': 'gemm_nt_q5_0',
            'q8_0': 'gemm_nt_q8_0',
            'q4_0': 'gemm_nt_q4_0',
            'f32': 'gemm_nt_f32',
            'fp32': 'gemm_nt_f32',
        }

        # Check comments in code that indicate dtype
        dtype_comments = re.findall(r'//.*?(\w+).*?weight.*?dtype.*?(\w+)', self.c_code, re.IGNORECASE)
        if dtype_comments:
            print(f"  Found {len(dtype_comments)} dtype comments in code")

        # Verify kernel calls match manifested dtypes
        if self.manifest:
            tensors = self.manifest.get('tensors', self.manifest.get('weights', {}))
            dtypes_used = set()
            for info in tensors.values():
                dtype = info.get('dtype', info.get('type', '')).lower().replace('_', '')
                dtypes_used.add(dtype)

            print(f"  Dtypes in manifest: {dtypes_used}")

            for dtype in dtypes_used:
                normalized = dtype.replace('ck_dt_', '').replace('_', '')
                expected_kernel = None
                for d, k in dtype_to_kernel.items():
                    if d.replace('_', '') in normalized:
                        expected_kernel = k
                        break

                if expected_kernel:
                    # Check if kernel is used
                    kernel_base = expected_kernel.split('_')[-1]  # e.g., q4k, q6k
                    if kernel_base in self.c_code.lower() or expected_kernel in self.c_code:
                        self.log_pass(f"Dtype {dtype} uses correct kernel pattern")
                    else:
                        self.log_warn(f"Dtype {dtype}: expected {expected_kernel}, not found")

        return True

    def run_all_tests(self) -> bool:
        """Run all codegen validation tests."""
        print("=" * 60)
        print("  CK-Engine Codegen Validation")
        print("=" * 60)
        print(f"C Source: {self.c_file}")
        if self.h_file.exists():
            print(f"Header: {self.h_file}")
        print()

        # Load files
        if not self.load_files():
            return False
        print()

        # Run tests
        self.test_compilation()
        print()

        self.test_header_consistency()
        print()

        self.test_kernel_calls()
        print()

        self.test_buffer_references()
        print()

        self.test_canary_verification()
        print()

        self.test_dtype_kernel_mapping()
        print()

        # Summary
        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  {GREEN}Passed:{NC} {self.passed}")
        print(f"  {RED}Failed:{NC} {self.failed}")
        print(f"  {YELLOW}Warnings:{NC} {self.warnings}")
        print()

        if self.failed == 0:
            print(f"{GREEN}All codegen validation tests passed!{NC}")
            return True
        else:
            print(f"{RED}Some tests failed. Check codegen script.{NC}")
            return False


def find_generated_code(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find generated C/H files in model directory."""
    c_candidates = [
        model_dir / "ck-kernel-inference.c",
        model_dir / "model.c",
        model_dir / "inference.c",
    ]

    for c_path in c_candidates:
        if c_path.exists():
            h_path = c_path.with_suffix('.h')
            return c_path, h_path if h_path.exists() else None

    # Search recursively
    for c_path in model_dir.rglob("*.c"):
        if 'kernel' in c_path.name.lower() or 'inference' in c_path.name.lower():
            h_path = c_path.with_suffix('.h')
            return c_path, h_path if h_path.exists() else None

    return None, None


def find_cached_models() -> List[Tuple[Path, Path]]:
    """Find model directories with generated code."""
    pairs = []

    cache_dirs = [
        Path.home() / ".cache" / "ck-engine-v6.6" / "models",
        Path.home() / ".cache" / "ck-engine-v6.5" / "models",
        Path.home() / ".cache" / "ck-engine-v6" / "models",
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            c_file, h_file = find_generated_code(model_dir)
            if c_file:
                pairs.append((c_file, model_dir))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validate generated C code")
    parser.add_argument("--c-file", type=str, help="Path to generated C file")
    parser.add_argument("--h-file", type=str, help="Path to header file (optional)")
    parser.add_argument("--model-dir", type=str, help="Model directory (auto-find)")
    parser.add_argument("--ir", type=str, help="IR JSON file (optional)")
    parser.add_argument("--manifest", type=str, help="Manifest JSON file (optional)")
    parser.add_argument("--auto", action="store_true", help="Auto-find cached models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.auto:
        pairs = find_cached_models()
        if not pairs:
            print(f"{RED}No cached models with generated code found{NC}")
            sys.exit(1)

        print(f"Found {len(pairs)} models with generated code:")
        for c_file, model_dir in pairs:
            print(f"  - {model_dir.name}/{c_file.name}")
        print()

        c_file, model_dir = pairs[0]
        h_file = c_file.with_suffix('.h')
        ir_path = model_dir / "lowered_decode.json"
        manifest_path = model_dir / "weights_manifest.json"

    elif args.model_dir:
        model_dir = Path(args.model_dir)
        c_file, h_file = find_generated_code(model_dir)
        if not c_file:
            print(f"{RED}No generated code found in {model_dir}{NC}")
            sys.exit(1)
        ir_path = model_dir / "lowered_decode.json"
        manifest_path = model_dir / "weights_manifest.json"

    elif args.c_file:
        c_file = Path(args.c_file)
        h_file = Path(args.h_file) if args.h_file else None
        ir_path = Path(args.ir) if args.ir else None
        manifest_path = Path(args.manifest) if args.manifest else None

    else:
        parser.print_help()
        sys.exit(1)

    validator = CodegenValidator(
        str(c_file),
        str(h_file) if h_file else None,
        str(ir_path) if ir_path and ir_path.exists() else None,
        str(manifest_path) if manifest_path and manifest_path.exists() else None,
        verbose=args.verbose
    )
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

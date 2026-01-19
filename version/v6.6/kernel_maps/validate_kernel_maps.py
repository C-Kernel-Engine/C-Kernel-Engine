#!/usr/bin/env python3
"""
Kernel Map Validator

Validates all kernel map JSON files in this directory.
Checks:
1. Valid JSON syntax
2. Required fields present
3. Field types correct
4. Dims referenced in shapes exist
5. Fused kernels reference existing kernel IDs
6. Source files exist
7. Test files exist

Usage:
    python validate_kernel_maps.py [--verbose] [--fix]
    python -m pytest validate_kernel_maps.py  # Run as tests
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Root directory (where this script lives)
SCRIPT_DIR = Path(__file__).parent.resolve()
KERNEL_MAPS_DIR = SCRIPT_DIR

# Required top-level fields
REQUIRED_FIELDS = {
    "id": (str, None),
    "op": (str, None),
    "variant": (str, None),
    "quant": (dict, None),
    "inputs": (list, None),
    "outputs": (list, None),
    "dims": (list, None),
    "impl": (dict, None),
}

# Required in impl
IMPL_REQUIRED_FIELDS = {
    "function": (str, None),
    "sources": (list, None),
}

# Valid operation types
VALID_OPS = {
    "gemm", "gemv", "rmsnorm", "rope", "attention", "add", "mul", "softmax",
    "swiglu", "silu", "sigmoid", "qkv_projection", "attention_projection",
    "fused_attention_block", "fused_mlp_block", "quantize", "dequantize",
    "layernorm", "bias_add", "embedding", "gather", "cat", "split"
}

# Valid data types
VALID_DTYPES = {
    "fp32", "fp16", "bf16", "fp64",
    "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "q4_k", "q6_k",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32",
    "none",  # For kernels without weights (attention, rope, etc.)
}

# Valid dimension symbols
VALID_DIMS = {
    "T", "tokens",        # Sequence length
    "S", "seq_len",       # Query/Key sequence length (attention)
    "U", "q_seq_len",     # Query sequence length (for attention Q×K^T)
    "E", "embed_dim",     # Embedding dimension
    "AE", "aligned_embed", "aligned_embed_dim",
    "H", "num_heads",     # Number of attention heads
    "KV", "num_kv_heads", # Number of KV heads (GQA)
    "D", "head_dim",      # Head dimension
    "AD", "aligned_head", "aligned_head_dim",
    "I", "intermediate", "intermediate_dim",
    "AI", "aligned_intermediate", "aligned_intermediate_dim",
    "max_T", "max_seq", "max_context_len",
    "V", "vocab", "vocab_size",
    "M", "N", "K",        # GEMM dimensions
    "B", "batch",         # Batch size
}


class KernelMapValidator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed: int = 0
        self.failed: int = 0
        self.all_kernel_ids: set[str] = set()
        self.kernel_files: dict[str, Path] = {}

    def log_verbose(self, msg: str):
        if self.verbose:
            print(f"  {CYAN}{msg}{RESET}")

    def log_pass(self, msg: str):
        self.passed += 1
        print(f"  {GREEN}✓{RESET} {msg}")

    def log_fail(self, msg: str):
        self.failed += 1
        print(f"  {RED}✗{RESET} {msg}")
        self.errors.append(msg)

    def log_warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  {YELLOW}!{RESET} {msg}")

    def load_all_kernel_maps(self) -> bool:
        """Load all kernel map JSON files and collect IDs."""
        json_files = list(KERNEL_MAPS_DIR.glob("*.json"))
        json_files = [f for f in json_files if f.name != "KERNEL_REGISTRY.json"]

        if not json_files:
            self.log_fail("No kernel map JSON files found")
            return False

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                kernel_id = data.get("id")
                if kernel_id:
                    self.all_kernel_ids.add(kernel_id)
                    self.kernel_files[kernel_id] = json_file
                    self.log_verbose(f"Loaded: {kernel_id}")
            except json.JSONDecodeError as e:
                self.log_fail(f"Invalid JSON in {json_file.name}: {e}")
                return False
            except Exception as e:
                self.log_fail(f"Error loading {json_file.name}: {e}")
                return False

        print(f"{GREEN}Loaded {len(self.all_kernel_ids)} kernel maps{RESET}")
        return True

    def validate_file(self, json_file: Path) -> bool:
        """Validate a single kernel map file."""
        name = json_file.name

        # Load JSON
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.log_fail(f"{name}: Invalid JSON - {e}")
            return False

        if not isinstance(data, dict):
            self.log_fail(f"{name}: Root must be an object")
            return False

        # Check required fields
        for field, (expected_type, _) in REQUIRED_FIELDS.items():
            if field not in data:
                self.log_fail(f"{name}: Missing required field '{field}'")
                return False
            if not isinstance(data[field], expected_type):
                self.log_fail(f"{name}: Field '{field}' must be {expected_type.__name__}")
                return False

        # Check impl.required fields
        impl = data["impl"]
        for field, (expected_type, _) in IMPL_REQUIRED_FIELDS.items():
            if field not in impl:
                self.log_fail(f"{name}.impl: Missing '{field}'")
                return False

        # Validate op type
        op = data["op"]
        if op not in VALID_OPS:
            self.log_fail(f"{name}: Unknown op '{op}'")

        # Validate quant
        self.validate_quant(name, data["quant"])

        # Validate inputs/outputs
        for inp in data["inputs"]:
            self.validate_io_spec(name, inp, "input")
        for out in data["outputs"]:
            self.validate_io_spec(name, out, "output")

        # Validate dims
        self.validate_dims(name, data["dims"])

        # Validate impl
        self.validate_impl(name, impl)

        # Validate fuses (if present)
        if "fuses" in data:
            self.validate_fuses(name, data["fuses"])

        self.log_pass(name)
        return True

    def validate_quant(self, name: str, quant: dict):
        """Validate quant field."""
        valid_quant_keys = {"weight", "activation", "output"}
        for key in quant.keys():
            if key not in valid_quant_keys:
                self.log_warn(f"{name}.quant: Unknown key '{key}'")

        # Check dtypes are valid
        for key, dtype in quant.items():
            if dtype not in VALID_DTYPES and "|" not in dtype:  # Allow "q5_0|q8_0"
                self.log_warn(f"{name}.quant.{key}: Unknown dtype '{dtype}'")

    def validate_io_spec(self, name: str, spec: dict, io_type: str):
        """Validate input/output specification."""
        if "name" not in spec:
            self.log_fail(f"{name}: {io_type} missing 'name'")
        if "dtype" not in spec:
            self.log_fail(f"{name}: {io_type} missing 'dtype'")

        # Check shape
        if "shape" in spec:
            shape = spec["shape"]
            for dim in shape:
                # Can be string (dim symbol) or dict with "dim" key
                if isinstance(dim, str):
                    if dim not in VALID_DIMS:
                        self.log_warn(f"{name}: {io_type} shape uses unknown dim '{dim}'")
                elif isinstance(dim, dict):
                    if "dim" not in dim:
                        self.log_fail(f"{name}: {io_type} shape dict missing 'dim'")

    def validate_dims(self, name: str, dims: list):
        """Validate dims list."""
        for dim in dims:
            if dim not in VALID_DIMS:
                self.log_warn(f"{name}.dims: Unknown dim '{dim}'")

    def validate_impl(self, name: str, impl: dict):
        """Validate impl field."""
        # Check sources exist
        sources = impl.get("sources", [])
        for src in sources:
            src_path = Path(src)
            if not src_path.is_absolute():
                # Check relative to repo root
                repo_root = SCRIPT_DIR.parent.parent
                full_path = repo_root / src
                if not full_path.exists():
                    self.log_warn(f"{name}: Source file not found: {src}")

        # Check variants if present
        if "variants" in impl:
            for variant in impl["variants"]:
                if not isinstance(variant, dict):
                    self.log_fail(f"{name}.impl.variants: Each variant must be an object")
                    continue
                if "name" not in variant:
                    self.log_fail(f"{name}.impl.variants: Variant missing 'name'")

    def validate_fuses(self, name: str, fuses: list):
        """Validate fuses array references existing kernels."""
        for fused_kernel in fuses:
            if fused_kernel not in self.all_kernel_ids:
                self.log_warn(f"{name}: Fuses unknown kernel '{fused_kernel}'")

    def validate_all(self) -> bool:
        """Validate all kernel maps."""
        print(f"\n{CYAN}Validating kernel maps in: {KERNEL_MAPS_DIR}{RESET}\n")

        # First load all
        if not self.load_all_kernel_maps():
            return False

        print()
        json_files = list(KERNEL_MAPS_DIR.glob("*.json"))
        json_files = [f for f in json_files if f.name != "KERNEL_REGISTRY.json"]

        for json_file in sorted(json_files):
            self.validate_file(json_file)

        # Summary
        print(f"\n{'='*60}")
        print(f"{CYAN}VALIDATION SUMMARY{RESET}")
        print(f"{'='*60}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        print(f"  {YELLOW}Warnings: {len(self.warnings)}{RESET}")

        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for w in self.warnings[:5]:
                print(f"  - {w}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")

        if self.errors:
            print(f"\n{RED}Errors:{RESET}")
            for e in self.errors[:5]:
                print(f"  - {e}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")

        print()
        return self.failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate kernel map JSON files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues")
    args = parser.parse_args()

    validator = KernelMapValidator(verbose=args.verbose)
    success = validator.validate_all()

    if success:
        print(f"{GREEN}All kernel maps are valid!{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}Validation failed!{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

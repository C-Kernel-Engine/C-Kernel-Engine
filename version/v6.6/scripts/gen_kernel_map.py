#!/usr/bin/env python3
"""
Generate kernel map template from source and tests.

This script helps create kernel_maps/*.json by:
1. Finding the kernel source file
2. Parsing function signature
3. Finding unit test
4. Running tests to discover SIMD variants
5. Generating JSON template with TODOs for manual completion

Usage:
  python3 version/v6.6/scripts/gen_kernel_map.py --kernel gemv_q5_0_q8_0
  python3 version/v6.6/scripts/gen_kernel_map.py --kernel rmsnorm_forward --output kernel_maps/
  python3 version/v6.6/scripts/gen_kernel_map.py --kernel mega_fused_attention_prefill --skip-tests
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Kernel source roots
KERNEL_ROOTS = [
    "src/kernels",
    "src/kernels/fused",
    "src",  # For kernels in src/*.c
]

# Unit test root
TEST_ROOT = "unittest"

# SIMD variants to check
SIMD_VARIANTS = [
    {"name": "ref", "requires": [], "compile_flags": []},
    {"name": "sse", "requires": ["sse4.1"], "compile_flags": ["-msse4.1"]},
    {"name": "avx", "requires": ["avx"], "compile_flags": ["-mavx"]},
    {"name": "avx2", "requires": ["avx2", "fma"], "compile_flags": ["-mavx2", "-mfma"]},
    {"name": "avx512", "requires": ["avx512f", "fma"], "compile_flags": ["-mavx512f", "-mfma"]},
    {"name": "vnni", "requires": ["avx512vnni"], "compile_flags": ["-mavx512vnni"]},
    {"name": "amx", "requires": ["amx-int8"], "compile_flags": ["-mamx-int8"]},
]

# Pattern to match function definitions
FUNC_PATTERN = re.compile(
    r'^(?:static\s+)?(?:inline\s+)?'
    r'(void|int|float|size_t|int32_t|uint32_t)\s+'
    r'(\w+)\s*\(([^)]*)\)',
    re.MULTILINE
)

# Quant type patterns
QUANT_PATTERNS = [
    (r'q4_0', 'q4_0'),
    (r'q4_1', 'q4_1'),
    (r'q5_0', 'q5_0'),
    (r'q5_1', 'q5_1'),
    (r'q8_0', 'q8_0'),
    (r'q4_k|q4k', 'q4_k'),
    (r'q6_k|q6k', 'q6_k'),
    (r'q8_k|q8k', 'q8_k'),
    (r'bf16', 'bf16'),
    (r'f16|fp16', 'f16'),
    (r'int8', 'int8'),
    (r'int4', 'int4'),
]

# Operation type patterns
OP_PATTERNS = [
    (r'^gemv_', 'gemv'),
    (r'^gemm_', 'gemm'),
    (r'^vec_dot_', 'vec_dot'),
    (r'^dot_', 'dot'),
    (r'rmsnorm', 'rmsnorm'),
    (r'layernorm', 'layernorm'),
    (r'attention', 'attention'),
    (r'softmax', 'softmax'),
    (r'rope', 'rope'),
    (r'swiglu', 'swiglu'),
    (r'gelu', 'gelu'),
    (r'relu', 'relu'),
    (r'sigmoid', 'sigmoid'),
    (r'embedding', 'embedding'),
    (r'mlp', 'mlp'),
    (r'topk', 'topk'),
    (r'quantize', 'quantize'),
    (r'dequant', 'dequant'),
    (r'mega_fused', 'fused_block'),
    (r'fused', 'fused'),
]


def find_kernel_source(kernel_name: str) -> Optional[Tuple[str, int, str]]:
    """Find kernel source file and line number.

    Returns: (filepath, line_number, signature) or None
    """
    for root in KERNEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except Exception:
                continue

            for match in FUNC_PATTERN.finditer(content):
                func_name = match.group(2)
                if func_name == kernel_name:
                    # Count line number
                    line_num = content[:match.start()].count('\n') + 1
                    signature = f"{match.group(1)} {match.group(2)}({match.group(3)})"
                    return filepath, line_num, signature.strip()

    return None


def find_unit_test(kernel_name: str) -> Optional[str]:
    """Find unit test file for kernel."""
    if not os.path.isdir(TEST_ROOT):
        return None

    # Try common patterns
    patterns = [
        f"test_{kernel_name}.py",
        f"test_{kernel_name.replace('_', '')}.py",
    ]

    # Also try partial matches
    for fname in os.listdir(TEST_ROOT):
        if not fname.endswith('.py'):
            continue
        # Check if kernel name (or part of it) is in the test name
        base = kernel_name.lower()
        if base in fname.lower():
            return os.path.join(TEST_ROOT, fname)

    # Try exact patterns
    for pattern in patterns:
        path = os.path.join(TEST_ROOT, pattern)
        if os.path.exists(path):
            return path

    return None


def infer_op_type(kernel_name: str) -> str:
    """Infer operation type from kernel name."""
    for pattern, op in OP_PATTERNS:
        if re.search(pattern, kernel_name, re.IGNORECASE):
            return op
    return "unknown"


def infer_quant_types(kernel_name: str) -> Dict[str, str]:
    """Infer quantization types from kernel name."""
    found = []
    for pattern, qtype in QUANT_PATTERNS:
        if re.search(pattern, kernel_name, re.IGNORECASE):
            found.append(qtype)

    if not found:
        return {"weight": "fp32", "activation": "fp32", "output": "fp32"}

    if len(found) == 1:
        # Single quant type - probably weight
        return {"weight": found[0], "activation": "fp32", "output": "fp32"}
    elif len(found) >= 2:
        # Two quant types - weight and activation
        return {"weight": found[0], "activation": found[1], "output": "fp32"}

    return {"weight": "fp32", "activation": "fp32", "output": "fp32"}


def parse_signature(signature: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse C function signature to extract params.

    Returns: (return_type, list of param dicts)
    """
    # Extract return type and params
    match = re.match(r'(\w+)\s+\w+\s*\(([^)]*)\)', signature)
    if not match:
        return "void", []

    ret_type = match.group(1)
    params_str = match.group(2).strip()

    if not params_str:
        return ret_type, []

    params = []
    for param in params_str.split(','):
        param = param.strip()
        # Parse "const float *x" or "int M"
        parts = param.split()
        if len(parts) >= 2:
            name = parts[-1].lstrip('*')
            dtype_parts = parts[:-1]
            is_pointer = '*' in param
            is_const = 'const' in dtype_parts

            # Infer dtype
            dtype = "fp32"
            if 'float' in dtype_parts:
                dtype = "fp32"
            elif 'void' in dtype_parts:
                dtype = "void"  # Usually quantized
            elif 'int' in dtype_parts or 'int32_t' in dtype_parts:
                dtype = "int32"
            elif 'uint8_t' in dtype_parts:
                dtype = "uint8"
            elif 'int8_t' in dtype_parts:
                dtype = "int8"

            params.append({
                "name": name,
                "dtype": dtype,
                "is_pointer": is_pointer,
                "is_const": is_const,
            })

    return ret_type, params


def infer_inputs_outputs(params: List[Dict], quant: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
    """Infer inputs and outputs from params and quant info."""
    inputs = []
    outputs = []

    for p in params:
        if p["dtype"] in ("int32", "int64"):
            # Dimension parameter, not a buffer
            continue

        if p["is_pointer"]:
            if p["is_const"]:
                # const pointer = input
                dtype = p["dtype"]
                if dtype == "void":
                    # Infer from position/name
                    if 'W' in p["name"] or 'weight' in p["name"].lower():
                        dtype = quant["weight"]
                    elif 'x' in p["name"] or 'input' in p["name"].lower():
                        dtype = quant["activation"]
                    else:
                        dtype = quant["activation"]

                inputs.append({
                    "name": p["name"],
                    "dtype": dtype,
                    "shape": ["TODO"],
                    "desc": "TODO: describe"
                })
            else:
                # non-const pointer = output
                outputs.append({
                    "name": p["name"],
                    "dtype": quant["output"],
                    "shape": ["TODO"],
                    "desc": "TODO: describe"
                })

    return inputs, outputs


def discover_simd_variants(kernel_name: str, test_file: Optional[str]) -> List[Dict]:
    """Discover which SIMD variants exist and pass tests."""
    # For now, just check if variant-named functions exist in source
    # A full implementation would run tests

    variants = []

    # Check for variant-specific functions in source
    for root in KERNEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for fname in os.listdir(root):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except Exception:
                continue

            for variant in SIMD_VARIANTS:
                vname = variant["name"]
                # Check for kernel_name_variant pattern
                patterns = [
                    f"{kernel_name}_{vname}",
                    f"{kernel_name.replace('_', '')}_{vname}",
                ]
                for pattern in patterns:
                    if re.search(rf'\b{pattern}\b', content):
                        variants.append(variant.copy())
                        break

    # Always include ref if we found the main function
    if not any(v["name"] == "ref" for v in variants):
        # Check if main function exists without suffix (acts as ref)
        variants.insert(0, SIMD_VARIANTS[0].copy())  # ref

    return variants


def generate_kernel_map(
    kernel_name: str,
    source_file: str,
    line_num: int,
    signature: str,
    test_file: Optional[str],
    variants: List[Dict],
) -> Dict:
    """Generate kernel map JSON."""

    op = infer_op_type(kernel_name)
    quant = infer_quant_types(kernel_name)
    ret_type, params = parse_signature(signature)
    inputs, outputs = infer_inputs_outputs(params, quant)

    # Build variant string
    variant_str = f"{quant['weight']}"
    if quant['activation'] != 'fp32':
        variant_str += f"_w_{quant['activation']}_a"
    variant_str += f"_{quant['output']}_out"

    # Infer dims from common patterns
    dims = []
    for p in params:
        if p["dtype"] in ("int32", "int64") and not p["is_pointer"]:
            dims.append(p["name"])
    if not dims:
        dims = ["M", "K"]  # Default

    kernel_map = {
        "id": kernel_name,
        "op": op,
        "variant": variant_str,
        "quant": quant,
        "inputs": inputs,
        "outputs": outputs,
        "scratch": [],
        "dims": dims,
        "parallelization": {
            "supported": ["TODO: e.g., output, token, feature"],
            "preferred": {"prefill": "TODO", "decode": "TODO"},
            "strategies": [
                {
                    "name": "TODO",
                    "partition_dim": "TODO",
                    "param_style": "TODO: ith_nth or range",
                    "notes": "TODO"
                }
            ]
        },
        "constraints": {
            "alignment": {"TODO_DIM": "TODO: e.g., 256"},
            "notes": "TODO"
        },
        "impl": {
            "function": kernel_name,
            "sources": [source_file],
            "variants": variants if variants else [{"name": "ref", "requires": []}]
        },
        "tests": {
            "unit": [test_file] if test_file else ["TODO: unittest/test_*.py"],
            "parity": [
                {
                    "kind": "numpy",
                    "command": f"python {test_file} --compare-numpy" if test_file else "TODO",
                    "tolerance": "1e-5",
                    "notes": "TODO"
                }
            ]
        },
        "_generated": {
            "by": "gen_kernel_map.py",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source_line": f"{source_file}:{line_num}",
            "needs_review": [
                "inputs[*].shape",
                "outputs[*].shape",
                "parallelization",
                "constraints",
                "scratch"
            ]
        },
        "notes": "TODO: describe what this kernel does"
    }

    return kernel_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate kernel map from source")
    parser.add_argument("--kernel", "-k", required=True,
                        help="Kernel function name (e.g., gemv_q5_0_q8_0)")
    parser.add_argument("--output", "-o", default="version/v6.6/kernel_maps",
                        help="Output directory for kernel map")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip test discovery")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing kernel map")
    args = parser.parse_args()

    kernel_name = args.kernel

    # Step 1: Find kernel source
    print(f"[search] looking for {kernel_name} in src/kernels/...")
    result = find_kernel_source(kernel_name)

    if not result:
        print(f"[error] kernel '{kernel_name}' not found in source")
        return 1

    source_file, line_num, signature = result
    print(f"[found]  {source_file}:{line_num}")
    print(f"[parse]  signature: {signature}")

    # Infer info
    op = infer_op_type(kernel_name)
    quant = infer_quant_types(kernel_name)
    print(f"[infer]  op: {op}")
    print(f"[infer]  quant: weight={quant['weight']}, activation={quant['activation']}, output={quant['output']}")

    # Step 2: Find unit test
    print(f"\n[search] looking for unit test...")
    test_file = find_unit_test(kernel_name)
    if test_file:
        print(f"[found]  {test_file}")
    else:
        print(f"[warn]   no unit test found")

    # Step 3: Discover SIMD variants
    variants = []
    if not args.skip_tests:
        print(f"\n[scan]   discovering SIMD variants...")
        variants = discover_simd_variants(kernel_name, test_file)
        for v in variants:
            print(f"         - {v['name']}")

    # Step 4: Generate kernel map
    print(f"\n[generate] creating kernel map...")
    kernel_map = generate_kernel_map(
        kernel_name, source_file, line_num, signature, test_file, variants
    )

    # Step 5: Write output
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{kernel_name}.json")

    if os.path.exists(output_path) and not args.force:
        print(f"[error]  {output_path} already exists (use --force to overwrite)")
        return 1

    with open(output_path, 'w') as f:
        json.dump(kernel_map, f, indent=2)

    print(f"[ok]     wrote {output_path}")

    # Show TODOs
    todos = kernel_map.get("_generated", {}).get("needs_review", [])
    if todos:
        print(f"\n[TODO]   Please fill in manually:")
        for todo in todos:
            print(f"         - {todo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

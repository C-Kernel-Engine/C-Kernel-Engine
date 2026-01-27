#!/usr/bin/env python3
"""
Generate KERNEL_REGISTRY.json by scanning src/kernels/**/*.c

Usage:
    python scripts/gen_kernel_registry.py
    python scripts/gen_kernel_registry.py --output version/v6.6/kernel_maps/KERNEL_REGISTRY.json

The script:
1. Scans all .c files in src/kernels/
2. Extracts CK_DECLARE_KERNEL declarations
3. Categorizes by operation type
4. Generates KERNEL_REGISTRY.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

# Regex to find kernel declarations
KERNEL_DECLARE_RE = re.compile(
    r'CK_DECLARE_KERNEL\s*\(\s*(\w+)\s*\)\s*=\s*\{([^}]+)\}',
    re.MULTILINE
)

# Operation categorization
OP_CATEGORIES = {
    "gemm": ["gemm_", "gemv_", "matmul_", "dot_"],
    "rmsnorm": ["rmsnorm_", "layernorm_"],
    "rope": ["rope_"],
    "attention": ["attention_", "flash_attn", "causal_attn"],
    "swiglu": ["swiglu_", "silu_", "sigmoid_"],
    "add": ["add_", "residual_", "bias_add_"],
    "quantize": ["quantize_", "dequantize_", "vec_dot_"],
    "embedding": ["embedding_", "gather_"],
    " softmax": ["softmax_"],
}

# File to category mapping
FILE_CATEGORIES = {
    "gemm_kernels": "gemm",
    "gemm_kernels_q": "gemm",
    "rmsnorm_kernels": "rmsnorm",
    "rope_kernels": "rope",
    "attention_kernels": "attention",
    "swiglu_kernels": "swiglu",
    "quantize_kernels": "quantize",
}


def extract_kernels_from_file(filepath: Path) -> List[Dict[str, Any]]:
    """Extract kernel declarations from a .c file."""
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"[warn] Could not read {filepath}: {e}")
        return []

    kernels = []
    for match in KERNEL_DECLARE_RE.finditer(content):
        kernel_name = match.group(1)
        body = match.group(2)

        kernel = {
            "name": kernel_name,
            "source": filepath.name,
            "dtypes": [],
            "layout": None,
            "category": infer_category(kernel_name, filepath.name),
        }

        # Extract dtypes from body
        dtype_match = re.search(r'\.out_dtype\s*=\s*(\w+)', body)
        if dtype_match:
            kernel["dtypes"].append(dtype_match.group(1))

        # Extract layout
        layout_match = re.search(r'\.trans_a\s*=\s*(\w+)', body)
        if layout_match:
            kernel["layout"] = layout_match.group(1)

        kernels.append(kernel)

    return kernels


def infer_category(kernel_name: str, filename: str) -> str:
    """Infer the operation category from kernel name or filename."""
    # Check filename first
    for prefix, category in FILE_CATEGORIES.items():
        if prefix in filename:
            return category

    # Check kernel name
    for category, prefixes in OP_CATEGORIES.items():
        for prefix in prefixes:
            if prefix in kernel_name.lower():
                return category

    return "utility"


def scan_kernels_dir(kernels_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Scan src/kernels directory and return categorized kernels."""
    all_kernels = []

    for cfile in kernels_dir.rglob("*.c"):
        kernels = extract_kernels_from_file(cfile)
        all_kernels.extend(kernels)

    # Group by category
    categorized = {}
    for kernel in all_kernels:
        cat = kernel.pop("category")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(kernel)

    return categorized


def generate_registry(kernels_dir: Path, output_path: Path) -> Dict[str, Any]:
    """Generate the complete kernel registry."""
    categorized = scan_kernels_dir(kernels_dir)

    # Build registry structure
    registry = {
        "_meta": {
            "description": "Comprehensive registry of all kernels in src/kernels/",
            "version": "v6.6",
            "generated_from": str(kernels_dir),
            "total_kernels": sum(len(k) for k in categorized.values()),
            "categories": list(categorized.keys()),
        }
    }

    # Add each category
    for category, kernels in sorted(categorized.items()):
        cat_key = f"{category}_kernels" if not category.endswith("y") else f"{category[:-1]}ies_kernels"
        if category in ("add", "quantize", "embedding"):
            cat_key = f"{category}_kernels"

        registry[cat_key] = {
            "description": f"{category.capitalize()} operations",
            "count": len(kernels),
            "kernels": sorted(kernels, key=lambda k: k["name"])
        }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)

    return registry


def main():
    parser = argparse.ArgumentParser(description="Generate kernel registry from source files")
    parser.add_argument(
        "--output", "-o",
        default="version/v6.6/kernel_maps/KERNEL_REGISTRY.json",
        help="Output path for registry JSON"
    )
    parser.add_argument(
        "--kernels-dir", "-k",
        default="src/kernels",
        help="Directory containing kernel source files"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    kernels_dir = Path(args.kernels_dir)
    if not kernels_dir.exists():
        print(f"[error] Kernels directory not found: {kernels_dir}")
        return 1

    if args.verbose:
        print(f"[info] Scanning {kernels_dir}")

    registry = generate_registry(kernels_dir, Path(args.output))

    print(f"[ok] Generated {registry['_meta']['total_kernels']} kernels")
    print(f"[ok] Written to {args.output}")

    # Summary by category
    print("\n[summary]")
    for key in sorted(registry.keys()):
        if key != "_meta":
            cat_info = registry[key]
            print(f"  {cat_info['count']:3d} {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Kernel source scanner for CK-Engine v6.6.

This module scans src/kernels/**/*.c and returns a categorized registry of
public kernel entrypoints. It is used by tooling that cross-checks kernel maps
against actual C implementations.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Root of kernel sources
KERNEL_ROOT = "src/kernels"

# Pattern to match function definitions
FUNC_PATTERN = re.compile(
    r'^(?:static\s+)?(?:inline\s+)?'
    r'(void|int|float|size_t|int32_t|uint32_t)\s+'
    r'(\w+)\s*\(',
    re.MULTILINE
)

# Classification rules (order matters - first match wins)
CLASSIFICATION_RULES = [
    # Optimizer kernels
    (r'adamw_|sgd_|zero_gradients|gradient_', 'optimizer'),

    # Training/backward kernels
    (r'_backward|_backward_', 'training'),

    # Fusion kernels (check before inference)
    (r'mega_fused_|fused_|_fused', 'fusion'),

    # Quantization kernels
    (r'quantize_|dequant_|vec_dot_|dot_q|convert_f|ck_fp\d+_to_|_to_fp\d+', 'quantization'),

    # Utility kernels
    (r'axpy_|scal_|weighted_sum|moe_accumulate|add_forward|add_inplace|add_scaled|_init$|_cleanup$|hsum_', 'utility'),

    # Everything else is inference
    (r'.*', 'inference'),
]

# Sub-categorization for inference kernels
INFERENCE_SUBCATEGORIES = {
    'gemm': r'gemm_|gemv_',
    'attention': r'attention_|softmax_|causal_softmax',
    'normalization': r'rmsnorm_|layernorm_',
    'activation': r'swiglu_|gelu_|relu_|sigmoid_',
    'positional': r'rope_',
    'embedding': r'embedding_',
    'mlp': r'mlp_',
    'kv_cache': r'kv_cache_',
    'sampling': r'topk_|argmax_|cross_entropy',
    'vision': r'im2patch|patch2im',
}

# Skip these (internal/helper functions)
SKIP_PATTERNS = [
    r'^_',           # Private functions
    r'^main$',       # Test mains
    r'_ref$',        # Reference implementations (keep dispatch version)
    r'_scalar$',     # Scalar fallbacks
    r'_avx\d*$',     # SIMD variants (keep dispatch version)
    r'_sse\d*$',
    r'_vnni$',
    r'_amx$',
]


def extract_functions(filepath: str) -> List[Tuple[str, str]]:
    """Extract function names and return types from a C file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return []

    functions = []
    for match in FUNC_PATTERN.finditer(content):
        ret_type = match.group(1)
        func_name = match.group(2)
        functions.append((func_name, ret_type))

    return functions


def should_skip(func_name: str) -> bool:
    """Check if function should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, func_name):
            return True
    return False


def classify_function(func_name: str, source_file: str) -> str:
    """Classify a function into a category."""
    if '/fused/' in source_file or '\\fused\\' in source_file:
        return 'fusion'

    for pattern, category in CLASSIFICATION_RULES:
        if re.search(pattern, func_name, re.IGNORECASE):
            return category

    return 'inference'


def get_inference_subcategory(func_name: str) -> str:
    """Get subcategory for inference kernels."""
    for subcat, pattern in INFERENCE_SUBCATEGORIES.items():
        if re.search(pattern, func_name, re.IGNORECASE):
            return subcat
    return 'other'


def extract_dtypes(func_name: str) -> List[str]:
    """Extract data types from function name."""
    dtypes = []

    quant_patterns = [
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

    for pattern, dtype in quant_patterns:
        if re.search(pattern, func_name, re.IGNORECASE):
            dtypes.append(dtype)

    if not dtypes:
        dtypes.append('fp32')

    return dtypes


def scan_kernel_sources(root: str = KERNEL_ROOT,
                        generated_by: str = "kernel_source_scan.py") -> Dict:
    """Scan kernel sources and return a categorized registry."""
    registry = {
        '_meta': {
            'description': 'Auto-generated kernel source scan',
            'version': 'v6.6',
            'generated_by': generated_by,
        },
        'inference': defaultdict(list),
        'training': [],
        'optimizer': [],
        'fusion': [],
        'quantization': [],
        'utility': [],
    }

    seen_functions: Set[str] = set()

    for root_dir, _, files in os.walk(root):
        for fname in files:
            if not fname.endswith('.c'):
                continue

            filepath = os.path.join(root_dir, fname)
            rel_path = os.path.relpath(filepath, start='.')

            functions = extract_functions(filepath)
            for func_name, _ret_type in functions:
                if func_name in seen_functions:
                    continue
                if should_skip(func_name):
                    continue

                seen_functions.add(func_name)

                category = classify_function(func_name, filepath)
                dtypes = extract_dtypes(func_name)

                entry = {
                    'name': func_name,
                    'source': fname,
                    'path': rel_path,
                    'dtypes': dtypes,
                }

                if category == 'inference':
                    subcat = get_inference_subcategory(func_name)
                    registry['inference'][subcat].append(entry)
                else:
                    registry[category].append(entry)

    registry['inference'] = dict(registry['inference'])

    inference_count = sum(len(v) for v in registry['inference'].values())
    registry['_meta']['counts'] = {
        'inference': inference_count,
        'training': len(registry['training']),
        'optimizer': len(registry['optimizer']),
        'fusion': len(registry['fusion']),
        'quantization': len(registry['quantization']),
        'utility': len(registry['utility']),
        'total': (inference_count + len(registry['training']) +
                  len(registry['optimizer']) + len(registry['fusion']) +
                  len(registry['quantization']) + len(registry['utility'])),
    }

    return registry


def scan_function_names(root: str = KERNEL_ROOT) -> Set[str]:
    """Return a set of public kernel function names found in sources."""
    registry = scan_kernel_sources(root=root)
    names: Set[str] = set()

    for subcat in registry.get('inference', {}).values():
        for entry in subcat:
            names.add(entry['name'])
    for cat in ('training', 'optimizer', 'fusion', 'quantization', 'utility'):
        for entry in registry.get(cat, []):
            names.add(entry['name'])

    return names

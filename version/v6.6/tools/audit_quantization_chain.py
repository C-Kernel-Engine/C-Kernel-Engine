#!/usr/bin/env python3
"""
audit_quantization_chain.py - Verify quantization consistency through the pipeline

This tool traces the dtype of each weight tensor through:
1. Weights manifest (what the GGUF/BUMP file actually contains)
2. IR Lower (what dtype the IR thinks it is)
3. Generated C code (what kernel is actually called)

Any mismatch = potential source of garbled output.

Usage:
    python audit_quantization_chain.py <model_cache_dir>
    python audit_quantization_chain.py <model_cache_dir> --layer 0  # specific layer
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# Quantization type mappings
DTYPE_TO_KERNEL_SUFFIX = {
    'q4_0': 'q4_0',
    'q4_1': 'q4_1',
    'q4_k': 'q4_k',
    'q5_0': 'q5_0',
    'q5_1': 'q5_1',
    'q5_k': 'q5_k',
    'q6_k': 'q6_k',
    'q8_0': 'q8_0',
    'q8_k': 'q8_k',
    'fp32': 'f32',
    'f32': 'f32',
    'fp16': 'f16',
    'f16': 'f16',
}

KERNEL_SUFFIX_TO_DTYPE = {v: k for k, v in DTYPE_TO_KERNEL_SUFFIX.items()}


@dataclass
class WeightInfo:
    """Information about a single weight tensor."""
    name: str
    layer: int
    role: str  # wq, wk, wv, wo, w1, w2, ln1_gamma, etc.
    manifest_dtype: str = ""
    ir_dtype: str = ""
    codegen_kernel: str = ""
    manifest_offset: int = 0
    ir_offset: int = 0
    codegen_offset: int = 0
    size: int = 0

    @property
    def is_consistent(self) -> bool:
        """Check if dtype is consistent across pipeline stages."""
        if not self.manifest_dtype or not self.ir_dtype:
            return True  # Can't verify

        # Normalize dtypes
        m_dtype = self.manifest_dtype.lower().replace('_', '')
        i_dtype = self.ir_dtype.lower().replace('_', '')

        # Handle fp32/f32 equivalence
        if m_dtype in ('fp32', 'f32') and i_dtype in ('fp32', 'f32'):
            return True

        return m_dtype == i_dtype

    @property
    def kernel_matches_dtype(self) -> bool:
        """Check if the kernel suffix matches the expected dtype."""
        if not self.codegen_kernel or not self.ir_dtype:
            return True

        # Special cases: some kernels don't have dtype suffix
        dtype_agnostic_kernels = [
            'rmsnorm_forward',
            'rope_forward_qk',
            'attention_forward',
            'kv_cache_store',
            'swiglu_forward',
            'add_inplace_f32',
            'ck_residual_add',
        ]
        for kernel in dtype_agnostic_kernels:
            if kernel in self.codegen_kernel:
                return True

        expected_suffix = DTYPE_TO_KERNEL_SUFFIX.get(self.ir_dtype.lower(), '')
        if not expected_suffix:
            return True

        # Check if kernel contains the expected suffix
        return expected_suffix in self.codegen_kernel.lower()


class QuantizationAuditor:
    """Audits quantization consistency through the pipeline."""

    def __init__(self, model_dir: Path, verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.verbose = verbose
        self.weights: Dict[str, WeightInfo] = {}
        self.issues: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def load_manifest(self):
        """Load weights manifest to get actual dtypes."""
        manifest_path = self.model_dir / "weights_manifest.json"
        if not manifest_path.exists():
            print(f"Warning: No weights_manifest.json found")
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Parse header weights
        for name, info in manifest.get('header_weights', {}).items():
            key = f"header.{name}"
            self.weights[key] = WeightInfo(
                name=name,
                layer=-1,
                role=name,
                manifest_dtype=info.get('dtype', ''),
                manifest_offset=info.get('offset', 0),
                size=info.get('size', 0)
            )

        # Parse layer weights
        for layer_idx, layer_data in enumerate(manifest.get('layer_weights', [])):
            for role, info in layer_data.items():
                if isinstance(info, dict):
                    key = f"layer.{layer_idx}.{role}"
                    self.weights[key] = WeightInfo(
                        name=f"layer.{layer_idx}.{role}",
                        layer=layer_idx,
                        role=role,
                        manifest_dtype=info.get('dtype', ''),
                        manifest_offset=info.get('offset', 0),
                        size=info.get('size', 0)
                    )

        self.log(f"Loaded {len(self.weights)} weights from manifest")

    def load_lowered_ir(self):
        """Load lowered IR to get IR-level dtypes."""
        for mode in ['decode', 'prefill']:
            ir_path = self.model_dir / f"lowered_{mode}.json"
            if not ir_path.exists():
                continue

            with open(ir_path) as f:
                ir = json.load(f)

            # Parse weight entries from memory section
            for entry in ir.get('memory', {}).get('weights', {}).get('entries', []):
                name = entry.get('name', '')
                dtype = entry.get('dtype', '')
                offset = entry.get('offset', 0)

                # Find matching weight
                key = name.replace('.', '.').lower()

                # Try exact match first
                for wkey, winfo in self.weights.items():
                    if wkey.lower() == key or winfo.name.lower() == name.lower():
                        winfo.ir_dtype = dtype
                        winfo.ir_offset = offset
                        break
                else:
                    # Create new entry if not found
                    layer = -1
                    role = name
                    if 'layer.' in name:
                        parts = name.split('.')
                        try:
                            layer = int(parts[1])
                            role = parts[2] if len(parts) > 2 else name
                        except (IndexError, ValueError):
                            pass

                    self.weights[name] = WeightInfo(
                        name=name,
                        layer=layer,
                        role=role,
                        ir_dtype=dtype,
                        ir_offset=offset
                    )

            self.log(f"Updated weights from {mode} IR")
            break  # Only need one mode

    def parse_generated_c(self):
        """Parse generated C code to find kernel calls and their weight arguments."""
        c_path = self.model_dir / "model_v6_6.c"
        if not c_path.exists():
            print(f"Warning: No model_v6_6.c found")
            return

        content = c_path.read_text()

        # Parse kernel calls with their arguments
        # Pattern: kernel_name(..., W_LAYER_X_YY, ...)
        kernel_patterns = [
            # gemv calls: gemv_q5_0(y, W, x, M, K)
            (r'(gemv_\w+)\s*\([^,]+,\s*[^,]*\+\s*(W_\w+)', 'matmul'),
            # gemm calls: gemm_nt_q5_0(a, W, bias, out, M, N, K)
            (r'(gemm_nt_\w+)\s*\([^,]+,\s*[^,]*\+\s*(W_\w+)', 'matmul'),
            # rmsnorm: rmsnorm_forward(..., gamma, ...)
            (r'(rmsnorm_forward)\s*\([^,]+,\s*[^,]*\+\s*(W_\w+)', 'norm'),
            # embedding: embedding_forward_q8_0(..., emb, ...)
            (r'(embedding_forward_\w+)\s*\([^)]*\+\s*(W_\w+)', 'embed'),
        ]

        for pattern, op_type in kernel_patterns:
            for match in re.finditer(pattern, content):
                kernel = match.group(1)
                weight_define = match.group(2)

                # Convert W_LAYER_0_WQ to layer.0.wq
                weight_key = self._define_to_key(weight_define)

                if weight_key in self.weights:
                    self.weights[weight_key].codegen_kernel = kernel
                else:
                    # Try to find by matching role
                    for key, winfo in self.weights.items():
                        if weight_define.endswith(winfo.role.upper()):
                            winfo.codegen_kernel = kernel
                            break

        self.log(f"Parsed kernel calls from generated C")

    def _define_to_key(self, define: str) -> str:
        """Convert W_LAYER_0_WQ to layer.0.wq format."""
        define = define.upper()

        if define.startswith('W_LAYER_'):
            # W_LAYER_0_WQ -> layer.0.wq
            parts = define[8:].split('_', 1)
            if len(parts) == 2:
                layer = parts[0]
                role = parts[1].lower()
                return f"layer.{layer}.{role}"
        elif define.startswith('W_'):
            # W_TOKEN_EMB -> header.token_emb
            return f"header.{define[2:].lower()}"

        return define.lower()

    def audit(self, layer_filter: Optional[int] = None) -> List[str]:
        """Run the audit and return issues."""
        self.load_manifest()
        self.load_lowered_ir()
        self.parse_generated_c()

        issues = []

        # Group weights by layer for organized output
        by_layer = defaultdict(list)
        for key, winfo in self.weights.items():
            if layer_filter is not None and winfo.layer != layer_filter:
                continue
            by_layer[winfo.layer].append(winfo)

        # Check each weight
        for layer_idx in sorted(by_layer.keys()):
            weights = by_layer[layer_idx]
            layer_name = f"Layer {layer_idx}" if layer_idx >= 0 else "Header"

            for winfo in sorted(weights, key=lambda w: w.role):
                # Check dtype consistency
                if not winfo.is_consistent:
                    issue = (f"{layer_name} {winfo.role}: dtype mismatch - "
                            f"manifest={winfo.manifest_dtype}, ir={winfo.ir_dtype}")
                    issues.append(issue)

                # Check kernel matches dtype
                if winfo.codegen_kernel and winfo.ir_dtype:
                    if not winfo.kernel_matches_dtype:
                        expected = DTYPE_TO_KERNEL_SUFFIX.get(winfo.ir_dtype.lower(), winfo.ir_dtype)
                        issue = (f"{layer_name} {winfo.role}: kernel mismatch - "
                                f"using {winfo.codegen_kernel} but dtype is {winfo.ir_dtype} "
                                f"(expected *_{expected})")
                        issues.append(issue)

                # Check offset consistency
                if winfo.manifest_offset and winfo.ir_offset:
                    if winfo.manifest_offset != winfo.ir_offset:
                        issue = (f"{layer_name} {winfo.role}: offset mismatch - "
                                f"manifest={winfo.manifest_offset}, ir={winfo.ir_offset}")
                        issues.append(issue)

        self.issues = issues
        return issues

    def print_report(self, layer_filter: Optional[int] = None):
        """Print detailed audit report."""
        print("\n" + "=" * 80)
        print("QUANTIZATION CHAIN AUDIT")
        print("=" * 80)

        # Group by layer
        by_layer = defaultdict(list)
        for key, winfo in self.weights.items():
            if layer_filter is not None and winfo.layer != layer_filter:
                continue
            by_layer[winfo.layer].append(winfo)

        for layer_idx in sorted(by_layer.keys()):
            weights = by_layer[layer_idx]
            layer_name = f"Layer {layer_idx}" if layer_idx >= 0 else "Header"

            print(f"\n{layer_name}")
            print("-" * 60)
            print(f"{'Role':<15} {'Manifest':<10} {'IR':<10} {'Kernel':<25} {'Status'}")
            print("-" * 60)

            for winfo in sorted(weights, key=lambda w: w.role):
                manifest_dtype = winfo.manifest_dtype or "-"
                ir_dtype = winfo.ir_dtype or "-"
                kernel = winfo.codegen_kernel or "-"

                # Determine status
                if winfo.is_consistent and winfo.kernel_matches_dtype:
                    status = "[OK]"
                elif not winfo.is_consistent:
                    status = "[DTYPE MISMATCH]"
                elif not winfo.kernel_matches_dtype:
                    status = "[KERNEL MISMATCH]"
                else:
                    status = "[?]"

                print(f"{winfo.role:<15} {manifest_dtype:<10} {ir_dtype:<10} {kernel:<25} {status}")

        # Print issues summary
        if self.issues:
            print("\n" + "=" * 80)
            print(f"ISSUES FOUND: {len(self.issues)}")
            print("=" * 80)
            for issue in self.issues:
                print(f"  [!] {issue}")
        else:
            print("\n" + "=" * 80)
            print("[OK] No quantization mismatches found")
            print("=" * 80)

    def print_layer_dtype_summary(self):
        """Print a compact summary of dtypes per layer."""
        print("\n" + "=" * 80)
        print("LAYER DTYPE SUMMARY")
        print("=" * 80)

        # Collect unique dtypes per layer
        by_layer = defaultdict(lambda: defaultdict(set))
        for key, winfo in self.weights.items():
            if winfo.layer < 0:
                continue
            dtype = winfo.ir_dtype or winfo.manifest_dtype
            if dtype:
                by_layer[winfo.layer][winfo.role].add(dtype)

        if not by_layer:
            print("No layer data available")
            return

        # Get all roles
        all_roles = set()
        for layer_data in by_layer.values():
            all_roles.update(layer_data.keys())

        # Print header
        roles = sorted(all_roles)
        header = f"{'Layer':<6}" + "".join(f"{r:<10}" for r in roles)
        print(header)
        print("-" * len(header))

        # Print each layer
        for layer_idx in sorted(by_layer.keys()):
            row = f"{layer_idx:<6}"
            for role in roles:
                dtypes = by_layer[layer_idx].get(role, set())
                dtype_str = ",".join(sorted(dtypes)) if dtypes else "-"
                row += f"{dtype_str:<10}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Audit quantization consistency through IR pipeline"
    )
    parser.add_argument(
        'model_dir',
        help='Path to model cache directory'
    )
    parser.add_argument(
        '-l', '--layer',
        type=int,
        help='Filter to specific layer'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print compact dtype summary'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Directory not found: {model_dir}")
        sys.exit(1)

    auditor = QuantizationAuditor(model_dir, verbose=args.verbose)
    issues = auditor.audit(layer_filter=args.layer)

    if args.json:
        result = {
            'model_dir': str(model_dir),
            'issues': issues,
            'weights': {
                k: {
                    'name': v.name,
                    'layer': v.layer,
                    'role': v.role,
                    'manifest_dtype': v.manifest_dtype,
                    'ir_dtype': v.ir_dtype,
                    'codegen_kernel': v.codegen_kernel,
                }
                for k, v in auditor.weights.items()
            }
        }
        print(json.dumps(result, indent=2))
    else:
        if args.summary:
            auditor.print_layer_dtype_summary()
        else:
            auditor.print_report(layer_filter=args.layer)

    sys.exit(1 if issues else 0)


if __name__ == '__main__':
    main()

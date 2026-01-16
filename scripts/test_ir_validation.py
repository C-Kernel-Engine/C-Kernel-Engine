#!/usr/bin/env python3
"""
Layer 3: IR Structure Validation
=================================

Validates that the IR (lowered JSON) is correct and consistent.

Tests:
1. IR structure completeness
2. Symbol consistency (E, H, D, etc.)
3. Operation sequence per layer
4. Weight dtype vs manifest match
5. Buffer allocation (no overlaps)
6. Dimension consistency

Usage:
    python scripts/test_ir_validation.py --ir path/to/lowered_decode.json --manifest path/to/weights_manifest.json
    python scripts/test_ir_validation.py --model-dir path/to/model_dir
    python scripts/test_ir_validation.py --auto  # Auto-find cached models
"""

import os
import sys
import json
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


class IRValidator:
    """Validate IR structure and consistency."""

    def __init__(self, ir_path: str, manifest_path: str, verbose: bool = False):
        self.ir_path = Path(ir_path)
        self.manifest_path = Path(manifest_path)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

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
        """Load IR and manifest files."""
        try:
            with open(self.ir_path) as f:
                self.ir = json.load(f)
            self.log_pass(f"Loaded IR: {self.ir_path.name}")
        except Exception as e:
            self.log_fail(f"Failed to load IR: {e}")
            return False

        try:
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            self.log_pass(f"Loaded manifest: {self.manifest_path.name}")
        except Exception as e:
            self.log_fail(f"Failed to load manifest: {e}")
            return False

        return True

    def test_ir_structure(self) -> bool:
        """Test 1: IR has required structure."""
        self.log_info("Test 1: IR Structure Completeness")

        required_fields = ['version', 'kind', 'mode']
        optional_fields = ['model', 'config', 'symbols', 'sections']

        all_present = True
        for field in required_fields:
            if field in self.ir:
                if self.verbose:
                    print(f"  ✓ {field}: {self.ir[field]}")
            else:
                self.log_fail(f"Missing required field: {field}")
                all_present = False

        # Check version
        version = self.ir.get('version', 0)
        if version >= 3:
            self.log_pass(f"IR version {version} is supported")
        else:
            self.log_warn(f"IR version {version} may be outdated")

        # Check mode
        mode = self.ir.get('mode', '')
        if mode in ['decode', 'prefill', 'both']:
            self.log_pass(f"Mode: {mode}")
        else:
            self.log_fail(f"Unknown mode: {mode}")
            all_present = False

        return all_present

    def _get_symbol_value(self, symbols: Dict, key: str) -> Optional[int]:
        """Extract symbol value, handling both direct ints and {'value': int} format."""
        if key not in symbols:
            return None
        val = symbols[key]
        if isinstance(val, dict):
            return val.get('value')
        return val

    def test_symbol_consistency(self) -> bool:
        """Test 2: Symbols are consistent with config."""
        self.log_info("Test 2: Symbol Consistency")

        symbols = self.ir.get('symbols', {})
        config = self.ir.get('config', {})

        if not symbols and not config:
            self.log_warn("No symbols or config found")
            return True

        issues = []

        # Get symbol values (handle both formats)
        E = self._get_symbol_value(symbols, 'E')
        H = self._get_symbol_value(symbols, 'H')
        D = self._get_symbol_value(symbols, 'D')
        I = self._get_symbol_value(symbols, 'I')

        # Check E (embed_dim)
        if E is not None and 'embed_dim' in config:
            if E != config['embed_dim']:
                issues.append(f"E={E} but embed_dim={config['embed_dim']}")

        # Check H (num_heads)
        if H is not None and 'num_heads' in config:
            if H != config['num_heads']:
                issues.append(f"H={H} but num_heads={config['num_heads']}")

        # Check D (head_dim)
        if D is not None and 'head_dim' in config:
            if D != config['head_dim']:
                issues.append(f"D={D} but head_dim={config['head_dim']}")

        # Check head_dim = embed_dim / num_heads (for non-GQA)
        if E is not None and H is not None and D is not None and H > 0:
            expected_d = E // H
            if D != expected_d:
                # This is OK for GQA, just warn
                self.log_warn(f"D={D} but E/H={expected_d} (GQA?)")

        # Check I (intermediate_dim)
        if I is not None and 'intermediate_dim' in config:
            if I != config['intermediate_dim']:
                issues.append(f"I={I} but intermediate_dim={config['intermediate_dim']}")

        if not issues:
            self.log_pass(f"All symbols consistent: E={E}, H={H}, D={D}")
            return True
        else:
            for issue in issues:
                self.log_fail(f"Symbol mismatch: {issue}")
            return False

    def test_layer_operations(self) -> bool:
        """Test 3: Each layer has expected operations."""
        self.log_info("Test 3: Layer Operation Sequence")

        sections = self.ir.get('sections', [])
        if not sections:
            self.log_warn("No sections found in IR")
            return True

        # Expected operation patterns for transformer layers
        expected_ops_patterns = [
            # Attention block
            ['rmsnorm', 'linear', 'rope', 'attention'],
            # Or simplified
            ['norm', 'qkv', 'rope', 'attn'],
            # MLP block
            ['rmsnorm', 'linear', 'linear', 'linear'],  # gate, up, down
            # Or fused
            ['norm', 'mlp'],
        ]

        layer_count = 0
        issues = []

        for section in sections:
            layers = section.get('layers', [])
            for layer in layers:
                layer_id = layer.get('layer_id', layer_count)
                ops = layer.get('ops', [])

                if not ops:
                    issues.append(f"Layer {layer_id}: no operations")
                    continue

                op_names = [op.get('op', op.get('type', '')) for op in ops]

                # Check for basic operations
                has_norm = any('norm' in op.lower() for op in op_names)
                has_linear = any('linear' in op.lower() or 'gemm' in op.lower() or 'proj' in op.lower() for op in op_names)
                has_attention = any('attn' in op.lower() or 'attention' in op.lower() for op in op_names)

                if not has_norm:
                    issues.append(f"Layer {layer_id}: missing normalization")
                if not has_linear:
                    issues.append(f"Layer {layer_id}: missing linear/projection")
                if not has_attention:
                    # MLP layers don't have attention, that's OK
                    pass

                layer_count += 1

                if self.verbose:
                    print(f"  Layer {layer_id}: {len(ops)} ops - {', '.join(op_names[:5])}...")

        if layer_count == 0:
            self.log_warn("No layers found")
        else:
            self.log_pass(f"Found {layer_count} layers")

        if not issues:
            self.log_pass("All layers have valid operation sequences")
            return True
        else:
            for issue in issues[:5]:  # Limit output
                self.log_fail(issue)
            if len(issues) > 5:
                self.log_fail(f"... and {len(issues) - 5} more issues")
            return False

    def test_weight_dtype_match(self) -> bool:
        """Test 4: Weight dtypes in IR match manifest."""
        self.log_info("Test 4: Weight Dtype vs Manifest")

        # Get weight dtypes from manifest
        manifest_dtypes = {}
        if 'tensors' in self.manifest:
            for name, info in self.manifest['tensors'].items():
                dtype = info.get('dtype', info.get('type', ''))
                manifest_dtypes[name] = dtype.lower().replace('_', '')
        elif 'weights' in self.manifest:
            for name, info in self.manifest['weights'].items():
                dtype = info.get('dtype', info.get('type', ''))
                manifest_dtypes[name] = dtype.lower().replace('_', '')

        if not manifest_dtypes:
            self.log_warn("No weight dtypes found in manifest")
            return True

        # Extract weight references from IR
        ir_weight_refs = []
        sections = self.ir.get('sections', [])
        for section in sections:
            # Header ops
            for op in section.get('header', {}).get('ops', []):
                weights = op.get('weights', [])
                for w in weights:
                    dtype = op.get('weight_dtype', op.get('dtype', ''))
                    ir_weight_refs.append((w, dtype.lower().replace('_', '')))

            # Layer ops
            for layer in section.get('layers', []):
                for op in layer.get('ops', []):
                    weights = op.get('weights', op.get('weight', []))
                    if isinstance(weights, str):
                        weights = [weights]
                    dtype = op.get('weight_dtype', op.get('dtype', ''))
                    for w in weights:
                        ir_weight_refs.append((w, dtype.lower().replace('_', '')))

        if not ir_weight_refs:
            self.log_warn("No weight references found in IR ops")
            return True

        # Compare
        matches = 0
        mismatches = 0

        for weight_name, ir_dtype in ir_weight_refs:
            if not ir_dtype:
                continue

            # Find in manifest (try variations)
            manifest_dtype = None
            for mname, mdtype in manifest_dtypes.items():
                if weight_name in mname or mname in weight_name:
                    manifest_dtype = mdtype
                    break

            if manifest_dtype and ir_dtype:
                if ir_dtype == manifest_dtype or ir_dtype in manifest_dtype or manifest_dtype in ir_dtype:
                    matches += 1
                else:
                    mismatches += 1
                    if self.verbose:
                        print(f"  {weight_name}: IR={ir_dtype}, Manifest={manifest_dtype}")

        if matches > 0 and mismatches == 0:
            self.log_pass(f"All {matches} weight dtypes match")
            return True
        elif mismatches > 0:
            self.log_fail(f"Weight dtype mismatches: {mismatches}/{matches + mismatches}")
            return False
        else:
            self.log_warn("Could not verify weight dtypes")
            return True

    def test_buffer_allocation(self) -> bool:
        """Test 5: Buffer allocations don't overlap."""
        self.log_info("Test 5: Buffer Allocation")

        buffers = []

        sections = self.ir.get('sections', [])
        for section in sections:
            # Section-level buffers
            for buf in section.get('buffers', []):
                offset = buf.get('offset', 0)
                size = buf.get('size', buf.get('bytes', 0))
                name = buf.get('name', 'unnamed')
                if size > 0:
                    buffers.append((name, offset, size))

            # Layer-level buffers
            for layer in section.get('layers', []):
                for buf in layer.get('buffers', []):
                    offset = buf.get('offset', 0)
                    size = buf.get('size', buf.get('bytes', 0))
                    name = buf.get('name', f"layer_{layer.get('layer_id', '?')}_unnamed")
                    if size > 0:
                        buffers.append((name, offset, size))

        if not buffers:
            self.log_warn("No buffer allocations found in IR")
            return True

        # Sort by offset and check for overlaps
        buffers.sort(key=lambda x: x[1])

        overlaps = []
        for i in range(len(buffers) - 1):
            name1, offset1, size1 = buffers[i]
            name2, offset2, size2 = buffers[i + 1]

            end1 = offset1 + size1
            if end1 > offset2:
                overlaps.append((name1, name2, end1 - offset2))

        if not overlaps:
            total_size = max(b[1] + b[2] for b in buffers) if buffers else 0
            self.log_pass(f"No buffer overlaps ({len(buffers)} buffers, {total_size / (1024*1024):.1f} MB total)")
            return True
        else:
            for name1, name2, overlap_bytes in overlaps[:5]:
                self.log_fail(f"Buffer overlap: {name1} and {name2} ({overlap_bytes} bytes)")
            return False

    def test_dimension_consistency(self) -> bool:
        """Test 6: Dimensions are consistent throughout IR."""
        self.log_info("Test 6: Dimension Consistency")

        symbols = self.ir.get('symbols', {})
        config = self.ir.get('config', {})

        # Expected dimensions - use helper to handle dict format
        embed_dim = self._get_symbol_value(symbols, 'E') or config.get('embed_dim', 0)
        num_heads = self._get_symbol_value(symbols, 'H') or config.get('num_heads', 0)
        head_dim = self._get_symbol_value(symbols, 'D') or config.get('head_dim', 0)
        intermediate = self._get_symbol_value(symbols, 'I') or config.get('intermediate_dim', 0)
        vocab_size = self._get_symbol_value(symbols, 'V') or config.get('vocab_size', 0)
        max_seq_len = self._get_symbol_value(symbols, 'T') or config.get('max_seq_len', 0)

        issues = []

        # Check key invariants
        if embed_dim and num_heads and head_dim:
            # For standard attention: embed_dim = num_heads * head_dim
            expected_embed = num_heads * head_dim
            if embed_dim != expected_embed:
                # Could be GQA, check if it's a reasonable ratio
                ratio = embed_dim / (num_heads * head_dim) if num_heads * head_dim else 0
                if ratio not in [1, 2, 4, 8]:  # Common GQA ratios
                    issues.append(f"embed_dim={embed_dim} but num_heads*head_dim={expected_embed}")

        # Check intermediate dim is reasonable
        if embed_dim and intermediate:
            ratio = intermediate / embed_dim
            if ratio < 1 or ratio > 10:
                issues.append(f"Unusual intermediate ratio: {ratio:.1f}x embed_dim")

        # Check vocab size
        if vocab_size and vocab_size < 1000:
            issues.append(f"Vocab size seems small: {vocab_size}")

        # Check max seq len
        if max_seq_len and max_seq_len < 512:
            self.log_warn(f"Max sequence length is short: {max_seq_len}")

        if not issues:
            self.log_pass(f"Dimensions consistent: E={embed_dim}, H={num_heads}, D={head_dim}, I={intermediate}")
            return True
        else:
            for issue in issues:
                self.log_fail(issue)
            return False

    def print_ir_summary(self):
        """Print a summary of the IR structure."""
        print()
        print(f"{CYAN}IR Summary:{NC}")

        config = self.ir.get('config', {})
        symbols = self.ir.get('symbols', {})

        print(f"  Model: {self.ir.get('model', 'unknown')}")
        print(f"  Mode: {self.ir.get('mode', 'unknown')}")
        print(f"  Version: {self.ir.get('version', 'unknown')}")
        print()
        print(f"  embed_dim (E): {symbols.get('E', config.get('embed_dim', '?'))}")
        print(f"  num_heads (H): {symbols.get('H', config.get('num_heads', '?'))}")
        print(f"  head_dim (D): {symbols.get('D', config.get('head_dim', '?'))}")
        print(f"  intermediate (I): {symbols.get('I', config.get('intermediate_dim', '?'))}")
        print(f"  num_layers: {config.get('num_layers', '?')}")
        print(f"  vocab_size (V): {symbols.get('V', config.get('vocab_size', '?'))}")
        print(f"  max_seq_len (T): {symbols.get('T', config.get('max_seq_len', '?'))}")
        print()

    def run_all_tests(self) -> bool:
        """Run all IR validation tests."""
        print("=" * 60)
        print("  CK-Engine IR Validation")
        print("=" * 60)
        print(f"IR: {self.ir_path}")
        print(f"Manifest: {self.manifest_path}")
        print()

        # Load files
        if not self.load_files():
            return False
        print()

        # Print summary
        self.print_ir_summary()

        # Run tests
        self.test_ir_structure()
        print()

        self.test_symbol_consistency()
        print()

        self.test_layer_operations()
        print()

        self.test_weight_dtype_match()
        print()

        self.test_buffer_allocation()
        print()

        self.test_dimension_consistency()
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
            print(f"{GREEN}All IR validation tests passed!{NC}")
            return True
        else:
            print(f"{RED}Some tests failed. Check IR generation.{NC}")
            return False


def find_ir_files(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find IR and manifest files in a model directory."""
    ir_candidates = [
        model_dir / "lowered_decode.json",
        model_dir / "lowered_prefill.json",
        model_dir / "ir_decode.json",
        model_dir / "ir.json",
    ]

    manifest_candidates = [
        model_dir / "weights_manifest.json",
        model_dir / "manifest.json",
    ]

    ir_path = None
    for path in ir_candidates:
        if path.exists():
            ir_path = path
            break

    manifest_path = None
    for path in manifest_candidates:
        if path.exists():
            manifest_path = path
            break

    return ir_path, manifest_path


def find_cached_models() -> List[Tuple[Path, Path]]:
    """Find IR + manifest pairs in cache directories."""
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

            ir_path, manifest_path = find_ir_files(model_dir)
            if ir_path and manifest_path:
                pairs.append((ir_path, manifest_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validate IR structure")
    parser.add_argument("--ir", type=str, help="Path to IR JSON file")
    parser.add_argument("--manifest", type=str, help="Path to manifest JSON file")
    parser.add_argument("--model-dir", type=str, help="Model directory (auto-find IR and manifest)")
    parser.add_argument("--auto", action="store_true", help="Auto-find cached models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.auto:
        pairs = find_cached_models()
        if not pairs:
            print(f"{RED}No cached IR/manifest pairs found{NC}")
            sys.exit(1)

        print(f"Found {len(pairs)} IR/manifest pairs")
        for ir, manifest in pairs:
            print(f"  - {ir.parent.name}/{ir.name}")
        print()

        ir_path, manifest_path = pairs[0]

    elif args.model_dir:
        model_dir = Path(args.model_dir)
        ir_path, manifest_path = find_ir_files(model_dir)
        if not ir_path:
            print(f"{RED}No IR file found in {model_dir}{NC}")
            sys.exit(1)
        if not manifest_path:
            print(f"{RED}No manifest file found in {model_dir}{NC}")
            sys.exit(1)

    elif args.ir and args.manifest:
        ir_path = Path(args.ir)
        manifest_path = Path(args.manifest)

    else:
        parser.print_help()
        sys.exit(1)

    validator = IRValidator(str(ir_path), str(manifest_path), verbose=args.verbose)
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

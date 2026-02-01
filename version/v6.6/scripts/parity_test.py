#!/usr/bin/env python3
"""
parity_test.py - Layer-by-layer parity comparison between llama.cpp and CKE

This tool compares intermediate activations dumped from both runtimes to
identify EXACTLY where they diverge.

Usage:
    # Run comparison with existing dumps
    python version/v6.6/scripts/parity_test.py --ck-dump ck_parity_dumps/dump.bin

    # Compare both runtimes
    python version/v6.6/scripts/parity_test.py --gguf model.gguf --tokens 1

    # With tolerance
    python version/v6.6/scripts/parity_test.py --ck-dump ck_parity_dumps/dump.bin --atol 1e-3
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# Binary Dump Reader
# =============================================================================

CKDUMP_MAGIC = b"CKDMP\x00\x00"

# Header format (128 bytes total)
HEADER_SPEC = struct.Struct(
    "<8s"      # magic (8)
    "I"        # version (4)
    "i"        # layer_id (4)
    "32s"      # op_name (32)
    "I"        # dtype (4)
    "I"        # rank (4)
    "4q"       # shape[4] (32)
    "I"        # elem_count (4)
    "i"        # token_id (4)
    "24x"      # reserved (24)
)
HEADER_SIZE = 128


class ParityDump:
    """Represents a single tensor dump."""

    def __init__(self, layer_id: int, op_name: str, data: np.ndarray,
                 token_id: int, dtype: str):
        self.layer_id = layer_id
        self.op_name = op_name
        self.data = data
        self.token_id = token_id
        self.dtype = dtype

    def __repr__(self):
        return f"ParityDump(layer={self.layer_id}, op={self.op_name}, " \
               f"shape={self.data.shape}, token={self.token_id})"


def read_dump_file(path: Path) -> List[ParityDump]:
    """Read all tensor dumps from a binary file."""
    dumps = []

    if not path.exists():
        print(f"[WARNING] Dump file not found: {path}")
        return dumps

    with open(path, "rb") as f:
        while True:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                break

            unpacked = HEADER_SPEC.unpack(header_bytes)
            magic, version, layer_id, op_name_bytes, dtype, rank, \
                shape0, shape1, shape2, shape3, elem_count, token_id = unpacked

            # Verify magic
            if magic != CKDUMP_MAGIC:
                print(f"[WARNING] Invalid magic at offset {f.tell() - HEADER_SIZE}")
                continue

            # Parse op name (null-terminated)
            op_name = op_name_bytes.split(b"\x00")[0].decode("utf-8", errors="ignore")

            # Get actual shape based on rank
            shape = [shape0, shape1, shape2, shape3][:rank]
            if not any(shape):
                shape = [elem_count]

            # Read data
            dtype_map = {0: np.float32, 1: np.float16, 2: np.float16}
            np_dtype = dtype_map.get(dtype, np.float32)
            elem_size = 4 if dtype == 0 else 2
            data_size = elem_count * elem_size

            data_bytes = f.read(data_size)
            if len(data_bytes) < data_size:
                print(f"[WARNING] Unexpected EOF reading {op_name}")
                break

            # Convert to numpy
            data = np.frombuffer(data_bytes, dtype=np_dtype).astype(np.float32)
            data = data.reshape(shape)

            dtype_name = ["fp32", "fp16", "bf16"][dtype] if dtype < 3 else "unknown"
            dumps.append(ParityDump(layer_id, op_name, data, token_id, dtype_name))

    return dumps


# =============================================================================
# Reference Data (llama.cpp or expected values)
# =============================================================================

class ReferenceData:
    """Reference activations for comparison."""

    def __init__(self, dump_path: Optional[Path] = None):
        self.dumps = []
        self.by_layer_op: Dict[Tuple[int, str], List[ParityDump]] = {}

        if dump_path and dump_path.exists():
            self.load(dump_path)

    def load(self, path: Path):
        """Load reference dumps from file."""
        self.dumps = read_dump_file(path)
        for d in self.dumps:
            key = (d.layer_id, d.op_name)
            if key not in self.by_layer_op:
                self.by_layer_op[key] = []
            self.by_layer_op[key].append(d)

    def get(self, layer_id: int, op_name: str, token_id: int = 0) -> Optional[ParityDump]:
        """Get a specific dump by key."""
        key = (layer_id, op_name)
        if key not in self.by_layer_op:
            return None
        for d in self.by_layer_op[key]:
            if d.token_id == token_id:
                return d
        # Return first if token_id not found
        return self.by_layer_op[key][0] if self.by_layer_op[key] else None


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_dumps(
    ref_dump: ParityDump,
    test_dump: ParityDump,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict:
    """Compare two dumps and return statistics."""

    # Trim to same size if needed
    ref_data = ref_dump.data.flatten()
    test_data = test_dump.data.flatten()

    min_len = min(len(ref_data), len(test_data))
    ref_data = ref_data[:min_len]
    test_data = test_data[:min_len]

    if min_len == 0:
        return {
            "status": "ERROR",
            "max_abs_diff": float('inf'),
            "mean_abs_diff": float('inf'),
            "max_rel_err": float('inf'),
            "diverge_idx": None,
            "has_nan": False,
            "has_inf": False,
            "ref_range": (0, 0),
            "test_range": (0, 0),
            "ref_shape": ref_dump.data.shape,
            "test_shape": test_dump.data.shape,
            "size_mismatch": True,
        }

    # Compute differences
    abs_diff = np.abs(ref_data - test_data)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Relative error (handle near-zero values)
    safe_ref = np.abs(ref_data) + atol
    rel_err = abs_diff / safe_ref
    max_rel_err = np.max(rel_err)
    mean_rel_err = np.mean(rel_err)

    # Find first divergence
    diverge_idx = None
    diverge_thresh = atol
    if max_abs_diff > atol:
        diverge_indices = np.where(abs_diff > diverge_thresh)[0]
        if len(diverge_indices) > 0:
            diverge_idx = int(diverge_indices[0])

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(test_data)) or np.any(np.isnan(ref_data))
    has_inf = np.any(np.isinf(test_data)) or np.any(np.isinf(ref_data))

    # Determine status
    if has_nan or has_inf:
        status = "ERROR"
    elif max_abs_diff <= atol:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "max_rel_err": float(max_rel_err),
        "mean_rel_err": float(mean_rel_err),
        "diverge_idx": diverge_idx,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "ref_range": (float(np.min(ref_data)), float(np.max(ref_data))),
        "test_range": (float(np.min(test_data)), float(np.max(test_data))),
        "ref_shape": ref_dump.data.shape,
        "test_shape": test_dump.data.shape,
        "size_mismatch": False,
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_parity_test(
    ck_dump_path: Path,
    ref_dump_path: Optional[Path] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[int, List[Dict]]:
    """Run parity test comparing CKE dumps against reference.

    Args:
        ck_dump_path: Path to CKE dump file
        ref_dump_path: Optional path to reference (llama.cpp) dump
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: Print results

    Returns:
        (exit_code, results_list)
    """

    # Load CKE dumps
    ck_dumps = read_dump_file(ck_dump_path)
    if not ck_dumps:
        print(f"[ERROR] No dumps found in {ck_dump_path}")
        return 1, []

    # Load reference if provided
    reference = ReferenceData(ref_dump_path) if ref_dump_path else ReferenceData()

    if verbose:
        print("=" * 80)
        print("LAYER-BY-LAYER PARITY TEST")
        print("=" * 80)
        print(f"CKE dump: {ck_dump_path} ({len(ck_dumps)} tensors)")
        if ref_dump_path:
            print(f"Reference: {ref_dump_path} ({len(reference.dumps)} tensors)")
        else:
            print(f"Reference: None (checking for NaN/Inf only)")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        print("=" * 80)

    # Organize CKE dumps
    ck_by_layer_op: Dict[Tuple[int, str], ParityDump] = {}
    for d in ck_dumps:
        key = (d.layer_id, d.op_name)
        if key not in ck_by_layer_op:
            ck_by_layer_op[key] = d

    # Get all keys
    all_keys = sorted(set(ck_by_layer_op.keys()) | set(reference.by_layer_op.keys()))

    results = []

    if verbose:
        print(f"\n{'Layer':<6} {'Op':<18} {'Status':<8} {'Max Diff':<12} {'Mean Diff':<12} {'Ref Range':<20} {'Test Range':<20}")
        print("-" * 130)

    for layer_id, op_name in all_keys:
        ck_dump = ck_by_layer_op.get((layer_id, op_name))
        ref_dump = reference.get(layer_id, op_name)

        if notck_dump:
            status = "MISSING"
            max_diff = mean_diff = "N/A"
            ref_range = test_range = "N/A"
            result = {
                "layer": layer_id,
                "op": op_name,
                "status": status,
                "max_abs_diff": float('inf'),
                "ck_missing": True,
            }
        else:
            # Check for NaN/Inf first (even without reference)
            has_nan = np.any(np.isnan(ck_dump.data))
            has_inf = np.any(np.isinf(ck_dump.data))

            if has_nan or has_inf:
                status = "ERROR"
                max_diff = "NaN/Inf"
                mean_diff = "NaN/Inf"
                ref_range = "N/A"
                test_range = f"[{np.nanmin(ck_dump.data):.2e}, {np.nanmax(ck_dump.data):.2e}]"
                result = {
                    "layer": layer_id,
                    "op": op_name,
                    "status": status,
                    "max_abs_diff": float('inf'),
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "ck_range": test_range,
                }
            elif ref_dump is None:
                status = "WARN"
                max_diff = mean_diff = "N/A"
                ref_range = "N/A"
                test_range = f"[{np.min(ck_dump.data):.2e}, {np.max(ck_dump.data):.2e}]"
                result = {
                    "layer": layer_id,
                    "op": op_name,
                    "status": status,
                    "max_abs_diff": 0,
                    "ref_missing": True,
                }
            else:
                comp = compare_dumps(ref_dump, ck_dump, atol, rtol)
                status = comp["status"]
                max_diff = f"{comp['max_abs_diff']:.2e}"
                mean_diff = f"{comp['mean_abs_diff']:.2e}"
                ref_range = f"[{comp['ref_range'][0]:.2e}, {comp['ref_range'][1]:.2e}]"
                test_range = f"[{comp['test_range'][0]:.2e}, {comp['test_range'][1]:.2e}]"
                result = {
                    "layer": layer_id,
                    "op": op_name,
                    **comp,
                }

        results.append(result)

        if verbose:
            # Color code
            if status == "PASS":
                color = "\033[92m"
            elif status in ("FAIL", "ERROR"):
                color = "\033[91m"
            elif status == "WARN":
                color = "\033[93m"
            else:
                color = "\033[90m"
            reset = "\033[0m"

            print(f"{layer_id:<6} {op_name:<18} {color}{status:<8}{reset} "
                  f"{max_diff:<12} {mean_diff:<12} {ref_range:<20} {test_range:<20}")

    if verbose:
        print("-" * 130)

        # Summary
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        errors = sum(1 for r in results if r["status"] == "ERROR")
        warnings = sum(1 for r in results if r["status"] == "WARN")
        total = len(results)

        print(f"\nSUMMARY: {passed}/{total} passed, {failed} failed, {errors} errors, {warnings} warnings")

        # Show first failures/errors
        if errors > 0:
            print("\n" + "=" * 80)
            print("ERRORS (NaN/Inf detected):")
            print("=" * 80)
            for r in results:
                if r["status"] == "ERROR":
                    print(f"  Layer {r['layer']}, Op {r['op']}: "
                          f"NaN={r.get('has_nan', False)}, Inf={r.get('has_inf', False)}")
                    if 'ck_range' in r:
                        print(f"    CKE range: {r['ck_range']}")

        if failed > 0:
            print("\n" + "=" * 80)
            print("FIRST FAILURES (divergence from reference):")
            print("=" * 80)
            count = 0
            for r in results:
                if r["status"] == "FAIL" and count < 5:
                    print(f"  Layer {r['layer']}, Op {r['op']}: "
                          f"max_diff={r['max_abs_diff']:.2e}, "
                          f"mean_diff={r['mean_abs_diff']:.2e}")
                    if 'diverge_idx' in r and r['diverge_idx'] is not None:
                        print(f"    First diverge @ idx {r['diverge_idx']}")
                    if 'ref_range' in r:
                        print(f"    Ref range:   {r['ref_range']}")
                        print(f"    Test range:  {r['test_range']}")
                    count += 1

        # Suggest next steps
        if errors > 0:
            print("\n" + "=" * 80)
            print("SUGGESTED ACTIONS:")
            print("=" * 80)
            print("1. Check the ERROR ops above — these contain NaN/Inf")
            print("2. Verify weight offsets for these ops in the manifest")
            print("3. Check kernel implementation for correctness")
            print("=" * 80)

    # Determine exit code
    if errors > 0:
        return 1, results
    elif failed > 0:
        return 2, results
    else:
        return 0, results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer-by-layer parity test for CKE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check CKE dump for NaN/Inf (no reference)
    python version/v6.6/scripts/parity_test.py --ck-dump ck_parity_dumps/dump.bin

    # Compare with llama.cpp reference
    python version/v6.6/scripts/parity_test.py \\
        --ck-dump ck_parity_dumps/dump.bin \\
        --ref-dump llama_parity_dumps/dump.bin

    # With custom tolerance
    python version/v6.6/scripts/parity_test.py \\
        --ck-dump ck_parity_dumps/dump.bin --atol 1e-3
        """
    )
    parser.add_argument("--ck-dump", type=Path, required=True,
                       help="Path to CKE dump file")
    parser.add_argument("--ref-dump", type=Path,
                       help="Path to reference (llama.cpp) dump file")
    parser.add_argument("--atol", type=float, default=1e-4,
                       help="Absolute tolerance (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                       help="Relative tolerance (default: 1e-3)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress output")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    args = parser.parse_args()

    exit_code, results = run_parity_test(
        ck_dump_path=args.ck_dump,
        ref_dump_path=args.ref_dump,
        atol=args.atol,
        rtol=args.rtol,
        verbose=not args.quiet,
    )

    if args.json:
        import json
        print(json.dumps(results, indent=2))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

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
import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# Binary Dump Reader
# =============================================================================

CKDUMP_MAGIC = b"CKDMP\x00\x00\x00"

# CKDMP format has drifted across branches; support the known variants.
# Keep the parser permissive and infer the record layout from the header bytes.
HEADER_FORMATS = [
    {
        "name": "rank_120",
        "spec": struct.Struct("<8sIi32sII4qIi24x"),  # 120 bytes
        "decoder": lambda u: {
            "magic": u[0],
            "version": u[1],
            "layer_id": u[2],
            "op_name_bytes": u[3],
            "dtype": u[4],
            "rank": u[5],
            "shape": [u[6], u[7], u[8], u[9]],
            "elem_count": u[10],
            "token_id": u[11],
        },
    },
    {
        "name": "rank_124",
        "spec": struct.Struct("<8sIi32sII4qIi28x"),  # 124 bytes
        "decoder": lambda u: {
            "magic": u[0],
            "version": u[1],
            "layer_id": u[2],
            "op_name_bytes": u[3],
            "dtype": u[4],
            "rank": u[5],
            "shape": [u[6], u[7], u[8], u[9]],
            "elem_count": u[10],
            "token_id": u[11],
        },
    },
    {
        "name": "rank_dump_type_128",
        "spec": struct.Struct("<8sIi32sII4qIiI28x"),  # 128 bytes
        "decoder": lambda u: {
            "magic": u[0],
            "version": u[1],
            "layer_id": u[2],
            "op_name_bytes": u[3],
            "dtype": u[4],
            "rank": u[5],
            "shape": [u[6], u[7], u[8], u[9]],
            "elem_count": u[10],
            "token_id": u[11],
        },
    },
    {
        "name": "legacy_no_rank_124",
        "spec": struct.Struct("<8sIi32sI4qIQi24x"),  # 124 bytes
        "decoder": lambda u: {
            "magic": u[0],
            "version": u[1],
            "layer_id": u[2],
            "op_name_bytes": u[3],
            "dtype": u[4],
            "rank": 0,
            "shape": [u[5], u[6], u[7], u[8]],
            "elem_count": u[9],
            "token_id": int(u[10]),
        },
    },
]
MAX_HEADER_SIZE = max(f["spec"].size for f in HEADER_FORMATS)
MIN_HEADER_SIZE = min(f["spec"].size for f in HEADER_FORMATS)


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


def _is_plausible_header(h: Dict) -> bool:
    if h["magic"] != CKDUMP_MAGIC:
        return False
    if h["version"] <= 0 or h["version"] > 1000:
        return False
    if h["dtype"] < 0 or h["dtype"] > 16:
        return False
    if h["elem_count"] < 0 or h["elem_count"] > (1 << 31):
        return False
    rank = int(h.get("rank", 0))
    if rank < 0 or rank > 4:
        return False
    shape = h.get("shape", [0, 0, 0, 0])
    if len(shape) != 4:
        return False
    if any(int(s) < 0 for s in shape):
        return False
    return True


def _decode_header(header_bytes: bytes) -> Optional[Tuple[Dict, int]]:
    # Prefer larger formats first so we can parse newer writers when possible.
    for fmt in sorted(HEADER_FORMATS, key=lambda x: x["spec"].size, reverse=True):
        size = fmt["spec"].size
        if len(header_bytes) < size:
            continue
        try:
            unpacked = fmt["spec"].unpack(header_bytes[:size])
        except struct.error:
            continue
        header = fmt["decoder"](unpacked)
        if _is_plausible_header(header):
            return header, size
    return None


def _normalize_layer_and_op(layer_id: int, op_name: str) -> Tuple[int, str]:
    op = op_name.strip()
    if " (" in op:
        op = op.split(" (", 1)[0]

    # Many llama dumps encode layer in the op name (e.g. Qcur-0).
    m = re.match(r"^(.*?)-(\d+)$", op)
    if m:
        op = m.group(1)
        parsed_layer = int(m.group(2))
        if layer_id < 0:
            layer_id = parsed_layer

    aliases = {
        "inp_embd": "token_embedding",
        "token_embd": "token_embedding",
        "attn_norm": "attn_norm",
        "qcur": "q_proj",
        "kcur": "k_proj",
        "vcur": "v_proj",
        "qcur_normed": "qcur_normed",
        "kcur_normed": "kcur_normed",
        "attn_out": "attn_output",
        "sa_out": "attn_output",
        "ffn_gate": "gate_proj",
        "ffn_up": "up_proj",
        "ffn_down": "down_proj",
        "final_norm": "final_norm",
        "ln_final": "final_norm",
        "result_norm": "final_norm",
    }
    return layer_id, aliases.get(op.lower(), op)


def _dtype_to_elem_size(dtype: int) -> int:
    if dtype == 0:
        return 4
    if dtype in (1, 2):
        return 2
    if dtype == 3:
        return 1
    return 4


def _choose_elem_size(blob: bytes, rec_start: int, header_size: int, elem_count: int, dtype: int) -> int:
    preferred = _dtype_to_elem_size(dtype)
    candidates = [preferred, 4, 2, 1]
    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)

    for elem_size in unique_candidates:
        next_off = rec_start + header_size + elem_count * elem_size
        if next_off > len(blob):
            continue
        if next_off == len(blob):
            return elem_size
        if blob[next_off:next_off + len(CKDUMP_MAGIC)] == CKDUMP_MAGIC:
            return elem_size
        if next_off + MIN_HEADER_SIZE <= len(blob):
            if _decode_header(blob[next_off:next_off + MAX_HEADER_SIZE]) is not None:
                return elem_size

    # Fallback: any size that stays in-bounds.
    for elem_size in unique_candidates:
        if rec_start + header_size + elem_count * elem_size <= len(blob):
            return elem_size
    return preferred


def read_dump_file(path: Path) -> List[ParityDump]:
    """Read all tensor dumps from a binary file."""
    dumps = []

    if not path.exists():
        print(f"[WARNING] Dump file not found: {path}")
        return dumps

    blob = path.read_bytes()
    offset = 0
    warned_dtype_mismatch = False

    while offset + MIN_HEADER_SIZE <= len(blob):
        rec_start = offset
        decoded = _decode_header(blob[offset:offset + MAX_HEADER_SIZE])
        if decoded is None:
            nxt = blob.find(CKDUMP_MAGIC, offset + 1)
            if nxt < 0:
                print(f"[WARNING] Could not decode header at offset {rec_start}")
                break
            print(f"[WARNING] Resyncing dump parse: bad header at {rec_start}, next magic at {nxt}")
            offset = nxt
            continue

        h, header_size = decoded
        op_name = h["op_name_bytes"].split(b"\x00")[0].decode("utf-8", errors="ignore")

        rank = int(h["rank"])
        elem_count = int(h["elem_count"])
        token_id = int(h["token_id"])
        dtype = int(h["dtype"])
        shape = [int(h["shape"][0]), int(h["shape"][1]), int(h["shape"][2]), int(h["shape"][3])][:rank]
        if not any(shape):
            shape = [elem_count]

        elem_size = _choose_elem_size(blob, rec_start, header_size, elem_count, dtype)
        expected_elem_size = _dtype_to_elem_size(dtype)
        if elem_size != expected_elem_size and not warned_dtype_mismatch:
            print(
                f"[WARNING] CKDMP dtype/data-size mismatch detected (offset={rec_start}, op={op_name}); "
                "using size-inferred parse to keep alignment"
            )
            warned_dtype_mismatch = True

        data_start = rec_start + header_size
        data_end = data_start + elem_count * elem_size
        if data_end > len(blob):
            print(f"[WARNING] Unexpected EOF reading {op_name}")
            break
        data_bytes = blob[data_start:data_end]

        if elem_size == 4:
            np_dtype = np.float32
            dtype_name = "fp32"
        elif elem_size == 2:
            np_dtype = np.float16
            dtype_name = "fp16" if dtype == 1 else "bf16"
        else:
            np_dtype = np.int8
            dtype_name = "int8"

        data = np.frombuffer(data_bytes, dtype=np_dtype).astype(np.float32)
        try:
            data = data.reshape(shape)
        except ValueError:
            data = data.reshape(-1)

        norm_layer, norm_op = _normalize_layer_and_op(int(h["layer_id"]), op_name)
        dumps.append(ParityDump(norm_layer, norm_op, data, token_id, dtype_name))
        offset = data_end

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

        if not ck_dump:
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

    # Summary counters are needed for both verbose and quiet exit logic.
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    warnings = sum(1 for r in results if r["status"] == "WARN")
    total = len(results)

    if verbose:
        print("-" * 130)

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

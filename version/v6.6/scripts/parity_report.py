#!/usr/bin/env python3
"""
parity_report.py - Full parity report between llama.cpp and CKE

Generates JSON report for parity_viewer.html or pretty terminal output.

Usage:
    python version/v6.6/scripts/parity_report.py --ck-dump ck_parity_dumps/dump.bin --output report.json
    python version/v6.6/scripts/parity_report.py --ck-dump ck_parity_dumps/dump.bin --view  # Open viewer
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# Binary Dump Reader
# =============================================================================

CKDUMP_MAGIC = b"CKDMP\x00\x00"
HEADER_SPEC = struct.Struct("<8sIi32sI4qIQi24x")
HEADER_SIZE = 128


class ParityDump:
    def __init__(self, layer_id: int, op_name: str, data: np.ndarray,
                 token_id: int, dtype: str):
        self.layer_id = layer_id
        self.op_name = op_name
        self.data = data
        self.token_id = token_id
        self.dtype = dtype


def read_dump_file(path: Path) -> List[ParityDump]:
    """Read all tensor dumps from a binary file."""
    dumps = []
    if not path.exists():
        return dumps

    with open(path, "rb") as f:
        while True:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                break

            unpacked = HEADER_SPEC.unpack(header_bytes)
            magic, version, layer_id, op_name_bytes, dtype, rank, \
                shape0, shape1, shape2, shape3, elem_count, token_id = unpacked

            if magic != CKDUMP_MAGIC:
                continue

            op_name = op_name_bytes.split(b"\x00")[0].decode("utf-8", errors="ignore")
            shape = [shape0, shape1, shape2, shape3][:rank]
            if not any(shape):
                shape = [elem_count]

            dtype_map = {0: np.float32, 1: np.float16, 2: np.float16}
            np_dtype = dtype_map.get(dtype, np.float32)
            elem_size = 4 if dtype == 0 else 2
            data_size = elem_count * elem_size

            data_bytes = f.read(data_size)
            if len(data_bytes) < data_size:
                break

            data = np.frombuffer(data_bytes, dtype=np_dtype).astype(np.float32)
            data = data.reshape(shape)

            dtype_name = ["fp32", "fp16", "bf16"][dtype] if dtype < 3 else "unknown"
            dumps.append(ParityDump(layer_id, op_name, data, token_id, dtype_name))

    return dumps


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_tensors(ref_data: np.ndarray, test_data: np.ndarray,
                   op_name: str, atol: float = 1e-4, rtol: float = 1e-3) -> Dict:
    """Compare two tensors and return statistics."""
    ref_flat = ref_data.flatten()
    test_flat = test_data.flatten()

    min_len = min(len(ref_flat), len(test_flat))
    ref_flat = ref_flat[:min_len]
    test_flat = test_flat[:min_len]

    abs_diff = np.abs(ref_flat - test_flat)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    safe_ref = np.abs(ref_data.flatten()) + atol
    rel_err = abs_diff / safe_ref[:min_len]
    max_rel_err = np.max(rel_err)

    has_nan = np.any(np.isnan(test_flat)) or np.any(np.isnan(ref_flat))
    has_inf = np.any(np.isinf(test_flat)) or np.any(np.isinf(ref_flat))

    diverge_idx = None
    if max_abs_diff > atol and not (has_nan or has_inf):
        diverge_indices = np.where(abs_diff > atol)[0]
        if len(diverge_indices) > 0:
            diverge_idx = int(diverge_indices[0])

    status = "ERROR" if (has_nan or has_inf) else ("PASS" if max_abs_diff <= atol else "FAIL")

    return {
        "op": op_name,
        "status": status,
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "max_rel_err": float(max_rel_err),
        "diverge_idx": diverge_idx,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "ref_range": [float(np.min(ref_data)), float(np.max(ref_data))],
        "cke_range": [float(np.min(test_data)), float(np.max(test_data))],
        "ref_shape": list(ref_data.shape),
        "cke_shape": list(test_data.shape),
        "notes": f"NaN={has_nan}, Inf={has_inf}" if (has_nan or has_inf) else "",
    }


# =============================================================================
# Report Generator
# =============================================================================

def generate_report_data(
    ck_dumps: List[ParityDump],
    ref_dumps: Optional[List[ParityDump]] = None,
    manifest: Optional[Dict] = None,
    model_info: Optional[Dict] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Dict:
    """Generate parity report data structure."""

    # Index dumps
    ck_index: Dict[Tuple[int, str], ParityDump] = {}
    for d in ck_dumps:
        ck_index[(d.layer_id, d.op_name)] = d

    ref_index: Dict[Tuple[int, str], ParityDump] = {}
    if ref_dumps:
        for d in ref_dumps:
            ref_index[(d.layer_id, d.op_name)] = d

    # All keys
    all_keys = sorted(set(ck_index.keys()) | set(ref_index.keys()))

    operations = []
    summary = {"total": len(all_keys), "passed": 0, "failed": 0, "errors": 0, "warnings": 0}

    for layer_id, op_name in all_keys:
        ck_dump = ck_index.get((layer_id, op_name))
        ref_dump = ref_index.get((layer_id, op_name))

        if not ck_dump:
            operations.append({
                "layer": layer_id,
                "op": op_name,
                "status": "missing",
                "notes": "Missing CKE dump"
            })
            summary["warnings"] += 1
        elif not ref_dump:
            # No reference - just check CKE for NaN/Inf
            has_nan = np.any(np.isnan(ck_dump.data))
            has_inf = np.any(np.isinf(ck_dump.data))
            status = "error" if (has_nan or has_inf) else "pass"
            operations.append({
                "layer": layer_id,
                "op": op_name,
                "status": status,
                "cke_range": [float(np.min(ck_dump.data)), float(np.max(ck_dump.data))],
                "cke_shape": list(ck_dump.data.shape),
                "notes": f"NaN={has_nan}, Inf={has_inf}"
            })
            if status == "error":
                summary["errors"] += 1
            else:
                summary["passed"] += 1
        else:
            comp = compare_tensors(ref_dump.data, ck_dump.data, op_name, atol, rtol)
            operations.append({
                "layer": layer_id,
                "op": op_name,
                **comp
            })
            if comp["status"] == "PASS":
                summary["passed"] += 1
            elif comp["status"] == "FAIL":
                summary["failed"] += 1
            else:
                summary["errors"] += 1

    # Get performance data if available
    performance = {}
    # TODO: Add timing info from dumps or separate source

    return {
        "format_version": "v1.0",
        "timestamp": str(np.datetime64('now')),
        "tolerance": {"atol": atol, "rtol": rtol},
        "summary": summary,
        "model_info": model_info or {},
        "performance": performance,
        "operations": operations,
    }


# =============================================================================
# Terminal Report (Pretty Print)
# =============================================================================

def print_terminal_report(data: Dict):
    """Print a formatted terminal report."""

    print("=" * 120)
    print(f"PARITY REPORT - {data['model_info'].get('model', 'Unknown')}")
    print(f"Generated: {data['timestamp']}")
    print("=" * 120)

    # Summary
    s = data["summary"]
    print(f"\n{'Operations:':<20} {s['total']}")
    print(f"  {'✓ PASS':<10} {s['passed']}")
    print(f"  {'✗ FAIL':<10} {s['failed']}")
    print(f"  {'⚠ ERROR':<10} {s['errors']}")
    print(f"  {'⚠ WARN':<10} {s['warnings']}")

    # Model info
    if data.get("model_info"):
        print(f"\n{'Model Info':<20}")
        for key, val in data["model_info"].items():
            print(f"  {key:<20} {val}")

    print(f"\n{'=' * 120}")
    print(f"{'Layer':<6} {'Op':<22} {'Status':<8} {'Max Abs':<12} {'Mean Abs':<12} {'Max Rel':<10} {'Ref Range':<22} {'CKE Range':<22}")
    print("-" * 120)

    for op in data["operations"]:
        status = op["status"].upper()

        # Color codes
        if status == "PASS":
            color = "\033[92m"
        elif status == "FAIL":
            color = "\033[91m"
        elif status == "ERROR":
            color = "\033[93m"
        elif status == "MISSING":
            color = "\033[90m"
        else:
            color = "\033[97m"
        reset = "\033[0m"

        ref_str = f"[{op.get('ref_range', ['N/A', 'N/A'])[0]:.2e}, {op.get('ref_range', ['N/A', 'N/A'])[1]:.2e}]" if 'ref_range' in op else "N/A"
        cke_str = f"[{op['cke_range'][0]:.2e}, {op['cke_range'][1]:.2e}]" if 'cke_range' in op else "N/A"

        print(f"{op['layer']:<6} {op['op']:<22} {color}{status:<8}{reset} "
              f"{op.get('max_abs_diff', 'N/A'):<12.2e} "
              f"{op.get('mean_abs_diff', 'N/A'):<12.2e} "
              f"{op.get('max_rel_err', 0):<10.2%} "
              f"{ref_str:<22} {cke_str:<22}")

    print("=" * 120)

    # Show first errors/failures
    errors = [op for op in data["operations"] if op["status"] == "error"]
    failures = [op for op in data["operations"] if op["status"] == "fail"]

    if errors:
        print("\n⚠ ERRORS (NaN/Inf detected):")
        for op in errors[:5]:
            print(f"  Layer {op['layer']}, Op {op['op']}: {op['notes']}")

    if failures:
        print("\n⚠ FAILURES (divergence from reference):")
        for op in failures[:5]:
            print(f"  Layer {op['layer']}, Op {op['op']}: "
                  f"max_diff={op['max_abs_diff']:.2e}, first diverge @ {op.get('diverge_idx', 'N/A')}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate parity report between llama.cpp and CKE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ck-dump", type=Path, help="Path to CKE dump file")
    parser.add_argument("--ref-dump", type=Path, help="Path to reference (llama.cpp) dump file")
    parser.add_argument("--manifest", type=Path, help="Weights manifest JSON")
    parser.add_argument("--model-info", type=Path, help="Model config JSON")
    parser.add_argument("--output", "-o", type=Path, default=Path("parity_report.json"),
                       help="Output JSON file for viewer")
    parser.add_argument("--view", action="store_true", help="Open in parity_viewer.html")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--ck-model", type=str, default="Qwen--Qwen3-0.6B-GGUF",
                       help="Model name (for finding in cache)")

    args = parser.parse_args()

    # Find model directory
    model_dir = None
    if args.ck_dump:
        model_dir = args.ck_dump.parent.parent
    else:
        cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
        model_dir = cache_dir / args.ck_model

    # Load CKE dumps
    ck_dump_path = args.ck_dump or (model_dir / "ck_parity_dumps" / "dump.bin")
    ck_dumps = read_dump_file(ck_dump_path)

    if not ck_dumps:
        print(f"[ERROR] No CKE dumps found at: {ck_dump_path}")
        print("[HINT] Run model with CK_PARITY_DUMP defined, or use ck-cli --parity-dump")
        return 1

    print(f"[OK] Loaded {len(ck_dumps)} tensors from CKE dump")

    # Load reference dumps if provided
    ref_dumps = None
    if args.ref_dump:
        ref_dumps = read_dump_file(args.ref_dump)
        print(f"[OK] Loaded {len(ref_dumps)} tensors from reference dump")

    # Load model info
    model_info = None
    config_path = model_dir / "config.json" if model_dir else None
    if config_path and config_path.exists():
        with open(config_path) as f:
            model_info = json.load(f)
    elif args.model_info:
        with open(args.model_info) as f:
            model_info = json.load(f)

    # Generate report data
    report_data = generate_report_data(
        ck_dumps=ck_dumps,
        ref_dumps=ref_dumps,
        model_info=model_info,
        atol=args.atol,
        rtol=args.rtol
    )

    # Write JSON
    output_path = args.output
    output_path.write_text(json.dumps(report_data, indent=2))
    print(f"[OK] Wrote report to: {output_path}")

    # Print terminal report
    print_terminal_report(report_data)

    # View in browser
    if args.view:
        viewer_path = Path(__file__).parent.parent / "tools" / "parity_viewer.html"
        if viewer_path.exists():
            # Open viewer with report loaded
            report_url = f"file://{viewer_path.absolute()}?report={output_path.absolute()}"
            print(f"[INFO] Opening viewer: {report_url}")
            webbrowser.open(report_url)
        else:
            print(f"[ERROR] Viewer not found at: {viewer_path}")

    return 0 if report_data["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

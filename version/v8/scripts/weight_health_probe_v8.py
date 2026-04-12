#!/usr/bin/env python3
"""
weight_health_probe_v8.py

Offline probe that reads checkpoint manifests + BUMP weight files from a
training run directory and emits a compact ``weight_health_latest.json``
summary.  The visualizer's Weight Health tab consumes this JSON.

Three diagnostic tiers:

1. **Checkpoint delta health** — compare init (step 0 / earliest) vs latest
   checkpoint per tensor: norm delta, max absolute delta, fraction of
   elements unchanged (within epsilon), NaN/Inf/zero counts.

2. **Gradient reachability** — if analysis-checkpoint JSONs contain per-
   parameter gradient norms, summarise latest grad norm, mean grad norm
   across checkpoints, and zero-heavy ratio.

3. **Stale-parameter flags** — conservative heuristics that flag tensors
   with "no meaningful update", "persistently near-zero grad", or
   "unchanged row/column".  We deliberately avoid the term "dead neuron"
   because in transformer architectures (SwiGLU, attention) that label
   is misleading.

Usage
-----
    python3 weight_health_probe_v8.py --run <run_dir>
    python3 weight_health_probe_v8.py --run <run_dir> --init-step 0 --latest-step 50000

Output: ``<run_dir>/weight_health_latest.json``
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Epsilon for "unchanged element" detection — two floats are considered
# identical if |a - b| < UNCHANGED_EPS.
UNCHANGED_EPS = 1e-9

# Gradient norm below this is classified as "near-zero".
GRAD_NEAR_ZERO_THRESHOLD = 1e-7

# Fraction of near-zero grad checkpoints above which we flag "persistently
# near-zero grad".
GRAD_ZERO_HEAVY_RATIO = 0.8

# If norm-delta / init-norm < this, flag "no meaningful update".
RELATIVE_MOVEMENT_THRESHOLD = 1e-5

# If this fraction of elements in a row or column is unchanged, flag it.
UNCHANGED_ROW_COL_FRACTION = 0.99

# BUMP file header constants (must match ckernel_model_load.c).
BUMP_MAGIC = b"CKBUMP"
BUMP_HEADER_SIZE = 128


# ---------------------------------------------------------------------------
# BUMP reader
# ---------------------------------------------------------------------------

def _read_bump_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Parse a weights_manifest*.json and return the tensor list."""
    with open(manifest_path, "r") as f:
        payload = json.load(f)
    # v7 runtime checkpoint format uses "entries".
    entries = payload.get("entries")
    if isinstance(entries, list) and entries:
        return entries
    tensors = payload.get("tensors")
    if isinstance(tensors, list):
        return tensors
    # Older manifests store tensor info at top level under "weights".
    weights = payload.get("weights")
    if isinstance(weights, list):
        return weights
    return []


def _dtype_byte_size(dtype_str: str) -> int:
    """Return the byte size of a single element for common dtype strings."""
    table = {
        "float32": 4, "f32": 4,
        "float16": 2, "f16": 2,
        "bfloat16": 2, "bf16": 2,
        "int8": 1, "q8_0": 1,
        "int32": 4, "i32": 4,
    }
    return table.get(dtype_str.lower().strip(), 4)


def _read_tensor_from_bump(
    bump_path: Path,
    offset: int,
    num_elements: int,
    dtype_str: str,
) -> list[float]:
    """Read a tensor from a BUMP file and return as a flat list of floats."""
    bpe = _dtype_byte_size(dtype_str)
    byte_count = num_elements * bpe

    with open(bump_path, "rb") as f:
        f.seek(offset)
        raw = f.read(byte_count)

    if len(raw) < byte_count:
        return []

    dt = dtype_str.lower().strip()
    if dt in ("float32", "f32"):
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}f", raw[: count * 4]))
    if dt in ("float16", "f16"):
        import array
        halfs = array.array("e")
        halfs.frombytes(raw[: (len(raw) // 2) * 2])
        return [float(v) for v in halfs]
    if dt in ("bfloat16", "bf16"):
        values: list[float] = []
        for i in range(0, len(raw) - 1, 2):
            bf16_bits = raw[i] | (raw[i + 1] << 8)
            f32_bits = bf16_bits << 16
            values.append(struct.unpack("<f", struct.pack("<I", f32_bits))[0])
        return values
    # Fallback: treat as float32.
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw[: count * 4]))


# ---------------------------------------------------------------------------
# Per-tensor statistics
# ---------------------------------------------------------------------------

def _tensor_stats(values: list[float]) -> dict[str, Any]:
    """Compute summary statistics for a flat list of floats."""
    n = len(values)
    if n == 0:
        return {"count": 0}

    nan_count = sum(1 for v in values if math.isnan(v))
    inf_count = sum(1 for v in values if math.isinf(v))
    zero_count = sum(1 for v in values if v == 0.0)

    finite = [v for v in values if math.isfinite(v)]
    if finite:
        mean = sum(finite) / len(finite)
        sq_sum = sum(v * v for v in finite)
        frob_norm = math.sqrt(sq_sum)
        abs_max = max(abs(v) for v in finite)
        abs_min = min(abs(v) for v in finite)
    else:
        mean = 0.0
        frob_norm = 0.0
        abs_max = 0.0
        abs_min = 0.0

    return {
        "count": n,
        "nan": nan_count,
        "inf": inf_count,
        "zero": zero_count,
        "zero_frac": zero_count / n if n > 0 else 0.0,
        "mean": mean,
        "frobenius_norm": frob_norm,
        "abs_max": abs_max,
        "abs_min": abs_min,
    }


def _delta_stats(
    init_values: list[float],
    latest_values: list[float],
) -> dict[str, Any]:
    """Compute per-element delta statistics between two flat tensors."""
    n = min(len(init_values), len(latest_values))
    if n == 0:
        return {"count": 0}

    deltas = [latest_values[i] - init_values[i] for i in range(n)]
    abs_deltas = [abs(d) for d in deltas if math.isfinite(d)]
    unchanged = sum(1 for d in abs_deltas if d < UNCHANGED_EPS)

    if abs_deltas:
        norm_delta = math.sqrt(sum(d * d for d in abs_deltas))
        max_delta = max(abs_deltas)
        mean_delta = sum(abs_deltas) / len(abs_deltas)
    else:
        norm_delta = 0.0
        max_delta = 0.0
        mean_delta = 0.0

    return {
        "count": n,
        "norm_delta": norm_delta,
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "unchanged_count": unchanged,
        "unchanged_frac": unchanged / n if n > 0 else 0.0,
    }


def _check_unchanged_rows_cols(
    init_values: list[float],
    latest_values: list[float],
    shape: list[int],
) -> dict[str, Any]:
    """Check for rows/columns that are entirely unchanged (2D tensors only)."""
    if len(shape) != 2:
        return {}
    rows, cols = shape
    n = rows * cols
    if n == 0 or len(init_values) < n or len(latest_values) < n:
        return {}

    unchanged_rows = 0
    for r in range(rows):
        start = r * cols
        row_unchanged = sum(
            1 for c in range(cols)
            if abs(latest_values[start + c] - init_values[start + c]) < UNCHANGED_EPS
        )
        if row_unchanged >= cols * UNCHANGED_ROW_COL_FRACTION:
            unchanged_rows += 1

    unchanged_cols = 0
    for c in range(cols):
        col_unchanged = sum(
            1 for r in range(rows)
            if abs(latest_values[r * cols + c] - init_values[r * cols + c]) < UNCHANGED_EPS
        )
        if col_unchanged >= rows * UNCHANGED_ROW_COL_FRACTION:
            unchanged_cols += 1

    result: dict[str, Any] = {}
    if unchanged_rows > 0:
        result["unchanged_rows"] = unchanged_rows
        result["unchanged_rows_frac"] = unchanged_rows / rows
    if unchanged_cols > 0:
        result["unchanged_cols"] = unchanged_cols
        result["unchanged_cols_frac"] = unchanged_cols / cols
    return result


# ---------------------------------------------------------------------------
# Gradient reachability from analysis checkpoints
# ---------------------------------------------------------------------------

def _collect_grad_reachability(
    analysis_roots: list[Path],
) -> dict[str, dict[str, Any]]:
    """Collect per-parameter gradient statistics from analysis checkpoints."""
    step_to_path: dict[int, Path] = {}
    for root in analysis_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("analysis_checkpoint_step_*.json")):
            m = re.search(r"_step_(\d+)\.json$", candidate.name)
            if not m:
                continue
            step = int(m.group(1))
            if step not in step_to_path:
                step_to_path[step] = candidate

    if not step_to_path:
        return {}

    param_norms: dict[str, list[float]] = {}

    for step in sorted(step_to_path.keys()):
        try:
            with open(step_to_path[step], "r") as f:
                payload = json.load(f)
        except Exception:
            continue
        gradients = payload.get("gradients")
        if not isinstance(gradients, dict):
            continue
        for param, info in gradients.items():
            norm = info.get("norm") if isinstance(info, dict) else None
            if isinstance(norm, (int, float)) and math.isfinite(norm):
                param_norms.setdefault(param, []).append(float(norm))

    result: dict[str, dict[str, Any]] = {}
    for param, norms in param_norms.items():
        n = len(norms)
        near_zero = sum(1 for v in norms if v < GRAD_NEAR_ZERO_THRESHOLD)
        result[param] = {
            "checkpoints_seen": n,
            "latest_norm": norms[-1],
            "mean_norm": sum(norms) / n if n > 0 else 0.0,
            "max_norm": max(norms),
            "min_norm": min(norms),
            "near_zero_count": near_zero,
            "near_zero_ratio": near_zero / n if n > 0 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Stale-parameter heuristics
# ---------------------------------------------------------------------------

def _classify_flags(
    delta: dict[str, Any],
    init_stats: dict[str, Any],
    latest_stats: dict[str, Any],
    grad_info: dict[str, Any] | None,
    row_col: dict[str, Any],
) -> list[str]:
    """Return list of human-readable flag strings for a tensor."""
    flags: list[str] = []

    init_norm = init_stats.get("frobenius_norm", 0.0)
    norm_delta = delta.get("norm_delta", 0.0)
    unchanged_frac = delta.get("unchanged_frac", 0.0)

    # Flag 1: no meaningful update.
    if init_norm > 0 and norm_delta / init_norm < RELATIVE_MOVEMENT_THRESHOLD:
        flags.append("no meaningful update")
    elif delta.get("count", 0) > 0 and unchanged_frac > 0.99:
        flags.append("no meaningful update")

    # Flag 2: NaN or Inf detected.
    if latest_stats.get("nan", 0) > 0:
        flags.append(f"NaN detected ({latest_stats['nan']})")
    if latest_stats.get("inf", 0) > 0:
        flags.append(f"Inf detected ({latest_stats['inf']})")

    # Flag 3: persistently near-zero grad.
    if grad_info is not None:
        if grad_info.get("near_zero_ratio", 0.0) >= GRAD_ZERO_HEAVY_RATIO:
            flags.append("persistently near-zero grad")

    # Flag 4: unchanged row/column.
    if row_col.get("unchanged_rows_frac", 0.0) > 0.1:
        n = row_col.get("unchanged_rows", 0)
        flags.append(f"unchanged rows ({n})")
    if row_col.get("unchanged_cols_frac", 0.0) > 0.1:
        n = row_col.get("unchanged_cols", 0)
        flags.append(f"unchanged columns ({n})")

    return flags


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def _find_checkpoint_pairs(
    run_dir: Path,
    init_step: int | None,
    latest_step: int | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Find init and latest checkpoint manifest+bump pairs."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        # Try run_dir itself (some layouts keep weights at root).
        checkpoints_dir = run_dir

    manifests: dict[int, Path] = {}
    for p in sorted(checkpoints_dir.glob("weights_step_*_manifest.json")):
        m = re.search(r"weights_step_(\d+)_manifest\.json$", p.name)
        if m:
            manifests[int(m.group(1))] = p

    # Also consider init manifest at run root.
    for candidate in [
        run_dir / "weights_manifest.json",
        run_dir / "weights_step_0_manifest.json",
    ]:
        if candidate.exists() and 0 not in manifests:
            manifests[0] = candidate

    if not manifests:
        return None, None, None, None

    steps = sorted(manifests.keys())
    init_s = init_step if init_step is not None else steps[0]
    latest_s = latest_step if latest_step is not None else steps[-1]

    if init_s == latest_s and len(steps) > 1:
        latest_s = steps[-1]
        if init_s == latest_s:
            init_s = steps[0]

    init_manifest = manifests.get(init_s)
    latest_manifest = manifests.get(latest_s)

    def _bump_for(manifest_path: Path, step: int) -> Path | None:
        parent = manifest_path.parent
        # Try both zero-padded and unpadded step names.
        for name in [f"weights_step_{step:08d}.bump", f"weights_step_{step}.bump"]:
            bump = parent / name
            if bump.exists():
                return bump
        # Check run-root level weights.bump as init fallback.
        if step == 0:
            # Prefer weights_init.bump (preserved init) over weights.bump
            # (which may be overwritten with the final checkpoint).
            for name in ["weights_init.bump", "weights.bump"]:
                bump = run_dir / name
                if bump.exists():
                    return bump
            # Also check .ck_build/ for the original seed.
            bump = run_dir / ".ck_build" / "weights.bump"
            if bump.exists():
                return bump
        return None

    init_bump = _bump_for(init_manifest, init_s) if init_manifest else None
    latest_bump = _bump_for(latest_manifest, latest_s) if latest_manifest else None

    return init_manifest, init_bump, latest_manifest, latest_bump


def _load_tensor_map(
    manifest_path: Path,
    bump_path: Path | None,
    max_elements_per_tensor: int = 500_000,
) -> dict[str, dict[str, Any]]:
    """Load tensors from manifest + optional BUMP file."""
    tensors = _read_bump_manifest(manifest_path)
    result: dict[str, dict[str, Any]] = {}

    for t in tensors:
        name = t.get("name") or t.get("tensor_name") or ""
        if not name:
            continue

        shape = t.get("shape", t.get("dims", []))
        if isinstance(shape, list):
            shape = [int(s) for s in shape]
        else:
            shape = []

        num_elements = 1
        for s in shape:
            num_elements *= s

        dtype_str = str(t.get("dtype", t.get("type", "float32")))
        offset = t.get("offset", t.get("data_offset"))

        entry: dict[str, Any] = {
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "num_elements": num_elements,
        }

        # Read actual values if BUMP available and tensor is small enough.
        if (
            bump_path is not None
            and bump_path.exists()
            and offset is not None
            and num_elements <= max_elements_per_tensor
            and num_elements > 0
        ):
            values = _read_tensor_from_bump(bump_path, int(offset), num_elements, dtype_str)
            if values:
                entry["values"] = values

        result[name] = entry

    return result


def run_probe(
    run_dir: Path,
    init_step: int | None = None,
    latest_step: int | None = None,
    max_elements: int = 500_000,
) -> dict[str, Any]:
    """Run the weight health probe and return the summary payload."""
    init_manifest, init_bump, latest_manifest, latest_bump = _find_checkpoint_pairs(
        run_dir, init_step, latest_step,
    )

    report: dict[str, Any] = {
        "schema": "ck.weight_health.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "init_checkpoint": str(init_manifest) if init_manifest else None,
        "latest_checkpoint": str(latest_manifest) if latest_manifest else None,
        "tensors": {},
        "summary": {},
    }

    if init_manifest is None or latest_manifest is None:
        report["summary"]["error"] = "Could not find init and/or latest checkpoint manifests."
        return report

    print(f"Loading init checkpoint: {init_manifest}")
    init_tensors = _load_tensor_map(init_manifest, init_bump, max_elements)
    print(f"Loading latest checkpoint: {latest_manifest}")
    latest_tensors = _load_tensor_map(latest_manifest, latest_bump, max_elements)

    # Gradient reachability from analysis checkpoints.
    analysis_roots = [run_dir, run_dir / "checkpoints"]
    grad_reach = _collect_grad_reachability(analysis_roots)

    all_names = sorted(set(list(init_tensors.keys()) + list(latest_tensors.keys())))
    tensor_reports: dict[str, dict[str, Any]] = {}

    total_params = 0
    flagged_count = 0
    flag_summary: dict[str, int] = {}

    for name in all_names:
        init_t = init_tensors.get(name, {})
        latest_t = latest_tensors.get(name, {})
        shape = latest_t.get("shape", init_t.get("shape", []))
        num_elements = latest_t.get("num_elements", init_t.get("num_elements", 0))
        total_params += num_elements

        entry: dict[str, Any] = {
            "shape": shape,
            "dtype": latest_t.get("dtype", init_t.get("dtype", "unknown")),
            "num_elements": num_elements,
        }

        # Compute stats if values available.
        init_values = init_t.get("values", [])
        latest_values = latest_t.get("values", [])

        if init_values:
            entry["init_stats"] = _tensor_stats(init_values)
        if latest_values:
            entry["latest_stats"] = _tensor_stats(latest_values)
        if init_values and latest_values:
            entry["delta"] = _delta_stats(init_values, latest_values)
            entry["row_col"] = _check_unchanged_rows_cols(init_values, latest_values, shape)
        else:
            entry["delta"] = {}
            entry["row_col"] = {}

        # Grad reachability.
        grad_info = grad_reach.get(name)
        if grad_info is not None:
            entry["grad_reachability"] = grad_info

        # Flags.
        flags = _classify_flags(
            entry.get("delta", {}),
            entry.get("init_stats", {}),
            entry.get("latest_stats", {}),
            grad_info,
            entry.get("row_col", {}),
        )
        if flags:
            entry["flags"] = flags
            flagged_count += 1
            for f in flags:
                key = f.split("(")[0].strip()
                flag_summary[key] = flag_summary.get(key, 0) + 1

        # Drop raw values from report — keep it compact.
        entry.pop("values", None)
        tensor_reports[name] = entry

    report["tensors"] = tensor_reports
    report["summary"] = {
        "total_tensors": len(all_names),
        "total_parameters": total_params,
        "tensors_with_values": sum(
            1 for t in tensor_reports.values()
            if t.get("delta", {}).get("count", 0) > 0
        ),
        "flagged_tensors": flagged_count,
        "flag_counts": flag_summary,
        "grad_reachability_tensors": len(grad_reach),
    }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight health probe for CK training runs.",
    )
    parser.add_argument("--run", type=Path, required=True, help="Path to run directory.")
    parser.add_argument("--init-step", type=int, default=None, help="Override init checkpoint step.")
    parser.add_argument("--latest-step", type=int, default=None, help="Override latest checkpoint step.")
    parser.add_argument(
        "--max-elements", type=int, default=500_000,
        help="Max elements per tensor to read from BUMP (default: 500k).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: <run>/weight_health_latest.json).")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    if not run_dir.is_dir():
        print(f"Error: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    report = run_probe(run_dir, args.init_step, args.latest_step, args.max_elements)

    output_path = args.output or (run_dir / "weight_health_latest.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    summary = report.get("summary", {})
    print(f"\nWeight Health Report: {output_path}")
    print(f"  Tensors: {summary.get('total_tensors', 0)}")
    print(f"  Parameters: {summary.get('total_parameters', 0):,}")
    print(f"  Analysed (with values): {summary.get('tensors_with_values', 0)}")
    print(f"  Flagged: {summary.get('flagged_tensors', 0)}")
    flags = summary.get("flag_counts", {})
    for flag_name, count in sorted(flags.items()):
        print(f"    {flag_name}: {count}")


if __name__ == "__main__":
    main()

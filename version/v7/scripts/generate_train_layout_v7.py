#!/usr/bin/env python3
"""
generate_train_layout_v7.py

Build a deterministic contiguous training-memory layout from IR2 tensor metadata.
Codegen remains dumb: it consumes per-tensor sizes/metadata but does not plan offsets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALIGN_BYTES_DEFAULT = 64

DTYPE_BYTES = {
    "fp32": 4,
    "f32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "f16": 2,
    "int32": 4,
    "i32": 4,
    "int32_t": 4,
}

# Fixed region ordering keeps offsets deterministic across runs and hosts.
# Deterministic layout is critical for parity and reproducibility checks.
REGION_ORDER = [
    "params",
    "grads",
    "optimizer_m",
    "optimizer_v",
    "grad_activations",
    "activations",
    "saved",
    "scratch",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _align_up(value: int, align: int) -> int:
    if align <= 1:
        return int(value)
    return int(((int(value) + align - 1) // align) * align)


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return int(n)


def _tensor_numel(meta: Dict[str, Any]) -> Optional[int]:
    n = meta.get("numel")
    if isinstance(n, int) and n > 0:
        return int(n)
    s = _shape_numel(meta.get("shape"))
    if isinstance(s, int) and s > 0:
        return int(s)
    return None


def _dtype_nbytes(dtype: str) -> int:
    d = str(dtype or "fp32").lower()
    return int(DTYPE_BYTES.get(d, 4))


def _classify_region(tid: str, meta: Dict[str, Any]) -> str:
    # IR2 tensor kind + tensor id prefix decide region placement.
    # Keep this mapping simple and explicit so codegen never has to infer memory policy.
    kind = str(meta.get("kind", "") or "").lower()
    persistent = bool(meta.get("persistent", False))

    if tid.startswith("weight.") or kind == "weight":
        return "params"
    if tid.startswith("grad.weight.") or kind == "grad_weight":
        return "grads"
    if tid.startswith("grad.act.") or kind == "grad_activation":
        return "grad_activations"

    if kind in ("saved", "save", "saved_tensor"):
        return "saved"
    if kind in ("aux", "tmp", "scratch"):
        return "scratch"

    if kind == "activation":
        return "saved" if persistent else "activations"

    if tid.startswith("saved.") or tid.startswith("save."):
        return "saved"
    if tid.startswith("aux.") or tid.startswith("tmp."):
        return "scratch"

    return "activations"


def _manifest_index(manifest: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(manifest, dict):
        return out
    for e in manifest.get("entries", []) or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name")
        if isinstance(name, str) and name:
            out[name] = e
    return out


def _build_optimizer_specs(layout_tensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Derive optimizer state tensors directly from grad.weight tensors so
    # optimizer coverage stays aligned with trainable-weight coverage.
    specs: List[Dict[str, Any]] = []
    for t in layout_tensors:
        tid = str(t.get("id", ""))
        if not tid.startswith("grad.weight."):
            continue
        wname = tid[len("grad.weight.") :]
        numel = int(t.get("numel", 0) or 0)
        if numel <= 0:
            continue
        for region, prefix in (("optimizer_m", "optimizer.m."), ("optimizer_v", "optimizer.v.")):
            specs.append(
                {
                    "id": f"{prefix}{wname}",
                    "dtype": "fp32",
                    "numel": numel,
                    "bytes": int(numel * 4),
                    "kind": region,
                    "persistent": True,
                    "requires_grad": False,
                    "region": region,
                    "shape": None,
                    "producer": None,
                    "source_grad": tid,
                }
            )
    return specs


def build_layout(ir2: Dict[str, Any], manifest: Optional[Dict[str, Any]], align_bytes: int, strict: bool) -> Dict[str, Any]:
    tensors = ir2.get("tensors") or {}
    if not isinstance(tensors, dict):
        raise RuntimeError("IR2 missing tensor registry")

    manifest_by_name = _manifest_index(manifest)

    layout_tensors: List[Dict[str, Any]] = []
    missing_numel: List[str] = []

    for tid, meta in sorted(tensors.items(), key=lambda x: str(x[0])):
        if not isinstance(tid, str):
            continue
        m = meta if isinstance(meta, dict) else {}
        numel = _tensor_numel(m)
        if not isinstance(numel, int) or numel <= 0:
            missing_numel.append(tid)
            continue
        dtype = str(m.get("dtype", "fp32") or "fp32")
        nbytes = _dtype_nbytes(dtype)
        region = _classify_region(tid, m)

        row: Dict[str, Any] = {
            "id": tid,
            "dtype": dtype,
            "numel": int(numel),
            "bytes": int(numel * nbytes),
            "kind": str(m.get("kind", "") or "activation"),
            "persistent": bool(m.get("persistent", False)),
            "requires_grad": bool(m.get("requires_grad", False)),
            "region": region,
            "shape": m.get("shape") if isinstance(m.get("shape"), list) else None,
            "producer": m.get("producer") if isinstance(m.get("producer"), dict) else None,
        }

        if tid.startswith("weight."):
            wname = tid[len("weight.") :]
            ent = manifest_by_name.get(wname)
            if isinstance(ent, dict):
                row["bump_offset"] = int(ent.get("offset", 0) or 0)
                row["bump_size"] = int(ent.get("size", 0) or 0)

        layout_tensors.append(row)

    if strict and missing_numel:
        raise RuntimeError(
            "Missing numel metadata for %d tensor(s): %s"
            % (len(missing_numel), ", ".join(missing_numel[:24]))
        )

    optimizer_specs = _build_optimizer_specs(layout_tensors)
    layout_tensors.extend(optimizer_specs)

    by_region: Dict[str, List[Dict[str, Any]]] = {k: [] for k in REGION_ORDER}
    for t in layout_tensors:
        region = str(t.get("region", "scratch"))
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(t)

    offset = 0
    region_rows: List[Dict[str, Any]] = []
    placed: List[Dict[str, Any]] = []

    # Place tensors by region in fixed order, then by tensor id for determinism.
    for region in REGION_ORDER + sorted([r for r in by_region.keys() if r not in REGION_ORDER]):
        entries = by_region.get(region, [])
        if not entries:
            continue
        offset = _align_up(offset, align_bytes)
        region_base = offset
        for t in sorted(entries, key=lambda x: str(x.get("id", ""))):
            offset = _align_up(offset, align_bytes)
            row = dict(t)
            row["offset"] = int(offset)
            row["end"] = int(offset + int(row["bytes"]))
            placed.append(row)
            offset = int(row["end"])
        region_rows.append(
            {
                "name": region,
                "offset": int(region_base),
                "bytes": int(offset - region_base),
                "count": len(entries),
            }
        )

    total_bytes = _align_up(offset, align_bytes)
    checkpoint_policy = str(ir2.get("checkpoint_policy", "none") or "none")
    checkpoint_summary = ir2.get("checkpoint_summary") if isinstance(ir2.get("checkpoint_summary"), dict) else {}
    region_bytes = {str(r.get("name", "")): int(r.get("bytes", 0) or 0) for r in region_rows}

    return {
        "format": "layout-train-v7",
        "generated_at": _utc_now_iso(),
        "align_bytes": int(align_bytes),
        "checkpoint_policy": checkpoint_policy,
        "config": ir2.get("config") if isinstance(ir2.get("config"), dict) else {},
        "summary": {
            "tensor_count": len(placed),
            "region_count": len(region_rows),
            "missing_numel": missing_numel,
            "region_bytes": region_bytes,
            "checkpoint_policy": checkpoint_policy,
            "checkpoint_rematerialize_ops": int(checkpoint_summary.get("checkpoint_rematerialize_ops", 0) or 0),
        },
        "regions": region_rows,
        "tensors": placed,
        "total_bytes": int(total_bytes),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate contiguous training memory layout from IR2 tensor metadata.")
    p.add_argument("--ir2", type=Path, required=True, help="Path to ir2_train_backward.json")
    p.add_argument("--manifest", type=Path, default=None, help="Optional weights_manifest.json for bump offsets")
    p.add_argument("--output", type=Path, required=True, help="Output layout_train.json path")
    p.add_argument("--align-bytes", type=int, default=ALIGN_BYTES_DEFAULT, help="Alignment for tensor offsets")
    p.add_argument("--strict", action="store_true", help="Fail if any tensor lacks numel metadata")
    args = p.parse_args()

    ir2 = _load_json(args.ir2)
    manifest = _load_json(args.manifest) if args.manifest and args.manifest.exists() else None
    layout = build_layout(ir2=ir2, manifest=manifest, align_bytes=max(1, int(args.align_bytes)), strict=bool(args.strict))
    _save_json(args.output, layout)
    print(f"Wrote training layout: {args.output} (tensors={len(layout.get('tensors', []))} total_bytes={layout.get('total_bytes', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

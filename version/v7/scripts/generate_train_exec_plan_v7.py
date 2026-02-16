#!/usr/bin/env python3
"""
generate_train_exec_plan_v7.py

Generate deterministic train execution schedule metadata (IR3-call style)
from IR2. This artifact does not plan memory; it only records runtime call
ordering and dispatch hints so codegen can stay declarative.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shape2(ref: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(ref, dict):
        return None
    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None
    try:
        a = int(shape[-2])
        b = int(shape[-1])
    except Exception:
        return None
    if a <= 0 or b <= 0:
        return None
    return (a, b)


def _resolve_ref_tensor(ref: Dict[str, Any], op_by_id: Dict[int, Dict[str, Any]]) -> Optional[str]:
    if not isinstance(ref, dict):
        return None
    t = ref.get("tensor")
    if isinstance(t, str) and t:
        return t
    from_op = ref.get("from_op")
    from_out = ref.get("from_output")
    if isinstance(from_op, int) and isinstance(from_out, str):
        src = op_by_id.get(from_op)
        if isinstance(src, dict):
            outs = src.get("dataflow", {}).get("outputs", {})
            sref = outs.get(from_out)
            if isinstance(sref, dict):
                st = sref.get("tensor")
                if isinstance(st, str) and st:
                    return st
    return None


def _op_io(
    op: Dict[str, Any],
    op_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    io_inputs: Dict[str, Any] = {}
    io_outputs: Dict[str, Any] = {}
    io_weights: Dict[str, Any] = {}
    df = op.get("dataflow", {})

    for key, ref in (df.get("inputs") or {}).items():
        if not isinstance(key, str):
            continue
        if not isinstance(ref, dict):
            continue
        io_inputs[key] = dict(ref)
        t = _resolve_ref_tensor(ref, op_by_id)
        if isinstance(t, str) and t:
            io_inputs[key]["tensor"] = t

    for key, ref in (df.get("outputs") or {}).items():
        if not isinstance(key, str) or not isinstance(ref, dict):
            continue
        io_outputs[key] = dict(ref)

    for key, ref in (op.get("weights") or {}).items():
        if not isinstance(key, str) or not isinstance(ref, dict):
            continue
        io_weights[key] = dict(ref)

    return io_inputs, io_outputs, io_weights


def _infer_gemm_forward_mnk(op: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, int]:
    df = op.get("dataflow", {}) if isinstance(op, dict) else {}
    in_ref = (df.get("inputs") or {}).get("input")
    out_ref = (df.get("outputs") or {}).get("y")
    w_ref = (op.get("weights") or {}).get("W") if isinstance(op, dict) else None

    d_model = int(cfg.get("embed_dim", cfg.get("hidden_size", 128)) or 128)
    m = 1
    n = d_model
    k = d_model

    out_shape = _shape2(out_ref)
    if out_shape is not None:
        m, n = out_shape

    in_shape = _shape2(in_ref)
    if in_shape is not None:
        if out_shape is None:
            m = int(in_shape[0])
        k = int(in_shape[1])

    w_shape = _shape2(w_ref)
    if w_shape is not None:
        if in_shape is None:
            k = int(w_shape[0])
        if out_shape is None:
            n = int(w_shape[1])

    return {"m": int(max(1, m)), "n": int(max(1, n)), "k": int(max(1, k))}


def _infer_gemm_backward_shape(
    op: Dict[str, Any],
    cfg: Dict[str, Any],
    op_by_id: Dict[int, Dict[str, Any]],
) -> Dict[str, int]:
    io_inputs, io_outputs, _io_weights = _op_io(op, op_by_id)

    d_output_shape = _shape2(io_inputs.get("in_0"))
    input_shape = _shape2(io_inputs.get("in_1"))
    weight_shape = _shape2(io_inputs.get("in_2"))

    d_model = int(cfg.get("embed_dim", cfg.get("hidden_size", 128)) or 128)
    hidden = int(cfg.get("hidden_size", cfg.get("hidden_dim", d_model)) or d_model)

    m = 1
    aligned_in = d_model
    aligned_out = hidden

    if d_output_shape is not None:
        m = int(d_output_shape[0])
        aligned_out = int(d_output_shape[1])
    if input_shape is not None:
        m = int(input_shape[0])
        aligned_in = int(input_shape[1])
    if weight_shape is not None:
        if input_shape is None:
            aligned_in = int(weight_shape[0])
        if d_output_shape is None:
            aligned_out = int(weight_shape[1])

    d_input_ref = io_outputs.get("d_input")
    d_weight_ref = io_outputs.get("d_weight")
    d_bias_ref = io_outputs.get("d_bias")
    d_input_shape = _shape2(d_input_ref)
    d_weight_shape = _shape2(d_weight_ref)
    d_bias_shape = _shape2(d_bias_ref)

    if d_input_shape is not None:
        m = int(d_input_shape[0])
        aligned_in = int(d_input_shape[1])
    if d_weight_shape is not None:
        if aligned_in > 0 and d_weight_shape[0] > 0:
            aligned_in = int(d_weight_shape[0])
        if d_weight_shape[1] > 0:
            aligned_out = int(d_weight_shape[1])
    if d_bias_shape is not None and d_bias_shape[1] > 0:
        aligned_out = int(d_bias_shape[1])

    k = aligned_in
    n = aligned_out
    return {
        "m": int(max(1, m)),
        "n": int(max(1, n)),
        "k": int(max(1, k)),
        "aligned_in": int(max(1, aligned_in)),
        "aligned_out": int(max(1, aligned_out)),
    }


def _choose_dispatch(shape: Dict[str, int], threads: int) -> Dict[str, Any]:
    m = int(shape.get("m", 1) or 1)
    n = int(shape.get("n", 1) or 1)
    work = int(m * n)
    if threads <= 1 or work < 256:
        return {
            "strategy": "serial",
            "split_axis": "none",
            "tile_m": m,
            "tile_n": n,
            "threads": 1,
            "schedule": "static",
            "min_work": 256,
        }

    if m >= max(4, threads):
        tile_m = int(max(1, math.ceil(float(m) / float(threads))))
        tile_n = int(min(256, max(64, n)))
        return {
            "strategy": "tile_2d",
            "split_axis": "m",
            "tile_m": tile_m,
            "tile_n": tile_n,
            "threads": int(threads),
            "schedule": "static",
            "min_work": 256,
        }

    if n >= 64:
        tile_n = int(max(32, math.ceil(float(n) / float(threads))))
        return {
            "strategy": "tile_2d",
            "split_axis": "n",
            "tile_m": m,
            "tile_n": tile_n,
            "threads": int(threads),
            "schedule": "static",
            "min_work": 256,
        }

    return {
        "strategy": "serial",
        "split_axis": "none",
        "tile_m": m,
        "tile_n": n,
        "threads": 1,
        "schedule": "static",
        "min_work": 256,
    }


def _build_plan(ir2: Dict[str, Any], threads: int, mode: str) -> Dict[str, Any]:
    cfg = dict(ir2.get("config") or {})
    forward_ops = sorted(list(ir2.get("forward") or []), key=lambda o: int(o.get("op_id", 0)))
    backward_ops = sorted(list(ir2.get("backward") or []), key=lambda o: int(o.get("op_id", 0)))
    all_ops = forward_ops + backward_ops
    op_by_id = {int(op["op_id"]): op for op in all_ops if isinstance(op, dict) and "op_id" in op}

    out_ops: List[Dict[str, Any]] = []
    for phase, ops in (("forward", forward_ops), ("backward", backward_ops)):
        for op in ops:
            op_id = int(op.get("op_id", -1) or -1)
            kid = str(op.get("kernel_id", "") or "")
            row: Dict[str, Any] = {
                "op_id": op_id,
                "phase": phase,
                "op": str(op.get("op", "") or ""),
                "kernel_id": kid,
                "layer": int(op.get("layer", -1) or -1),
                "role": str(op.get("role", "") or ""),
                "dispatch": {
                    "strategy": "none",
                    "split_axis": "none",
                    "threads": 1,
                    "schedule": "static",
                    "min_work": 0,
                },
                "reduction": {"type": "none"},
                "deterministic_reduction": True,
                "barrier_after": False,
            }

            if kid == "gemm_blocked_serial":
                shape = _infer_gemm_forward_mnk(op, cfg)
                row["shape"] = shape
                row["dispatch"] = _choose_dispatch(shape, threads)
                row["deterministic_reduction"] = True
            elif kid == "gemm_backward_f32":
                shape = _infer_gemm_backward_shape(op, cfg, op_by_id)
                row["shape"] = shape
                row["dispatch"] = _choose_dispatch(shape, threads)
                row["reduction"] = {"type": "none"}
                row["deterministic_reduction"] = True

            out_ops.append(row)

    return {
        "schema": "ck.train.exec.v1",
        "generated_at": _utc_now_iso(),
        "runtime": {
            "threads": int(max(1, threads)),
            "mode": str(mode),
            "schedule": "static",
        },
        "model": {
            "template": str(cfg.get("arch", cfg.get("template", "train"))),
            "dtype": str(cfg.get("weight_dtype", "fp32")),
        },
        "summary": {
            "forward_ops": len(forward_ops),
            "backward_ops": len(backward_ops),
            "total_ops": len(out_ops),
            "gemm_ops": sum(1 for r in out_ops if r.get("kernel_id") in ("gemm_blocked_serial", "gemm_backward_f32")),
        },
        "ops": out_ops,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate v7 training execution plan (IR3-call metadata)")
    p.add_argument("--ir2", type=Path, required=True, help="Path to ir2_train_backward.json")
    p.add_argument("--output", type=Path, required=True, help="Path to train_exec_plan.json")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1), help="Planned thread count for dispatch hints")
    p.add_argument("--mode", choices=["deterministic", "fast"], default="deterministic", help="Dispatch mode label")
    args = p.parse_args()

    ir2 = _load_json(args.ir2)
    plan = _build_plan(ir2, threads=int(args.threads), mode=str(args.mode))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote train exec plan: {args.output} (ops={len(plan.get('ops') or [])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
codegen_train_runtime_v7.py

Generate a compile-ready C training runtime skeleton from v7 IR2 train artifacts.

Goal:
- Emit explicit C functions:
  - ck_train_forward_step(...)
  - ck_train_backward_step(...)
  - ck_train_optimizer_step(...)
  - ck_train_step(...)
- Keep generation deterministic and data-driven from IR2 + kernel registry.
- Prefer compile-readiness and operator visibility over aggressive optimization.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"


FALLBACK_DECLS: Dict[str, str] = {
    "gradient_clip_norm_f32": (
        "float gradient_clip_norm_f32(float *grad, size_t numel, float max_norm);"
    ),
    "adamw_clip_update_multi_f32": (
        "void adamw_clip_update_multi_f32(float *const *grads, float *const *weights, "
        "float *const *m_states, float *const *v_states, const size_t *numels, int tensor_count, "
        "float lr, float beta1, float beta2, float eps, float weight_decay, float max_grad_norm, int step);"
    ),
    # Registry currently omits this declaration for some builds.
    "gemm_blocked_serial": (
        "void gemm_blocked_serial(const float *A, const float *B, const float *bias, "
        "float *C, int M, int N, int K);"
    ),
    # IR2 uses dense_embedding_lookup as logical kernel id.
    "dense_embedding_lookup": (
        "void embedding_forward(const int32_t *token_ids, int token_count, int vocab_size, "
        "const float *token_embeddings, const float *pos_embeddings, float *output, "
        "int embed_dim, int aligned_embed_dim, int context_window, int add_pos);"
    ),
}

FALLBACK_FN_BY_KERNEL_ID: Dict[str, str] = {
    "dense_embedding_lookup": "embedding_forward",
}


# Training codegen is intentionally strict: layout must provide concrete offsets
# for every IR2 tensor before we emit any runtime pointer math.
def _load_train_layout_offsets(
    layout: Optional[Dict[str, Any]],
    tensor_numel: Dict[str, int],
) -> Dict[str, Any]:
    """
    Read layout_train.json and return:
      - total_floats
      - tensor_offsets_floats[tid] -> offset
      - optimizer_m_offsets[name]  -> offset
      - optimizer_v_offsets[name]  -> offset
      - region_offsets_floats[name] -> (offset, size)
    """
    if not isinstance(layout, dict):
        raise RuntimeError("Missing --layout payload for train runtime codegen")

    total_bytes = layout.get("total_bytes")
    if not isinstance(total_bytes, int) or total_bytes <= 0:
        raise RuntimeError("layout_train.json missing positive total_bytes")
    if (total_bytes % 4) != 0:
        raise RuntimeError("layout_train.json total_bytes is not float-aligned")

    tensor_offsets: Dict[str, int] = {}
    opt_m_offsets: Dict[str, int] = {}
    opt_v_offsets: Dict[str, int] = {}

    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeError("layout_train.json missing tensors[]")

    for row in tensors:
        if not isinstance(row, dict):
            continue
        tid = row.get("id")
        off_b = row.get("offset")
        nbytes = row.get("bytes")
        if not isinstance(tid, str) or not isinstance(off_b, int) or not isinstance(nbytes, int):
            continue
        if (off_b % 4) != 0 or (nbytes % 4) != 0:
            raise RuntimeError(f"layout tensor `{tid}` has non-float-aligned offset/size")

        off_f = int(off_b // 4)
        size_f = int(nbytes // 4)

        if tid.startswith("optimizer.m."):
            name = tid[len("optimizer.m.") :]
            opt_m_offsets[name] = off_f
            continue
        if tid.startswith("optimizer.v."):
            name = tid[len("optimizer.v.") :]
            opt_v_offsets[name] = off_f
            continue

        if tid in tensor_numel:
            expected = int(tensor_numel[tid])
            if size_f < expected:
                raise RuntimeError(
                    f"layout tensor `{tid}` too small: layout={size_f} expected={expected}"
                )
            tensor_offsets[tid] = off_f

    missing_tensors = [tid for tid in sorted(tensor_numel.keys()) if tid not in tensor_offsets]
    if missing_tensors:
        raise RuntimeError(
            "layout_train.json missing offsets for %d IR tensors: %s"
            % (len(missing_tensors), ", ".join(missing_tensors[:24]))
        )

    region_offsets: Dict[str, Tuple[int, int]] = {}
    regions = layout.get("regions")
    if isinstance(regions, list):
        for r in regions:
            if not isinstance(r, dict):
                continue
            name = r.get("name")
            off_b = r.get("offset")
            size_b = r.get("bytes")
            if not isinstance(name, str) or not isinstance(off_b, int) or not isinstance(size_b, int):
                continue
            if (off_b % 4) != 0 or (size_b % 4) != 0:
                raise RuntimeError(f"layout region `{name}` has non-float-aligned offset/size")
            region_offsets[name] = (int(off_b // 4), int(size_b // 4))

    return {
        "total_floats": int(total_bytes // 4),
        "tensor_offsets_floats": tensor_offsets,
        "optimizer_m_offsets": opt_m_offsets,
        "optimizer_v_offsets": opt_v_offsets,
        "region_offsets_floats": region_offsets,
    }


def _load_train_exec_plan(exec_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    by_op: Dict[int, Dict[str, Any]] = {}
    if isinstance(exec_plan, dict):
        for row in (exec_plan.get("ops") or []):
            if not isinstance(row, dict):
                continue
            op_id = row.get("op_id")
            if not isinstance(op_id, int):
                try:
                    op_id = int(op_id)
                except Exception:
                    continue
            by_op[int(op_id)] = row
    return {
        "by_op_id": by_op,
        "runtime": (exec_plan.get("runtime") if isinstance(exec_plan, dict) else {}),
        "schema": (exec_plan.get("schema") if isinstance(exec_plan, dict) else None),
    }


def _exec_plan_shape_mnk(op_plan: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    if not isinstance(op_plan, dict):
        return None
    shape = op_plan.get("shape")
    if not isinstance(shape, dict):
        return None
    try:
        m = int(shape.get("m", 0) or 0)
        n = int(shape.get("n", 0) or 0)
        k = int(shape.get("k", 0) or 0)
    except Exception:
        return None
    if m <= 0 or n <= 0 or k <= 0:
        return None
    return (m, n, k)


def _exec_plan_gemm_backward_dims(op_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not isinstance(op_plan, dict):
        return None
    shape = op_plan.get("shape")
    if not isinstance(shape, dict):
        return None
    try:
        aligned_in = int(shape.get("aligned_in", 0) or 0)
        aligned_out = int(shape.get("aligned_out", 0) or 0)
    except Exception:
        return None
    if aligned_in <= 0 or aligned_out <= 0:
        return None
    return {"aligned_in": aligned_in, "aligned_out": aligned_out}


def _exec_plan_dispatch_comment(op_plan: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(op_plan, dict):
        return None
    disp = op_plan.get("dispatch")
    if not isinstance(disp, dict):
        return None
    strategy = str(disp.get("strategy", "none") or "none")
    split_axis = str(disp.get("split_axis", "none") or "none")
    threads = disp.get("threads", 1)
    schedule = str(disp.get("schedule", "static") or "static")
    tile_m = disp.get("tile_m")
    tile_n = disp.get("tile_n")
    return f"strategy={strategy} split_axis={split_axis} threads={threads} schedule={schedule} tile_m={tile_m} tile_n={tile_n}"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _classify_slot_section(tid: str) -> str:
    t = str(tid)
    if t.startswith("weight."):
        return "weights"
    if t.startswith("grad.weight."):
        return "grad_weights"
    if t.startswith("grad.act."):
        return "grad_activations"
    if t.startswith("act."):
        return "activations"
    if t.startswith("saved."):
        return "saved"
    if t.startswith("tmp."):
        return "temporaries"
    if t.startswith("aux."):
        return "aux"
    if t.startswith("optimizer.m."):
        return "optimizer_m"
    if t.startswith("optimizer.v."):
        return "optimizer_v"
    return "other"


def _slot_writable_flags(tid: str) -> Tuple[int, int]:
    t = str(tid)
    if t.startswith("weight."):
        return (0, 0)
    if t.startswith("grad.weight.") or t.startswith("grad.act."):
        return (0, 1)
    if t.startswith("optimizer.m.") or t.startswith("optimizer.v."):
        return (0, 0)
    if t.startswith("tmp."):
        return (0, 1)
    return (1, 1)



def _c_ident(name: str, prefix: str = "t_") -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not out:
        out = "unnamed"
    if out[0].isdigit():
        out = prefix + out
    return out


def _parse_decl(decl: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """
    Parse declarations like:
      void fn(const float *x, int n, float eps);
    Returns:
      ("fn", [("const float *", "x"), ("int", "n"), ("float", "eps")])
    """
    if not isinstance(decl, str) or "(" not in decl or ")" not in decl:
        return None
    one = " ".join(decl.strip().split())
    m = re.match(r"^(?:void|float|int|size_t)\s+([A-Za-z_]\w*)\s*\((.*)\)\s*;?$", one)
    if not m:
        return None
    fn = m.group(1)
    raw_args = m.group(2).strip()
    if raw_args == "" or raw_args == "void":
        return (fn, [])

    parts = [p.strip() for p in raw_args.split(",")]
    args: List[Tuple[str, str]] = []
    for part in parts:
        mm = re.match(r"^(.*?)([A-Za-z_]\w*)$", part)
        if not mm:
            return None
        arg_type = mm.group(1).strip()
        arg_name = mm.group(2).strip()
        args.append((arg_type, arg_name))
    return (fn, args)


def _kernel_registry_map(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k in registry.get("kernels", []):
        kid = k.get("id")
        if isinstance(kid, str) and kid:
            out[kid] = k
    return out


def _dtype_is_i32(dtype: str) -> bool:
    d = str(dtype or "").lower()
    return d in ("i32", "int32", "int32_t")


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


def _collect_tensors(ir2: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tensors: Dict[str, Dict[str, Any]] = {}
    for tid, meta in (ir2.get("tensors") or {}).items():
        if isinstance(tid, str):
            m = dict(meta) if isinstance(meta, dict) else {}
            tensors[tid] = m

    ops = (ir2.get("forward") or []) + (ir2.get("backward") or [])
    op_by_id = {int(op["op_id"]): op for op in ops if isinstance(op, dict) and "op_id" in op}
    for op in ops:
        if not isinstance(op, dict):
            continue
        df = op.get("dataflow", {})
        for r in (df.get("inputs") or {}).values():
            t = _resolve_ref_tensor(r, op_by_id)
            if t and t not in tensors:
                dtype = "int32" if str(t).startswith("input.target") else "fp32"
                tensors[t] = {"dtype": dtype, "kind": "activation"}
        for r in (df.get("outputs") or {}).values():
            if isinstance(r, dict):
                t = r.get("tensor")
                if isinstance(t, str) and t:
                    if t not in tensors:
                        tensors[t] = {"dtype": "fp32", "kind": r.get("kind", "activation")}
        for wk, wref in (op.get("weights") or {}).items():
            if not isinstance(wref, dict):
                continue
            wname = wref.get("name")
            if not isinstance(wname, str) or not wname:
                continue
            tid = "weight.%s" % wname
            if tid not in tensors:
                tensors[tid] = {
                    "dtype": wref.get("dtype", "fp32"),
                    "kind": "weight",
                    "from_weight_key": wk,
                }
    return tensors


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return n


def _tensor_numel_map(tensors: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    out: Dict[str, int] = {}
    missing: List[str] = []
    for tid, meta in sorted(tensors.items(), key=lambda x: x[0]):
        if not isinstance(tid, str):
            continue
        m = meta if isinstance(meta, dict) else {}
        n = m.get("numel")
        if isinstance(n, int) and n > 0:
            out[tid] = int(n)
            continue
        shape_n = _shape_numel(m.get("shape"))
        if isinstance(shape_n, int) and shape_n > 0:
            out[tid] = int(shape_n)
            continue
        missing.append(tid)
    return out, missing


def _arg_scalar_expr(arg_type: str, arg_name: str, cfg: Dict[str, Any]) -> str:
    name = str(arg_name).lower()
    is_float = "float" in arg_type and "*" not in arg_type
    is_size_t = "size_t" in arg_type
    is_int = (("int" in arg_type) or is_size_t) and ("*" not in arg_type)

    if is_float:
        if "eps" in name:
            return "1e-5f"
        if "beta1" in name:
            return "0.9f"
        if "beta2" in name:
            return "0.999f"
        if "weight_decay" in name:
            return "0.01f"
        if "learning_rate" in name or name == "lr":
            return "lr"
        if "base" in name:
            return "%.6ff" % float(cfg.get("rope_theta", 10000.0))
        return "1.0f"

    if is_size_t:
        return "1"

    if is_int:
        if "num_heads" in name and "kv" not in name:
            return str(int(cfg.get("num_heads", 1)))
        if "num_kv_heads" in name:
            return str(int(cfg.get("num_kv_heads", cfg.get("num_heads", 1))))
        if name in ("t", "tokens", "num_tokens", "token_count"):
            return "CK_NUM_TOKENS"
        if "vocab" in name:
            return str(int(cfg.get("vocab_size", 256)))
        if "aligned_context_window" in name or "context_window" in name:
            # Training runtime currently executes 1-token micro-steps.
            # Use runtime token count to avoid kernels assuming full-context buffers.
            return "CK_NUM_TOKENS"
        if "head_dim" in name:
            return str(int(cfg.get("head_dim", 64)))
        if "aligned_embed_dim" in name or "embed_dim" in name or "d_model" in name:
            return str(int(cfg.get("embed_dim", cfg.get("hidden_size", 128))))
        if "aligned_in" in name:
            return str(int(cfg.get("embed_dim", cfg.get("hidden_size", 128))))
        if "aligned_out" in name:
            return str(int(cfg.get("hidden_size", cfg.get("embed_dim", 128))))
        if "kv_stride_tokens" in name:
            return "CK_NUM_TOKENS"
        if "rotary_dim" in name:
            hd = int(cfg.get("head_dim", 64))
            return str(int(cfg.get("rotary_dim", hd)))
        if "numel" in name:
            return "1"
        if "num_threads" in name:
            return "ck_get_num_threads()"
        if "add_pos" in name or "pos_offset" in name:
            return "0"
        return "1"

    if "char *" in arg_type:
        return "\"none\""
    return "0"



def _infer_gemm_backward_dims(
    io_inputs: Dict[str, str],
    io_outputs: Dict[str, str],
    tensors: Dict[str, Dict[str, Any]],
    tensor_numel: Dict[str, int],
    cfg: Dict[str, Any],
) -> Dict[str, int]:
    # NOTE:
    # gemm_backward_f32 is one of the easiest places to create silent OOB writes if
    # aligned_in/aligned_out are wrong. Prefer IR-derived tensor shapes first;
    # use cfg defaults only as a last resort when metadata is missing.
    d_input_n: Optional[int] = None
    d_weight_n: Optional[int] = None
    d_bias_n: Optional[int] = None
    d_output_n: Optional[int] = None

    d_input_w: Optional[int] = None
    d_weight_in_w: Optional[int] = None
    d_weight_out_w: Optional[int] = None
    d_output_w: Optional[int] = None
    d_bias_w: Optional[int] = None

    def _shape2(tid: Optional[str]) -> Optional[Tuple[int, int]]:
        if not isinstance(tid, str):
            return None
        meta = tensors.get(tid) or {}
        sh = meta.get("shape")
        if not isinstance(sh, list) or len(sh) < 2:
            return None
        try:
            a = int(sh[-2])
            b = int(sh[-1])
        except Exception:
            return None
        if a <= 0 or b <= 0:
            return None
        return (a, b)

    for tid in io_inputs.values():
        n = tensor_numel.get(tid)
        if not isinstance(n, int) or n <= 0:
            continue
        lt = str(tid).lower()
        if d_output_n is None and (
            "d_output" in lt
            or "grad_output" in lt
            or lt.startswith("grad.act.")
        ):
            d_output_n = int(n)
            sh = _shape2(tid)
            if sh is not None:
                d_output_w = int(sh[1])

    for tid in io_outputs.values():
        n = tensor_numel.get(tid)
        if not isinstance(n, int) or n <= 0:
            continue
        lt = str(tid).lower()
        if d_input_n is None and (
            "d_input" in lt
            or lt.endswith(".input")
            or (lt.startswith("tmp.grad.act.") and ".input" in lt)
        ):
            d_input_n = int(n)
            sh = _shape2(tid)
            if sh is not None:
                d_input_w = int(sh[1])
        if d_weight_n is None and lt.startswith("tmp.grad.weight.") and lt.endswith(".w"):
            d_weight_n = int(n)
            sh = _shape2(tid)
            if sh is not None:
                d_weight_out_w = int(sh[0])
                d_weight_in_w = int(sh[1])
        if d_bias_n is None and lt.startswith("tmp.grad.weight.") and lt.endswith(".bias"):
            d_bias_n = int(n)
            d_bias_w = int(n)
        if d_bias_n is None and ("d_bias" in lt or lt.endswith(".bias")):
            d_bias_n = int(n)
            d_bias_w = int(n)

    aligned_in = 0
    aligned_out = 0

    # Prefer explicit shape-derived widths first; this avoids token-flattened
    # numel mismatches when CK_NUM_TOKENS > 1.
    if isinstance(d_input_w, int) and d_input_w > 0:
        aligned_in = int(d_input_w)
    elif isinstance(d_weight_in_w, int) and d_weight_in_w > 0:
        aligned_in = int(d_weight_in_w)
    elif isinstance(d_input_n, int) and d_input_n > 0:
        tok = max(1, int(cfg.get("train_tokens", cfg.get("tokens", 1)) or 1))
        if d_input_n % tok == 0:
            aligned_in = int(d_input_n // tok)
        else:
            aligned_in = int(d_input_n)

    if isinstance(d_weight_out_w, int) and d_weight_out_w > 0:
        aligned_out = int(d_weight_out_w)
    elif isinstance(d_output_w, int) and d_output_w > 0:
        aligned_out = int(d_output_w)
    elif isinstance(d_bias_w, int) and d_bias_w > 0:
        aligned_out = int(d_bias_w)
    elif isinstance(d_weight_n, int) and d_weight_n > 0 and aligned_in > 0 and (d_weight_n % aligned_in == 0):
        aligned_out = int(d_weight_n // aligned_in)
    elif isinstance(d_bias_n, int) and d_bias_n > 0:
        aligned_out = int(d_bias_n)

    if aligned_in <= 0 and isinstance(d_weight_n, int) and d_weight_n > 0 and aligned_out > 0 and (d_weight_n % aligned_out == 0):
        aligned_in = int(d_weight_n // aligned_out)

    if aligned_in <= 0:
        aligned_in = int(cfg.get("embed_dim", cfg.get("hidden_size", 128)))
    if aligned_out <= 0:
        aligned_out = int(cfg.get("hidden_size", cfg.get("embed_dim", 128)))

    return {
        "aligned_in": int(aligned_in),
        "aligned_out": int(aligned_out),
    }

def _infer_gemm_forward_mnk(op: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """Infer GEMM forward M/N/K directly from IR op shapes."""
    df = op.get("dataflow", {}) if isinstance(op, dict) else {}
    in_ref = (df.get("inputs") or {}).get("input")
    out_ref = (df.get("outputs") or {}).get("y")
    w_ref = (op.get("weights") or {}).get("W") if isinstance(op, dict) else None

    def _shape2(ref: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(ref, dict):
            return None
        sh = ref.get("shape")
        if not isinstance(sh, list) or len(sh) < 2:
            return None
        try:
            a = int(sh[-2])
            b = int(sh[-1])
            if a > 0 and b > 0:
                return (a, b)
        except Exception:
            return None
        return None

    m = 1
    n = int(cfg.get("embed_dim", cfg.get("hidden_size", 128)) or 128)
    k = int(cfg.get("embed_dim", cfg.get("hidden_size", 128)) or 128)

    out_shape = _shape2(out_ref)
    if out_shape is not None:
        m = int(out_shape[0])
        n = int(out_shape[1])

    in_shape = _shape2(in_ref)
    if in_shape is not None:
        k = int(in_shape[1])
        if out_shape is None:
            m = int(in_shape[0])

    w_shape = _shape2(w_ref)
    if w_shape is not None:
        # Use weight shape only as fallback when input/output dims are unavailable.
        if in_shape is None:
            k = int(w_shape[0])
        if out_shape is None:
            n = int(w_shape[1])

    return (max(1, int(m)), max(1, int(n)), max(1, int(k)))


def _sorted_pool_items(pool: Dict[str, str]) -> List[Tuple[str, str]]:
    def key_fn(item: Tuple[str, str]) -> Tuple[int, int, str]:
        k = str(item[0])
        m = re.match(r"^(?:in|out)_(\d+)$", k)
        if m:
            return (0, int(m.group(1)), k)
        return (1, 0, k)
    return sorted([(k, v) for k, v in pool.items() if isinstance(v, str)], key=key_fn)


def _choose_tensor_for_ptr(
    arg_type: str,
    arg_name: str,
    io_inputs: Dict[str, str],
    io_outputs: Dict[str, str],
    io_weights: Dict[str, str],
    tensors: Dict[str, Dict[str, Any]],
    tvars_f32: Dict[str, str],
    tvars_i32: Dict[str, str],
    fallback_state: Optional[Dict[str, int]] = None,
) -> str:
    want_i32 = "int32_t" in arg_type or "int32" in arg_type
    const_ptr = "const" in arg_type
    aname = str(arg_name)
    lname = aname.lower()

    if (not want_i32) and (not const_ptr) and ("loss" in lname):
        return "g_loss_scalar"

    def tid_dtype_is_i32(tid: str) -> bool:
        dtype = str((tensors.get(tid) or {}).get("dtype", "fp32")).lower()
        return _dtype_is_i32(dtype)

    def tid_to_var(tid: Optional[str]) -> Optional[str]:
        if not isinstance(tid, str):
            return None
        if want_i32:
            if tid_dtype_is_i32(tid) and tid in tvars_i32:
                return tvars_i32[tid]
            return None
        if (not tid_dtype_is_i32(tid)) and tid in tvars_f32:
            return tvars_f32[tid]
        return None

    def find_semantic(pool: Dict[str, str], token: str) -> Optional[str]:
        for k, tid in _sorted_pool_items(pool):
            lk = str(k).lower()
            if lk == token or token == lk:
                return tid
            if token and (token in lk or lk in token):
                return tid
        return None

    # Exact mapping first by key name.
    for pool in (io_inputs, io_outputs, io_weights):
        var = tid_to_var(pool.get(aname))
        if var:
            return var

    # Common backward-core semantic aliases.
    # IR2 often uses generic keys (in_0/in_1/in_2, d_input/d_weight/d_bias)
    # while declarations use kernel-specific argument names.
    if const_ptr:
        if lname in ("d_output", "grad_output", "dy"):
            var = tid_to_var(io_inputs.get("in_0"))
            if var:
                return var
        if ("input" in lname) and (not lname.startswith("d_")):
            var = tid_to_var(io_inputs.get("in_1"))
            if var:
                return var
        if ("weight" in lname) or lname.startswith("w") or ("w_" in lname):
            var = tid_to_var(io_inputs.get("in_2"))
            if var:
                return var

    if not const_ptr:
        if ("input" in lname) or lname.startswith("dx") or lname.startswith("d_x"):
            var = tid_to_var(io_outputs.get("d_input"))
            if var:
                return var
        if ("weight" in lname) or lname.startswith("dw") or lname.startswith("d_w") or ("w_" in lname):
            var = tid_to_var(io_outputs.get("d_weight"))
            if var:
                return var
        if ("bias" in lname) or lname.startswith("db") or lname.startswith("d_b"):
            var = tid_to_var(io_outputs.get("d_bias"))
            if var:
                return var

    if aname.startswith("d_"):
        base = aname[2:]
        for pool in (io_outputs, io_inputs, io_weights):
            var = tid_to_var(pool.get(base))
            if var:
                return var

    # Explicit accumulator semantics.
    if lname == "dst" or lname.endswith("_dst"):
        for cand in (io_outputs.get("dst"), io_inputs.get("dst")):
            var = tid_to_var(cand)
            if var:
                return var
    if lname == "src" or lname.endswith("_src"):
        for cand in (io_inputs.get("src"), io_outputs.get("src")):
            var = tid_to_var(cand)
            if var:
                return var

    # Input token/target hints.
    if "target" in lname:
        for k, tid in _sorted_pool_items(io_inputs):
            if "target" in str(k).lower():
                var = tid_to_var(tid)
                if var:
                    return var
    if "token" in lname or "input_id" in lname:
        for k, tid in _sorted_pool_items(io_inputs):
            lk = str(k).lower()
            if "token" in lk or "input" in lk:
                var = tid_to_var(tid)
                if var:
                    return var

    # Semantic fuzzy mapping.
    if lname.startswith("d_"):
        token = lname[2:]
        for pool in (io_outputs, io_inputs, io_weights):
            tid = find_semantic(pool, token)
            var = tid_to_var(tid)
            if var:
                return var
    for token in (lname, lname.split("_")[-1] if "_" in lname else ""):
        if not token:
            continue
        for pool in (io_inputs, io_outputs, io_weights):
            tid = find_semantic(pool, token)
            var = tid_to_var(tid)
            if var:
                return var

    # Ordered fallback with per-op state, so multiple pointer args map to different buffers.
    const_tids: List[str] = []
    for _k, tid in _sorted_pool_items(io_inputs):
        const_tids.append(tid)
    for _k, tid in _sorted_pool_items(io_weights):
        const_tids.append(tid)
    for _k, tid in _sorted_pool_items(io_outputs):
        const_tids.append(tid)

    mut_tids: List[str] = []
    for _k, tid in _sorted_pool_items(io_outputs):
        mut_tids.append(tid)
    for _k, tid in _sorted_pool_items(io_inputs):
        mut_tids.append(tid)

    # Dedup preserve order.
    def dedup(vals: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in vals:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    const_tids = dedup(const_tids)
    mut_tids = dedup(mut_tids)

    def filtered(tids: List[str]) -> List[str]:
        out: List[str] = []
        for tid in tids:
            if want_i32 and tid_dtype_is_i32(tid):
                out.append(tid)
            if (not want_i32) and (not tid_dtype_is_i32(tid)):
                out.append(tid)
        return out

    pool = filtered(const_tids if const_ptr else mut_tids)
    if pool:
        st = fallback_state if isinstance(fallback_state, dict) else {}
        skey = ("c_" if const_ptr else "m_") + ("i32" if want_i32 else "f32")
        idx = int(st.get(skey, 0) or 0)
        pick = pool[idx] if idx < len(pool) else pool[-1]
        st[skey] = idx + 1
        var = tid_to_var(pick)
        if var:
            return var

    if want_i32:
        return "g_dummy_i32"
    return "g_dummy_f32"


def _op_io(
    op: Dict[str, Any],
    op_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    io_inputs: Dict[str, str] = {}
    io_outputs: Dict[str, str] = {}
    io_weights: Dict[str, str] = {}
    df = op.get("dataflow", {})
    for key, ref in (df.get("inputs") or {}).items():
        if not isinstance(key, str):
            continue
        t = _resolve_ref_tensor(ref, op_by_id)
        if isinstance(t, str):
            io_inputs[key] = t
    for key, ref in (df.get("outputs") or {}).items():
        if not isinstance(key, str) or not isinstance(ref, dict):
            continue
        t = ref.get("tensor")
        if isinstance(t, str) and t:
            io_outputs[key] = t
    for key, wref in (op.get("weights") or {}).items():
        if not isinstance(key, str) or not isinstance(wref, dict):
            continue
        wname = wref.get("name")
        if isinstance(wname, str) and wname:
            io_weights[key] = "weight.%s" % wname
    return io_inputs, io_outputs, io_weights


def _choose_numel_expr(
    arg_name: str,
    io_inputs: Dict[str, str],
    io_outputs: Dict[str, str],
    io_weights: Dict[str, str],
    tensor_numel: Dict[str, int],
) -> str:
    lname = str(arg_name).lower()
    candidates: List[str] = []

    def add_tid(tid: Optional[str]) -> None:
        if not isinstance(tid, str):
            return
        if tid in tensor_numel and tid not in candidates:
            candidates.append(tid)

    # Direct key matches first.
    for pool in (io_outputs, io_inputs, io_weights):
        add_tid(pool.get(arg_name))

    # Heuristic key matches.
    for pool in (io_outputs, io_inputs, io_weights):
        for k, tid in pool.items():
            lk = str(k).lower()
            if lk and (lk in lname or lname in lk):
                add_tid(tid)

    if "dst" in lname:
        add_tid(io_outputs.get("dst"))
        add_tid(io_inputs.get("dst"))
    if "src" in lname:
        add_tid(io_inputs.get("src"))
        add_tid(io_outputs.get("src"))
    if "weight" in lname:
        for tid in io_weights.values():
            add_tid(tid)

    # Fall back to first output/input/weight tensor.
    for pool in (io_outputs, io_inputs, io_weights):
        for tid in pool.values():
            add_tid(tid)

    if candidates:
        return str(int(tensor_numel[candidates[0]]))
    return "1"


def generate_c(ir2: Dict[str, Any], registry: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None, layout: Optional[Dict[str, Any]] = None, exec_plan: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    cfg = dict(ir2.get("config") or {})
    tensors = _collect_tensors(ir2)
    tensor_numel, missing_numel = _tensor_numel_map(tensors)
    if missing_numel:
        raise RuntimeError(
            "IR2 tensor layout missing numel metadata for %d tensor(s): %s" % (
                len(missing_numel), ", ".join(missing_numel[:16])
            )
        )

    kernel_map = _kernel_registry_map(registry)
    forward_ops = sorted(list(ir2.get("forward") or []), key=lambda o: int(o.get("op_id", 0)))
    backward_ops = sorted(list(ir2.get("backward") or []), key=lambda o: int(o.get("op_id", 0)))
    all_ops = forward_ops + backward_ops
    op_by_id = {int(op["op_id"]): op for op in all_ops if isinstance(op, dict) and "op_id" in op}
    exec_plan_info = _load_train_exec_plan(exec_plan)
    exec_plan_by_op = exec_plan_info["by_op_id"]
    # rope_forward_qk expects precomputed cos/sin caches. If present in IR,
    # ck_train_init must seed deterministic caches before the first forward call.
    has_rope_forward_qk = any(str(op.get("kernel_id", "")) == "rope_forward_qk" for op in forward_ops)

    tvars_f32: Dict[str, str] = {}
    tvars_i32: Dict[str, str] = {}
    for tid, meta in sorted(tensors.items(), key=lambda x: x[0]):
        ident = _c_ident(tid, prefix="t_")
        dtype = str((meta or {}).get("dtype", "fp32"))
        if _dtype_is_i32(dtype):
            tvars_i32[tid] = ident
        else:
            tvars_f32[tid] = ident

    # Collect declarations required by kernels in this IR.
    used_decl: Dict[str, Tuple[str, List[Tuple[str, str]], str]] = {}
    skipped_ops: List[str] = []

    def resolve_decl_for_kernel(kernel_id: str) -> Optional[Tuple[str, List[Tuple[str, str]], str]]:
        k = kernel_map.get(kernel_id) or {}
        impl = k.get("impl") or {}
        fn = impl.get("function")
        decl = impl.get("c_declaration")

        if (not fn) and kernel_id in FALLBACK_FN_BY_KERNEL_ID:
            fn = FALLBACK_FN_BY_KERNEL_ID[kernel_id]
        if (not decl) and kernel_id in FALLBACK_DECLS:
            decl = FALLBACK_DECLS[kernel_id]

        if not fn and isinstance(decl, str):
            parsed = _parse_decl(decl)
            if parsed:
                fn = parsed[0]
        if not (fn and decl):
            return None
        parsed = _parse_decl(decl)
        if not parsed:
            return None
        return (parsed[0], parsed[1], decl)

    for op in all_ops:
        kid = op.get("kernel_id")
        if not isinstance(kid, str) or not kid:
            continue
        if kid in used_decl:
            continue
        resolved = resolve_decl_for_kernel(kid)
        if resolved is not None:
            used_decl[kid] = resolved

    # Optimizer kernels are required by generated ck_train_optimizer_step
    # even when the current IR2 does not carry explicit optimizer nodes.
    for opt_kernel in ("adamw_update_f32", "gradient_clip_norm_f32", "adamw_clip_update_multi_f32"):
        if opt_kernel not in used_decl:
            resolved = resolve_decl_for_kernel(opt_kernel)
            if resolved is not None:
                used_decl[opt_kernel] = resolved

    # Optimizer state (AdamW moments) for each grad.weight.* tensor.
    # AdamW kernel here is fp32-only, so non-fp32 params are skipped.
    opt_pairs: List[Tuple[str, str, str, str, str, int]] = []
    skipped_opt_non_fp32: List[str] = []
    grad_weight_tids = sorted([tid for tid in tensors.keys() if tid.startswith("grad.weight.")])
    for gtid in grad_weight_tids:
        wname = gtid[len("grad.weight."):]
        wtid = "weight.%s" % wname
        gvar = tvars_f32.get(gtid)
        wvar = tvars_f32.get(wtid)
        if not (gvar and wvar):
            continue
        wdtype = str((tensors.get(wtid) or {}).get("dtype", "fp32")).lower()
        if wdtype not in ("fp32", "f32"):
            skipped_opt_non_fp32.append(wname)
            continue
        mvar = _c_ident("m." + wname, prefix="m_")
        vvar = _c_ident("v." + wname, prefix="v_")
        numel = tensor_numel.get(gtid)
        if numel is None:
            numel = tensor_numel.get(wtid)
        if not isinstance(numel, int) or numel <= 0:
            raise RuntimeError("Missing positive numel for optimizer tensor pair: %s" % wname)
        opt_pairs.append((wname, gvar, wvar, mvar, vvar, int(numel)))

    # Runtime weight hydration order for ck_train_init (flattened fp32 payload).
    init_weight_specs: List[Tuple[str, str, int]] = []
    for tid in sorted(tensors.keys()):
        if not isinstance(tid, str) or (not tid.startswith("weight.")):
            continue
        var = tvars_f32.get(tid)
        if not var:
            continue
        wname = tid[len("weight."):]
        numel = tensor_numel.get(tid)
        if not isinstance(numel, int) or numel <= 0:
            raise RuntimeError("Missing positive numel for weight tensor: %s" % tid)
        init_weight_specs.append((wname, var, int(numel)))

    layout_info = _load_train_layout_offsets(layout, tensor_numel)
    tensor_offsets = layout_info["tensor_offsets_floats"]
    region_offsets = layout_info["region_offsets_floats"]

    tensor_offset_macros: Dict[str, str] = {}
    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        tensor_offset_macros[tid] = "OFF_%s" % var

    opt_m_offset_macros: Dict[str, str] = {}
    opt_v_offset_macros: Dict[str, str] = {}
    opt_m_offsets = layout_info["optimizer_m_offsets"]
    opt_v_offsets = layout_info["optimizer_v_offsets"]
    for wname, _gvar, _wvar, mvar, vvar, _numel in opt_pairs:
        if wname not in opt_m_offsets:
            raise RuntimeError("layout_train.json missing optimizer.m offset for %s" % wname)
        if wname not in opt_v_offsets:
            raise RuntimeError("layout_train.json missing optimizer.v offset for %s" % wname)
        opt_m_offset_macros[mvar] = "OFF_%s" % mvar
        opt_v_offset_macros[vvar] = "OFF_%s" % vvar

    # Deterministic slot registry used for runtime diagnostics/canary checks.
    # Keep ordering stable so a failing canary index maps to the same range/slot
    # across runs and can be decoded consistently in strict mode.
    slot_rows: List[Dict[str, Any]] = []
    for tid, _var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        wf, wb = _slot_writable_flags(tid)
        slot_rows.append({
            "name": tid,
            "offset": int(tensor_offsets[tid]),
            "numel": int(tensor_numel[tid]),
            "section": _classify_slot_section(tid),
            "writable_fwd": int(wf),
            "writable_bwd": int(wb),
        })

    for wname, _gvar, _wvar, _mvar, _vvar, numel in opt_pairs:
        wf, wb = _slot_writable_flags("optimizer.m." + wname)
        slot_rows.append({
            "name": "optimizer.m." + wname,
            "offset": int(opt_m_offsets[wname]),
            "numel": int(numel),
            "section": "optimizer_m",
            "writable_fwd": int(wf),
            "writable_bwd": int(wb),
        })
        wf, wb = _slot_writable_flags("optimizer.v." + wname)
        slot_rows.append({
            "name": "optimizer.v." + wname,
            "offset": int(opt_v_offsets[wname]),
            "numel": int(numel),
            "section": "optimizer_v",
            "writable_fwd": int(wf),
            "writable_bwd": int(wb),
        })

    slot_rows = sorted(slot_rows, key=lambda r: (int(r["offset"]), str(r["name"])))
    slot_name_to_idx: Dict[str, int] = {}
    prev_end = 0
    for idx, row in enumerate(slot_rows):
        off = int(row["offset"])
        numel = int(row["numel"])
        if numel <= 0:
            raise RuntimeError("Non-positive slot numel for `%s`" % row["name"])
        if off < prev_end:
            raise RuntimeError(
                "Overlapping train layout slots: `%s` starts at %d before prior end %d"
                % (row["name"], off, prev_end)
            )
        prev_end = max(prev_end, off + numel)
        slot_name_to_idx[str(row["name"])] = idx

    if prev_end > int(layout_info["total_floats"]):
        raise RuntimeError(
            "Slot coverage exceeds layout total floats: end=%d total=%d"
            % (prev_end, int(layout_info["total_floats"]))
        )

    canary_ranges: List[Tuple[int, int, int, int]] = []
    cursor = 0
    left_idx = -1
    for idx, row in enumerate(slot_rows):
        off = int(row["offset"])
        end = off + int(row["numel"])
        if off > cursor:
            canary_ranges.append((cursor, off - cursor, left_idx, idx))
        cursor = max(cursor, end)
        left_idx = idx
    if cursor < int(layout_info["total_floats"]):
        canary_ranges.append((cursor, int(layout_info["total_floats"]) - cursor, left_idx, -1))

    weight_slot_indices = [
        idx for idx, row in enumerate(slot_rows)
        if str(row["name"]).startswith("weight.")
    ]
    weight_snapshot_floats = sum(int(slot_rows[idx]["numel"]) for idx in weight_slot_indices)
    # Strict snapshot-oracle compares only forward activation slots here.
    # Exclude saved.* and aux.* because those are kernel-internal or reporting
    # artifacts and may not map 1:1 to oracle-forward tensors.
    activation_slot_indices = [
        idx for idx, row in enumerate(slot_rows)
        if str(row.get("section", "")) in ("activations",)
    ]
    activation_snapshot_floats = sum(int(slot_rows[idx]["numel"]) for idx in activation_slot_indices)
    optimizer_state_slot_indices = [
        idx for idx, row in enumerate(slot_rows)
        if str(row.get("section", "")) in ("optimizer_m", "optimizer_v")
    ]
    optimizer_state_snapshot_floats = sum(int(slot_rows[idx]["numel"]) for idx in optimizer_state_slot_indices)
    accum_snapshot_floats = 0
    if "grads" in region_offsets:
        accum_snapshot_floats += int(region_offsets["grads"][1])
    if "grad_activations" in region_offsets:
        accum_snapshot_floats += int(region_offsets["grad_activations"][1])

    lines: List[str] = []
    ap = lines.append
    ap("/*")
    ap(" * Auto-generated by codegen_train_runtime_v7.py")
    ap(" *")
    ap(" * Compile-ready v7 training runtime skeleton.")
    ap(" * Source: IR2 backward artifact (forward + backward op chains).")
    ap(" */")
    ap("")
    ap("#include <stdint.h>")
    ap("#include <stddef.h>")
    ap("#include <string.h>")
    ap("#include <stdlib.h>")
    ap("#include <math.h>")
    ap("#include <time.h>")
    ap("#include \"ckernel_engine.h\"")
    ap("")
    ap("#ifndef CK_NUM_TOKENS")
    ap("#define CK_NUM_TOKENS 1")
    ap("#endif")
    ap("#ifndef CK_GRAD_ACCUM_STEPS")
    ap("#define CK_GRAD_ACCUM_STEPS 1")
    ap("#endif")
    ap("#if (CK_GRAD_ACCUM_STEPS < 1)")
    ap("#undef CK_GRAD_ACCUM_STEPS")
    ap("#define CK_GRAD_ACCUM_STEPS 1")
    ap("#endif")
    ap("#ifndef CK_MAX_GRAD_NORM")
    ap("#define CK_MAX_GRAD_NORM 0.0f")
    ap("#endif")
    ap("")

    ap("/* Training GEMM wrapper (threadpool dispatch + serial fallback). */")
    ap("void gemm_blocked_serial_train_parallel_dispatch(const float *A, const float *B, const float *bias, float *C, int M, int N, int K);")
    ap("void gemm_backward_f32_train_parallel_dispatch(const float *d_output, const float *input, const float *W, float *d_input, float *d_W, float *d_b, int T, int aligned_in, int aligned_out, int num_threads);")
    ap("void gemm_backward_f32_train_parallel_dispatch_v2(const float *d_output, const float *input, const float *W, float *d_input, float *d_W, float *d_b, int T, int aligned_in, int aligned_out, int num_threads);")
    ap("/* Fallback declarations for kernels with incomplete registry declarations. */")
    for kid in sorted(used_decl.keys()):
        fn, args, decl = used_decl[kid]
        ap("/* %s -> %s */" % (kid, fn))
        ap(decl if decl.endswith(";") else (decl + ";"))
    ap("")

    ap("#define CK_TRAIN_TOTAL_FLOATS ((size_t)%d)" % int(layout_info["total_floats"]))
    ap("#define CK_CANARY_TAIL_FLOATS ((size_t)16)")
    rope_heads = int(cfg.get("num_heads", 8) or 8)
    rope_kv_heads = int(cfg.get("num_kv_heads", max(1, rope_heads // 2)) or max(1, rope_heads // 2))
    rope_head_dim = int(cfg.get("head_dim", max(1, int(cfg.get("d_model", 128) or 128) // max(1, rope_heads))) or max(1, int(cfg.get("d_model", 128) or 128) // max(1, rope_heads)))
    rope_aligned_head_dim = int(cfg.get("aligned_head_dim", rope_head_dim) or rope_head_dim)
    rope_rotary_dim = int(cfg.get("rope_rotary_dim", rope_head_dim) or rope_head_dim)
    rope_theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
    ap("#define CK_ROPE_NUM_HEADS %d" % int(rope_heads))
    ap("#define CK_ROPE_NUM_KV_HEADS %d" % int(rope_kv_heads))
    ap("#define CK_ROPE_HEAD_DIM %d" % int(rope_head_dim))
    ap("#define CK_ROPE_ALIGNED_HEAD_DIM %d" % int(rope_aligned_head_dim))
    ap("#define CK_ROPE_ROTARY_DIM %d" % int(rope_rotary_dim))
    ap("#define CK_ROPE_THETA %.6ff" % float(rope_theta))
    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        ap("#define %s ((size_t)%d)" % (tensor_offset_macros[tid], int(tensor_offsets[tid])))
    for _wname, _gvar, _wvar, mvar, vvar, _numel in sorted(opt_pairs, key=lambda x: x[0]):
        ap("#define %s ((size_t)%d)" % (opt_m_offset_macros[mvar], int(opt_m_offsets[_wname])))
        ap("#define %s ((size_t)%d)" % (opt_v_offset_macros[vvar], int(opt_v_offsets[_wname])))
    ap("")

    if "grads" in region_offsets:
        goff, gsize = region_offsets["grads"]
        ap("#define CK_GRADS_OFFSET ((size_t)%d)" % int(goff))
        ap("#define CK_GRADS_SIZE   ((size_t)%d)" % int(gsize))
    else:
        ap("#define CK_GRADS_OFFSET ((size_t)0)")
        ap("#define CK_GRADS_SIZE   ((size_t)0)")
    if "grad_activations" in region_offsets:
        gaoff, gasize = region_offsets["grad_activations"]
        ap("#define CK_GRAD_ACT_OFFSET ((size_t)%d)" % int(gaoff))
        ap("#define CK_GRAD_ACT_SIZE   ((size_t)%d)" % int(gasize))
    else:
        ap("#define CK_GRAD_ACT_OFFSET ((size_t)0)")
        ap("#define CK_GRAD_ACT_SIZE   ((size_t)0)")
    ap("")
    ap("typedef struct {")
    ap("    const char *name;")
    ap("    size_t offset_floats;")
    ap("    size_t numel;")
    ap("    const char *section;")
    ap("    int writable_fwd;")
    ap("    int writable_bwd;")
    ap("} CKTensorSlot;")
    ap("typedef struct {")
    ap("    size_t start_floats;")
    ap("    size_t len_floats;")
    ap("    int left_slot_idx;")
    ap("    int right_slot_idx;")
    ap("} CKCanaryRange;")
    ap("#define CK_CANARY_VALUE_U32 0xDEADBEEF")
    ap("#ifndef CK_RUNTIME_CANARY_CHECKS")
    ap("#define CK_RUNTIME_CANARY_CHECKS 0")
    ap("#endif")
    ap("#ifndef CK_RUNTIME_BOUNDS_ASSERT")
    ap("#define CK_RUNTIME_BOUNDS_ASSERT 0")
    ap("#endif")
    ap("#ifndef CK_RUNTIME_FAULT_INJECT")
    ap("#define CK_RUNTIME_FAULT_INJECT 0")
    ap("#endif")
    ap("#ifndef CK_FAULT_INJECT_OP_ID")
    ap("#define CK_FAULT_INJECT_OP_ID (-1)")
    ap("#endif")
    ap("#ifndef CK_ABLATE_QK_NORM_BACKWARD")
    ap("#define CK_ABLATE_QK_NORM_BACKWARD 0")
    ap("#endif")
    ap("#ifndef CK_ABLATE_ROPE_BACKWARD_QK")
    ap("#define CK_ABLATE_ROPE_BACKWARD_QK 0")
    ap("#endif")
    ap("#ifndef CK_ABLATE_ATTENTION_BACKWARD")
    ap("#define CK_ABLATE_ATTENTION_BACKWARD 0")
    ap("#endif")
    ap("#ifndef CK_ABLATE_GEMM_BACKWARD_BIAS")
    ap("#define CK_ABLATE_GEMM_BACKWARD_BIAS 0")
    ap("#endif")
    ap("#define CK_TENSOR_SLOT_COUNT %d" % len(slot_rows))
    ap("static const CKTensorSlot g_tensor_slots[CK_TENSOR_SLOT_COUNT] = {")
    for row in slot_rows:
        ap('    {"%s", (size_t)%d, (size_t)%d, "%s", %d, %d},' % (
            str(row["name"]), int(row["offset"]), int(row["numel"]), str(row["section"]), int(row["writable_fwd"]), int(row["writable_bwd"])
        ))
    ap("};")
    ap("#define CK_CANARY_RANGE_COUNT %d" % len(canary_ranges))
    ap("static const CKCanaryRange g_canary_ranges[CK_CANARY_RANGE_COUNT > 0 ? CK_CANARY_RANGE_COUNT : 1] = {")
    if canary_ranges:
        for start, length, left, right in canary_ranges:
            ap("    {(size_t)%d, (size_t)%d, %d, %d}," % (int(start), int(length), int(left), int(right)))
    else:
        ap("    {(size_t)0, (size_t)0, -1, -1},")
    ap("};")
    ap("#define CK_WEIGHT_SLOT_COUNT %d" % len(weight_slot_indices))
    ap("static const int g_weight_slot_indices[CK_WEIGHT_SLOT_COUNT > 0 ? CK_WEIGHT_SLOT_COUNT : 1] = {")
    if weight_slot_indices:
        for idx in weight_slot_indices:
            ap("    %d," % int(idx))
    else:
        ap("    -1,")
    ap("};")
    ap("#define CK_WEIGHT_SNAPSHOT_FLOATS ((size_t)%d)" % int(weight_snapshot_floats))
    ap("#define CK_ACTIVATION_SLOT_COUNT %d" % len(activation_slot_indices))
    ap("static const int g_activation_slot_indices[CK_ACTIVATION_SLOT_COUNT > 0 ? CK_ACTIVATION_SLOT_COUNT : 1] = {")
    if activation_slot_indices:
        for idx in activation_slot_indices:
            ap("    %d," % int(idx))
    else:
        ap("    -1,")
    ap("};")
    ap("#define CK_ACTIVATION_SNAPSHOT_FLOATS ((size_t)%d)" % int(activation_snapshot_floats))
    ap("#define CK_OPTIMIZER_STATE_SLOT_COUNT %d" % len(optimizer_state_slot_indices))
    ap("static const int g_optimizer_state_slot_indices[CK_OPTIMIZER_STATE_SLOT_COUNT > 0 ? CK_OPTIMIZER_STATE_SLOT_COUNT : 1] = {")
    if optimizer_state_slot_indices:
        for idx in optimizer_state_slot_indices:
            ap("    %d," % int(idx))
    else:
        ap("    -1,")
    ap("};")
    ap("#define CK_OPTIMIZER_STATE_SNAPSHOT_FLOATS ((size_t)%d)" % int(optimizer_state_snapshot_floats))
    ap("#define CK_ACCUM_SNAPSHOT_FLOATS ((size_t)(CK_GRADS_SIZE + CK_GRAD_ACT_SIZE))")
    ap("")

    ap("static float *g_memory = NULL;")
    ap("static size_t g_memory_floats = 0;")
    ap("static float g_dummy_f32[1];")
    ap("static int32_t g_dummy_i32[1];")
    ap("static float g_loss_scalar[1];")
    ap("static float g_rope_cos_cache[((CK_NUM_TOKENS > 0 ? CK_NUM_TOKENS : 1) * ((CK_ROPE_ROTARY_DIM / 2) > 0 ? (CK_ROPE_ROTARY_DIM / 2) : 1))];")
    ap("static float g_rope_sin_cache[((CK_NUM_TOKENS > 0 ? CK_NUM_TOKENS : 1) * ((CK_ROPE_ROTARY_DIM / 2) > 0 ? (CK_ROPE_ROTARY_DIM / 2) : 1))];")
    ap("")

    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        ap("static float *%s; /* %s */" % (var, tid))
    for tid, var in sorted(tvars_i32.items(), key=lambda x: x[1]):
        ap("static int32_t %s[%d]; /* %s */" % (var, int(tensor_numel[tid]), tid))
    ap("")

    if opt_pairs:
        ap("/* Optimizer state (AdamW moments) */")
        seen_m = set()
        seen_v = set()
        for _wname, _gvar, _wvar, mvar, vvar, _numel in opt_pairs:
            if mvar not in seen_m:
                ap("static float *%s;" % mvar)
                seen_m.add(mvar)
            if vvar not in seen_v:
                ap("static float *%s;" % vvar)
                seen_v.add(vvar)
    ap("static int g_opt_step = 0;")
    ap("static int g_accum_step = 0;")
    ap("static int g_profile_steps = 0;")
    ap("static int g_profile_optimizer_steps = 0;")
    ap("static double g_last_step_ms = 0.0;")
    ap("static double g_last_fwd_ms = 0.0;")
    ap("static double g_last_bwd_ms = 0.0;")
    ap("static double g_last_opt_ms = 0.0;")
    ap("static int g_last_fwd_calls = 0;")
    ap("static int g_last_bwd_calls = 0;")
    ap("static int g_last_opt_calls = 0;")
    ap("static int g_last_opt_applied = 0;")
    ap("static double g_total_step_ms = 0.0;")
    ap("static double g_total_fwd_ms = 0.0;")
    ap("static double g_total_bwd_ms = 0.0;")
    ap("static double g_total_opt_ms = 0.0;")
    ap("")
    ap("static inline double ck_now_ms(void) {")
    ap("    struct timespec ts;")
    ap("    clock_gettime(CLOCK_MONOTONIC, &ts);")
    ap("    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;")
    ap("}")
    ap("")

    if init_weight_specs:
        ap("/* Runtime init hydration order (flattened fp32 payload) */")
        ap("#define CK_INIT_WEIGHT_COUNT %d" % len(init_weight_specs))
        ap("static const size_t g_init_weight_offsets[CK_INIT_WEIGHT_COUNT] = {")
        for _wname, _wvar, _numel in init_weight_specs:
            tid = "weight.%s" % _wname
            ap("    %s," % tensor_offset_macros[tid])
        ap("};")
        ap("static int g_init_weight_numel[CK_INIT_WEIGHT_COUNT] = {")
        for _wname, _wvar, _numel in init_weight_specs:
            ap("    %d," % int(_numel))
        ap("};")
        ap("static const char *g_init_weight_names[CK_INIT_WEIGHT_COUNT] = {")
        for _wname, _wvar, _numel in init_weight_specs:
            ap('    "%s",' % _wname)
        ap("};")
    else:
        ap("#define CK_INIT_WEIGHT_COUNT 0")
    ap("")

    ap("int ck_train_alloc(void) {")
    ap("    if (g_memory != NULL) return 0;")
    ap("    size_t alloc_floats = CK_TRAIN_TOTAL_FLOATS + CK_CANARY_TAIL_FLOATS;")
    ap("    g_memory = (float*)calloc(alloc_floats, sizeof(float));")
    ap("    if (g_memory == NULL) return -1;")
    ap("    g_memory_floats = alloc_floats;")
    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        ap("    %s = g_memory + %s;" % (var, tensor_offset_macros[tid]))
    for _wname, _gvar, _wvar, mvar, vvar, _numel in sorted(opt_pairs, key=lambda x: x[0]):
        ap("    %s = g_memory + %s;" % (mvar, opt_m_offset_macros[mvar]))
        ap("    %s = g_memory + %s;" % (vvar, opt_v_offset_macros[vvar]))
    ap("    return 0;")
    ap("}")
    ap("")

    ap("void ck_train_free(void) {")
    ap("    if (g_memory != NULL) {")
    ap("        free(g_memory);")
    ap("        g_memory = NULL;")
    ap("    }")
    ap("    g_memory_floats = 0;")
    for _tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        ap("    %s = NULL;" % var)
    for _wname, _gvar, _wvar, mvar, vvar, _numel in sorted(opt_pairs, key=lambda x: x[0]):
        ap("    %s = NULL;" % mvar)
        ap("    %s = NULL;" % vvar)
    ap("}")
    ap("")
    ap("int ck_train_forward_step(void);")
    ap("int ck_train_backward_step(void);")
    ap("int ck_train_backward_step_trace(int *failed_op_id, int *first_corrupt_idx);")
    ap("int ck_train_optimizer_step(float lr);")
    ap("int ck_train_flush_optimizer(float lr);")
    ap("int ck_train_get_accum_counter(void);")
    ap("int ck_train_get_accum_steps(void);")
    ap("int ck_train_set_accum_counter(int accum_step);")
    ap("int ck_train_get_opt_step(void);")
    ap("int ck_train_set_opt_step(int opt_step);")
    ap("void ck_train_reset_profile(void);")
    ap("int ck_train_get_last_step_profile(double *step_ms, double *fwd_ms, double *bwd_ms, double *opt_ms, int *fwd_calls, int *bwd_calls, int *opt_calls, int *opt_applied);")
    ap("int ck_train_get_cumulative_profile(double *total_ms, double *fwd_ms, double *bwd_ms, double *opt_ms, int *steps, int *optimizer_steps);")
    ap("int ck_train_get_optimizer_state_snapshot_numel(void);")
    ap("int ck_train_export_optimizer_state_snapshot(float *dst, int dst_numel);")
    ap("int ck_train_import_optimizer_state_snapshot(const float *src, int src_numel);")
    ap("int ck_train_get_accum_snapshot_numel(void);")
    ap("int ck_train_export_accum_snapshot(float *dst, int dst_numel);")
    ap("int ck_train_import_accum_snapshot(const float *src, int src_numel);")
    ap("")

    ap("static inline float ck_canary_value_f32(void) {")
    ap("    union { uint32_t u; float f; } v;")
    ap("    v.u = (uint32_t)CK_CANARY_VALUE_U32;")
    ap("    return v.f;")
    ap("}")
    ap("")
    ap("static int g_diag_failed_op_id = -1;")
    ap("static int g_diag_failed_canary_idx = -1;")
    ap("int ck_train_get_last_diag_failed_op(void) { return g_diag_failed_op_id; }")
    ap("int ck_train_get_last_diag_failed_canary(void) { return g_diag_failed_canary_idx; }")
    ap("")
    ap("static int ck_bounds_check_span_f32(const float *ptr, size_t need_numel) {")
    ap("    if (!CK_RUNTIME_BOUNDS_ASSERT) return 0;")
    ap("    if (g_memory == NULL || ptr == NULL) return -1;")
    ap("    if (ptr < g_memory) return -2;")
    ap("    const size_t off = (size_t)(ptr - g_memory);")
    ap("    if (off > CK_TRAIN_TOTAL_FLOATS) return -3;")
    ap("    if (need_numel > CK_TRAIN_TOTAL_FLOATS) return -4;")
    ap("    if ((off + need_numel) > CK_TRAIN_TOTAL_FLOATS) return -5;")
    ap("    return 0;")
    ap("}")
    ap("")
    ap("static int ck_train_snapshot_weights(float *dst, size_t dst_numel) {")
    ap("    if (g_memory == NULL || dst == NULL) return -1;")
    ap("    if (dst_numel < CK_WEIGHT_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_WEIGHT_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_weight_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        memcpy(dst + cursor, g_memory + s->offset_floats, s->numel * sizeof(float));")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_get_weight_snapshot_numel(void) {")
    ap("    return (int)CK_WEIGHT_SNAPSHOT_FLOATS;")
    ap("}")
    ap("")
    ap("int ck_train_export_weight_snapshot(float *dst, int dst_numel) {")
    ap("    if (dst_numel < 0) return -1;")
    ap("    return ck_train_snapshot_weights(dst, (size_t)dst_numel);")
    ap("}")
    ap("")
    ap("static int ck_train_snapshot_activations(float *dst, size_t dst_numel) {")
    ap("    if (g_memory == NULL || dst == NULL) return -1;")
    ap("    if (dst_numel < CK_ACTIVATION_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_ACTIVATION_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_activation_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        memcpy(dst + cursor, g_memory + s->offset_floats, s->numel * sizeof(float));")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_get_activation_snapshot_numel(void) {")
    ap("    return (int)CK_ACTIVATION_SNAPSHOT_FLOATS;")
    ap("}")
    ap("")
    ap("int ck_train_export_activation_snapshot(float *dst, int dst_numel) {")
    ap("    if (dst_numel < 0) return -1;")
    ap("    return ck_train_snapshot_activations(dst, (size_t)dst_numel);")
    ap("}")
    ap("")
    ap("static int ck_train_snapshot_optimizer_state(float *dst, size_t dst_numel) {")
    ap("    if (g_memory == NULL || dst == NULL) return -1;")
    ap("    if (dst_numel < CK_OPTIMIZER_STATE_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_OPTIMIZER_STATE_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_optimizer_state_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        memcpy(dst + cursor, g_memory + s->offset_floats, s->numel * sizeof(float));")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_get_optimizer_state_snapshot_numel(void) {")
    ap("    return (int)CK_OPTIMIZER_STATE_SNAPSHOT_FLOATS;")
    ap("}")
    ap("")
    ap("int ck_train_export_optimizer_state_snapshot(float *dst, int dst_numel) {")
    ap("    if (dst_numel < 0) return -1;")
    ap("    return ck_train_snapshot_optimizer_state(dst, (size_t)dst_numel);")
    ap("}")
    ap("")
    ap("int ck_train_import_optimizer_state_snapshot(const float *src, int src_numel) {")
    ap("    if (g_memory == NULL || src == NULL) return -1;")
    ap("    if (src_numel < (int)CK_OPTIMIZER_STATE_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_OPTIMIZER_STATE_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_optimizer_state_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        memcpy(g_memory + s->offset_floats, src + cursor, s->numel * sizeof(float));")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_get_accum_snapshot_numel(void) {")
    ap("    return (int)CK_ACCUM_SNAPSHOT_FLOATS;")
    ap("}")
    ap("")
    ap("int ck_train_export_accum_snapshot(float *dst, int dst_numel) {")
    ap("    if (g_memory == NULL || dst == NULL) return -1;")
    ap("    if (dst_numel < (int)CK_ACCUM_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    if (CK_GRADS_SIZE > 0) {")
    ap("        memcpy(dst + cursor, g_memory + CK_GRADS_OFFSET, CK_GRADS_SIZE * sizeof(float));")
    ap("        cursor += CK_GRADS_SIZE;")
    ap("    }")
    ap("    if (CK_GRAD_ACT_SIZE > 0) {")
    ap("        memcpy(dst + cursor, g_memory + CK_GRAD_ACT_OFFSET, CK_GRAD_ACT_SIZE * sizeof(float));")
    ap("        cursor += CK_GRAD_ACT_SIZE;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_import_accum_snapshot(const float *src, int src_numel) {")
    ap("    if (g_memory == NULL || src == NULL) return -1;")
    ap("    if (src_numel < (int)CK_ACCUM_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    if (CK_GRADS_SIZE > 0) {")
    ap("        memcpy(g_memory + CK_GRADS_OFFSET, src + cursor, CK_GRADS_SIZE * sizeof(float));")
    ap("        cursor += CK_GRADS_SIZE;")
    ap("    }")
    ap("    if (CK_GRAD_ACT_SIZE > 0) {")
    ap("        memcpy(g_memory + CK_GRAD_ACT_OFFSET, src + cursor, CK_GRAD_ACT_SIZE * sizeof(float));")
    ap("        cursor += CK_GRAD_ACT_SIZE;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("int ck_train_import_weight_snapshot(const float *src, int src_numel) {")
    ap("    if (g_memory == NULL || src == NULL) return -1;")
    ap("    if (src_numel < (int)CK_WEIGHT_SNAPSHOT_FLOATS) return -2;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_WEIGHT_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_weight_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        memcpy(g_memory + s->offset_floats, src + cursor, s->numel * sizeof(float));")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return (int)cursor;")
    ap("}")
    ap("")
    ap("void ck_train_plant_canaries(void) {")
    ap("    if (g_memory == NULL) return;")
    ap("    const float cv = ck_canary_value_f32();")
    ap("    for (int i = 0; i < CK_CANARY_RANGE_COUNT; ++i) {")
    ap("        const CKCanaryRange *r = &g_canary_ranges[i];")
    ap("        for (size_t j = 0; j < r->len_floats; ++j) {")
    ap("            g_memory[r->start_floats + j] = cv;")
    ap("        }")
    ap("    }")
    ap("    for (size_t j = 0; j < CK_CANARY_TAIL_FLOATS; ++j) {")
    ap("        g_memory[CK_TRAIN_TOTAL_FLOATS + j] = cv;")
    ap("    }")
    ap("}")
    ap("")
    ap("int ck_train_check_canaries(const char *phase_name, int *first_corrupt_idx) {")
    ap("    (void)phase_name;")
    ap("    if (first_corrupt_idx != NULL) *first_corrupt_idx = -1;")
    ap("    if (g_memory == NULL) return -1;")
    ap("    const uint32_t expected = (uint32_t)CK_CANARY_VALUE_U32;")
    ap("    for (int i = 0; i < CK_CANARY_RANGE_COUNT; ++i) {")
    ap("        const CKCanaryRange *r = &g_canary_ranges[i];")
    ap("        for (size_t j = 0; j < r->len_floats; ++j) {")
    ap("            uint32_t got = 0u;")
    ap("            memcpy(&got, g_memory + r->start_floats + j, sizeof(uint32_t));")
    ap("            if (got != expected) {")
    ap("                if (first_corrupt_idx != NULL) *first_corrupt_idx = i;")
    ap("                return 1;")
    ap("            }")
    ap("        }")
    ap("    }")
    ap("    for (size_t j = 0; j < CK_CANARY_TAIL_FLOATS; ++j) {")
    ap("        uint32_t got = 0u;")
    ap("        memcpy(&got, g_memory + CK_TRAIN_TOTAL_FLOATS + j, sizeof(uint32_t));")
    ap("        if (got != expected) {")
    ap("            if (first_corrupt_idx != NULL) *first_corrupt_idx = CK_CANARY_RANGE_COUNT + (int)j;")
    ap("            return 2;")
    ap("        }")
    ap("    }")
    ap("    return 0;")
    ap("}")
    ap("")
    ap("int ck_train_check_weights_readonly(const float *weight_snapshot, int *first_corrupt_idx) {")
    ap("    if (first_corrupt_idx != NULL) *first_corrupt_idx = -1;")
    ap("    if (g_memory == NULL || weight_snapshot == NULL) return -1;")
    ap("    size_t cursor = 0;")
    ap("    for (int i = 0; i < CK_WEIGHT_SLOT_COUNT; ++i) {")
    ap("        int slot_idx = g_weight_slot_indices[i];")
    ap("        if (slot_idx < 0 || slot_idx >= CK_TENSOR_SLOT_COUNT) continue;")
    ap("        const CKTensorSlot *s = &g_tensor_slots[slot_idx];")
    ap("        const float *cur = g_memory + s->offset_floats;")
    ap("        const float *ref = weight_snapshot + cursor;")
    ap("        for (size_t j = 0; j < s->numel; ++j) {")
    ap("            if (cur[j] != ref[j]) {")
    ap("                if (first_corrupt_idx != NULL) *first_corrupt_idx = slot_idx;")
    ap("                return 1;")
    ap("            }")
    ap("        }")
    ap("        cursor += s->numel;")
    ap("    }")
    ap("    return 0;")
    ap("}")
    ap("")
    ap("/*")
    ap(" * Layout bridge helpers (token-major <-> head-major).")
    ap(" *")
    ap(" * IR tensors for q/k/v/attn are modeled token-major [T, H*Dh].")
    ap(" * Attention/RoPE kernels here consume head-major [H, T, aligned_Dh].")
    ap(" * Keep this conversion explicit in codegen to avoid silent contract drift.")
    ap(" */")
    ap("static inline void ck_reorder_token_to_head_major(const float *src, float *dst, int tokens, int heads, int head_dim, int aligned_head_dim) {")
    ap("    if (src == NULL || dst == NULL || tokens <= 0 || heads <= 0 || head_dim <= 0 || aligned_head_dim <= 0) return;")
    ap("    for (int h = 0; h < heads; ++h) {")
    ap("        for (int t = 0; t < tokens; ++t) {")
    ap("            const size_t src_base = ((size_t)t * (size_t)heads + (size_t)h) * (size_t)head_dim;")
    ap("            const size_t dst_base = ((size_t)h * (size_t)tokens + (size_t)t) * (size_t)aligned_head_dim;")
    ap("            for (int d = 0; d < head_dim; ++d) dst[dst_base + (size_t)d] = src[src_base + (size_t)d];")
    ap("            for (int d = head_dim; d < aligned_head_dim; ++d) dst[dst_base + (size_t)d] = 0.0f;")
    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    ap("static inline void ck_reorder_head_to_token_major(const float *src, float *dst, int tokens, int heads, int head_dim, int aligned_head_dim) {")
    ap("    if (src == NULL || dst == NULL || tokens <= 0 || heads <= 0 || head_dim <= 0 || aligned_head_dim <= 0) return;")
    ap("    for (int t = 0; t < tokens; ++t) {")
    ap("        for (int h = 0; h < heads; ++h) {")
    ap("            const size_t src_base = ((size_t)h * (size_t)tokens + (size_t)t) * (size_t)aligned_head_dim;")
    ap("            const size_t dst_base = ((size_t)t * (size_t)heads + (size_t)h) * (size_t)head_dim;")
    ap("            for (int d = 0; d < head_dim; ++d) dst[dst_base + (size_t)d] = src[src_base + (size_t)d];")
    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    ap("int ck_train_memory_diagnostic(const float *oracle_acts, const float *oracle_grads, float tolerance) {")
    ap("    (void)oracle_acts;")
    ap("    (void)oracle_grads;")
    ap("    (void)tolerance;")
    ap("    if (g_memory == NULL) return -1;")
    ap("    g_diag_failed_op_id = -1;")
    ap("    g_diag_failed_canary_idx = -1;")
    ap("    int first = -1;")
    ap("    ck_train_plant_canaries();")
    ap("    if (ck_train_check_canaries(\"after_plant\", &first) != 0) return -10 - first;")
    ap("    float *snap = NULL;")
    ap("    if (CK_WEIGHT_SNAPSHOT_FLOATS > 0) {")
    ap("        snap = (float*)malloc(CK_WEIGHT_SNAPSHOT_FLOATS * sizeof(float));")
    ap("        if (snap == NULL) return -2;")
    ap("        if (ck_train_snapshot_weights(snap, CK_WEIGHT_SNAPSHOT_FLOATS) < 0) { free(snap); return -3; }")
    ap("    }")
    ap("    int fwd = ck_train_forward_step();")
    ap("    if (ck_train_check_canaries(\"after_forward\", &first) != 0) { if (snap) free(snap); return -100 - first; }")
    ap("    if (snap != NULL && ck_train_check_weights_readonly(snap, &first) != 0) { free(snap); return -200 - first; }")
    ap("    int failed_op = -1;")
    ap("    int failed_canary = -1;")
    ap("    int bwd = ck_train_backward_step_trace(&failed_op, &failed_canary);")
    ap("    if (bwd < 0) {")
    ap("        g_diag_failed_op_id = failed_op;")
    ap("        g_diag_failed_canary_idx = failed_canary;")
    ap("        if (snap) free(snap);")
    ap("        return -5000;")
    ap("    }")
    ap("    if (ck_train_check_canaries(\"after_backward\", &first) != 0) { if (snap) free(snap); return -300 - first; }")
    ap("    int opt = ck_train_optimizer_step(1e-3f);")
    ap("    if (ck_train_check_canaries(\"after_optimizer\", &first) != 0) { if (snap) free(snap); return -400 - first; }")
    ap("    if (snap != NULL) free(snap);")
    ap("    return fwd + bwd + opt;")
    ap("}")
    ap("")

    token_input_i32_vars: List[str] = []
    target_input_i32_vars: List[str] = []
    for tid, var in sorted(tvars_i32.items(), key=lambda x: x[1]):
        lt = str(tid).lower()
        if "token" in lt and "target" not in lt:
            token_input_i32_vars.append(var)
        if "target" in lt:
            target_input_i32_vars.append(var)

    grad_vars_f32: List[str] = []
    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        if isinstance(tid, str) and tid.startswith("grad."):
            grad_vars_f32.append(var)
    ap("void ck_train_reset_buffers(void) {")
    ap("    if (g_memory != NULL && g_memory_floats > 0) {")
    ap("        memset(g_memory, 0, g_memory_floats * sizeof(float));")
    ap("    }")
    for var in sorted(set(tvars_i32.values())):
        ap("    memset(%s, 0, sizeof(%s));" % (var, var))
    if opt_pairs:
        ap("    g_opt_step = 0;")
    ap("    g_accum_step = 0;")
    ap("    ck_train_reset_profile();")
    ap("    g_loss_scalar[0] = 0.0f;")
    ap("}")
    ap("")
    ap("int ck_train_init(const float *bump, const int *manifest_sizes, int num_params) {")
    ap("    int alloc_rc = ck_train_alloc();")
    ap("    if (alloc_rc != 0) { return alloc_rc; }")
    ap("    ck_train_reset_buffers();")
    ap("    if (CK_INIT_WEIGHT_COUNT == 0) {")
    ap("        return 0;")
    ap("    }")
    ap("    if (bump == NULL) {")
    ap("        return -1;")
    ap("    }")
    ap("    if (num_params < CK_INIT_WEIGHT_COUNT) {")
    ap("        return -2;")
    ap("    }")
    ap("    const float *cursor = bump;")
    ap("    for (int i = 0; i < CK_INIT_WEIGHT_COUNT; ++i) {")
    ap("        int expected = g_init_weight_numel[i];")
    ap("        int src_n = expected;")
    ap("        int copy_n = expected;")
    ap("        if (manifest_sizes != NULL && manifest_sizes[i] > 0) {")
    ap("            src_n = manifest_sizes[i];")
    ap("            if (src_n < copy_n) copy_n = src_n;")
    ap("        }")
    ap("        if (copy_n > 0) {")
    ap("            memcpy(g_memory + g_init_weight_offsets[i], cursor, (size_t)copy_n * sizeof(float));")
    ap("        }")
    ap("        if (copy_n < expected) {")
    ap("            memset(g_memory + g_init_weight_offsets[i] + copy_n, 0, (size_t)(expected - copy_n) * sizeof(float));")
    ap("        }")
    ap("        cursor += (src_n > 0 ? src_n : expected);")
    ap("    }")
    if has_rope_forward_qk:
        # Do not remove: rope_forward_qk kernels read cos/sin tables directly.
        # Missing/NULL caches can silently skew intermediate activations while
        # keeping final logits deceptively close for tiny-token runs.
        ap("    if (CK_ROPE_ROTARY_DIM > 0) {")
        ap("        rope_precompute_cache_split(g_rope_cos_cache, g_rope_sin_cache, CK_NUM_TOKENS, CK_ROPE_ROTARY_DIM, CK_ROPE_THETA);")
        ap("    }")
    ap("    return CK_INIT_WEIGHT_COUNT;")
    ap("}")
    ap("")
    ap("void ck_zero_grad(void) {")
    ap("    if (g_memory == NULL) return;")
    ap("    if (CK_GRADS_SIZE > 0) {")
    ap("        memset(g_memory + CK_GRADS_OFFSET, 0, CK_GRADS_SIZE * sizeof(float));")
    ap("    }")
    ap("    if (CK_GRAD_ACT_SIZE > 0) {")
    ap("        memset(g_memory + CK_GRAD_ACT_OFFSET, 0, CK_GRAD_ACT_SIZE * sizeof(float));")
    ap("    }")
    ap("}")
    ap("")

    f32_ptr_numel: Dict[str, int] = {}
    for tid, var in tvars_f32.items():
        n = int(tensor_numel.get(tid, 0) or 0)
        if n > 0:
            f32_ptr_numel[var] = n
    for _wname, _gvar, _wvar, mvar, vvar, _numel in opt_pairs:
        n = int(_numel)
        if n > 0:
            f32_ptr_numel[mvar] = n
            f32_ptr_numel[vvar] = n

    tmp_float_pool: List[Tuple[str, str, int]] = []
    for _tid, _var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        if not str(_tid).startswith("tmp."):
            continue
        _n = int(tensor_numel.get(_tid, 0) or 0)
        if _n > 0:
            tmp_float_pool.append((_tid, _var, _n))

    def _pick_tmp_scratch_var(min_numel: int, banned_vars: set[str]) -> Optional[str]:
        if min_numel <= 0:
            return None
        best: Optional[Tuple[int, str]] = None
        for _tid, _var, _n in tmp_float_pool:
            if _var in banned_vars:
                continue
            if _n < min_numel:
                continue
            cand = (_n, _var)
            if best is None or cand < best:
                best = cand
        return best[1] if best is not None else None

    def emit_ops(fn_name: str, ops: List[Dict[str, Any]], phase: str, *, trace_canary: bool = False) -> None:
        if trace_canary:
            ap("int %s(int *failed_op_id, int *first_corrupt_idx) {" % fn_name)
            ap("    if (failed_op_id != NULL) *failed_op_id = -1;")
            ap("    if (first_corrupt_idx != NULL) *first_corrupt_idx = -1;")
            ap("    int first = -1;")
        else:
            ap("int %s(void) {" % fn_name)
        ap("    /* %s ops: %d */" % (phase, len(ops)))
        ap("    int call_count = 0;")
        for op in ops:
            kid = op.get("kernel_id")
            op_id = op.get("op_id")
            op_name = op.get("op")
            if not isinstance(kid, str):
                skipped_ops.append("op_id=%s missing kernel_id" % op_id)
                ap("    /* skip op_id=%s (%s): missing kernel_id */" % (op_id, op_name))
                continue

            resolved = used_decl.get(kid)
            if resolved is None:
                skipped_ops.append("op_id=%s kernel_id=%s no callable declaration" % (op_id, kid))
                ap("    /* skip op_id=%s (%s): kernel `%s` has no callable declaration */" % (op_id, op_name, kid))
                continue

            fn, args, _decl = resolved
            io_inputs, io_outputs, io_weights = _op_io(op, op_by_id)

            # qk_norm_forward is in-place and head-major in kernel API, while IR
            # models token-major out-of-place tensors.
            if kid == "qk_norm_forward":
                q_in_tid = io_inputs.get("q")
                k_in_tid = io_inputs.get("k")
                q_out_tid = io_outputs.get("q") or q_in_tid
                k_out_tid = io_outputs.get("k") or k_in_tid
                q_in_var = tvars_f32.get(q_in_tid or "", "g_dummy_f32")
                k_in_var = tvars_f32.get(k_in_tid or "", "g_dummy_f32")
                q_out_var = tvars_f32.get(q_out_tid or "", q_in_var)
                k_out_var = tvars_f32.get(k_out_tid or "", k_in_var)
                q_gamma_var = tvars_f32.get(io_weights.get("q_gamma", ""), "g_dummy_f32")
                k_gamma_var = tvars_f32.get(io_weights.get("k_gamma", ""), "g_dummy_f32")
                q_numel = int(tensor_numel.get(q_out_tid or q_in_tid or "", 0) or 0)
                k_numel = int(tensor_numel.get(k_out_tid or k_in_tid or "", 0) or 0)
                banned: set[str] = {q_in_var, k_in_var, q_out_var, k_out_var}
                q_scratch = _pick_tmp_scratch_var(q_numel, banned)
                if q_scratch is not None:
                    banned.add(q_scratch)
                k_scratch = _pick_tmp_scratch_var(k_numel, banned)

                ap("    /* op_id=%s op=%s kernel_id=%s (token-major IR <-> head-major kernel bridge) */" % (op_id, op_name, kid))
                if q_numel > 0 and q_out_var != q_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (q_out_var, q_in_var, int(q_numel)))
                if k_numel > 0 and k_out_var != k_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (k_out_var, k_in_var, int(k_numel)))
                if q_scratch and k_scratch:
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_scratch, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_scratch, int(k_numel), op_id))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_out_var, q_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_out_var, k_scratch))
                    ap("    qk_norm_forward(%s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, 1e-5f);" % (
                        q_scratch, k_scratch, q_gamma_var, k_gamma_var
                    ))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_scratch, q_out_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_scratch, k_out_var))
                else:
                    ap("    /* missing tmp.* scratch for qk_norm bridge; fallback direct call */")
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                    ap("    qk_norm_forward(%s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, 1e-5f);" % (
                        q_out_var, k_out_var, q_gamma_var, k_gamma_var
                    ))
                ap("    call_count++;")
                continue

            # rope_forward_qk is in-place and head-major, while IR tensors are token-major.
            # Keep IR-visible tensors token-major by converting around kernel invocation.
            if kid == "rope_forward_qk":
                q_in_tid = io_inputs.get("q")
                k_in_tid = io_inputs.get("k")
                q_out_tid = io_outputs.get("q") or q_in_tid
                k_out_tid = io_outputs.get("k") or k_in_tid
                q_in_var = tvars_f32.get(q_in_tid or "", "g_dummy_f32")
                k_in_var = tvars_f32.get(k_in_tid or "", "g_dummy_f32")
                q_out_var = tvars_f32.get(q_out_tid or "", q_in_var)
                k_out_var = tvars_f32.get(k_out_tid or "", k_in_var)
                q_numel = int(tensor_numel.get(q_out_tid or q_in_tid or "", 0) or 0)
                k_numel = int(tensor_numel.get(k_out_tid or k_in_tid or "", 0) or 0)

                banned: set[str] = {q_in_var, k_in_var, q_out_var, k_out_var}
                q_scratch = _pick_tmp_scratch_var(q_numel, banned)
                if q_scratch is not None:
                    banned.add(q_scratch)
                k_scratch = _pick_tmp_scratch_var(k_numel, banned)

                ap("    /* op_id=%s op=%s kernel_id=%s (IR token-major <-> kernel head-major bridge) */" % (op_id, op_name, kid))
                if q_numel > 0 and q_out_var != q_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (q_out_var, q_in_var, int(q_numel)))
                if k_numel > 0 and k_out_var != k_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (k_out_var, k_in_var, int(k_numel)))

                if q_scratch and k_scratch:
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_scratch, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_scratch, int(k_numel), op_id))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_out_var, q_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_out_var, k_scratch))
                    ap("    rope_forward_qk_with_rotary_dim(%s, %s, g_rope_cos_cache, g_rope_sin_cache, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, 0, CK_ROPE_ROTARY_DIM);" % (
                        q_scratch, k_scratch
                    ))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_scratch, q_out_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_scratch, k_out_var))
                else:
                    ap("    /* missing tmp.* scratch for RoPE bridge; fallback direct call */")
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                    ap("    rope_forward_qk_with_rotary_dim(%s, %s, g_rope_cos_cache, g_rope_sin_cache, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, 0, CK_ROPE_ROTARY_DIM);" % (
                        q_out_var, k_out_var
                    ))
                ap("    call_count++;")
                continue

            # attention_forward_* kernels in this runtime are head-major contracts.
            # IR tensors are token-major, so bridge layouts explicitly here.
            if kid == "attention_forward_causal_head_major_gqa_flash_strided":
                q_tid = io_inputs.get("q")
                k_tid = io_inputs.get("k")
                v_tid = io_inputs.get("v")
                y_tid = io_outputs.get("out") or io_outputs.get("output") or io_outputs.get("y")
                q_var = tvars_f32.get(q_tid or "", "g_dummy_f32")
                k_var = tvars_f32.get(k_tid or "", "g_dummy_f32")
                v_var = tvars_f32.get(v_tid or "", "g_dummy_f32")
                y_var = tvars_f32.get(y_tid or "", "g_dummy_f32")
                q_numel = int(tensor_numel.get(q_tid or "", 0) or 0)
                k_numel = int(tensor_numel.get(k_tid or "", 0) or 0)
                v_numel = int(tensor_numel.get(v_tid or "", 0) or 0)
                y_numel = int(tensor_numel.get(y_tid or "", 0) or 0)

                banned: set[str] = {q_var, k_var, v_var, y_var}
                q_scratch = _pick_tmp_scratch_var(q_numel, banned)
                if q_scratch is not None:
                    banned.add(q_scratch)
                k_scratch = _pick_tmp_scratch_var(k_numel, banned)
                if k_scratch is not None:
                    banned.add(k_scratch)
                v_scratch = _pick_tmp_scratch_var(v_numel, banned)
                if v_scratch is not None:
                    banned.add(v_scratch)
                y_scratch = _pick_tmp_scratch_var(y_numel, banned)

                if q_scratch and k_scratch and v_scratch and y_scratch:
                    ap("    /* op_id=%s op=%s kernel_id=%s (token-major IR -> head-major kernel bridge) */" % (op_id, op_name, kid))
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_var, int(q_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_scratch, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_var, int(k_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_scratch, int(k_numel), op_id))
                    if v_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (v_var, int(v_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (v_scratch, int(v_numel), op_id))
                    if y_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (y_var, int(y_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (y_scratch, int(y_numel), op_id))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_var, q_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_var, k_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (v_var, v_scratch))
                    ap("    attention_forward_causal_head_major_gqa_flash_strided(%s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, CK_NUM_TOKENS);" % (
                        q_scratch, k_scratch, v_scratch, y_scratch
                    ))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (y_scratch, y_var))
                    ap("    call_count++;")
                    continue
                else:
                    ap("    /* op_id=%s op=%s kernel_id=%s: missing tmp.* scratch for layout bridge; falling back to direct kernel call */" % (op_id, op_name, kid))

            # attention_backward_* kernels consume/produce head-major tensors.
            # IR tensors are token-major, so bridge layouts explicitly.
            if kid in ("attention_backward_causal_head_major_gqa", "attention_backward_causal_head_major"):
                dy_tid = io_inputs.get("in_0")
                q_tid = io_inputs.get("in_1")
                k_tid = io_inputs.get("in_2")
                v_tid = io_inputs.get("in_3")
                attn_tid = io_inputs.get("in_4")
                dq_tid = io_outputs.get("d_q")
                dk_tid = io_outputs.get("d_k")
                dv_tid = io_outputs.get("d_v")
                ds_tid = io_outputs.get("d_scores")

                dy_var = tvars_f32.get(dy_tid or "", "g_dummy_f32")
                q_var = tvars_f32.get(q_tid or "", "g_dummy_f32")
                k_var = tvars_f32.get(k_tid or "", "g_dummy_f32")
                v_var = tvars_f32.get(v_tid or "", "g_dummy_f32")
                attn_var = tvars_f32.get(attn_tid or "", "g_dummy_f32")
                dq_var = tvars_f32.get(dq_tid or "", "g_dummy_f32")
                dk_var = tvars_f32.get(dk_tid or "", "g_dummy_f32")
                dv_var = tvars_f32.get(dv_tid or "", "g_dummy_f32")
                ds_var = tvars_f32.get(ds_tid or "", "g_dummy_f32")

                dy_numel = int(tensor_numel.get(dy_tid or "", 0) or 0)
                q_numel = int(tensor_numel.get(q_tid or "", 0) or 0)
                k_numel = int(tensor_numel.get(k_tid or "", 0) or 0)
                v_numel = int(tensor_numel.get(v_tid or "", 0) or 0)
                dq_numel = int(tensor_numel.get(dq_tid or "", 0) or 0)
                dk_numel = int(tensor_numel.get(dk_tid or "", 0) or 0)
                dv_numel = int(tensor_numel.get(dv_tid or "", 0) or 0)
                ds_numel = int(tensor_numel.get(ds_tid or "", 0) or 0)

                banned: set[str] = {dy_var, q_var, k_var, v_var, attn_var, dq_var, dk_var, dv_var, ds_var}
                dy_scratch = _pick_tmp_scratch_var(dy_numel, banned)
                if dy_scratch is not None:
                    banned.add(dy_scratch)
                q_scratch = _pick_tmp_scratch_var(q_numel, banned)
                if q_scratch is not None:
                    banned.add(q_scratch)
                k_scratch = _pick_tmp_scratch_var(k_numel, banned)
                if k_scratch is not None:
                    banned.add(k_scratch)
                v_scratch = _pick_tmp_scratch_var(v_numel, banned)
                if v_scratch is not None:
                    banned.add(v_scratch)
                dq_scratch = _pick_tmp_scratch_var(dq_numel, banned)
                if dq_scratch is not None:
                    banned.add(dq_scratch)
                dk_scratch = _pick_tmp_scratch_var(dk_numel, banned)
                if dk_scratch is not None:
                    banned.add(dk_scratch)
                dv_scratch = _pick_tmp_scratch_var(dv_numel, banned)

                if dy_scratch and q_scratch and k_scratch and v_scratch and dq_scratch and dk_scratch and dv_scratch:
                    ap("    /* op_id=%s op=%s kernel_id=%s (token-major IR <-> head-major kernel bridge) */" % (op_id, op_name, kid))
                    if dy_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dy_var, int(dy_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dy_scratch, int(dy_numel), op_id))
                    if q_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_var, int(q_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_scratch, int(q_numel), op_id))
                    if k_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_var, int(k_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_scratch, int(k_numel), op_id))
                    if v_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (v_var, int(v_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (v_scratch, int(v_numel), op_id))
                    if dq_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_var, int(dq_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_scratch, int(dq_numel), op_id))
                    if dk_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_var, int(dk_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_scratch, int(dk_numel), op_id))
                    if dv_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dv_var, int(dv_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dv_scratch, int(dv_numel), op_id))

                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dy_var, dy_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_var, q_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_var, k_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (v_var, v_scratch))

                    ap("    %s(%s, %s, %s, %s, %s, %s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, CK_NUM_TOKENS);" % (
                        fn, dy_scratch, q_scratch, k_scratch, v_scratch, attn_var, dq_scratch, dk_scratch, dv_scratch, ds_var
                    ))

                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dq_scratch, dq_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dk_scratch, dk_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dv_scratch, dv_var))
                    ap("#if CK_ABLATE_ATTENTION_BACKWARD")
                    ap("    /* Ablation fallback: bypass attention backward outputs with passthrough/zeros. */")
                    if dq_numel > 0:
                        cp = min(int(dq_numel), int(dy_numel))
                        if cp > 0:
                            ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (dq_var, dy_var, int(cp)))
                        if int(dq_numel) > cp:
                            ap("    memset(%s + (size_t)%d, 0, (size_t)%d * sizeof(float));" % (dq_var, int(cp), int(dq_numel - cp)))
                    if dk_numel > 0:
                        ap("    memset(%s, 0, (size_t)%d * sizeof(float));" % (dk_var, int(dk_numel)))
                    if dv_numel > 0:
                        ap("    memset(%s, 0, (size_t)%d * sizeof(float));" % (dv_var, int(dv_numel)))
                    if ds_numel > 0:
                        ap("    memset(%s, 0, (size_t)%d * sizeof(float));" % (ds_var, int(ds_numel)))
                    ap("#endif")
                    ap("    call_count++;")
                    continue
                else:
                    ap("    /* op_id=%s op=%s kernel_id=%s: missing tmp.* scratch for backward layout bridge; falling back to direct kernel call */" % (op_id, op_name, kid))

            # rope_backward_qk expects head-major q/k gradient tensors.
            if kid == "rope_backward_qk_f32":
                dq_out_tid = io_inputs.get("in_0")
                dk_out_tid = io_inputs.get("in_1")
                dq_tid = io_outputs.get("d_q")
                dk_tid = io_outputs.get("d_k")

                dq_out_var = tvars_f32.get(dq_out_tid or "", "g_dummy_f32")
                dk_out_var = tvars_f32.get(dk_out_tid or "", "g_dummy_f32")
                dq_var = tvars_f32.get(dq_tid or "", "g_dummy_f32")
                dk_var = tvars_f32.get(dk_tid or "", "g_dummy_f32")

                dq_out_numel = int(tensor_numel.get(dq_out_tid or "", 0) or 0)
                dk_out_numel = int(tensor_numel.get(dk_out_tid or "", 0) or 0)
                dq_numel = int(tensor_numel.get(dq_tid or "", 0) or 0)
                dk_numel = int(tensor_numel.get(dk_tid or "", 0) or 0)

                banned: set[str] = {dq_out_var, dk_out_var, dq_var, dk_var}
                dq_out_scratch = _pick_tmp_scratch_var(dq_out_numel, banned)
                if dq_out_scratch is not None:
                    banned.add(dq_out_scratch)
                dk_out_scratch = _pick_tmp_scratch_var(dk_out_numel, banned)
                if dk_out_scratch is not None:
                    banned.add(dk_out_scratch)
                dq_scratch = _pick_tmp_scratch_var(dq_numel, banned)
                if dq_scratch is not None:
                    banned.add(dq_scratch)
                dk_scratch = _pick_tmp_scratch_var(dk_numel, banned)

                if dq_out_scratch and dk_out_scratch and dq_scratch and dk_scratch:
                    ap("    /* op_id=%s op=%s kernel_id=%s (token-major IR <-> head-major kernel bridge) */" % (op_id, op_name, kid))
                    if dq_out_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_out_var, int(dq_out_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_out_scratch, int(dq_out_numel), op_id))
                    if dk_out_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_out_var, int(dk_out_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_out_scratch, int(dk_out_numel), op_id))
                    if dq_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_var, int(dq_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_scratch, int(dq_numel), op_id))
                    if dk_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_var, int(dk_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_scratch, int(dk_numel), op_id))

                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dq_out_var, dq_out_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dk_out_var, dk_out_scratch))
                    ap("    rope_backward_qk(%s, %s, %s, %s, g_rope_cos_cache, g_rope_sin_cache, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, 0);" % (
                        dq_out_scratch, dk_out_scratch, dq_scratch, dk_scratch
                    ))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dq_scratch, dq_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dk_scratch, dk_var))
                    ap("#if CK_ABLATE_ROPE_BACKWARD_QK")
                    ap("    /* Ablation fallback: treat RoPE backward as identity for dq/dk. */")
                    if dq_numel > 0:
                        cp = min(int(dq_numel), int(dq_out_numel))
                        if cp > 0:
                            ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (dq_var, dq_out_var, int(cp)))
                        if int(dq_numel) > cp:
                            ap("    memset(%s + (size_t)%d, 0, (size_t)%d * sizeof(float));" % (dq_var, int(cp), int(dq_numel - cp)))
                    if dk_numel > 0:
                        cp = min(int(dk_numel), int(dk_out_numel))
                        if cp > 0:
                            ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (dk_var, dk_out_var, int(cp)))
                        if int(dk_numel) > cp:
                            ap("    memset(%s + (size_t)%d, 0, (size_t)%d * sizeof(float));" % (dk_var, int(cp), int(dk_numel - cp)))
                    ap("#endif")
                    ap("    call_count++;")
                    continue
                else:
                    ap("    /* op_id=%s op=%s kernel_id=%s: missing tmp.* scratch for rope backward layout bridge; falling back to direct kernel call */" % (op_id, op_name, kid))

            # qk_norm_backward expects head-major q/k tensors for grads + activations.
            if kid == "qk_norm_backward_f32":
                dq_out_tid = io_inputs.get("in_0")
                dk_out_tid = io_inputs.get("in_1")
                q_in_tid = io_inputs.get("in_2")
                k_in_tid = io_inputs.get("in_3")
                q_gamma_tid = io_inputs.get("in_4")
                k_gamma_tid = io_inputs.get("in_5")
                dq_tid = io_outputs.get("d_q")
                dk_tid = io_outputs.get("d_k")
                dq_gamma_tid = io_outputs.get("d_q_gamma")
                dk_gamma_tid = io_outputs.get("d_k_gamma")

                dq_out_var = tvars_f32.get(dq_out_tid or "", "g_dummy_f32")
                dk_out_var = tvars_f32.get(dk_out_tid or "", "g_dummy_f32")
                q_in_var = tvars_f32.get(q_in_tid or "", "g_dummy_f32")
                k_in_var = tvars_f32.get(k_in_tid or "", "g_dummy_f32")
                q_gamma_var = tvars_f32.get(q_gamma_tid or "", "g_dummy_f32")
                k_gamma_var = tvars_f32.get(k_gamma_tid or "", "g_dummy_f32")
                dq_var = tvars_f32.get(dq_tid or "", "g_dummy_f32")
                dk_var = tvars_f32.get(dk_tid or "", "g_dummy_f32")
                dq_gamma_var = tvars_f32.get(dq_gamma_tid or "", "g_dummy_f32")
                dk_gamma_var = tvars_f32.get(dk_gamma_tid or "", "g_dummy_f32")

                dq_out_numel = int(tensor_numel.get(dq_out_tid or "", 0) or 0)
                dk_out_numel = int(tensor_numel.get(dk_out_tid or "", 0) or 0)
                q_in_numel = int(tensor_numel.get(q_in_tid or "", 0) or 0)
                k_in_numel = int(tensor_numel.get(k_in_tid or "", 0) or 0)
                dq_numel = int(tensor_numel.get(dq_tid or "", 0) or 0)
                dk_numel = int(tensor_numel.get(dk_tid or "", 0) or 0)
                dq_gamma_numel = int(tensor_numel.get(dq_gamma_tid or "", 0) or 0)
                dk_gamma_numel = int(tensor_numel.get(dk_gamma_tid or "", 0) or 0)

                banned: set[str] = {dq_out_var, dk_out_var, q_in_var, k_in_var, q_gamma_var, k_gamma_var, dq_var, dk_var, dq_gamma_var, dk_gamma_var}
                dq_out_scratch = _pick_tmp_scratch_var(dq_out_numel, banned)
                if dq_out_scratch is not None:
                    banned.add(dq_out_scratch)
                dk_out_scratch = _pick_tmp_scratch_var(dk_out_numel, banned)
                if dk_out_scratch is not None:
                    banned.add(dk_out_scratch)
                q_in_scratch = _pick_tmp_scratch_var(q_in_numel, banned)
                if q_in_scratch is not None:
                    banned.add(q_in_scratch)
                k_in_scratch = _pick_tmp_scratch_var(k_in_numel, banned)
                if k_in_scratch is not None:
                    banned.add(k_in_scratch)
                dq_scratch = _pick_tmp_scratch_var(dq_numel, banned)
                if dq_scratch is not None:
                    banned.add(dq_scratch)
                dk_scratch = _pick_tmp_scratch_var(dk_numel, banned)

                if dq_out_scratch and dk_out_scratch and q_in_scratch and k_in_scratch and dq_scratch and dk_scratch:
                    ap("    /* op_id=%s op=%s kernel_id=%s (token-major IR <-> head-major kernel bridge) */" % (op_id, op_name, kid))
                    if dq_out_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_out_var, int(dq_out_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_out_scratch, int(dq_out_numel), op_id))
                    if dk_out_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_out_var, int(dk_out_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_out_scratch, int(dk_out_numel), op_id))
                    if q_in_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_in_var, int(q_in_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_in_scratch, int(q_in_numel), op_id))
                    if k_in_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_in_var, int(k_in_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_in_scratch, int(k_in_numel), op_id))
                    if dq_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_var, int(dq_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dq_scratch, int(dq_numel), op_id))
                    if dk_numel > 0:
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_var, int(dk_numel), op_id))
                        ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (dk_scratch, int(dk_numel), op_id))

                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dq_out_var, dq_out_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dk_out_var, dk_out_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (q_in_var, q_in_scratch))
                    ap("    ck_reorder_token_to_head_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (k_in_var, k_in_scratch))

                    ap("    qk_norm_backward(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, 1e-5f);" % (
                        dq_out_scratch, dk_out_scratch, q_in_scratch, k_in_scratch, q_gamma_var, k_gamma_var,
                        dq_scratch, dk_scratch, dq_gamma_var, dk_gamma_var
                    ))

                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dq_scratch, dq_var))
                    ap("    ck_reorder_head_to_token_major(%s, %s, CK_NUM_TOKENS, CK_ROPE_NUM_KV_HEADS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM);" % (dk_scratch, dk_var))
                    ap("#if CK_ABLATE_QK_NORM_BACKWARD")
                    ap("    /* Ablation fallback: treat qk_norm backward as identity for dq/dk and zero gamma grads. */")
                    if dq_numel > 0:
                        cp = min(int(dq_numel), int(dq_out_numel))
                        if cp > 0:
                            ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (dq_var, dq_out_var, int(cp)))
                        if int(dq_numel) > cp:
                            ap("    memset(%s + (size_t)%d, 0, (size_t)%d * sizeof(float));" % (dq_var, int(cp), int(dq_numel - cp)))
                    if dk_numel > 0:
                        cp = min(int(dk_numel), int(dk_out_numel))
                        if cp > 0:
                            ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (dk_var, dk_out_var, int(cp)))
                        if int(dk_numel) > cp:
                            ap("    memset(%s + (size_t)%d, 0, (size_t)%d * sizeof(float));" % (dk_var, int(cp), int(dk_numel - cp)))
                    if dq_gamma_numel > 0:
                        ap("    memset(%s, 0, (size_t)%d * sizeof(float));" % (dq_gamma_var, int(dq_gamma_numel)))
                    if dk_gamma_numel > 0:
                        ap("    memset(%s, 0, (size_t)%d * sizeof(float));" % (dk_gamma_var, int(dk_gamma_numel)))
                    ap("#endif")
                    ap("    call_count++;")
                    continue
                else:
                    ap("    /* op_id=%s op=%s kernel_id=%s: missing tmp.* scratch for qk_norm backward layout bridge; falling back to direct kernel call */" % (op_id, op_name, kid))

            op_plan = exec_plan_by_op.get(int(op_id)) if isinstance(op_id, int) else None
            # gemm_backward_f32 uses op-local inferred dims from IR tensor sizes.
            # Avoid generic defaults here: wrong aligned_out can corrupt
            # tmp.grad.weight.* and trip canary/heap checks.
            gemm_dims = None
            if kid == "gemm_backward_f32":
                # Keep backward dims sourced from IR tensor numel map only.
                # Plan metadata is advisory for dispatch, not numeric kernel contracts.
                gemm_dims = _infer_gemm_backward_dims(io_inputs, io_outputs, tensors, tensor_numel, cfg)
            gemm_fwd_mnk = None
            if kid == "gemm_blocked_serial":
                gemm_fwd_mnk = _exec_plan_shape_mnk(op_plan) or _infer_gemm_forward_mnk(op, cfg)
            swiglu_dim: Optional[int] = None
            if kid in ("swiglu_forward", "swiglu_forward_exact", "swiglu_backward", "swiglu_backward_exact"):
                # swiglu kernels expect `dim` as per-token hidden width (D), where
                # input is [T, 2D], output/grad_output is [T, D], d_input is [T, 2D].
                tok = int(cfg.get("train_tokens", cfg.get("tokens", 1)) or 1)
                tok = max(1, tok)

                def _num(tid: Optional[str]) -> int:
                    if not isinstance(tid, str):
                        return 0
                    return int(tensor_numel.get(tid, 0) or 0)

                cands: List[int] = []

                if kid in ("swiglu_forward", "swiglu_forward_exact"):
                    # Forward: prefer output [T,D], fallback to input [T,2D].
                    for key in ("out", "y"):
                        n = _num(io_outputs.get(key))
                        if n > 0 and (n % tok) == 0:
                            cands.append(n // tok)
                    for key in ("input", "x", "in_0"):
                        n = _num(io_inputs.get(key))
                        denom = 2 * tok
                        if n > 0 and denom > 0 and (n % denom) == 0:
                            cands.append(n // denom)
                else:
                    # Backward: prefer grad_output [T,D], fallback to d_input/input [T,2D].
                    for key in ("d_output", "grad_output", "dy", "in_0"):
                        n = _num(io_inputs.get(key))
                        if n > 0 and (n % tok) == 0:
                            cands.append(n // tok)
                    for key in ("input", "x", "in_1"):
                        n = _num(io_inputs.get(key))
                        denom = 2 * tok
                        if n > 0 and denom > 0 and (n % denom) == 0:
                            cands.append(n // denom)
                    for key in ("d_input", "dx"):
                        n = _num(io_outputs.get(key))
                        denom = 2 * tok
                        if n > 0 and denom > 0 and (n % denom) == 0:
                            cands.append(n // denom)

                cands = [int(v) for v in cands if int(v) > 0]
                if cands:
                    swiglu_dim = int(min(cands))
                else:
                    # Legacy fallback for unusual IR wiring.
                    raw: List[int] = []
                    for tid in list(io_outputs.values()) + list(io_inputs.values()):
                        n = _num(tid)
                        if n > 0:
                            raw.append(n)
                    if raw:
                        swiglu_dim = int(min(raw))

            call_args: List[str] = []
            ptr_fallback_state: Dict[str, int] = {}
            bounds_checks: List[Tuple[str, int]] = []
            for atype, aname in args:
                is_ptr = "*" in atype
                if is_ptr:
                    if kid == "gemm_blocked_serial":
                        lname = str(aname).lower()
                        mapped: Optional[str] = None
                        if lname in ("a", "x", "in", "input", "lhs"):
                            mapped = tvars_f32.get(io_inputs.get("input", ""))
                        elif lname in ("b", "w", "weight", "rhs"):
                            mapped = tvars_f32.get(io_weights.get("W", ""))
                        elif "bias" in lname:
                            mapped = tvars_f32.get(io_weights.get("bias", ""))
                            if mapped is None:
                                # Some GEMM ops (e.g. logits) intentionally have no bias.
                                mapped = "NULL"
                        elif lname in ("c", "y", "out", "output", "dst"):
                            mapped = tvars_f32.get(io_outputs.get("y", ""))
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    elif kid == "gemm_backward_f32":
                        lname = str(aname).lower()
                        mapped = None
                        # Prefer positional pointer mapping first because registry declarations
                        # may use opaque arg names after normalization.
                        ptr_idx = int(ptr_fallback_state.get("__gemm_bwd_ptr_idx", 0) or 0)
                        ptr_fallback_state["__gemm_bwd_ptr_idx"] = int(ptr_idx + 1)
                        if ptr_idx == 0:
                            mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                        elif ptr_idx == 1:
                            mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                        elif ptr_idx == 2:
                            mapped = tvars_f32.get(io_inputs.get("in_2", ""))
                        elif ptr_idx == 3:
                            mapped = tvars_f32.get(io_outputs.get("d_input", ""))
                        elif ptr_idx == 4:
                            mapped = tvars_f32.get(io_outputs.get("d_W", ""))
                        elif ptr_idx == 5:
                            mapped = tvars_f32.get(io_outputs.get("d_bias", "")) or "NULL"
                            if mapped != "NULL":
                                mapped = "(CK_ABLATE_GEMM_BACKWARD_BIAS ? NULL : %s)" % mapped
                        # Fallback name-based mapping for unexpected signatures.
                        if mapped is None:
                            if lname in ("d_output", "grad_output", "dy"):
                                mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                            elif lname in ("input", "x", "in"):
                                mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                            elif lname in ("w", "weight"):
                                mapped = tvars_f32.get(io_inputs.get("in_2", ""))
                            elif lname in ("d_input", "dx"):
                                mapped = tvars_f32.get(io_outputs.get("d_input", ""))
                            elif lname in ("d_w", "dw", "d_weight", "dweight"):
                                mapped = tvars_f32.get(io_outputs.get("d_W", ""))
                            elif lname in ("d_b", "db", "d_bias", "dbias", "bias"):
                                mapped = tvars_f32.get(io_outputs.get("d_bias", ""))
                                if mapped is None:
                                    mapped = "NULL"
                                if mapped != "NULL":
                                    mapped = "(CK_ABLATE_GEMM_BACKWARD_BIAS ? NULL : %s)" % mapped
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    elif kid in ("attention_backward_causal_head_major_gqa", "attention_backward_causal_head_major"):
                        lname = str(aname).lower()
                        mapped: Optional[str] = None
                        if lname in ("d_output", "grad_output", "dy"):
                            mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                        elif lname == "q":
                            mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                        elif lname == "k":
                            mapped = tvars_f32.get(io_inputs.get("in_2", ""))
                        elif lname == "v":
                            mapped = tvars_f32.get(io_inputs.get("in_3", ""))
                        elif lname in ("attn_weights", "weights", "softmax_weights"):
                            mapped = tvars_f32.get(io_inputs.get("in_4", ""))
                        elif lname in ("d_q", "dq"):
                            mapped = tvars_f32.get(io_outputs.get("d_q", ""))
                        elif lname in ("d_k", "dk"):
                            mapped = tvars_f32.get(io_outputs.get("d_k", ""))
                        elif lname in ("d_v", "dv"):
                            mapped = tvars_f32.get(io_outputs.get("d_v", ""))
                        elif lname in ("d_scores", "dscores"):
                            mapped = tvars_f32.get(io_outputs.get("d_scores", ""))
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    elif kid == "rmsnorm_backward":
                        # Keep backward argument binding stable for:
                        # rmsnorm_backward(d_output, input, gamma, rstd_cache, d_input, d_gamma, ...)
                        lname = str(aname).lower()
                        mapped = None
                        if lname in ("d_output", "grad_output", "dy"):
                            mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                        elif lname in ("input", "x"):
                            mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                        elif lname in ("gamma", "weight", "w"):
                            mapped = tvars_f32.get(io_inputs.get("in_2", ""))
                        elif lname in ("rstd_cache", "rstd", "inv_rms", "inv_rms_cache"):
                            mapped = tvars_f32.get(io_inputs.get("in_3", ""))
                        elif lname in ("d_input", "dx"):
                            mapped = tvars_f32.get(io_outputs.get("d_input", ""))
                        elif lname in ("d_gamma", "dgamma"):
                            mapped = tvars_f32.get(io_outputs.get("d_gamma", ""))
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    elif kid == "qk_norm_backward_f32":
                        # qk_norm_backward(d_q_out, d_k_out, q_in, k_in, q_gamma, k_gamma, d_q_in, d_k_in, d_q_gamma, d_k_gamma, ...)
                        lname = str(aname).lower()
                        mapped = None
                        if lname in ("d_q_out", "dq_out", "d_q", "dq"):
                            mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                        elif lname in ("d_k_out", "dk_out"):
                            mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                        elif lname in ("q_in", "q"):
                            mapped = tvars_f32.get(io_inputs.get("in_2", ""))
                        elif lname in ("k_in", "k"):
                            mapped = tvars_f32.get(io_inputs.get("in_3", ""))
                        elif lname in ("q_gamma", "gamma_q"):
                            mapped = tvars_f32.get(io_inputs.get("in_4", ""))
                        elif lname in ("k_gamma", "gamma_k"):
                            mapped = tvars_f32.get(io_inputs.get("in_5", ""))
                        elif lname in ("d_q_in", "dq_in"):
                            mapped = tvars_f32.get(io_outputs.get("d_q", ""))
                        elif lname in ("d_k_in", "dk_in"):
                            mapped = tvars_f32.get(io_outputs.get("d_k", ""))
                        elif lname in ("d_q_gamma", "dq_gamma", "d_gamma_q"):
                            mapped = tvars_f32.get(io_outputs.get("d_q_gamma", ""))
                        elif lname in ("d_k_gamma", "dk_gamma", "d_gamma_k"):
                            mapped = tvars_f32.get(io_outputs.get("d_k_gamma", ""))
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    elif kid == "rope_backward_qk_f32":
                        # rope_backward_qk(d_q_out, d_k_out, d_q, d_k, cos_cache, sin_cache, ...)
                        lname = str(aname).lower()
                        mapped = None
                        if lname in ("d_q_out", "dq_out"):
                            mapped = tvars_f32.get(io_inputs.get("in_0", ""))
                        elif lname in ("d_k_out", "dk_out"):
                            mapped = tvars_f32.get(io_inputs.get("in_1", ""))
                        elif lname in ("d_q", "dq"):
                            mapped = tvars_f32.get(io_outputs.get("d_q", ""))
                        elif lname in ("d_k", "dk"):
                            mapped = tvars_f32.get(io_outputs.get("d_k", ""))
                        elif lname in ("cos_cache", "cos"):
                            mapped = "g_rope_cos_cache"
                        elif lname in ("sin_cache", "sin"):
                            mapped = "g_rope_sin_cache"
                        if mapped is not None:
                            expr = mapped
                        else:
                            expr = _choose_tensor_for_ptr(
                                atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                                fallback_state=ptr_fallback_state,
                            )
                    else:
                        expr = _choose_tensor_for_ptr(
                            atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32,
                            fallback_state=ptr_fallback_state,
                        )
                    call_args.append(expr)
                    if ("float" in atype) and ("*" in atype):
                        n = f32_ptr_numel.get(expr)
                        if isinstance(n, int) and n > 0:
                            bounds_checks.append((expr, int(n)))
                else:
                    lname = str(aname).lower()
                    if gemm_fwd_mnk is not None:
                        m_i, n_i, k_i = gemm_fwd_mnk
                        if lname in ("m", "rows"):
                            call_args.append(str(int(m_i)))
                            continue
                        if lname in ("n", "cols", "out_dim"):
                            call_args.append(str(int(n_i)))
                            continue
                        if lname in ("k", "inner", "in_dim"):
                            call_args.append(str(int(k_i)))
                            continue
                    if swiglu_dim is not None and lname in ("dim", "hidden_dim", "intermediate_dim"):
                        call_args.append(str(int(swiglu_dim)))
                        continue
                    if ("numel" in lname) or ("size_t" in atype):
                        call_args.append(_choose_numel_expr(aname, io_inputs, io_outputs, io_weights, tensor_numel))
                    elif gemm_dims is not None and "aligned_in" in lname:
                        call_args.append(str(int(gemm_dims.get("aligned_in", 1))))
                    elif gemm_dims is not None and "aligned_out" in lname:
                        call_args.append(str(int(gemm_dims.get("aligned_out", 1))))
                    else:
                        call_args.append(_arg_scalar_expr(atype, aname, cfg))

            ap("    /* op_id=%s op=%s kernel_id=%s */" % (op_id, op_name, kid))
            disp_note = _exec_plan_dispatch_comment(op_plan)
            if disp_note:
                ap("    /* dispatch_plan: %s */" % disp_note)
            seen_bounds: set[str] = set()
            for expr, n in bounds_checks:
                if expr in seen_bounds:
                    continue
                seen_bounds.add(expr)
                ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (expr, int(n), op_id))
            call_fn = fn
            if kid == "gemm_blocked_serial":
                call_fn = "gemm_blocked_serial_train_parallel_dispatch"
            elif kid == "gemm_backward_f32":
                call_fn = "gemm_backward_f32_train_parallel_dispatch_v2"
            ap("    %s(%s);" % (call_fn, ", ".join(call_args)))
            if phase.startswith("backward"):
                inj_tid: Optional[str] = None
                for _k, _tid in _sorted_pool_items(io_outputs):
                    if isinstance(_tid, str) and _tid in tvars_f32 and int(tensor_numel.get(_tid, 0) or 0) > 0:
                        inj_tid = _tid
                        break
                if inj_tid is not None:
                    inj_var = tvars_f32[inj_tid]
                    inj_numel = int(tensor_numel.get(inj_tid, 0) or 0)
                    ap("#if CK_RUNTIME_FAULT_INJECT")
                    ap("    if ((CK_RUNTIME_FAULT_INJECT != 0) && (CK_FAULT_INJECT_OP_ID == %s)) {" % op_id)
                    ap("        /* Deliberate +1 write for diagnostics: hit tail canary deterministically. */")
                    ap("        g_memory[CK_TRAIN_TOTAL_FLOATS] = 123.0f;")
                    ap("    }")
                    ap("#endif")
            ap("    call_count++;")
            if trace_canary:
                ap("    if (ck_train_check_canaries(\"trace_after_op_%s\", &first) != 0) {" % op_id)
                ap("        if (failed_op_id != NULL) *failed_op_id = %s;" % op_id)
                ap("        if (first_corrupt_idx != NULL) *first_corrupt_idx = first;")
                ap("        return -2;")
                ap("    }")

        ap("    return call_count;")
        ap("}")
        ap("")

    emit_ops("ck_train_forward_step", forward_ops, "forward")
    emit_ops("ck_train_backward_step", backward_ops, "backward")
    emit_ops("ck_train_backward_step_trace", backward_ops, "backward_trace", trace_canary=True)

    ap("int ck_train_optimizer_step(float lr) {")
    if "adamw_update_f32" not in used_decl:
        ap("    (void)lr;")
        ap("    /* adamw_update_f32 is unavailable; optimizer kernel call path is disabled. */")
        ap("    return 0;")
    elif not opt_pairs:
        ap("    (void)lr;")
        ap("    /* No grad.weight.* tensors available for optimizer updates. */")
        ap("    return 0;")
    else:
        ap("    const float beta1 = 0.9f;")
        ap("    const float beta2 = 0.999f;")
        ap("    const float eps = 1e-8f;")
        ap("    const float weight_decay = 0.01f;")
        ap("    const float max_grad_norm = (float)CK_MAX_GRAD_NORM;")
        ap("    int call_count = 0;")
        ap("    /*")
        ap("     * Match PyTorch grad-accum semantics: optimizer consumes averaged")
        ap("     * gradient over the current accumulation window, not the raw sum.")
        ap("     */")
        ap("    int accum_denom = (g_accum_step > 0) ? g_accum_step : CK_GRAD_ACCUM_STEPS;")
        ap("    if (accum_denom < 1) accum_denom = 1;")
        ap("    const float accum_scale = 1.0f / (float)accum_denom;")
        ap("    if (accum_denom > 1) {")
        for _wname, gvar, _wvar, _mvar, _vvar, numel in opt_pairs:
            n_expr = str(int(numel))
            ap("        for (size_t gi = 0; gi < (size_t)%s; ++gi) { %s[gi] *= accum_scale; }" % (n_expr, gvar))
        ap("    }")
        ap("    g_opt_step += 1;")
        if skipped_opt_non_fp32:
            ap("    /* skipped non-fp32 optimizer params: %d */" % len(skipped_opt_non_fp32))

        if "adamw_clip_update_multi_f32" in used_decl and len(opt_pairs) > 0:
            ap("    /* Fused multi-tensor AdamW + per-tensor norm clip update. */")
            ap("    float *grads[%d] = {%s};" % (len(opt_pairs), ", ".join([gvar for (_wname, gvar, _wvar, _mvar, _vvar, _numel) in opt_pairs])))
            ap("    float *weights[%d] = {%s};" % (len(opt_pairs), ", ".join([wvar for (_wname, _gvar, wvar, _mvar, _vvar, _numel) in opt_pairs])))
            ap("    float *m_states[%d] = {%s};" % (len(opt_pairs), ", ".join([mvar for (_wname, _gvar, _wvar, mvar, _vvar, _numel) in opt_pairs])))
            ap("    float *v_states[%d] = {%s};" % (len(opt_pairs), ", ".join([vvar for (_wname, _gvar, _wvar, _mvar, vvar, _numel) in opt_pairs])))
            ap("    size_t numels[%d] = {%s};" % (len(opt_pairs), ", ".join(["(size_t)%d" % int(numel) for (_wname, _gvar, _wvar, _mvar, _vvar, numel) in opt_pairs])))
            ap("    adamw_clip_update_multi_f32(grads, weights, m_states, v_states, numels, %d, lr, beta1, beta2, eps, weight_decay, max_grad_norm, g_opt_step);" % len(opt_pairs))
            ap("    call_count += %d;" % len(opt_pairs))
        else:
            for wname, gvar, wvar, mvar, vvar, numel in opt_pairs:
                n_expr = str(int(numel))
                ap("    /* AdamW update: %s */" % wname)
                if "gradient_clip_norm_f32" in used_decl:
                    ap("    gradient_clip_norm_f32(%s, (size_t)%s, max_grad_norm);" % (gvar, n_expr))
                ap("    adamw_update_f32(%s, %s, %s, %s, (size_t)%s, lr, beta1, beta2, eps, weight_decay, g_opt_step);" % (
                    gvar, wvar, mvar, vvar, n_expr
                ))
                ap("    call_count++;")
        ap("    return call_count;")
    ap("}")
    ap("")
    ap("int ck_train_flush_optimizer(float lr) {")
    ap("    if (g_accum_step <= 0) return 0;")
    ap("    double t_opt0 = ck_now_ms();")
    ap("    int opt = ck_train_optimizer_step(lr);")
    ap("    double t_opt1 = ck_now_ms();")
    ap("    double opt_ms = t_opt1 - t_opt0;")
    ap("    g_accum_step = 0;")
    ap("    ck_zero_grad();")
    ap("    g_last_step_ms = opt_ms;")
    ap("    g_last_fwd_ms = 0.0;")
    ap("    g_last_bwd_ms = 0.0;")
    ap("    g_last_opt_ms = opt_ms;")
    ap("    g_last_fwd_calls = 0;")
    ap("    g_last_bwd_calls = 0;")
    ap("    g_last_opt_calls = opt;")
    ap("    g_last_opt_applied = (opt > 0) ? 1 : 0;")
    ap("    g_total_step_ms += opt_ms;")
    ap("    g_total_opt_ms += opt_ms;")
    ap("    if (opt > 0) g_profile_optimizer_steps += 1;")
    ap("    return opt;")
    ap("}")
    ap("")
    ap("int ck_train_get_accum_counter(void) { return g_accum_step; }")
    ap("int ck_train_get_accum_steps(void) { return CK_GRAD_ACCUM_STEPS; }")
    ap("int ck_train_set_accum_counter(int accum_step) {")
    ap("    if (accum_step < 0) accum_step = 0;")
    ap("    if (CK_GRAD_ACCUM_STEPS > 0) accum_step = accum_step % CK_GRAD_ACCUM_STEPS;")
    ap("    g_accum_step = accum_step;")
    ap("    return g_accum_step;")
    ap("}")
    ap("int ck_train_get_opt_step(void) { return g_opt_step; }")
    ap("int ck_train_set_opt_step(int opt_step) {")
    ap("    if (opt_step < 0) opt_step = 0;")
    ap("    g_opt_step = opt_step;")
    ap("    return g_opt_step;")
    ap("}")
    ap("void ck_train_reset_profile(void) {")
    ap("    g_profile_steps = 0;")
    ap("    g_profile_optimizer_steps = 0;")
    ap("    g_last_step_ms = 0.0;")
    ap("    g_last_fwd_ms = 0.0;")
    ap("    g_last_bwd_ms = 0.0;")
    ap("    g_last_opt_ms = 0.0;")
    ap("    g_last_fwd_calls = 0;")
    ap("    g_last_bwd_calls = 0;")
    ap("    g_last_opt_calls = 0;")
    ap("    g_last_opt_applied = 0;")
    ap("    g_total_step_ms = 0.0;")
    ap("    g_total_fwd_ms = 0.0;")
    ap("    g_total_bwd_ms = 0.0;")
    ap("    g_total_opt_ms = 0.0;")
    ap("}")
    ap("int ck_train_get_last_step_profile(double *step_ms, double *fwd_ms, double *bwd_ms, double *opt_ms, int *fwd_calls, int *bwd_calls, int *opt_calls, int *opt_applied) {")
    ap("    if (step_ms != NULL) *step_ms = g_last_step_ms;")
    ap("    if (fwd_ms != NULL) *fwd_ms = g_last_fwd_ms;")
    ap("    if (bwd_ms != NULL) *bwd_ms = g_last_bwd_ms;")
    ap("    if (opt_ms != NULL) *opt_ms = g_last_opt_ms;")
    ap("    if (fwd_calls != NULL) *fwd_calls = g_last_fwd_calls;")
    ap("    if (bwd_calls != NULL) *bwd_calls = g_last_bwd_calls;")
    ap("    if (opt_calls != NULL) *opt_calls = g_last_opt_calls;")
    ap("    if (opt_applied != NULL) *opt_applied = g_last_opt_applied;")
    ap("    return 0;")
    ap("}")
    ap("int ck_train_get_cumulative_profile(double *total_ms, double *fwd_ms, double *bwd_ms, double *opt_ms, int *steps, int *optimizer_steps) {")
    ap("    if (total_ms != NULL) *total_ms = g_total_step_ms;")
    ap("    if (fwd_ms != NULL) *fwd_ms = g_total_fwd_ms;")
    ap("    if (bwd_ms != NULL) *bwd_ms = g_total_bwd_ms;")
    ap("    if (opt_ms != NULL) *opt_ms = g_total_opt_ms;")
    ap("    if (steps != NULL) *steps = g_profile_steps;")
    ap("    if (optimizer_steps != NULL) *optimizer_steps = g_profile_optimizer_steps;")
    ap("    return 0;")
    ap("}")
    ap("")

    ap("int ck_train_set_batch(const int32_t *token_ids, const int32_t *targets) {")
    ap("    if (token_ids != NULL) {")
    if token_input_i32_vars:
        for var in sorted(set(token_input_i32_vars)):
            ap("        memcpy(%s, token_ids, sizeof(int32_t) * CK_NUM_TOKENS);" % var)
    else:
        ap("        memcpy(g_dummy_i32, token_ids, sizeof(int32_t) * CK_NUM_TOKENS);")
    ap("    }")
    ap("    if (targets != NULL) {")
    if target_input_i32_vars:
        for var in sorted(set(target_input_i32_vars)):
            ap("        memcpy(%s, targets, sizeof(int32_t) * CK_NUM_TOKENS);" % var)
    else:
        ap("        memcpy(g_dummy_i32, targets, sizeof(int32_t) * CK_NUM_TOKENS);")
    ap("    }")
    ap("    return 0;")
    ap("}")
    ap("")

    ap("int ck_train_step(const int32_t *token_ids, const int32_t *targets, float *loss_out, float lr) {")
    ap("    double t_step0 = ck_now_ms();")
    ap("    ck_train_set_batch(token_ids, targets);")
    ap("    /*")
    ap("     * Grad-accum window contract (must match PyTorch):")
    ap("     * - zero grads once at window start")
    ap("     * - accumulate forward/backward across micro-steps")
    ap("     * - run optimizer exactly at accumulation boundary")
    ap("     */")
    ap("    if (g_accum_step <= 0) {")
    ap("        /* Window start: clear gradients before first micro-step. */")
    ap("        ck_zero_grad();")
    ap("    }")
    ap("    int first = -1;")
    ap("    if (CK_RUNTIME_CANARY_CHECKS) ck_train_plant_canaries();")
    ap("    double t_fwd0 = ck_now_ms();")
    ap("    int fwd = ck_train_forward_step();")
    ap("    double t_fwd1 = ck_now_ms();")
    ap("    if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_forward\", &first) != 0) return -1000 - first;")
    ap("    double t_bwd0 = ck_now_ms();")
    ap("    int bwd = ck_train_backward_step();")
    ap("    double t_bwd1 = ck_now_ms();")
    ap("    if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_backward\", &first) != 0) return -1100 - first;")
    ap("    g_accum_step += 1;")
    ap("    int opt = 0;")
    ap("    int opt_applied = 0;")
    ap("    double opt_ms = 0.0;")
    ap("    if (g_accum_step >= CK_GRAD_ACCUM_STEPS) {")
    ap("        /* Boundary reached: apply one optimizer update per window. */")
    ap("        double t_opt0 = ck_now_ms();")
    ap("        opt = ck_train_optimizer_step(lr);")
    ap("        double t_opt1 = ck_now_ms();")
    ap("        opt_ms = t_opt1 - t_opt0;")
    ap("        opt_applied = 1;")
    ap("        g_accum_step = 0;")
    ap("        if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_optimizer\", &first) != 0) return -1200 - first;")
    ap("    }")
    ap("    double t_step1 = ck_now_ms();")
    ap("    g_last_step_ms = t_step1 - t_step0;")
    ap("    g_last_fwd_ms = t_fwd1 - t_fwd0;")
    ap("    g_last_bwd_ms = t_bwd1 - t_bwd0;")
    ap("    g_last_opt_ms = opt_ms;")
    ap("    g_last_fwd_calls = fwd;")
    ap("    g_last_bwd_calls = bwd;")
    ap("    g_last_opt_calls = opt;")
    ap("    g_last_opt_applied = opt_applied;")
    ap("    g_profile_steps += 1;")
    ap("    g_total_step_ms += g_last_step_ms;")
    ap("    g_total_fwd_ms += g_last_fwd_ms;")
    ap("    g_total_bwd_ms += g_last_bwd_ms;")
    ap("    g_total_opt_ms += g_last_opt_ms;")
    ap("    if (opt_applied && opt > 0) g_profile_optimizer_steps += 1;")
    ap("    if (loss_out != NULL) {")
    ap("        *loss_out = g_loss_scalar[0];")
    ap("    }")
    ap("    return fwd + bwd + opt;")
    ap("}")
    ap("")

    summary = {
        "forward_ops": len(forward_ops),
        "backward_ops": len(backward_ops),
        "callable_kernels": len(used_decl),
        "optimizer_pairs": len(opt_pairs),
        "optimizer_skipped_non_fp32": skipped_opt_non_fp32,
        "init_weight_count": len(init_weight_specs),
        "init_weight_order": [wname for (wname, _wvar, _numel) in init_weight_specs],
        "init_weight_numel": [int(_numel) for (_wname, _wvar, _numel) in init_weight_specs],
        "tensor_numel_count": len(tensor_numel),
        "memory_total_floats": int(layout_info["total_floats"]),
        "memory_regions": sorted(list(region_offsets.keys())),
        "tensor_slot_count": len(slot_rows),
        "canary_range_count": len(canary_ranges),
        "canary_tail_floats": 16,
        "exec_plan_schema": exec_plan_info.get("schema"),
        "exec_plan_runtime": exec_plan_info.get("runtime"),
        "exec_plan_ops": len(exec_plan_by_op),
        "weight_snapshot_floats": int(weight_snapshot_floats),
        "activation_snapshot_floats": int(activation_snapshot_floats),
        "optimizer_state_snapshot_floats": int(optimizer_state_snapshot_floats),
        "accum_snapshot_floats": int(accum_snapshot_floats),
        "tensor_slots": [
            {
                "index": int(i),
                "name": str(row["name"]),
                "offset": int(row["offset"]),
                "numel": int(row["numel"]),
                "section": str(row["section"]),
                "writable_fwd": int(row["writable_fwd"]),
                "writable_bwd": int(row["writable_bwd"]),
            }
            for i, row in enumerate(slot_rows)
        ],
        "canary_ranges": [
            {
                "index": int(i),
                "start": int(start),
                "length": int(length),
                "left_slot_idx": int(left),
                "right_slot_idx": int(right),
                "left_slot": (str(slot_rows[left]["name"]) if (left >= 0 and left < len(slot_rows)) else None),
                "right_slot": (str(slot_rows[right]["name"]) if (right >= 0 and right < len(slot_rows)) else None),
            }
            for i, (start, length, left, right) in enumerate(canary_ranges)
        ],
        "backward_op_trace": [
            {
                "op_id": int(op.get("op_id", -1)),
                "op": str(op.get("op", "")),
                "kernel_id": str(op.get("kernel_id", "")),
            }
            for op in backward_ops
        ],
        "skipped_ops": skipped_ops,
    }
    return ("\n".join(lines), summary)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate compile-ready v7 training C runtime skeleton from IR2.")
    p.add_argument("--ir2", type=Path, required=True, help="Path to ir2_train_backward_*.json")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Kernel registry JSON path")
    p.add_argument("--manifest", type=Path, default=None, help="Optional weights manifest for exact optimizer numel mapping")
    p.add_argument("--layout", type=Path, required=True, help="Path to layout_train.json (authoritative offsets)")
    p.add_argument("--exec-plan", type=Path, default=None, help="Optional train_exec_plan.json (dispatch metadata)")
    p.add_argument("--output", type=Path, required=True, help="Output C file path")
    p.add_argument("--summary-out", type=Path, default=None, help="Optional summary JSON")
    args = p.parse_args()

    ir2 = _load_json(args.ir2)
    reg = _load_json(args.registry)
    manifest = _load_json(args.manifest) if args.manifest is not None else None
    layout = _load_json(args.layout) if args.layout is not None else None
    exec_plan = _load_json(args.exec_plan) if args.exec_plan is not None else None
    c_src, summary = generate_c(ir2, reg, manifest=manifest, layout=layout, exec_plan=exec_plan)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(c_src, encoding="utf-8")
    print("Generated C: %s" % args.output)
    print("forward_ops=%d backward_ops=%d callable_kernels=%d skipped=%d" % (
        summary["forward_ops"],
        summary["backward_ops"],
        summary["callable_kernels"],
        len(summary["skipped_ops"]),
    ))

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Summary: %s" % args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

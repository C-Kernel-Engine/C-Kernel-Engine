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
            return "1"
        if "add_pos" in name or "pos_offset" in name:
            return "0"
        return "1"

    if "char *" in arg_type:
        return "\"none\""
    return "0"



def _infer_gemm_backward_dims(io_outputs: Dict[str, str], tensor_numel: Dict[str, int], cfg: Dict[str, Any]) -> Dict[str, int]:
    # NOTE:
    # gemm_backward_f32 is one of the easiest places to create silent OOB writes if
    # aligned_in/aligned_out are wrong. Prefer IR-derived tensor sizes first;
    # use cfg defaults only as a last resort when metadata is missing.
    d_input_n: Optional[int] = None
    d_weight_n: Optional[int] = None
    d_bias_n: Optional[int] = None

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
        if d_weight_n is None and lt.startswith("tmp.grad.weight.") and lt.endswith(".w"):
            d_weight_n = int(n)
        if d_bias_n is None and lt.startswith("tmp.grad.weight.") and lt.endswith(".bias"):
            d_bias_n = int(n)

    aligned_in = 0
    aligned_out = 0
    if isinstance(d_input_n, int) and d_input_n > 0:
        aligned_in = int(d_input_n)
    if isinstance(d_weight_n, int) and d_weight_n > 0 and aligned_in > 0 and (d_weight_n % aligned_in == 0):
        aligned_out = int(d_weight_n // aligned_in)
    if aligned_out <= 0 and isinstance(d_bias_n, int) and d_bias_n > 0:
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


def generate_c(ir2: Dict[str, Any], registry: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None, layout: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
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
    for opt_kernel in ("adamw_update_f32", "gradient_clip_norm_f32"):
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
    ap("#include \"ckernel_engine.h\"")
    ap("")
    ap("#ifndef CK_NUM_TOKENS")
    ap("#define CK_NUM_TOKENS 1")
    ap("#endif")
    ap("")

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

            # qk_norm_forward is in-place in kernel API, but IR models distinct outputs.
            # Preserve IR semantics by copying input buffers into output buffers, then running in-place.
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

                ap("    /* op_id=%s op=%s kernel_id=%s (IR out-of-place -> kernel in-place) */" % (op_id, op_name, kid))
                if q_numel > 0 and q_out_var != q_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (q_out_var, q_in_var, int(q_numel)))
                if k_numel > 0 and k_out_var != k_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (k_out_var, k_in_var, int(k_numel)))
                if q_numel > 0:
                    ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                if k_numel > 0:
                    ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                ap("    qk_norm_forward(%s, %s, %s, %s, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, 1e-5f);" % (
                    q_out_var, k_out_var, q_gamma_var, k_gamma_var
                ))
                ap("    call_count++;")
                continue

            # rope_forward_qk is also in-place and requires cos/sin cache pointers.
            # IR models explicit q/k outputs, so stage into output buffers first.
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

                ap("    /* op_id=%s op=%s kernel_id=%s (IR out-of-place -> kernel in-place) */" % (op_id, op_name, kid))
                if q_numel > 0 and q_out_var != q_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (q_out_var, q_in_var, int(q_numel)))
                if k_numel > 0 and k_out_var != k_in_var:
                    ap("    memcpy(%s, %s, (size_t)%d * sizeof(float));" % (k_out_var, k_in_var, int(k_numel)))
                if q_numel > 0:
                    ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (q_out_var, int(q_numel), op_id))
                if k_numel > 0:
                    ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (k_out_var, int(k_numel), op_id))
                ap("    rope_forward_qk_with_rotary_dim(%s, %s, g_rope_cos_cache, g_rope_sin_cache, CK_ROPE_NUM_HEADS, CK_ROPE_NUM_KV_HEADS, CK_NUM_TOKENS, CK_ROPE_HEAD_DIM, CK_ROPE_ALIGNED_HEAD_DIM, 0, CK_ROPE_ROTARY_DIM);" % (
                    q_out_var, k_out_var
                ))
                ap("    call_count++;")
                continue

            # gemm_backward_f32 uses op-local inferred dims from IR tensor sizes.
            # Avoid generic defaults here: wrong aligned_out can corrupt
            # tmp.grad.weight.* and trip canary/heap checks.
            gemm_dims = _infer_gemm_backward_dims(io_outputs, tensor_numel, cfg) if kid == "gemm_backward_f32" else None
            gemm_fwd_mnk = _infer_gemm_forward_mnk(op, cfg) if kid == "gemm_blocked_serial" else None
            swiglu_dim: Optional[int] = None
            if kid in ("swiglu_forward", "swiglu_forward_exact", "swiglu_backward", "swiglu_backward_exact"):
                cands: List[int] = []
                for pool in (io_outputs, io_inputs):
                    for tid in pool.values():
                        n = int(tensor_numel.get(tid, 0) or 0)
                        if n > 0:
                            cands.append(n)
                if cands:
                    swiglu_dim = int(min(cands))

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
            seen_bounds: set[str] = set()
            for expr, n in bounds_checks:
                if expr in seen_bounds:
                    continue
                seen_bounds.add(expr)
                ap("    if (CK_RUNTIME_BOUNDS_ASSERT && ck_bounds_check_span_f32(%s, (size_t)%d) != 0) return -6000 - %s;" % (expr, int(n), op_id))
            ap("    %s(%s);" % (fn, ", ".join(call_args)))
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
        ap("    const float max_grad_norm = 1.0f;")
        ap("    int call_count = 0;")
        ap("    g_opt_step += 1;")
        if skipped_opt_non_fp32:
            ap("    /* skipped non-fp32 optimizer params: %d */" % len(skipped_opt_non_fp32))
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
    ap("    ck_zero_grad();")
    ap("    ck_train_set_batch(token_ids, targets);")
    ap("    int first = -1;")
    ap("    if (CK_RUNTIME_CANARY_CHECKS) ck_train_plant_canaries();")
    ap("    int fwd = ck_train_forward_step();")
    ap("    if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_forward\", &first) != 0) return -1000 - first;")
    ap("    int bwd = ck_train_backward_step();")
    ap("    if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_backward\", &first) != 0) return -1100 - first;")
    ap("    int opt = ck_train_optimizer_step(lr);")
    ap("    if (CK_RUNTIME_CANARY_CHECKS && ck_train_check_canaries(\"step_optimizer\", &first) != 0) return -1200 - first;")
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
        "weight_snapshot_floats": int(weight_snapshot_floats),
        "activation_snapshot_floats": int(activation_snapshot_floats),
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
    p.add_argument("--output", type=Path, required=True, help="Output C file path")
    p.add_argument("--summary-out", type=Path, default=None, help="Optional summary JSON")
    args = p.parse_args()

    ir2 = _load_json(args.ir2)
    reg = _load_json(args.registry)
    manifest = _load_json(args.manifest) if args.manifest is not None else None
    layout = _load_json(args.layout) if args.layout is not None else None
    c_src, summary = generate_c(ir2, reg, manifest=manifest, layout=layout)

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

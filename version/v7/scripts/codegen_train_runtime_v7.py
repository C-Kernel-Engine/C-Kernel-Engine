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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    m = re.match(r"^void\s+([A-Za-z_]\w*)\s*\((.*)\)\s*;?$", one)
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


def _manifest_numel_map(manifest: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(manifest, dict):
        return out
    for e in manifest.get("entries", []) or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name")
        if not isinstance(name, str) or not name:
            continue
        n = _shape_numel(e.get("shape"))
        if n is not None:
            out[name] = n
            continue
        # Inference manifests often omit `shape` but include byte `size`.
        size = e.get("size")
        dtype = str(e.get("dtype", "")).lower()
        if isinstance(size, int) and size > 0:
            if dtype in ("fp32", "f32") and (size % 4 == 0):
                out[name] = size // 4
            elif dtype in ("bf16", "bfloat16") and (size % 2 == 0):
                out[name] = size // 2
    return out


def _ir_weight_numel_map(ops: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for op in ops:
        if not isinstance(op, dict):
            continue
        for _wk, wref in (op.get("weights") or {}).items():
            if not isinstance(wref, dict):
                continue
            name = wref.get("name")
            if not isinstance(name, str) or not name:
                continue
            n = _shape_numel(wref.get("shape"))
            if n is not None and n > 0:
                out[name] = n
    return out


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
        return "CK_TENSOR_CAP_F32"

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
            return str(int(cfg.get("context_len", 128)))
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
            return "CK_TENSOR_CAP_F32"
        if "num_threads" in name:
            return "1"
        if "add_pos" in name or "pos_offset" in name:
            return "0"
        return "1"

    if "char *" in arg_type:
        return "\"none\""
    return "0"


def _choose_tensor_for_ptr(
    arg_type: str,
    arg_name: str,
    io_inputs: Dict[str, str],
    io_outputs: Dict[str, str],
    io_weights: Dict[str, str],
    tensors: Dict[str, Dict[str, Any]],
    tvars_f32: Dict[str, str],
    tvars_i32: Dict[str, str],
) -> str:
    want_i32 = "int32_t" in arg_type or "int32" in arg_type
    const_ptr = "const" in arg_type
    aname = str(arg_name)

    def pick_tid_by_key(m: Dict[str, str], key: str) -> Optional[str]:
        tid = m.get(key)
        if isinstance(tid, str):
            return tid
        return None

    candidates: List[str] = []
    for pool in (io_inputs, io_outputs, io_weights):
        t = pick_tid_by_key(pool, aname)
        if t:
            candidates.append(t)
    if aname.startswith("d_"):
        base = aname[2:]
        for pool in (io_outputs, io_inputs, io_weights):
            t = pick_tid_by_key(pool, base)
            if t:
                candidates.append(t)
    if "_" in aname:
        short = aname.split("_")[-1]
        for pool in (io_inputs, io_outputs, io_weights):
            t = pick_tid_by_key(pool, short)
            if t:
                candidates.append(t)

    pool_order: List[Dict[str, str]]
    if const_ptr:
        pool_order = [io_inputs, io_weights, io_outputs]
    else:
        pool_order = [io_outputs, io_inputs, io_weights]
    for pool in pool_order:
        for t in pool.values():
            if isinstance(t, str):
                candidates.append(t)

    seen = set()
    ordered: List[str] = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    for tid in ordered:
        dtype = str((tensors.get(tid) or {}).get("dtype", "fp32")).lower()
        if want_i32 and _dtype_is_i32(dtype):
            if tid in tvars_i32:
                return tvars_i32[tid]
        if not want_i32 and not _dtype_is_i32(dtype):
            if tid in tvars_f32:
                return tvars_f32[tid]

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


def generate_c(ir2: Dict[str, Any], registry: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    cfg = dict(ir2.get("config") or {})
    tensors = _collect_tensors(ir2)
    kernel_map = _kernel_registry_map(registry)
    forward_ops = sorted(list(ir2.get("forward") or []), key=lambda o: int(o.get("op_id", 0)))
    backward_ops = sorted(list(ir2.get("backward") or []), key=lambda o: int(o.get("op_id", 0)))
    all_ops = forward_ops + backward_ops
    ir_weight_numel = _ir_weight_numel_map(all_ops)
    manifest_numel = _manifest_numel_map(manifest)
    op_by_id = {int(op["op_id"]): op for op in all_ops if isinstance(op, dict) and "op_id" in op}

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

    # Optimizer kernel is required by generated ck_train_optimizer_step
    # even when IR2 does not carry explicit optimizer ops.
    if "adamw_update_f32" not in used_decl:
        resolved = resolve_decl_for_kernel("adamw_update_f32")
        if resolved is not None:
            used_decl["adamw_update_f32"] = resolved

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
    ap("#include \"ckernel_engine.h\"")
    ap("")
    ap("#ifndef CK_NUM_TOKENS")
    ap("#define CK_NUM_TOKENS 1")
    ap("#endif")
    ap("#ifndef CK_TENSOR_CAP_F32")
    ap("#define CK_TENSOR_CAP_F32 4096")
    ap("#endif")
    ap("#ifndef CK_TENSOR_CAP_I32")
    ap("#define CK_TENSOR_CAP_I32 512")
    ap("#endif")
    ap("")

    ap("/* Fallback declarations for kernels with incomplete registry declarations. */")
    for kid in sorted(used_decl.keys()):
        fn, args, decl = used_decl[kid]
        ap("/* %s -> %s */" % (kid, fn))
        ap(decl if decl.endswith(";") else (decl + ";"))
    ap("")

    ap("static float g_dummy_f32[CK_TENSOR_CAP_F32];")
    ap("static int32_t g_dummy_i32[CK_TENSOR_CAP_I32];")
    ap("static float g_loss_scalar[1];")
    ap("")

    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        ap("static float %s[CK_TENSOR_CAP_F32]; /* %s */" % (var, tid))
    for tid, var in sorted(tvars_i32.items(), key=lambda x: x[1]):
        ap("static int32_t %s[CK_TENSOR_CAP_I32]; /* %s */" % (var, tid))
    ap("")

    # Optimizer state (AdamW moments) for each grad.weight.* tensor.
    # AdamW kernel here is fp32-only, so non-fp32 params are skipped.
    opt_pairs: List[Tuple[str, str, str, str, str, Optional[int]]] = []
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
        numel = ir_weight_numel.get(wname)
        if numel is None:
            numel = manifest_numel.get(wname)
        opt_pairs.append((wname, gvar, wvar, mvar, vvar, numel))

    if opt_pairs:
        ap("/* Optimizer state (AdamW moments) */")
        seen = set()
        for _wname, _gvar, _wvar, mvar, vvar, _numel in opt_pairs:
            if mvar not in seen:
                ap("static float %s[CK_TENSOR_CAP_F32];" % mvar)
                seen.add(mvar)
            if vvar not in seen:
                ap("static float %s[CK_TENSOR_CAP_F32];" % vvar)
                seen.add(vvar)
        ap("static int g_opt_step = 0;")
    ap("")

    grad_vars_f32: List[str] = []
    for tid, var in sorted(tvars_f32.items(), key=lambda x: x[1]):
        if isinstance(tid, str) and tid.startswith("grad."):
            grad_vars_f32.append(var)

    ap("void ck_train_reset_buffers(void) {")
    for var in sorted(set(tvars_f32.values())):
        ap("    memset(%s, 0, sizeof(%s));" % (var, var))
    for var in sorted(set(tvars_i32.values())):
        ap("    memset(%s, 0, sizeof(%s));" % (var, var))
    if opt_pairs:
        seen = set()
        for _wname, _gvar, _wvar, mvar, vvar, _numel in opt_pairs:
            if mvar not in seen:
                ap("    memset(%s, 0, sizeof(%s));" % (mvar, mvar))
                seen.add(mvar)
            if vvar not in seen:
                ap("    memset(%s, 0, sizeof(%s));" % (vvar, vvar))
                seen.add(vvar)
        ap("    g_opt_step = 0;")
    ap("    g_loss_scalar[0] = 0.0f;")
    ap("}")
    ap("")
    ap("int ck_train_init(const float *bump, const int *manifest_sizes, int num_params) {")
    ap("    (void)bump;")
    ap("    (void)manifest_sizes;")
    ap("    (void)num_params;")
    ap("    ck_train_reset_buffers();")
    ap("    return 0;")
    ap("}")
    ap("")
    ap("void ck_zero_grad(void) {")
    if grad_vars_f32:
        for var in sorted(set(grad_vars_f32)):
            ap("    memset(%s, 0, sizeof(%s));" % (var, var))
    else:
        ap("    /* No grad.* tensors found in IR; zero-grad is a no-op. */")
    ap("}")
    ap("")

    def emit_ops(fn_name: str, ops: List[Dict[str, Any]], phase: str) -> None:
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

            call_args: List[str] = []
            for atype, aname in args:
                is_ptr = "*" in atype
                if is_ptr:
                    expr = _choose_tensor_for_ptr(
                        atype, aname, io_inputs, io_outputs, io_weights, tensors, tvars_f32, tvars_i32
                    )
                    call_args.append(expr)
                else:
                    call_args.append(_arg_scalar_expr(atype, aname, cfg))

            ap("    /* op_id=%s op=%s kernel_id=%s */" % (op_id, op_name, kid))
            ap("    %s(%s);" % (fn, ", ".join(call_args)))
            ap("    call_count++;")

        ap("    return call_count;")
        ap("}")
        ap("")

    emit_ops("ck_train_forward_step", forward_ops, "forward")
    emit_ops("ck_train_backward_step", backward_ops, "backward")

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
        ap("    int call_count = 0;")
        ap("    g_opt_step += 1;")
        if skipped_opt_non_fp32:
            ap("    /* skipped non-fp32 optimizer params: %d */" % len(skipped_opt_non_fp32))
        for wname, gvar, wvar, mvar, vvar, numel in opt_pairs:
            n_expr = str(numel) if isinstance(numel, int) and numel > 0 else "CK_TENSOR_CAP_F32"
            ap("    /* AdamW update: %s */" % wname)
            ap("    adamw_update_f32(%s, %s, %s, %s, (size_t)%s, lr, beta1, beta2, eps, weight_decay, g_opt_step);" % (
                gvar, wvar, mvar, vvar, n_expr
            ))
            ap("    call_count++;")
        ap("    return call_count;")
    ap("}")
    ap("")

    ap("int ck_train_step(const int32_t *token_ids, const int32_t *targets, float *loss_out, float lr) {")
    ap("    ck_zero_grad();")
    ap("    if (token_ids != NULL) {")
    ap("        memcpy(g_dummy_i32, token_ids, sizeof(int32_t) * CK_NUM_TOKENS);")
    ap("    }")
    ap("    if (targets != NULL) {")
    ap("        memcpy(g_dummy_i32, targets, sizeof(int32_t) * CK_NUM_TOKENS);")
    ap("    }")
    ap("    int fwd = ck_train_forward_step();")
    ap("    int bwd = ck_train_backward_step();")
    ap("    int opt = ck_train_optimizer_step(lr);")
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
        "skipped_ops": skipped_ops,
    }
    return ("\n".join(lines), summary)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate compile-ready v7 training C runtime skeleton from IR2.")
    p.add_argument("--ir2", type=Path, required=True, help="Path to ir2_train_backward_*.json")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Kernel registry JSON path")
    p.add_argument("--manifest", type=Path, default=None, help="Optional weights manifest for exact optimizer numel mapping")
    p.add_argument("--output", type=Path, required=True, help="Output C file path")
    p.add_argument("--summary-out", type=Path, default=None, help="Optional summary JSON")
    args = p.parse_args()

    ir2 = _load_json(args.ir2)
    reg = _load_json(args.registry)
    manifest = _load_json(args.manifest) if args.manifest is not None else None
    c_src, summary = generate_c(ir2, reg, manifest=manifest)

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

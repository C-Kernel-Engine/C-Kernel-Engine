#!/usr/bin/env python3
"""
lower_ir2_backward_v7.py

Synthesize IR2 (forward + backward) from IR1 train-forward:
- reverse traversal of forward ops
- grad-rule driven backward kernel expansion
- explicit grad accumulation ops for activations and weights
- fail-fast contracts (optional strict mode)
- preserve deterministic, IR-derived lowering (no model-family special-cases in emitter)
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import build_ir_v7


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = V7_ROOT / "kernel_maps" / "kernel_bindings.json"
DEFAULT_GRAD_RULES_PATH = SCRIPT_DIR / "grad_rules_v7.json"

BACKWARD_BRIDGE_PLAN_SPECS: Dict[str, Dict[str, Any]] = {
    "attn_backward_core": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "semantic_op_from_forward": True,
        "pre_roles": (
            {"role": "dy", "input_key": "in_0", "head_group": "num_heads"},
            {"role": "q", "input_key": "in_1", "head_group": "num_heads"},
            {"role": "k", "input_key": "in_2", "head_group": "num_kv_heads"},
            {"role": "v", "input_key": "in_3", "head_group": "num_kv_heads"},
        ),
        "post_roles": (
            {"role": "d_q", "output_key": "d_q", "head_group": "num_heads"},
            {"role": "d_k", "output_key": "d_k", "head_group": "num_kv_heads"},
            {"role": "d_v", "output_key": "d_v", "head_group": "num_kv_heads"},
        ),
    },
    "rope_qk_backward_core": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "semantic_op": "rope_qk",
        "pre_roles": (
            {"role": "grad_q", "input_key": "in_0", "head_group": "num_heads"},
            {"role": "grad_k", "input_key": "in_1", "head_group": "num_kv_heads"},
        ),
        "post_roles": (
            {"role": "d_q", "output_key": "d_q", "head_group": "num_heads"},
            {"role": "d_k", "output_key": "d_k", "head_group": "num_kv_heads"},
        ),
    },
    "qk_norm_backward_core": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "semantic_op": "qk_norm",
        "pre_roles": (
            {"role": "grad_q", "input_key": "in_0", "head_group": "num_heads"},
            {"role": "grad_k", "input_key": "in_1", "head_group": "num_kv_heads"},
            {"role": "q", "input_key": "in_2", "head_group": "num_heads"},
            {"role": "k", "input_key": "in_3", "head_group": "num_kv_heads"},
        ),
        "post_roles": (
            {"role": "d_q", "output_key": "d_q", "head_group": "num_heads"},
            {"role": "d_k", "output_key": "d_k", "head_group": "num_kv_heads"},
        ),
    },
    "checkpoint_rematerialize_saved_tensor": {
        "kernel_layout": "head_major",
        "tensor_layout": "token_major",
        "semantic_op": "checkpoint_rematerialize_saved_tensor",
        "pre_roles": (
            {"role": "q", "input_key": "q", "head_group": "num_heads"},
            {"role": "k", "input_key": "k", "head_group": "num_kv_heads"},
            {"role": "v", "input_key": "v", "head_group": "num_kv_heads"},
        ),
        "post_roles": (),
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sanitize_tensor_id(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("_", ".", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _kernel_ids(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for kernel in registry.get("kernels", []):
        kid = kernel.get("id")
        if isinstance(kid, str) and kid:
            out[kid] = kernel
    return out


def _binding_ids(bindings_doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(bindings_doc.get("bindings", {}))


def _resolve_ir1_config(ir1: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(ir1.get("config", {}))
    rope_layout = build_ir_v7._normalize_rope_layout_value(config.get("rope_layout"))
    if rope_layout:
        config["rope_layout"] = rope_layout
        return config

    template_name = str(ir1.get("template_name", "") or "").strip().lower()
    template_doc = build_ir_v7._load_builtin_template_doc(template_name)
    if isinstance(template_doc, dict):
        attention_contract = ((template_doc.get("contract") or {}).get("attention_contract") or {})
        template_rope_layout = build_ir_v7._normalize_rope_layout_value(attention_contract.get("rope_layout"))
        if template_rope_layout:
            config["rope_layout"] = template_rope_layout
    return config


def _resolve_backward_kernel_id(spec: Dict[str, Any], config: Dict[str, Any]) -> str:
    kernel_id = str(spec.get("kernel_id", ""))
    if kernel_id in {"rope_backward_qk_f32", "rope_backward_qk_pairwise_f32"}:
        return build_ir_v7._resolve_rope_backward_qk_kernel(config, kernel_id)
    return kernel_id


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return int(n)


def _cfg_int(config: Dict[str, Any], keys: List[str], default: int) -> int:
    for k in keys:
        if k in config:
            try:
                v = int(config.get(k))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(default)


def _default_activation_numel(config: Dict[str, Any]) -> int:
    d_model = _cfg_int(config, ["embed_dim", "hidden_size", "d_model"], 128)
    hidden = _cfg_int(config, ["intermediate_size", "hidden_dim"], max(2 * d_model, d_model))
    vocab = _cfg_int(config, ["vocab_size"], 256)
    num_heads = _cfg_int(config, ["num_heads"], 1)
    num_kv_heads = _cfg_int(config, ["num_kv_heads"], num_heads)
    head_dim = _cfg_int(config, ["head_dim"], max(1, d_model // max(1, num_heads)))
    q_dim = max(1, num_heads * head_dim)
    kv_dim = max(1, num_kv_heads * head_dim)
    return max(1, d_model, hidden, 2 * hidden, vocab, q_dim, kv_dim)


def _aligned_head_dim(config: Dict[str, Any]) -> int:
    head_dim = _cfg_int(config, ["head_dim"], 1)
    return _cfg_int(config, ["aligned_head_dim"], max(1, head_dim))


def _active_tokens(config: Dict[str, Any]) -> int:
    return _cfg_int(config, ["train_tokens", "tokens", "seq_len"], 1)


def _head_major_bridge_shape_numel(config: Dict[str, Any], head_group: str) -> Tuple[List[int], int]:
    tokens = max(1, _active_tokens(config))
    aligned_head_dim = max(1, _aligned_head_dim(config))
    group = str(head_group or "num_heads").strip().lower()
    if group in ("num_kv_heads", "kv", "k", "v"):
        heads = _cfg_int(config, ["num_kv_heads"], _cfg_int(config, ["num_heads"], 1))
    else:
        heads = _cfg_int(config, ["num_heads"], 1)
    heads = max(1, int(heads))
    numel = max(1, tokens * heads * aligned_head_dim)
    return [heads, tokens, aligned_head_dim], numel


def _tensor_numel(tensors: Dict[str, Dict[str, Any]], tensor_id: Optional[str]) -> Optional[int]:
    if not isinstance(tensor_id, str) or not tensor_id:
        return None
    meta = tensors.get(tensor_id)
    if not isinstance(meta, dict):
        return None
    n = meta.get("numel")
    if isinstance(n, int) and n > 0:
        return int(n)
    shape_n = _shape_numel(meta.get("shape"))
    if isinstance(shape_n, int) and shape_n > 0:
        return int(shape_n)
    return None


def _last_forward_logits_op(ops: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for op in reversed(ops):
        if op.get("phase") != "forward":
            continue
        if op.get("op") == "logits" and op.get("dataflow", {}).get("outputs"):
            return op
    return None


def _op_output_tensor(op: Dict[str, Any], out_name: str) -> Optional[str]:
    outputs = op.get("dataflow", {}).get("outputs", {})
    ref = outputs.get(out_name)
    if isinstance(ref, dict):
        t = ref.get("tensor")
        if isinstance(t, str) and t:
            return t
    return None


def _find_weight_ref(op: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    w = op.get("weights", {}).get(key)
    if isinstance(w, dict):
        return w
    return None


def _ensure_tensor(
    tensors: Dict[str, Dict[str, Any]],
    tensor_id: str,
    dtype: str,
    kind: str,
    requires_grad: bool,
    persistent: bool,
    producer: Optional[Dict[str, Any]] = None,
    shape: Optional[List[int]] = None,
    numel: Optional[int] = None,
) -> None:
    if tensor_id in tensors:
        cur = tensors[tensor_id]
        if shape is not None and "shape" not in cur:
            cur["shape"] = list(shape)
        if isinstance(numel, int) and numel > 0:
            cur_numel = cur.get("numel")
            if not isinstance(cur_numel, int) or cur_numel <= 0:
                cur["numel"] = int(numel)
        return
    tensors[tensor_id] = {
        "dtype": dtype,
        "kind": kind,
        "requires_grad": bool(requires_grad),
        "persistent": bool(persistent),
        "producer": producer,
        "shape": list(shape) if isinstance(shape, list) else None,
        "numel": int(numel) if isinstance(numel, int) and numel > 0 else None,
    }


def _bridge_ref(
    tensors: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    *,
    core_op_id: int,
    role: str,
    head_group: str,
) -> Dict[str, Any]:
    shape, numel = _head_major_bridge_shape_numel(config, head_group)
    tid = "tmp.bridge.bwd.op%d.%s" % (int(core_op_id), _sanitize_tensor_id(role))
    _ensure_tensor(
        tensors,
        tensor_id=tid,
        dtype="fp32",
        kind="tmp",
        requires_grad=False,
        persistent=False,
        producer=None,
        shape=shape,
        numel=numel,
    )
    return {
        "tensor": tid,
        "kind": "tmp",
        "shape": shape,
        "numel": numel,
        "head_group": head_group,
    }


def _normalize_checkpoint_policy(policy: Any) -> str:
    raw = str(policy or "none").strip().lower()
    aliases = {
        "": "none",
        "off": "none",
        "save_all": "none",
        "recompute_attention": "recompute_attn",
        "recompute_attn_weights": "recompute_attn",
    }
    return aliases.get(raw, raw)


def _checkpoint_recomputes_attention(policy: str) -> bool:
    return _normalize_checkpoint_policy(policy) == "recompute_attn"


def synthesize_ir2_backward(
    ir1: Dict[str, Any],
    registry: Dict[str, Any],
    bindings_doc: Dict[str, Any],
    grad_rules: Dict[str, Any],
    strict: bool,
    allow_partial: bool,
    checkpoint_policy: str
) -> Dict[str, Any]:
    if ir1.get("format") != "ir1-train-v7":
        raise RuntimeError("Expected ir1-train-v7 input")

    kernels = _kernel_ids(registry)
    bindings = _binding_ids(bindings_doc)
    rules = grad_rules.get("rules", {}) or {}
    checkpoint_policy = _normalize_checkpoint_policy(checkpoint_policy)
    if checkpoint_policy not in {"none", "recompute_attn"}:
        raise RuntimeError("Unsupported checkpoint policy: %s" % checkpoint_policy)

    config = _resolve_ir1_config(ir1)
    activation_default_numel = _default_activation_numel(config)

    forward_ops = deepcopy(ir1.get("ops", []))
    tensors: Dict[str, Dict[str, Any]] = deepcopy(ir1.get("tensors", {}))

    backward_ops: List[Dict[str, Any]] = []
    issues: List[str] = []
    warnings: List[str] = []
    unresolved: List[Dict[str, Any]] = []

    next_op_id = 0
    if forward_ops:
        next_op_id = max(int(o.get("op_id", 0)) for o in forward_ops) + 1

    # Maps forward activation tensor -> canonical accumulated grad tensor (grad.act.*).
    grad_for_tensor: Dict[str, str] = {}
    # Tracks writers per gradient destination for invariant reporting/debugging.
    grad_writers: Dict[str, List[int]] = {}
    bridge_enabled = str(ir1.get("bridge_lowering", "legacy") or "legacy").strip().lower() == "explicit"
    checkpoint_rematerialize_ops = 0
    checkpoint_demoted_saved_tensors: List[str] = []

    def emit_backward_op(op: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal next_op_id
        op["op_id"] = next_op_id
        backward_ops.append(op)
        next_op_id += 1
        return op

    def canonical_grad_activation(source_tensor: str) -> str:
        safe = _sanitize_tensor_id(source_tensor)
        tid = "grad.act.%s" % safe
        src_meta = tensors.get(source_tensor) if isinstance(source_tensor, str) else None
        src_shape = src_meta.get("shape") if isinstance(src_meta, dict) else None
        src_numel = _tensor_numel(tensors, source_tensor) or activation_default_numel
        _ensure_tensor(
            tensors,
            tensor_id=tid,
            dtype="fp32",
            kind="grad_activation",
            requires_grad=False,
            persistent=True,
            producer=None,
            shape=src_shape if isinstance(src_shape, list) else None,
            numel=src_numel,
        )
        return tid

    def canonical_grad_weight(param_name: str) -> str:
        safe = _sanitize_tensor_id(param_name)
        tid = "grad.weight.%s" % safe
        src_tid = "weight.%s" % param_name
        src_meta = tensors.get(src_tid) if isinstance(src_tid, str) else None
        src_shape = src_meta.get("shape") if isinstance(src_meta, dict) else None
        src_numel = _tensor_numel(tensors, src_tid)
        _ensure_tensor(
            tensors,
            tensor_id=tid,
            dtype="fp32",
            kind="grad_weight",
            requires_grad=False,
            persistent=True,
            producer=None,
            shape=src_shape if isinstance(src_shape, list) else None,
            numel=src_numel,
        )
        return tid

    def emit_accumulate(src_tensor: str, dst_tensor: str, forward_ref: int, layer: int, role: str, target_param: Optional[str]) -> None:
        if "gradient_accumulate_f32" not in kernels:
            issues.append("Missing kernel id `gradient_accumulate_f32` for accumulation")
        if "gradient_accumulate_f32" not in bindings:
            issues.append("Missing bindings for kernel id `gradient_accumulate_f32`")

        op = {
            "phase": "backward",
            "op": "grad_accumulate",
            "kernel_id": "gradient_accumulate_f32",
            "forward_ref": int(forward_ref),
            "layer": int(layer),
            "role": role,
            "target_param": target_param,
            "dataflow": {
                "inputs": {
                    "dst": {"tensor": dst_tensor, "kind": "grad_accumulator"},
                    "src": {"tensor": src_tensor, "kind": "tmp_grad"}
                },
                "outputs": {
                    "dst": {"tensor": dst_tensor, "kind": "grad_accumulator"}
                }
            },
            "attrs": {"semantics": "dst += src"}
        }
        created = emit_backward_op(op)
        grad_writers.setdefault(dst_tensor, []).append(int(created["op_id"]))

    def build_backward_bridge_plan(
        *,
        plan_key: str,
        core_op_id: int,
        semantic_op: str,
        bwd_inputs: Dict[str, Dict[str, Any]],
        bwd_outputs: Dict[str, Dict[str, Any]],
        runtime_contract: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        spec = BACKWARD_BRIDGE_PLAN_SPECS.get(str(plan_key))
        if not isinstance(spec, dict):
            return None

        plan: Dict[str, Any] = {
            "version": 1,
            "mode": "explicit",
            "semantic_op": str(semantic_op),
            "kernel_layout": str(spec.get("kernel_layout", "head_major") or "head_major"),
            "tensor_layout": str(spec.get("tensor_layout", "token_major") or "token_major"),
            "pre": [],
            "post": [],
        }
        if isinstance(runtime_contract, dict) and runtime_contract:
            plan["runtime_contract"] = dict(runtime_contract)

        for row in spec.get("pre_roles", ()):
            input_key = str(row.get("input_key", "") or "")
            src = bwd_inputs.get(input_key)
            if not isinstance(src, dict) or not isinstance(src.get("tensor"), str):
                return None
            role = str(row.get("role", input_key) or input_key)
            head_group = str(row.get("head_group", "num_heads") or "num_heads")
            scratch = _bridge_ref(tensors, config, core_op_id=core_op_id, role=role, head_group=head_group)
            plan["pre"].append(
                {
                    "role": role,
                    "head_group": head_group,
                    "input_tensor": src["tensor"],
                    "output_tensor": scratch["tensor"],
                    "layout_in": "token_major",
                    "layout_out": "head_major",
                }
            )

        for row in spec.get("post_roles", ()):
            output_key = str(row.get("output_key", "") or "")
            dst = bwd_outputs.get(output_key)
            if not isinstance(dst, dict) or not isinstance(dst.get("tensor"), str):
                return None
            role = str(row.get("role", output_key) or output_key)
            head_group = str(row.get("head_group", "num_heads") or "num_heads")
            scratch = _bridge_ref(tensors, config, core_op_id=core_op_id, role=role, head_group=head_group)
            plan["post"].append(
                {
                    "role": role,
                    "head_group": head_group,
                    "input_tensor": scratch["tensor"],
                    "output_tensor": dst["tensor"],
                    "layout_in": "head_major",
                    "layout_out": "token_major",
                }
            )
        return plan

    def attach_backward_bridge_plan(
        core_op: Dict[str, Any],
        *,
        kernel_id: str,
        fwd_op: Dict[str, Any],
        bwd_inputs: Dict[str, Dict[str, Any]],
        bwd_outputs: Dict[str, Dict[str, Any]],
    ) -> None:
        if not bridge_enabled:
            return
        core_op_id = int(core_op.get("op_id", -1))
        if core_op_id < 0:
            return
        plan_key = str(core_op.get("op", "") or "")
        semantic_op = str(fwd_op.get("op", "") or "attention")
        spec = BACKWARD_BRIDGE_PLAN_SPECS.get(plan_key)
        if not isinstance(spec, dict):
            return
        if not bool(spec.get("semantic_op_from_forward", False)):
            semantic_op = str(spec.get("semantic_op", semantic_op) or semantic_op)
        plan = build_backward_bridge_plan(
            plan_key=plan_key,
            core_op_id=core_op_id,
            semantic_op=semantic_op,
            bwd_inputs=bwd_inputs,
            bwd_outputs=bwd_outputs,
        )
        if isinstance(plan, dict):
            core_op["bridge_plan"] = plan

    def _apply_forward_checkpoint_policy(op: Dict[str, Any]) -> None:
        if not _checkpoint_recomputes_attention(checkpoint_policy):
            return
        if str(op.get("op", "") or "") not in ("attn", "attn_sliding"):
            return
        runtime_contract = op.get("runtime_contract") if isinstance(op.get("runtime_contract"), dict) else {}
        base_kernel_id = str(
            runtime_contract.get("base_kernel_id")
            or runtime_contract.get("kernel_id")
            or op.get("kernel_id")
            or ""
        )
        if not base_kernel_id:
            return
        updated_contract = dict(runtime_contract)
        updated_contract["version"] = int(updated_contract.get("version", 1) or 1)
        updated_contract["semantic_op"] = str(updated_contract.get("semantic_op") or op.get("op") or "attn")
        updated_contract["base_kernel_id"] = base_kernel_id
        updated_contract["kernel_id"] = base_kernel_id
        updated_contract["materialize_saved_attn_weights"] = False
        updated_contract["checkpoint_policy"] = checkpoint_policy
        updated_contract.pop("requires_zero_sliding_window", None)
        op["runtime_kernel_id"] = base_kernel_id
        op["runtime_contract"] = updated_contract
        bridge_plan = op.get("bridge_plan")
        if isinstance(bridge_plan, dict):
            bridge_plan["runtime_contract"] = dict(updated_contract)
        saved = op.get("save_for_backward")
        if isinstance(saved, dict):
            attn_saved = saved.get("attn_weights")
            if isinstance(attn_saved, dict):
                attn_saved["checkpoint_storage"] = "recompute"

    def emit_checkpoint_rematerialize_saved_tensor(
        *,
        fwd_op: Dict[str, Any],
        fwd_inputs: Dict[str, Dict[str, Any]],
        saved_attn: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        nonlocal checkpoint_rematerialize_ops
        saved_tid = saved_attn.get("tensor")
        if not isinstance(saved_tid, str) or not saved_tid:
            return None
        q_ref = fwd_inputs.get("q")
        k_ref = fwd_inputs.get("k")
        v_ref = fwd_inputs.get("v")
        if not all(isinstance(x, dict) and isinstance(x.get("tensor"), str) for x in (q_ref, k_ref, v_ref)):
            return None

        saved_meta = tensors.get(saved_tid) if isinstance(tensors.get(saved_tid), dict) else {}
        saved_shape = saved_meta.get("shape") if isinstance(saved_meta.get("shape"), list) else saved_attn.get("shape")
        saved_numel = _tensor_numel(tensors, saved_tid) or saved_attn.get("numel")
        saved_numel = int(saved_numel) if isinstance(saved_numel, int) and saved_numel > 0 else None
        tensors.setdefault(saved_tid, {})
        tensors[saved_tid]["dtype"] = "fp32"
        tensors[saved_tid]["kind"] = "aux"
        tensors[saved_tid]["persistent"] = False
        tensors[saved_tid]["requires_grad"] = False
        tensors[saved_tid]["checkpoint_storage"] = "recompute"
        tensors[saved_tid]["shape"] = list(saved_shape) if isinstance(saved_shape, list) else tensors[saved_tid].get("shape")
        if saved_numel is not None:
            tensors[saved_tid]["numel"] = int(saved_numel)
        if saved_tid not in checkpoint_demoted_saved_tensors:
            checkpoint_demoted_saved_tensors.append(saved_tid)

        out_shape, out_numel = _head_major_bridge_shape_numel(config, "num_heads")
        out_tensor = "aux.checkpoint.op%d.attn_out" % int(fwd_op.get("op_id", -1))
        _ensure_tensor(
            tensors,
            tensor_id=out_tensor,
            dtype="fp32",
            kind="aux",
            requires_grad=False,
            persistent=False,
            producer=None,
            shape=out_shape,
            numel=out_numel,
        )

        base_kernel_id = str(
            ((fwd_op.get("runtime_contract") or {}).get("base_kernel_id"))
            or ((fwd_op.get("runtime_contract") or {}).get("kernel_id"))
            or fwd_op.get("kernel_id")
            or "attention_forward_causal_head_major_gqa_flash_strided"
        )
        runtime_contract = {
            "version": 1,
            "semantic_op": str(fwd_op.get("op") or "attn"),
            "kernel_id": "attention_forward_causal_head_major_gqa_exact",
            "base_kernel_id": base_kernel_id,
            "materialize_saved_attn_weights": True,
            "checkpoint_policy": checkpoint_policy,
        }
        rematerialize_contract = {
            "version": 1,
            "mode": "forward_recompute",
            "semantic_op": str(fwd_op.get("op") or "attn"),
            "saved_key": "attn_weights",
            "saved_tensor": saved_tid,
            "runtime_contract": dict(runtime_contract),
        }

        remat = emit_backward_op(
            {
                "phase": "backward",
                "op": "checkpoint_rematerialize_saved_tensor",
                "kernel_id": base_kernel_id,
                "forward_ref": int(fwd_op.get("op_id", -1)),
                "layer": int(fwd_op.get("layer", -1)),
                "section": fwd_op.get("section"),
                "role": "checkpoint_rematerialize",
                "runtime_contract": runtime_contract,
                "rematerialize_contract": rematerialize_contract,
                "save_for_backward": {
                    "attn_weights": {
                        "tensor": saved_tid,
                        "kind": "aux",
                        "shape": list(saved_shape) if isinstance(saved_shape, list) else None,
                        "numel": saved_numel,
                        "checkpoint_storage": "recompute",
                    }
                },
                "dataflow": {
                    "inputs": {
                        "q": dict(q_ref or {}),
                        "k": dict(k_ref or {}),
                        "v": dict(v_ref or {}),
                    },
                    "outputs": {
                        "out": {
                            "tensor": out_tensor,
                            "kind": "aux",
                            "dtype": "fp32",
                            "shape": out_shape,
                            "numel": out_numel,
                        }
                    },
                },
                "attrs": {
                    "checkpoint_policy": checkpoint_policy,
                    "rematerialize_saved": "attn_weights",
                },
            }
        )
        tensors[saved_tid]["producer"] = {"op_id": int(remat.get("op_id", -1)), "output_name": "attn_weights"}
        if bridge_enabled:
            plan = build_backward_bridge_plan(
                plan_key="checkpoint_rematerialize_saved_tensor",
                core_op_id=int(remat.get("op_id", -1)),
                semantic_op="checkpoint_rematerialize_saved_tensor",
                bwd_inputs={
                    "q": dict(q_ref or {}),
                    "k": dict(k_ref or {}),
                    "v": dict(v_ref or {}),
                },
                bwd_outputs={},
                runtime_contract=runtime_contract,
            )
            if isinstance(plan, dict):
                remat["bridge_plan"] = plan
        checkpoint_rematerialize_ops += 1
        return remat

    def _shape_numel_from_write_source(
        write_spec: Dict[str, Any],
        *,
        fwd_inputs: Dict[str, Dict[str, Any]],
        fwd_outputs: Dict[str, Dict[str, Any]],
        fwd_weights: Dict[str, Dict[str, Any]],
        save_for_backward: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[List[int]], Optional[int]]:
        source_specs = (
            ("shape_from_saved", save_for_backward, "tensor"),
            ("shape_from_input", fwd_inputs, "tensor"),
            ("shape_from_output", fwd_outputs, "tensor"),
            ("shape_from_weight", fwd_weights, "tensor"),
        )
        for field, pool, tensor_key in source_specs:
            ref_name = str(write_spec.get(field, "") or "").strip()
            if not ref_name:
                continue
            ref = pool.get(ref_name)
            if not isinstance(ref, dict):
                continue
            tensor_id = ref.get(tensor_key)
            if not isinstance(tensor_id, str) or not tensor_id:
                continue
            meta = tensors.get(tensor_id)
            shape = None
            if isinstance(meta, dict):
                meta_shape = meta.get("shape")
                if isinstance(meta_shape, list) and meta_shape:
                    shape = list(meta_shape)
            if shape is None:
                ref_shape = ref.get("shape")
                if isinstance(ref_shape, list) and ref_shape:
                    shape = list(ref_shape)
            numel = _tensor_numel(tensors, tensor_id)
            if not isinstance(numel, int) or numel <= 0:
                ref_numel = ref.get("numel")
                if isinstance(ref_numel, int) and ref_numel > 0:
                    numel = int(ref_numel)
                else:
                    numel = None
            return shape, numel
        return None, None

    for op in forward_ops:
        if isinstance(op, dict):
            _apply_forward_checkpoint_policy(op)

    # Seed backward graph from explicit CE gradient on final logits.
    # This is the only non-reverse-traversal entrypoint for gradient flow.
    logits_op = _last_forward_logits_op(forward_ops)
    if logits_op is None:
        msg = "Cannot seed backward graph: no `logits` op found in IR1"
        if strict:
            issues.append(msg)
        else:
            warnings.append(msg)
    else:
        logits_tensor = _op_output_tensor(logits_op, "y")
        if not logits_tensor:
            issues.append("logits op found but output `y` is missing")
        else:
            loss_rule = rules.get("cross_entropy_loss", {})
            loss_kernel_id = "softmax_cross_entropy_loss"
            if isinstance(loss_rule, dict):
                bwd_ops = loss_rule.get("backward_ops", []) or []
                if bwd_ops and isinstance(bwd_ops[0], dict):
                    loss_kernel_id = str(bwd_ops[0].get("kernel_id") or loss_kernel_id)

            if loss_kernel_id not in kernels:
                issues.append("Missing kernel `%s` required for loss seed" % loss_kernel_id)
            if loss_kernel_id not in bindings:
                issues.append("Missing bindings `%s` required for loss seed" % loss_kernel_id)

            grad_logits = canonical_grad_activation(logits_tensor)
            grad_for_tensor[logits_tensor] = grad_logits
            # aux.loss is a scalar reporting output from the loss seed op.
            # It is intentionally modeled as an aux tensor (not an activation slot)
            # and should be excluded from strict forward activation-oracle matching.
            _ensure_tensor(
                tensors,
                tensor_id="aux.loss",
                dtype="fp32",
                kind="aux",
                requires_grad=False,
                persistent=False,
                producer={"op_id": next_op_id, "output_name": "loss_out"},
                shape=[1],
                numel=1,
            )
            emit_backward_op(
                {
                    "phase": "backward",
                    "op": "loss_backward",
                    "kernel_id": loss_kernel_id,
                    "forward_ref": int(logits_op.get("op_id", -1)),
                    "layer": int(logits_op.get("layer", -1)),
                    "role": "loss_seed",
                    "dataflow": {
                        "inputs": {
                            "logits": {"tensor": logits_tensor, "kind": "activation"},
                            "targets": {"tensor": "input.targets", "kind": "input"}
                        },
                        "outputs": {
                            "d_logits": {"tensor": grad_logits, "kind": "grad_activation"},
                            "loss_out": {"tensor": "aux.loss", "kind": "aux"}
                        }
                    }
                }
            )
            grad_writers.setdefault(grad_logits, []).append(next_op_id - 1)

    # Synthesize backward ops by reverse traversal.
    for fwd in reversed(forward_ops):
        if fwd.get("phase") != "forward":
            continue
        if not fwd.get("requires_grad", False):
            continue

        outputs = fwd.get("dataflow", {}).get("outputs", {}) or {}
        output_grad_keys = []
        for out_name, out_ref in outputs.items():
            if not isinstance(out_ref, dict):
                continue
            tensor_id = out_ref.get("tensor")
            if isinstance(tensor_id, str) and tensor_id in grad_for_tensor:
                output_grad_keys.append(out_name)

        # If no downstream gradient reached this op, skip it.
        if not output_grad_keys:
            continue

        rule_name = fwd.get("grad_rule")
        if not isinstance(rule_name, str) or not rule_name:
            unresolved.append({
                "forward_op_id": fwd.get("op_id"),
                "op": fwd.get("op"),
                "reason": "missing_grad_rule"
            })
            continue

        rule = rules.get(rule_name)
        if not isinstance(rule, dict):
            unresolved.append({
                "forward_op_id": fwd.get("op_id"),
                "op": fwd.get("op"),
                "grad_rule": rule_name,
                "reason": "grad_rule_not_found"
            })
            continue

        status = str(rule.get("status", "todo"))
        if status != "ready":
            unresolved.append({
                "forward_op_id": fwd.get("op_id"),
                "op": fwd.get("op"),
                "grad_rule": rule_name,
                "reason": "grad_rule_not_ready",
                "detail": str(rule.get("reason", ""))
            })
            if not allow_partial:
                issues.append(
                    "Grad rule `%s` for op=%s (id=%s) is not ready: %s" % (
                        rule_name, fwd.get("op"), fwd.get("op_id"), rule.get("reason", "")
                    )
                )
            continue

        backward_specs = list(rule.get("backward_ops", []) or [])
        if not backward_specs:
            unresolved.append({
                "forward_op_id": fwd.get("op_id"),
                "op": fwd.get("op"),
                "grad_rule": rule_name,
                "reason": "empty_backward_ops"
            })
            continue

        save_for_backward = fwd.get("save_for_backward", {}) or {}
        fwd_inputs = fwd.get("dataflow", {}).get("inputs", {}) or {}
        fwd_outputs = fwd.get("dataflow", {}).get("outputs", {}) or {}
        fwd_weights = fwd.get("weights", {}) or {}

        for spec in backward_specs:
            kernel_id = _resolve_backward_kernel_id(spec, config)
            role = str(spec.get("role", "core"))
            rematerialize_saved_attn = False
            rematerialize_saved_attn_ref: Optional[Dict[str, Any]] = None

            if kernel_id not in kernels:
                issues.append("Missing kernel `%s` for backward op=%s" % (kernel_id, fwd.get("op")))
            if kernel_id not in bindings:
                issues.append("Missing bindings `%s` for backward op=%s" % (kernel_id, fwd.get("op")))

            bwd_inputs: Dict[str, Dict[str, Any]] = {}
            read_specs = list(spec.get("reads", []) or [])
            blocked = False
            for idx, read in enumerate(read_specs):
                key = "in_%d" % idx
                if not isinstance(read, str) or ":" not in read:
                    bwd_inputs[key] = {"tensor": str(read), "kind": "raw"}
                    continue
                kind, ref = read.split(":", 1)
                kind = kind.strip()
                ref = ref.strip()

                if kind == "grad":
                    out = fwd_outputs.get(ref)
                    if not isinstance(out, dict):
                        warnings.append("Backward read missing output `%s` in op_id=%s" % (ref, fwd.get("op_id")))
                        blocked = True
                        continue
                    src_tensor = out.get("tensor")
                    if not isinstance(src_tensor, str) or src_tensor not in grad_for_tensor:
                        warnings.append("No gradient available for tensor `%s` (op_id=%s)" % (src_tensor, fwd.get("op_id")))
                        blocked = True
                        continue
                    bwd_inputs[key] = {
                        "tensor": grad_for_tensor[src_tensor],
                        "kind": "grad_activation",
                        "from_forward_output": ref,
                    }
                elif kind == "saved":
                    s = save_for_backward.get(ref)
                    if not isinstance(s, dict):
                        warnings.append("save_for_backward `%s` missing for op_id=%s" % (ref, fwd.get("op_id")))
                        blocked = True
                        continue
                    if (
                        _checkpoint_recomputes_attention(checkpoint_policy)
                        and ref == "attn_weights"
                        and str(fwd.get("op", "") or "") in ("attn", "attn_sliding")
                    ):
                        rematerialize_saved_attn = True
                        rematerialize_saved_attn_ref = s
                    bwd_inputs[key] = {
                        "tensor": s.get("tensor"),
                        "kind": s.get("kind", "saved_activation"),
                        "from_saved": ref,
                    }
                elif kind == "saved_weight":
                    w = _find_weight_ref(fwd, ref)
                    if w is None:
                        warnings.append("saved_weight `%s` missing for op_id=%s" % (ref, fwd.get("op_id")))
                        blocked = True
                        continue
                    bwd_inputs[key] = {
                        "tensor": w.get("tensor"),
                        "kind": "weight",
                        "from_saved_weight": ref,
                        "param_name": w.get("name"),
                    }
                else:
                    bwd_inputs[key] = {"tensor": "%s:%s" % (kind, ref), "kind": kind}

            if blocked:
                # We could not materialize the required backward inputs for this spec.
                unresolved.append(
                    {
                        "forward_op_id": fwd.get("op_id"),
                        "op": fwd.get("op"),
                        "grad_rule": rule_name,
                        "kernel_id": kernel_id,
                        "reason": "missing_backward_input"
                    }
                )
                continue

            bwd_outputs: Dict[str, Dict[str, Any]] = {}
            writes = list(spec.get("writes", []) or [])
            for w in writes:
                if not isinstance(w, dict):
                    continue
                out_name = str(w.get("name", "out"))
                kind = str(w.get("kind", "aux"))
                if kind == "grad_activation":
                    target_input = str(w.get("target_input", ""))
                    inp = fwd_inputs.get(target_input)
                    if not isinstance(inp, dict):
                        unresolved.append(
                            {
                                "forward_op_id": fwd.get("op_id"),
                                "op": fwd.get("op"),
                                "reason": "missing_target_input_for_grad_activation",
                                "target_input": target_input,
                            }
                        )
                        continue
                    src_tensor = inp.get("tensor")
                    if not isinstance(src_tensor, str) or not src_tensor:
                        continue
                    # Backward kernels write into tmp grads first; accumulation into canonical
                    # grad.act.* tensors is emitted as explicit separate ops.
                    tmp_tensor = "tmp.grad.act.op%d.%s" % (int(fwd.get("op_id", -1)), _sanitize_tensor_id(target_input))
                    src_meta = tensors.get(src_tensor) if isinstance(src_tensor, str) else None
                    src_shape = src_meta.get("shape") if isinstance(src_meta, dict) else None
                    src_numel = _tensor_numel(tensors, src_tensor) or activation_default_numel
                    _ensure_tensor(
                        tensors,
                        tensor_id=tmp_tensor,
                        dtype="fp32",
                        kind="tmp_grad_activation",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
                        shape=src_shape if isinstance(src_shape, list) else None,
                        numel=src_numel,
                    )
                    bwd_outputs[out_name] = {
                        "tensor": tmp_tensor,
                        "kind": "tmp_grad_activation",
                        "target_input": target_input,
                        "source_tensor": src_tensor,
                    }
                elif kind in ("tmp_grad_weight", "grad_weight"):
                    target_weight = str(w.get("target_weight", ""))
                    weight_ref = fwd_weights.get(target_weight)
                    if not isinstance(weight_ref, dict):
                        # Some grad rules expose optional targets (for example d_bias on
                        # projection kernels where a model may not have an actual bias tensor).
                        # If the target is marked optional, skip emission without flagging
                        # unresolved coverage.
                        if bool(w.get("optional_target_weight", False)):
                            continue
                        unresolved.append(
                            {
                                "forward_op_id": fwd.get("op_id"),
                                "op": fwd.get("op"),
                                "reason": "missing_target_weight_for_grad_weight",
                                "target_weight": target_weight,
                            }
                        )
                        continue
                    param_name = str(weight_ref.get("name", target_weight))
                    tmp_tensor = "tmp.grad.weight.op%d.%s" % (int(fwd.get("op_id", -1)), _sanitize_tensor_id(target_weight))
                    src_wtid = str(weight_ref.get("tensor", ""))
                    src_wmeta = tensors.get(src_wtid) if isinstance(src_wtid, str) else None
                    src_wshape = src_wmeta.get("shape") if isinstance(src_wmeta, dict) else None
                    src_wnumel = _tensor_numel(tensors, src_wtid)
                    _ensure_tensor(
                        tensors,
                        tensor_id=tmp_tensor,
                        dtype="fp32",
                        kind="tmp_grad_weight",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
                        shape=src_wshape if isinstance(src_wshape, list) else None,
                        numel=src_wnumel,
                    )
                    bwd_outputs[out_name] = {
                        "tensor": tmp_tensor,
                        "kind": "tmp_grad_weight",
                        "target_weight": target_weight,
                        "param_name": param_name,
                    }
                else:
                    aux_tensor = str(w.get("tensor") or ("aux.op%d.%s" % (int(fwd.get("op_id", -1)), out_name)))
                    aux_shape: Optional[List[int]] = [activation_default_numel]
                    aux_numel = activation_default_numel
                    resolved_shape, resolved_numel = _shape_numel_from_write_source(
                        w,
                        fwd_inputs=fwd_inputs,
                        fwd_outputs=fwd_outputs,
                        fwd_weights=fwd_weights,
                        save_for_backward=save_for_backward,
                    )
                    if isinstance(resolved_shape, list) and resolved_shape:
                        aux_shape = resolved_shape
                    if isinstance(resolved_numel, int) and resolved_numel > 0:
                        aux_numel = resolved_numel
                    _ensure_tensor(
                        tensors,
                        tensor_id=aux_tensor,
                        dtype="fp32",
                        kind="aux",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
                        shape=aux_shape,
                        numel=aux_numel,
                    )
                    bwd_outputs[out_name] = {"tensor": aux_tensor, "kind": "aux"}

            if rematerialize_saved_attn and isinstance(rematerialize_saved_attn_ref, dict):
                remat = emit_checkpoint_rematerialize_saved_tensor(
                    fwd_op=fwd,
                    fwd_inputs=fwd_inputs,
                    saved_attn=rematerialize_saved_attn_ref,
                )
                if not isinstance(remat, dict):
                    issues.append(
                        "checkpoint_policy `%s` could not rematerialize attn_weights for op_id=%s"
                        % (checkpoint_policy, fwd.get("op_id"))
                    )
                    continue

            core = emit_backward_op(
                {
                    "phase": "backward",
                    "op": "%s_backward_%s" % (fwd.get("op"), role),
                    "kernel_id": kernel_id,
                    "forward_ref": int(fwd.get("op_id", -1)),
                    "layer": int(fwd.get("layer", -1)),
                    "section": fwd.get("section"),
                    "role": role,
                    "grad_rule": rule_name,
                    "dataflow": {"inputs": bwd_inputs, "outputs": bwd_outputs},
                }
            )
            attach_backward_bridge_plan(
                core,
                kernel_id=kernel_id,
                fwd_op=fwd,
                bwd_inputs=bwd_inputs,
                bwd_outputs=bwd_outputs,
            )

            # Emit explicit accumulation ops so fanout and weight updates remain visible
            # in IR2 rather than hidden inside fused backward kernels.
            for out_name, out_ref in bwd_outputs.items():
                if not isinstance(out_ref, dict):
                    continue
                if out_ref.get("kind") == "tmp_grad_activation":
                    src_tensor = str(out_ref.get("source_tensor", ""))
                    if not src_tensor:
                        continue
                    src_meta = fwd_inputs.get(out_ref.get("target_input"), {})
                    if isinstance(src_meta, dict) and src_meta.get("requires_grad") is False:
                        continue
                    dst_grad = canonical_grad_activation(src_tensor)
                    emit_accumulate(
                        src_tensor=str(out_ref["tensor"]),
                        dst_tensor=dst_grad,
                        forward_ref=int(fwd.get("op_id", -1)),
                        layer=int(fwd.get("layer", -1)),
                        role="activation",
                        target_param=None,
                    )
                    grad_for_tensor[src_tensor] = dst_grad
                elif out_ref.get("kind") == "tmp_grad_weight":
                    param_name = str(out_ref.get("param_name", ""))
                    if not param_name:
                        continue
                    dst_grad = canonical_grad_weight(param_name)
                    emit_accumulate(
                        src_tensor=str(out_ref["tensor"]),
                        dst_tensor=dst_grad,
                        forward_ref=int(fwd.get("op_id", -1)),
                        layer=int(fwd.get("layer", -1)),
                        role="weight",
                        target_param=param_name,
                    )

    if strict:
        # Strict mode only escalates unresolved rule coverage when partial mode is off.
        unresolved_ready = [
            x for x in unresolved
            if x.get("reason") in ("grad_rule_not_ready", "grad_rule_not_found", "missing_grad_rule")
        ]
        if unresolved_ready and not allow_partial:
            issues.append("Unresolved backward coverage for %d ops" % len(unresolved_ready))

    grad_weight_tensors = sorted([k for k, v in tensors.items() if isinstance(v, dict) and v.get("kind") == "grad_weight"])
    grad_act_tensors = sorted([k for k, v in tensors.items() if isinstance(v, dict) and v.get("kind") == "grad_activation"])
    backward_kernel_ops = [o for o in backward_ops if o.get("op") != "grad_accumulate"]
    accumulate_ops = [o for o in backward_ops if o.get("op") == "grad_accumulate"]

    if strict and issues:
        raise RuntimeError("IR2 backward synthesis failed:\n- " + "\n- ".join(issues))

    return {
        "format": "ir2-train-v7",
        "version": 1,
        "bridge_lowering": str(ir1.get("bridge_lowering", "legacy") or "legacy"),
        "checkpoint_policy": checkpoint_policy,
        "config": config,
        "template_name": ir1.get("template_name"),
        "num_layers": ir1.get("num_layers"),
        "forward": forward_ops,
        "backward": backward_ops,
        "tensors": tensors,
        "stats": {
            "forward_ops": len(forward_ops),
            "backward_ops": len(backward_ops),
            "backward_kernel_ops": len(backward_kernel_ops),
            "accumulate_ops": len(accumulate_ops),
            "checkpoint_rematerialize_ops": int(checkpoint_rematerialize_ops),
            "grad_activation_tensors": len(grad_act_tensors),
            "grad_weight_tensors": len(grad_weight_tensors),
            "issues": len(issues),
            "warnings": len(warnings),
            "unresolved": len(unresolved),
        },
        "checkpoint_summary": {
            "policy": checkpoint_policy,
            "checkpoint_rematerialize_ops": int(checkpoint_rematerialize_ops),
            "demoted_saved_tensors": sorted(checkpoint_demoted_saved_tensors),
        },
        "gradient_summary": {
            "grad_weight_tensors": grad_weight_tensors,
            "grad_activation_tensors": grad_act_tensors,
            "grad_weight_writers": grad_writers,
        },
        "unresolved": unresolved,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Lower IR1 train-forward -> IR2 backward train graph for v7.")
    ap.add_argument("--ir1", required=True, help="Input ir1 train-forward json")
    ap.add_argument("--output", required=True, help="Output ir2 json")
    ap.add_argument("--grad-rules", default=str(DEFAULT_GRAD_RULES_PATH), help="grad_rules_v7.json path")
    ap.add_argument("--checkpoint-policy", default="none", help="Checkpoint policy tag (none/recompute_attn)")
    ap.add_argument("--strict", action="store_true", help="Fail on unresolved backward coverage")
    ap.add_argument("--allow-partial", action="store_true", help="Allow unresolved TODO grad rules")
    ap.add_argument("--summary-out", default=None, help="Optional compact summary json")
    args = ap.parse_args()

    ir1 = _load_json(Path(args.ir1))
    registry = _load_json(KERNEL_REGISTRY_PATH)
    bindings_doc = _load_json(KERNEL_BINDINGS_PATH)
    grad_rules = _load_json(Path(args.grad_rules))

    try:
        ir2 = synthesize_ir2_backward(
            ir1=ir1,
            registry=registry,
            bindings_doc=bindings_doc,
            grad_rules=grad_rules,
            strict=bool(args.strict),
            allow_partial=bool(args.allow_partial),
            checkpoint_policy=str(args.checkpoint_policy),
        )
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    _save_json(Path(args.output), ir2)
    print(
        "Wrote IR2 train backward: %s (forward=%d backward=%d unresolved=%d)" % (
            args.output,
            len(ir2.get("forward", [])),
            len(ir2.get("backward", [])),
            len(ir2.get("unresolved", [])),
        )
    )

    if args.summary_out:
        summary = {
            "format": ir2.get("format"),
            "stats": ir2.get("stats", {}),
            "issues": ir2.get("issues", []),
            "warnings": ir2.get("warnings", []),
            "unresolved": ir2.get("unresolved", []),
            "gradient_summary": ir2.get("gradient_summary", {}),
        }
        _save_json(Path(args.summary_out), summary)
        print("Wrote summary:", args.summary_out)

    if ir2.get("issues"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Audit v8 generated circuit artifacts for template/dataflow consistency.

This validates the graph after IR generation and lowering. It is intentionally
artifact-first: if ir1/lowered JSON is wrong, generated C will be wrong too.
The optional C audit is only a final codegen-preservation check.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ops(doc: dict[str, Any]) -> list[dict[str, Any]]:
    ops = doc.get("ops") or doc.get("operations") or []
    return ops if isinstance(ops, list) else []


def _template_op_items(section: Any) -> list[dict[str, Any]]:
    if not isinstance(section, list):
        return []
    out: list[dict[str, Any]] = []
    for item in section:
        if isinstance(item, str):
            out.append({"op": item})
        elif isinstance(item, dict) and isinstance(item.get("op"), str):
            out.append(item)
    return out


CRITICAL_TEMPLATE_INPUTS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("x",),
    "q_proj": ("x",),
    "q_gate_proj": ("x",),
    "k_proj": ("x",),
    "v_proj": ("x",),
    "kv_a_proj": ("x",),
    "kv_a_layernorm": ("input",),
    "kv_lora_decompress": ("compressed_kv",),
    "partial_rope_concat": ("q_packed", "k_nope", "k_pe"),
    "mla_attention": ("query", "key", "value"),
    "mlp_gate_up": ("x",),
    "moe_swiglu_expert_mlp": ("hidden", "indices", "routing_weights"),
    "shared_swiglu_expert_mlp": ("hidden", "routed"),
    "full_softmax_topk_router": ("logits",),
    "gated_shared_swiglu_expert_mlp": ("hidden", "routed"),
    "mlp_up": ("x",),
    "mamba_in_proj": ("x",),
    "recurrent_qkv_proj": ("x",),
    "recurrent_gate_proj": ("x",),
    "recurrent_alpha_proj": ("x",),
    "recurrent_beta_proj": ("x",),
    "out_proj": ("x",),
    "mlp_down": ("x",),
    "mamba_out_proj": ("x",),
    "recurrent_out_proj": ("x",),
    "logits": ("x",),
}


def audit_template_explicit_edges(template: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    explicit: list[str] = []
    ignored: list[str] = []
    blocks = template.get("block_types") if isinstance(template.get("block_types"), dict) else {}
    for block_name, block in blocks.items():
        if not isinstance(block, dict):
            continue
        sections: list[tuple[str, Any]] = [("header", block.get("header")), ("footer", block.get("footer"))]
        body = block.get("body")
        if isinstance(body, dict):
            sections.append(("body", body.get("ops")))
            ops_by_kind = body.get("ops_by_kind")
            if isinstance(ops_by_kind, dict):
                for kind, ops in ops_by_kind.items():
                    sections.append((f"body:{kind}", ops))
        else:
            sections.append(("body", body))
        for section_name, raw_ops in sections:
            for index, item in enumerate(_template_op_items(raw_ops)):
                op = str(item.get("op") or "")
                required = CRITICAL_TEMPLATE_INPUTS.get(op)
                if not required:
                    ignored.append(f"{block_name}.{section_name}[{index}].{op}")
                    continue
                graph_slots = item.get("graph_slots") if isinstance(item.get("graph_slots"), dict) else {}
                inputs = graph_slots.get("inputs") if isinstance(graph_slots.get("inputs"), dict) else {}
                for input_name in required:
                    label = f"{block_name}.{section_name}[{index}].{op}.{input_name}"
                    if input_name in inputs:
                        explicit.append(f"{label}={inputs[input_name]}")
                    else:
                        missing.append(label)
    return {
        "template": template.get("name"),
        "explicit_count": len(explicit),
        "missing_count": len(missing),
        "explicit": explicit,
        "missing": missing,
    }


def _op_id(op: dict[str, Any]) -> int:
    return int(op.get("op_id", op.get("idx", -1)))


def _layer(op: dict[str, Any]) -> int | None:
    if "layer" not in op or op.get("layer") is None:
        return None
    try:
        return int(op.get("layer"))
    except Exception:
        return None


def _input_ref(op: dict[str, Any], name: str) -> dict[str, Any]:
    return (((op.get("dataflow") or {}).get("inputs") or {}).get(name) or {})


def _lookup_tensor(mapping: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        item = mapping.get(name)
        if isinstance(item, dict):
            return item
    return {}


def _input_buffer(op: dict[str, Any], name: str) -> str:
    aliases = {
        "x": ("x", "input", "A"),
        "projected": ("projected", "x", "A"),
        "dt": ("dt", "x", "A"),
        "gate": ("gate", "z"),
    }
    names = aliases.get(name, (name,))
    return str((_lookup_tensor(op.get("activations") or {}, names).get("buffer") or ""))


def _output_buffer(op: dict[str, Any], name: str) -> str:
    aliases = {"y": ("y", "out", "C", "output"), "out": ("out", "y", "C", "output")}
    names = aliases.get(name, (name,))
    return str((_lookup_tensor(op.get("outputs") or {}, names).get("buffer") or ""))


SEMANTIC_FP32_BUFFERS = {
    "embedded_input",
    "main_stream",
    "residual",
    "layer_output",
    "attn_scratch",
    "mlp_scratch",
    "logits",
}


def _audit_physical_activation_views(ops: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for op in ops:
        idx = op.get("idx", op.get("op_id", "?"))
        op_name = str(op.get("op") or "")
        layer = op.get("layer", "?")
        for arg_name, spec in (op.get("activations") or {}).items():
            if not isinstance(spec, dict):
                continue
            dtype = str(spec.get("dtype") or "")
            buffer = str(spec.get("buffer") or "")
            if dtype.startswith("q") and buffer in SEMANTIC_FP32_BUFFERS:
                errors.append(
                    f"op {idx} layer {layer} {op_name}.{arg_name}: quantized activation dtype={dtype} "
                    f"is bound to semantic FP32 buffer={buffer}; expected a quantized physical view such as layer_input/main_stream_q8"
                )
    return errors


def _by_layer_name(ops: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for op in ops:
        layer = _layer(op)
        name = str(op.get("op") or "")
        if layer is None or layer < 0 or not name:
            continue
        out.setdefault((layer, name), op)
    return out


def _same_producer(errors: list[str], consumer: dict[str, Any], input_name: str, producer: dict[str, Any], label: str) -> None:
    ref = _input_ref(consumer, input_name)
    if int(ref.get("from_op", -999999)) != _op_id(producer):
        errors.append(
            f"{label}: {consumer.get('op')}.{input_name} comes from op {ref.get('from_op')} "
            f"but expected {producer.get('op')} op {_op_id(producer)}"
        )


def _slot(errors: list[str], op: dict[str, Any], input_name: str, expected: str, label: str) -> None:
    ref = _input_ref(op, input_name)
    if str(ref.get("slot") or "") != expected:
        errors.append(f"{label}: {op.get('op')}.{input_name} slot={ref.get('slot')} expected={expected}")


def audit_ir1(ir1: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ops = _ops(ir1)
    by = _by_layer_name(ops)
    layers = sorted({layer for layer, _ in by})
    for layer in layers:
        norm = by.get((layer, "block_rmsnorm"))
        residual_save = by.get((layer, "residual_save"))
        residual_add = by.get((layer, "residual_add"))
        if norm is None:
            continue
        label = f"layer {layer}"
        for proj_name in ("mamba_in_proj", "q_proj", "q_gate_proj", "k_proj", "v_proj", "kv_a_proj", "mlp_up", "mlp_gate_up", "moe_router"):
            proj = by.get((layer, proj_name))
            if proj is not None:
                _same_producer(errors, proj, "x", norm, f"{label} pre-norm projection")
                _slot(errors, proj, "x", "layer_input", f"{label} pre-norm projection")
        split = by.get((layer, "mamba_in_proj_split"))
        mamba_in = by.get((layer, "mamba_in_proj"))
        if split is not None and mamba_in is not None:
            _same_producer(errors, split, "projected", mamba_in, f"{label} mamba split")
        conv = by.get((layer, "mamba_conv1d_silu"))
        if conv is not None and split is not None:
            _same_producer(errors, conv, "x", split, f"{label} mamba conv")
        dt = by.get((layer, "mamba_dt_softplus"))
        if dt is not None and split is not None:
            _same_producer(errors, dt, "dt", split, f"{label} mamba dt")
        scan = by.get((layer, "mamba_selective_scan"))
        if scan is not None:
            if conv is not None:
                for input_name in ("x", "B", "C"):
                    _same_producer(errors, scan, input_name, conv, f"{label} mamba scan")
            if dt is not None:
                _same_producer(errors, scan, "dt", dt, f"{label} mamba scan")
        gate_norm = by.get((layer, "mamba_rmsnorm_gate"))
        if gate_norm is not None:
            if scan is not None:
                _same_producer(errors, gate_norm, "x", scan, f"{label} mamba gated norm")
            if split is not None:
                _same_producer(errors, gate_norm, "gate", split, f"{label} mamba gated norm")
        mamba_out = by.get((layer, "mamba_out_proj"))
        if mamba_out is not None and gate_norm is not None:
            _same_producer(errors, mamba_out, "x", gate_norm, f"{label} mamba out")
        relu2 = by.get((layer, "relu2"))
        mlp_up = by.get((layer, "mlp_up"))
        if relu2 is not None and mlp_up is not None:
            _same_producer(errors, relu2, "x", mlp_up, f"{label} mlp relu2")
        mlp_down = by.get((layer, "mlp_down"))
        if mlp_down is not None and relu2 is not None:
            _same_producer(errors, mlp_down, "x", relu2, f"{label} mlp down")
        q = by.get((layer, "q_proj")); k = by.get((layer, "k_proj")); v = by.get((layer, "v_proj"))
        kv_a = by.get((layer, "kv_a_proj"))
        kv_norm = by.get((layer, "kv_a_layernorm"))
        kv_decomp = by.get((layer, "kv_lora_decompress"))
        partial = by.get((layer, "partial_rope_concat"))
        mla = by.get((layer, "mla_attention"))
        if kv_a is not None:
            _same_producer(errors, kv_a, "x", norm, f"{label} MLA kv_a")
            _slot(errors, kv_a, "x", "layer_input", f"{label} MLA kv_a")
        if kv_norm is not None and kv_a is not None:
            _same_producer(errors, kv_norm, "x", kv_a, f"{label} MLA kv_a norm")
        if kv_decomp is not None and kv_norm is not None:
            _same_producer(errors, kv_decomp, "compressed_kv", kv_norm, f"{label} MLA kv decompress")
        if partial is not None:
            if q is not None:
                _same_producer(errors, partial, "q_packed", q, f"{label} MLA partial rope")
            if kv_decomp is not None:
                _same_producer(errors, partial, "k_nope", kv_decomp, f"{label} MLA partial rope")
            if kv_a is not None:
                _same_producer(errors, partial, "k_pe", kv_a, f"{label} MLA partial rope")
        if mla is not None:
            if partial is not None:
                _same_producer(errors, mla, "query", partial, f"{label} MLA attention")
                _same_producer(errors, mla, "key", partial, f"{label} MLA attention")
            if kv_decomp is not None:
                _same_producer(errors, mla, "value", kv_decomp, f"{label} MLA attention")
        rope = by.get((layer, "rope_qk"))
        if rope is not None:
            if q is not None:
                _same_producer(errors, rope, "q", q, f"{label} rope")
            if k is not None:
                _same_producer(errors, rope, "k", k, f"{label} rope")
        attn = by.get((layer, "attn")) or by.get((layer, "attn_sliding")) or by.get((layer, "mla_attention"))
        if attn is not None:
            if rope is not None:
                _same_producer(errors, attn, "q", rope, f"{label} attention")
                k_ref = _input_ref(attn, "k")
                if k_ref.get("slot") == "kv_cache":
                    pass  # decode path reads K from the explicit KV cache stream inserted during lowering
                else:
                    _same_producer(errors, attn, "k", rope, f"{label} attention")
            if v is not None:
                v_ref = _input_ref(attn, "v")
                if v_ref.get("slot") == "kv_cache":
                    pass  # decode path reads V from the explicit KV cache stream inserted during lowering
                else:
                    _same_producer(errors, attn, "v", v, f"{label} attention")
        out_proj = by.get((layer, "out_proj"))
        if out_proj is not None and attn is not None:
            _same_producer(errors, out_proj, "x", attn, f"{label} attention out")
        if residual_add is not None and residual_save is not None:
            residual_ref = _input_ref(residual_add, "residual")
            if residual_ref and int(residual_ref.get("from_op", -999999)) != _op_id(residual_save):
                errors.append(f"{label} residual_add.residual does not consume residual_save")
    return errors


def audit_lowered(lowered: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ops = _ops(lowered)
    errors.extend(_audit_physical_activation_views(ops))
    by = _by_layer_name(ops)
    layers = sorted({layer for layer, _ in by})
    for layer in layers:
        label = f"layer {layer}"
        for proj_name in ("mamba_in_proj", "q_proj", "q_gate_proj", "k_proj", "v_proj", "kv_a_proj", "mlp_up", "mlp_gate_up", "moe_router"):
            proj = by.get((layer, proj_name))
            if proj is not None:
                buf = _input_buffer(proj, "x")
                if buf != "layer_input":
                    errors.append(f"{label}: lowered {proj_name}.x buffer={buf} expected=layer_input")
        checks = [
            ("mamba_in_proj_split", "projected", "recurrent_packed"),
            ("mamba_conv1d_silu", "x", "recurrent_conv_qkv"),
            ("mamba_dt_softplus", "dt", "recurrent_g"),
            ("mamba_selective_scan", "x", "recurrent_conv_qkv"),
            ("mamba_selective_scan", "dt", "recurrent_g"),
            ("mamba_rmsnorm_gate", "x", "recurrent_v"),
            ("mamba_rmsnorm_gate", "gate", "recurrent_z"),
            ("mamba_out_proj", "x", "recurrent_normed"),
            ("relu2", "x", "mlp_scratch"),
            ("mlp_down", "x", "mlp_scratch"),
        ]
        for op_name, input_name, expected in checks:
            op = by.get((layer, op_name))
            if op is not None:
                buf = _input_buffer(op, input_name)
                if buf != expected:
                    errors.append(f"{label}: lowered {op_name}.{input_name} buffer={buf} expected={expected}")
    return errors


def audit_c_source(c_path: Path) -> list[str]:
    if not c_path:
        return []
    text = c_path.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    pattern = re.compile(
        r"/\* Op \d+: gemv_bf16 \(mamba_in_proj\) layer=(\d+).*?gemv_bf16\(.*?W_LAYER_\1_MAMBA_IN_PROJ\),\s*\n\s*\(const float\*\)\(model->bump \+ ([^)]+)\),",
        re.S,
    )
    for layer, arg in pattern.findall(text):
        if arg.strip() != "A_LAYER_INPUT":
            errors.append(f"generated C layer {layer}: mamba_in_proj input={arg.strip()} expected=A_LAYER_INPUT")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, help="v8 template JSON to audit for explicit circuit edges")
    ap.add_argument("--require-explicit-template-edges", action="store_true", help="fail if critical template op inputs rely on implicit defaults")
    ap.add_argument("--ir1", type=Path, help="ir1_*.json artifact")
    ap.add_argument("--lowered", type=Path, help="lowered_*.json artifact")
    ap.add_argument("--c-source", type=Path, help="generated model_v8.c artifact")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    if args.template:
        template_report = audit_template_explicit_edges(_load_json(args.template))
        checks["template_explicit_edges"] = template_report
        if args.require_explicit_template_edges and template_report["missing_count"]:
            errors.extend([f"template missing explicit input edge: {item}" for item in template_report["missing"]])
    if args.ir1:
        e = audit_ir1(_load_json(args.ir1)); errors.extend(e); checks["ir1_errors"] = e
    if args.lowered:
        e = audit_lowered(_load_json(args.lowered)); errors.extend(e); checks["lowered_errors"] = e
    if args.c_source:
        e = audit_c_source(args.c_source); errors.extend(e); checks["c_errors"] = e
    report = {"status": "pass" if not errors else "fail", "error_count": len(errors), "errors": errors, "checks": checks}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

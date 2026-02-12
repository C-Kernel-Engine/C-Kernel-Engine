#!/usr/bin/env python3
"""
lower_ir2_backward_v7.py

Synthesize IR2 (forward + backward) from IR1 train-forward:
- reverse traversal of forward ops
- grad-rule driven backward kernel expansion
- explicit grad accumulation ops for activations and weights
- fail-fast contracts (optional strict mode)
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = V7_ROOT / "kernel_maps" / "kernel_bindings.json"
DEFAULT_GRAD_RULES_PATH = SCRIPT_DIR / "grad_rules_v7.json"


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
    producer: Optional[Dict[str, Any]] = None
) -> None:
    if tensor_id in tensors:
        return
    tensors[tensor_id] = {
        "dtype": dtype,
        "kind": kind,
        "requires_grad": bool(requires_grad),
        "persistent": bool(persistent),
        "producer": producer,
    }


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

    forward_ops = list(ir1.get("ops", []))
    tensors: Dict[str, Dict[str, Any]] = deepcopy(ir1.get("tensors", {}))

    backward_ops: List[Dict[str, Any]] = []
    issues: List[str] = []
    warnings: List[str] = []
    unresolved: List[Dict[str, Any]] = []

    next_op_id = 0
    if forward_ops:
        next_op_id = max(int(o.get("op_id", 0)) for o in forward_ops) + 1

    grad_for_tensor: Dict[str, str] = {}
    grad_writers: Dict[str, List[int]] = {}

    def emit_backward_op(op: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal next_op_id
        op["op_id"] = next_op_id
        backward_ops.append(op)
        next_op_id += 1
        return op

    def canonical_grad_activation(source_tensor: str) -> str:
        safe = _sanitize_tensor_id(source_tensor)
        tid = "grad.act.%s" % safe
        _ensure_tensor(
            tensors,
            tensor_id=tid,
            dtype="fp32",
            kind="grad_activation",
            requires_grad=False,
            persistent=True,
            producer=None,
        )
        return tid

    def canonical_grad_weight(param_name: str) -> str:
        safe = _sanitize_tensor_id(param_name)
        tid = "grad.weight.%s" % safe
        _ensure_tensor(
            tensors,
            tensor_id=tid,
            dtype="fp32",
            kind="grad_weight",
            requires_grad=False,
            persistent=True,
            producer=None,
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

    # Seed backward graph from explicit loss gradient on final logits.
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
            _ensure_tensor(
                tensors,
                tensor_id="aux.loss",
                dtype="fp32",
                kind="aux",
                requires_grad=False,
                persistent=False,
                producer={"op_id": next_op_id, "output_name": "loss_out"},
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
            kernel_id = str(spec.get("kernel_id", ""))
            role = str(spec.get("role", "core"))

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
                    tmp_tensor = "tmp.grad.act.op%d.%s" % (int(fwd.get("op_id", -1)), _sanitize_tensor_id(target_input))
                    _ensure_tensor(
                        tensors,
                        tensor_id=tmp_tensor,
                        dtype="fp32",
                        kind="tmp_grad_activation",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
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
                    _ensure_tensor(
                        tensors,
                        tensor_id=tmp_tensor,
                        dtype="fp32",
                        kind="tmp_grad_weight",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
                    )
                    bwd_outputs[out_name] = {
                        "tensor": tmp_tensor,
                        "kind": "tmp_grad_weight",
                        "target_weight": target_weight,
                        "param_name": param_name,
                    }
                else:
                    aux_tensor = str(w.get("tensor") or ("aux.op%d.%s" % (int(fwd.get("op_id", -1)), out_name)))
                    _ensure_tensor(
                        tensors,
                        tensor_id=aux_tensor,
                        dtype="fp32",
                        kind="aux",
                        requires_grad=False,
                        persistent=False,
                        producer={"op_id": next_op_id, "output_name": out_name},
                    )
                    bwd_outputs[out_name] = {"tensor": aux_tensor, "kind": "aux"}

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

            # Emit explicit accumulation ops.
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
        "checkpoint_policy": checkpoint_policy,
        "config": deepcopy(ir1.get("config", {})),
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
            "grad_activation_tensors": len(grad_act_tensors),
            "grad_weight_tensors": len(grad_weight_tensors),
            "issues": len(issues),
            "warnings": len(warnings),
            "unresolved": len(unresolved),
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
    ap.add_argument("--checkpoint-policy", default="none", help="Checkpoint policy tag (none/every_n_layers/...)")
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

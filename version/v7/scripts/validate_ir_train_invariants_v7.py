#!/usr/bin/env python3
"""
validate_ir_train_invariants_v7.py

Validate v7 train IR artifacts (IR1 + IR2):
- backward ops resolve to registered + bound kernels
- grad accumulation contracts are explicit
- trainable weights have grad.weight tensors
- unresolved coverage policy is enforced
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
KERNEL_REGISTRY_PATH = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_BINDINGS_PATH = V7_ROOT / "kernel_maps" / "kernel_bindings.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _kernel_ids(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for kernel in registry.get("kernels", []):
        kid = kernel.get("id")
        if isinstance(kid, str):
            out[kid] = kernel
    return out


def _binding_ids(bindings_doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(bindings_doc.get("bindings", {}))


def validate(ir1: Dict[str, Any], ir2: Dict[str, Any], strict_unresolved: bool, allow_partial: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    registry = _load_json(KERNEL_REGISTRY_PATH)
    bindings_doc = _load_json(KERNEL_BINDINGS_PATH)
    kernels = _kernel_ids(registry)
    bindings = _binding_ids(bindings_doc)

    backward_ops = [op for op in ir2.get("backward", []) if isinstance(op, dict)]
    trainable_weights = []
    for tname, tmeta in (ir1.get("tensors", {}) or {}).items():
        if not isinstance(tmeta, dict):
            continue
        if tmeta.get("kind") == "weight" and tmeta.get("requires_grad") is True:
            # tname is "weight.<param_name>"
            param = str(tname).replace("weight.", "", 1)
            trainable_weights.append(param)

    # Invariant 1: every backward op has kernel id and kernel+binding present.
    missing_kernel = []
    missing_binding = []
    kernel_less = []
    for op in backward_ops:
        kid = op.get("kernel_id")
        if not isinstance(kid, str) or not kid:
            kernel_less.append(op.get("op_id"))
            continue
        if kid not in kernels:
            missing_kernel.append({"op_id": op.get("op_id"), "kernel_id": kid})
        if kid not in bindings:
            missing_binding.append({"op_id": op.get("op_id"), "kernel_id": kid})
    if kernel_less or missing_kernel:
        failures.append("backward_ops_missing_kernel_or_registry")
    if missing_binding:
        failures.append("backward_ops_missing_bindings")
    rows.append(
        {
            "name": "kernel_coverage",
            "status": "PASS" if not (kernel_less or missing_kernel or missing_binding) else "FAIL",
            "kernel_less": kernel_less,
            "missing_kernel": missing_kernel,
            "missing_binding": missing_binding,
        }
    )

    # Invariant 2: grad accumulate ops must target explicit dst grad tensor.
    bad_accum = []
    grad_weight_writers: Dict[str, int] = {}
    for op in backward_ops:
        if op.get("op") != "grad_accumulate":
            continue
        outputs = op.get("dataflow", {}).get("outputs", {}) or {}
        dst = outputs.get("dst", {}) if isinstance(outputs, dict) else {}
        dst_tensor = dst.get("tensor") if isinstance(dst, dict) else None
        if not isinstance(dst_tensor, str) or not dst_tensor:
            bad_accum.append({"op_id": op.get("op_id"), "reason": "missing_dst_tensor"})
            continue
        if dst_tensor.startswith("grad.weight."):
            grad_weight_writers[dst_tensor] = grad_weight_writers.get(dst_tensor, 0) + 1
    if bad_accum:
        failures.append("invalid_grad_accumulate_ops")
    rows.append(
        {
            "name": "accumulate_contract",
            "status": "PASS" if not bad_accum else "FAIL",
            "bad_accumulate_ops": bad_accum,
        }
    )

    # Invariant 3: every trainable weight should have a grad.weight tensor.
    missing_grad_weight = []
    ir2_tensors = ir2.get("tensors", {}) or {}
    for param in sorted(set(trainable_weights)):
        tid = "grad.weight.%s" % param.replace("/", "_")
        if tid not in ir2_tensors:
            # fallback: allow sanitized dots exactly as lower script emits.
            tid2 = "grad.weight.%s" % param
            if tid2 not in ir2_tensors:
                missing_grad_weight.append(param)
                continue
        # soft warning if no writer
        if tid not in grad_weight_writers and ("grad.weight.%s" % param) not in grad_weight_writers:
            warnings.append("No grad_accumulate writer observed for %s" % param)
    if missing_grad_weight:
        if allow_partial:
            warnings.append("Missing grad.weight tensors for partial mode: %d" % len(missing_grad_weight))
        else:
            failures.append("missing_grad_weight_tensors")
    rows.append(
        {
            "name": "grad_weight_coverage",
            "status": "PASS" if not missing_grad_weight else ("WARN" if allow_partial else "FAIL"),
            "trainable_weights": len(set(trainable_weights)),
            "missing_grad_weight_tensors": missing_grad_weight,
        }
    )

    # Invariant 4: unresolved policy.
    unresolved = ir2.get("unresolved", []) or []
    unresolved_count = len(unresolved)
    unresolved_status = "PASS"
    if unresolved_count > 0:
        if strict_unresolved:
            unresolved_status = "FAIL"
            failures.append("unresolved_backward_coverage")
        else:
            unresolved_status = "WARN"
            warnings.append("Unresolved backward coverage present: %d" % unresolved_count)
    rows.append(
        {
            "name": "unresolved_policy",
            "status": unresolved_status,
            "count": unresolved_count,
        }
    )

    passed = len(failures) == 0
    return {
        "format": "v7-train-ir-invariants",
        "passed": passed,
        "strict_unresolved": bool(strict_unresolved),
        "allow_partial": bool(allow_partial),
        "checks": rows,
        "failures": failures,
        "warnings": warnings,
        "stats": {
            "backward_ops": len(backward_ops),
            "trainable_weights": len(set(trainable_weights)),
            "unresolved": unresolved_count,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v7 train IR invariants.")
    ap.add_argument("--ir1", required=True, help="ir1 train-forward json")
    ap.add_argument("--ir2", required=True, help="ir2 backward json")
    ap.add_argument("--output", required=True, help="Output invariant report json")
    ap.add_argument("--strict-unresolved", action="store_true", help="Fail if unresolved backward coverage exists")
    ap.add_argument("--allow-partial", action="store_true", help="Allow partial backward coverage (warn instead of fail for missing grad.weight)")
    args = ap.parse_args()

    ir1 = _load_json(Path(args.ir1))
    ir2 = _load_json(Path(args.ir2))
    report = validate(
        ir1=ir1,
        ir2=ir2,
        strict_unresolved=bool(args.strict_unresolved),
        allow_partial=bool(args.allow_partial),
    )
    _save_json(Path(args.output), report)

    print(
        "v7 train IR invariants: %s (checks=%d failures=%d warnings=%d)" % (
            "PASS" if report.get("passed") else "FAIL",
            len(report.get("checks", [])),
            len(report.get("failures", [])),
            len(report.get("warnings", [])),
        )
    )
    if report.get("failures"):
        for item in report["failures"]:
            print("  FAIL:", item)
    if report.get("warnings"):
        for item in report["warnings"]:
            print("  WARN:", item)

    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())

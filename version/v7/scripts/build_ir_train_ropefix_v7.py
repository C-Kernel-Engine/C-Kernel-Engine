#!/usr/bin/env python3
"""
build_ir_train_ropefix_v7.py

Narrow train-IR variant used to validate rope_layout-driven RoPE lowering
without changing the stable training builder in-place.

Workflow:
- prove qwen-family control cases remain unchanged
- prove llama/nanbeige-family manifests lower pairwise RoPE correctly
- merge the validated change back into build_ir_train_v7.py
- keep this file as reference/legacy bring-up history
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import build_ir_train_v7 as stable
import build_ir_v7


def _resolved_template_and_rope_kernel(
    manifest: Dict[str, Any],
    explicit_template: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    if explicit_template is not None:
        template = copy.deepcopy(stable._load_json(explicit_template))
    else:
        template = copy.deepcopy(stable._choose_template(manifest, explicit_template=None))
    template_kernels = dict(template.get("kernels") or {})
    rope_kernel = build_ir_v7._resolve_rope_qk_kernel(
        dict(manifest.get("config") or {}),
        template_kernels,
    )
    template["kernels"] = dict(template_kernels)
    template["kernels"]["rope_qk"] = rope_kernel
    return template, rope_kernel


def build_ir1_train_ropefix(
    manifest: Dict[str, Any],
    registry: Dict[str, Any],
    bindings_doc: Dict[str, Any],
    grad_rules: Dict[str, Any],
    max_layers: Optional[int],
    strict: bool,
    explicit_template: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_for_build = copy.deepcopy(manifest)
    template, rope_kernel = _resolved_template_and_rope_kernel(
        manifest_for_build,
        explicit_template=explicit_template,
    )
    manifest_for_build["template"] = template

    previous_rope_kernel = stable.FORWARD_KERNEL_BY_OP.get("rope_qk", "rope_forward_qk")
    stable.FORWARD_KERNEL_BY_OP["rope_qk"] = rope_kernel
    try:
        ir1 = stable.build_ir1_train(
            manifest=manifest_for_build,
            registry=registry,
            bindings_doc=bindings_doc,
            grad_rules=grad_rules,
            max_layers=max_layers,
            strict=strict,
        )
    finally:
        stable.FORWARD_KERNEL_BY_OP["rope_qk"] = previous_rope_kernel

    ropefix_meta = dict(ir1.get("ropefix") or {})
    ropefix_meta.update(
        {
            "status": "validated_variant",
            "rope_qk_kernel": rope_kernel,
        }
    )
    ir1["ropefix"] = ropefix_meta
    return ir1


def main() -> int:
    ap = argparse.ArgumentParser(description="Build train IR1 with rope_layout-driven rope_qk kernel selection.")
    ap.add_argument("--manifest", required=True, help="weights_manifest.json path")
    ap.add_argument("--output", required=True, help="Output ir1_train_forward.json")
    ap.add_argument("--grad-rules", default=str(stable.DEFAULT_GRAD_RULES_PATH), help="grad_rules_v7.json path")
    ap.add_argument("--template", default=None, help="Optional template override path")
    ap.add_argument("--max-layers", type=int, default=None, help="Optional cap for fast smoke runs")
    ap.add_argument("--tokens", type=int, default=1, help="Compile-time token count for train IR/runtime (default: 1)")
    ap.add_argument("--strict", action="store_true", help="Fail on unresolved weights/save-for-backward")
    ap.add_argument("--report-out", default=None, help="Optional report JSON path")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    grad_rules_path = Path(args.grad_rules)
    explicit_template = Path(args.template) if args.template else None

    manifest = stable._load_json(manifest_path)
    if not isinstance(manifest.get("config"), dict):
        manifest["config"] = {}
    manifest["config"]["train_tokens"] = max(1, int(args.tokens or 1))
    registry = stable._load_json(stable.KERNEL_REGISTRY_PATH)
    bindings_doc = stable._load_json(stable.KERNEL_BINDINGS_PATH)
    grad_rules = stable._load_json(grad_rules_path)

    ir1 = build_ir1_train_ropefix(
        manifest=manifest,
        registry=registry,
        bindings_doc=bindings_doc,
        grad_rules=grad_rules,
        max_layers=args.max_layers,
        strict=args.strict,
        explicit_template=explicit_template,
    )
    stable._save_json(output_path, ir1)
    print("Wrote IR1 train-forward (ropefix): %s (ops=%d tensors=%d)" % (output_path, len(ir1["ops"]), len(ir1["tensors"])))

    if args.report_out:
        report = {
            "format": ir1.get("format"),
            "ops": len(ir1.get("ops", [])),
            "tensors": len(ir1.get("tensors", {})),
            "issues": ir1.get("issues", []),
            "warnings": ir1.get("warnings", []),
            "stats": ir1.get("stats", {}),
            "ropefix": ir1.get("ropefix", {}),
        }
        stable._save_json(Path(args.report_out), report)
        print("Wrote report: %s" % args.report_out)

    if ir1.get("issues"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

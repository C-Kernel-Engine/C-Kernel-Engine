#!/usr/bin/env python3
"""
Static v7 contract checker.

This is intentionally lightweight and deterministic:
- checks required v7 contract docs exist
- checks required backward-capable symbols exist in kernel sources
- checks required training kernel-map coverage (registry + bindings)
- checks v7 parity harness script exists
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


ROOT = Path(__file__).resolve().parents[3]
V7 = ROOT / "version" / "v7"
SRC_KERNELS = ROOT / "src" / "kernels"


@dataclass
class Row:
    layer: str
    handoff: str
    contract: str
    status: str
    notes: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_kernel_source() -> str:
    parts: List[str] = []
    for cfile in sorted(SRC_KERNELS.glob("*.c")):
        try:
            parts.append(_read_text(cfile))
        except Exception:
            continue
    return "\n".join(parts)


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _has_symbol(src: str, symbol: str) -> bool:
    return re.search(r"\b" + re.escape(symbol) + r"\b", src) is not None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return None


def _render_table(rows: List[Row]) -> str:
    headers = ["Layer", "Handoff", "Contract", "Status", "Notes"]
    matrix = [headers] + [[r.layer, r.handoff, r.contract, r.status, r.notes] for r in rows]
    widths = [max(len(str(row[i])) for row in matrix) for i in range(len(headers))]

    def _fmt(row: List[str]) -> str:
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [_fmt(headers), sep]
    for r in rows:
        out.append(_fmt([r.layer, r.handoff, r.contract, r.status, r.notes]))
    return "\n".join(out)


def _status_counts(rows: List[Row]) -> dict:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def run_checks() -> List[Row]:
    rows: List[Row] = []

    # L1: contract docs present
    required_docs = [
        V7 / "contracts" / "IR_CONTRACT.md",
        V7 / "contracts" / "KERNEL_CONTRACT.md",
        V7 / "contracts" / "RUNTIME_CONTRACT.md",
        V7 / "README.md",
    ]
    missing_docs = [p.name for p in required_docs if not _exists(p)]
    if missing_docs:
        rows.append(
            Row(
                layer="L1",
                handoff="Spec -> IR/Kernel",
                contract="v7 contract docs exist",
                status="FAIL",
                notes="missing=" + ",".join(missing_docs),
            )
        )
    else:
        rows.append(
            Row(
                layer="L1",
                handoff="Spec -> IR/Kernel",
                contract="v7 contract docs exist",
                status="PASS",
                notes="contracts-present",
            )
        )

    # L2: required kernel symbols present
    src = _collect_kernel_source()
    required_symbols = [
        "rmsnorm_forward",
        "rmsnorm_backward",
        "swiglu_forward",
        "swiglu_backward",
        "qk_norm_backward",
        "softmax_cross_entropy_loss",
        "attention_forward_causal_head_major_gqa_exact",
        "attention_backward_causal_head_major_gqa",
    ]
    missing_symbols = [s for s in required_symbols if not _has_symbol(src, s)]
    if missing_symbols:
        rows.append(
            Row(
                layer="L2",
                handoff="Kernel sources -> v7 gate",
                contract="required fp32 backward-capable symbols",
                status="FAIL",
                notes="missing=" + ",".join(missing_symbols),
            )
        )
    else:
        rows.append(
            Row(
                layer="L2",
                handoff="Kernel sources -> v7 gate",
                contract="required fp32 backward-capable symbols",
                status="PASS",
                notes="symbols-present=" + str(len(required_symbols)),
            )
        )

    # L3: parity harness exists
    parity_harness = V7 / "scripts" / "run_parity_1token_v7.py"
    if _exists(parity_harness):
        rows.append(
            Row(
                layer="L3",
                handoff="Kernel API -> PyTorch parity",
                contract="v7 parity harness exists",
                status="PASS",
                notes="script=run_parity_1token_v7.py",
            )
        )
    else:
        rows.append(
            Row(
                layer="L3",
                handoff="Kernel API -> PyTorch parity",
                contract="v7 parity harness exists",
                status="FAIL",
                notes="missing=run_parity_1token_v7.py",
            )
        )

    # L4: v7 make target wiring
    makefile = ROOT / "Makefile"
    if not makefile.exists():
        rows.append(
            Row(
                layer="L4",
                handoff="Dev UX -> CI",
                contract="v7 make targets wired",
                status="FAIL",
                notes="missing=Makefile",
            )
        )
    else:
        text = _read_text(makefile)
        required_targets = ["v7-help:", "v7-validate-contracts:", "v7-parity-1tok:", "v7-gate:", "v7:"]
        missing_targets = [t[:-1] for t in required_targets if t not in text]
        if missing_targets:
            rows.append(
                Row(
                    layer="L4",
                    handoff="Dev UX -> CI",
                    contract="v7 make targets wired",
                    status="WARN",
                    notes="missing=" + ",".join(missing_targets),
                )
            )
        else:
            rows.append(
                Row(
                    layer="L4",
                    handoff="Dev UX -> CI",
                    contract="v7 make targets wired",
                    status="PASS",
                    notes="targets-present",
                )
            )

    # L5: training kernel-map coverage (IR2/backprop lowering prerequisites)
    registry_path = V7 / "kernel_maps" / "KERNEL_REGISTRY.json"
    bindings_path = V7 / "kernel_maps" / "kernel_bindings.json"
    if not _exists(registry_path) or not _exists(bindings_path):
        missing = []
        if not _exists(registry_path):
            missing.append("KERNEL_REGISTRY.json")
        if not _exists(bindings_path):
            missing.append("kernel_bindings.json")
        rows.append(
            Row(
                layer="L5",
                handoff="Kernel maps -> IR2 training lower",
                contract="required training kernel IDs are registered and bound",
                status="FAIL",
                notes="missing=" + ",".join(missing),
            )
        )
    else:
        registry = _load_json(registry_path)
        bindings_doc = _load_json(bindings_path)
        kernels = registry.get("kernels", []) if isinstance(registry, dict) else []
        bindings = bindings_doc.get("bindings", {}) if isinstance(bindings_doc, dict) else {}

        if not isinstance(kernels, list) or not isinstance(bindings, dict):
            rows.append(
                Row(
                    layer="L5",
                    handoff="Kernel maps -> IR2 training lower",
                    contract="required training kernel IDs are registered and bound",
                    status="FAIL",
                    notes="invalid-registry-or-bindings-format",
                )
            )
        else:
            function_to_ids = {}
            for k in kernels:
                if not isinstance(k, dict):
                    continue
                kid = k.get("id")
                fn = (k.get("impl") or {}).get("function")
                if isinstance(kid, str) and isinstance(fn, str):
                    function_to_ids.setdefault(fn, []).append(kid)

            required_training = {
                "rmsnorm_backward": ["rmsnorm_backward"],
                "matmul_backward": ["fc2_backward_kernel", "fc1_backward_kernel"],
                "qk_norm_backward": ["qk_norm_backward"],
                "residual_add_backward": ["ck_residual_add_backward"],
                "rope_backward_qk": ["rope_backward_qk"],
                "embedding_backward": ["embedding_backward"],
                "swiglu_backward": ["swiglu_backward_exact", "swiglu_backward"],
                "softmax_cross_entropy": ["softmax_cross_entropy_loss"],
                "attention_backward_gqa": ["attention_backward_causal_head_major_gqa"],
                "gradient_accumulate": ["gradient_accumulate_f32"],
                "gradient_scale": ["gradient_scale_f32"],
                "gradient_clip_norm": ["gradient_clip_norm_f32"],
                "optimizer_step_adamw": ["adamw_update_f32"],
            }

            missing_registry = []
            missing_bindings = []
            covered = []
            for logical_name, candidates in required_training.items():
                resolved_fn = None
                resolved_ids = []
                for fn in candidates:
                    ids = function_to_ids.get(fn, [])
                    if ids:
                        resolved_fn = fn
                        resolved_ids = ids
                        break
                if resolved_fn is None:
                    missing_registry.append("%s->%s" % (logical_name, "|".join(candidates)))
                    continue
                if not any(kid in bindings for kid in resolved_ids):
                    missing_bindings.append("%s:%s" % (logical_name, resolved_fn))
                    continue
                covered.append(logical_name)

            if missing_registry or missing_bindings:
                notes = []
                if missing_registry:
                    notes.append("missing-registry=" + ",".join(missing_registry))
                if missing_bindings:
                    notes.append("missing-bindings=" + ",".join(missing_bindings))
                rows.append(
                    Row(
                        layer="L5",
                        handoff="Kernel maps -> IR2 training lower",
                        contract="required training kernel IDs are registered and bound",
                        status="FAIL",
                        notes="; ".join(notes),
                    )
                )
            else:
                rows.append(
                    Row(
                        layer="L5",
                        handoff="Kernel maps -> IR2 training lower",
                        contract="required training kernel IDs are registered and bound",
                        status="PASS",
                        notes="covered=%d" % len(covered),
                    )
                )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v7 training contracts.")
    parser.add_argument("--strict", action="store_true", help="Treat WARN as failure.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    rows = run_checks()
    counts = _status_counts(rows)

    print("=" * 112)
    print("v7 CONTRACT REPORT")
    print("=" * 112)
    print(_render_table(rows))
    print("=" * 112)
    print("Summary: PASS=%d WARN=%d FAIL=%d" % (counts.get("PASS", 0), counts.get("WARN", 0), counts.get("FAIL", 0)))

    payload = {
        "rows": [r.__dict__ for r in rows],
        "summary": counts,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON: %s" % args.json_out)

    if counts.get("FAIL", 0) > 0:
        return 1
    if args.strict and counts.get("WARN", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

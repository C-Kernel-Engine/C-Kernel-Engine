#!/usr/bin/env python3
from __future__ import annotations

"""Collect v8 architecture contract health into one JSON report.

This is intentionally lightweight. The expensive checks are still owned by the
normal make targets. This script turns their current source/artifact surface
into a dashboard payload for docs/site/test-report.html.
"""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "version/v8/.cache/reports/architecture_contracts_latest.json"
AUDIT_SCRIPT = ROOT / "version/v8/scripts/audit_template_circuit_v8.py"


PROMOTED_TEMPLATES = [
    "qwen2",
    "qwen3",
    "qwen35",
    "qwen3vl",
    "qwen3_vl_vision",
    "gemma3",
    "gemma4",
    "gemma4_vision",
    "glm4",
    "nemotron_h",
    "kimi_vl",
    "llama",
]


CONTRACT_LANES = [
    {
        "id": "template_circuit",
        "label": "Template / Circuit",
        "description": "Critical projection edges, residual flow, and model-specific block order are explicit enough to audit.",
    },
    {
        "id": "lowered_ir_dataflow",
        "label": "Lowered IR Dataflow",
        "description": "Producer/consumer edges and semantic stream slots survive IR lowering.",
    },
    {
        "id": "generated_c_preservation",
        "label": "Generated C Preservation",
        "description": "Generated C call arguments preserve the lowered activation buffers for critical ops.",
    },
    {
        "id": "runtime_path_equivalence",
        "label": "Runtime Path Equivalence",
        "description": "Equivalent decode/prefill/state-update paths agree for recurrent, attention, and quantized views.",
    },
    {
        "id": "model_contract_coverage",
        "label": "Model Contract Coverage",
        "description": "Promoted model families have template, IR, generated-C, smoke, or parity coverage recorded.",
    },
]


MODEL_COVERAGE = [
    {"family": "qwen2", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "pass", "model": "pass", "notes": "fast v8 regression family"},
    {"family": "qwen3", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "pass", "model": "pass", "notes": "QK-norm + GGUF smoke"},
    {"family": "qwen3.5", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "partial", "model": "partial", "notes": "long-generation numerical parity monitored"},
    {"family": "qwen3-vl", "template": "pass", "ir": "pass", "generated_c": "partial", "runtime": "partial", "model": "pass", "notes": "promoted vision smoke; bridge parity still active"},
    {"family": "gemma3", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "pass", "model": "pass", "notes": "split-half RoPE and sliding attention covered"},
    {"family": "gemma4", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "partial", "model": "pass", "notes": "text coherent; vision bridge early"},
    {"family": "glm4", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "partial", "model": "pass", "notes": "partial pairwise RoPE + GGUF smoke"},
    {"family": "nemotron-h", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "partial", "model": "pass", "notes": "Mamba2 path equivalence monitored"},
    {"family": "kimi-vl", "template": "pass", "ir": "partial", "generated_c": "partial", "runtime": "partial", "model": "partial", "notes": "MLA/MoE scalar contracts added; full template lowering/tokenizer/vision bridge in bring-up"},
    {"family": "llama/nanbeige", "template": "pass", "ir": "pass", "generated_c": "pass", "runtime": "partial", "model": "partial", "notes": "Nanbeige active bring-up lane"},
]


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_template_circuit_v8", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _template_path(name: str) -> Path:
    return ROOT / "version/v8/circuits" / f"{name}.json"


def _safe_template_report(audit: Any, name: str) -> dict[str, Any]:
    path = _template_path(name)
    if not path.exists():
        return {
            "template": name,
            "status": "warn",
            "explicit_count": 0,
            "missing_count": 1,
            "warnings": [f"missing template file: {path}"],
        }
    doc = json.loads(path.read_text(encoding="utf-8"))
    report = audit.audit_template_explicit_edges(doc)
    missing = list(report.get("missing") or [])
    return {
        "template": name,
        "status": "pass" if not missing else "warn",
        "explicit_count": int(report.get("explicit_count") or 0),
        "missing_count": int(report.get("missing_count") or 0),
        "warnings": missing[:8],
    }


NOVELTY_ARTIFACT = ROOT / "version/v8/.cache/reports/model_novelty_latest.json"


def _model_novelty_section() -> dict[str, Any] | None:
    """Include the advisory model novelty report when a cached artifact exists.

    The artifact is produced out of band, for example:
        python3 version/v8/scripts/report_model_novelty_v8.py \
            --base <sha> --head <sha> \
            --json-out version/v8/.cache/reports/model_novelty_latest.json
    The report is advisory only: it never gates and never flips this
    dashboard to warn/fail. When the artifact is absent the section is
    omitted entirely rather than fabricated.
    """
    if not NOVELTY_ARTIFACT.exists():
        return None
    try:
        payload = json.loads(NOVELTY_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    core = payload.get("core_compiler") or {}
    core_files = int(core.get("files") or 0)
    details = (
        f"ADVISORY git-range novelty snapshot ({payload.get('base')}..{payload.get('head')}): "
        f"{core_files} core-compiler file(s) changed (target trend: zero); "
        f"{(payload.get('totals') or {}).get('files', 0)} file(s) total."
    )
    return {
        "id": "model_novelty",
        "label": "Model Novelty (Advisory)",
        "status": "pass",
        "checks_passed": 1,
        "checks_failed": 0,
        "warnings": 0,
        "details": details,
        "rows": [
            {
                "metric": "core_compiler_files_changed",
                "value": core_files,
                "target_trend": core.get("target_trend", "zero"),
                "advisory": True,
            }
        ],
    }


def _status_from_counts(failed: int, warnings: int) -> str:
    if failed:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def build_report() -> dict[str, Any]:
    audit = _load_audit_module()
    template_rows = [_safe_template_report(audit, name) for name in PROMOTED_TEMPLATES]
    template_failed = sum(1 for row in template_rows if row["status"] == "fail")
    template_warnings = sum(1 for row in template_rows if row["status"] == "warn")
    explicit_edges = sum(int(row["explicit_count"]) for row in template_rows)
    missing_edges = sum(int(row["missing_count"]) for row in template_rows)

    capability_evidence_path = ROOT / "version/v8/.cache/reports/mrope_capabilities_latest.json"
    capability_rows = []
    if capability_evidence_path.exists():
        try:
            capability_rows = list(json.loads(capability_evidence_path.read_text(encoding="utf-8")).get("rows") or [])
        except (OSError, json.JSONDecodeError):
            capability_rows = []
    capability_failed = sum(1 for row in capability_rows if row.get("status") != "pass")

    sections = [
        {
            "id": "template_circuit",
            "label": "Template / Circuit",
            "status": _status_from_counts(template_failed, template_warnings),
            "checks_passed": max(0, len(template_rows) - template_failed),
            "checks_failed": template_failed,
            "warnings": template_warnings,
            "details": f"{explicit_edges} explicit critical edges; {missing_edges} implicit/missing edges across promoted templates.",
            "rows": template_rows,
        },
        {
            "id": "lowered_ir_dataflow",
            "label": "Lowered IR Dataflow",
            "status": "pass",
            "checks_passed": 4,
            "checks_failed": 0,
            "warnings": 0,
            "details": "Covered by test-v8-template-circuit-audit, GLM4 synthetic BF16/quant dataflow tests, and safetensors-to-BUMP Nemotron checks.",
        },
        {
            "id": "generated_c_preservation",
            "label": "Generated C Preservation",
            "status": "pass",
            "checks_passed": 1,
            "checks_failed": 0,
            "warnings": 0,
            "details": "Generated-C mamba_in_proj buffer preservation is covered by the template circuit audit test.",
        },
        {
            "id": "runtime_path_equivalence",
            "label": "Runtime Path Equivalence",
            "status": "warn",
            "checks_passed": 5,
            "checks_failed": 0,
            "warnings": 2,
            "details": "Mamba2, DeltaNet, sliding attention, KV-cache, and threadpool parity tests exist; long-context Qwen3.5 and Gemma4 vision remain monitored.",
        },
        {
            "id": "numerical_kernel_capabilities",
            "label": "Numerical Kernel Capabilities",
            "status": "pass" if capability_rows and not capability_failed else "warn",
            "checks_passed": sum(1 for row in capability_rows if row.get("status") == "pass"),
            "checks_failed": capability_failed,
            "warnings": 0 if capability_rows else 1,
            "details": "Artifact-backed input storage, compute, reduction, rounding, output storage, shape, thread, and oracle evidence.",
            "rows": capability_rows,
        },
        {
            "id": "model_contract_coverage",
            "label": "Model Contract Coverage",
            "status": "warn",
            "checks_passed": sum(1 for row in MODEL_COVERAGE if row["model"] == "pass"),
            "checks_failed": 0,
            "warnings": sum(1 for row in MODEL_COVERAGE if "partial" in row.values()),
            "details": "Promoted family coverage from current v8 bring-up lanes.",
            "rows": MODEL_COVERAGE,
        },
    ]

    novelty_section = _model_novelty_section()
    if novelty_section is not None:
        sections.append(novelty_section)

    failed = sum(1 for section in sections if section["status"] == "fail")
    warnings = sum(int(section.get("warnings") or 0) for section in sections)
    return {
        "status": _status_from_counts(failed, warnings),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "sections_total": len(sections),
            "sections_passed": sum(1 for section in sections if section["status"] == "pass"),
            "sections_warn": sum(1 for section in sections if section["status"] == "warn"),
            "sections_failed": failed,
            "templates_total": len(template_rows),
            "templates_failed": template_failed,
            "templates_warn": template_warnings,
            "explicit_template_edges": explicit_edges,
            "missing_template_edges": missing_edges,
            "warnings": warnings,
        },
        "sections": sections,
        "families": MODEL_COVERAGE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())

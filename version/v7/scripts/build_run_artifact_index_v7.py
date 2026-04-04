#!/usr/bin/env python3
"""Build a compact per-run artifact index for cache-root training retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SECTIONS = [
    (
        "brief",
        "Run Brief",
        [
            "run_scope.json",
            "run_scope.md",
            "training_plan.json",
            "training.md",
            "agent.md",
            "seed_source.json",
        ],
    ),
    (
        "gates",
        "Gates",
        [
            "{spec_tag}_curriculum_audit.json",
            "{spec_tag}_curriculum_audit.md",
            "{spec_tag}_preflight.json",
            "{spec_tag}_compiler_smoke_report.json",
            "{spec_tag}_compiler_smoke_report.html",
            "training_parity_regimen_latest.json",
            "training_parity_regimen_latest.md",
            "replay_determinism_latest.json",
            "replay_accum_latest.json",
        ],
    ),
    (
        "evaluation",
        "Evaluation",
        [
            "{spec_tag}_probe_contract.json",
            "{spec_tag}_probe_report.json",
            "{spec_tag}_probe_report.html",
            "{spec_tag}_probe_autopsy.json",
            "{spec_tag}_probe_autopsy.md",
            "{spec_tag}_tested_prompts_report.html",
            "{spec_tag}_tested_prompts_report.md",
            "ir_report.html",
        ],
    ),
    (
        "training",
        "Training",
        [
            "train_{spec_tag}_stage_a.json",
            "train_{spec_tag}_stage_b.json",
            "train_{spec_tag}_sft.json",
            "training_pipeline_latest.json",
            "training_loss_curve_latest.json",
            "training_grad_norms_latest.json",
            "training_checkpoint_policy_latest.json",
            "training_step_profile_latest.json",
        ],
    ),
    (
        "dataset",
        "Dataset",
        [
            "dataset/README.md",
            "dataset/manifests/{prefix}_workspace_manifest.json",
            "dataset/manifests/{prefix}_mixture_manifest.json",
            "dataset/manifests/{prefix}_coherent_replay_manifest.json",
            "dataset/manifests/{prefix}_unified_curriculum_manifest.json",
            "dataset/manifests/{prefix}_sft_instruction_manifest.json",
            "dataset/manifests/{prefix}_neighbor_augmentation_manifest.json",
            "dataset/manifests/{prefix}_delta_manifest.json",
            "dataset/contracts/{prefix}_probe_report_contract.json",
            "dataset/tokenizer/{prefix}_tokenizer_manifest.json",
            "dataset/tokenizer/{prefix}_render_catalog.json",
        ],
    ),
]


def _entry(run_dir: Path, rel: str) -> dict[str, Any]:
    path = run_dir / rel
    return {
        "label": rel,
        "path": rel,
        "exists": path.exists(),
    }


def build_index(run_dir: Path, *, spec_tag: str, prefix: str) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    sections: list[dict[str, Any]] = []
    total = 0
    present = 0
    for key, title, patterns in SECTIONS:
        entries = []
        for pattern in patterns:
            rel = pattern.format(spec_tag=spec_tag, prefix=prefix)
            item = _entry(run_dir, rel)
            entries.append(item)
            total += 1
            present += int(bool(item["exists"]))
        sections.append(
            {
                "key": key,
                "title": title,
                "present": sum(1 for item in entries if item["exists"]),
                "total": len(entries),
                "entries": entries,
            }
        )

    context_dir = run_dir / "context"
    context_files = []
    if context_dir.exists():
        for child in sorted(context_dir.iterdir()):
            if child.is_file():
                context_files.append({"label": child.name, "path": str(child.relative_to(run_dir)), "exists": True})
    sections.append(
        {
            "key": "context",
            "title": "Context Copies",
            "present": len(context_files),
            "total": len(context_files),
            "entries": context_files,
        }
    )

    return {
        "schema": "ck.run_artifact_index.v1",
        "run_dir": str(run_dir),
        "spec_tag": spec_tag,
        "prefix": prefix,
        "summary": {
            "present": present,
            "total": total,
        },
        "sections": sections,
    }


def _write_md(path: Path, doc: dict[str, Any]) -> None:
    lines = [
        "# Run Artifact Index",
        "",
        f"- run_dir: `{doc['run_dir']}`",
        f"- spec_tag: `{doc['spec_tag']}`",
        f"- prefix: `{doc['prefix']}`",
        f"- present: `{doc['summary']['present']}/{doc['summary']['total']}`",
        "",
    ]
    for section in doc.get("sections") or []:
        lines.append(f"## {section['title']}")
        lines.append("")
        for item in section.get("entries") or []:
            status = "present" if item.get("exists") else "missing"
            lines.append(f"- `{status}` `{item['path']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a compact per-run artifact index")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--spec-tag", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--md-out", required=True, type=Path)
    args = ap.parse_args()

    doc = build_index(args.run, spec_tag=str(args.spec_tag), prefix=str(args.prefix))
    args.json_out.expanduser().resolve().write_text(json.dumps(doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_md(args.md_out.expanduser().resolve(), doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

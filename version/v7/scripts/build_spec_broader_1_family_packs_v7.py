#!/usr/bin/env python3
"""Emit per-family bootstrap packs and a readiness queue for spec_broader_1."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN_JSON = ROOT / "version" / "v7" / "reports" / "spec_broader_1_asset_plan_2026-04-04.json"
DEFAULT_OUT_DIR = ROOT / "version" / "v7" / "reports" / "spec_broader_1_family_packs"
DEFAULT_QUEUE_JSON = ROOT / "version" / "v7" / "reports" / "spec_broader_1_bootstrap_queue_2026-04-04.json"
DEFAULT_QUEUE_MD = ROOT / "version" / "v7" / "reports" / "spec_broader_1_bootstrap_queue_2026-04-04.md"


FAMILY_SUPPORT: dict[str, dict[str, Any]] = {
    "comparison_span_chart": {
        "renderer_status": "existing_renderer_family",
        "renderer_layout": "comparison_span_chart",
        "gold_seed_authoring_status": "can_start_now",
        "compiler_scope": "extend_existing_family",
        "notes": "Spec09 already renders this family and previous alignment work exists.",
    },
    "poster_stack": {
        "renderer_status": "existing_renderer_family",
        "renderer_layout": "poster_stack",
        "gold_seed_authoring_status": "can_start_now",
        "compiler_scope": "extend_existing_family",
        "notes": "Spec09 already renders stacked poster assets, though the family still needs richer section/chart primitives.",
    },
    "dashboard_cards": {
        "renderer_status": "existing_renderer_family",
        "renderer_layout": "dashboard_cards",
        "gold_seed_authoring_status": "can_start_now",
        "compiler_scope": "extend_existing_family",
        "notes": "Spec09 already supports dashboard-style boards, so gold authoring can begin immediately.",
    },
    "timeline_flow": {
        "renderer_status": "existing_renderer_family",
        "renderer_layout": "timeline_flow",
        "gold_seed_authoring_status": "can_start_now",
        "compiler_scope": "extend_existing_family",
        "notes": "Spec09 already has a timeline_flow renderer that can be used as the bootstrap path.",
    },
    "table_matrix": {
        "renderer_status": "precursor_renderer_family",
        "renderer_layout": "table_analysis",
        "gold_seed_authoring_status": "can_start_with_precursor",
        "compiler_scope": "promote_precursor_to_first_class_family",
        "notes": "Spec09 has table_analysis as a precursor, but the broader branch needs a cleaner first-class table_matrix family.",
    },
    "architecture_map": {
        "renderer_status": "new_renderer_family_required",
        "renderer_layout": None,
        "gold_seed_authoring_status": "blocked_on_compiler_family",
        "compiler_scope": "new_family_required",
        "notes": "System/topology assets exceed the current bounded system_diagram family and do not yet have a broader renderer.",
    },
}


REFERENCE_ASSETS: dict[str, list[str]] = {
    "comparison_span_chart": [
        "compute-bandwidth-chasm.svg",
        "quantization-formats.svg",
        "rope-layouts-compared.svg",
        "sentencepiece-vs-bpe-wordpiece.svg",
        "tokenizer-performance-comparison.svg",
    ],
    "poster_stack": [],
    "dashboard_cards": [],
    "timeline_flow": [
        "ir-v66-evolution-timeline.svg",
        "ir-timeline-why.svg",
    ],
    "table_matrix": [
        "quantization-formats.svg",
    ],
    "architecture_map": [
        "pipeline-overview.svg",
        "ir-pipeline-flow.svg",
        "kernel-registry-flow.svg",
    ],
}


QUEUE_BUCKETS = (
    "author_now",
    "author_with_precursor",
    "blocked_on_new_family_compiler",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _queue_bucket(support: dict[str, Any]) -> str:
    status = str(support.get("gold_seed_authoring_status") or "")
    if status == "can_start_now":
        return "author_now"
    if status == "can_start_with_precursor":
        return "author_with_precursor"
    return "blocked_on_new_family_compiler"


def _family_pack(plan: dict[str, Any], family_doc: dict[str, Any]) -> dict[str, Any]:
    family = str(family_doc["family"])
    support = FAMILY_SUPPORT[family]
    return {
        "schema": "ck.spec_broader_1_family_pack.v1",
        "generated_on": str(date.today()),
        "branch": plan["branch"],
        "family": family,
        "priority": family_doc["priority"],
        "lineage": family_doc["lineage"],
        "why_now": family_doc["why_now"],
        "readiness": {
            "renderer_status": support["renderer_status"],
            "renderer_layout": support["renderer_layout"],
            "gold_seed_authoring_status": support["gold_seed_authoring_status"],
            "compiler_scope": support["compiler_scope"],
            "notes": support["notes"],
        },
        "replay_reference_assets": [
            {
                "asset": asset,
                "path": f"docs/site/assets/{asset}",
            }
            for asset in REFERENCE_ASSETS.get(family, [])
        ],
        "dsl_additions": list(family_doc["dsl_additions"]),
        "compiler_additions": list(family_doc["compiler_additions"]),
        "selected_assets": list(family_doc["assets"]),
        "immediate_next_steps": _family_next_steps(family, support),
    }


def _family_next_steps(family: str, support: dict[str, Any]) -> list[str]:
    status = str(support["gold_seed_authoring_status"])
    if status == "can_start_now":
        return [
            f"Author 2 to 3 gold {family} DSL seeds for the highest-priority assets in this pack.",
            "Compile them through the existing renderer and write a family smoke report.",
            "Only after smoke passes, widen the family DSL with the listed additions.",
        ]
    if status == "can_start_with_precursor":
        return [
            f"Use the precursor renderer layout `{support['renderer_layout']}` to author exploratory gold seeds.",
            f"Promote `{family}` to a first-class family only after the exploratory seeds reveal the missing compiler behavior.",
            "Write a family-smoke report that clearly separates precursor success from required new-family work.",
        ]
    return [
        f"Design the `{family}` family DSL before authoring training rows.",
        "Implement the new compiler family and prove smoke parity on at least one representative asset.",
        "Only after the new family renders correctly should gold seed authoring and tokenizer freezing begin.",
    ]


def build_outputs(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_docs = list(plan["first_wave"]["families"])
    packs = [_family_pack(plan, family_doc) for family_doc in family_docs]

    queue_items: list[dict[str, Any]] = []
    for pack in packs:
        readiness = pack["readiness"]
        bucket = _queue_bucket(readiness)
        queue_items.append(
            {
                "family": pack["family"],
                "priority": pack["priority"],
                "queue_bucket": bucket,
                "renderer_status": readiness["renderer_status"],
                "renderer_layout": readiness["renderer_layout"],
                "asset_count": len(pack["selected_assets"]),
                "top_assets": [asset["asset"] for asset in pack["selected_assets"][:3]],
            }
        )

    order = {"author_now": 0, "author_with_precursor": 1, "blocked_on_new_family_compiler": 2}
    priority_rank = {"P0": 0, "P1": 1, "P2": 2}
    queue_items.sort(key=lambda row: (order[row["queue_bucket"]], priority_rank.get(row["priority"], 9), row["family"]))
    queue = {
        "schema": "ck.spec_broader_1_bootstrap_queue.v1",
        "generated_on": str(date.today()),
        "branch": plan["branch"],
        "summary": {
            "family_count": len(queue_items),
            "author_now": sum(1 for row in queue_items if row["queue_bucket"] == "author_now"),
            "author_with_precursor": sum(1 for row in queue_items if row["queue_bucket"] == "author_with_precursor"),
            "blocked_on_new_family_compiler": sum(
                1 for row in queue_items if row["queue_bucket"] == "blocked_on_new_family_compiler"
            ),
        },
        "queue": queue_items,
    }
    return packs, queue


def _render_family_md(pack: dict[str, Any]) -> str:
    readiness = pack["readiness"]
    lines: list[str] = []
    lines.append(f"# Spec Broader 1 Family Pack: {pack['family']}")
    lines.append("")
    lines.append(f"- Priority: `{pack['priority']}`")
    lines.append(f"- Lineage: `{pack['lineage']}`")
    lines.append(f"- Renderer status: `{readiness['renderer_status']}`")
    lines.append(f"- Renderer layout: `{readiness['renderer_layout'] or 'new family required'}`")
    lines.append(f"- Gold seed authoring: `{readiness['gold_seed_authoring_status']}`")
    lines.append(f"- Why now: {pack['why_now']}")
    lines.append("")
    lines.append("## Replay References")
    lines.append("")
    if pack["replay_reference_assets"]:
        for asset in pack["replay_reference_assets"]:
            lines.append(f"- `{asset['asset']}`")
    else:
        lines.append("- none yet; this family is new to the current covered library")
    lines.append("")
    lines.append("## DSL Additions")
    lines.append("")
    for token in pack["dsl_additions"]:
        lines.append(f"- `{token}`")
    lines.append("")
    lines.append("## Compiler Additions")
    lines.append("")
    for token in pack["compiler_additions"]:
        lines.append(f"- `{token}`")
    lines.append("")
    lines.append("## Selected Assets")
    lines.append("")
    for asset in pack["selected_assets"]:
        lines.append(
            f"- `{asset['asset']}` -> `{asset['proposed_form']}` ({asset['priority']}; needs `{', '.join(asset['needs'])}`)"
        )
    lines.append("")
    lines.append("## Immediate Next Steps")
    lines.append("")
    for step in pack["immediate_next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def _render_queue_md(queue: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Spec Broader 1 Bootstrap Queue")
    lines.append("")
    lines.append(f"- Families: `{queue['summary']['family_count']}`")
    lines.append(f"- Author now: `{queue['summary']['author_now']}`")
    lines.append(f"- Author with precursor: `{queue['summary']['author_with_precursor']}`")
    lines.append(f"- Blocked on new compiler family: `{queue['summary']['blocked_on_new_family_compiler']}`")
    lines.append("")
    lines.append("## Queue")
    lines.append("")
    for row in queue["queue"]:
        lines.append(
            f"- `{row['family']}` ({row['priority']}) -> `{row['queue_bucket']}` via `{row['renderer_layout'] or 'new family'}`; assets: `{', '.join(row['top_assets'])}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit spec_broader_1 family bootstrap packs and readiness queue.")
    ap.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON, help="Spec broader 1 plan JSON.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for per-family packs.")
    ap.add_argument("--queue-json", type=Path, default=DEFAULT_QUEUE_JSON, help="Readiness queue JSON output path.")
    ap.add_argument("--queue-md", type=Path, default=DEFAULT_QUEUE_MD, help="Readiness queue Markdown output path.")
    args = ap.parse_args()

    plan = _load_json(args.plan_json.expanduser().resolve())
    packs, queue = build_outputs(plan)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        slug = _slug(pack["family"])
        (out_dir / f"{slug}.json").write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{slug}.md").write_text(_render_family_md(pack) + "\n", encoding="utf-8")

    queue_json = args.queue_json.expanduser().resolve()
    queue_md = args.queue_md.expanduser().resolve()
    queue_json.parent.mkdir(parents=True, exist_ok=True)
    queue_md.parent.mkdir(parents=True, exist_ok=True)
    queue_json.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    queue_md.write_text(_render_queue_md(queue) + "\n", encoding="utf-8")
    print(json.dumps(queue, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

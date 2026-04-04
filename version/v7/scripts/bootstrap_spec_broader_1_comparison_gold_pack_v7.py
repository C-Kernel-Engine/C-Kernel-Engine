#!/usr/bin/env python3
"""Write the first spec_broader_1 comparison-span-chart gold pack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
OUT_DIR = REPORTS / "spec_broader_1_gold_mappings" / "comparison_span_chart"
STATUS_JSON = OUT_DIR / "spec_broader_1_comparison_gold_pack_status_20260404.json"
README = OUT_DIR / "README.md"


@dataclass(frozen=True)
class GoldCase:
    asset: str
    case_id: str
    rationale: str
    scene_text: str
    content_json: dict[str, Any]


def _scene_text(*, frame: str, background: str, tone: str = "mixed") -> str:
    return "\n".join(
        [
            "[scene]",
            "[canvas:wide]",
            "[layout:comparison_span_chart]",
            "[theme:infra_dark]",
            f"[tone:{tone}]",
            f"[frame:{frame}]",
            "[density:balanced]",
            "[inset:md]",
            "[gap:md]",
            "[hero:center]",
            "[columns:3]",
            "[emphasis:center]",
            "[rail:none]",
            f"[background:{background}]",
            "[connector:bracket]",
            "[topic:comparison_span_chart_generic]",
            "[header_band:@title.kicker|@title.headline|@title.subtitle]",
            "[compare_bar:@bars.primary.label|@bars.primary.value|@bars.primary.caption|accent=@bars.primary.accent|note=@bars.primary.note]",
            "[compare_bar:@bars.secondary.label|@bars.secondary.value|@bars.secondary.caption|accent=@bars.secondary.accent|note=@bars.secondary.note]",
            "[axis:@axis.label|@axis.note]",
            "[legend_row:amber=@legend.primary|green=@legend.secondary]",
            "[annotation:@annotation.label|@annotation.note|accent=@annotation.accent]",
            "[divider:dash]",
            "[span_bracket:@bars.primary.bracket_label|@bars.primary.bracket_value]",
            "[span_bracket:@bars.secondary.bracket_label|@bars.secondary.bracket_value]",
            "[floor_band:@floor.note]",
            "[thesis_box:@thesis.title|@thesis.line_1|@thesis.line_2]",
            "[conclusion_strip:@summary.conclusion]",
            "[footer_note:@summary.footer]",
            "[/scene]",
        ]
    )


CASES: tuple[GoldCase, ...] = (
    GoldCase(
        asset="performance-balance.svg",
        case_id="performance-balance",
        rationale="Primary comparison-span seed for the broader branch. It stays within the existing renderer while stressing multi-series span logic, threshold narrative, and thesis-centered composition.",
        scene_text=_scene_text(frame="none", background="none"),
        content_json={
            "title": {
                "kicker": "Systems tradeoff",
                "headline": "The Performance Gap",
                "subtitle": "Same floor, very different reachable spans.",
            },
            "bars": {
                "primary": {
                    "label": "GPU Compute",
                    "value": "67,000 GB/s eq",
                    "caption": "5,000x total span",
                    "accent": "amber",
                    "note": "HBM is fast but capacity-bound.",
                    "bracket_label": "GPU span",
                    "bracket_value": "5360x",
                },
                "secondary": {
                    "label": "CPU Compute",
                    "value": "1,800 GB/s eq",
                    "caption": "144x total span",
                    "accent": "green",
                    "note": "Fit and cost are closer to deployment reality.",
                    "bracket_label": "CPU span",
                    "bracket_value": "144x",
                },
            },
            "axis": {
                "label": "Bandwidth axis",
                "note": "Same ethernet floor under both paths",
            },
            "legend": {
                "primary": "GPU cluster",
                "secondary": "CPU server",
            },
            "annotation": {
                "label": "Bottleneck shift",
                "note": "Memory and topology dominate before peak FLOPs.",
                "accent": "amber",
            },
            "floor": {
                "note": "Ethernet equalizes the floor for every cluster.",
            },
            "thesis": {
                "title": "Optimize for the constraint, not the headline.",
                "line_1": "The GPU path wins on raw peak throughput but amplifies memory and network pressure.",
                "line_2": "The CPU path stays inside a smaller, more closable deployment gap.",
            },
            "summary": {
                "conclusion": "The comparison family should make the span itself legible, not just the endpoints.",
                "footer": "Broader training should keep values external and let the DSL own only the board structure.",
            },
        },
    ),
    GoldCase(
        asset="cpu-gpu-analysis.svg",
        case_id="cpu-gpu-analysis",
        rationale="Cost-bound variant of the comparison family. It teaches that the same family can carry deployment-cost reasoning without collapsing into a dual-panel memo.",
        scene_text=_scene_text(frame="card", background="mesh"),
        content_json={
            "title": {
                "kicker": "Deployment cost boundary",
                "headline": "CPU vs GPU Purchase Path",
                "subtitle": "Memory fit changes topology, and topology changes cost.",
            },
            "bars": {
                "primary": {
                    "label": "GPU Path",
                    "value": "$250k+",
                    "caption": "8 GPUs plus fabric",
                    "accent": "amber",
                    "note": "Multi-GPU fit pressure sets the purchase floor.",
                    "bracket_label": "Cluster cost",
                    "bracket_value": "8x+",
                },
                "secondary": {
                    "label": "CPU Path",
                    "value": "$32k",
                    "caption": "1 RAM-heavy server",
                    "accent": "green",
                    "note": "Single-box fit keeps the topology simple.",
                    "bracket_label": "Server cost",
                    "bracket_value": "1x",
                },
            },
            "axis": {
                "label": "Deployment spend",
                "note": "Topology changes the bill before optimization begins",
            },
            "legend": {
                "primary": "GPU cluster",
                "secondary": "CPU server",
            },
            "annotation": {
                "label": "Fit first",
                "note": "If the model does not fit, peak FLOPs do not rescue the plan.",
                "accent": "amber",
            },
            "floor": {
                "note": "Both paths still share the same external network realities.",
            },
            "thesis": {
                "title": "Memory fit is a purchasing decision, not just an optimization detail.",
                "line_1": "The expensive path starts paying for topology before training or inference begins.",
                "line_2": "The cheaper path keeps the architecture close to a single-box deployment boundary.",
            },
            "summary": {
                "conclusion": "The comparison DSL should expose cost span, topology note, and one explicit thesis region.",
                "footer": "Use the same family to compare spending, fit, and deployment complexity without changing raw geometry tokens.",
            },
        },
    ),
    GoldCase(
        asset="theory-of-constraints.svg",
        case_id="theory-of-constraints",
        rationale="Constraint-oriented comparison seed. It stays in the same family but makes the network floor and bottleneck narrative explicit.",
        scene_text=_scene_text(frame="panel", background="rings"),
        content_json={
            "title": {
                "kicker": "Constraint framing",
                "headline": "Topology Becomes the Equalizer",
                "subtitle": "At scale, both paths run into the same external constraint.",
            },
            "bars": {
                "primary": {
                    "label": "GPU Cluster",
                    "value": "900 GB/s in-node",
                    "caption": "50 GB/s between nodes",
                    "accent": "amber",
                    "note": "NVLink is fast, but it does not scale to infinity.",
                    "bracket_label": "In-node burst",
                    "bracket_value": "900",
                },
                "secondary": {
                    "label": "CPU Cluster",
                    "value": "460 GB/s in-node",
                    "caption": "50 GB/s between nodes",
                    "accent": "green",
                    "note": "The local gap is smaller and the shared floor is identical.",
                    "bracket_label": "In-node burst",
                    "bracket_value": "460",
                },
            },
            "axis": {
                "label": "Bandwidth ladder",
                "note": "Shared interconnect floor for both paths",
            },
            "legend": {
                "primary": "GPU local path",
                "secondary": "CPU local path",
            },
            "annotation": {
                "label": "Equalizer",
                "note": "Ethernet collapses the apparent architectural distance.",
                "accent": "amber",
            },
            "floor": {
                "note": "The network floor is the same constraint both clusters eventually inherit.",
            },
            "thesis": {
                "title": "Choose the gap you can close.",
                "line_1": "The larger local peak matters less when both systems converge on the same external bottleneck.",
                "line_2": "The better deployment path is the one with the smaller structural mismatch.",
            },
            "summary": {
                "conclusion": "A good comparison board makes the shared floor and the remaining local gap obvious at a glance.",
                "footer": "This is the broader-branch seed for constraint-focused comparison assets.",
            },
        },
    ),
)


def write_gold_pack(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_json = out_dir / STATUS_JSON.name
    readme_path = out_dir / README.name
    rows: list[dict[str, Any]] = []
    for case in CASES:
        scene_path = out_dir / f"{case.case_id}.scene.compact.dsl"
        content_path = out_dir / f"{case.case_id}.content.json"
        scene_path.write_text(case.scene_text.strip() + "\n", encoding="utf-8")
        content_path.write_text(json.dumps(case.content_json, indent=2) + "\n", encoding="utf-8")
        try:
            scene_ref = str(scene_path.relative_to(ROOT))
        except ValueError:
            scene_ref = scene_path.name
        try:
            content_ref = str(content_path.relative_to(ROOT))
        except ValueError:
            content_ref = content_path.name
        rows.append(
            {
                "case_id": case.case_id,
                "asset": case.asset,
                "scene_dsl": scene_ref,
                "content_json": content_ref,
                "rationale": case.rationale,
            }
        )

    status = {
        "schema": "ck.spec_broader_1_comparison_gold_pack.v1",
        "generated_on": "2026-04-04",
        "family": "comparison_span_chart",
        "renderer": "render_structured_scene_spec09_svg",
        "status": "gold_seed_bootstrapped",
        "notes": [
            "This pack externalizes visible copy and values into content.json while reusing the existing comparison_span_chart renderer.",
            "The broader branch should treat these as compiler-first seeds, not as training rows yet.",
            "A dedicated smoke report must pass before tokenizer or dataset expansion starts from this pack.",
        ],
        "cases": rows,
    }
    status_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# Spec Broader 1 Comparison Gold Mappings",
                "",
                "This directory is the first concrete gold pack for the broader branch.",
                "",
                "- family: `comparison_span_chart`",
                "- renderer: `render_structured_scene_spec09_svg`",
                "- contract: `scene.compact.dsl + content.json -> deterministic compiler -> SVG`",
                "",
                "These files are compiler-first gold seeds. They are not training rows until the family smoke report passes.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return status


def main() -> int:
    status = write_gold_pack()
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

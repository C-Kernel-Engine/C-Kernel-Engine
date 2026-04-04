#!/usr/bin/env python3
"""Generate a minimal graph-family spec13b dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_svg_structured_spec06_v7 import (
    _build_tokenizer_payload,
    _ordered_domain_tokens,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_spec13b_v7 import render_structured_scene_spec13b_svg


ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "version" / "v7" / "reports" / "spec13b_gold_mappings"


@dataclass(frozen=True)
class GraphCase:
    case_id: str
    layout: str
    topic_token: str
    prompt_topic: str
    theme: str
    tone: str
    density: str
    asset: str
    scene_text: str
    content_json: dict[str, Any]
    connector: str = "arrow"
    split: str = "train"

    @property
    def prompt_text(self) -> str:
        return " ".join(
            [
                "[task:svg]",
                f"[layout:{self.layout}]",
                f"[topic:{self.topic_token}]",
                f"[theme:{self.theme}]",
                f"[tone:{self.tone}]",
                f"[density:{self.density}]",
                f"[connector:{self.connector}]",
                "[OUT]",
            ]
        )


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flow_result_payload(raw: dict[str, Any]) -> dict[str, Any]:
    for key, value in raw.items():
        if key in {"header", "nodes", "edges", "footer"}:
            continue
        if isinstance(value, dict):
            return dict(value)
    return {"title": "Result", "body": ""}


def _normalize_flow_graph_content(raw: dict[str, Any]) -> dict[str, Any]:
    header = dict(raw.get("header") or {})
    footer = dict(raw.get("footer") or {})
    node_map = raw.get("nodes") if isinstance(raw.get("nodes"), dict) else {}
    edge_map = raw.get("edges") if isinstance(raw.get("edges"), dict) else {}

    node_keys = list(node_map.keys())[:4]
    if len(node_keys) < 4:
        raise ValueError(f"flow_graph case needs 4 nodes, found {len(node_keys)}")

    generic_nodes: dict[str, Any] = {}
    for idx, node_key in enumerate(node_keys, start=1):
        generic_nodes[f"stage_{idx}"] = dict(node_map.get(node_key) or {})

    edge_values = list(edge_map.values())[:4]
    while len(edge_values) < 4:
        edge_values.append("")
    generic_edges = {
        "stage_1_stage_2": str(edge_values[0] or ""),
        "stage_2_stage_3": str(edge_values[1] or ""),
        "stage_3_stage_4": str(edge_values[2] or ""),
        "stage_4_result": str(edge_values[3] or ""),
    }

    return {
        "header": header,
        "nodes": generic_nodes,
        "edges": generic_edges,
        "result": _flow_result_payload(raw),
        "footer": footer,
    }


def _generic_flow_scene(*, theme: str, tone: str, density: str, topic_token: str, connector: str = "arrow") -> str:
    return " ".join(
        [
            "[scene]",
            "[layout:flow_graph]",
            f"[theme:{theme}]",
            f"[tone:{tone}]",
            f"[density:{density}]",
            f"[connector:{connector}]",
            f"[topic:{topic_token}]",
            "[header_band:header]",
            "[decision_node:stage_1|nodes.stage_1]",
            "[decision_node:stage_2|nodes.stage_2]",
            "[decision_node:stage_3|nodes.stage_3]",
            "[decision_node:stage_4|nodes.stage_4]",
            "[outcome_panel:result|result]",
            "[decision_edge:stage_1->stage_2|edges.stage_1_stage_2]",
            "[decision_edge:stage_2->stage_3|edges.stage_2_stage_3]",
            "[decision_edge:stage_3->stage_4|edges.stage_3_stage_4]",
            "[decision_edge:stage_4->result|edges.stage_4_result]",
            "[footer_note:footer]",
            "[/scene]",
        ]
    )


def _decision_tree_content(
    *,
    headline: str,
    subtitle: str,
    footer_note: str,
    nodes: list[tuple[str, str, str, tuple[str, ...]]],
    outcomes: list[tuple[str, str, str, tuple[str, ...]]],
    edge_labels: dict[str, str],
    entry_label: str = "ENTRY",
) -> dict[str, Any]:
    return {
        "header": {"headline": headline, "subtitle": subtitle},
        "entry": {"label": entry_label},
        "nodes": {
            node_id: {
                "title": title,
                "body": body,
                **({"detail": list(details)} if details else {}),
            }
            for node_id, title, body, details in nodes
        },
        "outcomes": {
            outcome_id: {
                "title": title,
                "body": body,
                **({"detail": list(details)} if details else {}),
            }
            for outcome_id, title, body, details in outcomes
        },
        "edges": dict(edge_labels),
        "footer": {"note": footer_note},
    }


def _generic_tree_scene(
    *,
    theme: str,
    tone: str,
    density: str,
    connector: str,
    topic_token: str,
    node_ids: list[str],
    outcome_ids: list[str],
    edges: list[tuple[str, str, str]],
) -> str:
    parts = [
        "[scene]",
        "[layout:decision_tree]",
        f"[theme:{theme}]",
        f"[tone:{tone}]",
        f"[density:{density}]",
        f"[connector:{connector}]",
        f"[topic:{topic_token}]",
        "[header_band:header]",
        "[entry_badge:entry]",
    ]
    for node_id in node_ids:
        parts.append(f"[decision_node:{node_id}|nodes.{node_id}]")
    for outcome_id in outcome_ids:
        parts.append(f"[outcome_panel:{outcome_id}|outcomes.{outcome_id}]")
    for src, dst, edge_key in edges:
        parts.append(f"[decision_edge:{src}->{dst}|edges.{edge_key}]")
    parts.append("[footer_note:footer]")
    parts.append("[/scene]")
    return " ".join(parts)


def _decision_tree_cases() -> list[GraphCase]:
    topic_token = "decision_tree_generic"
    cases = [
        GraphCase(
            case_id="decision_tree_balanced_triad",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a three-branch decision tree",
            theme="infra_dark",
            tone="amber",
            density="balanced",
            connector="arrow",
            asset="generic-decision-tree-balanced.svg",
            scene_text=_generic_tree_scene(
                theme="infra_dark",
                tone="amber",
                density="balanced",
                connector="arrow",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4"],
                outcome_ids=["outcome_1", "outcome_2", "outcome_3"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_1", "node_4", "node_1_node_4"),
                    ("node_2", "outcome_1", "node_2_outcome_1"),
                    ("node_3", "outcome_2", "node_3_outcome_2"),
                    ("node_4", "outcome_3", "node_4_outcome_3"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Balanced three-branch layout with direct outcomes.",
                footer_note="Generic decision-tree content stays external to the planner.",
                nodes=[
                    ("node_1", "Start", "Choose the branch to follow.", ("set the objective",)),
                    ("node_2", "Path A", "Take the first route.", ("review fit", "confirm signal")),
                    ("node_3", "Path B", "Take the middle route.", ("compare options",)),
                    ("node_4", "Path C", "Take the last route.", ("defer if unclear",)),
                ],
                outcomes=[
                    ("outcome_1", "Outcome A", "Proceed on the first branch.", ()),
                    ("outcome_2", "Outcome B", "Proceed on the second branch.", ()),
                    ("outcome_3", "Outcome C", "Proceed on the third branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "A",
                    "node_1_node_3": "B",
                    "node_1_node_4": "C",
                    "node_2_outcome_1": "go",
                    "node_3_outcome_2": "go",
                    "node_4_outcome_3": "go",
                },
            ),
        ),
        GraphCase(
            case_id="decision_tree_editorial_depth",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a deeper editorial decision tree",
            theme="paper_editorial",
            tone="blue",
            density="airy",
            connector="dotted",
            asset="generic-decision-tree-editorial.svg",
            scene_text=_generic_tree_scene(
                theme="paper_editorial",
                tone="blue",
                density="airy",
                connector="dotted",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4", "node_5"],
                outcome_ids=["outcome_1", "outcome_2"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_2", "node_4", "node_2_node_4"),
                    ("node_3", "node_5", "node_3_node_5"),
                    ("node_4", "outcome_1", "node_4_outcome_1"),
                    ("node_5", "outcome_2", "node_5_outcome_2"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Deeper editorial tree with dotted connectors and airy spacing.",
                footer_note="Use external payloads to swap content without changing the scene grammar.",
                nodes=[
                    ("node_1", "Open", "Begin with the top decision.", ("frame the question",)),
                    ("node_2", "Assess Fit", "Inspect the first branch.", ("note the tradeoff",)),
                    ("node_3", "Assess Timing", "Inspect the second branch.", ("set the order",)),
                    ("node_4", "Refine", "Go one level deeper on fit.", ("tighten the scope",)),
                    ("node_5", "Sequence", "Go one level deeper on timing.", ("stage the rollout",)),
                ],
                outcomes=[
                    ("outcome_1", "Editorial Call", "Finalize the refined branch.", ()),
                    ("outcome_2", "Timing Call", "Finalize the staged branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "fit",
                    "node_1_node_3": "timing",
                    "node_2_node_4": "refine",
                    "node_3_node_5": "sequence",
                    "node_4_outcome_1": "lock",
                    "node_5_outcome_2": "lock",
                },
            ),
        ),
        GraphCase(
            case_id="decision_tree_signal_compact",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a compact decision tree with two levels",
            theme="signal_glow",
            tone="green",
            density="compact",
            connector="dashed",
            asset="generic-decision-tree-signal.svg",
            scene_text=_generic_tree_scene(
                theme="signal_glow",
                tone="green",
                density="compact",
                connector="dashed",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4", "node_5"],
                outcome_ids=["outcome_1", "outcome_2"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_2", "node_4", "node_2_node_4"),
                    ("node_3", "node_5", "node_3_node_5"),
                    ("node_4", "outcome_1", "node_4_outcome_1"),
                    ("node_5", "outcome_2", "node_5_outcome_2"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Compact signal-glow branch plan with dashed connectors.",
                footer_note="Family-explicit prompts teach structure and style, not domain semantics.",
                nodes=[
                    ("node_1", "Start", "Pick the direction.", ("read the signal",)),
                    ("node_2", "Left Gate", "Inspect the left path.", ("gather evidence",)),
                    ("node_3", "Right Gate", "Inspect the right path.", ("collect contrast",)),
                    ("node_4", "Left Check", "Final check on the left.", ("short path",)),
                    ("node_5", "Right Check", "Final check on the right.", ("short path",)),
                ],
                outcomes=[
                    ("outcome_1", "Left Result", "Proceed from the left branch.", ()),
                    ("outcome_2", "Right Result", "Proceed from the right branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "left",
                    "node_1_node_3": "right",
                    "node_2_node_4": "check",
                    "node_3_node_5": "check",
                    "node_4_outcome_1": "ship",
                    "node_5_outcome_2": "ship",
                },
            ),
        ),
        GraphCase(
            case_id="decision_tree_dual_gate",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a dual-gate decision tree",
            theme="infra_dark",
            tone="purple",
            density="balanced",
            connector="arrow",
            asset="generic-decision-tree-dual-gate.svg",
            scene_text=_generic_tree_scene(
                theme="infra_dark",
                tone="purple",
                density="balanced",
                connector="arrow",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4"],
                outcome_ids=["outcome_1", "outcome_2"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_3", "node_4", "node_3_node_4"),
                    ("node_2", "outcome_1", "node_2_outcome_1"),
                    ("node_4", "outcome_2", "node_4_outcome_2"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Balanced dual-gate tree with a short and a longer branch.",
                footer_note="The planner learns placement and connector style while copy stays external.",
                nodes=[
                    ("node_1", "Open", "Start at the first gate.", ("set the frame",)),
                    ("node_2", "Quick Gate", "Resolve the short path.", ("few checks",)),
                    ("node_3", "Deep Gate", "Resolve the long path.", ("more checks",)),
                    ("node_4", "Final Gate", "Close the long path.", ("commit the route",)),
                ],
                outcomes=[
                    ("outcome_1", "Fast Result", "Finish the short branch.", ()),
                    ("outcome_2", "Deep Result", "Finish the long branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "fast",
                    "node_1_node_3": "deep",
                    "node_3_node_4": "continue",
                    "node_2_outcome_1": "done",
                    "node_4_outcome_2": "done",
                },
            ),
        ),
        GraphCase(
            case_id="decision_tree_balanced_wide_train",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a wider decision tree with one deeper branch",
            theme="paper_editorial",
            tone="amber",
            density="airy",
            connector="dotted",
            asset="generic-decision-tree-wide-train.svg",
            scene_text=_generic_tree_scene(
                theme="paper_editorial",
                tone="amber",
                density="airy",
                connector="dotted",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4", "node_5"],
                outcome_ids=["outcome_1", "outcome_2", "outcome_3"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_1", "node_4", "node_1_node_4"),
                    ("node_2", "node_5", "node_2_node_5"),
                    ("node_3", "outcome_1", "node_3_outcome_1"),
                    ("node_5", "outcome_2", "node_5_outcome_2"),
                    ("node_4", "outcome_3", "node_4_outcome_3"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Wider editorial tree with one deeper branch and dotted connectors.",
                footer_note="Training cases should span multiple valid tree silhouettes before we judge holdout transfer.",
                nodes=[
                    ("node_1", "Start", "Open the wider tree.", ("set the question",)),
                    ("node_2", "Branch A", "Deeper left branch.", ("inspect first",)),
                    ("node_3", "Branch B", "Direct middle branch.", ("scan quickly",)),
                    ("node_4", "Branch C", "Direct right branch.", ("scan quickly",)),
                    ("node_5", "Branch A2", "Second layer on the left.", ("make the call",)),
                ],
                outcomes=[
                    ("outcome_1", "Outcome B", "Resolve the middle branch.", ()),
                    ("outcome_2", "Outcome A", "Resolve the deeper left branch.", ()),
                    ("outcome_3", "Outcome C", "Resolve the right branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "A",
                    "node_1_node_3": "B",
                    "node_1_node_4": "C",
                    "node_2_node_5": "refine",
                    "node_3_outcome_1": "done",
                    "node_5_outcome_2": "done",
                    "node_4_outcome_3": "done",
                },
            ),
        ),
        GraphCase(
            case_id="decision_tree_wide_deep_holdout",
            layout="decision_tree",
            topic_token=topic_token,
            prompt_topic="a wider and deeper decision tree",
            theme="paper_editorial",
            tone="amber",
            density="airy",
            connector="dotted",
            split="holdout",
            asset="generic-decision-tree-holdout.svg",
            scene_text=_generic_tree_scene(
                theme="paper_editorial",
                tone="amber",
                density="airy",
                connector="dotted",
                topic_token=topic_token,
                node_ids=["node_1", "node_2", "node_3", "node_4", "node_5", "node_6", "node_7", "node_8"],
                outcome_ids=["outcome_1", "outcome_2", "outcome_3"],
                edges=[
                    ("node_1", "node_2", "node_1_node_2"),
                    ("node_1", "node_3", "node_1_node_3"),
                    ("node_1", "node_4", "node_1_node_4"),
                    ("node_2", "node_5", "node_2_node_5"),
                    ("node_2", "node_6", "node_2_node_6"),
                    ("node_3", "node_7", "node_3_node_7"),
                    ("node_4", "node_8", "node_4_node_8"),
                    ("node_6", "outcome_1", "node_6_outcome_1"),
                    ("node_7", "outcome_2", "node_7_outcome_2"),
                    ("node_8", "outcome_3", "node_8_outcome_3"),
                ],
            ),
            content_json=_decision_tree_content(
                headline="Decision Tree",
                subtitle="Holdout case: wider top layer plus one deeper branch.",
                footer_note="Holdout measures structural transfer, not memorized asset identity.",
                nodes=[
                    ("node_1", "Start", "Open the tree.", ("set the goal",)),
                    ("node_2", "Branch A", "First major branch.", ("compare paths",)),
                    ("node_3", "Branch B", "Second major branch.", ("review options",)),
                    ("node_4", "Branch C", "Third major branch.", ("defer if needed",)),
                    ("node_5", "Leaf A1", "Short side leaf.", ("side note",)),
                    ("node_6", "Leaf A2", "Deeper A branch.", ("make the call",)),
                    ("node_7", "Leaf B1", "Deep B branch.", ("check sequence",)),
                    ("node_8", "Leaf C1", "Deep C branch.", ("check risk",)),
                ],
                outcomes=[
                    ("outcome_1", "Outcome A", "Finish the deep A branch.", ()),
                    ("outcome_2", "Outcome B", "Finish the B branch.", ()),
                    ("outcome_3", "Outcome C", "Finish the C branch.", ()),
                ],
                edge_labels={
                    "node_1_node_2": "A",
                    "node_1_node_3": "B",
                    "node_1_node_4": "C",
                    "node_2_node_5": "scan",
                    "node_2_node_6": "commit",
                    "node_3_node_7": "focus",
                    "node_4_node_8": "focus",
                    "node_6_outcome_1": "done",
                    "node_7_outcome_2": "done",
                    "node_8_outcome_3": "done",
                },
            ),
        ),
    ]
    return cases


def _flow_graph_cases() -> list[GraphCase]:
    cases = [
        GraphCase(
            case_id="pipeline_overview",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="pipeline overview",
            theme="infra_dark",
            tone="blue",
            density="balanced",
            connector="arrow",
            asset="pipeline-overview.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="blue", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "pipeline-overview-flow.content.json")),
        ),
        GraphCase(
            case_id="templates_to_ir",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="templates to IR",
            theme="infra_dark",
            tone="blue",
            density="balanced",
            connector="arrow",
            asset="ir-templates-to-ir.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="blue", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "templates-to-ir.content.json")),
        ),
        GraphCase(
            case_id="ir_lowering_pipeline",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="IR lowering pipeline",
            theme="infra_dark",
            tone="amber",
            density="balanced",
            connector="arrow",
            asset="ir-lowering-pipeline.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="amber", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "ir-lowering-pipeline.content.json")),
        ),
        GraphCase(
            case_id="kernel_registry_flow",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="kernel registry flow",
            theme="infra_dark",
            tone="green",
            density="balanced",
            connector="arrow",
            asset="kernel-registry-flow.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="green", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "kernel-registry-flow.content.json")),
        ),
        GraphCase(
            case_id="dataflow_stitching",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="IR dataflow stitching",
            theme="infra_dark",
            tone="amber",
            density="balanced",
            connector="arrow",
            asset="ir-dataflow-stitching.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="amber", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "ir-dataflow-stitching.content.json")),
        ),
        GraphCase(
            case_id="qwen_layer_dataflow",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="qwen layer dataflow",
            theme="infra_dark",
            tone="purple",
            density="balanced",
            connector="arrow",
            asset="qwen_layer_dataflow.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="purple", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "qwen-layer-dataflow.content.json")),
        ),
        GraphCase(
            case_id="ir_pipeline_flow",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="IR pipeline flow",
            theme="infra_dark",
            tone="amber",
            density="balanced",
            connector="arrow",
            split="holdout",
            asset="ir-pipeline-flow.svg",
            scene_text=_generic_flow_scene(theme="infra_dark", tone="amber", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "ir-pipeline-flow.content.json")),
        ),
        GraphCase(
            case_id="ir_artifact_lineage",
            layout="flow_graph",
            topic_token="pipeline_flow",
            prompt_topic="IR artifact lineage",
            theme="paper_editorial",
            tone="amber",
            density="balanced",
            connector="arrow",
            asset="ir-v66-artifact-lineage.svg",
            scene_text=_generic_flow_scene(theme="paper_editorial", tone="amber", density="balanced", topic_token="pipeline_flow", connector="arrow"),
            content_json=_normalize_flow_graph_content(_load_json(GOLD / "ir-artifact-lineage.content.json")),
        ),
    ]
    return cases


def _build_cases(family_mode: str) -> list[GraphCase]:
    if family_mode == "decision_tree":
        return _decision_tree_cases()
    if family_mode == "flow_graph":
        return _flow_graph_cases()
    if family_mode == "all":
        return _decision_tree_cases() + _flow_graph_cases()
    raise ValueError(f"unsupported family_mode: {family_mode}")


def _bridge_prompt_variants(case: GraphCase) -> list[str]:
    layout_label = "decision tree" if case.layout == "decision_tree" else "flow graph"
    prompts = [
        f"Create a {layout_label} infographic about {case.prompt_topic} in a {case.theme.replace('_', ' ')} style with {case.tone} tone and {case.density} spacing. Return scene DSL only. {case.prompt_text}",
        f"Plan a {layout_label} infographic for {case.prompt_topic}. Emit [scene] ... [/scene] only. {case.prompt_text}",
        f"Draft the scene DSL for a {layout_label} about {case.prompt_topic}. Keep the output compiler-facing and scene-only. {case.prompt_text}",
        f"Compose a complete {layout_label} infographic on {case.prompt_topic} with the specified style tags. Emit only the structured scene program. {case.prompt_text}",
    ]
    if case.layout == "flow_graph":
        prompts.extend(
            [
                f"Trace the system flow for {case.prompt_topic} as a flow graph infographic. Return only the scene program. {case.prompt_text}",
                f"Map the execution flow for {case.prompt_topic}. Keep only graph-family scene DSL. {case.prompt_text}",
            ]
        )
    if case.layout == "decision_tree":
        prompts.extend(
            [
                f"Draft a {case.density} decision tree with {case.connector} connectors for {case.prompt_topic}. Return only the scene program. {case.prompt_text}",
                f"Render a {case.theme.replace('_', ' ')} decision tree with {case.connector} branches and no prose outside the DSL. {case.prompt_text}",
                f"Match the requested decision-tree tags exactly: theme {case.theme}, tone {case.tone}, density {case.density}, connector {case.connector}. Output only scene DSL. {case.prompt_text}",
                f"Keep the tree structure and style faithful to the tags and end cleanly at [/scene]. {case.prompt_text}",
            ]
        )
    if case.case_id == "qwen_layer_dataflow":
        prompts.extend(
            [
                f"Show the layer path from token state through attention and MLP as a flow graph. Output scene DSL only. {case.prompt_text}",
                f"Render the one-layer Qwen compute path as a compiler-facing flow graph scene. End cleanly at [/scene]. {case.prompt_text}",
            ]
        )
    return prompts


def _hidden_prompt_variants(case: GraphCase) -> list[str]:
    layout_label = "decision tree" if case.layout == "decision_tree" else "flow graph"
    prompts = [
        f"Draft the scene program for a {layout_label} about {case.prompt_topic}. Keep only compiler-facing DSL. {case.prompt_text}",
        f"Compose a complete {layout_label} infographic on {case.prompt_topic} with the specified style tags. Output only the structured scene program. {case.prompt_text}",
    ]
    if case.layout == "decision_tree":
        prompts.append(
            f"Build a {case.connector} decision tree for {case.prompt_topic}. Stop right after [/scene]. {case.prompt_text}"
        )
    if case.case_id == "qwen_layer_dataflow":
        prompts.append(
            f"Map the Qwen layer compute path as a flow graph. Stop exactly after the scene program. {case.prompt_text}"
        )
    return prompts


def _is_holdout(case: GraphCase) -> bool:
    return str(case.split or "train") == "holdout"


def _family_title(family_mode: str) -> str:
    return {
        "decision_tree": "Decision-Tree Family",
        "flow_graph": "Flow-Graph Family",
    }.get(family_mode, "Structured Graph Families")


def _write_probe_report_contract(out_dir: Path, prefix: str, *, family_mode: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": f"Spec13b {_family_title(family_mode)} Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec13b.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a ",
                "Plan a ",
                "Draft the ",
                "Compose a ",
                "Trace the ",
                "Map the ",
                "Show the ",
                "Render the ",
            ],
        },
        "catalog": {
            "format": "json_rows",
            "path": f"{prefix}_render_catalog.json",
            "prompt_key": "prompt",
            "output_key": "output_tokens",
            "rendered_key": "svg_xml",
            "rendered_mime": "image/svg+xml",
            "split_key": "split",
        },
        "splits": [
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 12},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 12},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str, *, family_mode: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    family_note = {
        "decision_tree": "This line is decision-tree only. External payloads provide copy; the model learns structure, style, and connector planning.",
        "flow_graph": "This line is flow-graph only. External payloads provide copy; the model learns graph structure and style planning.",
    }.get(
        family_mode,
        "This line covers structured graph families only. External payloads provide copy; the model learns structure and style planning.",
    )
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": f"Learn generalized {_family_title(family_mode).lower()} scene DSL with a deterministic compiler-backed renderer.",
        "notes": [
            family_note,
            "The output contract stays symbolic and compiler-backed.",
            "Asset identity stays in prompt text, case_id, and content_json; the scene DSL should stay family-generic.",
            "This line is intended to learn generalized graph structure before broader SVG families.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec13b.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 384,
            "temperature": 0.0,
            "stop_on_text": [
                "Create a ",
                "Plan a ",
                "Draft the ",
                "Compose a ",
                "Trace the ",
                "Map the ",
                "Show the ",
                "Render the ",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec13b_scene_dsl", help="Output prefix")
    ap.add_argument("--family-mode", choices=("all", "decision_tree", "flow_graph"), default="all")
    ap.add_argument("--train-repeats", type=int, default=6)
    ap.add_argument("--holdout-repeats", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(args.family_mode)
    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    tokenizer_corpus: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for case in cases:
        split = "holdout" if _is_holdout(case) else "train"
        output_tokens = case.scene_text
        svg_xml = render_structured_scene_spec13b_svg(output_tokens, content=dict(case.content_json))

        rows: list[tuple[str, str, str, bool]] = [(case.prompt_text, "tag_canonical", split, True)]
        rows.extend((prompt, f"bridge_{idx}", split, True) for idx, prompt in enumerate(_bridge_prompt_variants(case), start=1))
        hidden_split = "probe_hidden_train" if split == "train" else "probe_hidden_holdout"
        rows.extend((prompt, f"hidden_{idx}", hidden_split, False) for idx, prompt in enumerate(_hidden_prompt_variants(case), start=1))

        for prompt, prompt_surface, row_split, training_prompt in rows:
            render_rows.append(
                {
                    "prompt": prompt,
                    "canonical_prompt": case.prompt_text,
                    "output_tokens": output_tokens,
                    "content_json": dict(case.content_json),
                    "svg_xml": svg_xml,
                    "split": row_split,
                    "layout": case.layout,
                    "topic": case.topic_token,
                    "case_id": case.case_id,
                    "router_case_id": case.case_id,
                    "prompt_topic": case.prompt_topic,
                    "theme": case.theme,
                    "tone": case.tone,
                    "density": case.density,
                    "source_asset": case.asset,
                    "prompt_surface": prompt_surface,
                    "training_prompt": bool(training_prompt),
                }
            )
            row_text = f"{prompt} {output_tokens}".strip()
            tokenizer_corpus.extend([row_text, prompt])
            if row_split == "train":
                seen_prompts.append(prompt)
                repeats = int(args.train_repeats) if prompt_surface == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    train_rows.append(row_text)
            elif row_split == "holdout":
                holdout_prompts.append(prompt)
                repeats = int(args.holdout_repeats) if prompt_surface == "tag_canonical" else 1
                for _ in range(max(1, repeats)):
                    holdout_rows.append(row_text)
            elif row_split == "probe_hidden_train":
                hidden_seen_prompts.append(prompt)
            elif row_split == "probe_hidden_holdout":
                hidden_holdout_prompts.append(prompt)

    def dedupe(rows: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            text = str(row or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    seen_prompts = dedupe(seen_prompts)
    holdout_prompts = dedupe(holdout_prompts)
    hidden_seen_prompts = dedupe(hidden_seen_prompts)
    hidden_holdout_prompts = dedupe(hidden_holdout_prompts)
    tokenizer_corpus = dedupe(tokenizer_corpus)

    domain_tokens = _ordered_domain_tokens(tokenizer_corpus)
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(tokenizer, tokenizer_meta, out_dir / f"{args.prefix}_tokenizer")

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    vocab_spec = {
        "format": "svg-structured-fixed-vocab.v7",
        "prefix": args.prefix,
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "vocab_size": int(tokenizer_meta["vocab_size"]),
        "num_merges": int(tokenizer_meta["num_merges"]),
    }
    (out_dir / f"{args.prefix}_vocab.json").write_text(json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix, family_mode=args.family_mode)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix, family_mode=args.family_mode)

    manifest = {
        "schema": "ck.generated_dataset.v1",
        "line_name": "spec13b_scene_dsl",
        "prefix": args.prefix,
        "family_mode": args.family_mode,
        "out_dir": str(out_dir),
        "layouts": sorted({case.layout for case in cases}),
        "case_ids": [case.case_id for case in cases],
        "source_assets": {case.case_id: case.asset for case in cases},
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "hidden_seen_prompts": str(out_dir / f"{args.prefix}_hidden_seen_prompts.txt"),
            "hidden_holdout_prompts": str(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path)
        },
        "counts": {
            "cases": len(cases),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_prompts": len(seen_prompts),
            "holdout_prompts": len(holdout_prompts),
            "hidden_seen_prompts": len(hidden_seen_prompts),
            "hidden_holdout_prompts": len(hidden_holdout_prompts)
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

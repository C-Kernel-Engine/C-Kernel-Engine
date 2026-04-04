#!/usr/bin/env python3
"""Generalized decision-graph IR for spec13b with legacy spec12 adapters."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionGraphNode:
    node_id: str
    title: str
    body: str
    details: tuple[str, ...] = ()
    kind: str = "decision"
    source_ref: str = ""
    order: int = 0


@dataclass(frozen=True)
class DecisionGraphEdge:
    source_id: str
    target_id: str
    label: str = ""
    label_ref: str = ""
    implied: bool = False


@dataclass(frozen=True)
class DecisionGraphIR:
    entry_label: str
    footer_note: str
    nodes: tuple[DecisionGraphNode, ...]
    edges: tuple[DecisionGraphEdge, ...]
    layout: str = "decision_tree"
    theme: str = "infra_dark"
    tone: str = "blue"
    density: str = "balanced"
    topic: str = ""
    inset: str = "md"
    gap: str = "md"
    background: str = "none"
    connector: str = "arrow"
    canvas: str = "wide"


LEGACY_OUTCOME_EDGES: tuple[tuple[str, str], ...] = (
    ("l0_l1", "contract_drift"),
    ("l2", "model_row"),
    ("l3_l4", "parity_branch"),
)


def _component_entries(scene: dict[str, Any], name: str) -> list[dict[str, str]]:
    return list(scene.get("components_by_name", {}).get(name, []))


def _payload_obj(content: dict[str, Any] | None, ref: str) -> Any:
    current: Any = content
    for part in str(ref or "").split("."):
        key = part.strip()
        if not key:
            continue
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            current = current[idx] if 0 <= idx < len(current) else None
        else:
            return None
    return current


def _payload_text(content: dict[str, Any] | None, ref: str) -> str:
    value = _payload_obj(content, ref)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    return str(value)


def _node_from_payload(node_id: str, ref: str, payload: Any, *, kind: str, order: int) -> DecisionGraphNode:
    if isinstance(payload, dict):
        details = tuple(str(item) for item in (payload.get("detail") or ()) if str(item).strip())
        return DecisionGraphNode(
            node_id=node_id,
            title=str(payload.get("title") or node_id),
            body=str(payload.get("body") or ""),
            details=details,
            kind=kind,
            source_ref=ref,
            order=order,
        )
    return DecisionGraphNode(
        node_id=node_id,
        title=node_id,
        body="",
        details=(),
        kind=kind,
        source_ref=ref,
        order=order,
    )


def lower_legacy_decision_scene(scene: dict[str, Any]) -> DecisionGraphIR:
    """Lower a spec12/spec13a decision-tree scene into a generalized graph IR."""

    content = scene.get("_content") if isinstance(scene.get("_content"), dict) else {}
    entry_entries = _component_entries(scene, "entry_badge")
    entry_ref = str(entry_entries[0].get("ref") or "entry") if entry_entries else "entry"
    entry_payload = _payload_obj(content, entry_ref)
    entry_label = str(entry_payload.get("label") if isinstance(entry_payload, dict) else "ENTRY")

    footer_entries = _component_entries(scene, "footer_note")
    footer_ref = str(footer_entries[0].get("ref") or "footer") if footer_entries else "footer"
    footer_payload = _payload_obj(content, footer_ref)
    footer_note = str(footer_payload.get("note") if isinstance(footer_payload, dict) else footer_payload or "")

    nodes: list[DecisionGraphNode] = []
    node_ids: set[str] = set()
    order = 0
    for entry in _component_entries(scene, "decision_node"):
        node_id = str(entry.get("node_id") or "").strip()
        ref = str(entry.get("ref") or "").strip()
        if not node_id or node_id in node_ids:
            continue
        nodes.append(_node_from_payload(node_id, ref, _payload_obj(content, ref), kind="decision", order=order))
        node_ids.add(node_id)
        order += 1
    for entry in _component_entries(scene, "outcome_panel"):
        panel_id = str(entry.get("panel_id") or "").strip()
        ref = str(entry.get("ref") or "").strip()
        if not panel_id or panel_id in node_ids:
            continue
        nodes.append(_node_from_payload(panel_id, ref, _payload_obj(content, ref), kind="outcome", order=order))
        node_ids.add(panel_id)
        order += 1

    edges: list[DecisionGraphEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in _component_entries(scene, "decision_edge"):
        src = str(entry.get("from_ref") or "").strip()
        dst = str(entry.get("to_ref") or "").strip()
        label_ref = str(entry.get("label_ref") or "").strip()
        if not src or not dst:
            continue
        pair = (src, dst)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edges.append(
            DecisionGraphEdge(
                source_id=src,
                target_id=dst,
                label=_payload_text(content, label_ref),
                label_ref=label_ref,
                implied=False,
            )
        )

    # Legacy spec12/spec13a decision trees relied on implicit branch->outcome routing.
    for src, dst in LEGACY_OUTCOME_EDGES:
        pair = (src, dst)
        if pair in seen_pairs or src not in node_ids or dst not in node_ids:
            continue
        edges.append(DecisionGraphEdge(source_id=src, target_id=dst, implied=True))
        seen_pairs.add(pair)

    return DecisionGraphIR(
        entry_label=entry_label,
        footer_note=footer_note,
        nodes=tuple(nodes),
        edges=tuple(edges),
        layout=str(scene.get("layout") or "decision_tree"),
        theme=str(scene.get("theme") or "infra_dark"),
        tone=str(scene.get("tone") or "blue"),
        density=str(scene.get("density") or "balanced"),
        topic=str(scene.get("topic") or ""),
        inset=str(scene.get("inset") or "md"),
        gap=str(scene.get("gap") or "md"),
        background=str(scene.get("background") or "none"),
        connector=str(scene.get("connector") or "arrow"),
        canvas=str(scene.get("canvas") or "wide"),
    )


def assign_layers(graph: DecisionGraphIR) -> dict[str, int]:
    nodes = {node.node_id: node for node in graph.nodes}
    indegree = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.source_id in nodes and edge.target_id in nodes:
            outgoing[edge.source_id].append(edge.target_id)
            indegree[edge.target_id] = indegree.get(edge.target_id, 0) + 1

    roots = [
        node.node_id
        for node in sorted(graph.nodes, key=lambda node: node.order)
        if node.kind != "outcome" and indegree.get(node.node_id, 0) == 0
    ]
    if not roots:
        roots = [node.node_id for node in sorted(graph.nodes, key=lambda node: node.order) if indegree.get(node.node_id, 0) == 0]
    if not roots and graph.nodes:
        roots = [graph.nodes[0].node_id]

    layers: dict[str, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in layers and layers[node_id] <= depth:
            continue
        layers[node_id] = depth
        for child in outgoing.get(node_id, ()):
            queue.append((child, depth + 1))

    max_depth = max(layers.values(), default=0)
    for node in sorted(graph.nodes, key=lambda node: node.order):
        if node.node_id in layers:
            continue
        if node.kind == "outcome":
            layers[node.node_id] = max_depth + 1
        else:
            max_depth += 1
            layers[node.node_id] = max_depth
    return layers


def grouped_layers(graph: DecisionGraphIR) -> list[list[DecisionGraphNode]]:
    layers = assign_layers(graph)
    grouped: dict[int, list[DecisionGraphNode]] = defaultdict(list)
    for node in sorted(graph.nodes, key=lambda item: (layers.get(item.node_id, 0), item.order)):
        grouped[layers.get(node.node_id, 0)].append(node)
    return [grouped[idx] for idx in sorted(grouped)]

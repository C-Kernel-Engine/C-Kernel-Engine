from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .nn import Module


_SLUG_RE = re.compile(r'[^a-z0-9]+')


def _slug(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = _SLUG_RE.sub('_', lowered).strip('_')
    return lowered or 'node'


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    op: str
    kind: str
    scope: str
    config: dict[str, Any]
    local_parameters: dict[str, dict[str, Any]]
    local_parameter_count: int
    total_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'op': self.op,
            'kind': self.kind,
            'scope': self.scope,
            'config': self.config,
            'local_parameters': self.local_parameters,
            'local_parameter_count': self.local_parameter_count,
            'total_parameter_count': self.total_parameter_count,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {'source': self.source, 'target': self.target, 'kind': self.kind}


@dataclass(frozen=True)
class AuthoringGraph:
    name: str
    root_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    total_parameters: int
    leaf_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema': 'ck.python_authoring.graph.v1',
            'name': self.name,
            'root_id': self.root_id,
            'total_parameters': self.total_parameters,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'leaf_order': list(self.leaf_order),
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f'# {self.name}',
            '',
            f'- Total modules: {len(self.nodes)}',
            f'- Total edges: {len(self.edges)}',
            f'- Total parameters: {self.total_parameters}',
            '',
            '| Node | Type | Kind | Local Params | Total Params | Config |',
            '| --- | --- | --- | ---: | ---: | --- |',
        ]
        for node in self.nodes:
            config_parts = ', '.join(f'{key}={value}' for key, value in node.config.items()) or '-'
            lines.append(
                f'| `{node.id}` | `{node.op}` | `{node.kind}` | {node.local_parameter_count} | {node.total_parameter_count} | {config_parts} |'
            )
        return "\n".join(lines)


def build_authoring_graph(model: Module, *, name: Optional[str] = None) -> AuthoringGraph:
    graph_name = str(name or model.name or model.__class__.__name__)
    root_id = _slug(graph_name)
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    leaf_order: list[str] = []

    def walk(module: Module, node_id: str, label: str, scope: str, parent_id: Optional[str]) -> None:
        module_dict = module.to_module_dict()
        nodes.append(
            GraphNode(
                id=node_id,
                label=label,
                op=module_dict['type'],
                kind=module_dict['kind'],
                scope=scope,
                config=module_dict['config'],
                local_parameters=module_dict['local_parameters'],
                local_parameter_count=module_dict['local_parameter_count'],
                total_parameter_count=module_dict['total_parameter_count'],
            )
        )
        if parent_id is not None:
            edges.append(GraphEdge(source=parent_id, target=node_id, kind='contains'))

        children = list(module.named_children())
        if not children:
            leaf_order.append(node_id)
            return

        previous_child_id: Optional[str] = None
        for child_name, child in children:
            child_id = f'{node_id}.{_slug(child_name)}'
            if previous_child_id is not None:
                edges.append(GraphEdge(source=previous_child_id, target=child_id, kind='flow'))
            child_scope = f'{scope}.{child_name}' if scope else child_name
            walk(child, child_id, child.name, child_scope, node_id)
            previous_child_id = child_id

    walk(model, root_id, graph_name, '', None)
    return AuthoringGraph(
        name=graph_name,
        root_id=root_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        total_parameters=model.parameter_count(recurse=True),
        leaf_order=tuple(leaf_order),
    )

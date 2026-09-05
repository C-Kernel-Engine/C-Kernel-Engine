#!/usr/bin/env python3
"""Advisory model novelty report for v8 bring-ups.

This tool answers one question: when a new model family is brought up, how
much code changes outside its circuit, tensor map, genuinely new kernels, and
evidence fixtures? If the DSL compiler and memory planner stay unchanged
across bring-ups, the architecture is scaling.

Two modes:

  Mode A (--base SHA [--head SHA]): classify every changed file in a git
  range into ownership buckets and report per-bucket file counts and line
  deltas. The core-compiler bucket is THE metric (target trend: zero).

  Mode B (--circuit NAME): report, from the kernel registry and circuit
  JSON, the operations used, how many are shared with other circuits vs
  unique to this one, the providers bound, provider status counts, and
  declared vs undeclared numerical boundaries when that metadata exists.
  Composite circuits (components + stitch edges, e.g. multimodal encoder /
  decoder compositions) are resolved recursively the same way
  build_ir_v8.py resolves them: ``extends`` parents are deep-merged,
  component circuits are loaded with cycle detection, and stitch edge
  ops/providers are aggregated with per-component attribution. Any
  composition form this resolver cannot understand raises an explicit
  "unsupported composition" error; zeros are never silently reported.

This report is ADVISORY only. It is not a CI gate and never fails the tree.
Missing metadata is reported as an explicit null with a "not tracked yet"
note; nothing is fabricated.

Stdlib only. Runnable from the repo root.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CIRCUITS_DIR = ROOT / "version/v8/circuits"
REGISTRY_PATH = ROOT / "version/v8/kernel_maps/KERNEL_REGISTRY.json"
KERNEL_MAPS_DIR = ROOT / "version/v8/kernel_maps"

SCHEMA = "cke.v8_model_novelty"
SCHEMA_VERSION = 1

ADVISORY_BANNER = (
    "ADVISORY: this report is informational only. It is not a CI gate, "
    "does not enforce dsl_policy.json caps, and never fails the tree."
)

# Bucket ids in classification order. The first matching rule wins.
BUCKET_ORDER = [
    "circuit",
    "model_map",
    "kernel_map",
    "kernel_c_source",
    "core_compiler",
    "converters",
    "tests_evidence",
    "docs",
    "other",
]

BUCKET_LABELS = {
    "circuit": "Circuit JSON (version/v8/circuits)",
    "model_map": "Model/tensor maps (version/v8/model_maps)",
    "kernel_map": "Kernel maps (version/v8/kernel_maps)",
    "kernel_c_source": "Kernel C source (src/kernels, include)",
    "core_compiler": "CORE COMPILER (DSL lowering, codegen, memory planner)",
    "converters": "Weight converters (version/v8/scripts/convert_*.py)",
    "tests_evidence": "Tests and contract/evidence fixtures",
    "docs": "Documentation",
    "other": "Other",
}

# The core-compiler surface: DSL lowering, code generators, capabilities,
# and the memory planner. resolve_*_contracts_v8.py scripts are included by
# glob below.
CORE_COMPILER_FILES = {
    "version/v8/scripts/build_ir_v8.py",
    "version/v8/scripts/codegen_core_v8.py",
    "version/v8/scripts/codegen_prefill_v8.py",
    "version/v8/scripts/codegen_v8.py",
    "version/v8/scripts/codegen_capabilities_v8.py",
    "version/v8/scripts/memory_planner_v8.py",
}
CORE_COMPILER_GLOBS = ("version/v8/scripts/resolve_*_contracts_v8.py",)


def classify_path(path: str) -> str:
    """Classify one repo-relative path into an ownership bucket."""
    if fnmatch.fnmatch(path, "version/v8/circuits/*.json"):
        return "circuit"
    if fnmatch.fnmatch(path, "version/v8/model_maps/*.json"):
        return "model_map"
    if fnmatch.fnmatch(path, "version/v8/kernel_maps/*.json"):
        return "kernel_map"
    if path in CORE_COMPILER_FILES or any(
        fnmatch.fnmatch(path, pattern) for pattern in CORE_COMPILER_GLOBS
    ):
        return "core_compiler"
    if path.startswith("src/kernels/") or path.startswith("include/"):
        return "kernel_c_source"
    if fnmatch.fnmatch(path, "version/v8/scripts/convert_*.py"):
        return "converters"
    if (
        path.startswith("tests/")
        or path.startswith("version/v8/tests/")
        or path.startswith("version/v8/test/")
        or path.startswith("version/v8/contracts/")
        or path.startswith("version/v8/schemas/")
    ):
        return "tests_evidence"
    if path.startswith("docs/") or path.endswith(".md"):
        return "docs"
    return "other"


def _git_numstat(base: str, head: str, repo: Path) -> list[tuple[int, int, str]]:
    """Return (added, deleted, path) rows for base..head. Binary files count 0/0."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--numstat", base, head],
            cwd=repo,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"error: git diff --numstat {base} {head} failed:\n{exc.output}"
        )
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_s, deleted_s = parts[0], parts[1]
        path = parts[-1]
        added = int(added_s) if added_s.isdigit() else 0
        deleted = int(deleted_s) if deleted_s.isdigit() else 0
        rows.append((added, deleted, path))
    return rows


def build_git_range_report(base: str, head: str = "HEAD", repo: Path = ROOT) -> dict[str, Any]:
    rows = _git_numstat(base, head, repo)
    buckets: dict[str, dict[str, Any]] = {
        bucket: {"files": 0, "added": 0, "deleted": 0, "paths": []}
        for bucket in BUCKET_ORDER
    }
    for added, deleted, path in rows:
        bucket = classify_path(path)
        entry = buckets[bucket]
        entry["files"] += 1
        entry["added"] += added
        entry["deleted"] += deleted
        entry["paths"].append(path)

    total_added = sum(row[0] for row in rows)
    total_deleted = sum(row[1] for row in rows)
    core = buckets["core_compiler"]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "advisory": True,
        "advisory_note": ADVISORY_BANNER,
        "mode": "git_range",
        "base": base,
        "head": head,
        "totals": {"files": len(rows), "added": total_added, "deleted": total_deleted},
        "buckets": buckets,
        "core_compiler": {
            "files": core["files"],
            "added": core["added"],
            "deleted": core["deleted"],
            "paths": core["paths"],
            "target_trend": "zero",
            "note": (
                "Core-compiler churn during a model bring-up is THE scaling "
                "metric: the DSL compiler and memory planner should stay "
                "unchanged when a new family lands in its circuit, maps, "
                "kernels, and evidence."
            ),
        },
    }


def _extract_circuit_ops(doc: dict[str, Any]) -> set[str]:
    """Collect operation names declared by a circuit's block structure."""
    ops: set[str] = set()

    def walk_ops(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, str):
                ops.add(item)
            elif isinstance(item, dict):
                op = item.get("op")
                if isinstance(op, str):
                    ops.add(op)

    block_types = doc.get("block_types") or {}
    if isinstance(block_types, dict):
        for block in block_types.values():
            if not isinstance(block, dict):
                continue
            walk_ops(block.get("header"))
            walk_ops(block.get("footer"))
            body = block.get("body")
            if isinstance(body, dict):
                by_kind = body.get("ops_by_kind")
                if isinstance(by_kind, dict):
                    for items in by_kind.values():
                        walk_ops(items)
                else:
                    for value in body.values():
                        walk_ops(value)
            elif isinstance(body, list):
                walk_ops(body)
    return ops


class UnsupportedCompositionError(RuntimeError):
    """Raised when a circuit's composition cannot be resolved faithfully.

    The report must never silently report zeros for a composition form it
    does not understand, so unresolvable references fail loudly here and
    are surfaced as a clear error by build_circuit_report().
    """


def _merge_circuit_docs(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge child over parent, mirroring build_ir_v8._merge_template_defaults."""
    merged = copy.deepcopy(parent)
    for key, value in child.items():
        if value is None:
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_circuit_docs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_circuit_doc(name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load a circuit with extends parents and components resolved.

    Mirrors build_ir_v8._load_builtin_template_doc: ``extends`` chains are
    deep-merged child-over-parent, component circuits are loaded
    recursively, and cyclic references raise. The resolved component docs
    are attached under ``_resolved_components`` for aggregation.
    """
    if name in stack:
        chain = " -> ".join((*stack, name))
        raise UnsupportedCompositionError(f"cyclic circuit reference: {chain}")
    path = CIRCUITS_DIR / f"{name}.json"
    if not path.exists():
        raise UnsupportedCompositionError(
            f"unsupported composition: circuit {name!r} referenced by "
            f"{' -> '.join(stack) or name} has no JSON at {path}"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedCompositionError(
            f"unsupported composition: circuit {name!r} is not readable JSON: {exc}"
        )
    if not isinstance(doc, dict):
        raise UnsupportedCompositionError(
            f"unsupported composition: circuit {name!r} JSON must be an object"
        )

    parent_name = str(doc.get("extends", "") or "").strip().lower()
    if parent_name:
        parent_doc = _resolve_circuit_doc(parent_name, (*stack, name))
        overrides = copy.deepcopy(doc)
        overrides.pop("extends", None)
        doc = _merge_circuit_docs(parent_doc, overrides)

    components = doc.get("components")
    if components is None:
        return doc
    if not isinstance(components, dict) or not components:
        raise UnsupportedCompositionError(
            f"unsupported composition: {name} components must be a non-empty object"
        )
    sequence = doc.get("sequence")
    if not isinstance(sequence, list) or set(map(str, sequence)) != set(map(str, components)):
        raise UnsupportedCompositionError(
            f"unsupported composition: {name} sequence must name every component exactly; "
            f"sequence={sequence!r} components={sorted(map(str, components))!r}"
        )
    resolved: dict[str, Any] = {}
    for component_name in map(str, sequence):
        ref = components.get(component_name)
        if not isinstance(ref, dict):
            raise UnsupportedCompositionError(
                f"unsupported composition: {name} component {component_name!r} must be an object"
            )
        circuit_name = str(ref.get("circuit", "") or "").strip().lower()
        block_name = str(ref.get("block", "") or "").strip()
        if not circuit_name or not block_name:
            raise UnsupportedCompositionError(
                f"unsupported composition: {name} component {component_name!r} "
                "must explicitly name circuit and block"
            )
        component_doc = _resolve_circuit_doc(circuit_name, (*stack, name))
        if component_doc.get("components") is None:
            block_types = component_doc.get("block_types") or {}
            if not isinstance(block_types, dict) or block_name not in block_types:
                raise UnsupportedCompositionError(
                    f"unsupported composition: {name} component {component_name!r} references "
                    f"missing block {circuit_name}.{block_name}"
                )
        resolved[component_name] = {
            "circuit": circuit_name,
            "block": block_name,
            "doc": component_doc,
        }
    doc["_resolved_components"] = resolved
    return doc


def _stitch_edges(doc: dict[str, Any], circuit_name: str) -> list[dict[str, Any]]:
    """Validate and return the stitch edges declared by a resolved circuit."""
    stitch = doc.get("stitch")
    if stitch is None:
        return []
    if not isinstance(stitch, list):
        raise UnsupportedCompositionError(
            f"unsupported composition: {circuit_name} stitch must be a list of edges"
        )
    edges: list[dict[str, Any]] = []
    for index, edge in enumerate(stitch):
        if not isinstance(edge, dict):
            raise UnsupportedCompositionError(
                f"unsupported composition: {circuit_name} stitch[{index}] must be an object"
            )
        edge_id = str(edge.get("id", "") or "").strip()
        op = str(edge.get("op", "") or "").strip()
        if not edge_id or not op:
            raise UnsupportedCompositionError(
                f"unsupported composition: {circuit_name} stitch[{index}] must declare id and op"
            )
        providers = (edge.get("required_contract") or {}).get("providers") or {}
        if not isinstance(providers, dict):
            raise UnsupportedCompositionError(
                f"unsupported composition: {circuit_name} stitch {edge_id!r} providers must be an object"
            )
        provider_map: dict[str, dict[str, str]] = {}
        for binding, reference in providers.items():
            if not isinstance(reference, dict):
                raise UnsupportedCompositionError(
                    f"unsupported composition: {circuit_name} stitch {edge_id!r} provider "
                    f"{binding!r} must be an object"
                )
            kernel_id = str(reference.get("kernel_id", "") or "").strip()
            provider_op = str(reference.get("op", "") or "").strip()
            if not kernel_id or not provider_op:
                raise UnsupportedCompositionError(
                    f"unsupported composition: {circuit_name} stitch {edge_id!r} provider "
                    f"{binding!r} must declare kernel_id and op"
                )
            provider_map[str(binding)] = {"kernel_id": kernel_id, "op": provider_op}
        edges.append(
            {
                "id": edge_id,
                "from": str(edge.get("from", "") or ""),
                "to": str(edge.get("to", "") or ""),
                "op": op,
                "providers": provider_map,
            }
        )
    return edges


def _collect_circuit_ops(doc: dict[str, Any], circuit_name: str) -> set[str]:
    """Aggregate ops from a resolved circuit: own blocks, components, stitch."""
    ops = _extract_circuit_ops(doc)
    for component in (doc.get("_resolved_components") or {}).values():
        ops |= _collect_circuit_ops(component["doc"], component["circuit"])
    for edge in _stitch_edges(doc, circuit_name):
        ops.add(edge["op"])
        for provider in edge["providers"].values():
            ops.add(provider["op"])
    return ops


def _collect_circuit_providers(
    doc: dict[str, Any], circuit_name: str, prefix: str = ""
) -> dict[str, str]:
    """Aggregate binding -> provider id with per-source qualified keys.

    A binding may map to a dispatch table (variant -> provider id) instead
    of a plain string; every variant provider is reported under
    ``binding[variant]`` so bound candidates are never dropped.
    """
    bindings: dict[str, str] = {}
    own = doc.get("kernels") or {}
    if isinstance(own, dict):
        for binding, provider_ref in own.items():
            key = prefix + str(binding)
            if isinstance(provider_ref, str):
                bindings[key] = provider_ref
            elif isinstance(provider_ref, dict):
                for variant, provider_id in provider_ref.items():
                    if not isinstance(provider_id, str):
                        raise UnsupportedCompositionError(
                            f"unsupported composition: {circuit_name} binding {binding!r} "
                            f"variant {variant!r} must name a provider id string"
                        )
                    bindings[f"{key}[{variant}]"] = provider_id
            else:
                raise UnsupportedCompositionError(
                    f"unsupported composition: {circuit_name} binding {binding!r} must be "
                    "a provider id string or a dispatch table of variant -> provider id"
                )
    for component_name, component in (doc.get("_resolved_components") or {}).items():
        for binding, provider_id in _collect_circuit_providers(
            component["doc"], component["circuit"]
        ).items():
            bindings[f"{prefix}{component_name}.{binding}"] = provider_id
    for edge in _stitch_edges(doc, circuit_name):
        for binding, provider in edge["providers"].items():
            bindings[f"{prefix}stitch.{edge['id']}.{binding}"] = provider["kernel_id"]
    return bindings


def _load_registry() -> dict[str, dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data.get("kernels", []) if "id" in entry}


def _provider_status(entry: dict[str, Any] | None) -> tuple[str, list[str]]:
    """Summarize one provider's tracked status from numerical capabilities."""
    if entry is None:
        return "missing_from_registry", []
    caps = entry.get("numerical_capabilities") or []
    statuses = sorted({str(cap.get("status")) for cap in caps if cap.get("status")})
    if not statuses:
        return "not_tracked", []
    if statuses == ["validated"]:
        return "validated", statuses
    return "observed", statuses


def _numerical_boundaries(
    provider_ids: list[str], registry: dict[str, dict[str, Any]]
) -> tuple[dict[str, int] | None, str]:
    """Declared vs undeclared numerical boundaries across bound providers.

    Kernel maps do not currently carry a structured ``boundaries`` field.
    When no provider map tracks it, return None (never fabricate counts).
    """
    tracked = False
    declared = 0
    undeclared = 0
    for provider_id in provider_ids:
        entry = registry.get(provider_id)
        source_file = (entry or {}).get("_source_file")
        if not source_file:
            continue
        map_path = KERNEL_MAPS_DIR / source_file
        if not map_path.exists():
            continue
        try:
            doc = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "boundaries" not in doc:
            continue
        tracked = True
        if doc["boundaries"]:
            declared += 1
        else:
            undeclared += 1
    if not tracked:
        return None, "numerical boundary metadata is not tracked in kernel maps yet"
    return {"declared": declared, "undeclared": undeclared}, ""


def build_circuit_report(circuit: str) -> dict[str, Any]:
    circuit_path = CIRCUITS_DIR / f"{circuit}.json"
    if not circuit_path.exists():
        available = sorted(path.stem for path in CIRCUITS_DIR.glob("*.json"))
        raise SystemExit(
            f"error: unknown circuit {circuit!r}; available: {', '.join(available)}"
        )
    try:
        return _build_circuit_report(circuit, circuit_path)
    except UnsupportedCompositionError as exc:
        raise SystemExit(f"error: {exc}")


def _build_circuit_report(circuit: str, circuit_path: Path) -> dict[str, Any]:
    doc = _resolve_circuit_doc(circuit)
    registry = _load_registry()

    composite = bool(doc.get("components"))
    composition: dict[str, Any] | None = None
    if composite:
        component_reports = []
        for component_name, component in sorted(
            (doc.get("_resolved_components") or {}).items()
        ):
            component_doc = component["doc"]
            component_ops = _collect_circuit_ops(component_doc, component["circuit"])
            component_providers = _collect_circuit_providers(
                component_doc, component["circuit"]
            )
            component_reports.append(
                {
                    "name": component_name,
                    "circuit": component["circuit"],
                    "block": component["block"],
                    "operations": sorted(component_ops),
                    "operation_count": len(component_ops),
                    "providers": dict(sorted(component_providers.items())),
                    "provider_count": len(component_providers),
                }
            )
        stitch_reports = [
            {
                "id": edge["id"],
                "from": edge["from"],
                "to": edge["to"],
                "op": edge["op"],
                "providers": {
                    binding: provider["kernel_id"]
                    for binding, provider in sorted(edge["providers"].items())
                },
                "provider_ops": {
                    binding: provider["op"]
                    for binding, provider in sorted(edge["providers"].items())
                },
            }
            for edge in _stitch_edges(doc, circuit)
        ]
        composition = {
            "components": component_reports,
            "stitch": stitch_reports,
            "component_count": len(component_reports),
            "stitch_edge_count": len(stitch_reports),
        }

    ops_used = _collect_circuit_ops(doc, circuit)
    circuits_compared = 0
    circuits_skipped: list[str] = []
    ops_elsewhere: dict[str, int] = {}
    for other_path in sorted(CIRCUITS_DIR.glob("*.json")):
        if other_path.stem == circuit:
            continue
        try:
            other_doc = _resolve_circuit_doc(other_path.stem)
        except UnsupportedCompositionError:
            # A broken unrelated circuit must not sink this report; record
            # the skip explicitly instead of silently dropping it.
            circuits_skipped.append(other_path.stem)
            continue
        circuits_compared += 1
        for op in _collect_circuit_ops(other_doc, other_path.stem):
            ops_elsewhere[op] = ops_elsewhere.get(op, 0) + 1
    shared = sorted(op for op in ops_used if op in ops_elsewhere)
    unique = sorted(op for op in ops_used if op not in ops_elsewhere)

    bindings = _collect_circuit_providers(doc, circuit)
    provider_detail = []
    status_counts = {"validated": 0, "observed": 0, "not_tracked": 0, "missing_from_registry": 0}
    for binding, provider_id in sorted(bindings.items()):
        entry = registry.get(provider_id)
        status, cap_statuses = _provider_status(entry)
        status_counts[status] += 1
        provider_detail.append(
            {
                "binding": binding,
                "provider": provider_id,
                "op": (entry or {}).get("op"),
                "status": status,
                "capability_statuses": cap_statuses,
            }
        )

    boundaries, boundaries_note = _numerical_boundaries(list(bindings.values()), registry)

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "advisory": True,
        "advisory_note": ADVISORY_BANNER,
        "mode": "circuit",
        "circuit": circuit,
        "circuit_path": str(circuit_path.relative_to(ROOT)),
        "composite": composite,
        "composition": composition,
        "operations": {
            "used": sorted(ops_used),
            "total": len(ops_used),
            "shared": shared,
            "shared_count": len(shared),
            "unique_to_circuit": unique,
            "unique_count": len(unique),
            "circuits_compared": circuits_compared,
            "circuits_skipped": circuits_skipped,
        },
        "providers": {
            "bindings": dict(sorted(bindings.items())),
            "total": len(bindings),
            "status_counts": status_counts,
            "detail": provider_detail,
        },
        "numerical_boundaries": boundaries,
        "numerical_boundaries_note": boundaries_note,
        "provider_provenance": None,
        "provider_provenance_note": (
            "kernel maps do not track added-by-PR or promotion-status "
            "provenance yet"
        ),
    }


def _render_git_range_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Novelty Report (ADVISORY)",
        "",
        ADVISORY_BANNER,
        "",
        f"Mode: git range `{report['base']}..{report['head']}`",
        "",
        "## Bucket summary",
        "",
        "| Bucket | Files | +Added | -Deleted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in BUCKET_ORDER:
        entry = report["buckets"][bucket]
        label = BUCKET_LABELS[bucket]
        if bucket == "core_compiler":
            label = f"**{label}**"
        lines.append(
            f"| {label} | {entry['files']} | {entry['added']} | {entry['deleted']} |"
        )
    totals = report["totals"]
    lines.append(f"| **Total** | **{totals['files']}** | **{totals['added']}** | **{totals['deleted']}** |")
    core = report["core_compiler"]
    lines += [
        "",
        "## Core-compiler surface (target trend: zero)",
        "",
        f"Files changed: **{core['files']}** (+{core['added']} / -{core['deleted']})",
    ]
    if core["paths"]:
        lines.append("")
        for path in core["paths"]:
            lines.append(f"- `{path}`")
    lines += [
        "",
        core["note"],
        "",
    ]
    return "\n".join(lines)


def _render_circuit_markdown(report: dict[str, Any]) -> str:
    ops = report["operations"]
    providers = report["providers"]
    lines = [
        "# Model Novelty Report (ADVISORY)",
        "",
        ADVISORY_BANNER,
        "",
        f"Mode: circuit `{report['circuit']}` (`{report['circuit_path']}`)",
    ]
    if report.get("composite"):
        composition = report["composition"] or {}
        component_count = composition.get("component_count", 0)
        edge_count = composition.get("stitch_edge_count", 0)
        lines.append(
            f"Composite circuit: yes ({component_count} "
            f"component{'s' if component_count != 1 else ''}, "
            f"{edge_count} stitch edge{'s' if edge_count != 1 else ''})"
        )
    else:
        lines.append("Composite circuit: no")
    lines += [
        "",
        "## Operations",
        "",
        f"- Operations used: {ops['total']}",
        f"- Shared with other circuits: {ops['shared_count']}",
        f"- Unique to this circuit: {ops['unique_count']}",
        f"- Circuits compared: {ops['circuits_compared']}",
    ]
    if ops.get("circuits_skipped"):
        lines.append(
            "- Circuits skipped (unresolvable composition): "
            + ", ".join(f"`{name}`" for name in ops["circuits_skipped"])
        )
    if ops["unique_to_circuit"]:
        lines.append("- Unique ops: " + ", ".join(f"`{op}`" for op in ops["unique_to_circuit"]))
    if report.get("composite"):
        composition = report["composition"] or {}
        lines += [
            "",
            "## Composition (per-component attribution)",
            "",
            "| Component | Circuit | Block | Ops | Providers |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for component in composition.get("components", []):
            lines.append(
                f"| `{component['name']}` | `{component['circuit']}` | "
                f"`{component['block']}` | {component['operation_count']} | "
                f"{component['provider_count']} |"
            )
        stitch_edges = composition.get("stitch", [])
        if stitch_edges:
            lines += [
                "",
                "### Stitch edges",
                "",
                "| Edge | From | To | Op | Providers |",
                "| --- | --- | --- | --- | --- |",
            ]
            for edge in stitch_edges:
                provider_list = ", ".join(
                    f"`{binding}` -> `{kernel_id}`"
                    for binding, kernel_id in edge["providers"].items()
                ) or "-"
                lines.append(
                    f"| `{edge['id']}` | `{edge['from']}` | `{edge['to']}` | "
                    f"`{edge['op']}` | {provider_list} |"
                )
    counts = providers["status_counts"]
    lines += [
        "",
        "## Providers",
        "",
        f"- Providers bound: {providers['total']}",
        f"- Status counts: validated={counts['validated']}, "
        f"observed={counts['observed']}, "
        f"not_tracked={counts['not_tracked']}, "
        f"missing_from_registry={counts['missing_from_registry']}",
        "",
        "| Binding | Provider | Op | Status |",
        "| --- | --- | --- | --- |",
    ]
    for detail in providers["detail"]:
        lines.append(
            f"| `{detail['binding']}` | `{detail['provider']}` | "
            f"{detail['op'] or '-'} | {detail['status']} |"
        )
    boundaries = report["numerical_boundaries"]
    lines += ["", "## Numerical boundaries", ""]
    if boundaries is None:
        lines.append(f"- declared vs undeclared: null ({report['numerical_boundaries_note']})")
    else:
        lines.append(
            f"- declared: {boundaries['declared']}, undeclared: {boundaries['undeclared']}"
        )
    lines += [
        f"- provider provenance (added-by PR / promotion status): null "
        f"({report['provider_provenance_note']})",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base git revision for range mode")
    parser.add_argument("--head", default="HEAD", help="head git revision (default: HEAD)")
    parser.add_argument("--circuit", help="circuit name for circuit mode (e.g. kimi_vl)")
    parser.add_argument("--json-out", type=Path, help="write the JSON report to this path")
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="repository root for git range mode (default: script's repo)",
    )
    args = parser.parse_args()

    if bool(args.base) == bool(args.circuit):
        parser.error("exactly one of --base (range mode) or --circuit (circuit mode) is required")

    if args.base:
        report = build_git_range_report(args.base, args.head, args.repo)
        markdown = _render_git_range_markdown(report)
    else:
        report = build_circuit_report(args.circuit)
        markdown = _render_circuit_markdown(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

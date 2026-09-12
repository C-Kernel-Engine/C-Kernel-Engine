#!/usr/bin/env python3
"""Audit allocator ownership in kernel sources and ratchet migration debt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
KERNEL_ROOT = ROOT / "src" / "kernels"
MAP_ROOT = ROOT / "version" / "v8" / "kernel_maps"
BASELINE = ROOT / "version" / "v8" / "contracts" / "kernel_allocation_baseline.json"
OWNERSHIP = ROOT / "version" / "v8" / "contracts" / "kernel_allocation_ownership.json"
CIRCUIT_ROOT = ROOT / "version" / "v8" / "circuits"
ALLOCATORS = ("malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign", "free")
CALL_RE = re.compile(r"\b(" + "|".join(ALLOCATORS) + r")\s*\(")
EXTERNAL_CREATE_RE = re.compile(r"\b(dnnl_[A-Za-z0-9_]*create)\s*\(")
FUNCTION_RE = re.compile(
    r"(?m)^[ \t]*(?:[A-Za-z_]\w*[ \t]+|\*[ \t]*)+"
    r"(?P<name>[A-Za-z_]\w*)[ \t]*\([^;{}]*\)\s*\{"
)
NON_FUNCTION_NAMES = {"if", "for", "while", "switch"}
NON_MAP_FILES = {
    "KERNEL_REGISTRY.json",
    "KERNEL_SOURCES.json",
    "kernel_bindings.json",
    "kernel_bindings.overlay.json",
}


def _mask_comments_and_strings(source: str) -> str:
    """Replace comments and literals with spaces while preserving offsets."""
    chars = list(source)
    index = 0
    state = "code"
    while index < len(chars):
        current = chars[index]
        following = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if current == "/" and following == "/":
                chars[index] = chars[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if current == "/" and following == "*":
                chars[index] = chars[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if current == '"':
                chars[index] = " "
                state = "string"
            elif current == "'":
                chars[index] = " "
                state = "character"
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            if current == "*" and following == "/":
                chars[index] = chars[index + 1] = " "
                state = "code"
                index += 2
                continue
            if current != "\n":
                chars[index] = " "
        else:
            if current == "\\":
                chars[index] = " "
                if index + 1 < len(chars) and chars[index + 1] != "\n":
                    chars[index + 1] = " "
                index += 2
                continue
            terminator = '"' if state == "string" else "'"
            if current == terminator:
                chars[index] = " "
                state = "code"
            elif current != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def _function_ranges(masked: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for match in FUNCTION_RE.finditer(masked):
        name = match.group("name")
        if name in NON_FUNCTION_NAMES:
            continue
        opening = masked.find("{", match.start(), match.end())
        depth = 0
        for index in range(opening, len(masked)):
            if masked[index] == "{":
                depth += 1
            elif masked[index] == "}":
                depth -= 1
                if depth == 0:
                    ranges.append((opening, index + 1, name))
                    break
    return ranges


def scan_source(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    masked = _mask_comments_and_strings(source)
    ranges = _function_ranges(masked)
    calls: list[dict[str, Any]] = []
    for match in CALL_RE.finditer(masked):
        function = next(
            (name for start, end, name in ranges if start <= match.start() < end),
            "<global>",
        )
        calls.append(
            {
                "allocator": match.group(1),
                "function": function,
                "line": source.count("\n", 0, match.start()) + 1,
            }
        )
    return calls


def scan_external_creates(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    masked = _mask_comments_and_strings(source)
    ranges = _function_ranges(masked)
    calls: list[dict[str, Any]] = []
    for match in EXTERNAL_CREATE_RE.finditer(masked):
        function = next(
            (name for start, end, name in ranges if start <= match.start() < end),
            "<global>",
        )
        calls.append({
            "api": match.group(1),
            "function": function,
            "line": source.count("\n", 0, match.start()) + 1,
        })
    return calls


def _classification(relative_path: str, function: str) -> str:
    if relative_path.endswith("attention_oracle_ggml.c"):
        return "oracle"
    if relative_path.endswith("fused_rmsnorm_linear.c") and function == "main":
        return "test"
    if relative_path.endswith("audio_kernels.c") and function == "audio_whisper_log_mel_window_wav_pcm16_f32":
        return "frontend"
    return "production"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _map_functions() -> dict[str, list[dict[str, Any]]]:
    functions: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(MAP_ROOT.glob("*.json")):
        if path.name in NON_MAP_FILES:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        impl = doc.get("impl") or {}
        names = {impl.get("function")}
        # Shared implementations retain the allocation debt of each mapped wrapper.
        names.update(impl.get("allocation_helpers", []))
        names.update(
            capability.get("function")
            for capability in doc.get("numerical_capabilities", [])
            if isinstance(capability, dict)
        )
        names.discard(None)
        for name in names:
            functions.setdefault(str(name), []).append(
                {
                    "id": doc.get("id", path.stem),
                    "path": str(path.relative_to(ROOT)),
                    "scratch_declared": bool(doc.get("scratch")),
                    "scratch_bound": any(
                        str(param.get("source", "")).startswith("scratch:")
                        for param in (doc.get("call_abi") or {}).get("params", [])
                        if isinstance(param, dict)
                    ),
                    "selection_status": (doc.get("selection") or {}).get("status"),
                }
            )
    return functions


def _walk_contract_ids(value: Any) -> set[str]:
    contracts: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "contract_id" and isinstance(child, str):
                contracts.add(child)
            contracts.update(_walk_contract_ids(child))
    elif isinstance(value, list):
        for child in value:
            contracts.update(_walk_contract_ids(child))
    return contracts


def _map_consumers() -> dict[str, list[str]]:
    """Return checked-in circuits that can select each map contract."""
    circuit_contracts: dict[str, set[str]] = {}
    circuit_strings: dict[str, set[str]] = {}

    def strings(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                found.update(strings(child))
        elif isinstance(value, list):
            for child in value:
                found.update(strings(child))
        return found

    for path in sorted(CIRCUIT_ROOT.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        circuit_contracts[path.stem] = _walk_contract_ids(doc)
        circuit_strings[path.stem] = strings(doc)

    consumers: dict[str, list[str]] = {}
    for path in sorted(MAP_ROOT.glob("*.json")):
        if path.name in NON_MAP_FILES:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        map_id = str(doc.get("id", path.stem))
        contracts = _walk_contract_ids(doc)
        consumers[map_id] = sorted(
            circuit
            for circuit in circuit_contracts
            if map_id in circuit_strings[circuit]
            or bool(contracts & circuit_contracts[circuit])
        )
    return consumers


def _allocation_groups(call_sites: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for call in call_sites:
        if call["allocator"] == "free":
            continue
        identity = f"{call['path']}::{call['function']}"
        row = groups.setdefault(
            identity,
            {
                "identity": identity,
                "path": call["path"],
                "function": call["function"],
                "classification": call["classification"],
                "allocation_calls": 0,
                "allocators": set(),
                "lines": [],
            },
        )
        row["allocation_calls"] += 1
        row["allocators"].add(call["allocator"])
        row["lines"].append(call["line"])
    return groups


def _load_ownership(
    groups: dict[str, dict[str, Any]],
    external_groups: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    document = json.loads(OWNERSHIP.read_text(encoding="utf-8"))
    entries = document.get("functions")
    if not isinstance(entries, dict):
        raise RuntimeError("kernel allocation ownership manifest has no functions object")
    failures: list[str] = []
    missing = sorted(set(groups) - set(entries))
    stale = sorted(set(entries) - set(groups))
    if missing:
        failures.append("unreviewed allocating functions: " + ", ".join(missing))
    if stale:
        failures.append("stale allocation ownership entries: " + ", ".join(stale))
    required = {
        "classification", "execution_role", "phase", "allocation_frequency",
        "size_formula", "lifetime", "owner", "selected_v8_path",
        "migration_target", "priority", "notes",
    }
    for identity in sorted(set(groups) & set(entries)):
        entry = entries[identity]
        absent = sorted(required - set(entry)) if isinstance(entry, dict) else sorted(required)
        if absent:
            failures.append(f"{identity} missing ownership fields: {', '.join(absent)}")
            continue
        if entry["classification"] != groups[identity]["classification"]:
            failures.append(f"{identity} classification does not match source audit")
    external_entries = document.get("external_library_risks", [])
    external_identities = {
        f"{entry.get('path')}::{entry.get('function')}"
        for entry in external_entries if isinstance(entry, dict)
    }
    missing_external = sorted(set(external_groups) - external_identities)
    stale_external = sorted(external_identities - set(external_groups))
    if missing_external:
        failures.append("unreviewed external create functions: " + ", ".join(missing_external))
    if stale_external:
        failures.append("stale external allocation risks: " + ", ".join(stale_external))
    for index, entry in enumerate(external_entries):
        if not isinstance(entry, dict):
            failures.append(f"external library risk {index} is not an object")
            continue
        external_required = {
            "path", "function", "library", "classification", "execution_role",
            "phase", "allocation_frequency", "size_formula", "lifetime", "owner",
            "selected_v8_path", "migration_target", "priority", "notes",
        }
        absent = sorted(external_required - set(entry))
        if absent:
            failures.append(
                f"external library risk {index} missing fields: {', '.join(absent)}"
            )
    return document, failures


def _providers_for_function(
    function: str,
    map_functions: dict[str, list[dict[str, Any]]],
    ownership_functions: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    direct = map_functions.get(function, [])
    if direct:
        return direct, "direct"
    providers: list[dict[str, Any]] = []
    for identity, annotation in ownership_functions.items():
        if function not in annotation.get("reachable_allocating_helpers", []):
            continue
        parent_function = identity.split("::", 1)[1]
        providers.extend(map_functions.get(parent_function, []))
    unique = {provider["id"]: provider for provider in providers}
    return [unique[key] for key in sorted(unique)], "transitive" if unique else "unmapped"


def build_report() -> dict[str, Any]:
    map_functions = _map_functions()
    map_consumers = _map_consumers()
    call_sites: list[dict[str, Any]] = []
    external_call_sites: list[dict[str, Any]] = []
    source_paths = sorted(
        path for path in KERNEL_ROOT.rglob("*")
        if path.suffix in {".c", ".h", ".cc", ".cpp", ".cxx"}
    )
    for path in source_paths:
        relative = str(path.relative_to(ROOT))
        for call in scan_source(path):
            call_sites.append(
                {
                    "path": relative,
                    **call,
                    "classification": _classification(relative, call["function"]),
                }
            )
        for call in scan_external_creates(path):
            external_call_sites.append({
                "path": relative,
                **call,
                "classification": _classification(relative, call["function"]),
            })

    identities = Counter(
        f"{call['path']}::{call['function']}::{call['allocator']}"
        for call in call_sites
    )
    external_identities = Counter(
        f"{call['path']}::{call['function']}::{call['api']}"
        for call in external_call_sites
    )
    allocating_functions = {
        call["function"]
        for call in call_sites
        if call["allocator"] != "free" and call["classification"] == "production"
    }
    mapped = [
        {"function": function, **provider}
        for function in sorted(allocating_functions)
        for provider in map_functions.get(function, [])
    ]
    missing_scratch = [
        row for row in mapped
        if not row["scratch_declared"] or not row["scratch_bound"]
    ]
    groups = _allocation_groups(call_sites)
    external_groups: dict[str, dict[str, Any]] = {}
    for call in external_call_sites:
        identity = f"{call['path']}::{call['function']}"
        row = external_groups.setdefault(identity, {"calls": 0, "apis": set(), "lines": []})
        row["calls"] += 1
        row["apis"].add(call["api"])
        row["lines"].append(call["line"])
    ownership, ownership_failures = _load_ownership(groups, external_groups)
    ownership_rows = []
    for identity, group in sorted(groups.items()):
        annotation = ownership["functions"].get(identity, {})
        providers, reachability = _providers_for_function(
            group["function"], map_functions, ownership["functions"]
        )
        row = {
            **{key: value for key, value in group.items() if key != "allocators"},
            "allocators": sorted(group["allocators"]),
            **annotation,
            "mapped_providers": providers,
            "provider_reachability": reachability,
            "circuit_consumers": sorted({
                circuit
                for provider in providers
                for circuit in map_consumers.get(provider["id"], [])
            }),
        }
        ownership_rows.append(row)
    external_rows = []
    for annotation in ownership.get("external_library_risks", []):
        provider_functions = annotation.get("provider_functions", [annotation["function"]])
        provider_rows: list[dict[str, Any]] = []
        reachability_values: set[str] = set()
        for function in provider_functions:
            rows, reachability = _providers_for_function(
                function, map_functions, ownership["functions"]
            )
            provider_rows.extend(rows)
            reachability_values.add(reachability)
        providers = list({row["id"]: row for row in provider_rows}.values())
        identity = f"{annotation['path']}::{annotation['function']}"
        source_evidence = external_groups.get(identity, {})
        external_rows.append({
            **annotation,
            "external_create_calls": source_evidence.get("calls", 0),
            "external_create_apis": sorted(source_evidence.get("apis", [])),
            "lines": source_evidence.get("lines", []),
            "mapped_providers": providers,
            "provider_reachability": (
                "direct" if "direct" in reachability_values
                else "transitive" if "transitive" in reachability_values
                else "unmapped"
            ),
            "circuit_consumers": sorted({
                circuit
                for provider in providers
                for circuit in map_consumers.get(provider["id"], [])
            }),
        })
    by_class = Counter(call["classification"] for call in call_sites)
    allocation_by_class = Counter(
        call["classification"] for call in call_sites if call["allocator"] != "free"
    )
    counts = {
        "allocator_calls": len(call_sites),
        "allocation_calls": sum(call["allocator"] != "free" for call in call_sites),
        "free_calls": sum(call["allocator"] == "free" for call in call_sites),
        "production_allocator_calls": by_class["production"],
        "production_allocation_calls": allocation_by_class["production"],
        "frontend_allocator_calls": by_class["frontend"],
        "oracle_allocator_calls": by_class["oracle"],
        "test_allocator_calls": by_class["test"],
        "mapped_allocating_providers": len(mapped),
        "mapped_allocating_without_scratch_contract": len(missing_scratch),
        "reviewed_allocating_functions": len(ownership_rows),
        "unreviewed_allocating_functions": len(ownership_failures),
        "external_library_allocation_risks": len(external_rows),
        "external_library_create_calls": len(external_call_sites),
    }
    warnings = []
    if counts["production_allocation_calls"]:
        warnings.append({
            "code": "production_allocator_debt",
            "count": counts["production_allocation_calls"],
            "message": "production kernel allocation calls remain",
        })
    if counts["mapped_allocating_without_scratch_contract"]:
        warnings.append({
            "code": "mapped_allocator_without_scratch",
            "count": counts["mapped_allocating_without_scratch_contract"],
            "message": "mapped allocating providers still lack complete scratch ownership",
        })
    if ownership_failures:
        warnings.append({
            "code": "allocation_ownership_incomplete",
            "count": len(ownership_failures),
            "message": "; ".join(ownership_failures),
        })
    return {
        "schema": "cke.v8.kernel_allocation_audit",
        "schema_version": 2,
        "provenance": {
            "source_revision": _source_revision(),
            "ownership_manifest_sha256": _sha256(OWNERSHIP),
            "allocation_baseline_sha256": _sha256(BASELINE),
        },
        "counts": counts,
        "warnings": warnings,
        "call_site_identities": dict(sorted(identities.items())),
        "external_create_call_site_identities": dict(sorted(external_identities.items())),
        "mapped_allocating_providers": mapped,
        "mapped_allocating_without_scratch_contract": missing_scratch,
        "ownership_inventory": ownership_rows,
        "external_library_allocation_risks": external_rows,
        "external_library_create_call_sites": external_call_sites,
        "ownership_manifest": str(OWNERSHIP.relative_to(ROOT)),
        "ownership_validation_failures": ownership_failures,
        "call_sites": call_sites,
    }


def validate_ratchet(report: dict[str, Any], baseline: dict[str, Any]) -> None:
    current = report["call_site_identities"]
    approved = baseline["maximum_call_site_identities"]
    additions = {
        identity: count
        for identity, count in current.items()
        if count > int(approved.get(identity, 0))
    }
    external_current = report.get("external_create_call_site_identities", {})
    external_approved = baseline.get("maximum_external_create_call_site_identities", {})
    external_additions = {
        identity: count
        for identity, count in external_current.items()
        if count > int(external_approved.get(identity, 0))
    }
    counts = report["counts"]
    failures: list[str] = []
    failures.extend(report.get("ownership_validation_failures", []))
    if additions:
        failures.append("new kernel allocator call sites: " + ", ".join(sorted(additions)))
    if external_additions:
        failures.append(
            "new external-library create call sites: "
            + ", ".join(sorted(external_additions))
        )
    if counts["production_allocation_calls"] > baseline["maximum_production_allocation_calls"]:
        failures.append("production allocation-call debt increased")
    if (
        counts["mapped_allocating_providers"]
        > baseline["maximum_mapped_allocating_providers"]
    ):
        failures.append("mapped allocating-provider debt increased")
    if (
        counts["mapped_allocating_without_scratch_contract"]
        > baseline["maximum_mapped_allocating_without_scratch_contract"]
    ):
        failures.append("mapped provider scratch-contract debt increased")
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.check:
        validate_ratchet(report, json.loads(BASELINE.read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        counts = report["counts"]
        prefix = "WARNING" if report["warnings"] else "PASS"
        print(
            f"{prefix}: kernel allocations: "
            f"allocations={counts['allocation_calls']} "
            f"frees={counts['free_calls']} "
            f"production={counts['production_allocation_calls']} "
            f"frontend_total={counts['frontend_allocator_calls']} "
            f"oracle_total={counts['oracle_allocator_calls']} "
            f"test_total={counts['test_allocator_calls']} "
            f"mapped_without_scratch={counts['mapped_allocating_without_scratch_contract']} "
            f"external_creates={counts['external_library_create_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit allocator ownership in kernel sources and ratchet migration debt."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
KERNEL_ROOT = ROOT / "src" / "kernels"
MAP_ROOT = ROOT / "version" / "v8" / "kernel_maps"
BASELINE = ROOT / "version" / "v8" / "contracts" / "kernel_allocation_baseline.json"
ALLOCATORS = ("malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign", "free")
CALL_RE = re.compile(r"\b(" + "|".join(ALLOCATORS) + r")\s*\(")
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


def _classification(relative_path: str, function: str) -> str:
    if relative_path.endswith("attention_oracle_ggml.c"):
        return "oracle"
    if relative_path.endswith("fused_rmsnorm_linear.c") and function == "main":
        return "test"
    if relative_path.endswith("audio_kernels.c") and function == "audio_whisper_log_mel_window_wav_pcm16_f32":
        return "frontend"
    return "production"


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


def build_report() -> dict[str, Any]:
    map_functions = _map_functions()
    call_sites: list[dict[str, Any]] = []
    for path in sorted(KERNEL_ROOT.rglob("*.c")):
        relative = str(path.relative_to(ROOT))
        for call in scan_source(path):
            call_sites.append(
                {
                    "path": relative,
                    **call,
                    "classification": _classification(relative, call["function"]),
                }
            )

    identities = Counter(
        f"{call['path']}::{call['function']}::{call['allocator']}"
        for call in call_sites
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
    return {
        "schema": "cke.v8.kernel_allocation_audit",
        "schema_version": 1,
        "counts": counts,
        "warnings": warnings,
        "call_site_identities": dict(sorted(identities.items())),
        "mapped_allocating_providers": mapped,
        "mapped_allocating_without_scratch_contract": missing_scratch,
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
    counts = report["counts"]
    failures: list[str] = []
    if additions:
        failures.append("new kernel allocator call sites: " + ", ".join(sorted(additions)))
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
            f"mapped_without_scratch={counts['mapped_allocating_without_scratch_contract']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

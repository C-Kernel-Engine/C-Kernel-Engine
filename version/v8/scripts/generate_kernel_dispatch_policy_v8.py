#!/usr/bin/env python3
"""Generate runtime GEMM dispatch tables from v8 kernel-map metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MAPS = ROOT / "version" / "v8" / "kernel_maps"
DEFAULT_OUTPUT = ROOT / "version" / "v8" / "src" / "ck_kernel_dispatch_policy_v8.inc"
POLICY_MAPS = (
    "gemm_nt_q4_k_q8_k.json",
    "gemm_nt_q6_k_q8_k.json",
)
INT_MAX = 2_147_483_647


def _identifier(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not text or text[0].isdigit():
        raise ValueError(f"invalid runtime dispatch policy id: {value!r}")
    return text


def _bound(row: dict[str, Any], key: str, *, minimum: bool) -> int:
    exact = row.get(key)
    if exact is not None:
        return int(exact)
    field = f"min_{key}" if minimum else f"max_{key}"
    return int(row.get(field, 1 if minimum else INT_MAX))


def _flag_expression(flags: list[str]) -> str:
    if not flags:
        return "0"
    values = [f"CK_GEMM_ROUTE_{_identifier(flag).upper()}" for flag in flags]
    return " | ".join(values)


def _load_policies(path: Path) -> tuple[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    map_id = str(data.get("id") or path.stem)
    implementation = data.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError(f"{path}: implementation must be an object")
    runtime = implementation.get("runtime_dispatch")
    if not isinstance(runtime, dict) or int(runtime.get("schema_version", 0)) != 1:
        raise ValueError(f"{path}: implementation.runtime_dispatch schema_version=1 required")
    policies = runtime.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise ValueError(f"{path}: runtime dispatch policies must be a non-empty object")
    return map_id, policies


def generate() -> str:
    lines = [
        "/* Generated from version/v8/kernel_maps. Do not edit manually. */",
        "/* Generator: version/v8/scripts/generate_kernel_dispatch_policy_v8.py */",
        "",
    ]
    for filename in POLICY_MAPS:
        path = MAPS / filename
        map_id, policies = _load_policies(path)
        for policy_id in sorted(policies):
            policy = policies[policy_id]
            routes = policy.get("routes") if isinstance(policy, dict) else None
            if not isinstance(routes, list) or not routes:
                raise ValueError(f"{path}: policy {policy_id!r} requires routes")
            symbol = f"ck_policy_{_identifier(map_id)}_{_identifier(policy_id)}"
            lines.append(f"static const ck_gemm_route_v8 {symbol}[] = {{")
            for index, route in enumerate(routes):
                if not isinstance(route, dict):
                    raise ValueError(f"{path}: policy {policy_id!r} route {index} must be an object")
                min_m = _bound(route, "m", minimum=True)
                max_m = _bound(route, "m", minimum=False)
                min_n = _bound(route, "n", minimum=True)
                max_n = _bound(route, "n", minimum=False)
                min_k = _bound(route, "k", minimum=True)
                max_k = _bound(route, "k", minimum=False)
                if min_m > max_m or min_n > max_n or min_k > max_k:
                    raise ValueError(f"{path}: policy {policy_id!r} route {index} has inverted bounds")
                tile_m = int(route.get("tile_m", 0))
                tile_n = int(route.get("tile_n", 0))
                max_threads = int(route.get("max_threads", 0))
                flags = route.get("flags", [])
                if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
                    raise ValueError(f"{path}: policy {policy_id!r} route {index} flags must be strings")
                lines.append(
                    "    {"
                    f"{min_m}, {max_m}, {min_n}, {max_n}, {min_k}, {max_k}, "
                    f"{tile_m}, {tile_n}, {max_threads}, {_flag_expression(flags)}"
                    "},"
                )
            lines.extend(
                [
                    "};",
                    f"#define {symbol.upper()}_COUNT "
                    f"(sizeof({symbol}) / sizeof({symbol}[0]))",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

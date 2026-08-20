#!/usr/bin/env python3
"""Verify Gemma4 heterogeneous attention widths and planner-owned write extents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arg_int(operation: dict[str, Any], name: str) -> int:
    for argument in operation.get("args", []):
        if argument.get("name") == name:
            return int(str(argument.get("expr")), 0)
    raise ValueError(
        f"layer {operation.get('layer')} {operation.get('op')} is missing ABI argument {name}"
    )


def _verify_memory(layout: dict[str, Any], phase: str) -> dict[str, Any]:
    validation = (layout.get("validation") or {}).get("activation_memory") or {}
    if validation.get("status") != "PASS":
        raise ValueError(f"{phase} activation-memory validation did not pass")
    writes = validation.get("writes") or []
    if not writes:
        raise ValueError(f"{phase} activation-memory validation has no write extents")
    for write in writes:
        required = int(write.get("required_bytes", -1))
        available = int(write.get("available_bytes", -1))
        if required < 0 or available < 0 or required > available:
            raise ValueError(
                f"{phase} invalid write extent: required={required} available={available} "
                f"op={write.get('op')} layer={write.get('layer')}"
            )
    return {
        "status": "PASS",
        "arena_bytes": int(validation.get("arena_bytes", 0) or 0),
        "checked_writes": len(writes),
    }


def verify_runtime(runtime_dir: Path) -> dict[str, Any]:
    manifest = _load(runtime_dir / "weights_manifest.json")
    config = manifest.get("config") or {}
    widths = [int(value) for value in config.get("layer_attention_output_dim") or []]
    value_widths = [int(value) for value in config.get("layer_v_head_dim") or []]
    windows = [int(value) for value in config.get("layer_sliding_window") or []]
    layer_count = int(config.get("num_hidden_layers") or len(widths))
    num_heads = int(config.get("num_heads") or 0)

    if layer_count <= 0 or num_heads <= 0:
        raise ValueError("Gemma4 manifest is missing layer/head dimensions")
    if not (len(widths) == len(value_widths) == len(windows) == layer_count):
        raise ValueError(
            "Gemma4 per-layer attention arrays do not match num_hidden_layers: "
            f"widths={len(widths)} values={len(value_widths)} windows={len(windows)} "
            f"layers={layer_count}"
        )

    expected = [num_heads * value_width for value_width in value_widths]
    if widths != expected:
        first = next(index for index, pair in enumerate(zip(widths, expected)) if pair[0] != pair[1])
        raise ValueError(
            f"Gemma4 layer {first} attention output width {widths[first]} != "
            f"heads*value_width {expected[first]}"
        )
    if len(set(widths)) < 2:
        raise ValueError("Gemma4 runtime lost heterogeneous sliding/full attention widths")

    phases: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        calls = _load(runtime_dir / f"lowered_{phase}_call.json")
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for operation in calls.get("operations", []):
            if operation.get("layer") is None:
                continue
            key = (str(operation.get("op")), int(operation.get("layer")))
            if key[0] not in {"quantize_out_proj_input", "out_proj"}:
                continue
            if key in by_key:
                raise ValueError(f"{phase} has duplicate {key[0]} for layer {key[1]}")
            by_key[key] = operation
        checked = 0
        for layer, width in enumerate(widths):
            for op_name, argument_name in (
                ("quantize_out_proj_input", "k"),
                ("out_proj", "K"),
            ):
                operation = by_key.get((op_name, layer))
                if operation is None:
                    raise ValueError(f"{phase} missing {op_name} for layer {layer}")
                observed = _arg_int(operation, argument_name)
                if observed != width:
                    raise ValueError(
                        f"{phase} layer {layer} {op_name}.{argument_name}={observed} "
                        f"!= planned attention width {width}"
                    )
                checked += 1
        phases[phase] = {
            "status": "PASS",
            "checked_attention_width_arguments": checked,
            "memory": _verify_memory(_load(runtime_dir / f"layout_{phase}.json"), phase),
        }

    return {
        "schema": "cke.v8.gemma4_runtime_contract",
        "schema_version": 1,
        "status": "PASS",
        "runtime_dir": str(runtime_dir),
        "layers": layer_count,
        "num_heads": num_heads,
        "attention_output_widths": sorted(set(widths)),
        "sliding_layers": sum(window > 0 for window in windows),
        "full_attention_layers": sum(window == 0 for window in windows),
        "phases": phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_runtime(args.runtime_dir.expanduser().resolve())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

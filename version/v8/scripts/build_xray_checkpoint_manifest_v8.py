#!/usr/bin/env python3
"""Build a canonical X-ray checkpoint manifest from call IR and a tensor report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = V8_ROOT / "schemas" / "checkpoint_manifest.schema.json"


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_call_checkpoints(call_ir: Dict[str, Any]) -> list[Dict[str, Any]]:
    entries = []
    seen = set()
    for operation in call_ir.get("operations", []):
        for checkpoint in operation.get("semantic_checkpoints", []):
            checkpoint_id = checkpoint["id"]
            if checkpoint_id in seen:
                raise ValueError(f"duplicate checkpoint in call IR: {checkpoint_id}")
            seen.add(checkpoint_id)
            entries.append(checkpoint)
    if not entries:
        raise ValueError("call IR contains no semantic_checkpoints")
    return entries


def _selector(checkpoint: Dict[str, Any]) -> str:
    tensor = str(checkpoint["tensor"])
    layer = int(checkpoint.get("layer", -1))
    return f"{tensor}@{layer}" if layer >= 0 else tensor


def build_manifest(
    *,
    backend: str,
    call_ir: Dict[str, Any],
    tensor_report: Dict[str, Any],
    model: str,
    source: str,
    phase: str,
    storage_dtype_override: str | None = None,
    requested: set[str] | None = None,
    loaded_library: Path | None = None,
) -> Dict[str, Any]:
    checkpoints = []
    torch_tensors = ((tensor_report.get("torch") or {}).get("tensors") or {})
    comparisons = tensor_report.get("comparisons") or {}
    for checkpoint in _flatten_call_checkpoints(call_ir):
        checkpoint_id = checkpoint["id"]
        if requested is not None and checkpoint_id not in requested:
            continue
        selector = _selector(checkpoint)
        if backend == "pytorch":
            source_meta = torch_tensors.get(selector)
            path_value = source_meta.get("path") if isinstance(source_meta, dict) else None
            shape = source_meta.get("shape") if isinstance(source_meta, dict) else None
        else:
            source_meta = comparisons.get(selector)
            path_value = source_meta.get("ck_path") if isinstance(source_meta, dict) else None
            shape = source_meta.get("shape") if isinstance(source_meta, dict) else None
            if (not shape or len(shape) == 1) and selector in torch_tensors:
                shape = torch_tensors[selector].get("shape")
        if not path_value:
            continue
        path = Path(path_value).resolve()
        if not path.is_file():
            raise ValueError(f"tensor report points to missing file: {path}")
        logical_shape = [int(value) for value in (shape or [])]
        if not logical_shape:
            raise ValueError(f"tensor report has no shape for {selector}")
        axes = list(checkpoint["axis_names"])
        if len(axes) != len(logical_shape):
            raise ValueError(
                f"checkpoint {checkpoint_id} axes {axes} do not match shape {logical_shape}"
            )
        physical_shape = [int(value) for value in source_meta.get("physical_shape", logical_shape)]
        valid_shape = [int(value) for value in source_meta.get("valid_shape", logical_shape)]
        physical_axes = list(source_meta.get("physical_axis_names", axes))
        checkpoint_entry = {
            "checkpoint_id": checkpoint_id,
            "producer": checkpoint["producer"],
            "phase": checkpoint["phase"],
            "layer": int(checkpoint["layer"]),
            "tensor_path": str(path),
            "storage_dtype": storage_dtype_override or checkpoint["storage_dtype"],
            "exported_dtype": "fp32",
            "logical_shape": logical_shape,
            "physical_shape": physical_shape,
            "logical_layout": checkpoint["logical_layout"],
            "axis_names": axes,
            "physical_axis_names": physical_axes,
            "resolved_contract_id": checkpoint["resolved_contract_id"],
            "kernel_id": checkpoint["kernel_id"],
            "function": checkpoint["function"],
            "sha256": sha256(path),
        }
        if physical_shape != logical_shape or "valid_shape" in source_meta:
            checkpoint_entry["valid_shape"] = valid_shape
        for key in ("capacity_shape", "physical_strides"):
            if key in source_meta:
                checkpoint_entry[key] = [int(value) for value in source_meta[key]]
        checkpoints.append(checkpoint_entry)
    if not checkpoints:
        raise ValueError(f"no {backend} tensors matched call-IR checkpoints")
    result = {
        "schema": "cke.checkpoint_manifest",
        "schema_version": 1,
        "backend": backend,
        "run": {"model": model, "phase": phase, "source": source},
        "checkpoints": checkpoints,
    }
    if loaded_library is not None:
        library = loaded_library.resolve()
        if not library.is_file():
            raise ValueError(f"loaded library does not exist: {library}")
        result["run"]["loaded_library"] = {"path": str(library), "sha256": sha256(library)}
    schema = load(SCHEMA)
    errors = list(Draft202012Validator(schema).iter_errors(result))
    if errors:
        raise ValueError(f"generated checkpoint manifest is invalid: {errors[0].message}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--call-ir", type=Path, required=True)
    parser.add_argument("--tensor-report", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode", "mixed_prefill", "teacher_forced"), default="prefill")
    parser.add_argument("--storage-dtype", choices=("fp32", "fp16", "bf16", "q8_0", "q8_k"))
    parser.add_argument("--loaded-library", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(
        backend=args.backend,
        call_ir=load(args.call_ir),
        tensor_report=load(args.tensor_report),
        model=args.model,
        source=args.source,
        phase=args.phase,
        storage_dtype_override=args.storage_dtype,
        requested=set(args.checkpoint) if args.checkpoint else None,
        loaded_library=args.loaded_library,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"checkpoints={len(result['checkpoints'])} output={args.output}")


if __name__ == "__main__":
    main()

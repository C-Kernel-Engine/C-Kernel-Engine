#!/usr/bin/env python3
"""
validate_bump_v4.py
===================

Validate v4 bump weights against IR layout + manifest.

Checks:
  - Every weight buffer in layout has a manifest entry
  - runtime_offset matches layout offset
  - size/dtype match
  - file_offset/size are within the weights file (optional)
  - optional cross-check vs input manifest (converter output)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_DIR = _SCRIPT_DIR.parent / "v3"
if _V3_DIR.is_dir():
    sys.path.insert(0, str(_V3_DIR))

import build_ir_v3 as v3


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_manifest_map(path: Path) -> List[Dict]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                raise ValueError(f"Malformed manifest.map line: {line}")
            name, dtype, file_off, size, rt_off = parts[:5]
            entries.append({
                "name": name,
                "dtype": dtype,
                "file_offset": int(file_off, 0),
                "size": int(size, 0),
                "runtime_offset": int(rt_off, 0),
            })
    return entries


def load_manifest(path: Path) -> List[Dict]:
    if path.suffix == ".map":
        return parse_manifest_map(path)
    data = load_json(path)
    return data.get("entries", [])


def gather_layout_weights(layout: Dict) -> Dict[str, Dict]:
    section = layout["sections"][0]
    buffers = []
    buffers.extend(section["header"]["buffers"])
    for layer in section.get("layers", []):
        buffers.extend(layer["buffers"])
    buffers.extend(section["footer"]["buffers"])

    weights = {}
    for buf in buffers:
        if buf.get("role") != "weight":
            continue
        if buf.get("tied_to"):
            continue
        offset = buf.get("offset")
        if isinstance(offset, str):
            offset = int(offset, 0)
        weights[buf["name"]] = {
            "offset": offset,
            "size": int(buf.get("size", 0)),
            "dtype": str(buf.get("dtype", "")).lower(),
            "shape": buf.get("shape", []),
        }
    return weights


def check_weights_file(weights_path: Path, entries: List[Dict], errors: List[str]) -> None:
    if not weights_path.exists():
        errors.append(f"weights file not found: {weights_path}")
        return
    size = weights_path.stat().st_size
    with weights_path.open("rb") as f:
        magic = f.read(8)
    if magic != b"BUMPWGT4":
        errors.append("weights file has invalid magic (expected BUMPWGT4)")
    for e in entries:
        file_off = int(e.get("file_offset", 0))
        size_e = int(e.get("size", 0))
        if file_off + size_e > size:
            errors.append(
                f"{e.get('name')}: file_offset+size out of bounds "
                f"(0x{file_off:X}+0x{size_e:X} > 0x{size:X})"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v4 bump weights against layout/manifest.")
    ap.add_argument("--layout", required=True, help="layout_decode.json or layout_prefill.json")
    ap.add_argument("--manifest", required=True, help="weights_manifest.json or weights_manifest.map")
    ap.add_argument("--weights", help="weights.bump (optional file bounds check)")
    ap.add_argument("--manifest-input", help="converter manifest (weights_manifest_input.json)")
    args = ap.parse_args()

    layout = load_json(Path(args.layout))
    weights = gather_layout_weights(layout)
    entries = load_manifest(Path(args.manifest))
    entries_by_name = {e["name"]: e for e in entries}

    errors = []
    warnings = []

    for name, buf in weights.items():
        entry = entries_by_name.get(name)
        if not entry:
            errors.append(f"missing manifest entry: {name}")
            continue
        rt_off = entry.get("runtime_offset")
        if rt_off is not None:
            if int(rt_off) != int(buf["offset"]):
                errors.append(
                    f"{name}: runtime_offset mismatch (manifest=0x{int(rt_off):X}, layout=0x{buf['offset']:X})"
                )
        size = int(entry.get("size", 0))
        if size != int(buf["size"]):
            errors.append(
                f"{name}: size mismatch (manifest={size}, layout={buf['size']})"
            )
        dtype = str(entry.get("dtype", "")).lower()
        if dtype and dtype != buf["dtype"]:
            errors.append(f"{name}: dtype mismatch (manifest={dtype}, layout={buf['dtype']})")

        # Optional size sanity via compute_size (best-effort).
        if buf["shape"] and buf["dtype"]:
            try:
                expected = v3.compute_size(buf["shape"], buf["dtype"])
                if expected != int(buf["size"]):
                    errors.append(
                        f"{name}: computed size {expected} != layout size {buf['size']} (dtype={buf['dtype']})"
                    )
            except Exception:
                pass

    if args.manifest_input:
        input_entries = load_manifest(Path(args.manifest_input))
        input_by_name = {e["name"]: e for e in input_entries}
        for name, entry in entries_by_name.items():
            if name not in input_by_name:
                warnings.append(f"{name}: missing in input manifest")
                continue
            inp = input_by_name[name]
            if int(inp.get("file_offset", 0)) != int(entry.get("file_offset", 0)):
                errors.append(f"{name}: file_offset mismatch between input/merged manifest")
            if int(inp.get("size", 0)) != int(entry.get("size", 0)):
                errors.append(f"{name}: size mismatch between input/merged manifest")

    if args.weights:
        check_weights_file(Path(args.weights), entries, errors)

    if warnings:
        print("[WARN] Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("[ERROR] Validation failed:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("[OK] Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

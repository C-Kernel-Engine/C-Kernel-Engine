#!/usr/bin/env python3
"""Build a balanced probe contract for spec07 scene DSL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_spec06_probe_contract_v7 import build_contract as _build_spec06_contract
from build_spec06_probe_contract_v7 import _detect_prefix


def build_contract(run_dir: Path, prefix: str, per_split: int) -> dict:
    payload = _build_spec06_contract(run_dir, prefix, per_split)
    payload["title"] = f"{prefix} Scene DSL Probe Report"
    output_adapter = dict(payload.get("output_adapter") or {})
    output_adapter["stop_markers"] = ["[/scene]"]
    output_adapter["renderer"] = "structured_svg_scene.v1"
    payload["output_adapter"] = output_adapter
    decode = dict(payload.get("decode") or {})
    decode["max_tokens"] = 128
    payload["decode"] = decode
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a balanced probe contract for a spec07 scene DSL run")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing spec07 scene DSL dataset artifacts")
    ap.add_argument("--prefix", default=None, help="Dataset prefix, auto-detected from dataset/tokenizer if omitted")
    ap.add_argument("--output", required=True, type=Path, help="Output contract JSON")
    ap.add_argument("--per-split", type=int, default=12, help="How many balanced cases to include per split")
    args = ap.parse_args()

    prefix = _detect_prefix(args.run.expanduser().resolve(), args.prefix)
    contract = build_contract(args.run, prefix, max(1, int(args.per_split)))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


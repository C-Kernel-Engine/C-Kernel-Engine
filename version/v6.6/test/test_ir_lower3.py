#!/usr/bin/env python3
"""
test_ir_lower3.py - Validate IR Lower 3 (call-ready) has no errors.

This test ensures:
- IR Lower 3 file exists
- No errors reported at top-level
- No per-op errors
"""

import sys
import json
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"


def find_call_ir() -> Optional[Path]:
    candidates = [
        "lowered_decode_call.json",
        "lowered_prefill_call.json",
        "lowered_call.json",
    ]
    for base in [GENERATED_DIR, CACHE_DIR]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
        # Fallback: any *_call.json
        for p in base.glob("*_call.json"):
            return p
    return None


def main() -> int:
    call_ir_path = find_call_ir()
    if not call_ir_path:
        print("ERROR: No IR Lower 3 file found (expected *_call.json)")
        return 1

    with open(call_ir_path, "r") as f:
        ir = json.load(f)

    errors = ir.get("errors", [])
    ops = ir.get("operations", ir.get("ops", []))
    op_errors = [op for op in ops if op.get("errors")]

    print(f"Loaded IR Lower 3: {call_ir_path}")
    print(f"Ops: {len(ops)}, top-level errors: {len(errors)}, op errors: {len(op_errors)}")

    if errors or op_errors:
        if errors:
            print("Top-level errors:")
            for e in errors[:5]:
                print(f"  {e}")
        if op_errors:
            print("Op errors (first 5):")
            for op in op_errors[:5]:
                print(f"  Op {op.get('idx')}: {op.get('errors')}")
        return 1

    # Sanity: ensure all ops have args
    missing_args = [op for op in ops if "args" not in op]
    if missing_args:
        print(f"ERROR: {len(missing_args)} ops missing args")
        return 1

    print("PASS: IR Lower 3 has no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""
v8 codegen wrapper.

v8 still reuses the stable v7 emitter, but vision-only bring-up needs a small
normalization layer because the v7 generator assumes every runtime has decoder
fields such as vocab_size. For encoder-only vision graphs we inject minimal
safe defaults and then delegate to the stable emitter.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import codegen_v7  # type: ignore  # noqa: E402


def _patch_codegen_config(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    cfg = dict(out.get("config", {}) or {})
    if "vocab_size" not in cfg and "n_vocab" not in cfg:
        cfg["vocab_size"] = 1
        cfg["n_vocab"] = 1
        cfg["_v8_codegen_default_vocab"] = True
    out["config"] = cfg
    return out


def _inject_vision_only_fallbacks(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    present = {str(buf.get("name", "")) for buf in act_buffers}
    fallback_defs = []
    for name in ("kv_cache", "rope_cache", "logits"):
        if name not in present:
            macro = f"A_{name.upper()}"
            fallback_defs.append(
                f"#ifndef {macro}\n#define {macro} 0\n#endif"
            )
    if not fallback_defs:
        return code

    block = "/* v8 vision-only fallback macros */\n" + "\n".join(fallback_defs) + "\n"
    insert_after = "#include <math.h>"
    if insert_after in code:
        return code.replace(insert_after, insert_after + "\n\n" + block, 1)
    return block + "\n" + code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v8 codegen wrapper over the stable v7 emitter")
    ap.add_argument("--ir", type=Path, required=True, help="Call-ready IR JSON (IR Lower 3)")
    ap.add_argument("--layout", type=Path, required=True, help="Memory layout JSON")
    ap.add_argument("--output", type=Path, required=True, help="Output C file")
    ap.add_argument("--init", type=Path, default=None, help="Optional init_call.json")
    ap.add_argument("--debug", action="store_true", help="Emit debug dumps")
    ap.add_argument("--profile", action="store_true", help="Emit profiling wrappers")
    ap.add_argument("--parity-dump", action="store_true", help="Emit parity dump helpers")
    ap.add_argument("--strict-contracts", action="store_true", help="Fail on strict contract/codegen errors")
    args = ap.parse_args(argv)

    with open(args.ir, "r", encoding="utf-8") as f:
        ir_obj = _patch_codegen_config(json.load(f))
    with open(args.layout, "r", encoding="utf-8") as f:
        layout_obj = _patch_codegen_config(json.load(f))

    init_call_obj = None
    init_path = args.init if args.init is not None else args.ir.parent / "init_call.json"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            init_call_obj = _patch_codegen_config(json.load(f))

    with tempfile.TemporaryDirectory(prefix="codegen_v8_") as td:
        td_path = Path(td)
        ir_path = td_path / "call.v8.json"
        layout_path = td_path / "layout.v8.json"
        ir_path.write_text(json.dumps(ir_obj, indent=2), encoding="utf-8")
        layout_path.write_text(json.dumps(layout_obj, indent=2), encoding="utf-8")
        code = codegen_v7.generate(
            ir_path,
            layout_path,
            debug=args.debug,
            init_call=init_call_obj,
            profile=args.profile,
            dump=args.parity_dump,
            strict_contracts=args.strict_contracts,
        )
        code = _inject_vision_only_fallbacks(code, layout_obj)

    args.output.write_text(code, encoding="utf-8")
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

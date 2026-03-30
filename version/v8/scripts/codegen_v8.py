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
import re
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
import codegen_prefill_v7  # type: ignore  # noqa: E402


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


def _inject_strict_vision_encoder_oracle(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    by_name = {str(buf.get("name", "")): buf for buf in act_buffers}
    image_buf = by_name.get("image_input")
    output_buf = by_name.get("vision_output")
    if not image_buf or not output_buf:
        return code

    cfg = dict(layout_obj.get("config", {}) or {})
    image_h = int(cfg.get("image_size", 0) or 0)
    image_w = int(cfg.get("image_size", 0) or 0)
    if image_h <= 0 or image_w <= 0:
        return code

    image_elems = int(image_buf.get("size_bytes", image_buf.get("size", 0)) or 0) // 4
    output_elems = int(output_buf.get("size_bytes", output_buf.get("size", 0)) or 0) // 4
    pixel_count = image_h * image_w
    if pixel_count <= 0 or image_elems <= 0 or output_elems <= 0:
        return code
    if image_elems % pixel_count != 0:
        return code

    channels = image_elems // pixel_count
    if channels <= 0:
        return code

    block = f"""    {{
        const char *strict_mtmd_oracle_env = getenv("CK_STRICT_MTMD_CLIP_ORACLE");
        int strict_mtmd_oracle = strict_mtmd_oracle_env ? (atoi(strict_mtmd_oracle_env) != 0) : 0;
        if (strict_mtmd_oracle && ck_strict_parity_enabled()) {{
            if (ck_strict_mtmd_clip_encode_planar_f32(
                    (const float*)(MEM + A_IMAGE_INPUT),
                    {channels},
                    {image_h},
                    {image_w},
                    (float*)(MEM + A_VISION_OUTPUT),
                    {output_elems})) {{
                model->pos++;
                return;
            }}
        }}
    }}
"""

    token_store_pat = re.compile(
        r"(    /\* Store token at offset [^\n]+\n"
        r"    \*\(int32_t\*\)\([^\n]+\) = token;\n)"
    )
    return token_store_pat.sub(r"\1\n" + block + "\n", code, count=1)


def _inject_activation_lookup_api(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    if not act_buffers:
        return code

    weights = (layout_obj.get("memory", {}) or {}).get("weights", {}) or {}
    activation_base = int(weights.get("base_offset", 0) or 0) + int(weights.get("size", 0) or 0)

    cases = []
    for buf in act_buffers:
        name = str(buf.get("name", "") or "")
        if not name:
            continue
        runtime_offset = activation_base + int(buf.get("offset", 0) or 0)
        size_bytes = int(buf.get("size_bytes", buf.get("size", 0)) or 0)
        c_name = json.dumps(name)
        cases.append(
            "    if (strcmp(name, {name}) == 0) {{\n"
            "        if (offset_out) *offset_out = (size_t){offset};\n"
            "        if (size_out) *size_out = (size_t){size};\n"
            "        return 1;\n"
            "    }}".format(name=c_name, offset=runtime_offset, size=size_bytes)
        )
    if not cases:
        return code

    block = """/* v8 activation lookup helpers for external hosts */
static int ck_lookup_named_activation_info(const char *name, size_t *offset_out, size_t *size_out) {
    if (!name) return 0;
{cases}
    return 0;
}

CK_EXPORT intptr_t ck_model_get_named_activation_runtime_offset(const char *name) {
    size_t offset = 0;
    if (!ck_lookup_named_activation_info(name, &offset, NULL)) return -1;
    return (intptr_t)offset;
}

CK_EXPORT intptr_t ck_model_get_named_activation_nbytes(const char *name) {
    size_t size = 0;
    if (!ck_lookup_named_activation_info(name, NULL, &size)) return -1;
    return (intptr_t)size;
}

CK_EXPORT uintptr_t ck_model_get_named_activation_ptr(const char *name) {
    size_t offset = 0;
    if (!g_model) return (uintptr_t)0;
    if (!ck_lookup_named_activation_info(name, &offset, NULL)) return (uintptr_t)0;
    return (uintptr_t)(g_model->bump + offset);
}
""".replace("{cases}", "\n".join(cases))
    return code + "\n\n" + block


def _normalize_prefill_for_decode_layout(
    prefill_obj: Dict[str, Any] | None,
    layout_obj: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Align appended prefill code with the target decode runtime layout.

    v8 appends a prefill entrypoint into the decode runtime C file. That means
    the prefill code must obey the decode runtime's activation layout, not the
    standalone prefill lowering defaults. In particular, decode runtimes often
    expose last-only logits storage `[1, vocab]`, while standalone prefill IR
    may request full `[T, vocab]` logits plus a `copy_last_logits` fixup.
    Emitting that full logits GEMM into the decode layout overruns A_LOGITS.
    """
    if prefill_obj is None:
        return None

    out = dict(prefill_obj)
    cfg = dict(out.get("config", {}) or {})
    layout_cfg = dict(layout_obj.get("config", {}) or {})
    decode_logits_layout = str(layout_cfg.get("logits_layout", cfg.get("logits_layout", "auto"))).lower()
    if decode_logits_layout != "last":
        return out

    cfg["logits_layout"] = "last"
    out["config"] = cfg

    ops = out.get("operations", [])
    if isinstance(ops, list):
        out["operations"] = [
            op for op in ops
            if str((op or {}).get("op", "")) != "copy_last_logits"
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v8 codegen wrapper over the stable v7 emitter")
    ap.add_argument("--ir", type=Path, required=True, help="Call-ready IR JSON (IR Lower 3)")
    ap.add_argument("--layout", type=Path, required=True, help="Memory layout JSON")
    ap.add_argument("--output", type=Path, required=True, help="Output C file")
    ap.add_argument("--prefill", type=Path, default=None, help="Optional lowered prefill IR for decoder runtimes")
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
    prefill_obj = None
    if args.prefill is not None:
        with open(args.prefill, "r", encoding="utf-8") as f:
            prefill_obj = _patch_codegen_config(json.load(f))
        prefill_obj = _normalize_prefill_for_decode_layout(prefill_obj, layout_obj)

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
        prefill_code = ""
        if prefill_obj is not None:
            prefill_path = td_path / "prefill.v8.json"
            prefill_path.write_text(json.dumps(prefill_obj, indent=2), encoding="utf-8")
            prefill_code = codegen_prefill_v7.generate_prefill(
                prefill_path,
                profile=args.profile,
                dump=args.parity_dump,
            )
        code = codegen_v7.generate(
            ir_path,
            layout_path,
            debug=args.debug,
            init_call=init_call_obj,
            profile=args.profile,
            dump=args.parity_dump,
            strict_contracts=args.strict_contracts,
        )
        if prefill_code:
            insert_marker = "#include <math.h>"
            if insert_marker in code:
                code = code.replace(
                    insert_marker,
                    insert_marker + "\n\n/* Prefill support enabled */\n#define CK_HAS_PREFILL 1",
                    1,
                )
            code = code + "\n\n" + prefill_code
        code = _inject_vision_only_fallbacks(code, layout_obj)
        code = _inject_strict_vision_encoder_oracle(code, layout_obj)
        code = _inject_activation_lookup_api(code, layout_obj)

    args.output.write_text(code, encoding="utf-8")
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

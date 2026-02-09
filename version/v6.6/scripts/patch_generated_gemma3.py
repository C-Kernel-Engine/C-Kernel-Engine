#!/usr/bin/env python3
"""
Patch generated model_v6_6.c for Gemma3 embedding scaling.

Why:
  llama.cpp Gemma3 applies inp_scaled = inp_embd * sqrt(n_embd) before layer 0.
  If this scale is missing, residual paths drift badly even when early attn_norm
  looks close.

This script patches generated C directly (temporary unblock), without touching
codegen or IR. It is idempotent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DECODE_SCALE_BLOCK = """    /* GEMMA3_EMBED_SCALE_PATCH: inp_scaled = inp_embd * sqrt(EMBED_DIM) */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        for (int i = 0; i < EMBED_DIM; ++i) {
            emb[i] *= emb_scale;
        }
    }
    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*)(model->bump + A_EMBEDDED_INPUT), -1, "inp_scaled", EMBED_DIM);
    #endif
"""

PREFILL_SCALE_BLOCK = """    /* GEMMA3_EMBED_SCALE_PATCH: inp_scaled = inp_embd * sqrt(EMBED_DIM) */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        const int n = num_tokens * EMBED_DIM;
        for (int i = 0; i < n; ++i) {
            emb[i] *= emb_scale;
        }
    }
    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*)(model->bump + A_EMBEDDED_INPUT), -1, "inp_scaled", num_tokens * EMBED_DIM);
    #endif
"""


def _check_gemma3(model_dir: Path) -> None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    model_type = str(cfg.get("model_type", "")).lower()
    if "gemma" not in model_type:
        raise SystemExit(f"Refusing to patch non-Gemma model_type={model_type!r} at {model_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch generated Gemma3 C with embedding scaling")
    ap.add_argument("--model-dir", type=Path, required=True, help="ck_build directory")
    ap.add_argument("--backup", action="store_true", help="Write model_v6_6.c.bak before patching")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    c_path = model_dir / "model_v6_6.c"
    if not c_path.exists():
        raise SystemExit(f"Missing generated file: {c_path}")

    _check_gemma3(model_dir)

    src = c_path.read_text(encoding="utf-8", errors="ignore")
    if "GEMMA3_EMBED_SCALE_PATCH" in src:
        print(f"[patch] already patched: {c_path}")
        return 0

    if args.backup:
        bak = c_path.with_suffix(c_path.suffix + ".bak")
        bak.write_text(src, encoding="utf-8")
        print(f"[patch] backup: {bak}")

    decode_re = re.compile(r"^(\s*/\* Op \d+: memcpy \(residual_save\) layer=0 section=body \*/)", re.MULTILINE)
    prefill_re = re.compile(r"^(\s*/\* Op \d+: memcpy \(residual_save\) layer=0 \*/)", re.MULTILINE)

    out, n_decode = decode_re.subn(DECODE_SCALE_BLOCK + r"\1", src, count=1)
    if n_decode != 1:
        raise SystemExit("Decode embedding anchor not found. Regenerate and retry.")

    out, n_prefill = prefill_re.subn(PREFILL_SCALE_BLOCK + r"\1", out, count=1)
    if n_prefill != 1:
        raise SystemExit("Prefill embedding anchor not found. Regenerate and retry.")

    c_path.write_text(out, encoding="utf-8")
    print(f"[patch] applied Gemma3 embedding scale patch to {c_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
test_v4_q4k_pipeline.py
=======================

End-to-end sanity pipeline for v4 Q4_K:
  1) Convert GGUF -> bump (1 layer)
  2) Build IR v4 (1 layer) + compile libmodel.so
  3) Run smoke test + PyTorch parity (random weights)
  4) Repeat for 2 layers
  5) Optionally convert full model if --full is set
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, verbose=False, env=None) -> None:
    if verbose:
        print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def emit_wrapper(out_dir: Path, verbose: bool) -> Path:
    decode_headers = sorted(out_dir.glob("generated_*_decode.h"))
    decode_sources = sorted(out_dir.glob("generated_*_decode.c"))
    if not decode_headers or not decode_sources:
        raise SystemExit(f"Missing generated decode C/H files in {out_dir}")

    decode_header = decode_headers[0].name
    decode_source = decode_sources[0].name
    prefix = decode_header.replace("generated_", "").replace(".h", "")
    safe_name = re.sub(r"[^A-Za-z0-9]", "_", prefix).upper()

    wrapper = f"""\
// AUTO-GENERATED v4 wrapper: {prefix}
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "{decode_header}"
#include "ckernel_model_load_v4.h"

#include "{decode_source}"

static {safe_name}Model g_model;
static int g_initialized = 0;
static int g_active_tokens = 0;
static int g_kv_cache_enabled = 0;
static int g_kv_cache_capacity = {safe_name}_MAX_SEQ_LEN;
static int g_kv_cache_tokens = 0;

static int32_t *g_tokens = NULL;
static int g_tokens_cap = 0;

static float *g_logits = NULL;
static size_t g_logits_cap = 0;

static int ensure_tokens_capacity(int n) {{
    if (n <= g_tokens_cap) return 0;
    int32_t *buf = (int32_t *)realloc(g_tokens, (size_t)n * sizeof(int32_t));
    if (!buf) return -1;
    g_tokens = buf;
    g_tokens_cap = n;
    return 0;
}}

static int ensure_logits_capacity(int n) {{
    size_t needed = (size_t)n * (size_t){safe_name}_VOCAB_SIZE;
    if (needed <= g_logits_cap) return 0;
    float *buf = (float *)realloc(g_logits, needed * sizeof(float));
    if (!buf) return -1;
    g_logits = buf;
    g_logits_cap = needed;
    return 0;
}}

static const char *manifest_path_from_weights(const char *weights_path,
                                             char *out,
                                             size_t out_len) {{
    if (!weights_path || !out || out_len == 0) return NULL;
    const char *slash = strrchr(weights_path, '/');
    size_t dir_len = slash ? (size_t)(slash - weights_path + 1) : 0;
    const char *fname = "weights_manifest.map";
    size_t fname_len = strlen(fname);
    if (dir_len + fname_len + 1 > out_len) return NULL;
    if (dir_len) {{
        memcpy(out, weights_path, dir_len);
    }}
    memcpy(out + dir_len, fname, fname_len + 1);
    return out;
}}

int ck_model_init(const char *weights_path) {{
    if (g_initialized) return 0;
    if (!weights_path) {{
        fprintf(stderr, "ck_model_init: missing weights path\\n");
        return -1;
    }}

    char manifest_path[1024];
    if (!manifest_path_from_weights(weights_path, manifest_path, sizeof(manifest_path))) {{
        fprintf(stderr, "ck_model_init: failed to resolve manifest path\\n");
        return -1;
    }}

    if ({prefix}_init(&g_model) != 0) {{
        fprintf(stderr, "ck_model_init: model init failed\\n");
        return -2;
    }}

    if (ck_load_weights_manifest_v4(g_model.memory, weights_path, manifest_path) != 0) {{
        fprintf(stderr, "ck_model_init: weights load failed\\n");
        return -3;
    }}

    g_initialized = 1;
    return 0;
}}

void ck_model_free(void) {{
    if (!g_initialized) return;
    {prefix}_free(&g_model);
    free(g_tokens);
    free(g_logits);
    g_tokens = NULL;
    g_logits = NULL;
    g_tokens_cap = 0;
    g_logits_cap = 0;
    g_initialized = 0;
}}

int ck_model_kv_cache_enable(int capacity) {{
    g_kv_cache_enabled = 1;
    g_kv_cache_capacity = capacity;
    return 0;
}}

int ck_model_embed_tokens(const int32_t *tokens, int n_tokens) {{
    if (!g_initialized) return -1;
    if (ensure_tokens_capacity(n_tokens) != 0) return -2;
    memcpy(g_tokens, tokens, (size_t)n_tokens * sizeof(int32_t));
    g_active_tokens = n_tokens;
    return 0;
}}

int ck_model_forward(float *logits_out) {{
    if (!g_initialized) return -1;
    if (g_active_tokens <= 0) return -2;
    if (ensure_logits_capacity(g_active_tokens) != 0) return -3;

    if ({prefix}_prefill(&g_model, g_tokens, g_active_tokens) != 0) {{
        fprintf(stderr, "ck_model_forward: prefill failed\\n");
        return -4;
    }}

    if (logits_out) {{
        memcpy(logits_out,
               g_model.logits,
               (size_t)g_active_tokens * (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    }}
    g_kv_cache_tokens = g_active_tokens;
    return 0;
}}

int ck_model_decode(int32_t token, float *logits_out) {{
    if (!g_initialized) return -1;
    if (!g_kv_cache_enabled) return -2;
    if (g_kv_cache_tokens >= g_kv_cache_capacity) return -3;

    if ({prefix}_decode(&g_model, &token, g_kv_cache_tokens) != 0) {{
        fprintf(stderr, "ck_model_decode: decode failed\\n");
        return -4;
    }}

    if (logits_out) {{
        memcpy(logits_out,
               g_model.logits + (size_t)g_kv_cache_tokens * {safe_name}_VOCAB_SIZE,
               (size_t){safe_name}_VOCAB_SIZE * sizeof(float));
    }}
    g_kv_cache_tokens += 1;
    g_active_tokens = g_kv_cache_tokens;
    return 0;
}}

int ck_model_get_vocab_size(void) {{
    return {safe_name}_VOCAB_SIZE;
}}

int ck_model_get_context_window(void) {{
    return {safe_name}_MAX_SEQ_LEN;
}}

int ck_model_get_active_tokens(void) {{
    return g_active_tokens;
}}

int ck_model_verify_canaries(void) {{
    if (!g_initialized) return -1;
    return {prefix}_verify_canaries(&g_model);
}}
"""

    model_c_path = out_dir / "model.c"
    model_c_path.write_text(wrapper)
    if verbose:
        print(f"[emit] {model_c_path}")
    return model_c_path


def compile_generated(out_dir: Path, verbose: bool) -> Path:
    model_c = emit_wrapper(out_dir, verbose)
    so_path = out_dir / "libmodel.so"

    kernel_sources = [str(p) for p in (ROOT / "src" / "kernels").glob("*.c")]
    extra_sources = [
        str(ROOT / "src" / "ckernel_model_load_v4.c"),
        str(ROOT / "src" / "ckernel_orchestration.c"),
        str(ROOT / "src" / "ckernel_strict.c"),
        str(ROOT / "src" / "cpu_features.c"),
    ]

    cmd = [
        "gcc",
        "-O3",
        "-fPIC",
        "-fopenmp",
        "-shared",
        f"-I{ROOT / 'include'}",
        f"-I{out_dir}",
        "-o",
        str(so_path),
        str(model_c),
    ] + kernel_sources + extra_sources + ["-lm"]

    run(cmd, verbose)
    return so_path


def run_case(gguf: Path, layers: int, validate: bool, verbose: bool, run_parity: bool) -> None:
    out_dir = ROOT / "build" / f"v4_q4k_l{layers}"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = out_dir / "weights.bump"
    manifest = out_dir / "weights_manifest_input.json"
    config = out_dir / "config.json"

    convert_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "convert_gguf_to_bump_v4.py"),
        "--gguf",
        str(gguf),
        "--output",
        str(weights),
        "--manifest-out",
        str(manifest),
        "--config-out",
        str(config),
        "--max-layers",
        str(layers),
    ]
    if validate:
        convert_cmd.append("--validate")
    run(convert_cmd, verbose)

    ir_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_ir_v4.py"),
        "--config",
        str(config),
        "--name",
        f"v4_q4k_l{layers}",
        "--prefix",
        str(out_dir),
        "--modes",
        "prefill,decode",
        "--dtype",
        "fp32",
        "--weight-dtype",
        "q4_k",
        "--weights-manifest",
        str(manifest),
        "--max-layers",
        str(layers),
        "--emit",
        "lib",
    ]
    run(ir_cmd, verbose)

    compile_generated(out_dir, verbose)

    run_env = os.environ.copy()
    lib_dir = str(ROOT / "build")
    run_env["LD_LIBRARY_PATH"] = lib_dir + (
        (":" + run_env["LD_LIBRARY_PATH"]) if run_env.get("LD_LIBRARY_PATH") else ""
    )

    smoke_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ck_model_smoke_v4.py"),
        "--model-dir",
        str(out_dir),
        "--weights",
        str(weights),
        "--prompt-len",
        "4",
        "--decode-steps",
        "2",
    ]
    run(smoke_cmd, verbose, env=run_env)

    if run_parity:
        parity_cmd = [
            sys.executable,
            str(ROOT / "unittest" / "test_multi_layer_parity.py"),
            "--layers",
            str(layers),
        ]
        run(parity_cmd, verbose, env=run_env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v4 Q4_K pipeline tests.")
    ap.add_argument("--gguf", required=True, help="Path to GGUF model file")
    ap.add_argument("--layers", default="1,2", help="Layer counts to test (comma-separated)")
    ap.add_argument("--validate", action="store_true", help="Enable converter validation checks")
    ap.add_argument("--no-parity", action="store_true", help="Skip PyTorch parity test")
    ap.add_argument("--full", action="store_true", help="Convert full model after 1/2 layer checks")
    ap.add_argument("--verbose", action="store_true", help="Print subprocess commands")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"GGUF file not found: {gguf}")

    layer_counts = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layer_counts:
        raise SystemExit("--layers must include at least one integer")

    run_parity = not args.no_parity

    for layers in layer_counts:
        print(f"[v4-q4k] testing {layers} layer(s)")
        run_case(gguf, layers, args.validate, args.verbose, run_parity)

    if args.full:
        out_dir = ROOT / "build" / "v4_q4k_full"
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / "weights.bump"
        manifest = out_dir / "weights_manifest_input.json"
        config = out_dir / "config.json"

        convert_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "convert_gguf_to_bump_v4.py"),
            "--gguf",
            str(gguf),
            "--output",
            str(weights),
            "--manifest-out",
            str(manifest),
            "--config-out",
            str(config),
        ]
        if args.validate:
            convert_cmd.append("--validate")
        run(convert_cmd, args.verbose)

        ir_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "build_ir_v4.py"),
            "--config",
            str(config),
            "--name",
            "v4_q4k_full",
            "--prefix",
            str(out_dir),
            "--modes",
            "prefill,decode",
            "--dtype",
            "fp32",
            "--weight-dtype",
            "q4_k",
            "--weights-manifest",
            str(manifest),
            "--emit",
            "lib",
        ]
        run(ir_cmd, args.verbose)
        compile_generated(out_dir, args.verbose)

    print("[v4-q4k] done")


if __name__ == "__main__":
    main()

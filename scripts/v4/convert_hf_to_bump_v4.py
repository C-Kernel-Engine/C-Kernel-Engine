#!/usr/bin/env python3
"""
convert_hf_to_bump_v4.py

Convert HF Llama-style weights into a v4 bump layout (RoPE-first, no pos_emb/bias).
This matches the v4 graph/layout expectations (token_emb + per-layer weights + final norm).
"""

import argparse
import json
import os
import struct

import numpy as np

import build_ir_v4 as v4
from convert_hf_to_bump import (
    HashingWriter,
    align_up_elems,
    get_optional,
    get_state_dict,
    get_tensor,
    load_config,
    pick,
    write_matrix_padded_f32,
    write_matrix_q4_k,
    write_qkv_packed_f32,
    write_qkv_packed_q4_k,
    write_row_q4_k,
    write_vector_f32,
    write_wo_packed_f32,
)

CACHE_ALIGN = 64
HEADER_SIZE = 128
FLOAT_SIZE = 4

CK_DT_FP32 = 0
CK_DT_Q4_K = 6


def build_dtype_table(weight_names, q4k):
    dtypes = []
    for name in weight_names:
        if not q4k:
            dtypes.append(CK_DT_FP32)
            continue
        if name in {"token_emb", "lm_head_weight"}:
            dtypes.append(CK_DT_Q4_K)
        elif name.endswith((".wq", ".wk", ".wv", ".wo", ".w1", ".w2")):
            dtypes.append(CK_DT_Q4_K)
        else:
            dtypes.append(CK_DT_FP32)
    return bytes(dtypes)


def main():
    parser = argparse.ArgumentParser(description="Convert HF weights to bump v4 format")
    parser.add_argument("--checkpoint", required=True, help="HF model directory (local)")
    parser.add_argument("--config", help="Optional config JSON (overrides model config)")
    parser.add_argument("--output", required=True, help="Output bump weights file")
    parser.add_argument("--context", type=int, help="Override context length (for small tests)")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output dtype: float32 (default) or q4_k/q4_k_m (weights only; norms stay fp32)",
    )
    parser.add_argument("--map-out", help="Optional JSON map of weight order/dtypes")
    parser.add_argument("--manifest-out", help="Optional JSON manifest with file offsets/sizes")
    args = parser.parse_args()

    dtype = str(args.dtype).lower().strip()
    q4k = dtype in ("q4_k", "q4_k_m", "q4k", "q4km")
    if not (dtype == "float32" or q4k):
        raise SystemExit("Unsupported --dtype (expected float32, q4_k, or q4_k_m)")

    state_dict, hf_config = get_state_dict(args.checkpoint, torch_dtype=None)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = hf_config.to_dict()

    num_layers = pick(cfg, ["num_hidden_layers", "num_layers"])
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    intermediate = pick(cfg, ["intermediate_size"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    num_kv_heads = pick(cfg, ["num_key_value_heads", "num_kv_heads"], num_heads)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)
    if args.context is not None:
        context_len = int(args.context)

    if not all([num_layers, embed_dim, intermediate, num_heads, vocab_size, context_len]):
        raise SystemExit("Config missing required fields for conversion")

    head_dim = embed_dim // num_heads
    qk_align_bytes = 256 * FLOAT_SIZE
    aligned_embed_dim = align_up_elems(embed_dim, FLOAT_SIZE, qk_align_bytes if q4k else CACHE_ALIGN)
    aligned_head_dim = align_up_elems(head_dim, FLOAT_SIZE, CACHE_ALIGN)
    aligned_intermediate = align_up_elems(intermediate, FLOAT_SIZE, qk_align_bytes if q4k else CACHE_ALIGN)
    aligned_context = align_up_elems(context_len, FLOAT_SIZE, CACHE_ALIGN)

    if q4k:
        if aligned_embed_dim != embed_dim:
            print(f"[warn] Q4_K padded embed_dim {embed_dim} -> {aligned_embed_dim}")
        if aligned_intermediate != intermediate:
            print(f"[warn] Q4_K padded intermediate {intermediate} -> {aligned_intermediate}")

    model_name = cfg.get("model_type", "model")
    graph = v4.build_graph_ir_v4(
        {
            "model_type": cfg.get("model_type", "llama"),
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_dim": intermediate,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "max_seq_len": context_len,
            "rope_theta": cfg.get("rope_theta", 10000.0),
            "dtype": "bf16" if q4k else "fp32",
            "tie_word_embeddings": cfg.get("tie_word_embeddings", True),
        },
        model_name,
        CACHE_ALIGN,
    )

    section = graph["sections"][0]
    weight_names = []

    for buf in section["buffers"]["header"]:
        if buf["role"] == "weight":
            weight_names.append(buf["name"])

    for layer in range(num_layers):
        for buf in section["buffers"]["layer"]:
            if buf["role"] != "weight":
                continue
            name = buf["name"].replace("{L}", str(layer))
            if buf.get("tied_to"):
                continue
            weight_names.append(name)

    for buf in section["buffers"]["footer"]:
        if buf["role"] == "weight":
            if buf.get("tied_to"):
                continue
            weight_names.append(buf["name"])

    dtype_table = build_dtype_table(weight_names, q4k)
    manifest_entries = []

    def record_entry(name: str, dtype_name: str, start: int, size: int) -> None:
        manifest_entries.append(
            {
                "name": name,
                "dtype": dtype_name,
                "file_offset": HEADER_SIZE + start,
                "size": size,
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w+b") as f:
        f.write(b"\x00" * HEADER_SIZE)
        w = HashingWriter(f)

        w.write(struct.pack("<I", len(dtype_table)))
        w.write(dtype_table)

        tok = get_tensor(
            state_dict,
            "model.embed_tokens.weight",
            alt_keys=("model.tok_embeddings.weight",),
        ).detach().cpu().numpy()
        start = w.bytes_written
        if q4k:
            write_matrix_q4_k(w, tok, vocab_size, embed_dim, aligned_embed_dim)
            dtype_name = "q4_k"
        else:
            write_matrix_padded_f32(w, tok, vocab_size, embed_dim, aligned_embed_dim)
            dtype_name = "fp32"
        record_entry("token_emb", dtype_name, start, w.bytes_written - start)

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}"
            ln1 = get_tensor(state_dict, f"{prefix}.input_layernorm.weight").detach().cpu().numpy()
            ln2 = get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight").detach().cpu().numpy()
            start = w.bytes_written
            write_vector_f32(w, ln1, aligned_embed_dim)
            record_entry(f"layer.{layer}.ln1_gamma", "fp32", start, w.bytes_written - start)

            wq = get_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight").detach().cpu().numpy()
            wk = get_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight").detach().cpu().numpy()
            wv = get_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight").detach().cpu().numpy()

            for name, mat, heads in (
                ("wq", wq, num_heads),
                ("wk", wk, num_kv_heads),
                ("wv", wv, num_kv_heads),
            ):
                start = w.bytes_written
                if q4k:
                    write_qkv_packed_q4_k(w, mat, heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                    dtype_name = "q4_k"
                else:
                    write_qkv_packed_f32(w, mat, heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                    dtype_name = "fp32"
                record_entry(f"layer.{layer}.{name}", dtype_name, start, w.bytes_written - start)

            wo = get_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight").detach().cpu().numpy()
            start = w.bytes_written
            if q4k:
                write_matrix_q4_k(w, wo, embed_dim, embed_dim, aligned_embed_dim, aligned_embed_dim)
                dtype_name = "q4_k"
            else:
                write_wo_packed_f32(w, wo, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                dtype_name = "fp32"
            record_entry(f"layer.{layer}.wo", dtype_name, start, w.bytes_written - start)

            start = w.bytes_written
            write_vector_f32(w, ln2, aligned_embed_dim)
            record_entry(f"layer.{layer}.ln2_gamma", "fp32", start, w.bytes_written - start)

            gate = get_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight").detach().cpu().numpy()
            up = get_tensor(state_dict, f"{prefix}.mlp.up_proj.weight").detach().cpu().numpy()
            down = get_tensor(state_dict, f"{prefix}.mlp.down_proj.weight").detach().cpu().numpy()

            start = w.bytes_written
            if q4k:
                for r in range(2 * aligned_intermediate):
                    row = np.zeros(aligned_embed_dim, dtype=np.float32)
                    if r < intermediate:
                        row[:embed_dim] = gate[r, :embed_dim].astype(np.float32)
                    elif aligned_intermediate <= r < (aligned_intermediate + intermediate):
                        row[:embed_dim] = up[r - aligned_intermediate, :embed_dim].astype(np.float32)
                    write_row_q4_k(w, row)
                dtype_name = "q4_k"
            else:
                w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
                w1[:intermediate, :embed_dim] = gate[:intermediate, :embed_dim]
                w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up[:intermediate, :embed_dim]
                w.write(w1.ravel().tobytes())
                dtype_name = "fp32"
            record_entry(f"layer.{layer}.w1", dtype_name, start, w.bytes_written - start)

            start = w.bytes_written
            if q4k:
                write_matrix_q4_k(w, down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim)
                dtype_name = "q4_k"
            else:
                write_matrix_padded_f32(w, down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim)
                dtype_name = "fp32"
            record_entry(f"layer.{layer}.w2", dtype_name, start, w.bytes_written - start)

        ln_f = get_tensor(state_dict, "model.norm.weight").detach().cpu().numpy()
        start = w.bytes_written
        write_vector_f32(w, ln_f, aligned_embed_dim)
        record_entry("final_ln_weight", "fp32", start, w.bytes_written - start)

        tie = cfg.get("tie_word_embeddings", True)
        if not tie and "lm_head.weight" not in state_dict:
            raise SystemExit("tie_word_embeddings=false but lm_head.weight is missing")
        if not tie:
            lm_head = get_tensor(state_dict, "lm_head.weight").detach().cpu().numpy()
            start = w.bytes_written
            if q4k:
                write_matrix_q4_k(w, lm_head, vocab_size, embed_dim, aligned_embed_dim)
                dtype_name = "q4_k"
            else:
                write_matrix_padded_f32(w, lm_head, vocab_size, embed_dim, aligned_embed_dim)
                dtype_name = "fp32"
            record_entry("lm_head_weight", dtype_name, start, w.bytes_written - start)

        checksum = w.digest()

        f.flush()
        f.seek(0)
        f.write(b"BUMPWGT4")
        f.write(struct.pack("<I", 4))  # version
        f.write(struct.pack("<I", 1))  # model_type (legacy)
        f.write(struct.pack("<I", int(num_layers)))
        f.write(struct.pack("<I", int(vocab_size)))
        f.write(struct.pack("<I", int(embed_dim)))
        f.write(struct.pack("<I", int(intermediate)))
        f.write(struct.pack("<I", int(context_len)))
        f.write(struct.pack("<I", int(num_heads)))
        f.write(struct.pack("<I", int(num_kv_heads)))
        f.write(struct.pack("<I", int(head_dim)))
        f.write(struct.pack("<Q", int(aligned_embed_dim)))
        f.write(struct.pack("<Q", int(aligned_head_dim)))
        f.write(struct.pack("<Q", int(aligned_intermediate)))
        f.write(struct.pack("<Q", int(aligned_context)))
        f.write(checksum)
        f.write(b"\x00" * 16)

    if args.map_out:
        os.makedirs(os.path.dirname(args.map_out) or ".", exist_ok=True)
        with open(args.map_out, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "model_type": cfg.get("model_type", "llama"),
                    "num_layers": num_layers,
                    "weights": [
                        {"name": name, "dtype": int(dtype_table[i])}
                        for i, name in enumerate(weight_names)
                    ],
                },
                mf,
                indent=2,
            )

    if args.manifest_out:
        os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "format": "ck-bumpwgt4-manifest-v1",
                    "weights_path": args.output,
                    "entries": manifest_entries,
                },
                mf,
                indent=2,
            )

    mb = (HEADER_SIZE + w.bytes_written) / (1024 * 1024)
    if q4k:
        print(
            f"Wrote {args.output} ({mb:.2f} MB, q4_k, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})"
        )
    else:
        print(f"Wrote {args.output} ({mb:.2f} MB, fp32, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})")


if __name__ == "__main__":
    main()

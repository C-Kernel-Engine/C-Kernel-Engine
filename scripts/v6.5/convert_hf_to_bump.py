#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct

import numpy as np

from q4_k_quantize import iter_q4_k_row_bytes


CACHE_ALIGN = 64
FLOAT_SIZE = 4
HEADER_SIZE = 128

CK_DT_FP32 = 0
CK_DT_Q4_K = 6


def align_up_elems(elems, elem_bytes=FLOAT_SIZE, align_bytes=CACHE_ALIGN):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned_bytes = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned_bytes // elem_bytes


class HashingWriter:
    def __init__(self, f):
        self._f = f
        self._h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes):
        if not data:
            return
        self._f.write(data)
        self._h.update(data)
        self.bytes_written += len(data)

    def digest(self) -> bytes:
        return self._h.digest()


def write_zero_f32(w: HashingWriter, count_floats: int) -> int:
    # Avoid giant allocations for long contexts.
    chunk_floats = 4096
    chunk = b"\x00" * (chunk_floats * FLOAT_SIZE)
    remaining = int(count_floats)
    while remaining > 0:
        n = chunk_floats if remaining > chunk_floats else remaining
        w.write(chunk[: n * FLOAT_SIZE])
        remaining -= n
    return count_floats * FLOAT_SIZE


def write_vector_f32(w: HashingWriter, vec, aligned_dim) -> int:
    buf = np.zeros(aligned_dim, dtype=np.float32)
    if vec is not None:
        flat = vec.astype(np.float32).reshape(-1)
        buf[:flat.size] = flat
    w.write(buf.tobytes())
    return buf.size * FLOAT_SIZE


def write_matrix_padded_f32(w: HashingWriter, mat, out_dim, in_dim, aligned_in, aligned_out=None) -> int:
    if aligned_out is None:
        aligned_out = out_dim
    buf = np.zeros((aligned_out, aligned_in), dtype=np.float32)
    if mat is not None:
        buf[:out_dim, :in_dim] = mat[:out_dim, :in_dim].astype(np.float32)
    w.write(buf.ravel().tobytes())
    return buf.size * FLOAT_SIZE


def write_row_q4_k(w: HashingWriter, row: np.ndarray) -> int:
    written = 0
    for blk in iter_q4_k_row_bytes(row):
        w.write(blk)
        written += len(blk)
    return written


def write_matrix_q4_k(w: HashingWriter, mat, out_dim, in_dim, aligned_in, aligned_out=None) -> int:
    if aligned_out is None:
        aligned_out = out_dim
    if aligned_in % 256 != 0:
        raise SystemExit(f"Q4_K requires aligned_in multiple of 256 (got {aligned_in})")

    bytes_written = 0
    for r in range(aligned_out):
        row = np.zeros(aligned_in, dtype=np.float32)
        if mat is not None and r < out_dim:
            row[:in_dim] = mat[r, :in_dim].astype(np.float32)
        bytes_written += write_row_q4_k(w, row)
    return bytes_written


def build_dtype_table(num_layers: int, use_q4k: bool) -> bytes:
    weight_dtype = CK_DT_Q4_K if use_q4k else CK_DT_FP32
    dtypes = [weight_dtype]  # token_emb
    dtypes.append(CK_DT_FP32)  # pos_emb
    for _ in range(num_layers):
        dtypes.extend([
            CK_DT_FP32,  # ln1_gamma
            CK_DT_FP32,  # ln2_gamma
            weight_dtype,  # wq
            CK_DT_FP32,  # bq
            weight_dtype,  # wk
            CK_DT_FP32,  # bk
            weight_dtype,  # wv
            CK_DT_FP32,  # bv
            weight_dtype,  # wo
            CK_DT_FP32,  # bo
            weight_dtype,  # w1
            CK_DT_FP32,  # b1
            weight_dtype,  # w2
            CK_DT_FP32,  # b2
        ])
    dtypes.append(CK_DT_FP32)  # final_norm
    dtypes.append(CK_DT_FP32)  # final_bias
    return bytes(dtypes)


def pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    if weight is None:
        return buf
    w = weight.astype(np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :embed_dim] = w[src_base:src_base + head_dim, :embed_dim]
    return buf


def write_qkv_packed_f32(w: HashingWriter, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim) -> int:
    buf = pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
    w.write(buf.ravel().tobytes())
    return buf.size * FLOAT_SIZE


def write_qkv_packed_q4_k(
    w: HashingWriter,
    weight,
    num_heads,
    head_dim,
    aligned_head_dim,
    embed_dim,
    aligned_embed_dim,
) -> int:
    if aligned_embed_dim % 256 != 0:
        raise SystemExit(f"Q4_K requires aligned_embed_dim multiple of 256 (got {aligned_embed_dim})")

    bytes_written = 0
    for h in range(num_heads):
        for r in range(aligned_head_dim):
            row = np.zeros(aligned_embed_dim, dtype=np.float32)
            if weight is not None and r < head_dim:
                src_row = h * head_dim + r
                row[:embed_dim] = weight[src_row, :embed_dim].astype(np.float32)
            bytes_written += write_row_q4_k(w, row)
    return bytes_written


def pack_qkv_bias(bias, num_heads, head_dim, aligned_head_dim):
    buf = np.zeros(num_heads * aligned_head_dim, dtype=np.float32)
    if bias is None:
        return buf
    b = bias.astype(np.float32).reshape(-1)
    for h in range(num_heads):
        src_base = h * head_dim
        dst_base = h * aligned_head_dim
        buf[dst_base:dst_base + head_dim] = b[src_base:src_base + head_dim]
    return buf


def write_qkv_bias_packed_f32(w: HashingWriter, bias, num_heads, head_dim, aligned_head_dim) -> int:
    buf = pack_qkv_bias(bias, num_heads, head_dim, aligned_head_dim)
    w.write(buf.tobytes())
    return buf.size * FLOAT_SIZE


def write_wo_packed_f32(w: HashingWriter, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim) -> int:
    total = 0
    wmat = weight.astype(np.float32) if weight is not None else None
    for h in range(num_heads):
        block = np.zeros((aligned_embed_dim, aligned_head_dim), dtype=np.float32)
        if wmat is not None:
            col_base = h * head_dim
            block[:embed_dim, :head_dim] = wmat[:embed_dim, col_base:col_base + head_dim]
        w.write(block.ravel().tobytes())
        total += block.size * FLOAT_SIZE
    return total


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def pick(cfg, keys, default=None):
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise SystemExit("torch is required to convert HF weights") from exc


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
        return AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required to convert HF weights") from exc


def get_state_dict(checkpoint, torch_dtype):
    torch = require_torch()
    AutoModelForCausalLM = require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    return model.state_dict(), model.config


def get_tensor(state, key, alt_keys=()):
    if key in state:
        return state[key]
    for alt in alt_keys:
        if alt in state:
            return state[alt]
    raise KeyError(f"Missing weight: {key}")


def get_optional(state, key, alt_keys=()):
    if key in state:
        return state[key]
    for alt in alt_keys:
        if alt in state:
            return state[alt]
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert HF Llama-style weights to bump format")
    parser.add_argument("--checkpoint", required=True, help="HF model directory (local)")
    parser.add_argument("--config", help="Optional config JSON (overrides model config)")
    parser.add_argument("--output", required=True, help="Output bump weights file")
    parser.add_argument("--context", type=int, help="Override context length (for small tests)")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output dtype: float32 (default) or q4_k/q4_k_m (weights only; norms/biases stay fp32)",
    )
    args = parser.parse_args()

    dtype = str(args.dtype).lower().strip()
    q4k = dtype in ("q4_k", "q4_k_m", "q4k", "q4km")
    if not (dtype == "float32" or q4k):
        raise SystemExit("Unsupported --dtype (expected float32, q4_k, or q4_k_m)")

    state_dict, hf_config = get_state_dict(args.checkpoint, torch_dtype=None)

    cfg = {}
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
    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)
    aligned_context = align_up_elems(context_len)

    if q4k:
        if embed_dim % 256 != 0 or intermediate % 256 != 0:
            raise SystemExit("Q4_K requires embed_dim and intermediate_size to be multiples of 256")
        if aligned_embed_dim != embed_dim or aligned_head_dim != head_dim or aligned_intermediate != intermediate:
            raise SystemExit("Q4_K conversion currently requires no padding (aligned dims must equal model dims)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    dtype_table = build_dtype_table(num_layers, q4k)

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
        if q4k:
            write_matrix_q4_k(w, tok, vocab_size, embed_dim, aligned_embed_dim)
        else:
            write_matrix_padded_f32(w, tok, vocab_size, embed_dim, aligned_embed_dim)

        write_zero_f32(w, context_len * aligned_embed_dim)

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}"
            ln1 = get_tensor(state_dict, f"{prefix}.input_layernorm.weight").detach().cpu().numpy()
            ln2 = get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight").detach().cpu().numpy()
            write_vector_f32(w, ln1, aligned_embed_dim)
            write_vector_f32(w, ln2, aligned_embed_dim)

            wq = get_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight").detach().cpu().numpy()
            wk = get_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight").detach().cpu().numpy()
            wv = get_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight").detach().cpu().numpy()

            bq = get_optional(state_dict, f"{prefix}.self_attn.q_proj.bias")
            bk = get_optional(state_dict, f"{prefix}.self_attn.k_proj.bias")
            bv = get_optional(state_dict, f"{prefix}.self_attn.v_proj.bias")

            if q4k:
                write_qkv_packed_q4_k(w, wq, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            else:
                write_qkv_packed_f32(w, wq, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            write_qkv_bias_packed_f32(
                w,
                None if bq is None else bq.detach().cpu().numpy(),
                num_heads,
                head_dim,
                aligned_head_dim,
            )
            if q4k:
                write_qkv_packed_q4_k(w, wk, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            else:
                write_qkv_packed_f32(w, wk, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            write_qkv_bias_packed_f32(
                w,
                None if bk is None else bk.detach().cpu().numpy(),
                num_kv_heads,
                head_dim,
                aligned_head_dim,
            )
            if q4k:
                write_qkv_packed_q4_k(w, wv, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            else:
                write_qkv_packed_f32(w, wv, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            write_qkv_bias_packed_f32(
                w,
                None if bv is None else bv.detach().cpu().numpy(),
                num_kv_heads,
                head_dim,
                aligned_head_dim,
            )

            wo = get_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight").detach().cpu().numpy()
            bo = get_optional(state_dict, f"{prefix}.self_attn.o_proj.bias")
            if q4k:
                # Q4_K path expects a fused [D x D] projection matrix in row-major.
                write_matrix_q4_k(w, wo, embed_dim, embed_dim, aligned_embed_dim, aligned_embed_dim)
            else:
                write_wo_packed_f32(w, wo, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
            write_vector_f32(w, None if bo is None else bo.detach().cpu().numpy(), aligned_embed_dim)

            gate = get_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight").detach().cpu().numpy()
            up = get_tensor(state_dict, f"{prefix}.mlp.up_proj.weight").detach().cpu().numpy()
            down = get_tensor(state_dict, f"{prefix}.mlp.down_proj.weight").detach().cpu().numpy()

            if q4k:
                # Avoid materializing the full [2*Hff x D] buffer for quantization.
                for r in range(2 * aligned_intermediate):
                    row = np.zeros(aligned_embed_dim, dtype=np.float32)
                    if r < intermediate:
                        row[:embed_dim] = gate[r, :embed_dim].astype(np.float32)
                    elif aligned_intermediate <= r < (aligned_intermediate + intermediate):
                        row[:embed_dim] = up[r - aligned_intermediate, :embed_dim].astype(np.float32)
                    write_row_q4_k(w, row)
            else:
                w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
                w1[:intermediate, :embed_dim] = gate[:intermediate, :embed_dim]
                w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up[:intermediate, :embed_dim]
                w.write(w1.ravel().tobytes())

            b1_gate = get_optional(state_dict, f"{prefix}.mlp.gate_proj.bias")
            b1_up = get_optional(state_dict, f"{prefix}.mlp.up_proj.bias")
            b1 = np.zeros((2 * aligned_intermediate,), dtype=np.float32)
            if b1_gate is not None:
                g = b1_gate.detach().cpu().numpy()
                b1[:intermediate] = g[:intermediate]
            if b1_up is not None:
                u = b1_up.detach().cpu().numpy()
                b1[aligned_intermediate:aligned_intermediate + intermediate] = u[:intermediate]
            w.write(b1.tobytes())

            if q4k:
                write_matrix_q4_k(w, down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim)
            else:
                write_matrix_padded_f32(w, down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim)

            b2 = get_optional(state_dict, f"{prefix}.mlp.down_proj.bias")
            write_vector_f32(w, None if b2 is None else b2.detach().cpu().numpy(), aligned_embed_dim)

        ln_f = get_tensor(state_dict, "model.norm.weight").detach().cpu().numpy()
        write_vector_f32(w, ln_f, aligned_embed_dim)
        write_zero_f32(w, aligned_embed_dim)

        f.flush()
        checksum = w.digest()
        f.seek(0)
        f.write(b"BUMPWGT3")
        f.write(struct.pack("I", 3))
        f.write(struct.pack("I", 1))
        f.write(struct.pack("I", int(num_layers)))
        f.write(struct.pack("I", int(vocab_size)))
        f.write(struct.pack("I", int(embed_dim)))
        f.write(struct.pack("I", int(context_len)))
        f.write(struct.pack("I", int(num_heads)))
        f.write(struct.pack("I", int(head_dim)))
        f.write(struct.pack("Q", int(aligned_embed_dim)))
        f.write(struct.pack("Q", int(aligned_head_dim)))
        f.write(struct.pack("Q", int(aligned_context)))
        f.write(checksum)
        f.write(b"\x00" * 32)

    mb = (HEADER_SIZE + w.bytes_written) / (1024 * 1024)
    if q4k:
        print(
            f"Wrote {args.output} ({mb:.2f} MB, q4_k, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})\n"
            "Run with: CK_WEIGHT_DTYPE=q4_k_m ./your_generated_binary ..."
        )
    else:
        print(f"Wrote {args.output} ({mb:.2f} MB, fp32, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})")


if __name__ == "__main__":
    main()

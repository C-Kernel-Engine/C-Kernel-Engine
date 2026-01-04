#!/usr/bin/env python3
"""
convert_gguf_to_bump_v4.py
==========================

Convert GGUF weights (llama.cpp format) into the v4 bump layout:
  - RoPE-first (no pos_emb)
  - No biases
  - w1 = concat(gate, up)
  - w2 = down

This keeps quantized tensors (Q4_K/Q6_K) in-place and only reorders where
the runtime expects head-major packing.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from typing import Dict, Optional

import numpy as np

import convert_gguf_to_bump as gguf
from convert_hf_to_bump import (
    write_matrix_padded_f32,
    write_qkv_packed_f32,
    write_vector_f32,
    write_wo_packed_f32,
)


HEADER_SIZE = 128
CACHE_ALIGN = 64
FLOAT_SIZE = 4
QK_ALIGN_BYTES = 256 * FLOAT_SIZE

CK_DT_FP32 = gguf.CK_DT_FP32
CK_DT_Q4_K = gguf.CK_DT_Q4_K
CK_DT_Q6_K = gguf.CK_DT_Q6_K


def read_matrix_f32(f, base: int, info: gguf.TensorInfo) -> np.ndarray:
    if len(info.dims) != 2:
        raise gguf.GGUFError(f"{info.name}: expected 2D, got dims={info.dims}")
    rows = int(info.ne1)
    cols = int(info.ne0)
    row_bytes = gguf.ggml_row_bytes(info.ggml_type, cols)
    nbytes = row_bytes * rows
    f.seek(base + info.offset, os.SEEK_SET)
    raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise gguf.GGUFError(f"{info.name}: unexpected EOF while reading matrix")

    if info.ggml_type == gguf.GGML_TYPE_F32:
        mat = np.frombuffer(raw, dtype=np.float32)
    elif info.ggml_type == gguf.GGML_TYPE_F16:
        mat = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif info.ggml_type == gguf.GGML_TYPE_BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        mat = u32.view(np.float32)
    elif info.ggml_type == gguf.GGML_TYPE_Q5_0:
        if cols % 32 != 0:
            raise gguf.GGUFError(
                f"{info.name}: Q5_0 requires cols % 32 == 0 (got cols={cols})"
            )
        blocks_per_row = cols // 32
        mat = np.empty((rows, cols), dtype=np.float32)
        offset = 0
        for r in range(rows):
            row = mat[r]
            for b in range(blocks_per_row):
                scale = np.frombuffer(raw[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
                qs = np.frombuffer(raw[offset + 2:offset + 18], dtype=np.uint8)
                qh = np.frombuffer(raw[offset + 18:offset + 22], dtype=np.uint8)
                for i in range(32):
                    low = (qs[i // 2] >> ((i & 1) * 4)) & 0x0F
                    high = (qh[i // 8] >> (i & 7)) & 0x01
                    val = int((high << 4) | low)
                    row[b * 32 + i] = (val - 16) * scale
                offset += 22
    elif info.ggml_type == gguf.GGML_TYPE_Q5_1:
        if cols % 32 != 0:
            raise gguf.GGUFError(
                f"{info.name}: Q5_1 requires cols % 32 == 0 (got cols={cols})"
            )
        blocks_per_row = cols // 32
        mat = np.empty((rows, cols), dtype=np.float32)
        offset = 0
        for r in range(rows):
            row = mat[r]
            for b in range(blocks_per_row):
                scale = np.frombuffer(raw[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
                bias = np.frombuffer(raw[offset + 2:offset + 4], dtype=np.float16)[0].astype(np.float32)
                qs = np.frombuffer(raw[offset + 4:offset + 20], dtype=np.uint8)
                qh = np.frombuffer(raw[offset + 20:offset + 24], dtype=np.uint8)
                for i in range(32):
                    low = (qs[i // 2] >> ((i & 1) * 4)) & 0x0F
                    high = (qh[i // 8] >> (i & 7)) & 0x01
                    val = int((high << 4) | low)
                    row[b * 32 + i] = val * scale + bias
                offset += 24
    elif info.ggml_type == gguf.GGML_TYPE_Q8_0:
        if cols % 32 != 0:
            raise gguf.GGUFError(
                f"{info.name}: Q8_0 requires cols % 32 == 0 (got cols={cols})"
            )
        blocks_per_row = cols // 32
        mat = np.empty((rows, cols), dtype=np.float32)
        offset = 0
        for r in range(rows):
            row = mat[r]
            for b in range(blocks_per_row):
                scale = np.frombuffer(raw[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
                qs = np.frombuffer(raw[offset + 2:offset + 34], dtype=np.int8).astype(np.float32)
                row[b * 32:(b + 1) * 32] = qs * scale
                offset += 34
    else:
        raise gguf.GGUFError(
            f"{info.name}: unsupported matrix type {gguf.ggml_type_name(info.ggml_type)}"
        )
    return mat.reshape(rows, cols)


def weight_dtype(info: gguf.TensorInfo, label: str) -> int:
    if info.ggml_type in (gguf.GGML_TYPE_Q4_K, gguf.GGML_TYPE_Q6_K):
        return gguf.ck_dtype_from_ggml_type(info.ggml_type)
    if info.ggml_type in (
        gguf.GGML_TYPE_F32,
        gguf.GGML_TYPE_F16,
        gguf.GGML_TYPE_BF16,
        gguf.GGML_TYPE_Q8_0,
        gguf.GGML_TYPE_Q5_0,
        gguf.GGML_TYPE_Q5_1,
    ):
        return CK_DT_FP32
    raise gguf.GGUFError(
        f"{info.name}: expected Q4_K/Q6_K/Q8_0/Q5_0/Q5_1/F32/F16/BF16 for {label}, got {gguf.ggml_type_name(info.ggml_type)}"
    )


def is_quantized(dt: int) -> bool:
    return dt in (CK_DT_Q4_K, CK_DT_Q6_K)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert GGUF (Q4_K/Q6_K/F16/BF16/F32) weights to bump v4")
    ap.add_argument("--gguf", required=True, help="Input GGUF file (e.g. model.Q4_K_M.gguf)")
    ap.add_argument("--output", help="Output bump weights file (required unless --inspect/--list)")
    ap.add_argument("--context", type=int, help="Override context length (max_position_embeddings)")
    ap.add_argument("--inspect", action="store_true", help="Show GGUF summary and exit")
    ap.add_argument("--list", action="store_true", help="List all tensors and exit")
    ap.add_argument("--max-layers", type=int, help="Limit number of layers to convert (for quick tests)")
    ap.add_argument("--validate", action="store_true", help="Validate quantized alignment before conversion")
    ap.add_argument("--config-out", help="Optional config JSON output path")
    ap.add_argument("--map-out", help="Optional JSON map of weight order/dtypes")
    ap.add_argument("--manifest-out", help="Optional JSON manifest with file offsets/sizes")
    args = ap.parse_args()

    if not args.output and not (args.inspect or args.list):
        raise SystemExit("--output is required unless --inspect/--list is set")

    with open(args.gguf, "rb") as f:
        r = gguf.GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise gguf.GGUFError("Invalid GGUF magic")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        meta: Dict[str, object] = {}
        arch_prefixes = ("general.", "llama.", "qwen2.", "qwen.")
        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            if args.inspect or args.list or key.startswith(arch_prefixes):
                meta[key] = gguf._gguf_read_value(r, vtype)
            else:
                gguf._gguf_skip_value(r, vtype)

        tensors: Dict[str, gguf.TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.key_str()
            n_dims = r.u32()
            dims = tuple(int(r.u64()) for _ in range(n_dims))
            ggml_type = r.u32()
            offset = r.u64()
            tensors[name] = gguf.TensorInfo(
                name=name,
                dims=dims,
                ggml_type=int(ggml_type),
                offset=int(offset),
            )

        alignment = int(meta.get("general.alignment", 32))
        data_start = gguf.align_up(r.tell(), alignment)
        r.seek(data_start)

        arch = str(meta.get("general.architecture", "llama")).lower()

        if args.inspect or args.list:
            counts: Dict[int, int] = {}
            bytes_by_type: Dict[int, int] = {}
            for info in tensors.values():
                counts[info.ggml_type] = counts.get(info.ggml_type, 0) + 1
                try:
                    bytes_by_type[info.ggml_type] = bytes_by_type.get(info.ggml_type, 0) + gguf.ggml_tensor_bytes(info)
                except Exception:
                    pass

            def fmt_bytes(n: int) -> str:
                if n >= 1024 * 1024 * 1024:
                    return f"{n / (1024 ** 3):.2f} GiB"
                if n >= 1024 * 1024:
                    return f"{n / (1024 ** 2):.2f} MiB"
                if n >= 1024:
                    return f"{n / 1024:.2f} KiB"
                return f"{n} B"

            print(f"[gguf] file={args.gguf}")
            print(f"[gguf] version={version} arch={arch} tensors={n_tensors} kv={n_kv} alignment={alignment}")
            print("[gguf] tensor types:")
            for tcode, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                b = bytes_by_type.get(tcode)
                b_str = fmt_bytes(b) if b is not None else "?"
                print(f"  - {gguf.ggml_type_name(tcode):>10}: {cnt:5d} tensors, bytes={b_str}")

            highlight = [
                "token_embd.weight",
                "output.weight",
                "output_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
            ]
            print("[gguf] key tensors:")
            for name in highlight:
                info = tensors.get(name)
                if not info:
                    continue
                print(f"  - {name}: {gguf.ggml_type_name(info.ggml_type)} dims={info.dims}")

            if args.list:
                print("[gguf] all tensors:")
                for name in sorted(tensors.keys()):
                    info = tensors[name]
                    print(f"  - {name}: {gguf.ggml_type_name(info.ggml_type)} dims={info.dims}")
            return

        def meta_int(key: str) -> Optional[int]:
            v = meta.get(key)
            if v is None:
                return None
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int, np.integer)):
                return int(v)
            return None

        def meta_float(key: str) -> Optional[float]:
            v = meta.get(key)
            if v is None:
                return None
            if isinstance(v, (float, np.floating)):
                return float(v)
            if isinstance(v, (int, np.integer)):
                return float(v)
            return None

        tok_name = "token_embd.weight"
        if tok_name not in tensors:
            raise gguf.GGUFError(f"Missing required tensor: {tok_name}")
        tok = tensors[tok_name]
        if len(tok.dims) != 2:
            raise gguf.GGUFError(f"{tok_name}: expected 2D, got dims={tok.dims}")

        def meta_int_arch(suffix: str) -> Optional[int]:
            prefixes = (arch, "llama", "qwen2", "qwen")
            seen = set()
            for prefix in prefixes:
                if not prefix or prefix in seen:
                    continue
                seen.add(prefix)
                value = meta_int(f"{prefix}.{suffix}")
                if value is not None:
                    return value
            return None

        def meta_float_arch(suffix: str) -> Optional[float]:
            prefixes = (arch, "llama", "qwen2", "qwen")
            seen = set()
            for prefix in prefixes:
                if not prefix or prefix in seen:
                    continue
                seen.add(prefix)
                value = meta_float(f"{prefix}.{suffix}")
                if value is not None:
                    return value
            return None

        embed_dim = meta_int_arch("embedding_length") or tok.ne0
        vocab_size = tok.ne1

        num_layers = meta_int_arch("block_count")
        if num_layers is None:
            layer_ids = []
            for name in tensors:
                if name.startswith("blk.") and ".attn_norm.weight" in name:
                    try:
                        layer_ids.append(int(name.split(".")[1]))
                    except Exception:
                        pass
            if not layer_ids:
                raise gguf.GGUFError("Could not infer num_layers (missing llama.block_count and no blk.* tensors found)")
            num_layers = max(layer_ids) + 1

        intermediate = meta_int_arch("feed_forward_length")
        if intermediate is None:
            gate0 = tensors.get("blk.0.ffn_gate.weight")
            if gate0 and len(gate0.dims) == 2:
                intermediate = gate0.ne1
        if intermediate is None:
            raise gguf.GGUFError("Could not determine intermediate_size (missing llama.feed_forward_length)")

        num_heads = meta_int_arch("attention.head_count")
        if num_heads is None:
            raise gguf.GGUFError("Missing attention.head_count (num_heads)")
        num_kv_heads = meta_int_arch("attention.head_count_kv") or num_heads

        context_len = meta_int_arch("context_length") or 0
        if args.context is not None:
            context_len = int(args.context)
        if context_len <= 0:
            raise gguf.GGUFError("Could not determine context length (use --context to override)")

        rope_theta = meta_float_arch("rope.freq_base") or 10000.0
        rms_eps = meta_float_arch("norm_rms_eps") or 1e-5

        if embed_dim != tok.ne0:
            raise gguf.GGUFError(f"{tok_name}: embedding_length mismatch (meta={embed_dim}, tensor.ne0={tok.ne0})")
        if embed_dim % num_heads != 0:
            raise gguf.GGUFError(f"hidden_size {embed_dim} not divisible by num_heads {num_heads}")

        head_dim = embed_dim // num_heads
        embed_kv = num_kv_heads * head_dim

        aligned_embed_dim = gguf.align_up_elems(embed_dim, FLOAT_SIZE, CACHE_ALIGN)
        aligned_head_dim = gguf.align_up_elems(head_dim, FLOAT_SIZE, CACHE_ALIGN)
        aligned_intermediate = gguf.align_up_elems(intermediate, FLOAT_SIZE, CACHE_ALIGN)
        aligned_context = gguf.align_up_elems(context_len, FLOAT_SIZE, CACHE_ALIGN)

        required = {
            "output_norm.weight",
        }
        for name in required:
            if name not in tensors:
                raise gguf.GGUFError(f"Missing required tensor: {name}")

        token_dt = weight_dtype(tok, "token_emb")
        quant_embed = is_quantized(token_dt)
        quant_intermediate = False

        if args.max_layers is not None:
            if args.max_layers <= 0:
                raise gguf.GGUFError(f"--max-layers must be >= 1 (got {args.max_layers})")
            if args.max_layers < num_layers:
                print(f"[info] limiting layers: {args.max_layers}/{num_layers}")
            num_layers = min(num_layers, args.max_layers)

        layer_infos = []
        dtype_table = [token_dt]
        weight_names = ["token_emb"]

        for layer in range(num_layers):
            attn_norm = tensors.get(f"blk.{layer}.attn_norm.weight")
            ffn_norm = tensors.get(f"blk.{layer}.ffn_norm.weight")
            if not attn_norm or not ffn_norm:
                raise gguf.GGUFError(f"Layer {layer}: missing attn_norm/ffn_norm tensors")

            wq = tensors.get(f"blk.{layer}.attn_q.weight")
            wk = tensors.get(f"blk.{layer}.attn_k.weight")
            wv = tensors.get(f"blk.{layer}.attn_v.weight")
            wo = tensors.get(f"blk.{layer}.attn_output.weight")
            gate = tensors.get(f"blk.{layer}.ffn_gate.weight")
            up = tensors.get(f"blk.{layer}.ffn_up.weight")
            down = tensors.get(f"blk.{layer}.ffn_down.weight")
            if not wq or not wk or not wv or not wo:
                raise gguf.GGUFError(f"Layer {layer}: missing attention projection tensors (q/k/v/o)")
            if not gate or not up or not down:
                raise gguf.GGUFError(f"Layer {layer}: missing ffn tensors (gate/up/down)")

            if wq.ne0 != embed_dim or wq.ne1 != embed_dim:
                raise gguf.GGUFError(f"{wq.name}: expected dims [ne0={embed_dim}, ne1={embed_dim}], got {wq.dims}")
            for tensor, label in ((wk, "K"), (wv, "V")):
                if tensor.ne0 != embed_dim or tensor.ne1 != embed_kv:
                    raise gguf.GGUFError(
                        f"{tensor.name}: expected dims [ne0={embed_dim}, ne1={embed_kv}] for {label}, got {tensor.dims}"
                    )
            if wo.ne0 != embed_dim or wo.ne1 != embed_dim:
                raise gguf.GGUFError(f"{wo.name}: expected dims [ne0={embed_dim}, ne1={embed_dim}], got {wo.dims}")

            for tensor, label in ((gate, "gate"), (up, "up")):
                if tensor.ne0 != embed_dim or tensor.ne1 != intermediate:
                    raise gguf.GGUFError(
                        f"{tensor.name}: expected dims [ne0={embed_dim}, ne1={intermediate}] for {label}, got {tensor.dims}"
                    )
            if down.ne0 != intermediate or down.ne1 != embed_dim:
                raise gguf.GGUFError(
                    f"{down.name}: expected dims [ne0={intermediate}, ne1={embed_dim}] for down, got {down.dims}"
                )

            wq_dt = weight_dtype(wq, "attn_q")
            wk_dt = weight_dtype(wk, "attn_k")
            wv_dt = weight_dtype(wv, "attn_v")
            wo_dt = weight_dtype(wo, "attn_output")
            gate_dt = weight_dtype(gate, "ffn_gate")
            up_dt = weight_dtype(up, "ffn_up")
            down_dt = weight_dtype(down, "ffn_down")
            if gate_dt != up_dt:
                raise gguf.GGUFError(
                    f"Layer {layer}: ffn_gate ({gguf.ggml_type_name(gate.ggml_type)}) and "
                    f"ffn_up ({gguf.ggml_type_name(up.ggml_type)}) must match"
                )

            if any(is_quantized(dt) for dt in (wq_dt, wk_dt, wv_dt, wo_dt, gate_dt, up_dt)):
                quant_embed = True
            if is_quantized(down_dt):
                quant_intermediate = True

            dtype_table.extend([
                CK_DT_FP32,  # ln1_gamma
                wq_dt,
                wk_dt,
                wv_dt,
                wo_dt,
                CK_DT_FP32,  # ln2_gamma
                gate_dt,     # w1
                down_dt,     # w2
            ])

            weight_names.extend([
                f"layer.{layer}.ln1_gamma",
                f"layer.{layer}.wq",
                f"layer.{layer}.wk",
                f"layer.{layer}.wv",
                f"layer.{layer}.wo",
                f"layer.{layer}.ln2_gamma",
                f"layer.{layer}.w1",
                f"layer.{layer}.w2",
            ])

            layer_infos.append({
                "attn_norm": attn_norm,
                "ffn_norm": ffn_norm,
                "wq": wq,
                "wk": wk,
                "wv": wv,
                "wo": wo,
                "gate": gate,
                "up": up,
                "down": down,
                "wq_dt": wq_dt,
                "wk_dt": wk_dt,
                "wv_dt": wv_dt,
                "wo_dt": wo_dt,
                "gate_dt": gate_dt,
                "down_dt": down_dt,
            })

        output_weight = tensors.get("output.weight")
        tie_word_embeddings = output_weight is None
        if output_weight is not None and len(output_weight.dims) != 2:
            raise gguf.GGUFError("output.weight must be a 2D matrix if present")

        dtype_table.append(CK_DT_FP32)  # final_norm
        weight_names.append("final_ln_weight")

        if output_weight is not None:
            out_dt = weight_dtype(output_weight, "lm_head")
            dtype_table.append(out_dt)
            weight_names.append("lm_head_weight")

        if output_weight is not None and is_quantized(weight_dtype(output_weight, "lm_head")):
            quant_embed = True

        if quant_embed:
            aligned_embed_dim = gguf.align_up_elems(embed_dim, FLOAT_SIZE, QK_ALIGN_BYTES)
            if aligned_embed_dim != embed_dim:
                print(f"[warn] K-quant padded embed_dim {embed_dim} -> {aligned_embed_dim}")
        else:
            if aligned_embed_dim != embed_dim:
                print(f"[warn] aligned_embed_dim={aligned_embed_dim} (embed_dim={embed_dim}) - padding FP32 weights")

        if quant_intermediate:
            aligned_intermediate = gguf.align_up_elems(intermediate, FLOAT_SIZE, QK_ALIGN_BYTES)
            if aligned_intermediate != intermediate:
                print(f"[warn] K-quant padded intermediate {intermediate} -> {aligned_intermediate}")
        else:
            if aligned_intermediate != intermediate:
                print(f"[warn] aligned_intermediate={aligned_intermediate} (intermediate={intermediate}) - padding FP32 weights")

        if args.validate:
            issues = []
            if quant_embed and aligned_embed_dim != embed_dim:
                issues.append(
                    f"K-quant weights require embed_dim aligned to 256 (embed_dim={embed_dim}, aligned={aligned_embed_dim})"
                )
            if quant_intermediate and aligned_intermediate != intermediate:
                issues.append(
                    f"K-quant weights require intermediate aligned to 256 (intermediate={intermediate}, aligned={aligned_intermediate})"
                )
            if issues:
                details = "\n  - " + "\n  - ".join(issues)
                raise gguf.GGUFError(f"Validation failed:{details}")

        dtype_table_bytes = bytes(dtype_table)
        manifest_entries = []

        def dtype_name(dt: int) -> str:
            if dt == CK_DT_Q4_K:
                return "q4_k"
            if dt == CK_DT_Q6_K:
                return "q6_k"
            return "fp32"

        def record_entry(name: str, dtype_str: str, start: int, size: int) -> None:
            manifest_entries.append(
                {
                    "name": name,
                    "dtype": dtype_str,
                    "file_offset": HEADER_SIZE + start,
                    "size": size,
                }
            )

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w+b") as out_f:
            out_f.write(b"\x00" * HEADER_SIZE)
            w = gguf.HashingWriter(out_f)

            w.write(struct.pack("<I", len(dtype_table_bytes)))
            w.write(dtype_table_bytes)

            # 1) token embeddings
            start = w.bytes_written
            if is_quantized(token_dt):
                gguf.copy_bytes_stream(f, data_start + tok.offset, gguf.ggml_tensor_bytes(tok), w)
            else:
                tok_mat = read_matrix_f32(f, data_start, tok)
                write_matrix_padded_f32(w, tok_mat, vocab_size, embed_dim, aligned_embed_dim)
            record_entry("token_emb", dtype_name(token_dt), start, w.bytes_written - start)

            # 2) per-layer weights
            for layer in range(num_layers):
                info = layer_infos[layer]

                ln1 = gguf.read_vector_f32(f, data_start, info["attn_norm"])
                start = w.bytes_written
                write_vector_f32(w, ln1, aligned_embed_dim)
                record_entry(f"layer.{layer}.ln1_gamma", "fp32", start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["wq_dt"]):
                    gguf.copy_qk_head_packed(
                        f, data_start, info["wq"], w,
                        group_count=num_heads,
                        head_dim=head_dim,
                        aligned_head_dim=aligned_head_dim,
                        aligned_embed_dim=aligned_embed_dim,
                    )
                else:
                    wq_mat = read_matrix_f32(f, data_start, info["wq"])
                    write_qkv_packed_f32(w, wq_mat, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                record_entry(f"layer.{layer}.wq", dtype_name(info["wq_dt"]), start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["wk_dt"]):
                    gguf.copy_qk_head_packed(
                        f, data_start, info["wk"], w,
                        group_count=num_kv_heads,
                        head_dim=head_dim,
                        aligned_head_dim=aligned_head_dim,
                        aligned_embed_dim=aligned_embed_dim,
                    )
                else:
                    wk_mat = read_matrix_f32(f, data_start, info["wk"])
                    write_qkv_packed_f32(w, wk_mat, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                record_entry(f"layer.{layer}.wk", dtype_name(info["wk_dt"]), start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["wv_dt"]):
                    gguf.copy_qk_head_packed(
                        f, data_start, info["wv"], w,
                        group_count=num_kv_heads,
                        head_dim=head_dim,
                        aligned_head_dim=aligned_head_dim,
                        aligned_embed_dim=aligned_embed_dim,
                    )
                else:
                    wv_mat = read_matrix_f32(f, data_start, info["wv"])
                    write_qkv_packed_f32(w, wv_mat, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                record_entry(f"layer.{layer}.wv", dtype_name(info["wv_dt"]), start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["wo_dt"]):
                    gguf.copy_bytes_stream(f, data_start + info["wo"].offset, gguf.ggml_tensor_bytes(info["wo"]), w)
                else:
                    wo_mat = read_matrix_f32(f, data_start, info["wo"])
                    write_wo_packed_f32(w, wo_mat, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
                record_entry(f"layer.{layer}.wo", dtype_name(info["wo_dt"]), start, w.bytes_written - start)

                ln2 = gguf.read_vector_f32(f, data_start, info["ffn_norm"])
                start = w.bytes_written
                write_vector_f32(w, ln2, aligned_embed_dim)
                record_entry(f"layer.{layer}.ln2_gamma", "fp32", start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["gate_dt"]):
                    gguf.copy_bytes_stream(f, data_start + info["gate"].offset, gguf.ggml_tensor_bytes(info["gate"]), w)
                    gguf.copy_bytes_stream(f, data_start + info["up"].offset, gguf.ggml_tensor_bytes(info["up"]), w)
                else:
                    gate_mat = read_matrix_f32(f, data_start, info["gate"])
                    up_mat = read_matrix_f32(f, data_start, info["up"])
                    w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
                    w1[:intermediate, :embed_dim] = gate_mat[:intermediate, :embed_dim]
                    w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up_mat[:intermediate, :embed_dim]
                    w.write(w1.ravel().tobytes())
                record_entry(f"layer.{layer}.w1", dtype_name(info["gate_dt"]), start, w.bytes_written - start)

                start = w.bytes_written
                if is_quantized(info["down_dt"]):
                    gguf.copy_bytes_stream(f, data_start + info["down"].offset, gguf.ggml_tensor_bytes(info["down"]), w)
                else:
                    down_mat = read_matrix_f32(f, data_start, info["down"])
                    write_matrix_padded_f32(w, down_mat, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim)
                record_entry(f"layer.{layer}.w2", dtype_name(info["down_dt"]), start, w.bytes_written - start)

            # 3) final RMSNorm
            final_norm = gguf.read_vector_f32(f, data_start, tensors["output_norm.weight"])
            start = w.bytes_written
            write_vector_f32(w, final_norm, aligned_embed_dim)
            record_entry("final_ln_weight", "fp32", start, w.bytes_written - start)

            # 4) optional lm_head
            if output_weight is not None:
                out_dt = weight_dtype(output_weight, "lm_head")
                start = w.bytes_written
                if is_quantized(out_dt):
                    gguf.copy_bytes_stream(f, data_start + output_weight.offset, gguf.ggml_tensor_bytes(output_weight), w)
                else:
                    out_mat = read_matrix_f32(f, data_start, output_weight)
                    write_matrix_padded_f32(w, out_mat, vocab_size, embed_dim, aligned_embed_dim)
                record_entry("lm_head_weight", dtype_name(out_dt), start, w.bytes_written - start)

            checksum = w.digest()

            out_f.flush()
            out_f.seek(0, os.SEEK_SET)
            out_f.write(b"BUMPWGT4")
            out_f.write(struct.pack("<I", 4))  # version
            out_f.write(struct.pack("<I", 1))  # model_type (legacy)
            out_f.write(struct.pack("<I", int(num_layers)))
            out_f.write(struct.pack("<I", int(vocab_size)))
            out_f.write(struct.pack("<I", int(embed_dim)))
            out_f.write(struct.pack("<I", int(intermediate)))
            out_f.write(struct.pack("<I", int(context_len)))
            out_f.write(struct.pack("<I", int(num_heads)))
            out_f.write(struct.pack("<I", int(num_kv_heads)))
            out_f.write(struct.pack("<I", int(head_dim)))
            out_f.write(struct.pack("<Q", int(aligned_embed_dim)))
            out_f.write(struct.pack("<Q", int(aligned_head_dim)))
            out_f.write(struct.pack("<Q", int(aligned_intermediate)))
            out_f.write(struct.pack("<Q", int(aligned_context)))
            out_f.write(checksum)
            out_f.write(b"\x00" * 16)

    if args.map_out:
        os.makedirs(os.path.dirname(args.map_out) or ".", exist_ok=True)
        with open(args.map_out, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "model_type": arch,
                    "num_layers": num_layers,
                    "tie_word_embeddings": tie_word_embeddings,
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

    if args.config_out:
        os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
        cfg = gguf.build_llama_config(
            model_type=arch,
            num_layers=num_layers,
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            intermediate_size=intermediate,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            context_window=context_len,
            rope_theta=rope_theta,
            rms_norm_eps=rms_eps,
        )
        cfg["tie_word_embeddings"] = bool(tie_word_embeddings)
        with open(args.config_out, "w", encoding="utf-8") as cf:
            json.dump(cfg, cf, indent=2)
            cf.write("\n")

    print(
        f"[gguf->bump v4] arch={arch} layers={num_layers} hidden={embed_dim} "
        f"heads={num_heads}/{num_kv_heads} ff={intermediate} vocab={vocab_size} "
        f"ctx={context_len} -> {args.output}"
    )


if __name__ == "__main__":
    main()

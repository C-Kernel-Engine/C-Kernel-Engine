#!/usr/bin/env python3
"""
convert_hf_to_bump_v7.py

Convert HF Llama-style weights into a bump layout aligned with the v7 pipeline.
This currently follows the v4 layout expectations (token_emb + per-layer weights + final norm).

BUMPWGT5 Support:
  - Version 5 adds embedded metadata JSON at EOF with hash verification
  - Maintains backward compatibility with BUMPWGT4 via magic-based detection
  - Metadata includes template, config, quant_summary for self-description
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime
from typing import Optional

import numpy as np

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_V4_DIR = _SCRIPT_DIR / "v4"
for path in (_SCRIPT_DIR, _V4_DIR):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

import ir_core_v7 as v4
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

# BUMP format versions
BUMP_VERSION_V4 = 4
BUMP_VERSION_V5 = 5
BUMP_META_FOOTER_MAGIC = b"BUMPV5MD"

CK_DT_FP32 = 0
CK_DT_Q4_K = 6


def tensor_to_numpy_f32(tensor):
    """Convert torch/numpy-like tensor to a CPU float32 numpy array."""
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor, dtype=np.float32)


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


def _canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode("utf-8")


def build_bumpv5_metadata(template_data: dict, config: dict, quant_summary: dict,
                          manifest_hash: Optional[str], created_by: str) -> dict:
    """Build BUMPWGT5 metadata JSON.

    Args:
        template_data: Full template JSON
        config: Model configuration (dims, heads, etc.)
        quant_summary: Per-layer quantization types
        manifest_hash: Hash of sidecar manifest (optional)
        created_by: String identifying the converter

    Returns:
        Metadata dictionary
    """
    # Build metadata with canonical field order
    metadata = {
        "schema_version": 1,
        "format": "BUMPWGT5",
        "created_by": created_by,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "template": template_data,
        "config": config,
        "quant_summary": quant_summary,
    }

    # Add manifest hash if provided
    if manifest_hash:
        metadata["manifest_hash"] = manifest_hash

    return metadata


def calculate_template_hash(template_data: dict) -> str:
    """Calculate SHA256 hash of template for verification."""
    # Canonical JSON for stable hashing
    return hashlib.sha256(_canonical_json_bytes(template_data)).hexdigest()


def calculate_manifest_hash(manifest: Optional[dict]) -> Optional[str]:
    """Calculate SHA256 hash of manifest dict (canonical JSON)."""
    if manifest is None:
        return None
    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def calculate_metadata_hash(metadata: dict) -> bytes:
    """Calculate SHA256 hash of metadata JSON.

    Returns bytes for inclusion in header.
    """
    # Canonical JSON: sorted keys, no whitespace
    return hashlib.sha256(_canonical_json_bytes(metadata)).digest()


def write_bumpv5_footer(f: "BinaryIO", meta_size: int, meta_sha256: bytes) -> None:
    f.write(BUMP_META_FOOTER_MAGIC)
    f.write(struct.pack("<Q", int(meta_size)))
    f.write(meta_sha256)


def load_template_for_model(model_type: str) -> dict:
    template_name = str(model_type).lower()
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates"))
    template_path = os.path.join(base_dir, f"{template_name}.json")
    if not os.path.exists(template_path):
        # Fallback to llama template if present
        fallback_path = os.path.join(base_dir, "llama.json")
        if template_name != "llama" and os.path.exists(fallback_path):
            template_path = fallback_path
        else:
            raise SystemExit(
                f"Missing template for model_type '{model_type}'. "
                f"Expected: {template_path}. Add a template or use --bump-version=4."
            )
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_unk_token(data: dict) -> Optional[str]:
    model = data.get("model", {})
    unk_token = model.get("unk_token")
    if isinstance(unk_token, dict):
        unk_token = unk_token.get("content")
    if not isinstance(unk_token, str) or not unk_token:
        unk_token = data.get("unk_token")
        if isinstance(unk_token, dict):
            unk_token = unk_token.get("content")
    return unk_token if isinstance(unk_token, str) and unk_token else None


def load_tokenizer_json(path: str, vocab_size: int) -> tuple[list[int], bytes, list[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json missing model.vocab")

    tokens_by_id = [""] * vocab_size
    max_id = -1
    for token, idx in vocab.items():
        if not isinstance(idx, int):
            continue
        if idx > max_id:
            max_id = idx
        if 0 <= idx < vocab_size:
            tokens_by_id[idx] = token

    missing_ids = [i for i, t in enumerate(tokens_by_id) if t == ""]
    if max_id + 1 != vocab_size or missing_ids:
        print(
            f"[tokenizer] Warning: vocab_size={vocab_size}, "
            f"max_id={max_id}, missing_ids={len(missing_ids)}"
        )
    if missing_ids:
        unk_token = _extract_unk_token(data)
        if unk_token:
            for idx in missing_ids:
                tokens_by_id[idx] = unk_token
            fill_desc = f"'{unk_token}'"
        else:
            for idx in missing_ids:
                tokens_by_id[idx] = f"<|ck_missing_{idx}|>"
            fill_desc = "<|ck_missing_{id}|>"
        print(f"[tokenizer] Repair: filled {len(missing_ids)} missing ids with {fill_desc}")

    offsets: list[int] = []
    strings_blob = bytearray()
    for token in tokens_by_id:
        offsets.append(len(strings_blob))
        if token:
            strings_blob.extend(token.encode("utf-8"))
        strings_blob.append(0)

    merges_data: list[int] = []
    if isinstance(merges, list):
        for entry in merges:
            if isinstance(entry, str):
                parts = entry.split()
            elif isinstance(entry, (list, tuple)):
                parts = list(entry)
            else:
                continue
            if len(parts) != 2:
                continue
            left, right = parts[0], parts[1]
            left_id = vocab.get(left)
            right_id = vocab.get(right)
            merged_id = vocab.get(left + right)
            if not isinstance(left_id, int) or not isinstance(right_id, int) or not isinstance(merged_id, int):
                continue
            if left_id < 0 or right_id < 0 or merged_id < 0:
                continue
            if left_id >= vocab_size or right_id >= vocab_size or merged_id >= vocab_size:
                continue
            merges_data.extend([left_id, right_id, merged_id])

    return offsets, bytes(strings_blob), merges_data


def main():
    parser = argparse.ArgumentParser(description="Convert HF weights to bump v7 format")
    parser.add_argument("--checkpoint", required=True, help="HF model directory (local)")
    parser.add_argument("--config", help="Optional config JSON (overrides model config)")
    parser.add_argument("--output", required=True, help="Output bump weights file")
    parser.add_argument("--context", type=int, help="Override context length (for small tests)")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output dtype: float32 (default) or q4_k/q4_k_m (weights only; norms stay fp32)",
    )
    parser.add_argument("--tokenizer-json", help="Tokenizer JSON (HuggingFace) to embed vocab + merges")
    parser.add_argument("--map-out", help="Optional JSON map of weight order/dtypes")
    parser.add_argument("--manifest-out", help="Optional JSON manifest with file offsets/sizes")
    parser.add_argument("--bump-version", type=int, default=BUMP_VERSION_V5, choices=[BUMP_VERSION_V4, BUMP_VERSION_V5],
                       help=f"BUMP format version (default: {BUMP_VERSION_V5}). V5 adds embedded metadata.")
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

    vocab_offsets = None
    vocab_strings = None
    vocab_merges = None
    num_merges = 0
    total_vocab_bytes = 0
    if args.tokenizer_json:
        if not os.path.exists(args.tokenizer_json):
            raise SystemExit(f"tokenizer.json not found: {args.tokenizer_json}")
        vocab_offsets, vocab_strings, vocab_merges = load_tokenizer_json(args.tokenizer_json, vocab_size)
        num_merges = len(vocab_merges) // 3
        total_vocab_bytes = len(vocab_strings)
        print(f"[tokenizer] loaded {len(vocab_offsets)} tokens, {num_merges} merges, {total_vocab_bytes} bytes")

    # Some families (e.g., Qwen3) expose an explicit attention head_dim that can
    # differ from hidden_size / num_heads.
    head_dim = pick(cfg, ["head_dim"], None)
    if head_dim is None:
        head_dim = embed_dim // num_heads
    head_dim = int(head_dim)
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
    manifest_dict = None

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

        tok = tensor_to_numpy_f32(
            get_tensor(
                state_dict,
                "model.embed_tokens.weight",
                alt_keys=("model.tok_embeddings.weight",),
            )
        )
        start = w.bytes_written
        if q4k:
            write_matrix_q4_k(w, tok, vocab_size, embed_dim, aligned_embed_dim)
            dtype_name = "q4_k"
        else:
            write_matrix_padded_f32(w, tok, vocab_size, embed_dim, aligned_embed_dim)
            dtype_name = "fp32"
        record_entry("token_emb", dtype_name, start, w.bytes_written - start)

        if vocab_offsets is not None and vocab_strings is not None and vocab_merges is not None:
            offsets_bytes = struct.pack(f"<{len(vocab_offsets)}i", *vocab_offsets)
            start = w.bytes_written
            w.write(offsets_bytes)
            record_entry("vocab_offsets", "i32", start, w.bytes_written - start)

            start = w.bytes_written
            w.write(vocab_strings)
            record_entry("vocab_strings", "u8", start, w.bytes_written - start)

            merges_bytes = b""
            if vocab_merges:
                merges_bytes = struct.pack(f"<{len(vocab_merges)}i", *vocab_merges)
            start = w.bytes_written
            if merges_bytes:
                w.write(merges_bytes)
            record_entry("vocab_merges", "i32", start, w.bytes_written - start)

        missing_qk_norm = 0
        for layer in range(num_layers):
            prefix = f"model.layers.{layer}"
            ln1 = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.input_layernorm.weight"))
            ln2 = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight"))
            start = w.bytes_written
            write_vector_f32(w, ln1, aligned_embed_dim)
            record_entry(f"layer.{layer}.ln1_gamma", "fp32", start, w.bytes_written - start)

            # Qwen3-style per-head Q/K norm gamma vectors.
            # Keep these in fp32 and pad to aligned_head_dim to match kernel expectations.
            q_norm = tensor_to_numpy_f32(get_optional(state_dict, f"{prefix}.self_attn.q_norm.weight"))
            k_norm = tensor_to_numpy_f32(get_optional(state_dict, f"{prefix}.self_attn.k_norm.weight"))
            if q_norm is None:
                q_norm = np.ones(head_dim, dtype=np.float32)
                missing_qk_norm += 1
            if k_norm is None:
                k_norm = np.ones(head_dim, dtype=np.float32)
                missing_qk_norm += 1

            start = w.bytes_written
            write_vector_f32(w, q_norm, aligned_head_dim)
            record_entry(f"layer.{layer}.q_norm", "fp32", start, w.bytes_written - start)
            start = w.bytes_written
            write_vector_f32(w, k_norm, aligned_head_dim)
            record_entry(f"layer.{layer}.k_norm", "fp32", start, w.bytes_written - start)

            wq = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight"))
            wk = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight"))
            wv = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight"))

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

            wo = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight"))
            start = w.bytes_written
            if q4k:
                write_matrix_q4_k(w, wo, embed_dim, embed_dim, aligned_embed_dim, aligned_embed_dim)
                dtype_name = "q4_k"
            else:
                # v7 fp32 path currently uses generic GEMM kernels for out_proj.
                # Those kernels expect plain row-major [embed_dim, q_dim], not
                # per-head packed layout. Keep this un-packed to preserve parity.
                q_dim = num_heads * head_dim
                aligned_q_dim = num_heads * aligned_head_dim
                write_matrix_padded_f32(w, wo, embed_dim, q_dim, aligned_q_dim, aligned_embed_dim)
                dtype_name = "fp32"
            record_entry(f"layer.{layer}.wo", dtype_name, start, w.bytes_written - start)

            start = w.bytes_written
            write_vector_f32(w, ln2, aligned_embed_dim)
            record_entry(f"layer.{layer}.ln2_gamma", "fp32", start, w.bytes_written - start)

            gate = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight"))
            up = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.mlp.up_proj.weight"))
            down = tensor_to_numpy_f32(get_tensor(state_dict, f"{prefix}.mlp.down_proj.weight"))

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

        if missing_qk_norm:
            print(
                f"[qk_norm] Warning: synthesized identity q_norm/k_norm for "
                f"{missing_qk_norm} tensors (missing in checkpoint)"
            )

        ln_f = tensor_to_numpy_f32(get_tensor(state_dict, "model.norm.weight"))
        start = w.bytes_written
        write_vector_f32(w, ln_f, aligned_embed_dim)
        record_entry("final_ln_weight", "fp32", start, w.bytes_written - start)

        tie = cfg.get("tie_word_embeddings", True)
        if not tie and "lm_head.weight" not in state_dict:
            raise SystemExit("tie_word_embeddings=false but lm_head.weight is missing")
        if not tie:
            lm_head = tensor_to_numpy_f32(get_tensor(state_dict, "lm_head.weight"))
            start = w.bytes_written
            if q4k:
                write_matrix_q4_k(w, lm_head, vocab_size, embed_dim, aligned_embed_dim)
                dtype_name = "q4_k"
            else:
                write_matrix_padded_f32(w, lm_head, vocab_size, embed_dim, aligned_embed_dim)
                dtype_name = "fp32"
            record_entry("lm_head_weight", dtype_name, start, w.bytes_written - start)

        checksum = w.digest()

        # Write header based on version
        f.flush()
        f.seek(0)

        if args.bump_version == BUMP_VERSION_V5:
            # BUMPWGT5: Build metadata and append at EOF
            print(f"[convert] Writing BUMPWGT5 with embedded metadata...")

            # Build config from HF config
            config = {
                "model": cfg.get("model_type", "llama"),
                "num_layers": int(num_layers),
                "embed_dim": int(embed_dim),
                "num_heads": int(num_heads),
                "num_kv_heads": int(num_kv_heads),
                "head_dim": int(head_dim),
                "intermediate_size": int(intermediate),
                "context_length": int(context_len),
                "rope_theta": float(cfg.get("rope_theta", 10000.0)),
                "vocab_size": int(vocab_size),
                "rms_eps": float(cfg.get("rms_norm_epsilon", 1e-5)),
            }

            # Build quantization summary
            quant_summary = {}
            for i in range(num_layers):
                quant_summary[f"layer.{i}"] = {
                    "wq": "q4_k" if q4k else "fp32",
                    "wk": "q4_k" if q4k else "fp32",
                    "wv": "q4_k" if q4k else "fp32",
                    "wo": "q4_k" if q4k else "fp32",
                    "w1": "q4_k" if q4k else "fp32",
                    "w2": "q4_k" if q4k else "fp32",
                }

            # Load template JSON
            template_data = load_template_for_model(cfg.get("model_type", "llama"))

            # Calculate hashes
            manifest_dict = None
            if args.manifest_out:
                manifest_dict = {
                    "format": "ck-bumpwgt5-manifest-v1",
                    "version": args.bump_version,
                    "weights_path": args.output,
                    "config": config,
                    "template": template_data,
                    "quant_summary": quant_summary,
                    "num_merges": num_merges,
                    "total_vocab_bytes": total_vocab_bytes,
                    "entries": manifest_entries,
                }
            manifest_hash = calculate_manifest_hash(manifest_dict)
            template_hash = calculate_template_hash(template_data)

            # Build metadata
            created_by = f"convert_hf_to_bump_v7.py v{BUMP_VERSION_V5}"
            metadata = build_bumpv5_metadata(
                template_data=template_data,
                config=config,
                quant_summary=quant_summary,
                manifest_hash=manifest_hash,
                created_by=created_by
            )

            # Add template hash to metadata
            metadata["template_hash"] = template_hash

            metadata_bytes = _canonical_json_bytes(metadata)
            meta_size = len(metadata_bytes)
            meta_hash = calculate_metadata_hash(metadata)

            # Write BUMPWGT5 header (same layout as BUMPWGT4)
            f.write(b"BUMPWGT5")
            f.write(struct.pack("<I", 5))  # version
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

            # Append metadata at EOF (no padding; footer locates blob)
            f.seek(0, os.SEEK_END)
            meta_offset = f.tell()
            f.write(metadata_bytes)
            write_bumpv5_footer(f, meta_size, meta_hash)

            print(f"[bumpv5] Metadata: {meta_size} bytes @ offset {meta_offset}")
            print(f"[bumpv5] Template: {template_data.get('name', cfg.get('model_type', 'llama'))}, quant_summary: {len(quant_summary)} layers")

        else:
            # BUMPWGT4: Legacy format
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
        manifest = manifest_dict or {
            "format": "ck-bumpwgt5-manifest-v1" if args.bump_version == BUMP_VERSION_V5 else "ck-bumpwgt4-manifest-v1",
            "version": args.bump_version,
            "weights_path": args.output,
            "config": config if args.bump_version == BUMP_VERSION_V5 else None,
            "template": template_data if args.bump_version == BUMP_VERSION_V5 else None,
            "quant_summary": quant_summary if args.bump_version == BUMP_VERSION_V5 else None,
            "num_merges": num_merges,
            "total_vocab_bytes": total_vocab_bytes,
            "entries": manifest_entries,
        }
        if args.bump_version != BUMP_VERSION_V5:
            for key in ("config", "template", "quant_summary"):
                manifest.pop(key, None)
        with open(args.manifest_out, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

    mb = (HEADER_SIZE + w.bytes_written) / (1024 * 1024)
    if q4k:
        print(
            f"Wrote {args.output} ({mb:.2f} MB, q4_k, BUMPv{args.bump_version}, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})"
        )
    else:
        print(f"Wrote {args.output} ({mb:.2f} MB, fp32, BUMPv{args.bump_version}, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})")
    if not args.tokenizer_json:
        print("[tokenizer] Warning: vocab/merges not embedded (pass --tokenizer-json)")


if __name__ == "__main__":
    main()
